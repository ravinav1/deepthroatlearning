"""Matched LP-v1 plus past-only reservoir-history specialist ablation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import sys
import time
import traceback
from unittest.mock import patch

for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[variable] = '2'
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.xgb_lp_specialists import (
    FEATURE_MODULES, digest, verify, build_features, add_physics_features,
    add_context_features, add_lp_features, add_normalized_features,
)
from sintef_project.features import _capacities
from sintef_project.past_state_features import build_past_state_features

UNITS = ['Haukeli_G1', 'Lio_G1', 'Hogga_G1']
RESERVOIRS = ['Kjelavatn', 'Staavatn', 'Langesae', 'Foersvatn', 'Venemo',
              'Botnedalsvatn', 'Bitdalsvatn', 'Vaamarvatn', 'Langeidvatn',
              'Totak', 'Songavatn', 'Vinjevatn', 'Bandak', 'Bordalsvatn', 'Byrtevatn']
WEIGHTS = [0., .25, .5, 1.]


def build_without_2022_targets() -> tuple:
    """Call the frozen feature builder without decoding any 2022 target rows.

    The frozen builder's feature computation is independent of labels. Its one
    training CSV read is intercepted locally: read all case metadata, read only
    pre-2022 target rows, and append zero sentinel labels for later case rows.
    These sentinels are discarded and never fitted or evaluated. Evaluation
    labels are read from the verified baseline NPZ only after the 2021 lock.
    All other reads delegate unchanged; no source module or raw file is edited.
    """
    target = (ROOT / 'sintef_project/data/kernel/Unit_commitment_decisions.csv').resolve()
    original_read = pd.read_csv

    def read_csv(path: object, *args: object, **kwargs: object) -> pd.DataFrame:
        if isinstance(path, (str, Path)) and Path(path).resolve() == target:
            if args or kwargs:
                raise ValueError('Frozen feature-loader target read changed unexpectedly')
            meta = original_read(target, usecols=['Run No', 'starttime'])
            dates = pd.to_datetime(meta.starttime, utc=True)
            keep = dates < pd.Timestamp('2022-01-01', tz='UTC')
            count = int(keep.sum())
            if not np.array_equal(keep.to_numpy(), np.arange(len(meta)) < count):
                raise ValueError('Target CSV must have chronological prefix order')
            prior = original_read(target, nrows=count)
            later = pd.DataFrame(np.zeros((len(meta) - count, len(prior.columns) - 3), dtype=np.uint8),
                                 columns=prior.columns[3:])
            later.insert(0, 'total_calculation_time', 0.)
            later.insert(0, 'starttime', meta.starttime.iloc[count:].to_numpy())
            later.insert(0, 'Run No', meta['Run No'].iloc[count:].to_numpy())
            return pd.concat([prior, later], ignore_index=True)
        return original_read(path, *args, **kwargs)

    with patch.object(pd, 'read_csv', read_csv):
        return build_features(ROOT)


def load_past_inputs(starts: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict]:
    """Read supplied histories and build only the frozen 225 case-level fields."""
    kernel = ROOT / 'sintef_project/data/kernel'
    volume = pd.read_csv(kernel / 'Historical_volume_2015_2024.csv')
    volume.index = pd.to_datetime(volume.pop('date'), utc=True)
    fragments = []
    for chunk in pd.read_csv(kernel / 'Historical_inflow_1958_2025.csv',
                             usecols=['date', *RESERVOIRS], chunksize=100000,
                             dtype={name: 'float64' for name in RESERVOIRS}):
        chunk.index = pd.to_datetime(chunk.pop('date'), utc=True)
        fragments.append(chunk.loc[(chunk.index >= starts.min() - pd.Timedelta(days=7))
                                    & (chunk.index < starts.max())])
    capacities = _capacities(ROOT / 'sintef_project/data/extended/Tokke_Vinje_topology.yaml')
    return build_past_state_features(starts, volume, pd.concat(fragments), capacities,
                                    reservoirs=RESERVOIRS, lookback_days=(1, 3, 7))


def main() -> None:
    """Keep baseline rows and hyperparameters; freeze the joint blend on 2021."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', default='lp-past-state-specialists-001')
    args = parser.parse_args()
    out = ROOT / 'artifacts' / args.run_id
    out.mkdir(exist_ok=False)
    start = time.time()
    report = {'run_id': args.run_id, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'command': shlex.join([sys.executable, *sys.argv]), 'units': UNITS,
              'weight_grid': WEIGHTS, 'seed': 4173, 'early_stopping': None,
              'change': 'Retain all 395 baseline columns and append 225 past-only fields. Same model parameters and sampled rows; authorized CPU thread cap2.',
              'selection_rule': '2021 maximum whole14-unit micro AUC for joint3-unit weight, ties lower weight; immutable lock before loading2022 evaluation labels. Evaluate only baseline and locked weight in2022.',
              'postprocessing': 'none; raw probabilities, no smoothing or projection',
              'label_access': 'Frozen builder locally receives real labels only through2021, later zero sentinels discarded;2022 evaluation NPZ opened only after2021 lock.',
              'environment': {'python': sys.version, 'platform': platform.platform(),
                              'cpu': platform.processor(), 'cpu_count': os.cpu_count(), 'threads': 2,
                              'packages': {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'lightgbm', 'scikit-learn', 'scipy', 'pyyaml']}},
              'source': {}, 'data': {}, 'inputs': {}, 'folds': {},
              'kaggle': {'submission_id': None, 'score': None, 'rank': None, 'observation_time': None}}
    try:
        gate_path = ROOT / 'artifacts/catboost-lp-specialists-001/manifest.json'
        if json.loads(gate_path.read_text()).get('status') != 'complete':
            raise ValueError('CatBoost specialist run has not completed; do not start another heavy job')
        base_dir = ROOT / 'artifacts/lgb-normalized-lp-001'
        base_path = base_dir / 'manifest.json'
        baseline = json.loads(base_path.read_text())
        if baseline.get('status') != 'complete' or baseline['arguments']['feature_set'] != 'normalized-lp':
            raise ValueError('Expected complete frozen normalized LP-v1 baseline')
        verify(ROOT, baseline['data'])
        verify(base_dir, baseline['artifacts'])
        verify(base_dir / 'source', baseline['source'])
        for name in FEATURE_MODULES:
            if digest(ROOT / name) != baseline['source'][name]:
                raise ValueError(f'Frozen feature code changed: {name}')
        if report['environment']['packages']['lightgbm'] != baseline['environment']['packages']['lightgbm']:
            raise ValueError('LightGBM version differs from baseline')
        params = baseline['model_parameters'].copy()
        params['n_jobs'] = 2
        if params['n_estimators'] != 350 or params['random_state'] != 4173:
            raise ValueError('Unexpected model recipe')
        report['model_parameters'] = lgb.LGBMClassifier(**params).get_params()
        report['data'] = baseline['data']
        report['inputs'] = {str(path.relative_to(ROOT)): digest(path) for path in [base_path, gate_path]}
        for name in [*FEATURE_MODULES, 'scripts/xgb_lp_specialists.py', 'scripts/lp_past_state_specialists.py',
                     'sintef_project/past_state_features.py', 'tests/test_past_state_features.py',
                     'tests/test_past_state_runner.py', 'docs/past_state_features.md']:
            report['source'][name] = digest(ROOT / name)
            destination = out / 'source' / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, destination)
        shutil.copy2(base_path, out / 'baseline_manifest.json')
        x, labels, train, mapping, meta = build_without_2022_targets()
        x, physics = add_physics_features(x, ROOT)
        x, context = add_context_features(x, ROOT)
        x, lp = add_lp_features(x, ROOT)
        x, normalized = add_normalized_features(x, ROOT)
        meta['feature_spec'].update(physics=physics, context=context, lp=lp, normalized=normalized)
        if x.columns.tolist() != baseline['features'] or meta['feature_spec'] != baseline['feature_spec'] or x.shape[1] != 395:
            raise ValueError('Baseline features/specification changed')
        cases = pd.concat([train, mapping], ignore_index=True)
        past, past_spec = load_past_inputs(pd.DatetimeIndex(cases.starttime))
        if past.shape != (len(cases), 225) or set(past.columns).intersection(x.columns):
            raise ValueError('Unexpected past-state family')
        np.save(out / 'past_case_features.npy', past.to_numpy())
        x = pd.concat([x, pd.DataFrame(np.repeat(past.to_numpy(), 168, axis=0),
                                      index=x.index, columns=past.columns)], axis=1)
        if x.shape[1] != 620:
            raise ValueError('Expected620 combined feature columns')
        report['features'] = x.columns.tolist()
        report['feature_spec'] = {**meta['feature_spec'], 'past_state': past_spec}
        report['shape'] = list(x.shape)
        units = meta['units']
        indices = [units.index(unit) for unit in UNITS]
        dates = pd.to_datetime(train.starttime, utc=True)
        retained = int((dates < pd.Timestamp('2022-01-01', tz='UTC')).sum()) * 168
        training_y = np.column_stack([labels[unit][:retained] for unit in UNITS]).astype(np.uint8)
        del labels, past
        selected = None
        for year in [2021, 2022]:
            if year == 2022 and not (out / 'selection_2021.json').is_file():
                raise ValueError('2021 selection must be locked before2022 labels')
            begin = pd.Timestamp(f'{year}-01-01', tz='UTC')
            end = pd.Timestamp(f'{year+1}-01-01', tz='UTC')
            train_cases = np.flatnonzero(dates < begin - pd.Timedelta(days=6))
            val_cases = np.flatnonzero((dates >= begin) & (dates < end))
            if not dates.iloc[train_cases].max() + pd.Timedelta(hours=167) < begin:
                raise ValueError('Target timestamps overlap validation')
            if train.iloc[train_cases]['Run No'].tolist() != baseline['folds'][str(year)]['train_ids']:
                raise ValueError('Training case IDs differ from baseline')
            sample_path = base_dir / f'train_sample_{year}.npy'
            sample = np.load(sample_path, allow_pickle=False)
            allowed = (train_cases[:, None] * 168 + np.arange(168)).ravel()
            if len(np.unique(sample)) != len(sample) or not np.isin(sample, allowed).all():
                raise ValueError('Saved sample contains invalid rows')
            np.save(out / f'train_sample_{year}.npy', sample)
            validation_path = base_dir / f'validation_{year}.npz'
            for path in [sample_path, validation_path]:
                report['inputs'][str(path.relative_to(ROOT))] = digest(path)
            with np.load(validation_path, allow_pickle=False) as z:
                y, base_pred, ids = z['labels'], z['predictions'].astype(np.float64), z['case_ids']
                if z['units'].tolist() != units or not np.array_equal(ids, train.iloc[val_cases]['Run No']):
                    raise ValueError('Validation ID/unit order differs')
            val_rows = (val_cases[:, None] * 168 + np.arange(168)).ravel()
            x_train, x_val = x.iloc[sample], x.iloc[val_rows]
            pred = np.empty((len(val_rows), 3), dtype=np.float32)
            fold = {'training_ids': train.iloc[train_cases]['Run No'].tolist(), 'validation_ids': ids.tolist(),
                    'training_dates': [str(dates.iloc[train_cases].min()), str(dates.iloc[train_cases].max())],
                    'validation_dates': [str(dates.iloc[val_cases].min()), str(dates.iloc[val_cases].max())],
                    'purge_days': 6, 'sampled_rows': len(sample), 'models': {}, 'scores': {}}
            report['folds'][str(year)] = fold
            for column, unit in enumerate(UNITS):
                fit_start = time.time()
                model = lgb.LGBMClassifier(**params)
                model.fit(x_train, training_y[sample, column])
                pred[:, column] = model.predict_proba(x_val)[:, 1]
                checkpoint = out / f'fold{year}_{unit}.txt'
                model.booster_.save_model(str(checkpoint))
                if model.booster_.current_iteration() != 350 or not np.isfinite(pred[:, column]).all():
                    raise ValueError('Invalid checkpoint or probabilities')
                fold['models'][unit] = {'checkpoint': checkpoint.name, 'sha256': digest(checkpoint),
                                       'training_seconds': time.time() - fit_start,
                                       'specialist_auc': float(roc_auc_score(y[:, indices[column]], pred[:, column])),
                                       'baseline_auc': float(roc_auc_score(y[:, indices[column]], base_pred[:, indices[column]]))}
                print(year, unit, fold['models'][unit], flush=True)
            for weight in (WEIGHTS if year == 2021 else sorted(set([0., selected]))):
                candidate = base_pred.copy()
                candidate[:, indices] = (1 - weight) * candidate[:, indices] + weight * pred
                fold['scores'][str(weight)] = float(roc_auc_score(y.ravel(), candidate.ravel()))
            if year == 2021:
                selected = max(WEIGHTS, key=lambda weight: fold['scores'][str(weight)])
                report['selected_weight'] = selected
                report['selection_locked_utc'] = datetime.now(timezone.utc).isoformat()
                (out / 'selection_2021.json').write_text(json.dumps(report, indent=2))
            chosen = base_pred.copy()
            chosen[:, indices] = (1 - selected) * chosen[:, indices] + selected * pred
            fold['selected_micro_auc'] = fold['scores'][str(selected)]
            np.savez_compressed(out / f'validation_{year}.npz', predictions=chosen, labels=y,
                                case_ids=ids, units=units, specialist_predictions=pred, specialist_units=UNITS)
            print('FOLD', year, 'joint_weight', selected, 'baseline', fold['scores']['0.0'],
                  'selected', fold['selected_micro_auc'], flush=True)
            (out / 'manifest.json').write_text(json.dumps(report, indent=2))
            del x_train, x_val
        report['improves_both_matched_folds'] = all(f['selected_micro_auc'] > f['scores']['0.0'] for f in report['folds'].values())
        report['meets_transfer_gate'] = all(f['selected_micro_auc'] >= .998 for f in report['folds'].values())
        report['status'] = 'complete'
    except BaseException:
        report['status'] = 'failed'
        report['error'] = traceback.format_exc()
        raise
    finally:
        report['runtime_seconds'] = time.time() - start
        report['artifacts'] = {str(p.relative_to(out)): digest(p) for p in out.rglob('*') if p.is_file() and p.name != 'manifest.json'}
        (out / 'manifest.json').write_text(json.dumps(report, indent=2))
        with (ROOT / 'docs/experiments.md').open('a') as handle:
            handle.write(f'\n## {args.run_id} — {report["timestamp_utc"]}\n\nStatus: {report["status"]}. '
                         f'Command: `{report["command"]}`. Runtime: {report["runtime_seconds"]:.1f}s. '
                         'Matched 350-tree three-unit LP-v1 specialist ablation: retain395 baseline columns, add225 strictly past-state fields; '
                         'same saved sample rows and parameters, two CPU threads. No early stopping, smoothing or projection. '
                         'Joint weights0/.25/.5/1 chosen on2021 before2022 labels; no fullfit or Kaggle submission. '
                         f'All code/data/checkpoint/sample/environment provenance: `artifacts/{args.run_id}/manifest.json`.\n')
            for year, fold in report['folds'].items():
                handle.write(f'\n- {year}: baseline {fold["scores"].get("0.0")}; locked weight {report.get("selected_weight")}; selected {fold.get("selected_micro_auc")}.\n')
            if 'error' in report:
                handle.write(f'\nFailure: `{report["error"].splitlines()[-1]}`.\n')


if __name__ == '__main__':
    main()
