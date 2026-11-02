"""Matched three-unit column-only locality ablation on frozen normalized-LP v1."""
from __future__ import annotations
import argparse
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
from datetime import datetime, timezone

for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[name] = '2'
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

UNITS = ['Haukeli_G1', 'Lio_G1', 'Hogga_G1']
WEIGHTS = [0., .25, .5, 1.]


def main() -> None:
    """Change only approved feature columns; freeze a joint weight on 2021."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', default='lp-locality-001')
    args = parser.parse_args()
    out = ROOT / 'artifacts' / args.run_id
    out.mkdir(exist_ok=False)
    started = time.time()
    report = {'run_id': args.run_id, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'command': shlex.join([sys.executable, *sys.argv]), 'units': UNITS,
              'weight_grid': WEIGHTS, 'seed': 4173, 'early_stopping': None,
              'change': 'Only approved per-unit feature columns change relative to matched LPv1 LightGBM; all model knobs and saved training-row indices fixed. CPU n_jobs reduced from baseline to2 as authorized.',
              'selection_rule': 'Jointly replace the same3unit columns at weights0,.25,.5,1; choose2021 whole14-unit microAUC, ties favor lowerweight. Lock before2022 validation NPZ or fitting. No perunitweight or2022 selection.',
              'postprocessing': 'none; raw probabilities, no projection or smoothing',
              'environment': {'python': sys.version, 'platform': platform.platform(), 'cpu': platform.processor(),
                              'cpu_count': os.cpu_count(), 'threads': 2,
                              'packages': {n: importlib.metadata.version(n) for n in ['numpy', 'pandas', 'lightgbm', 'scikit-learn', 'scipy', 'pyyaml']}},
              'source': {}, 'data': {}, 'inputs': {}, 'folds': {},
              'kaggle': {'submission_id': None, 'score': None, 'rank': None, 'observation_time': None}}
    try:
        base_dir = ROOT / 'artifacts/lgb-normalized-lp-001'
        base_path = base_dir / 'manifest.json'
        baseline = json.loads(base_path.read_text())
        protocol_path = ROOT / 'docs/lp_locality_columns.json'
        protocol = json.loads(protocol_path.read_text())
        if baseline.get('status') != 'complete' or baseline['arguments']['feature_set'] != 'normalized-lp':
            raise ValueError('Frozen LPv1 baseline must be complete')
        if digest(base_path) != protocol['baseline_manifest_sha256']:
            raise ValueError('Approved column protocol references another baseline manifest')
        if protocol['counts'] != {'Haukeli_G1': 81, 'Lio_G1': 83, 'Hogga_G1': 86}:
            raise ValueError('Approved locality counts changed')
        verify(ROOT, baseline['data'])
        verify(base_dir, baseline['artifacts'])
        verify(base_dir / 'source', baseline['source'])
        for name in FEATURE_MODULES:
            if digest(ROOT / name) != baseline['source'][name]:
                raise ValueError(f'Frozen feature code changed: {name}')
        if report['environment']['packages']['lightgbm'] != baseline['environment']['packages']['lightgbm']:
            raise ValueError('LightGBM version changed since baseline')
        params = baseline['model_parameters'].copy()
        params['n_jobs'] = 2
        if params['n_estimators'] != 350 or params['random_state'] != 4173:
            raise ValueError('Unexpected baseline model recipe')
        report['model_parameters'] = lgb.LGBMClassifier(**params).get_params()
        report['data'] = baseline['data']
        report['inputs'] = {str(base_path.relative_to(ROOT)): digest(base_path),
                            str(protocol_path.relative_to(ROOT)): digest(protocol_path)}
        report['column_protocol'] = protocol
        for name in [*FEATURE_MODULES, 'scripts/xgb_lp_specialists.py', 'scripts/lp_locality_ablation.py',
                     'docs/lp_locality_columns.json', 'docs/lp_locality_protocol.md']:
            report['source'][name] = digest(ROOT / name)
            dest = out / 'source' / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, dest)
        shutil.copy2(base_path, out / 'baseline_manifest.json')
        x, all_labels, train, _, metadata = build_features(ROOT)
        x, physics = add_physics_features(x, ROOT)
        x, context = add_context_features(x, ROOT)
        x, lp = add_lp_features(x, ROOT)
        x, normalized = add_normalized_features(x, ROOT)
        metadata['feature_spec'].update(physics=physics, context=context, lp=lp, normalized=normalized)
        if x.columns.tolist() != baseline['features'] or metadata['feature_spec'] != baseline['feature_spec']:
            raise ValueError('Full feature reconstruction differs from matched baseline')
        report['feature_spec'] = metadata['feature_spec']
        report['full_feature_order'] = x.columns.tolist()
        report['unit_columns'] = protocol['columns_by_unit']
        for unit in UNITS:
            cols = report['unit_columns'][unit]
            if len(cols) != protocol['counts'][unit] or len(set(cols)) != len(cols) or not set(cols).issubset(x.columns):
                raise ValueError('Invalid approved feature list')
            if set(cols).intersection(protocol['omitted_calendar']):
                raise ValueError('Calendar leaked into locality feature list')
        units = metadata['units']; indices = [units.index(u) for u in UNITS]
        dates = pd.to_datetime(train.starttime, utc=True)
        retained = int((dates < pd.Timestamp('2022-01-01', tz='UTC')).sum()) * 168
        # Shared loader parses labels, but retain only pre2022 training targets.
        # Decode2022 evaluation labels/probabilities only after the weight lock.
        training_y = np.column_stack([all_labels[u][:retained] for u in UNITS]).astype(np.uint8)
        del all_labels
        selected = None
        for year in [2021, 2022]:
            begin = pd.Timestamp(f'{year}-01-01', tz='UTC')
            end = pd.Timestamp(f'{year+1}-01-01', tz='UTC')
            tr_cases = np.flatnonzero(dates < begin - pd.Timedelta(days=6))
            va_cases = np.flatnonzero((dates >= begin) & (dates < end))
            if not dates.iloc[tr_cases].max() + pd.Timedelta(hours=167) < begin:
                raise ValueError('Training target times overlap validation')
            if train.iloc[tr_cases]['Run No'].tolist() != baseline['folds'][str(year)]['train_ids']:
                raise ValueError('Training case IDs differ from baseline')
            sample_path = base_dir / f'train_sample_{year}.npy'
            sample = np.load(sample_path, allow_pickle=False)
            allowed = (tr_cases[:, None] * 168 + np.arange(168)).ravel()
            if len(np.unique(sample)) != len(sample) or not np.isin(sample, allowed).all():
                raise ValueError('Stored training samples violate eligible rows')
            np.save(out / f'train_sample_{year}.npy', sample)
            report['inputs'][str(sample_path.relative_to(ROOT))] = digest(sample_path)
            val_path = base_dir / f'validation_{year}.npz'
            report['inputs'][str(val_path.relative_to(ROOT))] = digest(val_path)
            with np.load(val_path, allow_pickle=False) as z:
                y, base_pred = z['labels'], z['predictions'].astype(np.float64)
                ids = z['case_ids']
                if z['units'].tolist() != units or not np.array_equal(ids, train.iloc[va_cases]['Run No']):
                    raise ValueError('Validation ID/unit order mismatch')
            va = (va_cases[:, None] * 168 + np.arange(168)).ravel()
            pred = np.empty((len(va), 3), dtype=np.float32)
            fold = {'training_ids': train.iloc[tr_cases]['Run No'].tolist(), 'validation_ids': ids.tolist(),
                    'training_dates': [str(dates.iloc[tr_cases].min()), str(dates.iloc[tr_cases].max())],
                    'validation_dates': [str(dates.iloc[va_cases].min()), str(dates.iloc[va_cases].max())],
                    'purge_days': 6, 'sampled_rows': len(sample), 'models': {}, 'scores': {}}
            report['folds'][str(year)] = fold
            for j, unit in enumerate(UNITS):
                fit_start = time.time()
                cols = report['unit_columns'][unit]
                model = lgb.LGBMClassifier(**params)
                model.fit(x.loc[:, cols].iloc[sample], training_y[sample, j])
                pred[:, j] = model.predict_proba(x.loc[:, cols].iloc[va])[:, 1]
                path = out / f'fold{year}_{unit}.txt'
                model.booster_.save_model(str(path))
                if model.booster_.current_iteration() != 350 or not np.isfinite(pred[:, j]).all():
                    raise ValueError('Unexpected tree count or invalid predictions')
                fold['models'][unit] = {'checkpoint': path.name, 'sha256': digest(path),
                                        'columns': cols, 'training_seconds': time.time()-fit_start,
                                        'local_auc': float(roc_auc_score(y[:, indices[j]], pred[:, j])),
                                        'baseline_auc': float(roc_auc_score(y[:, indices[j]], base_pred[:, indices[j]]))}
                print(year, unit, fold['models'][unit]['local_auc'], 'baseline', fold['models'][unit]['baseline_auc'], flush=True)
            for weight in WEIGHTS:
                candidate = base_pred.copy()
                candidate[:, indices] = (1-weight)*candidate[:, indices] + weight*pred
                fold['scores'][str(weight)] = float(roc_auc_score(y.ravel(), candidate.ravel()))
            if year == 2021:
                selected = max(WEIGHTS, key=lambda w: fold['scores'][str(w)])
                report['selected_weight'] = selected
                report['selection_locked_utc'] = datetime.now(timezone.utc).isoformat()
                (out / 'selection_2021.json').write_text(json.dumps(report, indent=2))
            chosen = base_pred.copy()
            chosen[:, indices] = (1-selected)*chosen[:, indices] + selected*pred
            fold['selected_micro_auc'] = fold['scores'][str(selected)]
            np.savez_compressed(out / f'validation_{year}.npz', predictions=chosen, labels=y,
                                case_ids=ids, units=units, locality_predictions=pred, locality_units=UNITS)
            print('FOLD', year, 'jointweight', selected, 'baseline', fold['scores']['0.0'],
                  'selected', fold['selected_micro_auc'], flush=True)
            (out / 'manifest.json').write_text(json.dumps(report, indent=2))
        report['improves_both_matched_folds'] = all(f['selected_micro_auc'] > f['scores']['0.0'] for f in report['folds'].values())
        report['meets_transfer_gate'] = all(f['selected_micro_auc'] >= .998 for f in report['folds'].values())
        report['status'] = 'complete'
        lines = ['# LP locality matched ablation', '',
                 'Only the approved per-unit column lists changed. Model hyperparameters, seed and exact sampled training rows match the full-feature LPv1 LightGBM baseline; CPUthreads were reduced to2. No smoothing/projection, newfeaturecomputation or LPsource changes.', '',
                 '| Year | Baseline microAUC | Frozen-weight microAUC |', '|---|---:|---:|']
        for year, f in report['folds'].items():
            lines.append(f'| {year} | {f["scores"]["0.0"]:.8f} | {f["selected_micro_auc"]:.8f} |')
        lines += ['', f'2021-selected joint weight:{selected}. Improves both matchedfolds:{report["improves_both_matched_folds"]}. Meets>=.998both:{report["meets_transfer_gate"]}.', '',
                  'These are reused chronological developmentfolds. No fullfit or Kaggle submission was made. A positive matched result still needs comparison with the prioraccepted combinedcandidate before deciding any fullfit.', '',
                  f'Exactcode/data/model/sample provenance, perunitAUC, allgridresults and selected/rawlocalitypredictions:`artifacts/{args.run_id}/manifest.json`.']
        (ROOT / 'docs/lp_locality_results.md').write_text('\n'.join(lines)+'\n')
    except BaseException:
        report['status'] = 'failed'
        report['error'] = traceback.format_exc()
        raise
    finally:
        report['runtime_seconds'] = time.time()-started
        report['artifacts'] = {str(p.relative_to(out)): digest(p) for p in out.rglob('*') if p.is_file() and p.name != 'manifest.json'}
        (out / 'manifest.json').write_text(json.dumps(report, indent=2))
        with (ROOT / 'docs/experiments.md').open('a') as f:
            f.write(f'\n## {args.run_id} — {report["timestamp_utc"]}\n\nStatus:{report["status"]}. Command:`{report["command"]}`. Runtime:{report["runtime_seconds"]:.1f}s.\n\n')
            f.write('Hypothesis:localLP/physical/market inputs reduce calendar and distantreservoir shortcuts onHaukeli/Lio/Hogga. Exactapproved81/83/86column sets; same350treeLPv1LightGBM settings,seed4173 and saved sampleindices; onlycolumns change exceptauthorized CPUthreadcap2. Noearlystopping. Jointweights0,.25,.5,1 selected2021 thenlocked2022; noproj/smoothing.\n\n')
            for year, fold in report['folds'].items():
                f.write(f'- {year}: baseline={fold.get("scores",{}).get("0.0")}; selected={fold.get("selected_micro_auc")}; jointweight={report.get("selected_weight")}.\n')
            if 'error' in report:
                f.write(f'\nFailure:`{report["error"].splitlines()[-1]}`\n')
            f.write(f'\nAll parameters,features,IDs,source/data/checkpoint/sample/output hashes and environment:`artifacts/{args.run_id}/manifest.json`. No fullfit. Kaggle submission ID/score/rank/observation:none.\n')


if __name__ == '__main__':
    main()
