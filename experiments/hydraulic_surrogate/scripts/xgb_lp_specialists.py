"""Fixed XGBoost LP specialists; select joint replacement weight on 2021 only."""
from __future__ import annotations
import argparse
import hashlib
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
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sintef_project.features import build_features
from sintef_project.physics import add_physics_features
from sintef_project.context_features import add_context_features
from sintef_project.lp_features import add_lp_features
from sintef_project.normalized import add_normalized_features

SPECIALISTS = ['Haukeli_G1', 'Lio_G1', 'Hogga_G1']
WEIGHTS = [0., .25, .5, 1.]
PARAMS = dict(n_estimators=600, max_depth=6, learning_rate=.05, min_child_weight=20,
              reg_lambda=4., reg_alpha=0., max_bin=128, subsample=.9,
              colsample_bytree=.9, colsample_bylevel=1., colsample_bynode=1.,
              random_state=4173, n_jobs=2, objective='binary:logistic',
              tree_method='hist', device='cpu', eval_metric='logloss',
              grow_policy='depthwise', max_delta_step=0., validate_parameters=True,
              verbosity=0)
FEATURE_MODULES = ['sintef_project/features.py', 'sintef_project/physics.py',
                   'sintef_project/context_features.py', 'sintef_project/lp_features.py',
                   'sintef_project/normalized.py']


def digest(path: Path) -> str:
    """Return content SHA256."""
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def verify(base: Path, hashes: dict) -> None:
    """Check all declared source files and artifacts before use."""
    for relative, expected in hashes.items():
        if digest(base / relative) != expected:
            raise ValueError(f'Hash mismatch: {base / relative}')


def main() -> None:
    """Fit three fixed specialists per year using exact baseline sample rows."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', default='xgb-lp-specialists-001')
    parser.add_argument('--baseline', default='lgb-normalized-lp-001')
    args = parser.parse_args()
    out = ROOT / 'artifacts' / args.run_id
    out.mkdir(exist_ok=False)
    started = time.time()
    report = {
        'run_id': args.run_id, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'command': shlex.join([sys.executable, *sys.argv]), 'baseline_run': args.baseline,
        'specialists': SPECIALISTS, 'model_parameters': PARAMS,
        'weight_grid': WEIGHTS, 'postprocessing': 'none; retain raw probability scales; replace only the three designated columns jointly',
        'selection_rule': 'Fit/evaluate 2021; select joint weight maximizing whole-matrix 14-unit micro AUC, ties favor lower weight. Write lock before loading 2022 validation NPZ or fitting 2022 models. 2022 grid is diagnostic only.',
        'early_stopping': None, 'training_rounds': 600, 'seed': 4173,
        'environment': {'platform': platform.platform(), 'python': sys.version,
                        'cpu': platform.processor(), 'cpu_count': os.cpu_count(), 'threads': 2,
                        'packages': {n: importlib.metadata.version(n) for n in ['numpy', 'pandas', 'scikit-learn', 'xgboost', 'lightgbm', 'scipy', 'pyyaml']}},
        'source': {}, 'data': {}, 'inputs': {}, 'folds': {},
        'kaggle': {'submission_id': None, 'score': None, 'rank': None, 'observation_time': None},
    }
    try:
        base_dir = ROOT / 'artifacts' / args.baseline
        base_path = base_dir / 'manifest.json'
        base = json.loads(base_path.read_text())
        large = json.loads((ROOT / 'artifacts/lgb-physics-large-001/manifest.json').read_text())
        if base.get('status') != 'complete' or large.get('status') != 'complete':
            raise ValueError('Baseline LP and large physics must both complete before this run')
        if base['arguments']['feature_set'] != 'normalized-lp':
            raise ValueError('Requires frozen normalized-LP v1 baseline')
        verify(ROOT, base['data'])
        verify(base_dir, base['artifacts'])
        verify(base_dir / 'source', base['source'])
        for source in FEATURE_MODULES:
            if digest(ROOT / source) != base['source'][source]:
                raise ValueError(f'Feature code differs from baseline: {source}')
        report['data'] = base['data']
        report['inputs'][str(base_path.relative_to(ROOT))] = digest(base_path)
        for relative in [*FEATURE_MODULES, 'scripts/xgb_lp_specialists.py']:
            report['source'][relative] = digest(ROOT / relative)
            destination = out / 'source' / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, destination)
        # The shared feature loader parses the supplied training-label table;
        # only pre-2022 training targets are retained below. Heldout 2022 labels
        # and probabilities are accessed solely after the 2021 weight lock.
        x, all_labels, train, _, metadata = build_features(ROOT)
        x, physics_spec = add_physics_features(x, ROOT)
        x, context_spec = add_context_features(x, ROOT)
        x, lp_spec = add_lp_features(x, ROOT)
        x, normalized_spec = add_normalized_features(x, ROOT)
        metadata['feature_spec'].update(physics=physics_spec, context=context_spec,
                                        lp=lp_spec, normalized=normalized_spec)
        if x.columns.tolist() != base['features'] or metadata['feature_spec'] != base['feature_spec']:
            raise ValueError('Reconstructed features differ from baseline definitions/order')
        units = metadata['units']
        unit_indices = [units.index(name) for name in SPECIALISTS]
        dates = pd.to_datetime(train.starttime, utc=True)
        pre2022_rows = int((dates < pd.Timestamp('2022-01-01', tz='UTC')).sum()) * 168
        training_y = np.column_stack([all_labels[u][:pre2022_rows] for u in SPECIALISTS]).astype(np.uint8)
        del all_labels
        report['features'] = x.columns.tolist()
        report['feature_spec'] = metadata['feature_spec']
        report['shape'] = list(x.shape)
        report['effective_model_parameters'] = {k: ('NaN' if isinstance(v, float) and np.isnan(v) else v)
                                                for k, v in XGBClassifier(**PARAMS).get_params().items()}
        selected_weight = None
        for year in [2021, 2022]:
            begin = pd.Timestamp(f'{year}-01-01', tz='UTC')
            end = pd.Timestamp(f'{year + 1}-01-01', tz='UTC')
            train_cases = np.flatnonzero(dates < begin - pd.Timedelta(days=6))
            val_cases = np.flatnonzero((dates >= begin) & (dates < end))
            if not dates.iloc[train_cases].max() + pd.Timedelta(hours=167) < begin:
                raise ValueError('Chronological target-time overlap')
            sample_path = base_dir / f'train_sample_{year}.npy'
            sample = np.load(sample_path, allow_pickle=False)
            allowed = (train_cases[:, None] * 168 + np.arange(168)).ravel()
            if sample.ndim != 1 or len(np.unique(sample)) != len(sample) or not np.isin(sample, allowed).all():
                raise ValueError('Baseline sample contains ineligible or duplicate rows')
            if train.iloc[train_cases]['Run No'].tolist() != base['folds'][str(year)]['train_ids']:
                raise ValueError('Training IDs differ from baseline')
            np.save(out / f'train_sample_{year}.npy', sample)
            report['inputs'][str(sample_path.relative_to(ROOT))] = digest(sample_path)
            validation_path = base_dir / f'validation_{year}.npz'
            report['inputs'][str(validation_path.relative_to(ROOT))] = digest(validation_path)
            with np.load(validation_path, allow_pickle=False) as z:
                labels, base_pred = z['labels'], z['predictions'].astype(np.float64)
                if z['units'].tolist() != units or not np.array_equal(z['case_ids'], train.iloc[val_cases]['Run No']):
                    raise ValueError('Validation unit/case order differs')
                val_ids = z['case_ids']
            va = (val_cases[:, None] * 168 + np.arange(168)).ravel()
            xt, xv = x.iloc[sample], x.iloc[va]
            predictions = np.empty((len(va), len(SPECIALISTS)), dtype=np.float32)
            fold = {'training_case_ids': train.iloc[train_cases]['Run No'].tolist(),
                    'validation_case_ids': val_ids.tolist(), 'sampled_training_rows': len(sample),
                    'training_date_range': [str(dates.iloc[train_cases].min()), str(dates.iloc[train_cases].max())],
                    'validation_date_range': [str(dates.iloc[val_cases].min()), str(dates.iloc[val_cases].max())],
                    'purged_start_days': 6, 'models': {}, 'scores': {}}
            report['folds'][str(year)] = fold
            for j, unit in enumerate(SPECIALISTS):
                fit_start = time.time()
                model = XGBClassifier(**PARAMS)
                model.fit(xt, training_y[sample, j])
                predictions[:, j] = model.predict_proba(xv)[:, 1]
                checkpoint = out / f'fold{year}_{unit}.ubj'
                model.save_model(checkpoint)
                if not np.isfinite(predictions[:, j]).all():
                    raise ValueError('Nonfinite XGBoost probabilities')
                fold['models'][unit] = {'checkpoint': checkpoint.name, 'sha256': digest(checkpoint),
                                        'training_seconds': time.time() - fit_start,
                                        'xgb_auc': float(roc_auc_score(labels[:, unit_indices[j]], predictions[:, j])),
                                        'base_auc': float(roc_auc_score(labels[:, unit_indices[j]], base_pred[:, unit_indices[j]]))}
                print(year, unit, fold['models'][unit], flush=True)
            for weight in WEIGHTS:
                candidate = base_pred.copy()
                candidate[:, unit_indices] = (1 - weight) * candidate[:, unit_indices] + weight * predictions
                fold['scores'][str(weight)] = float(roc_auc_score(labels.ravel(), candidate.ravel()))
            if year == 2021:
                selected_weight = max(WEIGHTS, key=lambda w: fold['scores'][str(w)])
                report['selected_weight'] = selected_weight
                report['selection_locked_utc'] = datetime.now(timezone.utc).isoformat()
                (out / 'selection_2021.json').write_text(json.dumps(report, indent=2))
            chosen = base_pred.copy()
            chosen[:, unit_indices] = (1 - selected_weight) * chosen[:, unit_indices] + selected_weight * predictions
            fold['selected_micro_auc'] = fold['scores'][str(selected_weight)]
            np.savez_compressed(out / f'validation_{year}.npz', predictions=chosen, labels=labels,
                                case_ids=val_ids, units=units, specialist_predictions=predictions,
                                specialist_units=SPECIALISTS)
            print('FOLD', year, 'weight', selected_weight, 'baseline', fold['scores']['0.0'],
                  'selected', fold['selected_micro_auc'], flush=True)
            (out / 'manifest.json').write_text(json.dumps(report, indent=2))
        report['status'] = 'complete'
        with (ROOT / 'docs/xgb_lp_specialists.md').open('a') as f:
            f.write(f'\n## Completed result: {args.run_id}\n\nSelected joint weight on2021:{selected_weight}.\n\n| Year | Baseline micro AUC | Frozen-weight micro AUC |\n|---|---:|---:|\n')
            for year, fold in report['folds'].items():
                f.write(f'| {year} | {fold["scores"]["0.0"]:.8f} | {fold["selected_micro_auc"]:.8f} |\n')
            f.write(f'\nComplete model/source/data provenance and every grid score:`artifacts/{args.run_id}/manifest.json`. Checkpoint predictions and selected full matrices are saved peryear. No Kaggle score is implied.\n')
    except BaseException:
        report['status'] = 'failed'
        report['error'] = traceback.format_exc()
        raise
    finally:
        report['runtime_seconds'] = time.time() - started
        report['artifacts'] = {str(p.relative_to(out)): digest(p) for p in out.rglob('*') if p.is_file() and p.name != 'manifest.json'}
        (out / 'manifest.json').write_text(json.dumps(report, indent=2))
        with (ROOT / 'docs/experiments.md').open('a') as f:
            f.write(f'\n## {args.run_id} — {report["timestamp_utc"]}\n\nStatus:{report["status"]}. Command:`{report["command"]}`. Runtime:{report["runtime_seconds"]:.1f}s.\n\n')
            f.write(f'Hypothesis:complementary XGBoost specialists for{SPECIALISTS} improve the normalized-LP v1 LightGBM matrix. Parameters:{PARAMS}. No early stopping. Identical baseline purged case IDs and stored sampledtrainingrowindices; no preprocessing fit. Weights{WEIGHTS} jointly replace the threeunits; select2021 before2022 validation access and fitting. Fullmatrix microAUC, no projection or smoothing.\n\n')
            for year, fold in report['folds'].items():
                f.write(f'- {year}: baseline={fold.get("scores",{}).get("0.0")}; selected={fold.get("selected_micro_auc")}; selectedweight={report.get("selected_weight")}.\n')
            if 'error' in report:
                f.write(f'\nFailure:`{report["error"].splitlines()[-1]}`\n')
            f.write(f'\nComplete source snapshots/data and artifact checksums, environment, splits, samples, checkpoints and grid:`artifacts/{args.run_id}/manifest.json`. Kaggle submission ID/score/rank/observation:none.\n')


if __name__ == '__main__':
    main()
