"""Chronological hydro-scheduling experiments and reproducible inference."""
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
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, log_loss
from sintef_project.features import build_features


def checksum(path: Path) -> str:
    """Return a SHA256 content digest."""
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def metrics(y: np.ndarray, p: np.ndarray, units: list[str]) -> dict:
    """Compute the actual micro objective plus useful unit diagnostics."""
    return {
        'micro_auc': float(roc_auc_score(y.ravel(), p.ravel())),
        'log_loss': float(log_loss(y.ravel(), p.ravel(), labels=[0, 1])),
        'unit_auc': {u: float(roc_auc_score(y[:, j], p[:, j]))
                     if np.unique(y[:, j]).size == 2 else None for j, u in enumerate(units)},
    }


def select_features(x: pd.DataFrame, name: str) -> list[str]:
    """Choose all features or a documented ablation of engineered context."""
    if name in ['full', 'physics', 'normalized', 'normalized-context', 'normalized-lp', 'normalized-lp-context', 'normalized-lp-v2', 'normalized-lp-switching', 'normalized-lp-joint', 'normalized-lp-context-joint']:
        return list(x.columns)
    # Definition checked against and recorded with the feature builder's names.
    return [c for c in x.columns if c.startswith(('volume_', 'water_value_'))
            or c in ['price', 'horizon_hour', 'hour_of_day', 'season_sin', 'season_cos']]


def main() -> None:
    """Train fixed-round per-generator models with purged yearly holdouts."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--years', nargs='*', type=int, default=[2021, 2022])
    parser.add_argument('--trees', type=int, default=250)
    parser.add_argument('--leaves', type=int, default=31)
    parser.add_argument('--learning-rate', type=float, default=.06)
    parser.add_argument('--min-child-samples', type=int, default=100)
    parser.add_argument('--sample-fraction', type=float, default=.35)
    parser.add_argument('--feature-set', choices=['basic', 'full', 'physics', 'normalized', 'normalized-context', 'normalized-lp', 'normalized-lp-context', 'normalized-lp-v2', 'normalized-lp-switching', 'normalized-lp-joint', 'normalized-lp-context-joint'], default='full')
    parser.add_argument('--seed', type=int, default=4173)
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--final', action='store_true')
    parser.add_argument('--inference-from', type=Path)
    args = parser.parse_args()
    out = ROOT / 'artifacts' / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    start = time.time()
    manifest = {'run_id': args.run_id, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                'command': shlex.join([sys.executable, *sys.argv]), 'arguments': vars(args).copy(),
                'environment': {'platform': platform.platform(), 'python': sys.version,
                                'cpu': platform.processor(), 'cpu_count': os.cpu_count(),
                                'hardware': subprocess.run(['system_profiler', 'SPHardwareDataType'], capture_output=True, text=True).stdout.split('Serial Number')[0],
                                'packages': {x: importlib.metadata.version(x) for x in ['numpy', 'pandas', 'scikit-learn', 'lightgbm', 'pyyaml', 'scipy']}},
                'data': {}, 'source': {}, 'folds': {},
                'kaggle': {'submission_id': None, 'score': None, 'rank': None, 'observation_time': None},
                'early_stopping': 'none; fixed rounds', 'ensemble_weights': [1.0]}
    manifest['arguments']['inference_from'] = str(args.inference_from) if args.inference_from else None
    for path in sorted([*(ROOT / 'sintef_project').rglob('*.csv'), *(ROOT / 'sintef_project').rglob('*.yaml')]):
        manifest['data'][str(path.relative_to(ROOT))] = checksum(path)
    for path in sorted([* (ROOT / 'scripts').glob('*.py'), *(ROOT / 'sintef_project').glob('*.py')]):
        rel = path.relative_to(ROOT)
        manifest['source'][str(rel)] = checksum(path)
        destination = out / 'source' / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    try:
        x, labels, train, mapping, meta = build_features(ROOT)
        units = meta['units']
        if args.feature_set in ['physics', 'normalized', 'normalized-context', 'normalized-lp', 'normalized-lp-context', 'normalized-lp-v2', 'normalized-lp-switching', 'normalized-lp-joint', 'normalized-lp-context-joint']:
            from sintef_project.physics import add_physics_features
            x, physics_spec = add_physics_features(x, ROOT)
            meta['feature_spec']['physics'] = physics_spec
        if args.feature_set in ['normalized-context', 'normalized-lp', 'normalized-lp-context', 'normalized-lp-v2', 'normalized-lp-switching', 'normalized-lp-joint', 'normalized-lp-context-joint']:
            from sintef_project.context_features import add_context_features
            x, context_spec = add_context_features(x, ROOT)
            meta['feature_spec']['context'] = context_spec
        if args.feature_set in ['normalized-lp', 'normalized-lp-context', 'normalized-lp-switching', 'normalized-lp-context-joint']:
            from sintef_project.lp_features import add_lp_features
            x, lp_spec = add_lp_features(x, ROOT)
            meta['feature_spec']['lp'] = lp_spec
        if args.feature_set == 'normalized-lp-v2':
            from sintef_project.lp_features_v2 import add_lp_features_v2
            x, lp_spec = add_lp_features_v2(x, ROOT)
            meta['feature_spec']['lp_v2'] = lp_spec
        if args.feature_set == 'normalized-lp-joint':
            from sintef_project.lp_joint_plant import add_lp_joint_features
            x, joint_spec = add_lp_joint_features(x, ROOT)
            meta['feature_spec']['lp_joint'] = joint_spec
        if args.feature_set in ['normalized-lp-context', 'normalized-lp-context-joint']:
            from sintef_project.lp_context import add_lp_context_features
            x, lp_context_spec = add_lp_context_features(x, ROOT)
            meta['feature_spec']['lp_context'] = lp_context_spec
        if args.feature_set == 'normalized-lp-switching':
            from sintef_project.lp_switching import add_lp_switching_features
            x, switching_spec = add_lp_switching_features(x, ROOT)
            meta['feature_spec']['lp_switching'] = switching_spec
        if args.feature_set in ['normalized', 'normalized-context', 'normalized-lp', 'normalized-lp-context', 'normalized-lp-v2', 'normalized-lp-switching', 'normalized-lp-joint', 'normalized-lp-context-joint']:
            from sintef_project.normalized import add_normalized_features
            x, normalized_spec = add_normalized_features(x, ROOT)
            meta['feature_spec']['normalized'] = normalized_spec
        if args.feature_set == 'normalized-lp-context-joint':
            # Preserve the original 521-column context table and its exact order.
            # The appended cache contains only dimensionless predictors.
            from sintef_project.lp_joint_plant import add_lp_joint_features
            x, joint_spec = add_lp_joint_features(x, ROOT)
            meta['feature_spec']['lp_joint'] = joint_spec
        if args.inference_from:
            old = json.loads((args.inference_from / 'manifest.json').read_text())
            columns = old['features']
            if args.feature_set != old['arguments']['feature_set']:
                raise ValueError('Inference feature set differs from training')
            if manifest['data'] != old['data']:
                raise ValueError('Inference data hashes differ from training manifest')
            if meta['feature_spec'] != old['feature_spec']:
                raise ValueError('Inference feature formulas or cache provenance differ from training')
            feature_modules = ['features']
            if args.feature_set not in ['basic', 'full']:
                feature_modules += ['physics']
            if args.feature_set.startswith('normalized'):
                feature_modules += ['normalized']
            if args.feature_set == 'normalized-context' or args.feature_set.startswith('normalized-lp'):
                feature_modules += ['context_features']
            if args.feature_set.startswith('normalized-lp'):
                feature_modules += ['lp_features']
            extra_modules = {'normalized-lp-context': ['lp_context'],
                             'normalized-lp-v2': ['lp_features_v2'],
                             'normalized-lp-switching': ['lp_switching'],
                             'normalized-lp-joint': ['lp_joint_plant', 'lp_head_update', 'lp_shared_loss_update'],
                             'normalized-lp-context-joint': ['lp_context', 'lp_joint_plant', 'lp_head_update', 'lp_shared_loss_update']}
            feature_modules += extra_modules.get(args.feature_set, [])
            for module in feature_modules:
                source = f'sintef_project/{module}.py'
                if source in old['source'] and manifest['source'].get(source) != old['source'][source]:
                    raise ValueError(f'Feature source changed: {source}; use archived source snapshot')
            manifest['model_origin'] = {'manifest': str(args.inference_from / 'manifest.json'),
                                        'manifest_sha256': checksum(args.inference_from / 'manifest.json'),
                                        'checkpoints': {str(p): checksum(p) for p in sorted(args.inference_from.glob('final_*.txt'))}}
        else:
            columns = select_features(x, args.feature_set)
        if not columns:
            raise ValueError(f'No features selected; builder columns: {list(x.columns)}')
        x = x[columns]
        manifest.update(features=columns, feature_spec=meta.get('feature_spec'), shape=list(x.shape))
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        print(f'FEATURES {x.shape}', flush=True)
        y = np.column_stack([labels[u] for u in units]).astype(np.uint8)
        dates = pd.to_datetime(train.starttime, utc=True)
        model_params = dict(objective='binary', n_estimators=args.trees, num_leaves=args.leaves,
                            learning_rate=args.learning_rate, min_child_samples=args.min_child_samples,
                            colsample_bytree=.9, subsample=1.0, reg_lambda=1.0,
                            max_bin=255, n_jobs=args.threads, random_state=args.seed,
                            deterministic=True, force_col_wise=True, verbosity=-1)
        manifest['model_parameters'] = old['model_parameters'] if args.inference_from else lgb.LGBMClassifier(**model_params).get_params()
        rng = np.random.default_rng(args.seed)
        for year in ([] if args.inference_from else args.years):
            begin = pd.Timestamp(f'{year}-01-01', tz='UTC')
            end = pd.Timestamp(f'{year+1}-01-01', tz='UTC')
            train_cases = np.flatnonzero(dates < begin - pd.Timedelta(days=6))
            val_cases = np.flatnonzero((dates >= begin) & (dates < end))
            assert dates.iloc[train_cases].max() + pd.Timedelta(hours=167) < begin
            tr = (train_cases[:, None] * 168 + np.arange(168)).ravel()
            va = (val_cases[:, None] * 168 + np.arange(168)).ravel()
            sample = np.sort(rng.choice(tr, size=int(len(tr)*args.sample_fraction), replace=False))
            xt, xv = x.iloc[sample], x.iloc[va]
            pred = np.zeros((len(va), len(units)), dtype=np.float32)
            baseline = np.broadcast_to(y[tr].mean(0), pred.shape)
            fold = {'train_ids': train.iloc[train_cases]['Run No'].tolist(),
                    'validation_ids': train.iloc[val_cases]['Run No'].tolist(),
                    'train_dates': [str(dates.iloc[train_cases].min()), str(dates.iloc[train_cases].max())],
                    'validation_dates': [str(dates.iloc[val_cases].min()), str(dates.iloc[val_cases].max())],
                    'purge_days': 6, 'sampled_training_rows': len(sample),
                    'baseline': metrics(y[va], baseline, units)}
            np.save(out / f'train_sample_{year}.npy', sample)
            for j, unit in enumerate(units):
                model = lgb.LGBMClassifier(**model_params)
                model.fit(xt, y[sample, j])
                pred[:, j] = model.predict_proba(xv)[:, 1]
                model.booster_.save_model(str(out / f'fold{year}_{unit}.txt'))
                print(f'{year} {unit} auc={roc_auc_score(y[va,j],pred[:,j]):.6f}', flush=True)
            fold.update(metrics(y[va], pred, units))
            np.savez_compressed(out / f'validation_{year}.npz', predictions=pred, labels=y[va], case_ids=train.iloc[val_cases]['Run No'].to_numpy(), units=units)
            # Per-month and lead-day diagnostics expose seasonal/horizon failures.
            fold['month_auc'] = {str(m): float(roc_auc_score(y[va][np.repeat(dates.iloc[val_cases].dt.month.to_numpy()==m,168)].ravel(), pred[np.repeat(dates.iloc[val_cases].dt.month.to_numpy()==m,168)].ravel())) for m in range(1,13)}
            fold['lead_day_auc'] = {str(d): float(roc_auc_score(y[va][np.tile(np.arange(168)//24==d,len(val_cases))].ravel(), pred[np.tile(np.arange(168)//24==d,len(val_cases))].ravel())) for d in range(7)}
            manifest['folds'][str(year)] = fold
            print(f'FOLD {year} micro_auc={fold["micro_auc"]:.8f} baseline={fold["baseline"]["micro_auc"]:.8f}', flush=True)
            (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        if args.final or args.inference_from:
            nt = len(train) * 168
            test_x = x.iloc[nt:]
            pred = np.zeros((len(test_x), len(units)), dtype=np.float32)
            if not args.inference_from:
                sample = np.sort(rng.choice(nt, size=int(nt*args.sample_fraction), replace=False))
                np.save(out / 'train_sample_final.npy', sample)
                xt = x.iloc[sample]
            for j, unit in enumerate(units):
                if args.inference_from:
                    checkpoint = args.inference_from / f'final_{unit}.txt'
                    if checksum(checkpoint) != old['artifacts'][checkpoint.name]:
                        raise ValueError(f'Checkpoint hash changed: {checkpoint}')
                    model = lgb.Booster(model_file=str(checkpoint))
                    pred[:, j] = model.predict(test_x, num_threads=args.threads)
                else:
                    model = lgb.LGBMClassifier(**model_params)
                    model.fit(xt, y[sample, j])
                    pred[:, j] = model.predict_proba(test_x)[:, 1]
                    model.booster_.save_model(str(out / f'final_{unit}.txt'))
                    pd.Series(model.feature_importances_, index=columns).sort_values(ascending=False).to_csv(out / f'importance_{unit}.csv')
                print(f'FINAL {unit}', flush=True)
            sample_submission = pd.read_csv(ROOT / 'sintef_project/sample_submission.csv')
            cube = pred.reshape(len(mapping), 168, len(units))
            values = {}
            for col in sample_submission.columns[1:]:
                unit, hour = col.removeprefix('result_committed_').rsplit('_t',1)
                values[col] = cube[:, int(hour), units.index(unit)]
            submission = pd.DataFrame(values)
            submission.insert(0, 'Run No', mapping['Run No'].to_numpy())
            submission = submission.set_index('Run No').loc[sample_submission['Run No']].reset_index()
            assert list(submission.columns) == list(sample_submission.columns)
            assert submission.shape == sample_submission.shape
            assert np.isfinite(submission.iloc[:,1:]).all().all()
            assert ((submission.iloc[:,1:] >= 0) & (submission.iloc[:,1:] <= 1)).all().all()
            submission.to_csv(out / 'submission.csv', index=False, float_format='%.9g')
            np.save(out / 'test_predictions.npy', pred)
            manifest['submission_sha256'] = checksum(out / 'submission.csv')
        manifest['status'] = 'complete'
    except BaseException:
        manifest['status'] = 'failed'
        manifest['error'] = traceback.format_exc()
        raise
    finally:
        manifest['runtime_seconds'] = time.time()-start
        manifest['artifacts'] = {str(p.relative_to(out)): checksum(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'manifest.json'}
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        with (ROOT / 'docs/experiments.md').open('a') as f:
            f.write(f'\n## {args.run_id} — {manifest["timestamp_utc"]}\n\n')
            f.write(f'Status: {manifest["status"]}. Command: `{manifest["command"]}`. Runtime: {manifest["runtime_seconds"]:.1f}s.\n\n')
            f.write(f'Manifest and source snapshot: `artifacts/{args.run_id}/manifest.json`. Feature ablation: {args.feature_set}; fixed-round per-unit LightGBM; hypothesis: use optimizer inputs to learn unit commitments and improve on training unit prevalence.\n\n')
            for year, fold in manifest['folds'].items():
                f.write(f'- {year}: micro AUC **{fold["micro_auc"]:.8f}**, prevalence baseline {fold["baseline"]["micro_auc"]:.8f}; train {fold["train_dates"]}, validation {fold["validation_dates"]}; six-day purge.\n')
            if 'error' in manifest:
                f.write(f'\nFailure: `{manifest["error"].splitlines()[-1]}`\n')
            f.write('\nKaggle submission ID, score, rank and observation time: none (not submitted). Seeds, complete model configuration, data/source/artifact hashes, environment, exact IDs and feature definitions are in the manifest.\n')

if __name__ == '__main__':
    main()
