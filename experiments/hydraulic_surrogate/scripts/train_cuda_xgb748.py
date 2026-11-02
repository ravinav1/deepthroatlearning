"""Portable GPU-only full14-unit748-feature XGBoost ablation, one explicit year per run."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import io
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
import warnings

for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[variable] = '2'

PARAMETERS = dict(n_estimators=600, max_depth=6, learning_rate=.05, min_child_weight=20,
                  reg_lambda=4., reg_alpha=0., max_bin=128, subsample=.9,
                  colsample_bytree=.9, colsample_bylevel=1., colsample_bynode=1.,
                  random_state=4173, n_jobs=2, objective='binary:logistic',
                  tree_method='hist', device='cuda:1', eval_metric='logloss',
                  grow_policy='depthwise', max_delta_step=0., validate_parameters=True,
                  verbosity=0)
WEIGHTS = [0., .1, .25, .5]


def require(condition: bool, message: str) -> None:
    """Fail explicitly rather than changing device, data or protocol."""
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    """Return a streaming content SHA256."""
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def clean_json(value):
    """Represent nonfinite metadata explicitly rather than emitting invalid JSON."""
    import math
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def require_gpu_config(config: dict) -> None:
    """Reject XGBoost's silent device fallback after actual booster execution."""
    learner = config['learner']
    require(learner['generic_param']['device'] == PARAMETERS['device'], 'XGBoost device changed or fell back to CPU')
    require('grow_gpu_hist' in json.dumps(learner['gradient_booster']),
            'CUDA histogram updater was not selected; no alternate updater allowed')


def runtime_environment(xgb) -> dict:
    """Require a CUDA-enabled build, visible NVIDIA driver and offline uv venv."""
    uv = shutil.which('uv')
    require(uv is not None, 'Preinstalled uv is required; do not install/download it in this runner')
    require(sys.prefix != sys.base_prefix, 'Use the supplied offline uv system-site-packages venv launcher')
    cfg = Path(sys.prefix) / 'pyvenv.cfg'
    require(cfg.is_file() and 'include-system-site-packages = true' in cfg.read_text().lower(),
            'Expected an offline uv venv reusing preinstalled system packages')
    build = xgb.build_info()
    require(build.get('USE_CUDA') is True, 'Installed XGBoost was built without CUDA; CPU fallback forbidden')
    require(int(xgb.__version__.split('.')[0]) >= 2, 'XGBoost>=2 required for explicit device=cuda protocol')
    nvidia = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
                            capture_output=True, text=True)
    require(nvidia.returncode == 0 and len(nvidia.stdout.strip().splitlines()) >= 2,
            'The fixedcuda:1 protocol requires two visible NVIDIA devices')
    return {'xgboost': xgb.__version__, 'xgboost_build_info': clean_json(build),
            'nvidia_smi': nvidia.stdout.strip(), 'requested_device': PARAMETERS['device'], 'cpu_fallback': False,
            'uv': subprocess.run([uv, '--version'], capture_output=True, text=True, check=True).stdout.strip(),
            'venv_configuration': cfg.read_text()}


def reconstruct_sample(dates, year: int, baseline: dict, np, pd):
    """Reproduce original sequential RNG calls and require exact stored NPY bytes."""
    rng = np.random.default_rng(4173)
    sample = None
    for candidate_year in [2021, 2022]:
        begin = pd.Timestamp(f'{candidate_year}-01-01', tz='UTC')
        cases = np.flatnonzero(dates < begin - pd.Timedelta(days=6))
        rows = (cases[:, None] * 168 + np.arange(168)).ravel()
        sample = np.sort(rng.choice(rows, size=int(len(rows) * .4), replace=False))
        if candidate_year == year:
            break
    require(year in [2021, 2022] and sample is not None, 'Invalid reconstruction year')
    payload = io.BytesIO()
    np.save(payload, sample)
    expected = baseline['artifacts'][f'train_sample_{year}.npy']
    require(hashlib.sha256(payload.getvalue()).hexdigest() == expected,
            'Reconstructed training sample differs from exact baseline NPY; no fallback resampling permitted')
    return sample


def validate_lock(path: Path, expected: str, protocol: dict, protocol_sha: str) -> dict:
    """Require the fixed2021 global choice to beat both anchor and matched664 control."""
    require(len(expected) == 64 and digest(path) == expected, 'Explicit2021 lock SHA256 mismatch')
    lock = json.loads(path.read_text())
    require(lock['version'] == 'cuda_xgb748_2021_blend_lock_v1' and lock['selection_year'] == 2021,
            'Unexpected2021 selection-lock schema')
    require(lock['training_protocol_sha256'] == protocol_sha, 'Lock uses another training protocol')
    require(lock['candidate_weights'] == WEIGHTS and lock['selected_weight'] in WEIGHTS,
            'Fixed blend grid changed')
    require(lock['selector_sha256'] == protocol['local_selector_sha256'], 'Selector identity changed')
    scores = lock['scores_2021']
    require(set(scores) == {str(w) for w in WEIGHTS}
            and all(isinstance(v, (int, float)) and 0 <= v <= 1 for v in scores.values()),
            'Incomplete or invalid2021 grid')
    selected = lock['selected_weight']
    require(selected == max(WEIGHTS, key=lambda w: scores[str(w)])
            and lock['selected_2021_micro_auc'] == scores[str(selected)]
            and lock['anchor_2021_micro_auc'] == scores['0.0'], 'Inconsistent2021 selection')
    require(lock['training_2021_complete'] is True and lock['eligible_for_2022'] is True
            and selected > 0 and scores[str(selected)] > max(scores['0.0'], lock['control_2021_micro_auc']),
            '2021 did not beat anchor and matched664 control; no2022 training')
    require(lock['anchor_prediction_sha256'] == protocol['anchor']['2021']['prediction_sha256']
            and lock['control_prediction_sha256'] == protocol['control']['2021']['prediction_sha256'],
            'Anchor or matched control changed')
    for name in ['gpu_run_manifest_sha256', 'gpu_validation_2021_sha256']:
        require(isinstance(lock[name], str) and len(lock[name]) == 64
                and all(c in '0123456789abcdef' for c in lock[name]), 'Missing2021 provenance')
    return lock


def smoke(xgb, np, out: Path) -> dict:
    """Verify actual CUDA fitting and saved-model inference on synthetic rows only."""
    rng = np.random.default_rng(4173)
    x = rng.standard_normal((2048, 748), dtype=np.float32)
    y = rng.integers(0, 2, size=2048, dtype=np.uint8)
    params = {**PARAMETERS, 'n_estimators': 8, 'max_depth': 3}
    model = xgb.XGBClassifier(**params)
    started = time.perf_counter()
    model.fit(x, y)
    config = json.loads(model.get_booster().save_config())
    require_gpu_config(config)
    matrix = xgb.DMatrix(x, nthread=2)
    predictions = model.get_booster().predict(matrix)
    checkpoint = out / 'synthetic_smoke.ubj'
    model.save_model(checkpoint)
    restored = xgb.Booster()
    restored.load_model(checkpoint)
    restored.set_param({'device': PARAMETERS['device'], 'nthread': 2, 'tree_method': 'hist'})
    replay = restored.predict(matrix)
    np.testing.assert_array_equal(predictions, replay)
    require_gpu_config(json.loads(restored.save_config()))
    require(model.get_booster().num_boosted_rounds() == 8, 'Unexpected synthetic tree count')
    return {'synthetic_only': True, 'shape': [2048, 748], 'rounds': 8,
            'seconds': time.perf_counter() - started, 'native_config': config,
            'checkpoint_sha256': digest(checkpoint), 'exact_replay_rows': 2048,
            'note': 'No competition data read; does not establish real-data speed, memory or accuracy.'}


def train_fold(xgb, np, pd, loader, stage: Path, year: int, protocol: dict,
               out: Path, report: dict) -> None:
    """Train14 fixed GPU learners from one verified chronological fold."""
    from sklearn.metrics import roc_auc_score
    x, labels, train, _, meta, baseline = loader.build_verified_features(stage, year, protocol)
    units = meta['units']
    require(units == protocol['units'] and len(units) == 14, 'Unexpected unit order')
    dates = pd.to_datetime(train.starttime, utc=True)
    begin, end = pd.Timestamp(f'{year}-01-01', tz='UTC'), pd.Timestamp(f'{year+1}-01-01', tz='UTC')
    tr = np.flatnonzero(dates < begin - pd.Timedelta(days=6))
    va = np.flatnonzero((dates >= begin) & (dates < end))
    require(dates.iloc[tr].max() + pd.Timedelta(hours=167) < begin, 'Training targets overlap validation')
    inherited = baseline['folds'][str(year)]
    require(train.iloc[tr]['Run No'].tolist() == inherited['train_ids']
            and train.iloc[va]['Run No'].tolist() == inherited['validation_ids'], 'Purged IDs differ from baseline')
    sample = reconstruct_sample(dates, year, baseline, np, pd)
    require(np.isin(sample // 168, tr).all() and len(np.unique(sample)) == len(sample), 'Invalid sampled rows')
    np.save(out / f'train_sample_{year}.npy', sample)
    va_rows = (va[:, None] * 168 + np.arange(168)).ravel()
    y = np.column_stack([labels[unit] for unit in units]).astype(np.uint8)
    del labels
    x_train, x_val = x.iloc[sample], x.iloc[va_rows]
    del x
    validation_matrix = xgb.DMatrix(x_val, feature_names=protocol['feature_columns'], nthread=2)
    prediction = np.empty((len(va_rows), 14), dtype=np.float32)
    report.update(feature_spec=meta['feature_spec'], features=protocol['feature_columns'], units=units,
                  train_ids=inherited['train_ids'], validation_ids=inherited['validation_ids'],
                  train_dates=inherited['train_dates'], validation_dates=inherited['validation_dates'],
                  purge_days=6, sampled_training_rows=len(sample),
                  sample_sha256=digest(out / f'train_sample_{year}.npy'),
                  baseline_reference_sha256=digest(stage / 'artifacts/lgb-normalized-lp-context-joint-001/manifest.json'),
                  preprocessing='Exact664 prefix plus84 proven integer water-budget columns; no learned scaling', models={})
    for j, unit in enumerate(units):
        started = time.perf_counter()
        model = xgb.XGBClassifier(**PARAMETERS)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            model.fit(x_train, y[sample, j])
            booster = model.get_booster()
            config = json.loads(booster.save_config())
            require_gpu_config(config)
            require(booster.num_boosted_rounds() == 600 and booster.feature_names == protocol['feature_columns'],
                    'Unexpected tree count or feature order')
            prediction[:, j] = booster.predict(validation_matrix)
            require_gpu_config(json.loads(booster.save_config()))
            checkpoint = out / f'fold{year}_{unit}.ubj'
            model.save_model(checkpoint)
            restored = xgb.Booster()
            restored.load_model(checkpoint)
            restored.set_param({'device': PARAMETERS['device'], 'nthread': 2, 'tree_method': 'hist'})
            np.testing.assert_array_equal(restored.predict(validation_matrix), prediction[:, j])
            require_gpu_config(json.loads(restored.save_config()))
        require(np.isfinite(prediction[:, j]).all() and prediction[:, j].min() >= 0
                and prediction[:, j].max() <= 1, 'Invalid GPU probabilities')
        native_path = out / f'config_{year}_{unit}.json'
        native_path.write_text(json.dumps(config, indent=2))
        report['models'][unit] = {'checkpoint': checkpoint.name, 'checkpoint_sha256': digest(checkpoint),
                                  'native_config_sha256': digest(native_path),
                                  'expanded_parameters': clean_json(model.get_params()),
                                  'runtime_seconds': time.perf_counter() - started,
                                  'raw_unit_auc': float(roc_auc_score(y[va_rows, j], prediction[:, j])),
                                  'checkpoint_replay_max_error': 0., 'replay_rows': len(va_rows),
                                  'warnings': [str(item.message) for item in captured]}
        (out / 'manifest.json').write_text(json.dumps(report, indent=2, allow_nan=False))
        print('MODEL', year, unit, report['models'][unit]['raw_unit_auc'], flush=True)
        del model, booster, restored
    np.savez_compressed(out / f'validation_{year}.npz', predictions=prediction, labels=y[va_rows],
                        case_ids=train.iloc[va]['Run No'].to_numpy(), units=np.asarray(units, dtype=str))
    report.update(raw_micro_auc=float(roc_auc_score(y[va_rows].ravel(), prediction.ravel())),
                  validation_sha256=digest(out / f'validation_{year}.npz'),
                  completed_units=14, completed_rounds_per_unit=600,
                  postprocessing='none; raw probabilities returned for local fixed-grid selection')


def main() -> None:
    """Perform either a synthetic smoke or one separately authorized GPU fold."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['smoke', 'fit'], default='smoke')
    parser.add_argument('--year', type=int, choices=[2021, 2022])
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--shared-loader', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--output-root', type=Path, default=Path('/kaggle/working/xgb748-runs'))
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--supplement', type=Path)
    parser.add_argument('--shop-bundle', type=Path)
    parser.add_argument('--integer-bundle', type=Path)
    parser.add_argument('--stage-root', type=Path, default=Path('/kaggle/working/tdt4173-xgb748-inputs'))
    parser.add_argument('--smoke-manifest', type=Path)
    parser.add_argument('--selection-lock', type=Path)
    parser.add_argument('--selection-lock-sha256')
    args = parser.parse_args()
    out = args.output_root / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    report = {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'run_id': args.run_id,
              'command': shlex.join([sys.executable, *sys.argv]), 'mode': args.mode, 'year': args.year,
              'seed': 4173, 'model_parameters_requested': PARAMETERS, 'status': 'initializing',
              'real_feature_table_loaded': False, 'early_stopping': None,
              'environment': {'python': sys.version, 'platform': platform.platform(), 'packages': {}},
              'selection': 'Local2021 fixedglobal grid only after downloading OOF; no anchor probabilities uploaded',
              'kaggle_submission': None}
    try:
        protocol_sha = digest(args.protocol)
        protocol = json.loads(args.protocol.read_text())
        require(protocol['version'] == 'cuda_xgb_integer748_v1', 'Unexpected protocol version')
        require(protocol['trainer_sha256'] == digest(Path(__file__)), 'Trainer hash differs from protocol')
        require(protocol['shared_loader_sha256'] == digest(args.shared_loader), 'Shared staging-loader hash changed')
        require(protocol['model_parameters'] == PARAMETERS and protocol['candidate_weights'] == WEIGHTS
                and len(protocol['feature_columns']) == 748 and len(protocol['units']) == 14,
                'Frozen model or schema constants changed')
        report.update(protocol_sha256=protocol_sha, trainer_sha256=digest(Path(__file__)),
                      shared_loader_sha256=digest(args.shared_loader))
        if args.mode == 'fit':
            require(args.year is not None and args.bundle is not None and args.supplement is not None and args.shop_bundle is not None and args.integer_bundle is not None and args.smoke_manifest is not None,
                    'Fit requires one explicit year, verified bundle and successful CUDA smoke')
            if args.year == 2022:
                require(args.selection_lock is not None and args.selection_lock_sha256 is not None,
                        '2022 requires external local2021 selection lock and explicit hash')
                report['selection_lock'] = validate_lock(args.selection_lock, args.selection_lock_sha256, protocol, protocol_sha)
                report['selection_lock_sha256'] = digest(args.selection_lock)
            else:
                require(args.selection_lock is None, '2021 must precede blend selection')
        else:
            require(args.year is None and args.bundle is None and args.supplement is None and args.shop_bundle is None and args.integer_bundle is None, 'Synthetic smoke must not access competition inputs')
        for name in ['xgboost', 'numpy', 'pandas', 'scipy', 'scikit-learn', 'pyyaml']:
            report['environment']['packages'][name] = importlib.metadata.version(name)
        import numpy as np
        import pandas as pd
        import xgboost as xgb
        report['cuda'] = runtime_environment(xgb)
        if args.mode == 'smoke':
            report['smoke'] = smoke(xgb, np, out)
        else:
            prior_smoke = json.loads(args.smoke_manifest.read_text())
            require(prior_smoke['status'] == 'complete' and prior_smoke['mode'] == 'smoke'
                    and prior_smoke['protocol_sha256'] == protocol_sha and prior_smoke['smoke']['synthetic_only'] is True
                    and prior_smoke['cuda'] == report['cuda'], 'Successful matching CUDA smoke required')
            report['smoke_manifest_sha256'] = digest(args.smoke_manifest)
            spec = importlib.util.spec_from_file_location('tdt4173_shared_gpu_loader', args.shared_loader)
            loader = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(loader)
            bundle, supplement, shop_bundle, integer_bundle, feature_stage = loader.stage_inputs(args.bundle, args.supplement, args.shop_bundle, args.integer_bundle, args.stage_root, protocol)
            report['shop_bundle'] = shop_bundle
            report['integer_bundle'] = integer_bundle
            report['bundle'], report['supplement'] = bundle, supplement
            report['stage_root'] = str(feature_stage)
            report['real_feature_table_loaded'] = True
            train_fold(xgb, np, pd, loader, feature_stage, args.year, protocol, out, report)
        report['status'] = 'complete'
    except BaseException:
        report['status'] = 'failed'
        report['error'] = traceback.format_exc()
        raise
    finally:
        shutil.copy2(__file__, out / 'train_cuda_xgb748.py')
        if args.protocol.is_file():
            shutil.copy2(args.protocol, out / 'protocol.json')
        if args.shared_loader.is_file():
            shutil.copy2(args.shared_loader, out / 'shared_loader.py')
            core_path = args.shared_loader.with_name('train_cuda_tcn.py')
            if core_path.is_file():
                shutil.copy2(core_path, out / 'core_loader.py')
            inherited_loader = args.shared_loader.with_name('cuda_xgb608_loader.py')
            if inherited_loader.is_file():
                shutil.copy2(inherited_loader, out / 'loader608.py')
            parent_loader = args.shared_loader.with_name('cuda_xgb664_loader.py')
            if parent_loader.is_file():
                shutil.copy2(parent_loader, out / 'loader664.py')
        report['runtime_seconds'] = time.time() - started
        report['artifacts'] = {str(path.relative_to(out)): digest(path) for path in out.rglob('*')
                               if path.is_file() and path.name != 'manifest.json'}
        (out / 'manifest.json').write_text(json.dumps(report, indent=2, allow_nan=False))
        print(json.dumps({'status': report['status'], 'mode': args.mode, 'year': args.year,
                          'manifest': str(out / 'manifest.json'), 'runtime_seconds': report['runtime_seconds']}), flush=True)


if __name__ == '__main__':
    main()
