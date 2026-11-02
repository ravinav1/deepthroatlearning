"""Verify hosted GPU-XGB transfer bytes before invoking the unchanged selector."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import platform
import shlex
import shutil
import stat
import sys
import time
import traceback
import zipfile

import numpy as np
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

PROTOCOL_SHA = '789c2a6419c526079e18ea83ac5dd1143ef0b2c590e522b163ddf4e890a72401'
ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    """Stop before selection when downloaded provenance differs."""
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    """Return SHA256 without loading an entire archive/checkpoint."""
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            result.update(block)
    return result.hexdigest()


def safe_name(name: str) -> Path:
    """Reject noncanonical archive members and path traversal."""
    value = PurePosixPath(name)
    require(name and not value.is_absolute() and '..' not in value.parts
            and '\\' not in name and str(value) == name and name != '.', 'Unsafe archive path')
    return Path(*value.parts)


def import_archive(archive_path: Path, expected_sha: str, destination: Path,
                   gpu_name: str, smoke_name: str) -> None:
    """Extract only the expected two run namespaces after validating every member."""
    require(digest(archive_path) == expected_sha, 'Transfer archive differs from hosted SHA256')
    require(not destination.exists(), 'Download destination already exists')
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        require(len({i.filename for i in infos}) == len(infos), 'Duplicate archive names')
        for item in infos:
            path = safe_name(item.filename)
            require((path.parts[0] in [gpu_name, smoke_name] and len(path.parts) >= 2)
                    or item.filename == f'{gpu_name}.log',
                    'Unexpected archive namespace')
            require(not item.is_dir() and not stat.S_ISLNK(item.external_attr >> 16), 'Archive directories/symlinks forbidden')
        destination.mkdir()
        for item in infos:
            path = destination / safe_name(item.filename)
            path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(item) as incoming, path.open('xb') as outgoing:
                shutil.copyfileobj(incoming, outgoing)


def verify_native_config(config: dict, parameters: dict) -> None:
    """Independently check native GPU, tree count, objective and learning recipe."""
    learner = config['learner']
    generic, booster = learner['generic_param'], learner['gradient_booster']
    require(generic['device'] == 'cuda:1' and 'grow_gpu_hist' in json.dumps(booster), 'Native GPU fallback/updater changed')
    require(int(generic['nthread']) == 2 and int(generic['seed']) == 4173, 'Native seed/thread settings changed')
    require(learner['objective']['name'] == 'binary:logistic'
            and int(learner['learner_model_param']['num_feature']) == 608, 'Native objective/feature count changed')
    require(int(booster['gbtree_model_param']['num_trees']) == 600
            and int(booster['gbtree_model_param']['num_parallel_tree']) == 1, 'Native boosted-round count changed')
    tree = booster['tree_train_param']
    mapping = {'eta': 'learning_rate', 'max_depth': 'max_depth', 'min_child_weight': 'min_child_weight',
               'lambda': 'reg_lambda', 'alpha': 'reg_alpha', 'max_bin': 'max_bin',
               'subsample': 'subsample', 'colsample_bytree': 'colsample_bytree',
               'colsample_bylevel': 'colsample_bylevel', 'colsample_bynode': 'colsample_bynode',
               'max_delta_step': 'max_delta_step'}
    for native, requested in mapping.items():
        require(math.isclose(float(tree[native]), parameters[requested], abs_tol=1e-7, rel_tol=1e-7),
                f'Native learning parameter changed: {native}')
    require(tree['grow_policy'] == parameters['grow_policy'], 'Native grow policy changed')


def verify_return(directory: Path, smoke_path: Path, manifest_sha: str, smoke_sha: str,
                  protocol: dict, year: int, workspace: Path) -> dict:
    """Verify all models, rows, labels and source hashes without model loading."""
    manifest_path = directory / 'manifest.json'
    require(digest(manifest_path) == manifest_sha and digest(smoke_path) == smoke_sha,
            'Downloaded fold/smoke manifest differs from hosted SHA256')
    run, smoke = json.loads(manifest_path.read_text()), json.loads(smoke_path.read_text())
    require(run['status'] == 'complete' and run['mode'] == 'fit' and run['year'] == year,
            'Expected completed requested fold')
    require(run['protocol_sha256'] == smoke['protocol_sha256'] == PROTOCOL_SHA
            and run['trainer_sha256'] == protocol['trainer_sha256']
            and run['shared_loader_sha256'] == protocol['shared_loader_sha256'], 'Source/protocol identity changed')
    require(smoke['status'] == 'complete' and smoke['mode'] == 'smoke'
            and smoke['smoke']['synthetic_only'] is True and smoke['smoke']['exact_replay_rows'] == 2048
            and smoke['smoke']['shape'] == [2048, 608] and smoke['smoke']['rounds'] == 8
            and smoke['trainer_sha256'] == protocol['trainer_sha256']
            and smoke['shared_loader_sha256'] == protocol['shared_loader_sha256']
            and run['smoke_manifest_sha256'] == smoke_sha and run['cuda'] == smoke['cuda'],
            'Original successful matching synthetic GPU smoke not established')
    require(run['cuda']['requested_device'] == 'cuda:1' and run['cuda']['cpu_fallback'] is False
            and run['cuda']['xgboost_build_info']['USE_CUDA'] is True, 'CUDA build/device changed')
    require(run['seed'] == 4173 and run['model_parameters_requested'] == protocol['model_parameters']
            and run['early_stopping'] is None, 'Fixed model settings changed')
    units = protocol['units']
    expected_names = {'train_cuda_xgb608.py', 'shared_loader.py', 'core_loader.py', 'protocol.json',
                      f'train_sample_{year}.npy', f'validation_{year}.npz'}
    for unit in units:
        expected_names.update([f'fold{year}_{unit}.ubj', f'config_{year}_{unit}.json'])
    files = list(directory.rglob('*'))
    require(not any(path.is_symlink() for path in files), 'Returned directory contains symlink')
    actual_names = {path.relative_to(directory).as_posix() for path in files if path.is_file()}
    require(set(run['artifacts']) == expected_names and actual_names == expected_names | {'manifest.json'},
            'Returned artifact allowlist differs from frozen runner')
    for name, expected in run['artifacts'].items():
        require(digest(directory / name) == expected, f'Returned artifact changed: {name}')
    for name, expected in [('train_cuda_xgb608.py', protocol['trainer_sha256']),
                           ('shared_loader.py', protocol['shared_loader_sha256']),
                           ('core_loader.py', protocol['core_loader_sha256']), ('protocol.json', PROTOCOL_SHA)]:
        require(digest(directory / name) == expected, f'Returned frozen source changed: {name}')
    bundle_path = workspace / 'artifacts/kaggle-neural-bundle-001/payload/bundle_manifest.json'
    baseline_path = workspace / 'artifacts/lgb-normalized-lp-context-joint-001/manifest.json'
    require(digest(bundle_path) == protocol['bundle_manifest_sha256']
            and run['bundle'] == json.loads(bundle_path.read_text()), 'Sealed bundle provenance changed')
    require(digest(baseline_path) == run['baseline_reference_sha256'] == protocol['baseline_manifest_sha256'],
            'Frozen baseline identity changed')
    supplement_path = workspace / 'artifacts/cuda-xgb608-package-001/supplement/bundle_manifest.json'
    require(digest(supplement_path) == protocol['supplement_manifest_sha256']
            and run['supplement'] == json.loads(supplement_path.read_text()), 'Supplement provenance changed')
    baseline = json.loads(baseline_path.read_text())
    expected = protocol['folds'][str(year)]
    require(run['train_ids'] == expected['train_ids'] and run['validation_ids'] == expected['validation_ids']
            and run['train_dates'] == expected['train_dates'] and run['validation_dates'] == expected['validation_dates']
            and run['purge_days'] == 6 and run['features'] == protocol['feature_columns']
            and run['units'] == units, 'Feature schema or purged chronological fold changed')
    original_spec, portable_spec = baseline['feature_spec'], run['feature_spec']
    for item in [original_spec, portable_spec]:
        item['lp'].pop('cache_directory', None)
        item['lp_joint'].pop('cache_directory', None)
    require(original_spec == portable_spec, 'Feature formulas/source/cache hashes differ')
    sample_path = directory / f'train_sample_{year}.npy'
    require(digest(sample_path) == protocol['sample_sha256'][str(year)] == run['sample_sha256'],
            'Exact frozen training sample changed')
    sample = np.load(sample_path, allow_pickle=False)
    require(len(sample) == run['sampled_training_rows'] and np.issubdtype(sample.dtype, np.integer),
            'Recorded sample size/type changed')
    require(run['completed_units'] == 14 and run['completed_rounds_per_unit'] == 600
            and set(run['models']) == set(units), 'Incomplete model ensemble')
    for unit in units:
        info = run['models'][unit]
        checkpoint, config_name = f'fold{year}_{unit}.ubj', f'config_{year}_{unit}.json'
        require(info['checkpoint'] == checkpoint and info['checkpoint_sha256'] == run['artifacts'][checkpoint]
                and info['native_config_sha256'] == run['artifacts'][config_name], 'Model artifact hashes disagree')
        require(info['checkpoint_replay_max_error'] == 0. and info['replay_rows'] == len(expected['validation_ids']) * 168,
                'Hosted full-validation checkpoint replay missing')
        for key, value in protocol['model_parameters'].items():
            require(info['expanded_parameters'][key] == value, f'Expanded model parameter changed: {unit}/{key}')
        verify_native_config(json.loads((directory / config_name).read_text()), protocol['model_parameters'])
    values_path = directory / f'validation_{year}.npz'
    require(digest(values_path) == run['validation_sha256'], 'OOF hashes disagree')
    with np.load(values_path, allow_pickle=False) as values:
        require(set(values.files) == {'predictions', 'labels', 'case_ids', 'units'}, 'OOF schema changed')
        raw, labels, ids = values['predictions'], values['labels'], values['case_ids']
        require(values['units'].tolist() == units and ids.tolist() == expected['validation_ids'], 'OOF ordering changed')
    require(raw.dtype == np.float32 and raw.shape == labels.shape == (len(ids)*168, 14)
            and np.isfinite(raw).all() and raw.min() >= 0 and raw.max() <= 1
            and np.isin(labels, [0, 1]).all(), 'Invalid OOF probabilities/labels')
    raw_auc = float(roc_auc_score(labels.ravel(), raw.ravel()))
    require(abs(raw_auc - run['raw_micro_auc']) <= 1e-12, 'Independent pooled AUC differs')
    for j, unit in enumerate(units):
        require(abs(float(roc_auc_score(labels[:, j], raw[:, j])) - run['models'][unit]['raw_unit_auc']) <= 1e-12,
                f'Independent unit AUC differs: {unit}')
    return {'status': 'complete', 'raw_micro_auc': raw_auc, 'models_verified': 14,
            'artifact_hashes': run['artifacts'], 'hosted_checkpoint_replay_rows_per_model': len(ids)*168,
            'local_checkpoint_inference': False, 'source': 'frozen runner replay proof and downloaded artifact/config verification',
            'case_ids': ids.tolist(), 'units': units}


def main() -> None:
    """Import, verify, snapshot and execute the protocol-pinned local selector."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, default=ROOT)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--archive-sha256', required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--smoke-manifest-sha256', required=True)
    parser.add_argument('--year', type=int, choices=[2021, 2022], required=True)
    parser.add_argument('--gpu-run-name', required=True)
    parser.add_argument('--smoke-run-name', default='cuda-xgb608-smoke-001')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--selection-run-id', required=True)
    parser.add_argument('--selection-lock', type=Path)
    parser.add_argument('--selection-lock-sha256')
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    for name in [args.gpu_run_name, args.smoke_run_name, args.run_id, args.selection_run_id]:
        require(safe_name(name).name == name, 'Expected single directory name')
    out = workspace / 'artifacts' / args.run_id
    out.mkdir(exist_ok=False)
    start = time.time()
    report = {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'run_id': args.run_id,
              'command': shlex.join([sys.executable, *sys.argv]), 'year': args.year, 'status': 'initializing',
              'training': 'none', 'environment': {'python': sys.version, 'platform': platform.platform(),
                  'packages': {p: importlib.metadata.version(p) for p in ['numpy', 'scikit-learn']}}, 'kaggle': None}
    try:
        protocol_path = workspace / 'docs/cuda_xgb608_protocol.json'
        require(digest(protocol_path) == PROTOCOL_SHA, 'Frozen GPU protocol changed')
        protocol = json.loads(protocol_path.read_text())
        package = workspace / 'artifacts/cuda-xgb608-package-001/source'
        source_keys = {'scripts/select_cuda_xgb608_blend.py': 'local_selector_sha256',
                       'scripts/train_cuda_xgb608.py': 'trainer_sha256',
                       'scripts/cuda_xgb608_loader.py': 'shared_loader_sha256', 'scripts/train_cuda_tcn.py': 'core_loader_sha256',
                       'sintef_project/postprocess.py': 'postprocess_sha256'}
        snapshot = out / 'frozen_source'
        for name, key in source_keys.items():
            source = package / name if (package / name).exists() else workspace / name
            require(digest(source) == protocol[key] and digest(workspace / name) == protocol[key], 'Frozen selector dependency changed')
            target = snapshot / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        for pkg in ['scripts', 'sintef_project']:
            (snapshot / pkg / '__init__.py').write_text('')
        # Execute only exact frozen selector bytes; ROOT redirects data/artifacts, not code.
        sys.path.insert(0, str(snapshot))
        spec = importlib.util.spec_from_file_location('frozen_xgb_selector', snapshot / 'scripts/select_cuda_xgb608_blend.py')
        selector = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(selector)
        selector.ROOT = workspace
        require(digest(Path(sys.modules['scripts.train_cuda_xgb608'].__file__)) == protocol['trainer_sha256']
                and digest(Path(sys.modules['sintef_project.postprocess'].__file__)) == protocol['postprocess_sha256'],
                'Imported selector dependency changed')
        if args.year == 2022:
            require(args.selection_lock is not None and args.selection_lock_sha256 is not None, '2022 lock required')
            lock = selector.validate_lock(args.selection_lock, args.selection_lock_sha256, protocol, PROTOCOL_SHA)
            weights = protocol['candidate_weights']
            require(set(lock['scores_2021']) == {str(w) for w in weights}
                    and all(math.isfinite(v) for v in lock['scores_2021'].values()), 'Invalid frozen selection scores')
            selected = lock['selected_weight']
            require(selected == max(weights, key=lambda w: lock['scores_2021'][str(w)])
                    and lock['selected_2021_micro_auc'] == lock['scores_2021'][str(selected)]
                    and lock['anchor_2021_micro_auc'] == lock['scores_2021']['0.0'], 'Frozen selection inconsistent')
        imported = out / 'download'
        import_archive(args.archive, args.archive_sha256, imported, args.gpu_run_name, args.smoke_run_name)
        directory = imported / args.gpu_run_name
        smoke_path = imported / args.smoke_run_name / 'manifest.json'
        if args.year == 2022:
            require(digest(directory / 'manifest.json') == args.manifest_sha256, 'Downloaded2022manifest differs')
            returned_lock = json.loads((directory / 'manifest.json').read_text())
            require(returned_lock['selection_lock'] == lock
                    and returned_lock['selection_lock_sha256'] == args.selection_lock_sha256,
                    'Remote2022did not use exact local lock')
        report['verification'] = verify_return(directory, smoke_path, args.manifest_sha256,
                                               args.smoke_manifest_sha256, protocol, args.year, workspace)
        report['transfer_sha256'] = args.archive_sha256
        report['hosted_manifest_sha256'] = args.manifest_sha256
        report['hosted_smoke_sha256'] = args.smoke_manifest_sha256
        sys.argv = [str(snapshot / 'scripts/select_cuda_xgb608_blend.py'), '--year', str(args.year),
                    '--gpu-run', str(directory), '--protocol', str(protocol_path), '--run-id', args.selection_run_id]
        if args.year == 2022:
            sys.argv += ['--selection-lock', str(args.selection_lock), '--selection-lock-sha256', args.selection_lock_sha256]
        selector.main()
        selection_path = workspace / 'artifacts' / args.selection_run_id / 'manifest.json'
        selection = json.loads(selection_path.read_text())
        require(selection['status'] == 'complete', 'Frozen selector failed')
        for name in ['scripts/select_cuda_xgb608_blend.py', 'scripts/train_cuda_xgb608.py', 'sintef_project/postprocess.py']:
            require(digest(selection_path.parent / 'source' / name) == protocol[source_keys[name]], 'Source changed during selection')
        report.update(status='complete', selection_manifest_sha256=digest(selection_path),
                      selection_run=args.selection_run_id, scores=selection['scores'], selected_weight=selection['selected_weight'])
        print(json.dumps({k: report.get(k) for k in ['status', 'selected_weight', 'scores', 'selection_run']}), flush=True)
    except BaseException:
        report['status'] = 'failed'
        report['error'] = traceback.format_exc()
        raise
    finally:
        shutil.copy2(__file__, out / 'verify_cuda_xgb608_return.py')
        report['runtime_seconds'] = time.time()-start
        report['files'] = {p.relative_to(out).as_posix(): digest(p) for p in out.rglob('*')
                           if p.is_file() and '__pycache__' not in p.parts and p.name != 'manifest.json'}
        (out / 'manifest.json').write_text(json.dumps(report, indent=2))
        with (workspace / 'docs/experiments.md').open('a') as stream:
            stream.write(f'\n## {args.run_id} — {report["timestamp_utc"]}\n\nStatus:{report["status"]}. Command:`{report["command"]}`. Strict returned GPU-XGB archive/manifest/smoke/checkpoint/config/sample/source/OOF verifier, then exact protocol-pinned selector; no model fit. Frozen current608 context-joint blend anchor. Scores:{report.get("scores")}; selected:{report.get("selected_weight")}. Full source snapshots, hashes, environment and verification: `artifacts/{args.run_id}/manifest.json`. No Kaggle submission.\n')


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
