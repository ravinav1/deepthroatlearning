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
    require(generic['device'] == parameters['device'] and 'grow_gpu_hist' in json.dumps(booster), 'Native GPU fallback/updater changed')
    require(int(generic['nthread']) == 2 and int(generic['seed']) == 4173, 'Native seed/thread settings changed')
    require(learner['objective']['name'] == 'binary:logistic'
            and int(learner['learner_model_param']['num_feature']) == 608, 'Native objective/feature count changed')
    require(int(booster['gbtree_model_param']['num_trees']) == parameters['n_estimators']
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
                  protocol: dict, year: int, workspace: Path, protocol_sha: str) -> dict:
    """Verify all models, rows, labels and source hashes without model loading."""
    manifest_path = directory / 'manifest.json'
    require(digest(manifest_path) == manifest_sha and digest(smoke_path) == smoke_sha,
            'Downloaded fold/smoke manifest differs from hosted SHA256')
    run, smoke = json.loads(manifest_path.read_text()), json.loads(smoke_path.read_text())
    require(run['status'] == 'complete' and run['mode'] == 'fit' and run['year'] == year,
            'Expected completed requested fold')
    require(run['protocol_sha256'] == smoke['protocol_sha256'] == protocol_sha
            and run['trainer_sha256'] == protocol['trainer_sha256']
            and run['shared_loader_sha256'] == protocol['shared_loader_sha256'], 'Source/protocol identity changed')
    require(smoke['status'] == 'complete' and smoke['mode'] == 'smoke'
            and smoke['smoke']['synthetic_only'] is True and smoke['smoke']['exact_replay_rows'] == 2048
            and smoke['smoke']['shape'] == [2048, 608] and smoke['smoke']['rounds'] == 8
            and smoke['trainer_sha256'] == protocol['trainer_sha256']
            and smoke['shared_loader_sha256'] == protocol['shared_loader_sha256']
            and run['smoke_manifest_sha256'] == smoke_sha and run['cuda'] == smoke['cuda'],
            'Original successful matching synthetic GPU smoke not established')
    require(run['cuda']['requested_device'] == 'cuda:0' and run['cuda']['cpu_fallback'] is False
            and run['cuda']['xgboost_build_info']['USE_CUDA'] is True, 'CUDA build/device changed')
    smoke_expected = {'train_cuda_xgb608_capacity.py', 'shared_loader.py', 'core_loader.py',
                      'protocol.json', 'synthetic_smoke.ubj'}
    require(set(smoke['artifacts']) == smoke_expected, 'Smoke artifact allowlist differs')
    for name, expected_hash in smoke['artifacts'].items():
        require(digest(smoke_path.parent / name) == expected_hash, f'Smoke artifact changed: {name}')
    for name, expected_hash in [('train_cuda_xgb608_capacity.py', protocol['trainer_sha256']),
                               ('shared_loader.py', protocol['shared_loader_sha256']),
                               ('core_loader.py', protocol['core_loader_sha256']),
                               ('protocol.json', protocol_sha)]:
        require(digest(smoke_path.parent / name) == expected_hash, f'Smoke source changed: {name}')
    require(smoke['smoke']['checkpoint_sha256'] == smoke['artifacts']['synthetic_smoke.ubj'],
            'Smoke checkpoint hashes disagree')
    verify_native_config(smoke['smoke']['native_config'], {**protocol['model_parameters'], 'n_estimators': 8, 'max_depth': 3})
    require(run['seed'] == 4173 and run['model_parameters_requested'] == protocol['model_parameters']
            and run['early_stopping'] is None, 'Fixed model settings changed')
    units = protocol['units']
    expected_names = {'train_cuda_xgb608_capacity.py', 'shared_loader.py', 'core_loader.py', 'protocol.json',
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
    for name, expected in [('train_cuda_xgb608_capacity.py', protocol['trainer_sha256']),
                           ('shared_loader.py', protocol['shared_loader_sha256']),
                           ('core_loader.py', protocol['core_loader_sha256']), ('protocol.json', protocol_sha)]:
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
    original_spec, portable_spec = json.loads(json.dumps(baseline['feature_spec'])), json.loads(json.dumps(run['feature_spec']))
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
    require(run['completed_units'] == 14 and run['completed_rounds_per_unit'] == 1200
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

