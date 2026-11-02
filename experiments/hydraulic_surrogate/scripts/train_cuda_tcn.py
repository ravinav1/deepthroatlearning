"""Portable, gated CUDA TCN experiment; no installation or network operations.

Default mode is a synthetic-only CUDA smoke. Real fitting requires an explicit
single year, a successful matching smoke, a verified input bundle, and (for2022)
an externally produced2021 selection lock. No command trains both folds.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import random
import shlex
import shutil
import stat
import statistics
import subprocess
import sys
import time
import traceback
from typing import Any
from unittest.mock import patch
import zipfile

for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[key] = '2'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
sys.dont_write_bytecode = True
SEED = 4173
FEATURES = 395
HOURS = 168
BATCH = 16
EPOCHS = 24


def require(condition: bool, message: str) -> None:
    """Reject incomplete prerequisites rather than silently changing the run."""
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    """Return a streaming SHA256."""
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def safe_relative(name: str) -> Path:
    """Permit only plain relative POSIX paths inside a payload root."""
    path = PurePosixPath(name)
    require(bool(name) and not path.is_absolute() and '..' not in path.parts and '\\' not in name,
            f'Unsafe payload path: {name}')
    require(str(path) == name and name != '.', f'Noncanonical payload path: {name}')
    return Path(*path.parts)


def stage_bundle(source: Path, destination: Path, manifest_sha256: str,
                 archive_sha256: str | None = None) -> dict:
    """Verify an exact payload allowlist before copying to a separate work root."""
    source, destination = source.resolve(), destination.resolve()
    require(source != destination and (not source.is_dir() or not destination.is_relative_to(source)),
            'Stage destination must be separate from the read-only input')
    zipped = source.is_file()
    if zipped:
        if archive_sha256 is not None:
            require(digest(source) == archive_sha256, 'Input archive checksum mismatch')
        with zipfile.ZipFile(source) as archive:
            infos = archive.infolist()
            require(len({item.filename for item in infos}) == len(infos), 'Duplicate zip members')
            manifest_bytes = archive.read('bundle_manifest.json')
    else:
        manifest_bytes = (source / 'bundle_manifest.json').read_bytes()
    require(hashlib.sha256(manifest_bytes).hexdigest() == manifest_sha256, 'Input bundle manifest checksum mismatch')
    manifest = json.loads(manifest_bytes)
    entries = manifest['payload_files']
    names = [item['path'] for item in entries]
    require(len(names) == len(set(names)) and 'bundle_manifest.json' not in names, 'Invalid bundle allowlist')
    for name in names:
        safe_relative(name)
    allowed = set(names) | {'bundle_manifest.json'}
    reuse = destination.exists() and any(destination.iterdir())
    if reuse:
        found = {str(path.relative_to(destination)) for path in destination.rglob('*') if path.is_file()}
        require(found == allowed and not any(path.is_symlink() for path in destination.rglob('*')), 'Existing stage has unexpected files or symlinks')
    destination.mkdir(parents=True, exist_ok=True)
    if zipped:
        with zipfile.ZipFile(source) as archive:
            require(set(archive.namelist()) == allowed, 'Archive contains missing or unlisted members')
            for item in archive.infolist():
                require(not item.is_dir() and not stat.S_ISLNK(item.external_attr >> 16), 'Archive directories/symlinks are forbidden')
                if not reuse:
                    target = destination / safe_relative(item.filename)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(item) as incoming, target.open('xb') as outgoing:
                        shutil.copyfileobj(incoming, outgoing)
    else:
        found = {str(path.relative_to(source)) for path in source.rglob('*') if path.is_file()}
        require(found == allowed, 'Input directory contains missing or unlisted files')
        require(not any(path.is_symlink() for path in source.rglob('*')), 'Input symlinks are forbidden')
        if not reuse:
            for name in sorted(allowed):
                target = destination / safe_relative(name)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / name, target)
    for item in entries:
        target = destination / safe_relative(item['path'])
        require(target.stat().st_size == item['size_bytes'] and digest(target) == item['sha256'],
                f'Staged payload checksum/size mismatch: {item["path"]}')
    require(digest(destination / 'bundle_manifest.json') == manifest_sha256, 'Staged manifest changed')
    return manifest


def read_targets_through_year(path: Path, year: int, read_csv: Any) -> Any:
    """Read all case metadata but decode targets only through the selected year."""
    import numpy as np
    import pandas as pd
    meta = read_csv(path, usecols=['Run No', 'starttime'])
    dates = pd.to_datetime(meta.starttime, utc=True)
    allowed = dates < pd.Timestamp(f'{year + 1}-01-01', tz='UTC')
    count = int(allowed.sum())
    require(np.array_equal(allowed.to_numpy(), np.arange(len(meta)) < count), 'Expected chronological target prefix')
    prior = read_csv(path, nrows=count)
    later = pd.DataFrame(np.zeros((len(meta) - count, len(prior.columns) - 3), dtype=np.uint8), columns=prior.columns[3:])
    later.insert(0, 'total_calculation_time', 0.)
    later.insert(0, 'starttime', meta.starttime.iloc[count:].to_numpy())
    later.insert(0, 'Run No', meta['Run No'].iloc[count:].to_numpy())
    return pd.concat([prior, later], ignore_index=True)


def training_stats(cube: Any, case_indices: Any) -> tuple[Any, Any]:
    """Streaming float64 population moments from retained training cases only."""
    import numpy as np
    count = 0
    mean = np.zeros(cube.shape[2], dtype=np.float64)
    m2 = np.zeros_like(mean)
    for offset in range(0, len(case_indices), 32):
        block = cube[case_indices[offset:offset + 32]].reshape(-1, cube.shape[2]).astype(np.float64)
        require(np.isfinite(block).all(), 'Nonfinite training features')
        n = len(block)
        local_mean = block.mean(axis=0)
        local_m2 = np.sum((block - local_mean) ** 2, axis=0)
        delta = local_mean - mean
        new_count = count + n
        m2 += local_m2 + delta ** 2 * count * n / new_count
        mean += delta * n / new_count
        count = new_count
    require(count > 0, 'No retained training observations')
    scale = np.sqrt(np.maximum(m2 / count, 0.))
    scale[scale < 1e-6] = 1.
    return mean, scale


def normalized_batch(cube: Any, indices: Any, mean: Any, scale: Any) -> Any:
    """Normalize whole cases, clip, and return contiguous float32 N,C,T input."""
    import numpy as np
    value = np.clip((cube[indices].astype(np.float64) - mean) / scale, -10., 10.).astype(np.float32)
    require(np.isfinite(value).all(), 'Nonfinite normalized batch')
    return np.ascontiguousarray(value.transpose(0, 2, 1))


def build_model(torch: Any, features: int = FEATURES) -> Any:
    """Build the unchanged64-channel centered61-hour residual TCN architecture."""
    nn = torch.nn

    class HourNorm(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(64)

        def forward(self, x: Any) -> Any:
            return self.norm(x.transpose(1, 2)).transpose(1, 2)

    class ResidualBlock(nn.Module):
        def __init__(self, dilation: int) -> None:
            super().__init__()
            layers = []
            for _ in range(2):
                layers.extend([nn.Conv1d(64, 64, 3, padding=dilation, dilation=dilation, padding_mode='replicate'),
                               HourNorm(), nn.ReLU(), nn.Dropout(.1)])
            self.layers = nn.Sequential(*layers)

        def forward(self, x: Any) -> Any:
            return torch.relu(x + self.layers(x))

    return nn.Sequential(nn.Conv1d(features, 64, 1), nn.ReLU(),
                         *[ResidualBlock(d) for d in [1, 2, 4, 8]], nn.Conv1d(64, 14, 1))


def cuda_environment(torch: Any) -> dict:
    """Require genuine CUDA and record all visible devices; use cuda:0 only."""
    require(torch.cuda.is_available(), 'CUDA is required; CPU fallback is forbidden')
    require(torch.version.cuda is not None, 'PyTorch was built without CUDA')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    devices = []
    for index in range(torch.cuda.device_count()):
        prop = torch.cuda.get_device_properties(index)
        devices.append({'index': index, 'name': prop.name, 'total_memory_bytes': prop.total_memory,
                        'capability': [prop.major, prop.minor]})
    return {'torch': torch.__version__, 'cuda_build': torch.version.cuda,
            'cudnn': torch.backends.cudnn.version(), 'devices': devices, 'selected_device': 'cuda:0',
            'deterministic_algorithms': True, 'float32': True, 'amp': False, 'compile': False,
            'cpu_fallback': False, 'CUBLAS_WORKSPACE_CONFIG': os.environ['CUBLAS_WORKSPACE_CONFIG']}


def synthetic_smoke(torch: Any) -> dict:
    """Measure only synthetic full-case updates; no supplied input table is read."""
    model = build_model(torch).to('cuda:0').train()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    require(parameter_count == 126094, 'TCN parameter count changed')
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001, foreach=False)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    x = torch.randn(BATCH, FEATURES, HOURS)
    y = torch.randint(0, 2, (BATCH, 14, HOURS)).float()
    times, losses = [], []
    for step in range(12):
        if step == 2:
            torch.cuda.reset_peak_memory_stats(0)
        torch.cuda.synchronize(0)
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x.to('cuda:0'))
        require(tuple(logits.shape) == (BATCH, 14, HOURS) and logits.device.type == 'cuda', 'Synthetic output shape/device mismatch')
        loss = loss_fn(logits, y.to('cuda:0'))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        torch.cuda.synchronize(0)
        elapsed = time.perf_counter() - start
        require(torch.isfinite(loss).item(), 'Synthetic loss is nonfinite')
        losses.append(float(loss.detach().cpu()))
        if step >= 2:
            times.append(elapsed)
    return {'synthetic_only': True, 'input_shape': [BATCH, FEATURES, HOURS], 'warmup_steps': 2,
            'timed_steps': 10, 'step_seconds': times, 'median_step_seconds': statistics.median(times),
            'max_allocated_bytes': torch.cuda.max_memory_allocated(0),
            'max_reserved_bytes': torch.cuda.max_memory_reserved(0), 'synthetic_losses': losses,
            'parameters': parameter_count, 'output_shape': [BATCH, 14, HOURS],
            'timing_scope': 'host-to-device transfer, forward, backward, clipping, optimizer, synchronization; excludes real feature construction and loading'}


def validate_selection_lock(path: Path, expected_sha: str, protocol: dict, protocol_sha: str) -> dict:
    """Require a locally reviewed2021 choice before the separate2022 command."""
    require(len(expected_sha) == 64 and digest(path) == expected_sha, 'Explicit2021 selection-lock SHA256 mismatch')
    locked = json.loads(path.read_text())
    require(locked['version'] == 'cuda_tcn_2021_blend_lock_v1' and locked['selection_year'] == 2021,
            'Unexpected selection-lock schema/year')
    require(locked['training_protocol_sha256'] == protocol_sha, 'Selection lock uses another training protocol')
    require(locked['anchor_run'] == protocol['anchor']['run_id']
            and locked['anchor_sha256'] == protocol['anchor']['analysis_sha256'], 'Selection lock uses another anchor')
    require(locked['candidate_neural_weights'] == [0., .1, .25, .5]
            and locked['selected_neural_weight'] in locked['candidate_neural_weights'], 'Unexpected blend selection')
    require(locked.get('training_2021_complete') is True, '2021 fitting/prediction completion is not recorded')
    for key in ['neural_run_manifest_sha256', 'neural_validation_2021_sha256']:
        require(isinstance(locked[key], str) and len(locked[key]) == 64
                and all(c in '0123456789abcdef' for c in locked[key]), 'Missing2021 neural provenance hash')
    return locked


def build_verified_features(stage: Path, year: int, protocol: dict) -> tuple:
    """Import only staged frozen code and redact target rows after selected year."""
    import numpy as np
    import pandas as pd
    for name, module in list(sys.modules.items()):
        if name.startswith('sintef_project') and getattr(module, '__file__', None):
            require(Path(module.__file__).resolve().is_relative_to(stage), 'A foreign sintef_project module is already imported; use a fresh process')
    sys.path.insert(0, str(stage))
    from sintef_project.features import build_features
    from sintef_project.physics import add_physics_features
    from sintef_project.context_features import add_context_features
    from sintef_project.lp_features import add_lp_features
    from sintef_project.normalized import add_normalized_features
    target = (stage / 'sintef_project/data/kernel/Unit_commitment_decisions.csv').resolve()
    original_read = pd.read_csv

    def guarded_read(path: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(path, (str, Path)) and Path(path).resolve() == target:
            if kwargs.get('usecols') == ['Run No', 'starttime']:
                return original_read(path, *args, **kwargs)
            require(not args and not kwargs, 'Frozen full-target read signature changed')
            return read_targets_through_year(target, year, original_read)
        return original_read(path, *args, **kwargs)

    with patch.object(pd, 'read_csv', guarded_read):
        x, labels, train, mapping, meta = build_features(stage)
        x, physics = add_physics_features(x, stage)
        x, context = add_context_features(x, stage)
        x, lp = add_lp_features(x, stage)
        x, normalized = add_normalized_features(x, stage)
    require(x.columns.tolist() == protocol['feature_columns'] and x.shape[1] == FEATURES, 'Frozen395 feature order mismatch')
    require(np.isfinite(x.to_numpy()).all(), 'Nonfinite real features')
    meta['feature_spec'].update(physics=physics, context=context, lp=lp, normalized=normalized)
    baseline = json.loads((stage / 'artifacts/lgb-normalized-lp-001/manifest.json').read_text())
    expected_spec = json.loads(json.dumps(baseline['feature_spec']))
    actual_spec = json.loads(json.dumps(meta['feature_spec']))
    expected_spec['lp'].pop('cache_directory')
    actual_spec['lp'].pop('cache_directory')
    require(expected_spec == actual_spec, 'Frozen feature formulas changed beyond portable cache-root path')
    return x, labels, train, mapping, meta, baseline


def train_fold(torch: Any, stage: Path, year: int, protocol: dict, protocol_sha: str,
               out: Path, report: dict) -> None:
    """Train exactly24 epochs on complete purged cases; evaluate only afterward."""
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    x, labels, train, mapping, meta, baseline = build_verified_features(stage, year, protocol)
    cube = x.to_numpy(dtype=np.float32).reshape(len(train) + len(mapping), HOURS, FEATURES)
    del x
    dates = pd.to_datetime(train.starttime, utc=True)
    begin = pd.Timestamp(f'{year}-01-01', tz='UTC')
    end = pd.Timestamp(f'{year + 1}-01-01', tz='UTC')
    tr = np.flatnonzero(dates < begin - pd.Timedelta(days=6))
    va = np.flatnonzero((dates >= begin) & (dates < end))
    require(dates.iloc[tr].max() + pd.Timedelta(hours=167) < begin, 'Training targets overlap validation')
    require(train.iloc[tr]['Run No'].tolist() == baseline['folds'][str(year)]['train_ids']
            and train.iloc[va]['Run No'].tolist() == baseline['folds'][str(year)]['validation_ids'], 'Purged IDs differ from frozen baseline')
    units = meta['units']
    y = np.column_stack([labels[unit] for unit in units]).reshape(len(train), HOURS, 14)
    del labels
    mean, scale = training_stats(cube, tr)
    np.savez_compressed(out / 'normalization.npz', mean=mean, scale=scale, columns=protocol['feature_columns'], training_case_ids=train.iloc[tr]['Run No'].to_numpy())
    report.update(feature_spec=meta['feature_spec'], units=units, train_ids=train.iloc[tr]['Run No'].tolist(),
                  validation_ids=train.iloc[va]['Run No'].tolist(), purge_days=6,
                  all_training_case_hours=True, training_rows=len(tr) * HOURS,
                  validation_label_scope=f'Only case starts in{year}; no later target rows decoded',
                  baseline_reference_sha256=digest(stage / 'artifacts/lgb-normalized-lp-001/manifest.json'))
    model = build_model(torch).to('cuda:0')
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001, betas=(.9, .999), eps=1e-8, foreach=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=.0001)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(SEED)
    orders, epochs = [], []
    for epoch in range(EPOCHS):
        model.train()
        order = rng.permutation(tr)
        orders.append(train.iloc[order]['Run No'].to_numpy())
        started = time.perf_counter()
        loss_sum = 0.
        learning_rate = optimizer.param_groups[0]['lr']
        for offset in range(0, len(order), BATCH):
            indices = order[offset:offset + BATCH]
            inputs = torch.from_numpy(normalized_batch(cube, indices, mean, scale)).to('cuda:0')
            targets = torch.from_numpy(np.ascontiguousarray(y[indices].transpose(0, 2, 1)).astype(np.float32)).to('cuda:0')
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(inputs), targets)
            require(torch.isfinite(loss).item(), 'Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(indices)
        scheduler.step()
        torch.cuda.synchronize(0)
        epochs.append({'epoch': epoch + 1, 'training_bce': loss_sum / len(tr),
                       'learning_rate': learning_rate, 'seconds': time.perf_counter() - started})
        print(json.dumps(epochs[-1]), flush=True)
    np.save(out / 'epoch_training_case_orders.npy', np.stack(orders))

    def cpu_tree(value: Any) -> Any:
        if torch.is_tensor(value):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {key: cpu_tree(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(cpu_tree(item) for item in value)
        return value

    checkpoint = {'model_state_dict': cpu_tree(model.state_dict()), 'optimizer_state_dict': cpu_tree(optimizer.state_dict()),
                  'scheduler_state_dict': scheduler.state_dict(), 'epoch': EPOCHS, 'seed': SEED,
                  'protocol_sha256': protocol_sha, 'features': protocol['feature_columns'], 'units': units,
                  'trainer_sha256': digest(Path(__file__)), 'bundle_manifest_sha256': protocol['bundle_manifest_sha256'],
                  'normalization_sha256': digest(out / 'normalization.npz'),
                  'numpy_rng_state': rng.bit_generator.state, 'torch_rng_cpu': torch.get_rng_state(),
                  'torch_rng_cuda': [value.cpu() for value in torch.cuda.get_rng_state_all()]}
    torch.save(checkpoint, out / f'tcn_{year}.pt')
    # No validation evaluation or checkpoint choice was made during training.
    model.eval()
    predictions = np.empty((len(va), HOURS, 14), dtype=np.float32)
    with torch.inference_mode():
        for offset in range(0, len(va), BATCH):
            indices = va[offset:offset + BATCH]
            inputs = torch.from_numpy(normalized_batch(cube, indices, mean, scale)).to('cuda:0')
            probabilities = torch.sigmoid(model(inputs)).cpu().numpy().transpose(0, 2, 1)
            predictions[offset:offset + len(indices)] = probabilities
    prediction = predictions.reshape(-1, 14)
    truth = y[va].reshape(-1, 14)
    require(np.isfinite(prediction).all() and prediction.min() >= 0 and prediction.max() <= 1, 'Invalid OOF probabilities')
    np.savez_compressed(out / f'validation_{year}.npz', predictions=prediction, labels=truth,
                        case_ids=train.iloc[va]['Run No'].to_numpy(), units=units)
    # Fresh model + saved normalizer replay; trusted hash checked before loading.
    checkpoint_path = out / f'tcn_{year}.pt'
    checkpoint_hash = digest(checkpoint_path)
    loaded = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    require(digest(checkpoint_path) == checkpoint_hash, 'Checkpoint changed during trusted load')
    for key, expected in [('protocol_sha256', protocol_sha), ('trainer_sha256', digest(Path(__file__))),
                          ('bundle_manifest_sha256', protocol['bundle_manifest_sha256']),
                          ('features', protocol['feature_columns']), ('units', units), ('epoch', EPOCHS), ('seed', SEED)]:
        require(loaded[key] == expected, f'Checkpoint replay metadata mismatch: {key}')
    require(loaded['normalization_sha256'] == digest(out / 'normalization.npz'), 'Saved normalizer changed')
    with np.load(out / 'normalization.npz', allow_pickle=False) as saved_stats:
        require(saved_stats['columns'].tolist() == protocol['feature_columns']
                and np.array_equal(saved_stats['training_case_ids'], train.iloc[tr]['Run No'].to_numpy()), 'Normalizer column/training scope changed')
        saved_mean, saved_scale = saved_stats['mean'], saved_stats['scale']
        require(saved_mean.dtype == saved_scale.dtype == np.float64, 'Normalizer must preserve float64 statistics')
    fresh = build_model(torch).to('cuda:0')
    fresh.load_state_dict(loaded['model_state_dict'], strict=True)
    fresh.eval()
    repeated = np.empty_like(predictions)
    with torch.inference_mode():
        for offset in range(0, len(va), BATCH):
            indices = va[offset:offset + BATCH]
            inputs = torch.from_numpy(normalized_batch(cube, indices, saved_mean, saved_scale)).to('cuda:0')
            repeated[offset:offset + len(indices)] = torch.sigmoid(fresh(inputs)).cpu().numpy().transpose(0, 2, 1)
    replay_error = float(np.max(np.abs(repeated - predictions)))
    require(repeated.tobytes() == predictions.tobytes(), 'Fresh checkpoint OOF replay is not byte-identical')
    np.save(out / f'replay_predictions_{year}.npy', repeated.reshape(-1, 14))
    report.update(epochs=epochs, completed_epochs=EPOCHS,
                  raw_micro_auc=float(roc_auc_score(truth.ravel(), prediction.ravel())),
                  unit_auc={unit: float(roc_auc_score(truth[:, j], prediction[:, j])) for j, unit in enumerate(units)},
                  max_cuda_allocated_bytes=torch.cuda.max_memory_allocated(0),
                  max_cuda_reserved_bytes=torch.cuda.max_memory_reserved(0),
                  checkpoint_sha256=checkpoint_hash,
                  validation_sha256=digest(out / f'validation_{year}.npz'),
                  checkpoint_replay={'fresh_model': True, 'saved_float64_normalizer': True, 'eval_inference_mode': True,
                                     'bitwise_equal': True, 'max_abs_probability_difference': replay_error,
                                     'checkpoint_load': 'hash-verified own checkpoint, weights_only=True, map_location=cpu',
                                     'replay_predictions_sha256': digest(out / f'replay_predictions_{year}.npy')},
                  postprocessing='none; raw14-unit probabilities for local frozen blend selection')
    print('FOLD', year, 'raw_micro_auc', report['raw_micro_auc'], flush=True)


def main() -> None:
    """Run either a synthetic CUDA check or one explicitly selected fold."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['smoke', 'fit'], default='smoke')
    parser.add_argument('--year', type=int, choices=[2021, 2022])
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--output-root', type=Path, default=Path('/kaggle/working/tcn-runs'))
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--stage-root', type=Path, default=Path('/kaggle/working/tdt4173-tcn-inputs'))
    parser.add_argument('--smoke-manifest', type=Path)
    parser.add_argument('--selection-lock', type=Path)
    parser.add_argument('--selection-lock-sha256')
    args = parser.parse_args()
    out = args.output_root / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    report = {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'run_id': args.run_id,
              'command': shlex.join([sys.executable, *sys.argv]), 'mode': args.mode, 'year': args.year,
              'seed': SEED, 'status': 'initializing', 'real_feature_table_loaded': False,
              'early_stopping': None, 'checkpoint_selection': 'fixed final epoch24 only',
              'environment': {'python': sys.version, 'platform': platform.platform(), 'packages': {}},
              'kaggle_submission': None, 'blend_selection': 'local only, after downloading OOF; no anchor arrays uploaded'}
    try:
        protocol_sha = digest(args.protocol)
        protocol = json.loads(args.protocol.read_text())
        require(protocol['version'] == 'cuda_tcn_v2_joint_anchor', 'Unexpected training protocol')
        require(protocol['trainer_sha256'] == digest(Path(__file__)), 'Trainer does not match frozen protocol hash')
        require(len(protocol['feature_columns']) == FEATURES and protocol['epochs'] == EPOCHS
                and protocol['batch_cases'] == BATCH and protocol['seed'] == SEED, 'Protocol constants changed')
        report['protocol_sha256'] = protocol_sha
        report['trainer_sha256'] = digest(Path(__file__))
        if args.mode == 'fit':
            require(args.year is not None and args.bundle is not None and args.smoke_manifest is not None,
                    'Fit requires one explicit year, input bundle and completed synthetic smoke')
            if args.year == 2022:
                require(args.selection_lock is not None and args.selection_lock_sha256 is not None,
                        '2022 requires the externally reviewed local2021 selection lock and explicitSHA256')
                report['selection_lock'] = validate_selection_lock(args.selection_lock, args.selection_lock_sha256, protocol, protocol_sha)
                report['selection_lock_sha256'] = digest(args.selection_lock)
            else:
                require(args.selection_lock is None, '2021 must precede external blend selection')
        else:
            require(args.year is None, 'Synthetic smoke has no validation year')
        # No package install, download, uv environment creation or CPU fallback.
        for package in ['torch', 'numpy', 'pandas', 'scipy', 'scikit-learn', 'pyyaml']:
            report['environment']['packages'][package] = importlib.metadata.version(package)
        uv = shutil.which('uv')
        require(uv is not None, 'uv is unavailable; no installation attempted. Use a reviewed hosted uv environment.')
        report['environment']['uv'] = subprocess.run([uv, '--version'], capture_output=True, text=True, check=True).stdout.strip()
        config = Path(sys.prefix) / 'pyvenv.cfg'
        require(sys.prefix != sys.base_prefix and config.is_file()
                and 'include-system-site-packages = true' in config.read_text().lower(),
                'Run through the offline uv --system-site-packages environment from the notebook launcher')
        report['environment']['venv'] = {'prefix': sys.prefix, 'base_prefix': sys.base_prefix,
                                         'system_site_packages': True, 'package_actions': 'none'}
        import torch
        report['cuda'] = cuda_environment(torch)
        if args.mode == 'smoke':
            report['smoke'] = synthetic_smoke(torch)
        else:
            smoke = json.loads(args.smoke_manifest.read_text())
            require(smoke['status'] == 'complete' and smoke['mode'] == 'smoke'
                    and smoke['protocol_sha256'] == protocol_sha and smoke['smoke']['synthetic_only'] is True,
                    'A successful matching synthetic CUDA smoke is mandatory')
            require(smoke['cuda'] == report['cuda'], 'CUDA environment changed; rerun the synthetic smoke')
            report['smoke_manifest_sha256'] = digest(args.smoke_manifest)
            report['bundle'] = stage_bundle(args.bundle, args.stage_root,
                                             protocol['bundle_manifest_sha256'], protocol['bundle_archive_sha256'])
            report['stage_root'] = str(args.stage_root.resolve())
            report['real_feature_table_loaded'] = True
            torch.cuda.reset_peak_memory_stats(0)
            train_fold(torch, args.stage_root.resolve(), args.year, protocol, protocol_sha, out, report)
        report['status'] = 'complete'
    except BaseException:
        report['status'] = 'failed'
        report['error'] = traceback.format_exc()
        raise
    finally:
        shutil.copyfile(__file__, out / 'train_cuda_tcn.py')
        if args.protocol.is_file():
            shutil.copyfile(args.protocol, out / 'protocol.json')
        report['runtime_seconds'] = time.time() - started
        report['artifacts'] = {str(path.relative_to(out)): digest(path) for path in out.rglob('*') if path.is_file() and path.name != 'manifest.json'}
        (out / 'manifest.json').write_text(json.dumps(report, indent=2))
        print(json.dumps({'status': report['status'], 'mode': args.mode, 'year': args.year,
                          'manifest': str(out / 'manifest.json'), 'runtime_seconds': report['runtime_seconds']}), flush=True)


if __name__ == '__main__':
    main()
