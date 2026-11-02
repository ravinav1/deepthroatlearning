"""Measure a tiny synthetic-only TCN step on MPS, or record its blocker.

No competition data table or label is loaded, and no real model fit is offered.
The only metadata input is the frozen LP-v1 manifest's feature count and hashes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import shlex
import shutil
import statistics
import sys
import time
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SEED = 4173


def build_model(torch: Any, features: int) -> Any:
    """Build a 64-channel, centered 61-hour-receptive-field residual TCN."""
    nn = torch.nn

    class HourNorm(nn.Module):
        """Normalize hidden channels independently at each case/hour."""

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
                layers.extend([nn.Conv1d(64, 64, 3, padding=dilation,
                                          dilation=dilation, padding_mode='replicate'),
                               HourNorm(), nn.ReLU(), nn.Dropout(.1)])
            self.layers = nn.Sequential(*layers)

        def forward(self, x: Any) -> Any:
            return torch.relu(x + self.layers(x))

    return nn.Sequential(nn.Conv1d(features, 64, 1), nn.ReLU(),
                         *[ResidualBlock(d) for d in [1, 2, 4, 8]],
                         nn.Conv1d(64, 14, 1))


def main() -> None:
    """Record dependency/MPS checks and at most 12 synthetic optimizer steps."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', default='mps-tcn-smoke-001')
    args = parser.parse_args()
    out = ROOT / 'artifacts' / args.run_id
    out.mkdir(exist_ok=False)
    started = time.time()
    baseline = ROOT / 'artifacts/lgb-normalized-lp-001/manifest.json'
    metadata = json.loads(baseline.read_text())
    features = len(metadata['features'])
    record = {'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'command': shlex.join([sys.executable, *sys.argv]),
              'python': sys.version, 'platform': platform.platform(),
              'machine': platform.machine(), 'seed': SEED,
              'source_manifest': str(baseline.relative_to(ROOT)),
              'source_manifest_sha256': hashlib.sha256(baseline.read_bytes()).hexdigest(),
              'real_data_loaded': False, 'real_training': False,
              'input_shape': [16, features, 168], 'output_shape': [16, 14, 168],
              'input_tensor_bytes_formula': 16 * features * 168 * 4,
              'label_tensor_bytes_formula': 16 * 14 * 168 * 4,
              'measured_gpu_runtime': None, 'measured_gpu_memory': None,
              'warmup_steps': 2, 'timed_steps': 10, 'cpu_threads': 2,
              'architecture': {'channels': 64, 'dilations': [1, 2, 4, 8],
                               'convolutions_per_block': 2, 'kernel_size': 3,
                               'padding': 'case-local replication',
                               'normalization': 'LayerNorm channels per hour',
                               'activation': 'ReLU', 'dropout': .1,
                               'convolutional_receptive_field_hours': 61},
              'optimizer': {'name': 'AdamW', 'lr': .001, 'weight_decay': .0001,
                            'betas': [.9, .999], 'eps': 1e-8, 'gradient_norm_clip': 1.,
                            'foreach': False, 'loss': 'unweighted BCEWithLogitsLoss'},
              'kaggle': None}
    try:
        if importlib.util.find_spec('torch') is None:
            record.update(status='blocked', blocker='PyTorch is not installed; no compatible wheel was cached.')
            return
        import torch
        record['torch'] = importlib.metadata.version('torch')
        record['mps_built'] = torch.backends.mps.is_built()
        record['mps_available'] = torch.backends.mps.is_available()
        if not record['mps_available']:
            record.update(status='blocked', blocker='Installed PyTorch does not expose an available MPS device.')
            return
        if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1':
            raise ValueError('CPU fallback must be disabled for a GPU feasibility measurement')
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.manual_seed(SEED)
        torch.mps.manual_seed(SEED)
        torch.use_deterministic_algorithms(True)
        model = build_model(torch, features).to('mps')
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001,
                                     weight_decay=.0001, foreach=False)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        cpu_x = torch.randn(16, features, 168)
        cpu_y = torch.randint(0, 2, (16, 14, 168)).float()
        times, memory, losses = [], [], []
        for step in range(12):
            torch.mps.synchronize()
            start = time.perf_counter()
            x, y = cpu_x.to('mps'), cpu_y.to('mps')
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            assert list(logits.shape) == [16, 14, 168]
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            torch.mps.synchronize()
            elapsed = time.perf_counter() - start
            value = float(loss.detach().cpu())
            if not (0. <= value < float('inf')):
                raise ValueError('Nonfinite synthetic loss')
            losses.append(value)
            memory.append({'tensor_bytes': torch.mps.current_allocated_memory(),
                           'driver_bytes': torch.mps.driver_allocated_memory()})
            if step >= 2:
                times.append(elapsed)
        record['parameters'] = sum(p.numel() for p in model.parameters())
        record['measured_gpu_runtime'] = {'timed_step_seconds': times,
                                          'median_step_seconds': statistics.median(times),
                                          'mean_step_seconds': statistics.mean(times),
                                          'includes': 'host-to-device copy, forward, backward, optimizer, synchronization',
                                          'excludes': 'real data feature construction and loading; warmup'}
        record['measured_gpu_memory'] = {'max_sampled_tensor_bytes': max(m['tensor_bytes'] for m in memory),
                                         'max_sampled_driver_bytes': max(m['driver_bytes'] for m in memory),
                                         'sampling': 'after each synchronized update, not true peak',
                                         'recommended_max_bytes': torch.mps.recommended_max_memory(),
                                         'process_peak_rss_bytes_macos': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        record['synthetic_losses'] = losses
        record['status'] = 'complete'
    except BaseException:
        record['status'] = 'failed'
        record['error'] = traceback.format_exc()
        raise
    finally:
        destination = out / 'source/scripts/smoke_mps_tcn.py'
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(__file__), destination)
        record['source_sha256'] = hashlib.sha256(destination.read_bytes()).hexdigest()
        record['wall_seconds_including_setup'] = time.time() - started
        (out / 'manifest.json').write_text(json.dumps(record, indent=2))
        print(json.dumps({key: record.get(key) for key in ['status', 'blocker', 'torch', 'mps_available', 'measured_gpu_runtime', 'measured_gpu_memory']}), flush=True)
        with (ROOT / 'docs/experiments.md').open('a') as handle:
            handle.write(f'\n## {args.run_id} — {record["timestamp_utc"]}\n\n'
                         f'Status: {record["status"]}. Command: `{record["command"]}`. '
                         'Synthetic-only MPS TCN feasibility; no real feature table, labels, or fit. '
                         f'Proof: `artifacts/{args.run_id}/manifest.json`. '
                         'Blocked/failed checks do not establish GPU speed or memory consumption. No Kaggle submission.\n')


if __name__ == '__main__':
    main()
