"""Independently rebuild early OOF inputs and replay every saved base checkpoint."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import shlex
import sys
import time
import traceback

for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[variable] = '1'

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_nonlinear_pq_specialists import build_608


def digest(path: Path) -> str:
    """Hash artifacts using bounded memory."""
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    """Verify the complete full14 OOF cache without opening2020 target values."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--oof-manifest-sha256', required=True)
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id or args.run_id in ['', '.', '..']:
        raise ValueError('Invalid audit run ID')
    out = ROOT / 'artifacts' / args.run_id
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    report = {'run_id': args.run_id, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'command': shlex.join([sys.executable, *sys.argv]), 'status': 'initializing',
              'oof_manifest_sha256': args.oof_manifest_sha256, 'training': False,
              'screen_2020_labels_read': False, 'kaggle': None, 'folds': {},
              'source_sha256': digest(Path(__file__)),
              'environment': {'python': sys.version, 'platform': platform.platform(),
                              'packages': {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'lightgbm']}}}
    try:
        source = ROOT / 'artifacts/full14-early-oof-001'
        if digest(source / 'manifest.json') != args.oof_manifest_sha256:
            raise ValueError('Explicit completed OOF hash differs')
        run = json.loads((source / 'manifest.json').read_text())
        if run['status'] != 'complete' or run['screen_2020_labels_read'] or len(run['new_models']) != 33:
            raise ValueError('OOF cache incomplete or screen targets opened')
        protocol_path = ROOT / 'docs/full14_early_oof_protocol.json'
        protocol_sha = '58d98fe4d3f7ec3d7b0a102b6a0ca6a538790baaf11ddb0ef2d3677cefc96028'
        if digest(protocol_path) != protocol_sha or run['protocol_sha256'] != protocol_sha:
            raise ValueError('Frozen OOF protocol differs')
        protocol = json.loads(protocol_path.read_text())
        for group in ['source', 'data', 'inputs']:
            for name, expected in protocol[group].items():
                if digest(ROOT / name) != expected:
                    raise ValueError(f'Changed {group}: {name}')
        for name, expected in run['artifacts'].items():
            if digest(source / name) != expected:
                raise ValueError(f'Changed OOF output: {name}')
        previous = ROOT / protocol['reuse_directory']
        baseline = json.loads((ROOT / 'artifacts/lgb-normalized-lp-context-joint-001/manifest.json').read_text())
        x, labels, train, _, metadata = build_608(2019, baseline)
        if x.columns.tolist() != protocol['features'] or metadata['feature_spec'] != baseline['feature_spec']:
            raise ValueError('Fresh608 source/cache/formulas differ')
        units = protocol['units']
        if metadata['units'] != units:
            raise ValueError('Fresh unit order differs')
        dates = pd.to_datetime(train.starttime, utc=True)
        y = np.column_stack([labels[unit] for unit in units]).astype(np.uint8)
        rng = np.random.default_rng(4173)
        folds = {}
        for year in [2018, 2019, 2020]:
            begin = pd.Timestamp(f'{year}-01-01', tz='UTC')
            end = pd.Timestamp(f'{year}-12-31' if year == 2018 else f'{year}-12-25', tz='UTC')
            tr = np.flatnonzero(dates < begin - pd.Timedelta(days=6))
            va = np.flatnonzero((dates >= begin) & (dates <= end))
            if dates.iloc[tr].max() + pd.Timedelta(hours=167) >= begin:
                raise ValueError('Independent purge check failed')
            rows = (tr[:, None] * 168 + np.arange(168)).ravel()
            sampled = np.sort(rng.choice(rows, size=int(len(rows) * .4), replace=False))
            np.testing.assert_array_equal(sampled, np.load(source / f'train_sample_{year}.npy'))
            if digest(source / f'train_sample_{year}.npy') != digest(previous / f'train_sample_{year}.npy'):
                raise ValueError('Copied sample file bytes changed')
            prediction_rows = (va[:, None] * 168 + np.arange(168)).ravel()
            saved_features = np.load(source / f'features_{year}.npy', mmap_mode='r')
            np.testing.assert_array_equal(saved_features, x.iloc[prediction_rows].to_numpy())
            with np.load(source / f'oof_{year}.npz', allow_pickle=False) as z:
                values = {name: z[name] for name in z.files}
            np.testing.assert_array_equal(values['case_ids'], train.iloc[va]['Run No'])
            np.testing.assert_array_equal(values['dates'], dates.iloc[va].astype(str))
            if values['units'].tolist() != units or values['probability'].shape != (len(va), 168, 14):
                raise ValueError('Saved OOF schema differs')
            if not np.isfinite(values['probability']).all() or np.any((values['probability'] < 0) | (values['probability'] > 1)):
                raise ValueError('Invalid OOF probability range')
            if ('labels' in values) != (year < 2020):
                raise ValueError('Screen file must have no targets')
            if year < 2020:
                np.testing.assert_array_equal(values['labels'], y[prediction_rows].reshape(len(va), 168, 14))
            with np.load(previous / f'oof_{year}.npz', allow_pickle=False) as old:
                for j, unit in enumerate(['Vinje_G1', 'Vinje_G2', 'Vinje_G3']):
                    np.testing.assert_array_equal(values['probability'][:, :, units.index(unit)], old['probability'][:, :, j])
                    if digest(source / f'fold{year}_{unit}.txt') != digest(previous / f'fold{year}_{unit}.txt'):
                        raise ValueError('Reused checkpoint bytes changed')
            folds[year] = values
            report['folds'][str(year)] = {'train_cases': len(tr), 'prediction_cases': len(va),
                                          'sample_rows': len(sampled), 'features_exact_fresh_rebuild': True,
                                          'reused_three_unit_predictions_exact': True}
        del x, labels, y, saved_features
        gc.collect()
        replays = []
        for year in [2018, 2019, 2020]:
            features = np.load(source / f'features_{year}.npy', mmap_mode='r')
            for j, unit in enumerate(units):
                model = lgb.Booster(model_file=str(source / f'fold{year}_{unit}.txt'))
                if model.num_trees() != 350 or model.num_feature() != 608:
                    raise ValueError('Checkpoint capacity/schema differs')
                actual = model.predict(features, num_threads=2).astype(np.float32)
                np.testing.assert_array_equal(actual, folds[year]['probability'][:, :, j].ravel())
                replays.append({'year': year, 'unit': unit, 'rows': len(actual), 'max_error': 0.})
                print('INDEPENDENT_OOF_REPLAY', year, unit, 'exact', flush=True)
                del model
        report.update(status='passed', protocol_sha256=protocol_sha, all_42_checkpoint_replays_exact=True,
                      checkpoint_replays=replays, all_608_features_rebuilt_exact=True,
                      reused_nine_checkpoints_and_probabilities_exact=True, training_labels_exact=True,
                      residual_training_cases=724, screen_cases=360,
                      input_manifest_sha256=digest(source / 'manifest.json'))
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        report['runtime_seconds'] = time.monotonic() - started
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report['peak_rss_bytes'] = int(rss if sys.platform == 'darwin' else rss * 1024)
        (out / 'manifest.json').write_text(json.dumps(report, indent=2))
        (out / 'audit.py').write_bytes(Path(__file__).read_bytes())
    print('AUDIT_PASSED', digest(out / 'manifest.json'), report['runtime_seconds'], flush=True)


if __name__ == '__main__':
    main()
