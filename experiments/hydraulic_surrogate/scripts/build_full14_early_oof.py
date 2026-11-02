"""Extend nine frozen Vinje models to a full14 single608-LGB early OOF cache."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import shlex
import shutil
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
from sintef_project.nonlinear_pq import digest

VINJE = ['Vinje_G1', 'Vinje_G2', 'Vinje_G3']


def require(condition: bool, message: str) -> None:
    """Reject changed provenance or data before using it."""
    if not condition:
        raise ValueError(message)


def split_indices(dates: pd.Series, year: int) -> tuple[np.ndarray, np.ndarray]:
    """Match the frozen six-day purge and early OOF prediction boundaries."""
    require(year in [2018, 2019, 2020], 'Unexpected OOF year')
    begin = pd.Timestamp(f'{year}-01-01', tz='UTC')
    end = pd.Timestamp(f'{year}-12-31' if year == 2018 else f'{year}-12-25', tz='UTC')
    train = np.flatnonzero(dates < begin - pd.Timedelta(days=6))
    validation = np.flatnonzero((dates >= begin) & (dates <= end))
    require(len(train) > 0 and len(validation) > 0, 'Empty chronological partition')
    require(dates.iloc[train].max() + pd.Timedelta(hours=167) < begin, 'Training horizon overlaps OOF')
    return train, validation


def place_reused(values: np.ndarray, reused: np.ndarray, units: list[str]) -> None:
    """Copy the three frozen generator columns exactly into the full unit order."""
    require(len(units) == 14 and len(set(units)) == 14 and all(u in units for u in VINJE), 'Changed full unit schema')
    require(values.shape[:-1] == reused.shape[:-1] and values.shape[-1] == 14 and reused.shape[-1] == 3,
            'Changed reused prediction shape')
    require(values.dtype == reused.dtype == np.dtype('float32'), 'Changed reused probability dtype')
    require(np.isfinite(reused).all() and np.all((reused >= 0) & (reused <= 1)), 'Invalid reused probability')
    for j, unit in enumerate(VINJE):
        values[..., units.index(unit)] = reused[..., j]


def main() -> None:
    """Fit only the 33 absent models; leave 2020 labels unopened for later residual screening."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--protocol-sha256', required=True)
    args = parser.parse_args()
    require(Path(args.run_id).name == args.run_id and args.run_id not in ['', '.', '..'], 'Invalid run ID')
    out = ROOT / 'artifacts' / args.run_id
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    report = {'run_id': args.run_id, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'command': shlex.join([sys.executable, *sys.argv]), 'status': 'initializing',
              'scope': 'Full14 single608LGB early OOF anchor; not accepted five-component blend',
              'maximum_target_year': 2019, 'seed': 4173, 'kaggle': None, 'folds': {}, 'new_models': [],
              'environment': {'python': sys.version, 'platform': platform.platform(), 'machine': platform.machine(),
                              'cpu_count': os.cpu_count(), 'packages': {name: importlib.metadata.version(name) for name in
                              ['numpy', 'pandas', 'scipy', 'scikit-learn', 'lightgbm', 'PyYAML']}}}
    try:
        protocol_path = ROOT / 'docs/full14_early_oof_protocol.json'
        require(digest(protocol_path) == args.protocol_sha256, 'Explicit frozen protocol hash differs')
        protocol = json.loads(protocol_path.read_text())
        require(protocol['version'] == 'full14_single608_early_oof_v1', 'Unexpected protocol')
        for group in ['source', 'data', 'inputs']:
            for name, expected in protocol[group].items():
                require(digest(ROOT / name) == expected, f'Changed {group}: {name}')
        source_dir = ROOT / protocol['reuse_directory']
        previous = json.loads((source_dir / 'manifest.json').read_text())
        require(previous['status'] == 'complete' and previous['model_parameters'] == protocol['model_parameters'],
                'Reused run incomplete or model parameters changed')
        for name, expected in previous['artifacts'].items():
            require(digest(source_dir / name) == expected, f'Reused artifact changed: {name}')
        for name in protocol['source']:
            target = out / 'source' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        shutil.copy2(protocol_path, out / 'protocol.json')
        report.update(protocol_sha256=args.protocol_sha256, source=protocol['source'], data=protocol['data'],
                      inputs=protocol['inputs'], model_parameters=protocol['model_parameters'])
        baseline = json.loads((ROOT / 'artifacts/lgb-normalized-lp-context-joint-001/manifest.json').read_text())
        x, labels, train, _, meta = build_608(2019, baseline)
        units = meta['units']
        require(units == protocol['units'] and x.columns.tolist() == protocol['features'], 'Full608 feature/unit schema differs')
        require(meta['feature_spec'] == baseline['feature_spec'], 'Feature formulas or cache provenance differs')
        report.update(units=units, feature_names=x.columns.tolist(), feature_spec=meta['feature_spec'])
        y = np.column_stack([labels[unit] for unit in units]).astype(np.uint8)
        dates = pd.to_datetime(train.starttime, utc=True)
        rng = np.random.default_rng(4173)
        for year in [2018, 2019, 2020]:
            tr, va = split_indices(dates, year)
            old_fold = previous['folds'][str(year)]
            require(train.iloc[tr]['Run No'].tolist() == old_fold['train_ids'] and
                    train.iloc[va]['Run No'].tolist() == old_fold['prediction_ids'], 'Frozen case partition differs')
            rows = (tr[:, None] * 168 + np.arange(168)).ravel()
            sample = np.sort(rng.choice(rows, size=int(len(rows) * .4), replace=False))
            old_sample = source_dir / f'train_sample_{year}.npy'
            np.testing.assert_array_equal(sample, np.load(old_sample, allow_pickle=False))
            shutil.copy2(old_sample, out / old_sample.name)
            prediction_rows = (va[:, None] * 168 + np.arange(168)).ravel()
            xt, xv = x.iloc[sample], x.iloc[prediction_rows]
            require(all(dtype == np.dtype('float32') for dtype in xv.dtypes), 'Unexpected feature dtype')
            np.save(out / f'features_{year}.npy', xv.to_numpy(copy=True))
            reused_path = source_dir / f'oof_{year}.npz'
            with np.load(reused_path, allow_pickle=False) as saved:
                reused = {key: saved[key] for key in saved.files}
            np.testing.assert_array_equal(reused['case_ids'], train.iloc[va]['Run No'].to_numpy())
            np.testing.assert_array_equal(reused['dates'], dates.iloc[va].astype(str).to_numpy(dtype=str))
            prediction = np.empty((len(va), 168, 14), dtype=np.float32)
            place_reused(prediction, reused['probability'], units)
            reuse_out = out / 'reused_vinje'
            reuse_out.mkdir(exist_ok=True)
            shutil.copy2(reused_path, reuse_out / reused_path.name)
            for unit in VINJE:
                checkpoint = source_dir / f'fold{year}_{unit}.txt'
                shutil.copy2(checkpoint, out / checkpoint.name)
                replay = lgb.Booster(model_file=str(checkpoint)).predict(xv, num_threads=2).astype(np.float32)
                np.testing.assert_array_equal(replay, prediction[:, :, units.index(unit)].ravel())
            for j, unit in enumerate(units):
                if unit in VINJE:
                    continue
                unit_started = time.monotonic()
                model = lgb.LGBMClassifier(**protocol['model_parameters'])
                model.fit(xt, y[sample, j])
                raw = model.predict_proba(xv)[:, 1].astype(np.float32)
                prediction[:, :, j] = raw.reshape(len(va), 168)
                checkpoint = out / f'fold{year}_{unit}.txt'
                model.booster_.save_model(str(checkpoint))
                replay = lgb.Booster(model_file=str(checkpoint)).predict(xv, num_threads=2).astype(np.float32)
                np.testing.assert_array_equal(raw, replay)
                item = {'year': year, 'unit': unit, 'runtime_seconds': time.monotonic() - unit_started,
                        'checkpoint_sha256': digest(checkpoint), 'checkpoint_replay_exact': True}
                report['new_models'].append(item)
                print('OOF_MODEL', year, unit, json.dumps(item), flush=True)
                (out / 'manifest.json').write_text(json.dumps(report, indent=2))
                del model
            for j, unit in enumerate(VINJE):
                np.testing.assert_array_equal(prediction[:, :, units.index(unit)], reused['probability'][:, :, j])
            values = {'probability': prediction, 'case_ids': reused['case_ids'], 'dates': reused['dates'], 'units': np.asarray(units)}
            if year < 2020:
                values['labels'] = y[prediction_rows].reshape(len(va), 168, 14)
                np.testing.assert_array_equal(values['labels'][:, :, [units.index(u) for u in VINJE]], reused['labels'])
            np.savez_compressed(out / f'oof_{year}.npz', **values)
            report['folds'][str(year)] = {**old_fold, 'reused_models': 3, 'new_models': 11,
                                        'sample_sha256': digest(out / old_sample.name),
                                        'oof_sha256': digest(out / f'oof_{year}.npz'),
                                        'features_sha256': digest(out / f'features_{year}.npy'),
                                        'all_checkpoint_replay_exact': True}
            (out / 'manifest.json').write_text(json.dumps(report, indent=2))
        require(len(report['new_models']) == 33, 'Wrong number of fitted models')
        report.update(status='complete', preserved_vinje_predictions_bitwise=True, all_42_checkpoint_replays_exact=True,
                      screen_2020_labels_read=False, feature_storage='float32 unexpanded case-hour rows, full608 order')
        print(json.dumps({'status': 'complete', 'new_models': 33, 'reused_models': 9}), flush=True)
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        report['runtime_seconds'] = time.monotonic() - started
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report['peak_rss_bytes'] = int(rss if sys.platform == 'darwin' else rss * 1024)
        report['artifacts'] = {p.relative_to(out).as_posix(): digest(p) for p in out.rglob('*')
                               if p.is_file() and p.name != 'manifest.json'}
        (out / 'manifest.json').write_text(json.dumps(report, indent=2))
        record = (f'\n## {args.run_id} — {report["timestamp_utc"]}\n\nStatus {report["status"]}. '
                  f'Command `{report["command"]}`. Full14 single608LGB forward-OOF extension: 33 missing350-tree models, '
                  '9 frozen Vinje models and probabilities preserved bitwise, matched40% samples, six-day purge; '
                  '2018/2019 residual-training OOF and2020 screen predictions. No2020 labels decoded, residual fit, '
                  f'2021/22 training or Kaggle action. Runtime {report["runtime_seconds"]:.3f}s; '
                  f'peakRSS {report["peak_rss_bytes"]}; all source/data/sample/model/features/probability hashes and '
                  f'environment in `artifacts/{args.run_id}/manifest.json`.\n')
        (out / 'experiment_record.md').write_text(record)
        with (ROOT / 'docs/experiments.md').open('a') as stream:
            stream.write(record)


if __name__ == '__main__':
    main()
