"""Audit timestamp availability for past-state inputs without building model X."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shlex
import shutil
import sys
import time
import traceback

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sintef_project.features import _capacities


def digest(path: Path) -> str:
    """Compute a streaming file digest."""
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    """Read only case metadata and raw supplied reservoir histories."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', default='past-state-audit-001')
    args = parser.parse_args()
    out = ROOT / 'artifacts' / args.run_id
    out.mkdir(exist_ok=False)
    begin = time.time()
    record = {'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'command': shlex.join([sys.executable, *sys.argv]),
              'environment': {'python': sys.version, 'platform': platform.platform(),
                              'numpy': np.__version__, 'pandas': pd.__version__},
              'training': 'none', 'labels_loaded': False, 'hourly_model_X_built': False,
              'seeds': None, 'scores': None, 'kaggle': None, 'inputs': {}}
    try:
        kernel = ROOT / 'sintef_project/data/kernel'
        train = pd.read_csv(kernel / 'Unit_commitment_decisions.csv', usecols=['Run No', 'starttime'])
        test = pd.read_csv(ROOT / 'sintef_project/prediction_mapping.csv')
        cases = pd.concat([train.assign(split='train'), test.assign(split='test')], ignore_index=True)
        starts = pd.DatetimeIndex(pd.to_datetime(cases['starttime'], utc=True))
        volume = pd.read_csv(kernel / 'Historical_volume_2015_2024.csv')
        volume.index = pd.to_datetime(volume.pop('date'), utc=True)
        names = volume.columns.tolist()
        capacities = _capacities(ROOT / 'sintef_project/data/extended/Tokke_Vinje_topology.yaml')
        # Scan original input in bounded chunks; retain only the date range that
        # can be queried by a supplied case. Do not read targets or build X.
        fragments = []
        inflow_first = inflow_last = None
        for chunk in pd.read_csv(kernel / 'Historical_inflow_1958_2025.csv',
                                 usecols=['date', *names], chunksize=100000,
                                 dtype={name: 'float32' for name in names}):
            dates = pd.to_datetime(chunk.pop('date'), utc=True)
            inflow_first = str(dates.iloc[0]) if inflow_first is None else inflow_first
            inflow_last = str(dates.iloc[-1])
            chunk.index = pd.DatetimeIndex(dates)
            keep = (chunk.index >= starts.min() - pd.Timedelta(days=7)) & (chunk.index < starts.max())
            fragments.append(chunk.loc[keep])
        inflow = pd.concat(fragments)
        if inflow.index.has_duplicates or volume.index.has_duplicates:
            raise ValueError('Duplicate supplied timestamps')
        if not inflow.index.equals(inflow.index.floor('h')):
            raise ValueError('Inflow not hourly aligned')
        valid = np.isfinite(inflow)
        counts = valid.groupby(inflow.index.normalize()).sum()
        daily = inflow.where(valid).groupby(inflow.index.normalize()).sum(min_count=24) * .0036
        daily = daily.where(counts == 24)
        now = volume.reindex(starts).to_numpy()
        availability = pd.DataFrame({'Run No': cases['Run No'], 'starttime': starts, 'split': cases['split']})
        record['availability'] = {}
        record['reservoir_diagnostics'] = {}
        for days in [1, 3, 7]:
            previous = volume.reindex(starts - pd.Timedelta(days=days)).to_numpy()
            good_volume = np.isfinite(now) & np.isfinite(previous)
            query = pd.DatetimeIndex(np.concatenate([(starts - pd.Timedelta(days=d)).as_unit('ns').asi8
                                                     for d in range(1, days + 1)]), tz='UTC')
            sums = daily.reindex(query).to_numpy().reshape(days, len(starts), len(names))
            good_inflow = np.isfinite(sums).all(axis=0)
            availability[f'volume_complete_{days}d'] = good_volume.all(axis=1)
            availability[f'inflow_complete_{days}d'] = good_inflow.all(axis=1)
            volume_change = now - previous
            residual = np.sum(sums, axis=0) - volume_change
            record['availability'][str(days)] = {}
            for group, mask in [('all', np.ones(len(cases), dtype=bool)),
                                ('training_2015_2022', cases['split'].eq('train').to_numpy()),
                                ('validation_2021', starts.year == 2021),
                                ('validation_2022', starts.year == 2022),
                                ('test_2023_2024', cases['split'].eq('test').to_numpy())]:
                record['availability'][str(days)][group] = {
                    'cases': int(mask.sum()), 'all_reservoir_volume_complete': int(good_volume[mask].all(axis=1).sum()),
                    'all_reservoir_inflow_complete': int(good_inflow[mask].all(axis=1).sum()),
                    'missing_volume_case_ids': cases.loc[mask & ~good_volume.all(axis=1), 'Run No'].tolist(),
                    'missing_inflow_case_ids': cases.loc[mask & ~good_inflow.all(axis=1), 'Run No'].tolist()}
            for column, name in enumerate(names):
                keep = good_volume[:, column] & good_inflow[:, column]
                value = residual[keep, column] / capacities[name]
                record['reservoir_diagnostics'].setdefault(name, {})[str(days)] = {
                    'valid_cases': int(keep.sum()), 'negative_withdrawal_proxy_fraction': float((value < 0).mean()),
                    'withdrawal_proxy_capacity_q01_q50_q99': np.quantile(value, [.01, .5, .99]).tolist(),
                    'zero_storage_delta_fraction': float((volume_change[keep, column] == 0).mean())}
        record['history'] = {'volume_rows': len(volume), 'volume_first': str(volume.index.min()),
                              'volume_last': str(volume.index.max()), 'inflow_first': inflow_first,
                              'inflow_last': inflow_last, 'retained_inflow_rows': len(inflow),
                              'constant_volume_reservoirs': [name for name in names if volume[name].nunique() == 1],
                              'reservoir_order': names}
        availability.to_csv(out / 'availability.csv', index=False)
        inputs = ['sintef_project/ML_Task_TDT4173.pdf', 'sintef_project/Dataset_definitions_and_explanation.pdf',
                  'sintef_project/data/kernel/Unit_commitment_decisions.csv',
                  'sintef_project/data/kernel/Historical_volume_2015_2024.csv',
                  'sintef_project/data/kernel/Historical_inflow_1958_2025.csv',
                  'sintef_project/data/extended/Tokke_Vinje_topology.yaml', 'sintef_project/prediction_mapping.csv']
        record['inputs'] = {name: digest(ROOT / name) for name in inputs}
        record['status'] = 'complete'
        print(json.dumps({'availability': record['availability'], 'history': record['history']}), flush=True)
    except BaseException:
        record['status'] = 'failed'
        record['error'] = traceback.format_exc()
        raise
    finally:
        for name in ['scripts/audit_past_state.py', 'sintef_project/past_state_features.py',
                     'sintef_project/features.py', 'tests/test_past_state_features.py']:
            target = out / 'source' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        record['runtime_seconds'] = time.time() - begin
        record['artifacts'] = {str(p.relative_to(out)): digest(p) for p in out.rglob('*') if p.is_file()}
        (out / 'manifest.json').write_text(json.dumps(record, indent=2))
        with (ROOT / 'docs/experiments.md').open('a') as handle:
            handle.write(f'\n## {args.run_id} — {record["timestamp_utc"]}\n\nStatus: {record["status"]}. '
                         f'Command: `{record["command"]}`. Past-only input availability audit; no labels, '
                         'hourly feature matrix, training, selection, or Kaggle submission. '
                         f'Manifest with data/source/environment hashes: `artifacts/{args.run_id}/manifest.json`. '
                         f'Runtime: {record["runtime_seconds"]:.2f}s. Pure transform boundary tests recorded separately.\n')


if __name__ == '__main__':
    main()
