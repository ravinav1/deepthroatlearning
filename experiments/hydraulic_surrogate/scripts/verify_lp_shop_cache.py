"""Recheck every saved SHOP case, replay, feature and physical constraint without labels."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import importlib.metadata
import os
from pathlib import Path
import platform
import shlex
import shutil
import sys
import time
import traceback
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sintef_project.lp_features import HydroCaseData, UNIT_ORDER
from sintef_project.lp_shop_features import COLUMNS, compute_features, digest, feature_specification

SLACKS = ['minimum_slack', 'maximum_slack', 'bypass_slack', 'vest_slack']


def arrays(path: Path) -> dict:
    """Load numeric raw arrays only."""
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved}


def main() -> None:
    """Verify complete cache hashes, raw physical equations and feature arithmetic."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    out = ROOT / 'artifacts' / args.run_id
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    report = {'run_id': args.run_id, 'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'status': 'initializing',
              'command': shlex.join([sys.executable, *sys.argv]), 'cache_manifest_sha256': args.manifest_sha256,
              'source_sha256': digest(Path(__file__)), 'training': False, 'solver_runs': 0,
              'label_values_decoded': False, 'kaggle_submission': None,
              'environment': {'python': sys.version, 'platform': platform.platform(), 'machine': platform.machine(), 'cpu_count': os.cpu_count(), 'packages': {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'PyYAML']}},
              'max_hourly_balance_mm3': 0., 'max_total_balance_mm3': 0., 'max_inequality_violation': 0.,
              'slacks': {name: {'maximum': 0., 'cases_above_1e_8': 0} for name in SLACKS}}
    try:
        cache = args.cache.resolve()
        if digest(cache / 'manifest.json') != args.manifest_sha256:
            raise ValueError('Cache manifest changed')
        manifest = json.loads((cache / 'manifest.json').read_text())
        if manifest['status'] != 'complete' or manifest['feature_specification'] != feature_specification():
            raise ValueError('Cache incomplete or wrong feature specification')
        expected_paths = set(manifest['artifacts']) | {'manifest.json'}
        if {p.relative_to(cache).as_posix() for p in cache.rglob('*') if p.is_file()} != expected_paths:
            raise ValueError('Cache file allowlist differs')
        for name, expected in manifest['artifacts'].items():
            if digest(cache / name) != expected:
                raise ValueError(f'Cache artifact changed: {name}')
        for group in ['source', 'data']:
            for name, expected in manifest[group].items():
                if digest(ROOT / name) != expected:
                    raise ValueError(f'Cache dependency changed: {name}')
        base = (ROOT / manifest['baseline_manifest_path']).parent
        if digest(base / 'manifest.json') != manifest['baseline_manifest_sha256']:
            raise ValueError('Baseline cache manifest changed')
        baseline = json.loads((base / 'manifest.json').read_text())
        original_read = pd.read_csv
        label_path = (ROOT / 'sintef_project/data/kernel/Unit_commitment_decisions.csv').resolve()
        def guarded(path, *positional, **kwargs):
            if isinstance(path, (str, Path)) and Path(path).resolve() == label_path:
                if positional or kwargs != {'usecols': ['Run No', 'starttime']}:
                    raise ValueError('Audit may read commitment metadata only')
            return original_read(path, *positional, **kwargs)
        with patch.object(pd, 'read_csv', guarded):
            inputs = HydroCaseData(ROOT)
            if inputs.cases['Run No'].tolist() != manifest['case_ids'] or inputs.cases.starttime.astype(str).tolist() != manifest['case_dates']:
                raise ValueError('Case ID/date scope changed')
            saved_features = arrays(cache / 'features.npz')
            if (saved_features['case_ids'].tolist() != manifest['case_ids'] or saved_features['columns'].tolist() != COLUMNS
                    or saved_features['units'].tolist() != UNIT_ORDER or saved_features['features'].shape != (3647, 168, 56)):
                raise ValueError('Feature schema changed')
            network = inputs.network
            for index, case_id in enumerate(manifest['case_ids']):
                prefix = f'case_{case_id:04d}'
                case, result, replay = [arrays(cache / f'{prefix}_{kind}.npz') for kind in ['inputs', 'result', 'replay']]
                fresh = inputs.case(index)
                fresh['natural'] = fresh['natural'].astype(np.float64)
                for key, field in [('pmin', 'p_min'), ('start_cost', 'startcost_const'), ('stop_cost', 'stopcost_const')]:
                    fresh[key] = np.asarray([inputs.model['generator'][u][field] for u in UNIT_ORDER], dtype=np.float64)
                if set(case) != set(fresh) or set(result) != set(replay):
                    raise ValueError('Input/replay field set differs')
                for key in case:
                    np.testing.assert_array_equal(case[key], fresh[key], err_msg=f'{case_id}/{key}')
                for key in result:
                    np.testing.assert_array_equal(result[key], replay[key], err_msg=f'Replay {case_id}/{key}')
                    if not np.isfinite(result[key]).all():
                        raise ValueError('Nonfinite physical output')
                if int(result['status']) != 0:
                    raise ValueError('Nonoptimal cached case')
                base_path = base / f'{prefix}.npz'
                if digest(base_path) != baseline['artifacts'][base_path.name]:
                    raise ValueError('Baseline case changed')
                np.testing.assert_array_equal(compute_features(case, result, arrays(base_path), network), saved_features['features'][index])
                previous = np.concatenate([case['initial'].astype(np.float64)[:, None], result['storage'][:, :-1]], axis=1)
                balance = result['storage'] - previous - case['natural']
                for flows, ups, downs in [(result['flow'], network.unit_up, network.unit_down),
                                         (result['bypass'], network.bypass_up, network.bypass_down),
                                         (result['spill'], np.arange(9), network.spill_down)]:
                    for j, (up, down) in enumerate(zip(ups, downs)):
                        balance[up] += flows[j]
                        if down >= 0:
                            balance[down] -= flows[j]
                sea = (result['flow'][network.unit_down < 0].sum(dtype=np.float64)
                       + result['bypass'][network.bypass_down < 0].sum(dtype=np.float64)
                       + result['spill'][network.spill_down < 0].sum(dtype=np.float64))
                total_balance = float(result['storage'][:, -1].sum(dtype=np.float64) + sea
                                      - case['initial'].sum(dtype=np.float64) - case['natural'].sum(dtype=np.float64))
                hourly = float(np.abs(balance).max())
                if max(hourly, abs(total_balance)) > 1e-5:
                    raise ValueError('Physical water balance failed')
                state = result['state']
                change = np.diff(state, axis=1)
                violations = [-state, state-1, case['pmin'][:, None]*state-result['power'],
                              result['power']-case['pmax'][:, None]*state,
                              case['qmin'][:, None]*state-result['flow'], result['flow']-case['qmax'][:, None]*state,
                              change-result['raw_start'], -change-result['raw_stop'],
                              -result['segment_flow'], result['segment_flow']-case['widths'][:, :, None],
                              case['minimum']-result['storage']-result['minimum_slack'],
                              result['storage']-case['capacity'][:, None]-result['maximum_slack'],
                              case['bypass_minimum']-result['bypass']-result['bypass_slack'], result['bypass']-3.6]
                vest = (result['flow'][network.vest_units].sum(axis=0) + result['bypass'][network.vest_bypasses].sum(axis=0)
                        + result['spill'][network.vest_spills].sum(axis=0) + result['vest_slack'])
                violations.append(case['vest_minimum']-case['vest_natural']-vest)
                violations += [-result[key] for key in ['flow', 'power', 'storage', 'bypass', 'spill', *SLACKS]]
                violation = max(0., *(float(value.max()) for value in violations))
                if violation > 1e-5:
                    raise ValueError('Physical state/storage/routing constraint failed')
                energy = np.repeat(case['energy'][:, :, None], 168, axis=2)
                average = (case['energy']*case['widths']).sum(axis=1)/case['widths'].sum(axis=1)
                energy[:, :, case['price'] < 0] = average[:, None, None]
                np.testing.assert_array_equal(result['segment_flow'].sum(axis=1), result['flow'])
                np.testing.assert_array_equal((result['segment_flow']*energy).sum(axis=1), result['power'])
                cost = float((np.maximum(change, 0)*case['start_cost'][:, None]).sum()
                             + (np.maximum(-change, 0)*case['stop_cost'][:, None]).sum())
                np.testing.assert_allclose(cost, result['transition_cost'], rtol=0, atol=1e-5)
                penalty = 1e7 + 100*np.abs(case['terminal_value']).max()
                objective = float((result['power']*case['price'][None]).sum() + case['terminal_value']@result['storage'][:, -1]
                                  - (10/.0036)*result['bypass'].sum() - (1000/.0036)*result['spill'].sum()
                                  - penalty*sum(result[key].sum() for key in SLACKS) - cost)
                np.testing.assert_allclose(objective, result['objective'], rtol=2e-11, atol=1e-5)
                report['max_hourly_balance_mm3'] = max(report['max_hourly_balance_mm3'], hourly)
                report['max_total_balance_mm3'] = max(report['max_total_balance_mm3'], abs(total_balance))
                report['max_inequality_violation'] = max(report['max_inequality_violation'], violation)
                for key in SLACKS:
                    report['slacks'][key]['maximum'] = max(report['slacks'][key]['maximum'], float(result[key].max()))
                    report['slacks'][key]['cases_above_1e_8'] += int(np.max(result[key]) > 1e-8)
            report.update(status='complete', checked_cases=len(manifest['case_ids']), all_replays_bitwise=True,
                          all_features_recomputed_bitwise=True, raw_inputs_reconstructed_exactly=True)
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        shutil.copy2(__file__, out / 'verify_lp_shop_cache.py')
        report['runtime_seconds'] = time.monotonic()-started
        (out / 'manifest.json').write_text(json.dumps(report, indent=2))
        (out / 'experiment_record.md').write_text(f'\n## {args.run_id} — {report["timestamp_utc"]}\n\nStatus {report["status"]}. Command `{report["command"]}`. Label-free raw-cache verification; no solver/model runs. Full hash/replay/feature/input/physical-balance checks in `artifacts/{args.run_id}/manifest.json`; runtime {report["runtime_seconds"]:.3f}s. No validation or Kaggle claim.\n')
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
