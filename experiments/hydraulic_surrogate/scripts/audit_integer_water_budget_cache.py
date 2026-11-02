"""Audit a finalized 3647-case integer cache without optimization or label reads.

Rebuild the frozen physical matrices, check saved incumbents and conditional-LP
KKT certificates, verify compatible receipts, and independently reconstruct the
84 feature columns. This script never promotes an incomplete cache.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
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
from types import SimpleNamespace
from unittest.mock import patch

for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS']:
    os.environ[variable] = '1'

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sintef_project.lp_features import HydroCaseData, UNIT_ORDER
from sintef_project.lp_shop_relaxation import ShopRelaxationLP

PROTOCOL_SHA = '7571930990cadc542c7ad4f47e7f1195d7a239885c887ad45b051afd9b9377d9'
BASE_SHA = 'f8aaa986c7afdaa802101f84c9cc33ce89309339efcfd321930431b035e8df9a'
OLD_MANIFEST_SHA = 'd6da75c846cc0b90b2462839ec368c5abda3fd2ba8a8a8e4cee681272de29867'
TARGETS = ['Hogga_G1', 'Lio_G1', 'Haukeli_G1']
KINDS = ['state', 'delta_qmin_ratio', 'delta_power_fraction', 'start', 'stop',
         'conditional_margin_change']
COLUMNS = [f'integer_water_{kind}_{unit}' for unit in UNIT_ORDER for kind in KINDS]


def digest(path: Path) -> str:
    """Hash a local artifact without loading it all into memory."""
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def require(condition: bool, message: str) -> None:
    """Keep audit checks active even under optimized Python."""
    if not condition:
        raise ValueError(message)


def arrays(path: Path) -> dict:
    """Read numeric arrays without pickle."""
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def same(first: dict, second: dict) -> None:
    """Compare every field's dtype, shape and raw bytes."""
    require(set(first) == set(second), 'Array fields differ')
    for key in first:
        a, b = np.asarray(first[key]), np.asarray(second[key])
        require(a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes(),
                'Array bytes differ: ' + key)


def verify_hashes(directory: Path, inventory: dict) -> None:
    """Validate relative paths and every inventoried payload."""
    base = directory.resolve()
    for name, expected in inventory.items():
        path = (directory / name).resolve()
        require(path.is_relative_to(base), 'Artifact escapes its directory')
        require(digest(path) == expected, 'Changed artifact: ' + str(path))


def check_complete_manifest(manifest: dict, protocol: dict) -> None:
    """Reject a partial aggregate before opening derived features."""
    require(manifest.get('status') == 'complete', 'Cache is not complete')
    require(manifest.get('all_cases_proven_and_replayed') is True, 'Missing full proof gate')
    require(not manifest.get('failures'), 'Cache has failures')
    require(len(protocol['case_ids']) == 3647 and len(set(protocol['case_ids'])) == 3647,
            'Expected exactly 3647 unique cases')
    require(manifest['case_ids'] == protocol['case_ids'] and
            manifest['case_dates'] == protocol['case_dates'], 'Case metadata differs')
    require(set(manifest['completed_cases']) == {str(i) for i in protocol['case_ids']},
            'Missing complete case receipts')
    require(manifest['protocol_sha256'] == PROTOCOL_SHA, 'Wrong execution protocol')
    require(manifest['labels_decoded'] is False and manifest['training'] is False,
            'Cache must be input only')
    for key in ['source', 'data', 'feature_specification', 'prefix_columns', 'environment']:
        require(manifest[key] == protocol[key], 'Manifest/protocol differs: ' + key)


def independent_features(case: dict, result: dict, control: dict, network) -> np.ndarray:
    """Reconstruct the six physical features per unit without calling the producer."""
    count, hours = 14, 168
    value = np.empty((hours, count, 6), dtype=np.float64)
    value[:, :, 0] = np.clip(result['state'].T, 0, 1)
    value[:, :, 1] = ((result['flow'] - control['flow']) / case['qmin'][:, None]).T
    value[:, :, 2] = ((result['power'] - control['power']) / case['pmax'][:, None]).T
    value[0, :, 3:5] = 0
    value[1:, :, 3] = result['raw_start'].T
    value[1:, :, 4] = result['raw_stop'].T
    scale = 1 + np.abs(case['price']).mean()
    for unit, (upstream, downstream) in enumerate(zip(network.unit_up, network.unit_down)):
        before = control['water_shadow'][upstream].copy()
        after = result['water_shadow'][upstream].copy()
        if downstream >= 0:
            before -= control['water_shadow'][downstream]
            after -= result['water_shadow'][downstream]
        value[:, unit, 5] = (before - after) / (case['energy'][unit, 0] * scale)
    value = value.reshape(hours, 84).astype(np.float32)
    require(np.isfinite(value).all(), 'Nonfinite reconstructed feature')
    return value


def physical_certificate(solver: ShopRelaxationLP, case: dict, result: dict,
                         lp: dict, proof: dict, control: dict) -> dict:
    """Verify saved global native proof and fixed-state LP primal/dual certificates."""
    require(proof.get('mip_status') == 0 and proof.get('proven_optimal') is True,
            'Native MILP is not proven optimal')
    require(proof['selected_units'] == TARGETS and proof['mip_rel_gap_requested'] == 0,
            'Binary units or native gap request changed')
    require(proof['mip_time_limit_seconds'] in (60, 600) and
            proof['fixed_lp_time_limit_seconds_each'] == 1, 'Solver time budget changed')
    require(proof.get('fixed_lp_replay_bitwise') is True, 'Missing conditional replay')
    require(bool(lp['success']) and int(lp['status']) == 0, 'Conditional LP failed')
    for key in ['minimum_slack', 'maximum_slack', 'bypass_slack', 'vest_slack']:
        require(np.isfinite(result[key]).all() and np.max(np.abs(result[key])) == 0,
                'Nonzero physical slack: ' + key)
    captured = {}

    def no_solve(c, **kwargs):
        captured.update(c=c, **kwargs)
        return SimpleNamespace(success=True, status=0, x=lp['solution'],
                               fun=float(lp['objective']), nit=int(result['iterations']),
                               eqlin=SimpleNamespace(marginals=lp['eqlin_marginals']))

    # Build exact frozen matrices; no optimizer runs.
    with patch('sintef_project.lp_shop_relaxation.linprog', no_solve):
        reconstructed = solver.solve(case, 1., 1.)
    for key, value in reconstructed.items():
        np.testing.assert_allclose(value, result[key], rtol=2e-11, atol=1e-5, err_msg=key)
    selected = solver.state[[UNIT_ORDER.index(unit) for unit in TARGETS]].ravel()
    states = result['integer_states']
    require(states.shape == (3, 168) and np.isin(states, [0, 1]).all(), 'Invalid integer state')
    x = lp['solution']
    require(np.isfinite(x).all(), 'Nonfinite conditional primal')
    np.testing.assert_array_equal(x[selected], states.ravel())
    bounds = captured['bounds'].copy()
    bounds[selected, 0] = bounds[selected, 1] = states.ravel()
    ae, be, au, bu, c = [captured[k] for k in ['A_eq', 'b_eq', 'A_ub', 'b_ub', 'c']]
    residuals = {'eqlin_residual': be - ae @ x, 'ineqlin_residual': bu - au @ x,
                 'lower_residual': x - bounds[:, 0], 'upper_residual': bounds[:, 1] - x}
    for key, value in residuals.items():
        np.testing.assert_allclose(value, lp[key], atol=1e-7, rtol=1e-8, err_msg=key)
    require(max(np.max(np.abs(residuals['eqlin_residual'])),
                -np.min(residuals['ineqlin_residual']), -np.min(residuals['lower_residual']),
                -np.min(residuals['upper_residual'])) < 1e-5, 'Conditional primal infeasible')
    for key in ['eqlin_marginals', 'ineqlin_marginals', 'lower_marginals', 'upper_marginals']:
        require(np.isfinite(lp[key]).all(), 'Nonfinite dual: ' + key)
    stationarity = float(np.max(np.abs(c - ae.T @ lp['eqlin_marginals']
                                      - au.T @ lp['ineqlin_marginals']
                                      - lp['lower_marginals'] - lp['upper_marginals'])))
    require(stationarity < 1e-5, 'Conditional stationarity failed')
    require(lp['ineqlin_marginals'].max() < 1e-7 and lp['lower_marginals'].min() > -1e-7
            and lp['upper_marginals'].max() < 1e-7, 'Conditional dual sign failed')
    finite = np.isfinite(bounds[:, 1])
    require(np.max(np.abs(lp['upper_marginals'][~finite]), initial=0) < 1e-7,
            'Infinite upper bound has nonzero marginal')
    complementarity = max(float(np.max(np.abs(lp['ineqlin_marginals'] * lp['ineqlin_residual']))),
                          float(np.max(np.abs(lp['lower_marginals'] * lp['lower_residual']))),
                          float(np.max(np.abs(lp['upper_marginals'][finite] * lp['upper_residual'][finite]))))
    require(complementarity < 1e-5, 'Conditional complementarity failed')
    dual = float(lp['eqlin_marginals'] @ be + lp['ineqlin_marginals'] @ bu
                 + lp['lower_marginals'] @ bounds[:, 0]
                 + lp['upper_marginals'][finite] @ bounds[finite, 1])
    np.testing.assert_allclose([c @ x, dual], [float(lp['objective'])] * 2, atol=1e-5, rtol=1e-11)
    incumbent = result['integer_solution']
    require(np.isfinite(incumbent).all(), 'Nonfinite native incumbent')
    original = captured['bounds']
    violation = max(float(np.max(np.abs(ae @ incumbent - be))), float(np.max(au @ incumbent - bu)),
                    float(np.max(original[:, 0] - incumbent)), float(np.max(incumbent - original[:, 1])))
    require(violation < 1e-5, 'Native incumbent infeasible')
    np.testing.assert_allclose(incumbent[selected], states.ravel(), atol=1e-5, rtol=0)
    native, bound = proof['mip_objective_scaled'], proof['mip_dual_bound_scaled']
    require(np.isfinite([native, bound, proof['mip_gap']]).all(), 'Nonfinite native proof')
    np.testing.assert_allclose(c @ incumbent, native, atol=1e-5, rtol=2e-11)
    require(bound <= native + 1e-7 and proof['mip_gap'] >= 0, 'Native bound order failed')
    require(float(lp['objective']) <= native + 1e-6, 'Conditional LP worsens native incumbent')
    scale = 1 + np.abs(case['price']).mean()
    np.testing.assert_allclose(proof['objective_gap_eur'], (float(lp['objective']) - bound) * scale,
                               atol=1e-5, rtol=2e-11)
    require(result['objective'] <= control['objective'] + 1e-4, 'Subset objective failed')
    return {'native_incumbent_violation': violation, 'conditional_stationarity': stationarity,
            'conditional_complementarity': complementarity, 'native_relative_gap': proof['mip_gap'],
            'native_absolute_gap_eur': float((native - bound) * scale),
            'conditional_objective_gap_eur': proof['objective_gap_eur']}


def audit_case(folder: Path, receipt: dict, case: dict, control: dict,
               solver: ShopRelaxationLP, expected_attempt_protocol: str,
               protocol: dict) -> tuple[np.ndarray, dict]:
    """Verify both independent MILP replicas and all four conditional LP payloads."""
    verify_hashes(folder, receipt['artifacts'])
    require(receipt['status'] == 'complete' and len(receipt['accepted_attempts']) == 2,
            'Invalid complete receipt')
    same(arrays(folder / 'inputs.npz'), case)
    results, records, checks = [], [], []
    require(len(set(receipt['accepted_attempts'])) == 2, 'Duplicate accepted replica')
    for replica, name in enumerate(receipt['accepted_attempts'], 1):
        attempt = folder / name
        record = json.loads((attempt / 'proof.json').read_text())
        require(record['status'] == 'complete' and record['case_id'] == receipt['case_id'],
                'Attempt identity or status differs')
        # Preserved pilot proofs predate a protocol_sha256 field.
        require(record['environment'] == protocol['environment'], 'Attempt environment differs')
        if name.startswith('pilot_replica_'):
            require(receipt['case_id'] in protocol['pilot_case_ids'] and
                    name == f'pilot_replica_{replica}' and record['attempt'] == replica,
                    'Unapproved pilot replica')
            pilot = ROOT / protocol['pilot_directory']
            prefix = f'case_{receipt["case_id"]:04d}/attempt_{replica}'
            for filename in ['proof.json', 'result.npz', 'fixed_lp_1.npz', 'fixed_lp_2.npz']:
                require(digest(attempt / filename) == protocol['pilot_files'][prefix + '/' + filename]
                        == digest(pilot / prefix / filename), 'Preserved pilot bytes differ')
        else:
            require(record['protocol_sha256'] == expected_attempt_protocol, 'Attempt protocol differs')
            require(record['replica'] == replica and
                    name == f'replica_{replica}_{int(record["milp_limit_seconds"])}s' and
                    record['milp_limit_seconds'] == record['proof']['mip_time_limit_seconds'],
                    'Attempt replica or time differs')
        require(record['input_sha256'] == receipt['input_sha256'] and
                record['control_sha256'] == receipt['control_sha256'], 'Attempt inputs differ')
        require(record['all_slacks_zero'] and record['fixed_lp_all_arrays_bitwise'], 'Attempt gate failed')
        verify_hashes(attempt, record['artifacts'])
        lp, other = arrays(attempt / 'fixed_lp_1.npz'), arrays(attempt / 'fixed_lp_2.npz')
        same(lp, other)
        result = arrays(attempt / 'result.npz')
        checks.append(physical_certificate(solver, case, result, lp, record['proof'], control))
        if record['proof']['mip_time_limit_seconds'] == 600:
            first = folder / f'replica_{record["replica"]}_60s' / 'proof.json'
            failed = json.loads(first.read_text())
            require(failed['proof']['mip_status'] == 1 and failed['milp_limit_seconds'] == 60,
                    'Retry lacks original status-1 evidence')
            require(failed['case_id'] == receipt['case_id'] and
                    failed['replica'] == record['replica'], 'Retry identity differs')
            require(failed['status'] in ['complete', 'failed'] and
                    failed['protocol_sha256'] == expected_attempt_protocol and
                    failed['input_sha256'] == receipt['input_sha256'] and
                    failed['control_sha256'] == receipt['control_sha256'] and
                    failed['environment'] == protocol['environment'] and
                    failed['proof']['mip_time_limit_seconds'] == 60 and
                    datetime.fromisoformat(failed['timestamp_utc']) < datetime.fromisoformat(record['timestamp_utc']),
                    'Retry source, input or execution lineage differs')
            verify_hashes(first.parent, failed['artifacts'])
        results.append(result)
        records.append(record)
    require(records[0]['pid'] != records[1]['pid'], 'Replicas must use separate processes')
    same(results[0], results[1])
    same(results[0], arrays(folder / 'canonical_result.npz'))
    values = independent_features(case, results[0], control, solver.network)
    same({'features': values}, {'features': np.load(folder / 'features.npy', allow_pickle=False)})
    return values, {'case_id': receipt['case_id'], 'certificates': checks,
                    'replica_pids': [r['pid'] for r in records], 'all_replays_bitwise': True}


def main() -> None:
    """Require an externally pinned, finalized cache before full physical verification."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    require(Path(args.run_id).name == args.run_id and args.run_id not in ['', '.', '..'], 'Invalid run ID')
    cache = args.cache if args.cache.is_absolute() else ROOT / args.cache
    out = ROOT / 'artifacts' / args.run_id
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    report = {'status': 'initializing', 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'command': shlex.join([sys.executable, *sys.argv]), 'optimization_calls': 0,
              'labels_decoded': False, 'training': False, 'cases': [], 'kaggle_submission': None}
    try:
        require(digest(cache / 'manifest.json') == args.manifest_sha256, 'External cache hash differs')
        require(digest(cache / 'protocol.json') == PROTOCOL_SHA, 'Wrong frozen protocol')
        m = json.loads((cache / 'manifest.json').read_text())
        p = json.loads((cache / 'protocol.json').read_text())
        check_complete_manifest(m, p)
        require(p['targets'] == TARGETS and p['workers'] == 4 and p['solver_threads'] == 1,
                'Unexpected units or execution recipe')
        require(digest(cache / 'compatible_base_protocol.json') == BASE_SHA, 'Base protocol differs')
        base = json.loads((cache / 'compatible_base_protocol.json').read_text())
        mutable = {'version', 'created_utc', 'workers', 'source', 'resume_policy'}
        require(set(p) == set(base) | {'compatible_base', 'execution'}, 'Protocol field set differs')
        for key in set(base) - mutable:
            require(base[key] == p[key], 'Physical protocol changed: ' + key)
        for name, value in base['source'].items():
            require(p['source'][name] == value, 'Original mathematical source changed')
        for group in ['source', 'data', 'audits']:
            verify_hashes(ROOT, p[group])
        verify_hashes(cache / 'source', p['source'])
        verify_hashes(cache, m['artifacts'])
        require(digest(ROOT / p['prefix_protocol']) == p['prefix_protocol_sha256'], 'Prefix changed')
        continuous = ROOT / p['continuous_cache']
        require(digest(continuous / 'manifest.json') == p['continuous_cache_manifest_sha256'], 'Control cache changed')
        cm = json.loads((continuous / 'manifest.json').read_text())
        old = ROOT / 'artifacts/integer-water-budget-full-001'
        require(digest(old / 'manifest.json') == OLD_MANIFEST_SHA, 'Original finalized cache changed')
        require(m['reuse_manifest_sha256'] == OLD_MANIFEST_SHA == digest(cache / 'reuse_manifest.json'),
                'Copied reuse manifest differs')
        old_m = json.loads((old / 'manifest.json').read_text())
        expected_compatible = set(old_m['completed_cases'])
        require(set(map(str, m['resumed_case_ids'])) == expected_compatible, 'Resumed cases differ')
        require(set(map(str, p['pilot_case_ids'])).issubset(expected_compatible), 'Pilot not imported intact')
        pilot = ROOT / p['pilot_directory']
        require(digest(pilot / 'manifest.json') == p['pilot_manifest_sha256'] and
                digest(pilot / 'protocol.json') == p['pilot_protocol_sha256'], 'Pilot lineage changed')
        verify_hashes(pilot, p['pilot_files'])
        for canonical in p['canonical_diagnostic'].values():
            for item in canonical.values():
                require(digest(ROOT / item['path']) == item['sha256'], 'Canonical diagnostic changed')
        target = (ROOT / 'sintef_project/data/kernel/Unit_commitment_decisions.csv').resolve()
        original_read = pd.read_csv

        def metadata_only(path, *positional, **kwargs):
            if isinstance(path, (str, Path)) and Path(path).resolve() == target:
                require(not positional and kwargs == {'usecols': ['Run No', 'starttime']}, 'Target decoding forbidden')
            return original_read(path, *positional, **kwargs)

        with patch.object(pd, 'read_csv', metadata_only):
            data = HydroCaseData(ROOT)
            require(data.cases['Run No'].tolist() == p['case_ids'] and
                    data.cases.starttime.astype(str).tolist() == p['case_dates'], 'Fresh metadata differs')
            solver = ShopRelaxationLP(data.network)
            rebuilt = np.empty((3647, 168, 84), dtype=np.float32)
            compatible = []
            for index, case_id in enumerate(p['case_ids']):
                folder = cache / 'cases' / f'case_{case_id:04d}'
                require(digest(folder / 'complete.json') == m['completed_cases'][str(case_id)], 'Receipt changed')
                receipt = json.loads((folder / 'complete.json').read_text())
                require(receipt['case_id'] == case_id and receipt['protocol_sha256'] == PROTOCOL_SHA,
                        'Receipt identity differs')
                attempt_protocol = PROTOCOL_SHA
                if str(case_id) in expected_compatible:
                    compatibility = receipt['compatibility']
                    original_folder = old / 'cases' / f'case_{case_id:04d}'
                    original_path = folder / 'compatible_original_complete.json'
                    require(digest(original_path) == old_m['completed_cases'][str(case_id)] ==
                            digest(original_folder / 'complete.json'), 'Original receipt bytes differ')
                    require(compatibility == {'original_protocol_sha256': BASE_SHA,
                            'original_manifest_sha256': OLD_MANIFEST_SHA,
                            'original_complete_sha256': digest(original_path),
                            'original_complete_path': original_path.name,
                            'all_original_payload_bytes_preserved': True}, 'Compatibility receipt differs')
                    original_receipt = json.loads(original_path.read_text())
                    expected = json.loads(json.dumps(original_receipt))
                    expected['protocol_sha256'] = PROTOCOL_SHA
                    expected['compatibility'] = compatibility
                    expected['artifacts'][original_path.name] = digest(original_path)
                    require(expected == receipt, 'Imported receipt has undeclared changes')
                    verify_hashes(original_folder, original_receipt['artifacts'])
                    verify_hashes(folder, original_receipt['artifacts'])
                    attempt_protocol = BASE_SHA
                    compatible.append(case_id)
                else:
                    require('compatibility' not in receipt, 'Undeclared compatible case')
                canonical = p['canonical_diagnostic'].get(str(case_id))
                if canonical:
                    require(receipt['canonical_origin'] == {'kind': 'preserved_diagnostic', 'files': canonical},
                            'Canonical diagnostic origin differs')
                    for kind, filename in [('result', 'canonical_result.npz'), ('proof', 'canonical_proof.json')]:
                        require(digest(folder / filename) == canonical[kind]['sha256'],
                                'Canonical diagnostic payload differs')
                else:
                    first = receipt['accepted_attempts'][0]
                    require(receipt['canonical_origin'] == {'kind': 'first_accepted_replica', 'directory': first},
                            'Canonical replica origin differs')
                    require(digest(folder / 'canonical_proof.json') == digest(folder / first / 'proof.json'),
                            'Canonical first-replica proof differs')
                case = data.case(index)
                case['natural'] = np.asarray(case['natural'], dtype=np.float64)
                for key, field in [('pmin', 'p_min'), ('start_cost', 'startcost_const'), ('stop_cost', 'stopcost_const')]:
                    case[key] = np.asarray([data.model['generator'][u][field] for u in UNIT_ORDER], dtype=np.float64)
                ip, cp = continuous / f'case_{case_id:04d}_inputs.npz', continuous / f'case_{case_id:04d}_result.npz'
                require(digest(ip) == cm['artifacts'][ip.name] == receipt['input_sha256'], 'Physical inputs changed')
                require(digest(cp) == cm['artifacts'][cp.name] == receipt['control_sha256'], 'Physical control changed')
                same(case, arrays(ip))
                rebuilt[index], checked = audit_case(folder, receipt, case, arrays(cp), solver, attempt_protocol, p)
                report['cases'].append(checked)
                if (index + 1) % 100 == 0:
                    print(json.dumps({'audited': index + 1, 'total': 3647, 'seconds': time.monotonic() - started}), flush=True)
        feature_path = cache / 'features.npz'
        require(digest(feature_path) == m['features_sha256'], 'Final feature bytes differ')
        saved = arrays(feature_path)
        require(set(saved) == {'features', 'case_ids', 'units', 'columns'}, 'Unexpected feature schema')
        require(saved['case_ids'].tolist() == p['case_ids'] and saved['units'].tolist() == UNIT_ORDER
                and saved['columns'].tolist() == COLUMNS, 'Feature alignment differs')
        same({'features': rebuilt}, {'features': saved['features']})
        require(digest(cache / 'manifest.json') == args.manifest_sha256, 'Cache mutated during audit')
        report.update(status='passed', cache_manifest_sha256=args.manifest_sha256,
                      cache_protocol_sha256=PROTOCOL_SHA, features_sha256=m['features_sha256'],
                      case_count=3647, all_cases_physical_and_replay_verified=True,
                      compatible_receipts_verified=True, compatible_case_count=len(compatible),
                      compatible_case_ids=compatible, all_84_features_independently_reconstructed=True,
                      caveat='Native solver status/bound provides global MILP certificate; independently reconstructed KKT certifies conditional LP. No new optimization, target values or model scores.')
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        shutil.copy2(__file__, out / Path(__file__).name)
        report.update(runtime_seconds=time.monotonic() - started, audit_source_sha256=digest(Path(__file__)),
                      environment={'python': sys.version, 'platform': platform.platform(), 'numpy': np.__version__, 'pandas': pd.__version__},
                      peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (1 if sys.platform == 'darwin' else 1024))
        (out / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
        with (ROOT / 'docs/experiments.md').open('a') as stream:
            stream.write(f'\n\n## {args.run_id} — {report["timestamp_utc"]}\n\nFull integer-cache physical audit status {report["status"]}; {len(report["cases"])} cases checked. No optimization, target decoding, model fitting or submission. Command `{report["command"]}`. Complete source/protocol/data/result lineage and environment in `artifacts/{args.run_id}/manifest.json`. Runtime {report["runtime_seconds"]:.3f}s; all failures retained.\n')
        print(json.dumps({'status': report['status'], 'manifest_sha256': digest(out / 'manifest.json'), 'case_count': len(report['cases'])}))


if __name__ == '__main__':
    main()
