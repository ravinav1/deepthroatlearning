"""Two-case label-free timing probe of all14 binary states in the unchanged network LP."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
import resource
import shlex
import shutil
import sys
import time
import traceback
from unittest.mock import patch

for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[variable] = '1'

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.diagnose_integer_water_budget import arrays, digest
from scripts.pilot_integer_water_budget import capture_solve, environment
from sintef_project.lp_features import HydroCaseData, UNIT_ORDER

CASES = [1, 1826]
NEW_SOURCE = ['scripts/probe_all14_integer_water_budget_v2.py',
              'tests/test_all14_integer_probe_v2.py', 'docs/all14_integer_water_budget_probe_v2.md']


def identical_arrays(first: dict, second: dict) -> bool:
    """Require exact keys, dtype, shape, and bytes, including zero signs."""
    return set(first) == set(second) and all(
        first[key].dtype == second[key].dtype and first[key].shape == second[key].shape
        and first[key].tobytes() == second[key].tobytes() for key in first)


def all_slacks_exact_zero(result: dict) -> bool:
    """Reject any nonzero physical slack, including negative numerical values."""
    return all(np.max(np.abs(result[key])) == 0 for key in
               ['minimum_slack', 'maximum_slack', 'bypass_slack', 'vest_slack'])


def gate(records: list[dict]) -> dict:
    """Keep proven feasibility and measured resource requirements explicit."""
    return {'both_selected_cases_present': [r['case_id'] for r in records] == CASES,
            'all_solved_and_proven_optimal': len(records) == 2 and all(
                r['status'] == 'complete' and r['proof'].get('proven_optimal')
                and r['proof'].get('mip_status') == 0 for r in records),
            'all_fixed_lp_arrays_byte_exact': len(records) == 2 and all(r.get('fixed_lp_all_arrays_bitwise', False) for r in records),
            'all_slacks_zero': len(records) == 2 and all(r.get('all_slacks_zero', False) for r in records),
            'all_objective_subset_checks': len(records) == 2 and all(r.get('objective_subset_checks', False) for r in records),
            'each_worker_peak_rss_below_2gib': len(records) == 2 and all(0 < r['peak_rss_bytes'] < 2 * 1024**3 for r in records)}


def freeze(path: Path) -> None:
    """Freeze metadata-only cases and all original mathematical/input identities."""
    original_path = ROOT / 'docs/integer_water_budget_protocol.json'
    original = json.loads(original_path.read_text())
    cache = ROOT / original['cache']; cm = json.loads((cache / 'manifest.json').read_text())
    sources = dict(original['source'])
    sources['scripts/pilot_integer_water_budget.py'] = digest(ROOT / 'scripts/pilot_integer_water_budget.py')
    sources.update({name: digest(ROOT / name) for name in NEW_SOURCE})
    comparison = ROOT / 'artifacts/integer-water-budget-completion-001'
    inputs = {}; cases = []
    for case_id in CASES:
        index = cm['case_ids'].index(case_id)
        cases.append({'case_id': case_id, 'metadata_position': index, 'date': cm['case_dates'][index]})
        for kind in ['inputs', 'result']:
            p = cache / f'case_{case_id:04d}_{kind}.npz'
            assert digest(p) == cm['artifacts'][p.name]
            inputs[str(p.relative_to(ROOT))] = digest(p)
        for suffix in ['proof.json', 'integer.npz']:
            p = comparison / f'case_{case_id:04d}_{suffix}'
            inputs[str(p.relative_to(ROOT))] = digest(p)
    proof_path = 'artifacts/integer-water-budget-completion-independent-audit-001/manifest.json'
    inputs[proof_path] = digest(ROOT / proof_path)
    for previous in ['docs/all14_integer_water_budget_probe_protocol.json',
                     'artifacts/all14-integer-water-budget-preflight-001/manifest.json']:
        inputs[previous] = digest(ROOT / previous)
    protocol = {'version': 'all14_integer_water_budget_two_case_probe_v2',
                'created_utc': datetime.now(timezone.utc).isoformat(),
                'revision_reason': 'Unrun v1 exact-zero gate used signed maximum; v2 uses maximum absolute slack and rejects negative numerical residuals. Solver and physical inputs unchanged.',
                'original_protocol_sha256': digest(original_path), 'source': sources,
                'data': original['data'], 'inputs': inputs, 'cache': original['cache'],
                'cache_manifest_sha256': digest(cache / 'manifest.json'),
                'comparison': str(comparison.relative_to(ROOT)), 'cases': cases,
                'unit_indices': list(range(14)), 'unit_order': UNIT_ORDER,
                'milp_seconds': 60., 'fixed_lp_seconds_each': 1., 'mip_rel_gap': 0.,
                'workers': 1, 'attempts_per_case': 1, 'fresh_process_per_case': True,
                'peak_rss_limit_bytes': 2 * 1024**3, 'environment': environment(),
                'labels_decoded': False, 'training': False,
                'selection': 'Parent-prespecified case IDs1 and1826 from existing metadata-selected original controls; no labels or adaptive case choice.',
                'formulation': 'Only unit_indices changes from three prescribed units to all14. All curves, costs, network constraints, initial heads, terminal values, fixed-LP tolerances and native solver settings remain unchanged.',
                'scope': 'One60second MILP plus two1second fixed-state LP calls percase. No retries, fallback, rescue, label diagnostic, feature cache, model training, or submission.',
                'gate': 'Both cases proven optimal with exact entire fixed-LP array bytes, zero slacks, original relaxed and proven3unit objective subset checks, and perworker peakRSS below2GiB. Passing permits only proposal/review of a later diagnostic, not automatic continuation.'}
    with path.open('x') as stream:
        json.dump(protocol, stream, indent=2)
    print(digest(path), flush=True)


def worker(out_name: str, case_id: int, network, case: dict, control: dict,
           three_unit: dict, comparison_proof: dict) -> dict:
    """Run one fresh single-thread all-unit attempt, preserving failures/incumbents."""
    out = Path(out_name) / f'case_{case_id:04d}'
    out.mkdir(exist_ok=False)
    started = time.monotonic(); saved = []
    record = {'case_id': case_id, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'status': 'initializing', 'pid': os.getpid(), 'environment': environment(),
              'comparison_three_unit_proof': comparison_proof, 'proof': {},
              'relaxed_objective': float(control['objective']),
              'three_unit_objective': float(three_unit['objective'])}
    try:
        result, proof, saved = capture_solve(network, case, list(range(14)), 60.)
        record['proof'] = proof
        np.savez_compressed(out / 'result.npz', **result)
        assert proof['selected_units'] == UNIT_ORDER
        assert len(saved) == 2 and identical_arrays(saved[0], saved[1]), 'Full fixed-LP replay mismatch'
        assert result['integer_states'].shape == (14, 168)
        assert np.isin(result['integer_states'], [0, 1]).all()
        assert np.max(np.abs(result['state'] - result['integer_states'])) <= 1e-10
        subset = bool(result['objective'] <= control['objective'] + 1e-4 and result['objective'] <= three_unit['objective'] + 1e-4)
        assert subset, 'All14 feasible subset cannot exceed proven3unit or relaxed optimum'
        slacks = all_slacks_exact_zero(result)
        scale = 1 + np.abs(case['price']).mean()
        record.update(status='complete', objective=float(result['objective']),
                      objective_loss_vs_relaxed_eur=float(control['objective'] - result['objective']),
                      objective_loss_vs_three_unit_eur=float(three_unit['objective'] - result['objective']),
                      mip_dual_upper_bound_eur=float(-proof['mip_dual_bound_scaled'] * scale) if proof.get('mip_dual_bound_scaled') is not None else None,
                      native_absolute_gap_eur=float((proof['mip_objective_scaled'] - proof['mip_dual_bound_scaled']) * scale) if proof.get('mip_dual_bound_scaled') is not None else None,
                      fixed_lp_all_arrays_bitwise=True, all_slacks_zero=bool(slacks), objective_subset_checks=subset)
    except Exception as error:
        saved = getattr(error, 'saved_lp_calls', saved)
        record.update(status='failed', error=traceback.format_exc(),
                      proof=getattr(error, 'saved_integer_proof', record['proof']))
        np.savez_compressed(out / 'failed_incumbent.npz', **getattr(error, 'saved_integer_arrays', {}))
    finally:
        for i, fields in enumerate(saved, 1):
            np.savez_compressed(out / f'fixed_lp_{i}.npz', **fields)
        record.update(runtime_seconds=time.monotonic() - started,
                      peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (1 if sys.platform == 'darwin' else 1024))
        record['artifacts'] = {p.name: digest(p) for p in out.iterdir() if p.is_file()}
        (out / 'proof.json').write_text(json.dumps(record, indent=2))
    return record


def main() -> None:
    """Validate immutable closure and independent review before the bounded probe."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze-protocol', action='store_true')
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--protocol', type=Path, default=ROOT / 'docs/all14_integer_water_budget_probe_protocol_v2.json')
    parser.add_argument('--protocol-sha256')
    parser.add_argument('--independent-audit', type=Path)
    parser.add_argument('--independent-audit-sha256')
    parser.add_argument('--run-id')
    args = parser.parse_args()
    if args.freeze_protocol:
        freeze(args.protocol); return
    if not args.protocol_sha256 or not args.run_id:
        raise ValueError('Explicit unique run ID and protocol hash required')
    out = ROOT / 'artifacts' / args.run_id; out.mkdir(exist_ok=False)
    start = time.monotonic()
    report = {'run_id': args.run_id, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'status': 'initializing', 'command': shlex.join([sys.executable, *sys.argv]),
              'protocol_sha256': args.protocol_sha256, 'environment': environment(),
              'labels_decoded': False, 'training': False, 'kaggle_submission': None, 'cases': []}
    try:
        assert digest(args.protocol) == args.protocol_sha256
        protocol = json.loads(args.protocol.read_text())
        assert protocol['environment'] == environment()
        assert [c['case_id'] for c in protocol['cases']] == CASES
        assert protocol['unit_indices'] == list(range(14)) and protocol['unit_order'] == UNIT_ORDER
        assert protocol['milp_seconds'] == 60. and protocol['fixed_lp_seconds_each'] == 1.
        assert protocol['workers'] == protocol['attempts_per_case'] == 1
        assert protocol['peak_rss_limit_bytes'] == 2 * 1024**3
        for group in ['source', 'data', 'inputs']:
            for name, expected in protocol[group].items(): assert digest(ROOT / name) == expected, name
        if not args.preflight:
            assert args.independent_audit and args.independent_audit_sha256
            assert digest(args.independent_audit) == args.independent_audit_sha256
            independent = json.loads(args.independent_audit.read_text())
            assert independent['status'] == 'passed' and independent['protocol_sha256'] == args.protocol_sha256
            report['independent_audit_sha256'] = args.independent_audit_sha256
        shutil.copy2(args.protocol, out / 'protocol.json')
        for name in protocol['source']:
            target = out / 'source' / name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT / name, target)
        cache = ROOT / protocol['cache']; assert digest(cache / 'manifest.json') == protocol['cache_manifest_sha256']
        original_read = pd.read_csv
        target_path = (ROOT / 'sintef_project/data/kernel/Unit_commitment_decisions.csv').resolve()
        def guarded(path, *positional, **kwargs):
            if isinstance(path, (str, Path)) and Path(path).resolve() == target_path:
                if positional or kwargs != {'usecols': ['Run No', 'starttime']}:
                    raise ValueError('Probe may read target metadata only')
            return original_read(path, *positional, **kwargs)
        jobs = []
        with patch.object(pd, 'read_csv', guarded):
            data = HydroCaseData(ROOT)
            for selected in protocol['cases']:
                case_id, index = selected['case_id'], selected['metadata_position']
                assert int(data.cases.iloc[index]['Run No']) == case_id
                assert str(data.cases.iloc[index].starttime) == selected['date']
                rebuilt = data.case(index); rebuilt['natural'] = np.asarray(rebuilt['natural'], dtype=np.float64)
                for name, field in [('pmin', 'p_min'), ('start_cost', 'startcost_const'), ('stop_cost', 'stopcost_const')]:
                    rebuilt[name] = np.asarray([data.model['generator'][unit][field] for unit in UNIT_ORDER], dtype=np.float64)
                input_path = cache / f'case_{case_id:04d}_inputs.npz'
                case = arrays(input_path)
                assert identical_arrays(case, rebuilt), 'Frozen input byte mismatch'
                shutil.copy2(input_path, out / input_path.name)
                comparison = ROOT / protocol['comparison']
                previous = json.loads((comparison / f'case_{case_id:04d}_proof.json').read_text())
                assert previous['status'] == 'complete' and previous['proof']['proven_optimal']
                assert previous['input_sha256'] == digest(input_path)
                jobs.append((str(out), case_id, data.network, case,
                             arrays(cache / f'case_{case_id:04d}_result.npz'),
                             arrays(comparison / f'case_{case_id:04d}_integer.npz'), previous))
        report['fresh_input_reconstruction_byte_exact'] = True
        if args.preflight:
            report.update(status='passed_preflight_only', solves=0); return
        with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn'), max_tasks_per_child=1) as pool:
            for job in jobs:
                result = pool.submit(worker, *job).result()
                report['cases'].append(result)
                print(json.dumps({'case_id': result['case_id'], 'status': result['status'],
                                  'proven_optimal': result['proof'].get('proven_optimal'),
                                  'mip_status': result['proof'].get('mip_status'),
                                  'seconds': result['runtime_seconds'], 'peak_rss_bytes': result['peak_rss_bytes']}), flush=True)
        gates = gate(report['cases'])
        report.update(status='complete', gates=gates, feasibility_passed=all(gates.values()))
    except BaseException:
        report.update(status='failed', error=traceback.format_exc()); raise
    finally:
        report['runtime_seconds'] = time.monotonic() - start
        report['artifacts'] = {p.relative_to(out).as_posix(): digest(p) for p in out.rglob('*') if p.is_file() and p.name != 'manifest.json'}
        (out / 'manifest.json').write_text(json.dumps(report, indent=2))
        (out / 'experiment_record.md').write_text(f'\n## {args.run_id} — {report["timestamp_utc"]}\n\nStatus {report["status"]}. Two fixed cases1/1826; all14 binary states, unchanged network economics. One fresh single-thread process percase;60second MILP and two1second fixedLP caps, no retry/fallback. Command `{report["command"]}`. Runtime {report["runtime_seconds"]:.3f}s; feasibility gate {report.get("feasibility_passed")}. Full source/data/input/environment/status/bounds/replay/RSS proof: `artifacts/{args.run_id}/manifest.json`. No target values, model fit, cache modification, submission, or automatic continuation.\n')


if __name__ == '__main__':
    main()
