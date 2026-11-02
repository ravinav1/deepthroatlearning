"""Separately declared case2012 numerical completion; original failures stay failed."""
from __future__ import annotations

import os
for _name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[_name] = '1'

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from fractions import Fraction
import json
import multiprocessing
from pathlib import Path
import shlex
import shutil
import sys
import time
import traceback
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_integer_water_budget_cache import run_attempt, same_arrays
from scripts.pilot_integer_water_budget import environment
from scripts.audit_integer_water_budget_cache import (
    arrays, digest, require, verify_hashes, physical_certificate, np, pd,
)
from scripts.certify_integer_objective_gap import exact_bound
from sintef_project.lp_features import HydroCaseData, UNIT_ORDER
from sintef_project.lp_hyljelihyl_v2 import capture_problem

CASE_ID = 2012
TARGETS = ['Hogga_G1', 'Lio_G1', 'Haukeli_G1']
ORIGINAL_PROTOCOL = 'docs/integer_water_budget_four_worker_protocol_v2.json'
ORIGINAL_PROTOCOL_SHA = '7571930990cadc542c7ad4f47e7f1195d7a239885c887ad45b051afd9b9377d9'
CERTIFICATE = 'artifacts/integer-case2012-exact-certificate-001/manifest.json'
CERTIFICATE_SHA = 'a6892f16ca3677f40f21ac24fd896b7202f5f5212fbd5b83488961c15c26916c'
CERTIFICATE_AUDIT = 'artifacts/integer-case2012-exact-certificate-independent-audit-001/manifest.json'
CERTIFICATE_AUDIT_SHA = '121b90dcf8394a9967aab609134b19cf29e8c99591d2fae4315a024317e7c926'
ORIGINAL_CASE = 'artifacts/integer-water-budget-full-four-worker-001/cases/case_2012'
PAYLOADS = ['failed_incumbent.npz', 'fixed_lp_1.npz', 'fixed_lp_2.npz']
GUARD_ERROR = 'Fixed-state LP worsened the MILP incumbent objective'
AUDITOR_GUARD_ERROR = 'Conditional LP worsens native incumbent'
EXTRA_SOURCES = [
    'scripts/complete_integer_case2012.py', 'tests/test_integer_case2012_completion.py',
    'docs/integer_case2012_completion.md', 'scripts/certify_integer_objective_gap.py',
    'scripts/audit_integer_water_budget_cache.py', 'scripts/audit_hyljelihyl_river_results.py',
    'scripts/audit_hyljelihyl_river_cost_v2.py', 'scripts/diagnose_hyljelihyl_river_cost_v2.py',
    'scripts/probe_all14_integer_water_budget_v2.py', 'scripts/continue_integer_water_budget_cache_v2.py',
    'sintef_project/lp_hyljelihyl_v2.py',
]


def failed_record(record: dict) -> None:
    """Only the preserved native-optimal numerical guard is eligible for inspection."""
    proof = record['proof']
    require(record['case_id'] == CASE_ID and record['status'] == 'failed', 'Wrong case or original status')
    require(record['milp_limit_seconds'] == 60 and GUARD_ERROR in record['error'], 'Wrong failure or limit')
    require(proof['mip_status'] == 0 and proof['proven_optimal'] is True, 'Native solver did not report optimum')
    require(proof['selected_units'] == TARGETS and proof['mip_rel_gap_requested'] == 0,
            'Integer problem changed')
    require(proof['mip_time_limit_seconds'] == 60 and proof['fixed_lp_time_limit_seconds_each'] == 1,
            'Solver budgets changed')
    require(set(record['artifacts']) == set(PAYLOADS), 'Unexpected failure payload set')


def dimensions(problem: dict, native: dict, lp: dict, scale: float) -> None:
    """Reject truncated operands before calling the exact-rational certificate."""
    n = len(problem['c'])
    require(np.isfinite(scale) and scale >= 1, 'Invalid positive EUR scale')
    require(np.asarray(problem['c']).shape == (n,) and problem['bounds'].shape == (n, 2), 'Invalid variable dimensions')
    require(native['integer_solution'].shape == lp['solution'].shape == (n,), 'Primal dimensions differ')
    require(native['integer_states'].shape == (3, 168), 'Integer state dimensions differ')
    for kind, matrix, rhs in [('eqlin', 'A_eq', 'b_eq'), ('ineqlin', 'A_ub', 'b_ub')]:
        rows = len(problem[rhs])
        require(problem[matrix].shape == (rows, n) and np.asarray(problem[rhs]).shape == (rows,), 'Matrix dimensions differ')
        require(lp[kind + '_marginals'].shape == lp[kind + '_residual'].shape == (rows,), 'Dual dimensions differ')
    for kind in ['lower', 'upper']:
        require(lp[kind + '_marginals'].shape == lp[kind + '_residual'].shape == (n,), 'Bound dimensions differ')


def freeze(path: Path) -> None:
    """Pin this one-case exception and original evidence without solving anything."""
    require(digest(ROOT / ORIGINAL_PROTOCOL) == ORIGINAL_PROTOCOL_SHA, 'Original protocol changed')
    old = json.loads((ROOT / ORIGINAL_PROTOCOL).read_text())
    require(digest(ROOT / CERTIFICATE) == CERTIFICATE_SHA, 'Read-only certificate changed')
    certificate = json.loads((ROOT / CERTIFICATE).read_text())
    require(certificate['status'] == 'passed_read_only_certificate' and certificate['case_id'] == CASE_ID,
            'Wrong read-only certificate')
    require(digest(ROOT / CERTIFICATE_AUDIT) == CERTIFICATE_AUDIT_SHA, 'Certificate audit changed')
    review = json.loads((ROOT / CERTIFICATE_AUDIT).read_text())
    require(review['status'] == 'passed' and review['exact_arithmetic_verified'] is True
            and review['certificate_manifest_sha256'] == CERTIFICATE_SHA
            and review['certificate_source_sha256'] == certificate['source_sha256'], 'Certificate review differs')
    for name, expected in old['source'].items():
        require(digest(ROOT / name) == expected, 'Original mathematical source changed: ' + name)
    sources = {**old['source'], **{name: digest(ROOT / name) for name in EXTRA_SOURCES}}
    original = []
    for replica in [1, 2]:
        directory = ROOT / ORIGINAL_CASE / f'replica_{replica}_60s'
        record = json.loads((directory / 'proof.json').read_text())
        failed_record(record)
        require(record['protocol_sha256'] == ORIGINAL_PROTOCOL_SHA and record['replica'] == replica,
                'Original lineage differs')
        verify_hashes(directory, record['artifacts'])
        original.append({'directory': str(directory.relative_to(ROOT)),
                         'proof_sha256': digest(directory / 'proof.json'),
                         'payloads': record['artifacts']})
    protocol = {
        'version': 'integer_case2012_exact_certificate_completion_v1',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'allowed_case_ids': [CASE_ID],
        'acceptance_basis': 'case2012_dual_feasibility_certificate',
        'original_protocol': ORIGINAL_PROTOCOL, 'original_protocol_sha256': ORIGINAL_PROTOCOL_SHA,
        'original_attempts': original, 'original_status_preserved': 'failed',
        'source': sources, 'data': old['data'], 'environment': environment(),
        'cache': old['continuous_cache'], 'cache_manifest_sha256': old['continuous_cache_manifest_sha256'],
        'certificate': CERTIFICATE, 'certificate_manifest_sha256': CERTIFICATE_SHA,
        'certificate_source_sha256': certificate['source_sha256'],
        'certificate_audit': CERTIFICATE_AUDIT, 'certificate_audit_sha256': CERTIFICATE_AUDIT_SHA,
        'certificate_inputs': certificate['inputs'],
        'case_input_sha256': record['input_sha256'], 'case_control_sha256': record['control_sha256'],
        'targets': TARGETS, 'replicas': 2, 'workers': 2, 'solver_threads': 1,
        'milp_seconds': 60, 'conditional_lp_seconds_each': 1, 'retry': False,
        'absolute_eur_bound': {'numerator': '1', 'denominator': '100000'},
        'physical_contract': 'Original solver and physical auditor unchanged; accept only the specific objective-guard failure after all preceding physical/KKT checks, then check the remaining subset condition. Exact certificate required separately.',
        'replay_contract': 'Every fresh NPZ file and array must match BOTH original failed replicas. Two newly spawned processes, one attempt each; no alternate solution selection.',
        'aggregate_eligible': False, 'old_v2_eligible': False,
        'aggregate_contract': 'A later independently reviewed new aggregate namespace/protocol/loader/auditor must explicitly preserve ordinary/imported/exception lineage. No original proof or complete receipt may be rewritten.',
        'execution_contract': 'Preparation does not authorize solves. Root must authorize execution after independent source/protocol review; runtime also requires both externally pinned review manifests.',
        'labels_decoded': False, 'training': False,
    }
    for group in ['source', 'data', 'certificate_inputs']:
        for name, expected in protocol[group].items():
            require(digest(ROOT / name) == expected, 'Pinned input changed: ' + name)
    with path.open('x') as stream:
        json.dump(protocol, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'prepared_protocol_sha256': digest(path), 'optimization_calls': 0}))


def check_protocol(path: Path, expected: str) -> dict:
    """Validate every dependency and the exact one-case scope before worker creation."""
    require(digest(path) == expected, 'Completion protocol changed')
    p = json.loads(path.read_text())
    require(p['version'] == 'integer_case2012_exact_certificate_completion_v1'
            and p['allowed_case_ids'] == [CASE_ID], 'Exception scope changed')
    require(p['environment'] == environment(), 'Environment changed')
    require(p['original_protocol_sha256'] == ORIGINAL_PROTOCOL_SHA
            and digest(ROOT / p['original_protocol']) == ORIGINAL_PROTOCOL_SHA, 'Original protocol changed')
    for group in ['source', 'data', 'certificate_inputs']:
        for name, expected_hash in p[group].items():
            require(digest(ROOT / name) == expected_hash, 'Dependency changed: ' + name)
    require(digest(ROOT / p['certificate']) == p['certificate_manifest_sha256'] == CERTIFICATE_SHA,
            'Certificate changed')
    require(digest(ROOT / p['certificate_audit']) == p['certificate_audit_sha256'] == CERTIFICATE_AUDIT_SHA,
            'Certificate independent review changed')
    require(digest(ROOT / p['cache'] / 'manifest.json') == p['cache_manifest_sha256'], 'Physical input cache changed')
    for old in p['original_attempts']:
        folder = ROOT / old['directory']
        require(digest(folder / 'proof.json') == old['proof_sha256'], 'Original failure changed')
        record = json.loads((folder / 'proof.json').read_text())
        failed_record(record)
        verify_hashes(folder, old['payloads'])
    return p


def fresh_worker(out: str, replica: int, network, p: dict, protocol_sha: str) -> dict:
    """Call the original attempt runner; observe LP iterations in a separate sidecar."""
    import scripts.pilot_integer_water_budget as pilot
    original_linprog = pilot.linprog
    metadata = []

    def observe(*args, **kwargs):
        result = original_linprog(*args, **kwargs)
        metadata.append({'iterations': int(result.nit), 'status': int(result.status)})
        return result

    with patch.object(pilot, 'linprog', observe):
        result = run_attempt(out, CASE_ID, replica, 60., network, str(ROOT / p['cache']),
                             p['case_input_sha256'], p['case_control_sha256'], protocol_sha)
    return {'attempt': result, 'lp_call_metadata': metadata}


def verify_fresh(folder: Path, record: dict, p: dict) -> None:
    """Require exact original raw payloads and unchanged native optimization result."""
    failed_record(record)
    require(record['environment'] == p['environment'], 'Fresh environment differs')
    require(record['input_sha256'] == p['case_input_sha256']
            and record['control_sha256'] == p['case_control_sha256'], 'Fresh inputs differ')
    verify_hashes(folder, record['artifacts'])
    for old in p['original_attempts']:
        original = ROOT / old['directory']
        old_record = json.loads((original / 'proof.json').read_text())
        for name in PAYLOADS:
            require(digest(folder / name) == old['payloads'][name], 'Fresh payload file differs: ' + name)
            same_arrays(arrays(folder / name), arrays(original / name))
        for key in ['mip_status', 'mip_message', 'mip_gap', 'mip_node_count', 'mip_objective_scaled',
                    'mip_dual_bound_scaled', 'proven_optimal', 'has_incumbent', 'seed']:
            require(record['proof'][key] == old_record['proof'][key], 'Fresh native result differs: ' + key)


def verify_physics(network, case: dict, control: dict, folder: Path,
                   record: dict, iterations: int, expected_certificate: dict) -> tuple[dict, dict]:
    """Run unchanged physical checks, then permit only the declared numerical exception."""
    model, problem = capture_problem(network, case, 1.)
    native, lp = arrays(folder / 'failed_incumbent.npz'), arrays(folder / 'fixed_lp_1.npz')
    same_arrays(lp, arrays(folder / 'fixed_lp_2.npz'))
    selected = model.state[[UNIT_ORDER.index(unit) for unit in TARGETS]].ravel()
    fixed = {**problem, 'bounds': problem['bounds'].copy()}
    fixed['bounds'][selected, 0] = fixed['bounds'][selected, 1] = native['integer_states'].ravel()
    scale = float(1 + np.abs(case['price']).mean())
    dimensions(fixed, native, lp, scale)
    certificate = exact_bound(fixed, native['integer_solution'], lp,
                              record['proof']['mip_objective_scaled'], scale)
    require(certificate == expected_certificate, 'Exact certificate differs from the original evidence')
    bound = certificate['bound_eur']
    require(Fraction(int(bound['numerator']), int(bound['denominator'])) <= Fraction(1, 100000),
            'Exact EUR certificate exceeds the declared budget')

    def no_solve(c, **kwargs):
        return SimpleNamespace(success=bool(lp['success']), status=int(lp['status']),
                               x=lp['solution'], fun=float(lp['objective']), nit=iterations,
                               eqlin=SimpleNamespace(marginals=lp['eqlin_marginals']))

    with patch('sintef_project.lp_shop_relaxation.linprog', no_solve):
        result = model.solve(case, 1., 1.)
    result.update(native)
    # These supplemental assertions are established from saved payloads; never
    # write them into either the original or fresh failed attempt proof.
    check_proof = {**record['proof'], 'fixed_lp_replay_bitwise': True,
                   'objective_gap_eur': float((lp['objective'] - record['proof']['mip_dual_bound_scaled']) * scale)}
    try:
        physical_certificate(model, case, result, lp, check_proof, control)
    except ValueError as error:
        require(str(error) == AUDITOR_GUARD_ERROR, 'An unrelated physical check failed: ' + str(error))
    else:
        raise ValueError('Original objective guard unexpectedly passed')
    # The original auditor stops at its numerical guard. These are the unchanged
    # subsequent checks, explicitly applied under the new exception contract.
    np.testing.assert_allclose(check_proof['objective_gap_eur'],
        (float(lp['objective']) - record['proof']['mip_dual_bound_scaled']) * scale, atol=1e-5, rtol=2e-11)
    require(result['objective'] <= control['objective'] + 1e-4, 'Subset objective failed')
    xn = native['integer_solution']
    require(max(float(np.max(fixed['bounds'][:, 0] - xn)),
                float(np.max(xn - fixed['bounds'][:, 1]))) < 1e-5, 'Native fixed-bound violation')
    return result, {'certificate': certificate, 'original_physical_guard_still_fails': True,
                    'all_other_original_physical_checks_passed': True,
                    'conditional_objective_gap_eur': check_proof['objective_gap_eur']}


def main() -> None:
    """Prepare only by default; execution requires explicit pinned independent reviews."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--protocol', type=Path, default=ROOT / 'docs/integer_case2012_completion_protocol.json')
    ap.add_argument('--protocol-sha256')
    ap.add_argument('--freeze-protocol', action='store_true')
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--run-id')
    ap.add_argument('--certificate-audit', type=Path)
    ap.add_argument('--certificate-audit-sha256')
    ap.add_argument('--completion-audit', type=Path)
    ap.add_argument('--completion-audit-sha256')
    a = ap.parse_args()
    if a.freeze_protocol:
        require(not a.execute, 'Cannot freeze and execute together')
        freeze(a.protocol)
        return
    require(a.protocol_sha256 is not None, 'Explicit protocol SHA required')
    p = check_protocol(a.protocol, a.protocol_sha256)
    if not a.execute:
        print(json.dumps({'status': 'passed_preflight_only', 'optimization_calls': 0,
                          'protocol_sha256': a.protocol_sha256}))
        return
    require(a.run_id is not None and Path(a.run_id).name == a.run_id
            and a.run_id not in ['', '.', '..'], 'Unique run ID required')
    for name in ['certificate', 'completion']:
        path, expected = getattr(a, name + '_audit'), getattr(a, name + '_audit_sha256')
        require(path is not None and expected is not None and digest(path) == expected,
                'Externally pinned ' + name + ' audit required')
        review = json.loads(path.read_text())
        require(review['status'] == 'passed', 'Independent review failed')
        if name == 'certificate':
            require(expected == p['certificate_audit_sha256']
                    and review['certificate_manifest_sha256'] == CERTIFICATE_SHA
                    and review['certificate_source_sha256'] == p['certificate_source_sha256']
                    and review['exact_arithmetic_verified'] is True, 'Certificate review differs')
        else:
            require(review['protocol_sha256'] == a.protocol_sha256
                    and review['source_sha256'] == digest(Path(__file__)), 'Completion review differs')
    out = ROOT / 'artifacts' / a.run_id
    out.mkdir(exist_ok=False)
    start = time.monotonic()
    report = {'status': 'initializing', 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'command': shlex.join([sys.executable, *sys.argv]), 'case_id': CASE_ID,
              'protocol_sha256': a.protocol_sha256, 'environment': environment(),
              'acceptance_basis': p['acceptance_basis'], 'labels_decoded': False, 'training': False,
              'original_status_preserved': 'failed', 'old_v2_eligible': False,
              'aggregate_eligible': False, 'new_aggregate_namespace_required': True,
              'certificate_audit_sha256': a.certificate_audit_sha256,
              'completion_audit_sha256': a.completion_audit_sha256, 'replicas': []}
    try:
        target = (ROOT / 'sintef_project/data/kernel/Unit_commitment_decisions.csv').resolve()
        read_csv = pd.read_csv

        def metadata(path, *args, **kwargs):
            if isinstance(path, (str, Path)) and Path(path).resolve() == target:
                require(not args and kwargs == {'usecols': ['Run No', 'starttime']}, 'Labels forbidden')
            return read_csv(path, *args, **kwargs)

        with patch.object(pd, 'read_csv', metadata):
            data = HydroCaseData(ROOT)
        ip, cp = [ROOT / p['cache'] / f'case_2012_{suffix}.npz' for suffix in ['inputs', 'result']]
        require(digest(ip) == p['case_input_sha256'] and digest(cp) == p['case_control_sha256'], 'Case inputs changed')
        case, control = arrays(ip), arrays(cp)
        require(int(data.cases.iloc[CASE_ID - 1]['Run No']) == CASE_ID, 'Metadata case differs')
        rebuilt = data.case(CASE_ID - 1)
        rebuilt['natural'] = rebuilt['natural'].astype(np.float64)
        for key, field in [('pmin', 'p_min'), ('start_cost', 'startcost_const'), ('stop_cost', 'stopcost_const')]:
            rebuilt[key] = np.asarray([data.model['generator'][unit][field] for unit in UNIT_ORDER], dtype=np.float64)
        same_arrays(case, rebuilt)
        with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn'),
                                 max_tasks_per_child=1) as executor:
            futures = [executor.submit(fresh_worker, str(out), replica, data.network, p, a.protocol_sha256)
                       for replica in [1, 2]]
            fresh = [future.result() for future in futures]
        require(len({item['attempt']['record']['pid'] for item in fresh}) == 2, 'Fresh processes are not distinct')
        certificate = json.loads((ROOT / p['certificate']).read_text())
        results = []
        for replica, item in enumerate(fresh, 1):
            record = item['attempt']['record']
            folder = out / 'cases' / 'case_2012' / f'replica_{replica}_60s'
            require(record['protocol_sha256'] == a.protocol_sha256 and record['replica'] == replica,
                    'Fresh protocol or replica differs')
            require(record['timestamp_utc'] >= report['timestamp_utc'], 'Fresh attempt predates execution')
            verify_fresh(folder, record, p)
            calls = item['lp_call_metadata']
            require(len(calls) == 2 and calls[0] == calls[1] and calls[0]['status'] == 0,
                    'Conditional LP call metadata differs')
            result, checks = verify_physics(data.network, case, control, folder, record,
                                           calls[0]['iterations'], certificate['replicas'][replica - 1]['certificate'])
            results.append(result)
            report['replicas'].append({'replica': replica, 'pid': record['pid'],
                                      'fresh_failed_proof': str((folder / 'proof.json').relative_to(out)),
                                      'fresh_failed_proof_sha256': digest(folder / 'proof.json'),
                                      'lp_call_metadata': calls, 'checks': checks})
        same_arrays(results[0], results[1])
        np.savez_compressed(out / 'exception_case2012_result.npz', **results[0])
        # Prove the old failed evidence was not mutated while repeating the calls.
        check_protocol(a.protocol, a.protocol_sha256)
        report.update(status='complete_case2012_exception', original_attempts=p['original_attempts'],
                      original_protocol_sha256=ORIGINAL_PROTOCOL_SHA,
                      certificate_manifest_sha256=CERTIFICATE_SHA,
                      all_fresh_payloads_match_both_original_replicas=True,
                      all_physical_and_exact_certificate_checks_passed=True,
                      declared_optimization_calls={'milp': 2, 'conditional_lp': 4})
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        shutil.copy2(a.protocol, out / 'protocol.json')
        shutil.copy2(__file__, out / Path(__file__).name)
        report.update(runtime_seconds=time.monotonic() - start, source_sha256=digest(Path(__file__)),
                      artifacts={str(path.relative_to(out)): digest(path) for path in out.rglob('*')
                                 if path.is_file() and path != out / 'manifest.json'})
        (out / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({'status': report['status'], 'manifest_sha256': digest(out / 'manifest.json')}), flush=True)


if __name__ == '__main__':
    main()
