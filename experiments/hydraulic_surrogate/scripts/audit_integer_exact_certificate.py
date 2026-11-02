"""Independent exact-rational verification of preserved case2012 failure evidence."""
from datetime import datetime, timezone
from fractions import Fraction
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from unittest.mock import patch
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[name] = '1'
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_hyljelihyl_river_cost_v2 import sha, load_arrays, literal_arrays
from scripts.audit_hyljelihyl_river_results import check_lp
from sintef_project.lp_hyljelihyl_v2 import capture_problem
from sintef_project.lp_features import HydroCaseData, UNIT_ORDER


def decode(value):
    return Fraction(int(value['numerator']), int(value['denominator']))


def independent_exact(problem, native, lp, native_reported, scale):
    """Recompute via sparse row residuals and separate transpose dual products."""
    memo = {}

    def f(value):
        value = float(value)
        assert np.isfinite(value)
        if value not in memo:
            memo[value] = Fraction.from_float(value)
        return memo[value]

    c, xn, xl = [list(map(f, value)) for value in [problem['c'], native, lp['solution']]]
    n = len(c)
    assert len(xn) == len(xl) == n and problem['bounds'].shape == (n, 2) and scale >= 1
    delta = [a-b for a, b in zip(xl, xn)]
    residuals, duals, dual_products = {}, {}, []
    for tag, dual_name in [('eq', 'eqlin_marginals'), ('ub', 'ineqlin_marginals')]:
        matrix = problem['A_'+tag].tocsr()
        rhs = list(map(f, problem['b_'+tag]))
        dual = list(map(f, lp[dual_name]))
        assert matrix.shape == (len(rhs), n) and len(rhs) == len(dual)
        residual = [[], []]
        for row in range(matrix.shape[0]):
            start, end = matrix.indptr[row:row+2]
            indices, coefficients = matrix.indices[start:end], list(map(f, matrix.data[start:end]))
            for output, values in zip(residual, [xn, xl]):
                output.append(sum((a*values[j] for j, a in zip(indices, coefficients)), Fraction(0))-rhs[row])
        transpose = matrix.T.tocsr()
        product = []
        for col in range(n):
            start, end = transpose.indptr[col:col+2]
            product.append(sum((f(a)*dual[j] for j, a in zip(transpose.indices[start:end], transpose.data[start:end])), Fraction(0)))
        residuals[tag], duals[tag] = residual, dual
        dual_products.append(product)
    lower, upper = list(map(f, lp['lower_marginals'])), list(map(f, lp['upper_marginals']))
    assert len(lower) == len(upper) == n
    stationarity = [c[j]-dual_products[0][j]-dual_products[1][j]-lower[j]-upper[j] for j in range(n)]
    terms = {name: Fraction(0) for name in [
        'equality_residuals', 'fixed_inequality_complementarity', 'native_inequality_violations',
        'wrong_sign_inequality_terms', 'fixed_lower_complementarity', 'native_lower_violations',
        'wrong_sign_lower_terms', 'fixed_upper_complementarity', 'native_upper_violations',
        'wrong_sign_upper_terms', 'infinite_upper_terms', 'stationarity_terms',
        'reported_lp_dot_discrepancy', 'reported_native_dot_discrepancy']}
    identity = Fraction(0)
    for tag in ['eq', 'ub']:
        for rn, rl, y in zip(*residuals[tag], duals[tag]):
            identity += y*(rl-rn)
            if tag == 'eq':
                terms['equality_residuals'] += abs(y*rn)+abs(y*rl)
            else:
                terms['fixed_inequality_complementarity'] += abs(y*rl)
                key = 'native_inequality_violations' if y <= 0 else 'wrong_sign_inequality_terms'
                terms[key] += max(-y*rn, 0)
    for j, (lo, hi) in enumerate(problem['bounds']):
        a, b, lo = lower[j], upper[j], f(lo)
        identity += (a+b)*delta[j]
        terms['fixed_lower_complementarity'] += abs(a*(xl[j]-lo))
        key = 'native_lower_violations' if a >= 0 else 'wrong_sign_lower_terms'
        terms[key] += max(-a*(xn[j]-lo), 0)
        if np.isposinf(hi):
            terms['infinite_upper_terms'] += abs(b*delta[j])
        else:
            hi = f(hi)
            terms['fixed_upper_complementarity'] += abs(b*(xl[j]-hi))
            key = 'native_upper_violations' if b <= 0 else 'wrong_sign_upper_terms'
            terms[key] += max(-b*(xn[j]-hi), 0)
    identity += sum((r*d for r, d in zip(stationarity, delta)), Fraction(0))
    terms['stationarity_terms'] = sum((abs(r*d) for r, d in zip(stationarity, delta)), Fraction(0))
    true_lp = sum((a*b for a, b in zip(c, xl)), Fraction(0))
    true_native = sum((a*b for a, b in zip(c, xn)), Fraction(0))
    assert identity == true_lp-true_native
    terms['reported_lp_dot_discrepancy'] = abs(f(lp['objective'])-true_lp)
    terms['reported_native_dot_discrepancy'] = abs(f(native_reported)-true_native)
    bound = sum(terms.values(), Fraction(0))
    reported = f(lp['objective'])-f(native_reported)
    assert reported <= bound and all(value >= 0 for value in terms.values())
    return {'parts': terms, 'exact_difference': true_lp-true_native, 'reported_difference': reported,
            'bound': bound, 'euro_bound': bound*f(scale), 'max_stationarity': max(map(abs, stationarity)),
            'native_exact_eq_violation': max(map(abs, residuals['eq'][0])),
            'native_exact_ub_violation': max(max(residuals['ub'][0]), 0)}


def main():
    out = ROOT/'artifacts/integer-case2012-exact-certificate-independent-audit-001'
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    report = {'status': 'initializing', 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'command': [sys.executable, *sys.argv], 'optimization_calls': 0, 'labels_decoded': False,
              'eligibility_changed': False, 'retry_authorized': False, 'replicas': []}
    try:
        mp = ROOT/'artifacts/integer-case2012-exact-certificate-001/manifest.json'
        assert sha(mp) == 'a6892f16ca3677f40f21ac24fd896b7202f5f5212fbd5b83488961c15c26916c'
        m = json.loads(mp.read_text())
        source = ROOT/'scripts/certify_integer_objective_gap.py'
        assert sha(source) == m['source_sha256'] == '57374b3bb6568d56ed5089a7483780ee23c9c86f893d9eb277709bcb6ef30fe9'
        assert source.read_bytes() == (mp.parent/source.name).read_bytes()
        assert m['status'] == 'passed_read_only_certificate' and not m['eligibility_changed'] and not m['retry_authorized']
        assert m['optimization_calls'] == 0 and not m['labels_decoded']
        for name, expected in m['inputs'].items():
            assert sha(ROOT/name) == expected, name
        protocol_path = ROOT/'docs/integer_water_budget_four_worker_protocol_v2.json'
        assert sha(protocol_path) == '7571930990cadc542c7ad4f47e7f1195d7a239885c887ad45b051afd9b9377d9'
        protocol = json.loads(protocol_path.read_text())
        for group in ['source', 'data']:
            for name, expected in protocol[group].items():
                assert sha(ROOT/name) == expected, name
        target = (ROOT/'sintef_project/data/kernel/Unit_commitment_decisions.csv').resolve()
        read = pd.read_csv

        def guarded(path, *args, **kwargs):
            if isinstance(path, (str, Path)) and Path(path).resolve() == target:
                assert not args and kwargs == {'usecols': ['Run No', 'starttime']}
            return read(path, *args, **kwargs)

        with patch.object(pd, 'read_csv', guarded):
            data = HydroCaseData(ROOT)
            assert int(data.cases.iloc[2011]['Run No']) == 2012
            fresh = data.case(2011)
            fresh['natural'] = np.asarray(fresh['natural'], dtype=np.float64)
            for name, field in [('pmin', 'p_min'), ('start_cost', 'startcost_const'), ('stop_cost', 'stopcost_const')]:
                fresh[name] = np.array([data.model['generator'][unit][field] for unit in UNIT_ORDER], dtype=np.float64)
        case = load_arrays(ROOT/'artifacts/lp-shop-full-001/case_2012_inputs.npz')
        literal_arrays(fresh, case)
        model, problem = capture_problem(data.network, case, 1.)
        scale = float(1+np.abs(case['price']).mean())
        assert np.isfinite(scale) and scale >= 1
        indices = model.state[[0, 5, 10]].ravel()
        previous = None
        for replica, expected in zip([1, 2], m['replicas'], strict=True):
            folder = ROOT/f'artifacts/integer-water-budget-full-four-worker-001/cases/case_2012/replica_{replica}_60s'
            proof = json.loads((folder/'proof.json').read_text())
            assert proof['status'] == 'failed' and 'Fixed-state LP worsened' in proof['error']
            assert proof['protocol_sha256'] == sha(protocol_path) and proof['proof']['mip_status'] == 0
            assert proof['proof']['mip_gap'] == 0 and proof['proof']['proven_optimal']
            assert proof['proof']['selected_units'] == ['Hogga_G1', 'Lio_G1', 'Haukeli_G1']
            for name, digest in proof['artifacts'].items():
                assert sha(folder/name) == digest
            native = load_arrays(folder/'failed_incumbent.npz')
            lp = load_arrays(folder/'fixed_lp_1.npz')
            literal_arrays(lp, load_arrays(folder/'fixed_lp_2.npz'))
            xn = native['integer_solution']
            assert np.isfinite(xn).all() and np.isin(native['integer_states'], [0, 1]).all()
            np.testing.assert_array_equal(np.rint(xn[indices]), native['integer_states'].ravel())
            integrality = float(np.max(np.abs(xn[indices]-native['integer_states'].ravel())))
            assert integrality <= 1e-5
            native_feasibility = {'equality': float(np.max(np.abs(problem['A_eq']@xn-problem['b_eq']))),
                                  'inequality': float(max(0, np.max(problem['A_ub']@xn-problem['b_ub']))),
                                  'bounds': float(max(0, np.max(problem['bounds'][:, 0]-xn), np.max(xn-problem['bounds'][:, 1])))}
            assert max(native_feasibility.values()) <= 1e-5
            fixed = {**problem, 'bounds': problem['bounds'].copy()}
            fixed['bounds'][indices, 0] = fixed['bounds'][indices, 1] = native['integer_states'].ravel()
            kkt = check_lp(fixed, {('x' if name == 'solution' else 'objective_scaled' if name == 'objective' else name): value for name, value in lp.items()})
            for name, value in kkt.items():
                np.testing.assert_allclose(value, expected['conditional_lp_kkt'][name], rtol=0, atol=1e-14)
            native_reported = proof['proof']['mip_objective_scaled']
            exact = independent_exact(fixed, xn, lp, native_reported, scale)
            certificate = expected['certificate']
            for name, value in exact['parts'].items():
                assert value == decode(certificate['parts_scaled'][name]), name
            for name, key in [('exact_difference', 'exact_dot_difference_scaled'), ('reported_difference', 'reported_difference_scaled'),
                              ('bound', 'bound_scaled'), ('euro_bound', 'bound_eur'), ('max_stationarity', 'maximum_exact_stationarity_residual')]:
                assert exact[name] == decode(certificate[key]), name
            assert exact['euro_bound'] <= Fraction(1, 100000) == decode(certificate['absolute_eur_budget'])
            assert exact['reported_difference'] > Fraction.from_float(1e-6)
            for index in [model.base.low_slack, model.base.high_slack, model.base.bypass_slack, model.base.vest_slack]:
                assert np.max(np.abs(lp['solution'][index])) == 0
            if previous is not None:
                literal_arrays(previous[0], native)
                literal_arrays(previous[1], lp)
            previous = native, lp
            report['replicas'].append({'replica': replica, 'original_status': 'failed', 'all_rational_components_exact': True,
                                       'native_feasibility': native_feasibility, 'integrality_max_error': integrality,
                                       'conditional_LP_KKT': kkt, 'reported_worsening_eur': float(exact['reported_difference'])*scale,
                                       'certified_one_sided_upper_bound_eur': float(exact['euro_bound']),
                                       'native_exact_inequality_violation': float(exact['native_exact_ub_violation']),
                                       'native_solver_reported_bound_minus_incumbent_scaled': proof['proof']['mip_dual_bound_scaled']-native_reported})
        command = [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_exact_integer_objective_bound.py']
        test = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        (out/'tests.txt').write_text(test.stdout+test.stderr)
        assert test.returncode == 0, test.stderr
        report.update(status='passed', certificate_manifest_sha256=sha(mp), certificate_source_sha256=sha(source),
                      exact_arithmetic_verified=True, focused_tests_passed=10, random_exact_identity_trials=128,
                      test_command=command, tests_sha256=sha(ROOT/'tests/test_exact_integer_objective_bound.py'),
                      limitations='One-sided upper bound on reported conditional-LP minus reported native-incumbent objective for stored binary64 coefficients, finite correctly shaped operands and positive scale. Not exact original decimal physics, absolute objective distance, or an independent global MILP proof. Native integrality/primal feasibility remain tolerance-based. API assumes positive scale/full dimensions; independently asserted for actualcase. Original failed eligibility unchanged.')
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        shutil.copy2(__file__, out/Path(__file__).name)
        shutil.copy2(ROOT/'tests/test_exact_integer_objective_bound.py', out/'test_exact_integer_objective_bound.py')
        report['runtime_seconds'] = time.monotonic()-started
        report['helpers'] = {name: sha(ROOT/name) for name in ['scripts/audit_hyljelihyl_river_cost_v2.py', 'scripts/audit_hyljelihyl_river_results.py', 'sintef_project/lp_hyljelihyl_v2.py']}
        report['artifacts'] = {q.name: sha(q) for q in out.iterdir() if q.is_file() and q.name != 'manifest.json'}
        (out/'manifest.json').write_text(json.dumps(report, indent=2))
        print(json.dumps({'status': report['status'], 'manifest_sha256': sha(out/'manifest.json'), 'runtime_seconds': report['runtime_seconds']}), flush=True)


if __name__ == '__main__':
    main()
