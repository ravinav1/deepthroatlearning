"""Read-only exact-rational consistency certificate for failed integer case2012.

This explains the stored binary64 optimization problem. It does not change the
original failed status, authorize a retry, or independently prove MILP optimality.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import json
from pathlib import Path
import shlex
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_integer_water_budget_cache import (
    arrays, digest, require, same, verify_hashes, np, pd,
)
from sintef_project.lp_features import HydroCaseData, UNIT_ORDER
from sintef_project.lp_hyljelihyl_v2 import capture_problem
from scripts.audit_hyljelihyl_river_results import check_lp

CASE_ID = 2012
DIAGNOSIS_SHA = '44ae344854d7a58aa7f9e6dd8f5225e1f3b0ec4d1094db05e4f10c9b77eeb105'
ABSOLUTE_EUR_BUDGET = Fraction(1, 100000)


def exact_bound(problem: dict, native: np.ndarray, lp: dict,
                native_reported: float, scale: float) -> dict:
    """Bound reported LP-minus-native objective using exact binary64 rationals.

Every sum, dot product and residual below uses Fraction; no floating arithmetic
allowance is needed. The bound includes imperfect LP complementarity, marginal
signs, stationarity, both objective summation errors and infinite upper bounds.
    """
    cache = {0.: Fraction(0), 1.: Fraction(1), -1.: Fraction(-1)}

    def f(value):
        value = float(value)
        require(np.isfinite(value), 'Nonfinite exact-certificate operand')
        if value not in cache:
            cache[value] = Fraction.from_float(value)
        return cache[value]

    def vector(value):
        return [f(item) for item in value]

    def dot(a, b):
        require(len(a) == len(b), 'Exact dot dimensions differ')
        return sum((x*y for x,y in zip(a,b)), Fraction(0))

    c, xn, xl = vector(problem['c']), vector(native), vector(lp['solution'])
    difference = [a-b for a,b in zip(xl,xn)]
    stationarity = c.copy()
    parts = {name: Fraction(0) for name in [
        'equality_residuals', 'fixed_inequality_complementarity',
        'native_inequality_violations', 'wrong_sign_inequality_terms',
        'fixed_lower_complementarity', 'native_lower_violations', 'wrong_sign_lower_terms',
        'fixed_upper_complementarity', 'native_upper_violations', 'wrong_sign_upper_terms',
        'infinite_upper_terms', 'stationarity_terms', 'reported_lp_dot_discrepancy',
        'reported_native_dot_discrepancy']}
    algebra_terms = Fraction(0)
    for matrix_name, rhs_name, dual_name, equality in [
            ('A_eq', 'b_eq', 'eqlin_marginals', True),
            ('A_ub', 'b_ub', 'ineqlin_marginals', False)]:
        matrix = problem[matrix_name].tocoo()
        rhs, dual = vector(problem[rhs_name]), vector(lp[dual_name])
        rn, rl = [-v for v in rhs], [-v for v in rhs]
        for row, col, value in zip(matrix.row, matrix.col, matrix.data):
            a = f(value)
            rn[row] += a*xn[col]
            rl[row] += a*xl[col]
            stationarity[col] -= a*dual[row]
        for n, l, multiplier in zip(rn, rl, dual):
            algebra_terms += multiplier*(l-n)
            if equality:
                parts['equality_residuals'] += abs(multiplier)*(abs(n)+abs(l))
            else:
                parts['fixed_inequality_complementarity'] += abs(multiplier*l)
                parts['native_inequality_violations'] += max(-multiplier,0)*max(n,0)
                parts['wrong_sign_inequality_terms'] += max(multiplier,0)*max(-n,0)
    bounds = problem['bounds']
    lower, upper = vector(lp['lower_marginals']), vector(lp['upper_marginals'])
    for j, (a, b) in enumerate(zip(lower, upper)):
        stationarity[j] -= a+b
        algebra_terms += (a+b)*difference[j]
        lower_bound = f(bounds[j,0])
        parts['fixed_lower_complementarity'] += abs(a*(xl[j]-lower_bound))
        parts['native_lower_violations'] += max(a,0)*max(lower_bound-xn[j],0)
        parts['wrong_sign_lower_terms'] += max(-a,0)*max(xn[j]-lower_bound,0)
        if np.isfinite(bounds[j,1]):
            upper_bound = f(bounds[j,1])
            parts['fixed_upper_complementarity'] += abs(b*(xl[j]-upper_bound))
            parts['native_upper_violations'] += max(-b,0)*max(xn[j]-upper_bound,0)
            parts['wrong_sign_upper_terms'] += max(b,0)*max(upper_bound-xn[j],0)
        else:
            require(np.isposinf(bounds[j,1]), 'Unexpected nonfinite upper bound')
            parts['infinite_upper_terms'] += abs(b*difference[j])
    stationarity_dot = dot(stationarity,difference)
    parts['stationarity_terms'] = dot([abs(r) for r in stationarity], [abs(d) for d in difference])
    true_lp, true_native = dot(c,xl), dot(c,xn)
    exact_difference = true_lp-true_native
    require(algebra_terms+stationarity_dot == exact_difference, 'Exact stationarity identity failed')
    reported_lp, reported_native = f(lp['objective']), f(native_reported)
    parts['reported_lp_dot_discrepancy'] = abs(reported_lp-true_lp)
    parts['reported_native_dot_discrepancy'] = abs(reported_native-true_native)
    bound = sum(parts.values(),Fraction(0))
    reported_difference = reported_lp-reported_native
    require(reported_difference <= bound, 'Exact reported-objective bound failed')
    euro_bound = bound*f(scale)

    def encoded(value):
        return {'numerator': str(value.numerator), 'denominator': str(value.denominator),
                'approximate_decimal': float(value)}

    return {'exact_stationarity_identity': True, 'reported_difference_bounded': True,
            'parts_scaled': {key: encoded(value) for key,value in parts.items()},
            'exact_dot_difference_scaled': encoded(exact_difference),
            'reported_difference_scaled': encoded(reported_difference),
            'bound_scaled': encoded(bound), 'bound_eur': encoded(euro_bound),
            'absolute_eur_budget': encoded(ABSOLUTE_EUR_BUDGET),
            'within_existing_absolute_objective_budget': euro_bound <= ABSOLUTE_EUR_BUDGET,
            'maximum_exact_stationarity_residual': encoded(max(map(abs,stationarity)))}


def main() -> None:
    """Certify the existing failure evidence without solving or changing eligibility."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    require(Path(args.run_id).name == args.run_id and args.run_id not in ['', '.', '..'], 'Invalid run ID')
    out = ROOT/'artifacts'/args.run_id
    out.mkdir(exist_ok=False)
    start = time.monotonic()
    report = {'status': 'initializing', 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'command': shlex.join([sys.executable,*sys.argv]), 'case_id': CASE_ID,
              'optimization_calls': 0, 'labels_decoded': False, 'eligibility_changed': False,
              'retry_authorized': False, 'replicas': []}
    try:
        diagnosis_path = ROOT/'artifacts/integer-case2012-numerical-diagnostic-001/manifest.json'
        require(digest(diagnosis_path) == DIAGNOSIS_SHA, 'Pinned initial diagnosis differs')
        diagnosis = json.loads(diagnosis_path.read_text())
        from unittest.mock import patch
        target = (ROOT/'sintef_project/data/kernel/Unit_commitment_decisions.csv').resolve()
        old_read = pd.read_csv

        def metadata(path,*a,**kw):
            if isinstance(path,(str,Path)) and Path(path).resolve() == target:
                require(not a and kw == {'usecols':['Run No','starttime']}, 'Labels forbidden')
            return old_read(path,*a,**kw)

        with patch.object(pd,'read_csv',metadata):
            data = HydroCaseData(ROOT)
        ip = ROOT/f'artifacts/lp-shop-full-001/case_{CASE_ID:04d}_inputs.npz'
        require(digest(ip) == diagnosis['input_sha256'], 'Physical input changed')
        case = arrays(ip)
        require(int(data.cases.iloc[CASE_ID-1]['Run No']) == CASE_ID, 'Metadata position changed')
        fresh = data.case(CASE_ID-1)
        fresh['natural'] = fresh['natural'].astype(np.float64)
        for key,field in [('pmin','p_min'),('start_cost','startcost_const'),('stop_cost','stopcost_const')]:
            fresh[key] = np.asarray([data.model['generator'][u][field] for u in UNIT_ORDER],dtype=np.float64)
        same(case,fresh)
        model,problem = capture_problem(data.network,case,1.)
        selected = model.state[[UNIT_ORDER.index(u) for u in ['Hogga_G1','Lio_G1','Haukeli_G1']]].ravel()
        scale = 1+np.abs(case['price']).mean()
        all_native,all_lp = [],[]
        inputs = {str(ip.relative_to(ROOT)): digest(ip), str(diagnosis_path.relative_to(ROOT)): DIAGNOSIS_SHA}
        for replica in [1,2]:
            folder = ROOT/f'artifacts/integer-water-budget-full-four-worker-001/cases/case_{CASE_ID:04d}/replica_{replica}_60s'
            receipt = json.loads((folder/'proof.json').read_text())
            require(digest(folder/'proof.json') == diagnosis['replicas'][replica-1]['proof_sha256'], 'Failed proof changed')
            require(receipt['status'] == 'failed' and receipt['proof']['mip_status'] == 0
                    and 'Fixed-state LP worsened' in receipt['error'], 'Unexpected exception')
            verify_hashes(folder,receipt['artifacts'])
            for name in ['proof.json',*receipt['artifacts']]:
                inputs[str((folder/name).relative_to(ROOT))] = digest(folder/name)
            native,lp = arrays(folder/'failed_incumbent.npz'),arrays(folder/'fixed_lp_1.npz')
            same(lp,arrays(folder/'fixed_lp_2.npz'))
            fixed = {**problem,'bounds':problem['bounds'].copy()}
            fixed['bounds'][selected,0] = fixed['bounds'][selected,1] = native['integer_states'].ravel()
            lp_fields = {('x' if key=='solution' else 'objective_scaled' if key=='objective' else key):value for key,value in lp.items()}
            kkt = check_lp(fixed,lp_fields)
            certificate = exact_bound(fixed,native['integer_solution'],lp,
                                      receipt['proof']['mip_objective_scaled'],scale)
            require(certificate['within_existing_absolute_objective_budget'], 'Error bound exceeds fixed absolute EUR budget')
            for name,index in [('minimum',model.base.low_slack),('maximum',model.base.high_slack),
                               ('bypass',model.base.bypass_slack),('vest',model.base.vest_slack)]:
                require(np.max(np.abs(lp['solution'][index])) == 0, 'Nonzero physical slack: '+name)
            all_native.append(native);all_lp.append(lp)
            report['replicas'].append({'replica':replica,'original_status':'failed','conditional_lp_kkt':kkt,
                                       'certificate':certificate})
        same(all_native[0],all_native[1]);same(all_lp[0],all_lp[1])
        report.update(status='passed_read_only_certificate',inputs=inputs,
                      caveat='Exact rational certificate for stored binary64 LP coefficients/primal/duals/objectives. Not exact original decimal physics or an independent global MILP proof. Original v2 failures and ineligibility remain unchanged; a separately frozen/reviewed fresh completion is still required.')
    except BaseException:
        report.update(status='failed',error=traceback.format_exc());raise
    finally:
        shutil.copy2(__file__,out/Path(__file__).name)
        report.update(runtime_seconds=time.monotonic()-start,source_sha256=digest(Path(__file__)),
                      environment={'python':sys.version,'numpy':np.__version__})
        (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
        with (ROOT/'docs/experiments.md').open('a') as stream:
            stream.write(f'\n\n## {args.run_id} — {report["timestamp_utc"]}\n\nRead-only exact-rational numerical certificate status {report["status"]}. No solve, target decoding, retry or eligibility change. Command `{report["command"]}`. Exact operands/input hashes/rational components/source/environment and failure evidence in `artifacts/{args.run_id}/manifest.json`. Runtime {report["runtime_seconds"]:.3f}s. Original case2012 failures remain failed under v2.\n')
        print(json.dumps({'status':report['status'],'manifest_sha256':digest(out/'manifest.json'),
                          'bound_eur':report['replicas'][0]['certificate']['bound_eur']['approximate_decimal'] if report['replicas'] else None}))


if __name__ == '__main__':
    main()
