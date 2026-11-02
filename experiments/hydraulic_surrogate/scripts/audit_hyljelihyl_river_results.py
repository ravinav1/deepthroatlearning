"""Independently verify saved three-arm river solutions and descriptive metrics."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
from scripts.audit_hyljelihyl_river_cost_v2 import projection_matrix
from scripts.diagnose_hyljelihyl_river_cost_v2 import load_network, signatures
from sintef_project.lp_hyljelihyl_v2 import make_problem
from sintef_project.lp_shop_relaxation import effective_energy
from sintef_project.lp_features import UNIT_ORDER
from scipy.stats import rankdata

PROTOCOL_SHA = 'ba460ccb6085f9b032ea0647e9d19879801d0095ae5eece240ffbbdc33ab91f0'
PREFLIGHT_SHA = 'd34d49f5cf0eeb07fc84fbf8f164c9babf5519cc66729ba2abec7bf1bd31ff23'
REVIEW_SHA = '6268f5155b40869074ae3e673bef2c10e973c13d0e5d03302fd6c3c8b879e203'


def check_lp(problem: dict, saved: dict) -> dict:
    """Recompute primal/dual equations from matrices, never calling an optimizer."""
    c, ae, be, au, bu, bounds = [problem[k] for k in ['c', 'A_eq', 'b_eq', 'A_ub', 'b_ub', 'bounds']]
    x = saved['x']
    require(np.isfinite(x).all(), 'Nonfinite primal')
    residuals = {'eqlin_residual': be - ae @ x, 'ineqlin_residual': bu - au @ x,
                 'lower_residual': x - bounds[:, 0], 'upper_residual': bounds[:, 1] - x}
    for key, value in residuals.items():
        np.testing.assert_allclose(saved[key], value, atol=1e-7, rtol=1e-8, err_msg=key)
    for name in ['eqlin', 'ineqlin', 'lower', 'upper']:
        require(np.isfinite(saved[name + '_marginals']).all(), 'Nonfinite dual')
    ye, yu, yl, yh = [saved[k + '_marginals'] for k in ['eqlin', 'ineqlin', 'lower', 'upper']]
    finite = np.isfinite(bounds[:, 1])
    require(np.max(np.abs(yh[~finite]), initial=0) <= 1e-7, 'Nonzero infinite-bound marginal')
    checks = {
        'equality': float(np.max(np.abs(residuals['eqlin_residual']))),
        'inequality': float(max(0, -residuals['ineqlin_residual'].min())),
        'bounds': float(max(0, -residuals['lower_residual'].min(), -residuals['upper_residual'].min())),
        'stationarity': float(np.max(np.abs(c - ae.T @ ye - au.T @ yu - yl - yh))),
        'dual_sign': float(max(0, yu.max(), -yl.min(), yh.max())),
        'complementarity': float(max(np.max(np.abs(yu * residuals['ineqlin_residual'])),
                                     np.max(np.abs(yl * residuals['lower_residual'])),
                                     np.max(np.abs(yh[finite] * residuals['upper_residual'][finite])))),
        'scaled_primal_dual_gap': float(abs(c @ x - (be @ ye + bu @ yu + bounds[:, 0] @ yl + bounds[finite, 1] @ yh[finite]))),
    }
    require(np.isfinite(list(checks.values())).all() and max(checks.values()) <= 1e-5, 'LP certificate failed')
    np.testing.assert_allclose(c @ x, saved['objective_scaled'], atol=1e-5, rtol=2e-11)
    return checks


def auc(truth: np.ndarray, values: np.ndarray) -> float | None:
    """Compute tie-corrected Mann–Whitney AUC independently of sklearn."""
    y, score = np.asarray(truth).ravel().astype(bool), np.asarray(values).ravel()
    positive, negative = int(y.sum()), int((~y).sum())
    if not positive or not negative:
        return None
    return float((rankdata(score, method='average')[y].sum() - positive * (positive + 1) / 2)
                 / (positive * negative))


def independent_metrics(truth: np.ndarray, q: np.ndarray, power: np.ndarray, blocks: list) -> dict:
    """Recompute frozen ranking, threshold and switching diagnostics."""
    qp, pp = np.maximum(q, 0), np.maximum(power, 0)
    qp, pp = qp / (1 + qp), pp / (1 + pp)
    hard = power >= .5
    weak = [UNIT_ORDER.index(u) for u in ['Hogga_G1', 'Lio_G1', 'Haukeli_G1']]
    affected = [7, 8, 9, 12, 13]
    result = {
        'micro_auc_qratio': auc(truth, qp), 'micro_auc_pratio': auc(truth, pp),
        'weak3_micro_auc_qratio': auc(truth[:, weak], qp[:, weak]),
        'power_cutoff_hour_errors': int(np.count_nonzero(hard != truth)),
        'power_cutoff_error_fraction': float(np.mean(hard != truth)),
        'mean_absolute_case_unit_onfraction_error': float(np.abs(hard.mean(-1) - truth.mean(-1)).mean()),
        'mean_absolute_case_unit_switch_count_error': float(np.abs(np.count_nonzero(np.diff(hard.astype(int), axis=-1), axis=-1)
                                                                - np.count_nonzero(np.diff(truth.astype(int), axis=-1), axis=-1)).mean()),
        'per_unit': {}, 'per_date_block': {},
        'affected_five_q_auc': auc(truth[:, affected], q[:, affected] / (1 + q[:, affected])),
        'affected_five_cutoff_errors': int(np.count_nonzero(hard[:, affected] != truth[:, affected])),
    }
    for index, unit in enumerate(UNIT_ORDER):
        result['per_unit'][unit] = {'auc_qratio': auc(truth[:, index], qp[:, index]),
                                   'auc_pratio': auc(truth[:, index], pp[:, index]),
                                   'hour_errors': int(np.count_nonzero(hard[:, index] != truth[:, index]))}
    for block in dict.fromkeys(blocks):
        take = np.asarray(blocks) == block
        result['per_date_block'][block] = {'auc_qratio': auc(truth[take], qp[take]),
                                         'hour_errors': int(np.count_nonzero(hard[take] != truth[take]))}
    return result


def compare_nested(actual, expected) -> None:
    """Compare all reported metrics, including degenerate-unit null AUCs."""
    if isinstance(expected, dict):
        require(set(actual) == set(expected), 'Metric fields differ')
        for key in expected:
            compare_nested(actual[key], expected[key])
    elif expected is None:
        require(actual is None, 'Degenerate AUC differs')
    else:
        np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=0)


def main() -> None:
    """Check all physical evidence before independently reading the twelve label rows."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    require(Path(args.run_id).name == args.run_id and args.run_id not in ['', '.', '..'], 'Invalid run ID')
    run = args.run if args.run.is_absolute() else ROOT / args.run
    out = ROOT / 'artifacts' / args.run_id
    out.mkdir(exist_ok=False)
    start = time.monotonic()
    report = {'status': 'initializing', 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'command': shlex.join([sys.executable, *sys.argv]), 'optimization_calls': 0,
              'labels_decoded': False, 'cases': [], 'automatic_promotion': False}
    try:
        require(digest(run / 'manifest.json') == args.manifest_sha256, 'Actual run hash differs')
        require(digest(run / 'protocol.json') == PROTOCOL_SHA, 'Protocol differs')
        m = json.loads((run / 'manifest.json').read_text())
        p = json.loads((run / 'protocol.json').read_text())
        require(m['status'] == 'complete' and m['labels_decoded'] and len(m['cases']) == 12,
                'Require a fully completed diagnostic')
        require(m['independent_audit_sha256'] == REVIEW_SHA, 'Independent preflight differs')
        for group in ['source', 'data', 'inputs']:
            verify_hashes(ROOT, p[group])
        verify_hashes(run, m['artifacts'])
        pre = ROOT / 'artifacts/hyljelihyl-river-cost-v2-preflight-001'
        require(digest(pre / 'manifest.json') == PREFLIGHT_SHA, 'No-solve input seal differs')
        pre_m = json.loads((pre / 'manifest.json').read_text())
        verify_hashes(pre, pre_m['artifacts'])
        review = ROOT / 'artifacts/hyljelihyl-river-cost-v2-independent-preflight-002/manifest.json'
        require(digest(review) == REVIEW_SHA, 'Independent input/algebra proof differs')
        require(digest(run / 'physical_lock.json') == m['physical_lock_sha256'], 'Physical seal changed')
        seal = json.loads((run / 'physical_lock.json').read_text())
        require(seal['status'] == 'passed' and seal['case_count'] == 12 and seal['cases'] == m['cases']
                and seal['protocol_sha256'] == PROTOCOL_SHA, 'Physical seal contents differ')
        require(datetime.fromisoformat(seal['timestamp_utc']) <= datetime.fromisoformat(m['labels_opened_utc']),
                'Labels opened before physical seal')
        results_by_arm = {arm: [] for arm in p['arms']}
        original_cases = []
        worker_pids = []
        for selected, recorded, pre_case in zip(p['cases'], m['cases'], pre_m['cases']):
            cid = selected['case_id']
            require(cid == recorded['case_id'] == pre_case['case_id'], 'Case order differs')
            require(pd.Timestamp(selected['date']) + pd.Timedelta(days=7) <= pd.Timestamp('2021-01-01', tz='UTC'), 'Late labels forbidden')
            folder = run / f'case_{cid:04d}'
            original = arrays(folder / 'A_inputs.npz')
            split = arrays(folder / 'split_inputs.npz')
            same(original, arrays(pre / folder.name / 'A_inputs.npz'))
            same(split, arrays(pre / folder.name / 'split_inputs.npz'))
            nets = {}
            for name in ['A', 'split']:
                require((folder / f'{name}_network.json').read_bytes() == (pre / folder.name / f'{name}_network.json').read_bytes(), 'Topology differs')
                nets[name] = load_network(json.loads((folder / f'{name}_network.json').read_text()))
            old_model, pa, _ = make_problem(nets['A'], original, 'A')
            case_summary = {'case_id': cid, 'arms': {}}
            for arm in p['arms']:
                case = original if arm == 'A' else split
                model, problem, segments = make_problem(nets['A'] if arm == 'A' else nets['split'], case, arm)
                require(signatures(problem) == recorded['matrix_signatures'][arm] == pre_case['matrix_signatures'][arm], 'Frozen matrix differs')
                both = []
                for replica in [1, 2]:
                    work = folder / f'{arm}_{replica}'
                    proof = json.loads((work / 'proof.json').read_text())
                    job = json.loads((work / 'job.json').read_text())
                    require(digest(work / 'proof.json') == recorded['arms'][arm]['worker_proof_sha256'][replica-1], 'Worker seal differs')
                    require(proof['status'] == 'complete' and proof['native']['success'] and proof['native']['status'] == 0
                            and proof['all_slacks_zero'] and 0 < proof['peak_rss_bytes'] < p['peak_rss_limit_bytes'], 'Worker acceptance failed')
                    require(proof['environment'] == p['environment'] and proof['case_id'] == cid
                            and proof['arm'] == arm and proof['replica'] == replica, 'Worker identity differs')
                    require(digest(work / 'job.json') == proof['job_sha256'] and job['protocol_sha256'] == PROTOCOL_SHA,
                            'Job identity differs')
                    for field in ['input', 'network']:
                        require(digest(Path(job[field])) == job[field + '_sha256'], 'Worker input differs')
                    require(job['resources']['free_percent'] >= 30 and
                            job['resources']['swapouts'] <= m['initial_resources']['swapouts'], 'Resource gate differs')
                    verify_hashes(work, proof['artifacts'])
                    saved = arrays(work / 'primal_dual.npz')
                    same(saved, arrays(work / 'native_output.npz'))
                    certificate = check_lp(problem, saved)
                    x, base = saved['x'], model.base
                    result = arrays(work / 'result.npz')
                    for key in ['minimum_slack', 'maximum_slack', 'bypass_slack', 'vest_slack']:
                        require(np.isfinite(result[key]).all() and np.max(np.abs(result[key])) == 0, 'Nonzero physical slack')
                    scale = 1 + np.abs(case['price']).mean()
                    expected = {'flow': x[base.q].sum(axis=1),
                                'power': (x[base.q] * effective_energy(case, 168)).sum(axis=1),
                                'state': x[model.state], 'raw_start': x[model.start], 'raw_stop': x[model.stop],
                                'storage': x[base.v], 'bypass': x[base.bypass], 'spill': x[base.spill],
                                'water_shadow': -saved['eqlin_marginals'][:len(model.network.pools)*168].reshape(-1,168)*scale,
                                'objective': -float(saved['objective_scaled'])*scale}
                    for key, value in expected.items():
                        np.testing.assert_allclose(result[key], value, atol=1e-5, rtol=2e-11, err_msg=key)
                    if arm != 'A':
                        projected = projection_matrix(old_model, model) @ x[:model.n_variables]
                        violation = max(float(np.max(np.abs(pa['A_eq'] @ projected - pa['b_eq']))),
                                        float(np.max(pa['A_ub'] @ projected - pa['b_ub'])),
                                        float(np.max(pa['bounds'][:,0] - projected)),
                                        float(np.max(projected - pa['bounds'][:,1])))
                        require(violation <= 1e-5, 'Projected old constraints fail')
                        np.testing.assert_array_equal(result['river_volume'], x[base.bypass[11]])
                    if arm == 'C':
                        np.testing.assert_array_equal(result['river_segment_volume'], x[segments])
                        np.testing.assert_allclose(x[segments].sum(0), result['river_volume'], atol=1e-7, rtol=1e-10)
                        river_cost = float(np.sum(x[segments] * np.asarray(p['marginal_costs'])[:,None] / .0036))
                        np.testing.assert_allclose(result['river_cost'], river_cost, atol=1e-5, rtol=2e-11)
                    worker_pids.append(proof['pid'])
                    both.append((result, saved))
                same(both[0][0], both[1][0]); same(both[0][1], both[1][1])
                result = both[0][0]
                np.testing.assert_allclose(result['objective'], recorded['arms'][arm]['objective'], rtol=0, atol=0)
                if arm == 'A':
                    same(result, arrays(ROOT / p['cache'] / f'case_{cid:04d}_result.npz'))
                results_by_arm[arm].append(result)
                summary = {'objective': float(result['objective']), 'certificate': certificate}
                if arm != 'A':
                    summary.update(river_volume_mm3=float(result['river_volume'].sum()),
                                   river_hours_above40=int(np.count_nonzero(result['river_volume'] > 40*.0036 + 1e-9)),
                                   river_cost=float(result['river_cost']))
                case_summary['arms'][arm] = summary
            values = [case_summary['arms'][arm]['objective'] for arm in p['arms']]
            require(values[0] + 1e-4 >= values[1] and values[1] + 1e-4 >= values[2], 'Objective ordering fails')
            report['cases'].append(case_summary)
            original_cases.append(original)
        require(len(worker_pids) == 72 and len(set(worker_pids)) == 72, 'Fresh-worker process count differs')
        path = ROOT / 'sintef_project/data/kernel/Unit_commitment_decisions.csv'
        ids = [r['case_id'] for r in p['cases']]
        metadata = pd.read_csv(path, usecols=['Run No', 'starttime'])
        keep = metadata['Run No'].isin(ids).to_numpy()
        require(metadata.loc[keep, 'Run No'].tolist() == ids, 'Target row order differs')
        require((pd.to_datetime(metadata.loc[keep, 'starttime'], utc=True) + pd.Timedelta(days=7)
                 <= pd.Timestamp('2021-01-01', tz='UTC')).all(), 'Target date scope differs')
        columns = [f'result_committed_{u}_t{h}' for u in UNIT_ORDER for h in range(168)]
        frame = pd.read_csv(path, skiprows=(np.flatnonzero(~keep)+1).tolist(),
                            usecols=['Run No', *columns], dtype={c: 'uint8' for c in columns})
        require(frame['Run No'].tolist() == ids, 'Decoded IDs differ')
        truth = frame[columns].to_numpy().reshape(12,14,168)
        require(np.isin(truth, [0,1]).all(), 'Invalid targets')
        diagnostic = arrays(run / 'diagnostic.npz')
        np.testing.assert_array_equal(truth, diagnostic['truth'])
        report['labels_decoded'] = True
        report['metrics'] = {}
        for arm, results in results_by_arm.items():
            q = np.stack([r['flow']/case['qmin'][:,None] for r,case in zip(results, original_cases)])
            power = np.stack([r['power']/case['pmin'][:,None] for r,case in zip(results, original_cases)])
            np.testing.assert_array_equal(q, diagnostic[arm+'_qratio'])
            np.testing.assert_array_equal(power, diagnostic[arm+'_pratio'])
            measured = independent_metrics(truth, q, power, p['blocks'])
            compare_nested(m['metrics'][arm], measured)
            report['metrics'][arm] = measured
        report.update(status='passed', result_manifest_sha256=args.manifest_sha256,
                      protocol_sha256=PROTOCOL_SHA, all_72_physical_outputs_replayed_and_verified=True,
                      case_count=12, caveat='Reused pre2021 descriptive physics diagnostic, not validation or a submission. No automatic cache/model promotion.')
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        shutil.copy2(__file__, out / Path(__file__).name)
        report.update(runtime_seconds=time.monotonic()-start, source_sha256=digest(Path(__file__)),
                      environment={'python': sys.version, 'numpy': np.__version__, 'pandas': pd.__version__})
        (out / 'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
        with (ROOT / 'docs/experiments.md').open('a') as stream:
            stream.write(f'\n\n## {args.run_id} — {report["timestamp_utc"]}\n\nActual river result audit {report["status"]}; source/command/input/result/environment and all failure evidence in `artifacts/{args.run_id}/manifest.json`. Exact command `{report["command"]}`. No optimization/model fit; only twelve pre2021 target rows after full physical verification. Runtime {report["runtime_seconds"]:.3f}s. No leaderboard claim or automatic promotion.\n')
        print(json.dumps({'status': report['status'], 'manifest_sha256': digest(out/'manifest.json'), 'cases': len(report['cases'])}))


if __name__ == '__main__':
    main()
