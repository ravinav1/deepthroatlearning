"""Verify the actual fresh one-case completion without optimization or target reads.""" 
from __future__ import annotations

from datetime import datetime, timezone
from fractions import Fraction
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_integer_water_budget_cache import arrays, digest, same, verify_hashes, np, pd
from scripts.audit_integer_exact_certificate import independent_exact, decode
from scripts.audit_hyljelihyl_river_results import check_lp
from sintef_project.lp_hyljelihyl_v2 import capture_problem
from sintef_project.lp_features import HydroCaseData, UNIT_ORDER


def decoded(model, problem: dict, network, case: dict, native: dict, lp: dict, iterations: int) -> dict:
    """Independently decode the original nine-pool physical outputs."""
    base, x = model.base, lp['solution']
    scale = 1 + np.abs(case['price']).mean()
    energy = np.repeat(case['energy'][:, :, None], 168, axis=2)
    mean_energy = (case['energy'] * case['widths']).sum(axis=1) / case['widths'].sum(axis=1)
    energy[:, :, case['price'] < 0] = mean_energy[:, None, None]
    q = x[base.q]
    flow, power, state = q.sum(axis=1), (q * energy).sum(axis=1), x[model.state]
    delta = np.diff(state, axis=1)
    transition = float(np.sum(np.maximum(delta, 0) * case['start_cost'][:, None])
                       + np.sum(np.maximum(-delta, 0) * case['stop_cost'][:, None]))
    penalty = case.get('slack_penalty', 1e7 + 100*np.max(np.abs(case['terminal_value'])))
    objective = -float(lp['objective']) * scale
    reconstructed = float(np.sum(power * case['price'][None, :]) + case['terminal_value'] @ x[base.v][:, -1]
        - case.get('bypass_cost', 10/.0036) * x[base.bypass].sum()
        - case.get('spill_cost', 1000/.0036) * x[base.spill].sum()
        - penalty * sum(x[j].sum() for j in [base.low_slack, base.high_slack, base.bypass_slack, base.vest_slack]) - transition)
    sea = flow[network.unit_down < 0].sum() + x[base.bypass][network.bypass_down < 0].sum() + x[base.spill][network.spill_down < 0].sum()
    bounds = problem['bounds']
    return {'flow': flow, 'power': power, 'state': state, 'raw_start': x[model.start], 'raw_stop': x[model.stop],
        'state_lower': np.maximum(flow/case['qmax'][:, None], power/case['pmax'][:, None]),
        'state_upper': np.minimum(np.minimum(flow/case['qmin'][:, None], power/case['pmin'][:, None]), 1.),
        'segment_flow': q, 'storage': x[base.v], 'bypass': x[base.bypass], 'spill': x[base.spill],
        'water_shadow': -lp['eqlin_marginals'].reshape(len(network.pools), 168)*scale,
        'minimum_slack': x[base.low_slack], 'maximum_slack': x[base.high_slack],
        'bypass_slack': x[base.bypass_slack], 'vest_slack': x[base.vest_slack],
        'objective': objective, 'reconstructed_objective': reconstructed,
        'transition_cost': transition, 'gross_objective': objective + transition,
        'balance_residual': float(np.max(np.abs(problem['A_eq'] @ x - problem['b_eq']))),
        'inequality_residual': float(max(0., np.max(problem['A_ub'] @ x - problem['b_ub']))),
        'bound_residual': float(max(0., np.max(bounds[:, 0] - x), np.max(x - bounds[:, 1]))),
        'total_balance_error': float(x[base.v][:, -1].sum() + sea - np.sum(case['initial'], dtype=np.float64) - np.sum(case['natural'], dtype=np.float64)),
        'iterations': iterations, 'status': 0, 'negative_price_hours': int(np.sum(case['price'] < 0)), **native}


def main() -> None:
    """Audit saved actual worker evidence and newly decoded result, preserving failures."""
    out = ROOT / 'artifacts/integer-case2012-completion-result-independent-audit-001'
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    report = {'status': 'initializing', 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'optimization_calls': 0, 'labels_decoded': False, 'old_v2_eligible': False,
              'aggregate_eligible': False, 'replicas': []}
    try:
        directory = ROOT / 'artifacts/integer-case2012-completion-001'
        expected_sha = '7da83d48119d48a0826a6d3edf01a265ee932edb8ec7f899be6dc2bb9dce8984'
        assert digest(directory / 'manifest.json') == expected_sha
        m = json.loads((directory / 'manifest.json').read_text())
        assert m['status'] == 'complete_case2012_exception' and m['case_id'] == 2012
        assert m['acceptance_basis'] == 'case2012_dual_feasibility_certificate'
        assert not m['old_v2_eligible'] and not m['aggregate_eligible']
        assert m['original_status_preserved'] == 'failed'
        assert digest(directory / 'protocol.json') == m['protocol_sha256'] == '6499a285d267e8ccab0cb2063e682880c8837423fdd8606d9c83d11063cff989'
        p = json.loads((directory / 'protocol.json').read_text())
        for group in ['source', 'data', 'certificate_inputs']:
            for name, expected in p[group].items():
                assert digest(ROOT / name) == expected, name
        verify_hashes(directory, m['artifacts'])
        assert m['declared_optimization_calls'] == {'milp': 2, 'conditional_lp': 4}
        assert len(m['replicas']) == 2 and len({r['pid'] for r in m['replicas']}) == 2
        check = subprocess.run(['ps', '-p', ','.join(str(r['pid']) for r in m['replicas']), '-o', 'pid='], capture_output=True, text=True)
        assert check.returncode == 1 and not check.stdout.strip(), 'Actual workers still alive'
        (out / 'worker_exit_observation.txt').write_text(check.stdout + check.stderr + '\nps returned1, no matching PIDs.\n')
        target = (ROOT / 'sintef_project/data/kernel/Unit_commitment_decisions.csv').resolve()
        read_csv = pd.read_csv

        def metadata(path, *args, **kwargs):
            if isinstance(path, (str, Path)) and Path(path).resolve() == target:
                assert not args and kwargs == {'usecols': ['Run No', 'starttime']}
            return read_csv(path, *args, **kwargs)

        with patch.object(pd, 'read_csv', metadata):
            data = HydroCaseData(ROOT)
        case = arrays(ROOT / p['cache'] / 'case_2012_inputs.npz')
        control = arrays(ROOT / p['cache'] / 'case_2012_result.npz')
        model, problem = capture_problem(data.network, case, 1.)
        indices = model.state[[UNIT_ORDER.index(u) for u in p['targets']]].ravel()
        scale = float(1 + np.abs(case['price']).mean())
        result = arrays(directory / 'exception_case2012_result.npz')
        original_certificate = json.loads((ROOT / p['certificate']).read_text())
        for replica, observed in enumerate(m['replicas'], 1):
            folder = directory / 'cases/case_2012' / f'replica_{replica}_60s'
            record = json.loads((folder / 'proof.json').read_text())
            assert digest(folder / 'proof.json') == observed['fresh_failed_proof_sha256']
            assert record['pid'] == observed['pid'] and record['timestamp_utc'] >= m['timestamp_utc']
            assert record['protocol_sha256'] == m['protocol_sha256'] and record['status'] == 'failed'
            assert 'Fixed-state LP worsened the MILP incumbent objective' in record['error']
            native_proof = record['proof']
            assert native_proof['mip_status'] == 0 and native_proof['proven_optimal'] and native_proof['mip_gap'] == 0
            assert native_proof['mip_time_limit_seconds'] == 60 and native_proof['fixed_lp_time_limit_seconds_each'] == 1
            verify_hashes(folder, record['artifacts'])
            for old in p['original_attempts']:
                source = ROOT / old['directory']
                assert digest(source / 'proof.json') == old['proof_sha256']
                assert json.loads((source / 'proof.json').read_text())['status'] == 'failed'
                for name, sha in old['payloads'].items():
                    assert digest(folder / name) == digest(source / name) == sha
                    same(arrays(folder / name), arrays(source / name))
            native, lp = arrays(folder / 'failed_incumbent.npz'), arrays(folder / 'fixed_lp_1.npz')
            same(lp, arrays(folder / 'fixed_lp_2.npz'))
            fixed = {**problem, 'bounds': problem['bounds'].copy()}
            fixed['bounds'][indices, 0] = fixed['bounds'][indices, 1] = native['integer_states'].ravel()
            kkt = check_lp(fixed, {('x' if key == 'solution' else 'objective_scaled' if key == 'objective' else key): value for key, value in lp.items()})
            exact = independent_exact(fixed, native['integer_solution'], lp, native_proof['mip_objective_scaled'], scale)
            cert = observed['checks']['certificate']
            assert cert == original_certificate['replicas'][replica - 1]['certificate']
            for name, value in exact['parts'].items():
                assert value == decode(cert['parts_scaled'][name])
            assert exact['euro_bound'] == decode(cert['bound_eur']) <= Fraction(1, 100000)
            calls = observed['lp_call_metadata']
            assert calls == [{'iterations': 6577, 'status': 0}] * 2
            reconstruction = decoded(model, problem, data.network, case, native, lp, calls[0]['iterations'])
            assert set(result) == set(reconstruction)
            for key, value in reconstruction.items():
                np.testing.assert_allclose(result[key], value, atol=1e-5, rtol=2e-11, err_msg=key)
            for key in ['minimum_slack', 'maximum_slack', 'bypass_slack', 'vest_slack']:
                assert np.max(np.abs(result[key])) == 0
            assert result['objective'] <= control['objective'] + 1e-4
            report['replicas'].append({'pid': observed['pid'], 'native_and_lp_payloads_match_originals': True,
                'conditional_kkt': kkt, 'exact_bound_eur': float(exact['euro_bound']), 'all_decoded_fields_verified': True})
        report.update(status='passed', completion_manifest_sha256=expected_sha, protocol_sha256=m['protocol_sha256'],
            result_sha256=digest(directory / 'exception_case2012_result.npz'), all_workers_exited=True,
            original_failures_preserved=True, case2012_exception_independently_accepted=True,
            limitation='Only this separately declared case exception is accepted; original v2 and incomplete aggregate remain ineligible.')
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        shutil.copy2(__file__, out / Path(__file__).name)
        report.update(runtime_seconds=time.monotonic()-started, source_sha256=digest(Path(__file__)),
            helpers={name: digest(ROOT/name) for name in ['scripts/audit_integer_exact_certificate.py',
                'scripts/audit_integer_water_budget_cache.py', 'scripts/audit_hyljelihyl_river_results.py',
                'sintef_project/lp_hyljelihyl_v2.py']})
        (out / 'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
        with (ROOT/'docs/experiments.md').open('a') as stream:
            stream.write(f'\n\n## integer-case2012-completion-result-independent-audit-001 — {report["timestamp_utc"]}\n\n'
                f'Status {report["status"]}; actual fresh payload/native/KKT/exact-rational bound and independently decoded result audit. '
                f'Command `{sys.executable} scripts/audit_integer_case2012_result.py`; runtime {report["runtime_seconds"]:.3f}s. '
                'No optimizer or target reads. Old failures and aggregate ineligibility preserved; full evidence in manifest.\n')
        print(json.dumps({'status': report['status'], 'manifest_sha256': digest(out / 'manifest.json')}), flush=True)


if __name__ == '__main__':
    main()
