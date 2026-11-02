"""Audit completed persistent-worker physical/feature replay without solving or labels."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[key] = '1'
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_hyljelihyl_river_cost_v2 import sha, literal_arrays, load_arrays, projection_matrix
from scripts.audit_hyljelihyl_persistent_pilot import independent_features
from scripts.audit_hyljelihyl_river_results import check_lp
from scripts.diagnose_hyljelihyl_river_cost_v2 import load_network, signatures
from sintef_project.lp_hyljelihyl_v2 import make_problem


def decoded_result(model, problem, segments, saved, case):
    """Independently derive every decoded C result field from saved LP arrays."""
    base, x = model.base, saved['x']
    scale = 1+np.abs(case['price']).mean()
    q = x[base.q]
    flow = q.sum(axis=1)
    energy = np.repeat(np.asarray(case['energy'])[:, :, None], 168, axis=2)
    average = (case['energy']*case['widths']).sum(axis=1)/case['widths'].sum(axis=1)
    energy[:, :, case['price'] < 0] = average[:, None, None]
    power = (q*energy).sum(axis=1)
    lower = np.maximum(flow/case['qmax'][:, None], power/case['pmax'][:, None])
    upper = np.minimum(np.minimum(flow/case['qmin'][:, None], power/case['pmin'][:, None]), 1.)
    river_cost = float(np.sum(x[segments]*np.array([1., 65., 239., 564., 1107., 1758., 1758.])[:, None]/.0036))
    transition = float(np.sum(x[model.start]*case['start_cost'][:, None])+np.sum(x[model.stop]*case['stop_cost'][:, None]))
    penalty = case.get('slack_penalty', 1e7+100*np.max(np.abs(case['terminal_value'])))
    reconstructed = (np.sum(power*case['price'][None, :])+case['terminal_value']@x[base.v][:, -1]
                     -float(case.get('bypass_cost', 10/.0036))*x[base.bypass[:11]].sum()
                     -float(case.get('spill_cost', 1000/.0036))*x[base.spill].sum()
                     -penalty*sum(x[j].sum() for j in [base.low_slack, base.high_slack, base.bypass_slack, base.vest_slack])
                     -transition-river_cost)
    return {'flow': flow, 'power': power, 'state': x[model.state], 'raw_start': x[model.start], 'raw_stop': x[model.stop],
            'state_lower': lower, 'state_upper': upper, 'segment_flow': q, 'storage': x[base.v],
            'bypass': x[base.bypass], 'spill': x[base.spill],
            'water_shadow': -saved['eqlin_marginals'][:10*168].reshape(10, 168)*scale,
            'minimum_slack': x[base.low_slack], 'maximum_slack': x[base.high_slack],
            'bypass_slack': x[base.bypass_slack], 'vest_slack': x[base.vest_slack],
            'objective': -float(saved['objective_scaled'])*scale, 'status': 0,
            'river_volume': x[base.bypass[11]], 'river_segment_volume': x[segments],
            'river_cost': river_cost, 'reconstructed_objective': float(reconstructed), 'transition_cost': transition}


def main() -> None:
    """Verify all118 saved calls, two orders,36 canonical cases and62 feature bytes."""
    out = ROOT/'artifacts/hyljelihyl-river-persistent-result-independent-audit-002'
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    report = {'status': 'initializing', 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'command': [sys.executable, *sys.argv], 'optimization_calls': 0, 'labels_decoded': False,
              'training': False, 'model_cache_eligible': False, 'cases': []}
    try:
        run = ROOT/'artifacts/hyljelihyl-river-persistent-pilot-001'
        result_hash = '134ad85d2533e33ac218f4be70c4ef5d246cef37fb77a4656f6c4e855538dc49'
        assert sha(run/'manifest.json') == result_hash
        m = json.loads((run/'manifest.json').read_text())
        p = json.loads((run/'protocol.json').read_text())
        assert sha(run/'protocol.json') == m['protocol_sha256'] == 'a1d860345862538c22498ca675d25e8be6730fda6051255e5e450ba08f2e440e'
        assert m['status'] == 'passed' and not m['labels_decoded'] and not m['training'] and not m['model_cache_eligible']
        assert m['all_worker_replays_byte_exact'] and m['canonical_fresh_process_agreement'] and m['feature_replays_byte_exact']
        for group in ['source', 'data', 'inputs']:
            for name, expected in p[group].items():
                assert sha(ROOT/name) == expected, name
        for name, expected in m['artifacts'].items():
            assert sha(run/name) == expected, name
        pre = ROOT/'artifacts/hyljelihyl-river-persistent-preflight-001'
        assert sha(pre/'manifest.json') == '9b07b76654644fda4a072322d4926c2321c29ee5ac0c60141410625876cab421'
        pm = json.loads((pre/'manifest.json').read_text())
        for name, expected in pm['artifacts'].items():
            assert sha(pre/name) == expected, name
        review = ROOT/'artifacts/hyljelihyl-river-persistent-independent-preflight-002'
        assert sha(review/'manifest.json') == m['independent_audit_sha256'] == '0000e4d1c8c054056a8a30d60c1d12a6ff5ee556c1e4bf203e0647da76986996'
        rm = json.loads((review/'manifest.json').read_text())
        assert rm['status'] == 'passed' and rm['protocol_sha256'] == m['protocol_sha256']
        for name, expected in rm['artifacts'].items():
            assert sha(review/name) == expected
        assert sha(ROOT/'scripts/audit_hyljelihyl_persistent_pilot.py') == sha(review/'audit_hyljelihyl_persistent_pilot.py')
        assert m['prepared_input_hashes'] == pm['prepared_input_hashes']
        assert sha(run/'prepared_inputs.json') == m['prepared_input_seal_sha256'] == sha(pre/'prepared_inputs.json')
        pids = m['worker_pids']
        assert len(set(pids)) == 2
        # Spawn workers cannot exec another program in the frozen source. Their
        # known process family is absent at these exact PIDs after joined exit.
        observation_path = ROOT/'artifacts/hyljelihyl-river-persistent-worker-exit-observation-001/manifest.json'
        observation = json.loads(observation_path.read_text())
        assert observation['status'] == 'passed' and observation['returncode'] in [0, 1]
        assert observation['expected_worker_pids'] == pids
        live = {int(line.split()[0]) for line in observation['scoped_process_output'].splitlines() if line.strip()}
        assert not set(pids).intersection(live)
        shutil.copy2(observation_path, out/'worker_exit_observation.json')
        report['worker_exit_observation_sha256'] = sha(observation_path)
        workers = []
        for replica in [1, 2]:
            folder = run/f'worker_{replica}'
            assert sha(folder/'manifest.json') == m['worker_manifests_sha256'][replica-1]
            worker = json.loads((folder/'manifest.json').read_text())
            assert worker['status'] == 'complete' and worker['pid'] == pids[replica-1] and worker['replica'] == replica
            assert worker['environment'] == p['environment'] and worker['order'] == p['orders'][replica-1]
            assert [c['case_id'] for c in worker['cases']] == worker['order']
            assert 0 < worker['peak_rss_bytes'] < p['worker_rss_limit_bytes']
            workers.append(worker)
        assert len(m['cases']) == len(p['cases']) == 59
        initial = m['initial_resources']
        assert initial['free_percent'] >= 30 and initial['pressure_derived_free_bytes'] >= 3*1024**3
        canonical = ROOT/p['canonical_directory']
        for selected, recorded, earlier in zip(p['cases'], m['cases'], pm['cases'], strict=True):
            cid = selected['case_id']
            assert recorded == earlier and recorded['case_id'] == cid
            directory = run/'inputs'/f'case_{cid:04d}'
            for name, expected in m['prepared_input_hashes'][str(cid)].items():
                assert sha(directory/name) == expected == sha(pre/'inputs'/directory.name/name)
            case, split = load_arrays(directory/'A_inputs.npz'), load_arrays(directory/'split_inputs.npz')
            oldnet = load_network(json.loads((directory/'A_network.json').read_text()))
            newnet = load_network(json.loads((directory/'split_network.json').read_text()))
            oldmodel, pa, _ = make_problem(oldnet, case, 'A')
            model, problem, segments = make_problem(newnet, split, 'C')
            assert signatures(problem) == recorded['matrix_signatures']
            reference = load_arrays(ROOT/p['cache']/f'case_{cid:04d}_result.npz')
            copies = []
            for replica, worker in enumerate(workers, start=1):
                folder = run/f'worker_{replica}'/f'case_{cid:04d}'
                proof = json.loads((folder/'proof.json').read_text())
                assert proof == next(row for row in worker['cases'] if row['case_id'] == cid)
                assert proof['status'] == 'complete' and proof['native']['status'] == 0 and proof['all_slacks_zero']
                assert proof['pid'] == worker['pid']
                resources = proof['resources']
                assert resources['free_percent'] >= 30 and resources['swapouts'] <= initial['swapouts']
                assert resources['pressure_derived_free_bytes'] >= 3*1024**3
                for name, expected in proof['artifacts'].items():
                    assert sha(folder/name) == expected
                saved = load_arrays(folder/'primal_dual.npz')
                literal_arrays(saved, load_arrays(folder/'native_output.npz'))
                checks = check_lp(problem, saved)
                for name, value in proof['kkt'].items():
                    np.testing.assert_allclose(value, checks[name], rtol=0, atol=1e-12)
                result = load_arrays(folder/'result.npz')
                expected_result = decoded_result(model, problem, segments, saved, split)
                assert set(result) == set(expected_result)
                for name, expected in expected_result.items():
                    np.testing.assert_allclose(result[name], expected, rtol=2e-11, atol=1e-7, err_msg=name)
                for name in ['minimum_slack', 'maximum_slack', 'bypass_slack', 'vest_slack']:
                    assert np.max(np.abs(result[name])) == 0
                projected = projection_matrix(oldmodel, model) @ saved['x'][:model.n_variables]
                assert max(np.max(np.abs(pa['A_eq']@projected-pa['b_eq'])), np.max(pa['A_ub']@projected-pa['b_ub']),
                           np.max(pa['bounds'][:, 0]-projected), np.max(projected-pa['bounds'][:, 1])) <= 1e-5
                expected_features = independent_features(case, split, result, reference, oldnet, newnet)
                actual_features = np.load(folder/'features.npy', allow_pickle=False)
                assert expected_features.dtype == actual_features.dtype and expected_features.shape == actual_features.shape
                assert expected_features.tobytes() == actual_features.tobytes()
                copies.append((result, saved, actual_features))
            literal_arrays(copies[0][0], copies[1][0])
            literal_arrays(copies[0][1], copies[1][1])
            assert copies[0][2].tobytes() == copies[1][2].tobytes()
            if cid in p['canonical_case_ids']:
                for index, name in [(0, 'result.npz'), (1, 'primal_dual.npz')]:
                    literal_arrays(copies[0][index], load_arrays(canonical/f'case_{cid:04d}/C_1'/name))
                assert sha(run/'worker_1'/f'case_{cid:04d}'/'features.npy') == sha(directory/'canonical_features.npy')
            report['cases'].append({'case_id': cid, 'native_full_LP_certificate_passed': True,
                                    'all_decoded_fields_reconstructed': True, 'independent62_feature_bytes_exact': True,
                                    'both_worker_arrays_literal_equal': True, 'canonical_agreement': cid in p['canonical_case_ids']})
        report.update(status='passed', result_manifest_sha256=result_hash, protocol_sha256=m['protocol_sha256'],
                      case_count=59, calls_checked=118, distinct_worker_pids=pids, workers_exited=True,
                      both_prescribed_orders_exact=True, all_cases_physical_and_replay_verified=True,
                      canonical_reuse_verified=True, canonical_case_count=36, feature_reconstruction_byte_exact=True,
                      worker_peak_rss_bytes=[w['peak_rss_bytes'] for w in workers], pilot_wall_seconds=m['runtime_seconds'],
                      scope='Input-only execution/schema evidence; no complete3647 cache or learned model authorized by this audit.')
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        shutil.copy2(__file__, out/Path(__file__).name)
        report['runtime_seconds'] = time.monotonic()-started
        report['helpers'] = {name: sha(ROOT/name) for name in ['scripts/audit_hyljelihyl_persistent_pilot.py', 'scripts/audit_hyljelihyl_river_results.py', 'scripts/audit_hyljelihyl_river_cost_v2.py']}
        report['artifacts'] = {q.name: sha(q) for q in out.iterdir() if q.is_file() and q.name != 'manifest.json'}
        (out/'manifest.json').write_text(json.dumps(report, indent=2))
        print(json.dumps({'status': report['status'], 'manifest_sha256': sha(out/'manifest.json'), 'runtime_seconds': report['runtime_seconds']}), flush=True)


if __name__ == '__main__':
    main()
