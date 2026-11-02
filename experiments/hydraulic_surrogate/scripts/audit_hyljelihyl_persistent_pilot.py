"""Independent label-free source/input/62-feature review for persistent C pilot."""
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import platform
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
from scripts.audit_hyljelihyl_river_cost_v2 import sha, literal_arrays, load_arrays
from scripts.audit_hyljelihyl_river_cost_36 import partition


def independent_features(case, split, result, reference, oldnet, newnet):
    """Evaluate the published62 formulas without calling the feature producer."""
    columns = []
    scale = 1+np.mean(np.abs(case['price']))
    for unit in range(14):
        old_dual = reference['water_shadow'][oldnet.unit_up[unit]].copy()
        new_dual = result['water_shadow'][newnet.unit_up[unit]].copy()
        if oldnet.unit_down[unit] >= 0:
            old_dual -= reference['water_shadow'][oldnet.unit_down[unit]]
        if newnet.unit_down[unit] >= 0:
            new_dual -= result['water_shadow'][newnet.unit_down[unit]]
        state = result['state'][unit]
        # Preserve the sign of in-range zero, as the frozen clip operation does.
        columns += [np.where(state < 0, 0., np.where(state > 1, 1., state)),
                    (result['flow'][unit]-reference['flow'][unit])/case['qmin'][unit],
                    (result['power'][unit]-reference['power'][unit])/case['pmax'][unit],
                    (old_dual-new_dual)/(case['energy'][unit, 0]*scale)]
    columns += [result['storage'][9]/.26, result['storage'][4]/307.,
                result['river_volume']/(.0036*40), result['bypass'][10]/(.0036*40),
                result['bypass'][4]/(.0036*40),
                (result['water_shadow'][4]-result['water_shadow'][9])/(case['energy'][7, 0]*scale)]
    return np.column_stack(columns).astype(np.float32)


def main() -> None:
    """Review protocol closure and all prepared cases without running a solver."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--protocol-sha256', required=True)
    ap.add_argument('--preflight', type=Path, required=True)
    ap.add_argument('--preflight-sha256', required=True)
    ap.add_argument('--run-id', required=True)
    a = ap.parse_args()
    assert Path(a.run_id).name == a.run_id and a.run_id not in ['', '.', '..']
    out = ROOT/'artifacts'/a.run_id
    out.mkdir(exist_ok=False)
    start = time.monotonic()
    report = {'status': 'initializing', 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'command': [sys.executable, *sys.argv], 'environment': {'python': sys.version, 'platform': platform.platform()},
              'optimization_calls': 0, 'labels_decoded': False, 'training': False, 'full_cache_authorized': False}
    try:
        pp = ROOT/'docs/hyljelihyl_river_persistent_pilot_protocol.json'
        assert sha(pp) == a.protocol_sha256 == 'a1d860345862538c22498ca675d25e8be6730fda6051255e5e450ba08f2e440e'
        p = json.loads(pp.read_text())
        report['protocol_sha256'] = sha(pp)
        for group in ['source', 'data', 'inputs']:
            for name, expected in p[group].items():
                assert sha(ROOT/name) == expected, name
        assert sha(a.preflight) == a.preflight_sha256
        pre = json.loads(a.preflight.read_text())
        assert pre['status'] == 'passed' and pre['optimization_calls'] == 0 and not pre['labels_decoded']
        assert pre['protocol_sha256'] == sha(pp) and not pre['model_cache_eligible']
        for name, expected in pre['artifacts'].items():
            assert sha(a.preflight.parent/name) == expected, name
        assert p['workers'] == 2 and p['lp_seconds'] == 20.
        assert p['resource_minimum_free_percent'] == 30 and p['minimum_pressure_derived_headroom_bytes'] == 3*1024**3
        assert p['worker_rss_limit_bytes'] == 2*1024**3 and not p['labels_decoded'] and not p['training']
        previous = json.loads((ROOT/'docs/hyljelihyl_river_cost_36_protocol.json').read_text())
        for name, expected in previous['source'].items():
            assert p['source'][name] == expected
        canonical = ROOT/p['canonical_directory']
        assert sha(canonical/'manifest.json') == p['canonical_manifest_sha256'] == 'f403ed698047e6e47e94160fac245b019e6e286f8d9256014187bf6d1d29b4e7'
        proof = ROOT/p['canonical_audit']
        assert sha(proof) == p['canonical_audit_sha256'] == '0be8e29f96f08fe25c20a5b69182f1efff2a289e6a60ebafe715c9f13b14b85f'
        audit = json.loads(proof.read_text())
        assert audit['status'] == 'passed' and audit['result_manifest_sha256'] == p['canonical_manifest_sha256']
        cm = json.loads((canonical/'manifest.json').read_text())
        assert cm['status'] == 'complete' and p['canonical_case_ids'] == [c['case_id'] for c in previous['cases']]
        positions = sorted(set(c['metadata_position'] for c in previous['cases']) |
                           set(np.rint(np.linspace(0, 3646, 24)).astype(int).tolist()))
        assert positions == [c['metadata_position'] for c in p['cases']]
        ids = [c['case_id'] for c in p['cases']]
        assert len(set(ids)) == len(ids) == len(positions)
        assert p['orders'] == [ids, np.random.default_rng(4173).permutation(ids).tolist()]
        assert p['orders'][0] != p['orders'][1]
        seal = a.preflight.parent/'prepared_inputs.json'
        assert sha(seal) == pre['prepared_input_seal_sha256']
        sealed = json.loads(seal.read_text())
        assert sealed['protocol_sha256'] == sha(pp) and sealed['inputs'] == pre['prepared_input_hashes']
        from sintef_project.lp_features import HydroCaseData, UNIT_ORDER
        from sintef_project.lp_hyljelihyl_v2 import split_network, make_problem
        from sintef_project.hyljelihyl_river_features import specification, COLUMNS
        from scripts.diagnose_hyljelihyl_river_cost_v2 import signatures, json_network
        assert p['feature_specification'] == specification() and len(COLUMNS) == len(set(COLUMNS)) == 62
        expected_columns = [f'river_c_{kind}_{unit}' for unit in UNIT_ORDER
                            for kind in ['state', 'delta_qmin_ratio', 'delta_power_fraction', 'delta_economic_margin']]
        expected_columns += ['river_c_'+name for name in ['hyljelihyl_fill', 'remaining_vinje_fill', 'main_flow_40',
                                                        'hyljelihyl_bypass_40', 'remaining_bypass_40', 'river_shadow_wedge']]
        assert COLUMNS == expected_columns
        target = (ROOT/'sintef_project/data/kernel/Unit_commitment_decisions.csv').resolve()
        read = pd.read_csv

        def guarded(path, *args, **kwargs):
            if isinstance(path, (str, Path)) and Path(path).resolve() == target:
                assert not args and kwargs == {'usecols': ['Run No', 'starttime']}
            return read(path, *args, **kwargs)

        records = []
        with patch.object(pd, 'read_csv', guarded):
            data = HydroCaseData(ROOT)
            assert len(data.cases) == 3647
            oldnet, newnet = data.network, split_network(data.network)
            for selected, observed in zip(p['cases'], pre['cases'], strict=True):
                cid, pos, date = selected['case_id'], selected['metadata_position'], pd.Timestamp(selected['date'])
                assert int(data.cases.iloc[pos]['Run No']) == cid and data.cases.iloc[pos].starttime == date
                assert observed['case_id'] == cid
                directory = a.preflight.parent/'inputs'/f'case_{cid:04d}'
                actual = {q.name: sha(q) for q in directory.iterdir() if q.is_file()}
                assert actual == sealed['inputs'][str(cid)]
                case = data.case(pos)
                case['natural'] = np.asarray(case['natural'], dtype=np.float64)
                for name, field in [('pmin', 'p_min'), ('start_cost', 'startcost_const'), ('stop_cost', 'stopcost_const')]:
                    case[name] = np.array([data.model['generator'][unit][field] for unit in UNIT_ORDER], dtype=np.float64)
                literal_arrays(case, load_arrays(ROOT/p['cache']/f'case_{cid:04d}_inputs.npz'))
                literal_arrays(case, load_arrays(directory/'A_inputs.npz'))
                split = partition(case, data, date)
                literal_arrays(split, load_arrays(directory/'split_inputs.npz'))
                for name, net in [('A', oldnet), ('split', newnet)]:
                    assert json.loads((directory/f'{name}_network.json').read_text()) == json_network(net)
                local = data.water.loc[date+pd.Timedelta(days=7), ['Vaamarvatn', 'Totak', 'Venemo', 'Hyljelihyl']].to_numpy()
                assert np.isfinite(local).all() and np.all(local == local[0])
                _, problem, _ = make_problem(newnet, split, 'C', 20.)
                assert signatures(problem) == observed['matrix_signatures']
                if cid in p['canonical_case_ids']:
                    folder = canonical/f'case_{cid:04d}'
                    literal_arrays(case, load_arrays(folder/'A_inputs.npz'))
                    literal_arrays(split, load_arrays(folder/'split_inputs.npz'))
                    result = load_arrays(folder/'C_1/result.npz')
                    reference = load_arrays(ROOT/p['cache']/f'case_{cid:04d}_result.npz')
                    expected = independent_features(case, split, result, reference, oldnet, newnet)
                    actual_features = np.load(directory/'canonical_features.npy', allow_pickle=False)
                    assert expected.dtype == actual_features.dtype and expected.shape == actual_features.shape
                    assert expected.tobytes() == actual_features.tobytes()
                records.append({'case_id': cid, 'fresh_inputs_partition_matrix_and_seal_exact': True,
                                'canonical62_features_independently_reconstructed': cid in p['canonical_case_ids']})
        source = (ROOT/'scripts/pilot_hyljelihyl_river_persistent.py').read_text()
        assert "ctx=mp.get_context('spawn')" in source and 'for i in [1,2]' in source
        assert "protocol['orders'][replica-1]" in source and "solve(newnet,split,'C',20.)" in source
        assert 'assert digest(PROTOCOL)==protocol_sha256' in source
        assert "digest(out/'prepared_inputs.json')==input_seal_sha256" in source
        assert "for name,h in protocol['source'].items():assert digest(ROOT/name)==h,name" in source
        assert "for name,h in input_hashes[str(cid)].items():assert digest(source/name)==h,name" in source
        assert 'reference_path' in source and "protocol['inputs'][str(reference_path.relative_to(ROOT))]" in source
        assert 'read_labels' not in source and 'result_committed_' not in source
        for pattern in ['test_hyljelihyl_river_features.py', 'test_hyljelihyl_persistent_pilot.py']:
            test = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-p', pattern],
                                  cwd=ROOT, text=True, capture_output=True)
            (out/(pattern+'.txt')).write_text(test.stdout+test.stderr)
            assert test.returncode == 0, test.stderr
        report.update(status='passed', preflight_manifest_sha256=sha(a.preflight), cases=records,
                      selected_case_count=len(ids), canonical_cases=36, synthetic_tests_passed=6,
                      feature_formula_review='Own-topology opportunity-cost differences; fixed qmin/pmax/first-segment-energy/price scales; six shared flow/storage/wedge features; float32 finite output and exact664 prefix preservation.',
                      execution_review='Two spawned persistent single-thread workers, identical full case population with different fixed orders and fresh LP model percase. Child verifies source/protocol/environment/prepared inputs/control; all result/LP/feature arrays byte-identical,36 canonical fresh-process matches; no label decode or full-cache promotion.',
                      loader_review='Future726 loader rejects incomplete caches and requires external3647-case physical/replay/feature proof. Only source/synthetic rejection/prefix behavior reviewed here; no complete cache exists or is authorized.')
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        shutil.copy2(__file__, out/Path(__file__).name)
        report['runtime_seconds'] = time.monotonic()-start
        report['helper_sha256'] = {name: sha(ROOT/name) for name in ['scripts/audit_hyljelihyl_river_cost_v2.py', 'scripts/audit_hyljelihyl_river_cost_36.py']}
        report['artifacts'] = {q.name: sha(q) for q in out.iterdir() if q.is_file() and q.name != 'manifest.json'}
        (out/'manifest.json').write_text(json.dumps(report, indent=2))
        print(json.dumps({'status': report['status'], 'manifest_sha256': sha(out/'manifest.json'), 'runtime_seconds': report['runtime_seconds']}), flush=True)


if __name__ == '__main__':
    main()
