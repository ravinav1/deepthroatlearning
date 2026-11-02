"""Independently verify the fixed36-case extension and exact12-case reuse."""
from datetime import datetime, timezone
import argparse
import ast
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

for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[key] = '1'
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_hyljelihyl_river_cost_v2 import sha, literal_arrays, load_arrays


def partition(case: dict, data, date: pd.Timestamp) -> dict:
    """Independently reproduce the fixed physical inventory partition."""
    times = pd.date_range(date, periods=168, freq='h')
    expected = {key: np.asarray(value).copy() for key, value in case.items()}
    initial = float(data.volume.loc[date, 'Hyljelihyl'])
    natural = data.inflow.loc[times, 'Hyljelihyl'].to_numpy(dtype=np.float64)*.0036
    minimum = data.minimum.loc[times, 'Hyljelihyl'].to_numpy() if 'Hyljelihyl' in data.minimum else np.zeros(168)
    for key, value in [('initial', initial), ('capacity', .26)]:
        expected[key] = np.r_[np.asarray(case[key], dtype=np.float64), value]
        expected[key][4] -= value
    for key, value in [('natural', natural), ('minimum', minimum)]:
        expected[key] = np.vstack([np.asarray(case[key], dtype=np.float64), value])
        expected[key][4] -= value
    expected['terminal_value'] = np.r_[case['terminal_value'], case['terminal_value'][4]]
    expected['bypass_minimum'] = np.vstack([case['bypass_minimum'], case['bypass_minimum'][4], np.zeros(168)])
    expected['bypass_minimum'][4] = 0.
    return expected


def main() -> None:
    """Inspect frozen producer evidence without real optimization or labels."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--preflight', type=Path, required=True)
    parser.add_argument('--preflight-sha256', required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    assert Path(args.run_id).name == args.run_id and args.run_id not in ['', '.', '..']
    out = ROOT/'artifacts'/args.run_id
    out.mkdir(exist_ok=False)
    start = time.monotonic()
    report = {'status': 'initializing', 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'command': [sys.executable, *sys.argv], 'environment': {'python': sys.version, 'platform': platform.platform()},
              'real_optimization_calls': 0, 'labels_decoded': False, 'training': False}
    try:
        path = ROOT/'docs/hyljelihyl_river_cost_36_protocol.json'
        assert sha(path) == args.protocol_sha256
        p = json.loads(path.read_text())
        report['protocol_sha256'] = sha(path)
        for section in ['source', 'data', 'inputs']:
            for name, expected in p[section].items():
                assert sha(ROOT/name) == expected, name
        assert sha(args.preflight) == args.preflight_sha256
        pre = json.loads(args.preflight.read_text())
        assert pre['status'] == 'passed' and not pre['labels_decoded'] and pre['optimization_calls'] == 0
        assert pre['protocol_sha256'] == sha(path)
        for name, expected in pre['artifacts'].items():
            assert sha(args.preflight.parent/name) == expected, name
        old_path = ROOT/'docs/hyljelihyl_river_cost_protocol_v2.json'
        assert sha(old_path) == 'ba460ccb6085f9b032ea0647e9d19879801d0095ae5eece240ffbbdc33ab91f0'
        old = json.loads(old_path.read_text())
        for section in ['source', 'data']:
            for name, expected in old[section].items():
                assert p[section][name] == expected
        for name in ['arms', 'lp_seconds', 'process_timeout_seconds', 'replicas', 'workers', 'resource_minimum_free_percent',
                     'peak_rss_limit_bytes', 'breakpoints_m3s', 'marginal_costs', 'hyljelihyl_capacity_mm3',
                     'hyljelihyl_bypass_cap_m3s', 'main_river_cap', 'shared_hyljelihyl_lower_bypass_cap_m3s',
                     'currency', 'scope', 'metrics', 'automatic_promotion']:
            assert p[name] == old[name], name
        selection = json.loads((ROOT/'docs/lp_shop_relaxation_protocol.json').read_text())
        ids = [row['case_id'] for row in p['cases']]
        assert ids == selection['case_ids'] and len(ids) == len(set(ids)) == 36
        assert p['blocks'] == selection['date_blocks']
        assert [str(pd.Timestamp(row['date']).date()) for row in p['cases']] == selection['case_dates']
        reused_ids = [row['case_id'] for row in old['cases']]
        assert p['reuse']['case_ids'] == reused_ids and len(reused_ids) == 12
        assert p['new_case_ids'] == [i for i in ids if i not in reused_ids] and len(p['new_case_ids']) == 24
        reused = ROOT/p['reuse']['directory']
        assert sha(reused/'manifest.json') == p['reuse']['manifest_sha256'] == 'fbb1ac2019fd0a8bf81695d7bc7959dc5d887c6149ecd55f74be60383e3ed401'
        rm = json.loads((reused/'manifest.json').read_text())
        proof = ROOT/p['reuse']['audit']
        assert sha(proof) == p['reuse']['audit_sha256'] == '783a1d687a3e8e4a9833c0e7b23ca0a688738784630a0ed8aef7559594a25496'
        audit = json.loads(proof.read_text())
        assert audit['status'] == 'passed' and audit['result_manifest_sha256'] == p['reuse']['manifest_sha256']
        assert rm['status'] == 'complete' and rm['protocol_sha256'] == sha(old_path)
        for name, expected in rm['artifacts'].items():
            assert sha(reused/name) == expected, name
        for cid in reused_ids:
            original = reused/f'case_{cid:04d}'
            copied = args.preflight.parent/f'case_{cid:04d}'
            first = {q.relative_to(original).as_posix(): sha(q) for q in original.rglob('*') if q.is_file()}
            second = {q.relative_to(copied).as_posix(): sha(q) for q in copied.rglob('*') if q.is_file()}
            assert first == second and first
        from sintef_project.lp_features import HydroCaseData, UNIT_ORDER
        from sintef_project.lp_hyljelihyl_v2 import make_problem, split_network
        from scripts.diagnose_hyljelihyl_river_cost_v2 import signatures
        target = (ROOT/'sintef_project/data/kernel/Unit_commitment_decisions.csv').resolve()
        actual_read = pd.read_csv

        def guarded(path, *positional, **kwargs):
            if isinstance(path, (str, Path)) and Path(path).resolve() == target:
                assert not positional and kwargs == {'usecols': ['Run No', 'starttime']}
            return actual_read(path, *positional, **kwargs)

        records = []
        with patch.object(pd, 'read_csv', guarded):
            data = HydroCaseData(ROOT)
            for chosen, observed in zip(p['cases'], pre['cases'], strict=True):
                cid, pos, date = chosen['case_id'], chosen['metadata_position'], pd.Timestamp(chosen['date'])
                assert observed['case_id'] == cid
                assert int(data.cases.iloc[pos]['Run No']) == cid and data.cases.iloc[pos].starttime == date
                assert date + pd.Timedelta(days=7) <= pd.Timestamp('2021-01-01', tz='UTC')
                case = data.case(pos)
                case['natural'] = np.asarray(case['natural'], dtype=np.float64)
                for key, field in [('pmin', 'p_min'), ('start_cost', 'startcost_const'), ('stop_cost', 'stopcost_const')]:
                    case[key] = np.array([data.model['generator'][unit][field] for unit in UNIT_ORDER], dtype=np.float64)
                directory = args.preflight.parent/f'case_{cid:04d}'
                literal_arrays(case, load_arrays(ROOT/p['cache']/f'case_{cid:04d}_inputs.npz'))
                literal_arrays(case, load_arrays(directory/'A_inputs.npz'))
                split = partition(case, data, date)
                literal_arrays(split, load_arrays(directory/'split_inputs.npz'))
                local = data.water.loc[date+pd.Timedelta(days=7), ['Vaamarvatn', 'Totak', 'Venemo', 'Hyljelihyl']].to_numpy()
                assert np.isfinite(local).all() and np.all(local == local[0])
                for arm in ['A', 'B', 'C']:
                    _, problem, _ = make_problem(data.network if arm == 'A' else split_network(data.network), case if arm == 'A' else split, arm)
                    assert signatures(problem) == observed['matrix_signatures'][arm]
                if cid in reused_ids:
                    assert observed['reuse']['all_original_bytes_preserved']
                    old_record = next(row for row in rm['cases'] if row['case_id'] == cid)
                    assert observed['arms'] == old_record['arms'] and observed['matrix_signatures'] == old_record['matrix_signatures']
                else:
                    assert 'reuse' not in observed and 'arms' not in observed
                    assert not any(directory.glob('?_?'))
                records.append({'case_id': cid, 'input_partition_and_matrix_signatures_exact': True, 'reused': cid in reused_ids})
        old_source = (ROOT/'scripts/diagnose_hyljelihyl_river_cost_v2.py').read_text()
        source = (ROOT/'scripts/diagnose_hyljelihyl_river_cost_36.py').read_text()
        functions = lambda text: {n.name: ast.dump(n) for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)}
        before, after = functions(old_source), functions(source)
        for name in ['json_network', 'load_network', 'signatures', 'worker', 'read_labels']:
            assert before[name] == after[name], name
        assert source.index("(out/'physical_lock.json').write_text") < source.index("truth=read_labels(p['cases'])")
        assert "if 'reuse' in record:" in source and "'case_count':36" in source
        test = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_hyljelihyl_river_reuse.py'],
                              cwd=ROOT, text=True, capture_output=True)
        (out/'tests.txt').write_text(test.stdout+test.stderr)
        assert test.returncode == 0, test.stderr
        report.update(status='passed', preflight_manifest_sha256=sha(args.preflight), cases=records,
                      exact_twelve_completed_case_directories_preserved=True, new_case_ids=p['new_case_ids'],
                      fixed_original36_selection=True, worker_and_label_loader_AST_unchanged=True,
                      source_review='Population extension only; inherited v2 math and algebraic proof unchanged. Exact prior root-audited twelve-case bytes,24 new cases, all36×3 complete physical gates and seal before labels; no fallback, fit or automatic cache promotion.')
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        shutil.copy2(__file__, out/Path(__file__).name)
        report.update(runtime_seconds=time.monotonic()-start, helper_sha256=sha(ROOT/'scripts/audit_hyljelihyl_river_cost_v2.py'))
        report['artifacts'] = {q.name: sha(q) for q in out.iterdir() if q.is_file() and q.name != 'manifest.json'}
        (out/'manifest.json').write_text(json.dumps(report, indent=2))
        print(json.dumps({'status': report['status'], 'manifest_sha256': sha(out/'manifest.json'), 'runtime_seconds': report['runtime_seconds']}), flush=True)


if __name__ == '__main__':
    main()
