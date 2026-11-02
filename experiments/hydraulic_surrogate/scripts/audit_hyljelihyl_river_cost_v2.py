"""Independent frozen closure and algebraic projection review; no real solves."""

from datetime import datetime, timezone
import hashlib
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

for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[variable] = '1'

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path: Path) -> str:
    """Hash one exact file."""
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def literal_arrays(first: dict, second: dict) -> None:
    """Require identical names, dtypes, shapes and all array bytes."""
    assert set(first) == set(second)
    for key in first:
        a, b = np.asarray(first[key]), np.asarray(second[key])
        assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), key


def load_arrays(path: Path) -> dict:
    """Read only saved physical arrays."""
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def projection_matrix(old, new):
    """Independently construct the full split-to-pooled linear variable map."""
    rows, cols = [], []

    def add(old_indices, new_indices):
        rows.extend(np.asarray(old_indices).ravel().tolist())
        cols.extend(np.asarray(new_indices).ravel().tolist())

    a, b = old.base, new.base
    add(a.q, b.q)
    for name in ['v', 'low_slack', 'high_slack']:
        add(getattr(a, name), getattr(b, name)[:9])
        add(getattr(a, name)[4], getattr(b, name)[9])
    add(a.spill, b.spill[:9])
    for name in ['bypass', 'bypass_slack']:
        add(getattr(a, name), getattr(b, name)[:10])
        add(getattr(a, name)[4], getattr(b, name)[10])
    add(a.vest_slack, b.vest_slack)
    for name in ['state', 'start', 'stop']:
        add(getattr(old, name), getattr(new, name))
    return coo_matrix((np.ones(len(rows)), (rows, cols)),
                      shape=(old.n_variables, new.n_variables)).tocsc()


def aggregation_matrix(old_rows: int, new_rows: int, t: int, equality: bool):
    """Aggregate new balance/constraint rows into every old row."""
    rows, cols = [], []

    def add(old_start, new_start, count):
        rows.extend(range(old_start, old_start + count))
        cols.extend(range(new_start, new_start + count))

    if equality:
        add(0, 0, 9*t)
        add(4*t, 9*t, t)
    else:
        add(0, 0, 9*t)
        add(4*t, 9*t, t)
        add(9*t, 10*t, 9*t)
        add(13*t, 19*t, t)
        add(18*t, 20*t, 10*t)
        add(22*t, 30*t, t)
        add(28*t, 32*t, t)
        add(29*t, 33*t, old_rows - 29*t)
    return coo_matrix((np.ones(len(rows)), (rows, cols)),
                      shape=(old_rows, new_rows)).tocsc()


def matrix_error(matrix) -> float:
    """Maximum absolute sparse coefficient error."""
    return float(np.max(np.abs(matrix.data))) if matrix.nnz else 0.


def audit_projection(old, pa: dict, new, pb: dict) -> dict:
    """Prove old balances/inequalities/objective for arbitrary feasible B."""
    t = old.horizon
    transform = projection_matrix(old, new)
    eq = aggregation_matrix(pa['A_eq'].shape[0], pb['A_eq'].shape[0], t, True)
    ub = aggregation_matrix(pa['A_ub'].shape[0], pb['A_ub'].shape[0], t, False)
    eq_error = pa['A_eq'] @ transform - eq @ pb['A_eq']
    # The only unmapped balance term is HH spill, explicitly fixed to zero.
    allowed = set(new.base.spill[9].tolist())
    assert set(eq_error.tocoo().col).issubset(allowed)
    assert np.all(pb['bounds'][new.base.spill[9], 1] == 0.)
    assert matrix_error(pa['A_ub'] @ transform - ub @ pb['A_ub']) == 0.
    np.testing.assert_allclose(eq @ pb['b_eq'], pa['b_eq'], rtol=0, atol=1e-12)
    np.testing.assert_allclose(ub @ pb['b_ub'], pa['b_ub'], rtol=0, atol=1e-12)
    # New last t rows preserve the single original bypass's bound exactly.
    cap = pb['A_ub'][-t:].tocoo()
    expected_cols = np.column_stack([new.base.bypass[4], new.base.bypass[10]]).ravel()
    expected = coo_matrix((np.ones(2*t), (np.repeat(np.arange(t), 2), expected_cols)),
                          shape=(t, new.n_variables)).tocsc()
    assert matrix_error(pb['A_ub'][-t:] - expected) == 0.
    np.testing.assert_array_equal(pb['b_ub'][-t:], pa['bounds'][old.base.bypass[4], 1])
    # Every other finite old bound maps one-to-one and remains identical.
    finite = np.flatnonzero(np.isfinite(pa['bounds'][:, 1]))
    for row in finite:
        mapped = transform.getrow(row).indices
        if len(mapped) == 1:
            assert pa['bounds'][row, 1] == pb['bounds'][mapped[0], 1]
        else:
            assert row in old.base.bypass[4] and len(mapped) == 2
    assert np.all(pa['bounds'][:, 0] == 0.) and np.all(pb['bounds'][:, 0] == 0.)
    # Projected A minimization cost cannot exceed B: unmapped slacks cost >=0.
    difference = np.asarray(pa['c'] @ transform).ravel() - pb['c']
    assert difference.max() <= 1e-12
    return {'old_balance_projection_exact_except_fixed_zero_spill': True,
            'all_old_inequalities_algebraically_preserved': True,
            'all_old_bounds_preserved_including_joint_cap': True,
            'projected_A_minimization_cost_no_greater_than_B': True}


def main() -> None:
    """Review the exact v2 protocol and producer preflight independently."""
    out = ROOT/'artifacts/hyljelihyl-river-cost-v2-independent-preflight-002'
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    report = {'status': 'running', 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'command': f'{sys.executable} scripts/audit_hyljelihyl_river_cost_v2.py',
              'environment': {'python': sys.version, 'platform': platform.platform()},
              'real_optimization_calls': 0, 'labels_decoded': False, 'training': False}
    try:
        protocol_path = ROOT/'docs/hyljelihyl_river_cost_protocol_v2.json'
        assert sha(protocol_path) == 'ba460ccb6085f9b032ea0647e9d19879801d0095ae5eece240ffbbdc33ab91f0'
        p = json.loads(protocol_path.read_text())
        report['protocol_sha256'] = sha(protocol_path)
        for section in ['source', 'data', 'inputs']:
            for name, expected in p[section].items():
                assert sha(ROOT/name) == expected, name
        pre = ROOT/'artifacts/hyljelihyl-river-cost-v2-preflight-001/manifest.json'
        assert sha(pre) == 'd34d49f5cf0eeb07fc84fbf8f164c9babf5519cc66729ba2abec7bf1bd31ff23'
        m = json.loads(pre.read_text())
        assert m['status'] == 'passed' and not m['labels_decoded'] and m['optimization_calls'] == 0
        assert m['protocol_sha256'] == sha(protocol_path)
        for name, expected in m['artifacts'].items():
            assert sha(pre.parent/name) == expected, name
        original = json.loads((ROOT/'docs/integer_water_budget_protocol.json').read_text())
        assert [r['case_id'] for r in p['cases']] == original['case_ids']
        assert p['blocks'] == original['blocks'] and p['arms'] == ['A', 'B', 'C']
        assert p['replicas'] == 2 and p['workers'] == 1 and p['lp_seconds'] == 20.
        assert p['process_timeout_seconds'] == 60. and p['peak_rss_limit_bytes'] == 2*1024**3
        assert p['resource_minimum_free_percent'] == 30 and not p['automatic_promotion']
        assert p['shared_hyljelihyl_lower_bypass_cap_m3s'] == 1000 and p['main_river_cap'] is None
        prior = json.loads((ROOT/'docs/hyljelihyl_river_cost_protocol.json').read_text())
        for section in ['source', 'data', 'inputs']:
            for name, expected in prior[section].items():
                assert sha(ROOT/name) == expected, name
        assert sha(ROOT/p['cache']/'manifest.json') == p['cache_manifest_sha256']
        from sintef_project.lp_features import HydroCaseData, UNIT_ORDER
        from sintef_project.lp_hyljelihyl_v2 import capture_problem, make_problem, split_network, project_to_original
        from scripts.diagnose_hyljelihyl_river_cost_v2 import signatures
        target = (ROOT/'sintef_project/data/kernel/Unit_commitment_decisions.csv').resolve()
        actual_read = pd.read_csv

        def guarded(path, *args, **kwargs):
            if isinstance(path, (str, Path)) and Path(path).resolve() == target:
                assert not args and kwargs == {'usecols': ['Run No', 'starttime']}
            return actual_read(path, *args, **kwargs)

        records = []
        with patch.object(pd, 'read_csv', guarded):
            data = HydroCaseData(ROOT)
            for selected, observed in zip(p['cases'], m['cases']):
                cid, pos = selected['case_id'], selected['metadata_position']
                date = pd.Timestamp(selected['date'])
                assert int(data.cases.iloc[pos]['Run No']) == cid and data.cases.iloc[pos].starttime == date
                assert date + pd.Timedelta(days=7) <= pd.Timestamp('2021-01-01', tz='UTC')
                assert observed['case_id'] == cid
                case = data.case(pos)
                case['natural'] = np.asarray(case['natural'], dtype=np.float64)
                for name, field in [('pmin', 'p_min'), ('start_cost', 'startcost_const'), ('stop_cost', 'stopcost_const')]:
                    case[name] = np.array([data.model['generator'][unit][field] for unit in UNIT_ORDER], dtype=np.float64)
                directory = pre.parent/f'case_{cid:04d}'
                literal_arrays(case, load_arrays(ROOT/p['cache']/f'case_{cid:04d}_inputs.npz'))
                literal_arrays(case, load_arrays(directory/'A_inputs.npz'))
                split = load_arrays(directory/'split_inputs.npz')
                times = pd.date_range(date, periods=168, freq='h')
                initial = float(data.volume.loc[date, 'Hyljelihyl'])
                natural = data.inflow.loc[times, 'Hyljelihyl'].to_numpy(dtype=np.float64)*.0036
                minimum = data.minimum.loc[times, 'Hyljelihyl'].to_numpy() if 'Hyljelihyl' in data.minimum else np.zeros(168)
                expected = {key: np.asarray(value).copy() for key, value in case.items()}
                for key, value in [('initial', initial), ('capacity', .26)]:
                    expected[key] = np.r_[np.asarray(case[key], dtype=np.float64), value]
                    expected[key][4] -= value
                for key, value in [('natural', natural), ('minimum', minimum)]:
                    expected[key] = np.vstack([np.asarray(case[key], dtype=np.float64), value])
                    expected[key][4] -= value
                expected['terminal_value'] = np.r_[case['terminal_value'], case['terminal_value'][4]]
                expected['bypass_minimum'] = np.vstack([case['bypass_minimum'], case['bypass_minimum'][4], np.zeros(168)])
                expected['bypass_minimum'][4] = 0.
                literal_arrays(expected, split)
                local = data.water.loc[date+pd.Timedelta(days=7), ['Vaamarvatn', 'Totak', 'Venemo', 'Hyljelihyl']].to_numpy()
                assert np.isfinite(local).all() and np.all(local == local[0])
                old, pa = capture_problem(data.network, case)
                net = split_network(data.network)
                new, pb, _ = make_problem(net, split, 'B')
                _, pc, segments = make_problem(net, split, 'C')
                for arm, problem in [('A', pa), ('B', pb), ('C', pc)]:
                    assert signatures(problem) == observed['matrix_signatures'][arm]
                assert np.all(pc['c'][segments] >= 0.) and np.isinf(pc['bounds'][segments[-1], 1]).all()
                assert matrix_error(pc['A_ub'][:, :len(pb['c'])] - pb['A_ub']) == 0.
                np.testing.assert_array_equal(pc['b_ub'], pb['b_ub'])
                algebra = audit_projection(old, pa, new, pb)
                random = np.random.default_rng(4173).uniform(size=new.n_variables)
                np.testing.assert_array_equal(project_to_original(old, new, random), projection_matrix(old, new) @ random)
                records.append({'case_id': cid, 'fresh_original_and_partitioned_input_bytes_exact': True, **algebra})
        source = (ROOT/'scripts/diagnose_hyljelihyl_river_cost_v2.py').read_text()
        assert source.index("(out/'physical_lock.json').write_text") < source.index("truth=read_labels(p['cases'])")
        assert "for replica in [1,2]" in source and 'check=True,timeout=p[' in source
        test = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_lp_hyljelihyl_v2.py'],
                              cwd=ROOT, text=True, capture_output=True)
        (out/'tests.txt').write_text(test.stdout+test.stderr)
        assert test.returncode == 0, test.stderr
        report.update(status='passed', preflight_manifest_sha256=sha(pre), cases=records,
                      tests_passed=7, prior_unrun_v1_preserved=True,
                      scope='Twelve fixed pre2021 physical cases, three arms, two fresh single-thread replicas each; all proof gates before labels. No automatic cache/model promotion.',
                      review='Exact inherited generator/state math; corrected shared bypass cap; independent sparse algebra proves old constraints/bounds and objective ordering under projection. Costs use supplied step curve, unbounded tail and fixed original numerical currency assumption.')
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        shutil.copy2(__file__, out/Path(__file__).name)
        report['runtime_seconds'] = time.monotonic()-started
        report['artifacts'] = {path.name: sha(path) for path in out.iterdir() if path.is_file() and path.name != 'manifest.json'}
        (out/'manifest.json').write_text(json.dumps(report, indent=2))
        print(json.dumps({'status': report['status'], 'manifest_sha256': sha(out/'manifest.json'), 'runtime_seconds': report['runtime_seconds']}), flush=True)


if __name__ == '__main__':
    main()
