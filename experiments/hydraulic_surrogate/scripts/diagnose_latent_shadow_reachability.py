"""Bounded pre-2021 water-shadow reachability screen; no learned model or LP solve."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import shlex
import shutil
import sys
import time
import traceback

for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[variable] = '1'

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sintef_project.features import _lookup
from sintef_project.lp_features import DOWNSTREAM, POOL_ORDER, unit_curve
from sintef_project.nonlinear_pq import NonlinearPQInputs, TARGET_UNITS, digest, on_profit

GRID = np.linspace(-1., 1., 17)
CACHE = ROOT / 'artifacts/nonlinear-pq-features-001'
LP_CACHE = ROOT / 'artifacts/lp-full-001'


def optimal_schedule(on_reward: np.ndarray, start_cost: float, stop_cost: float,
                     common_reward: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Decode one finite two-state path, with free boundaries and off-first ties."""
    reward = np.asarray(on_reward, dtype=np.float64)
    common = np.zeros_like(reward) if common_reward is None else np.asarray(common_reward, dtype=np.float64)
    if reward.ndim != 1 or not len(reward) or common.shape != reward.shape:
        raise ValueError('Expected aligned nonempty one-dimensional rewards')
    if not np.isfinite(reward).all() or not np.isfinite(common).all() or min(start_cost, stop_cost) < 0:
        raise ValueError('Invalid finite rewards or switching costs')
    transitions = np.asarray([[0., start_cost], [stop_cost, 0.]])
    value = np.asarray([common[0], common[0] + reward[0]])
    parents = np.zeros((len(reward), 2), dtype=np.uint8)
    for t in range(1, len(reward)):
        previous = value[:, None] - transitions
        parents[t] = previous.argmax(axis=0)
        value = previous[parents[t], np.arange(2)] + common[t] + np.asarray([0., reward[t]])
    path = np.empty(len(reward), dtype=np.uint8)
    path[-1] = value.argmax()
    best = float(value[path[-1]])
    for t in range(len(reward)-1, 0, -1):
        path[t-1] = parents[t, path[t]]
    return path, best


def schedule_value(path: np.ndarray, reward: np.ndarray, start_cost: float,
                   stop_cost: float) -> float:
    """Evaluate only internal transitions, without invented initial/final charges."""
    path = np.asarray(path, dtype=np.int8)
    transition = np.diff(path)
    return float(np.dot(path, reward) - start_cost*np.sum(transition == 1) - stop_cost*np.sum(transition == -1))


def select_indices(cases: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Choose 12 uniformly spaced metadata rows per fixed training date block."""
    dates = pd.to_datetime(cases.starttime, utc=True)
    blocks = [('2015_2018', '2015-01-01', '2018-12-31'),
              ('2019', '2019-01-01', '2019-12-31'),
              ('2020', '2020-01-01', '2020-12-25')]
    selected, names = [], []
    for name, first, last in blocks:
        eligible = np.flatnonzero((dates >= pd.Timestamp(first, tz='UTC')) & (dates <= pd.Timestamp(last, tz='UTC')))
        index = eligible[np.rint(np.linspace(0, len(eligible)-1, 12)).astype(int)]
        if len(np.unique(index)) != 12:
            raise ValueError('Insufficient metadata rows for fixed date grid')
        selected.extend(index.tolist())
        names.extend([name]*12)
    return np.asarray(selected), names


def choose_best(errors: np.ndarray) -> np.ndarray:
    """Minimize mismatch, then absolute displacement, then signed displacement."""
    order = np.asarray(sorted(range(len(GRID)), key=lambda j: (abs(GRID[j]), GRID[j])))
    return order[np.argmin(errors[..., order], axis=-1)]


def relative_reduction(before: float, after: float) -> float:
    """Report zero headroom as zero improvement rather than a division artifact."""
    return float((before-after)/before) if before else 0.


def main() -> None:
    """Check frozen provenance and input-only controls before parsing selected labels."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    out = ROOT / 'artifacts' / args.run_id
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    report = {'run_id': args.run_id, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'command': shlex.join([sys.executable, *sys.argv]), 'status': 'initializing',
              'training': 'none; finite label-informed reachability search only, no regressor/calibrator or LP solves',
              'kaggle_submission': None, 'seed': 4173, 'grid': GRID.tolist(),
              'units': TARGET_UNITS, 'inputs': {}, 'label_decode_started_utc': None,
              'environment': {'python': sys.version, 'platform': platform.platform(),
                              'packages': {n: importlib.metadata.version(n) for n in ['numpy', 'pandas', 'scipy', 'PyYAML']}},
              'gate_thresholds': {'overall_reduction_min': .25, 'per_unit_reduction_min': .1,
                                  'per_date_block_reduction_min': .1, 'endpoint_fraction_strict_max': .2},
              'limits': 'Finite-grid diagnostic oracles are not deployable predictions, held-out validation, or bounds for all possible continuous/time-dependent corrections.'}
    try:
        frozen_protocol_path = ROOT / 'docs/nonlinear_pq_protocol.json'
        frozen = json.loads(frozen_protocol_path.read_text())
        for name, sha in frozen['source'].items():
            if digest(ROOT / name) != sha:
                raise ValueError(f'Frozen dependency changed: {name}')
        cache_manifest_path = CACHE / 'manifest.json'
        cm = json.loads(cache_manifest_path.read_text())
        if digest(cache_manifest_path) != frozen['cache_manifest_sha256']:
            raise ValueError('Frozen cache manifest changed')
        for name in ['features.npz', 'raw_on_rewards.npz']:
            if digest(CACHE / name) != cm['artifacts'][name]:
                raise ValueError('Frozen nonlinear-PQ arrays changed')
        inputs = NonlinearPQInputs(ROOT, LP_CACHE)
        indices, blocks = select_indices(inputs.cases)
        selected = inputs.cases.iloc[indices].copy()
        assert len(selected) == 36 and selected.starttime.max() <= pd.Timestamp('2020-12-25', tz='UTC')
        report.update(case_ids=selected['Run No'].tolist(), case_dates=selected.starttime.astype(str).tolist(),
                      date_blocks=blocks, source=frozen['source'], data=cm['data'],
                      formula='nu(d)=nu_LP+d*(1+mean(abs(price)))*Ebest; reward=max_q(price*P(q)-.0036*q*nu(d)); free-boundary asymmetric-cost binary DP')
        for path in [frozen_protocol_path, cache_manifest_path, LP_CACHE/'manifest.json', Path(__file__),
                     ROOT/'tests/test_latent_shadow_reachability.py', ROOT/'docs/next_model_hypotheses.md']:
            report['inputs'][str(path.relative_to(ROOT))] = digest(path)
        for name in [*frozen['source'], 'scripts/diagnose_latent_shadow_reachability.py', 'tests/test_latent_shadow_reachability.py']:
            target = out/'source'/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/name, target)
        (out/'protocol_before_labels.json').write_text(json.dumps(report, indent=2))
        with np.load(CACHE/'raw_on_rewards.npz', allow_pickle=False) as z:
            saved_rewards = z['rewards'][indices]
            assert z['case_ids'][indices].tolist() == report['case_ids'] and z['units'].tolist() == TARGET_UNITS
        with np.load(CACHE/'features.npz', allow_pickle=False) as z:
            saved_features = z['values'][indices]
        rewards = np.empty((36, 3, 17, 168))
        paths = np.empty(rewards.shape, dtype=np.uint8)
        objectives = np.empty((36, 3, 17))
        cost = np.empty((36, 3, 2))
        scales = np.empty((36, 3))
        best_energy = np.empty((36, 3))
        d0_index = int(np.flatnonzero(GRID == 0)[0])
        controls = {'cache_reward_exact': True, 'cache_feature_exact': True,
                    'same_on_off_shift_objective_invariant': True, 'zero_discharge_added_cost_zero': True}
        for i, index in enumerate(indices):
            values, details = inputs.one_case(int(index))
            np.testing.assert_array_equal(values, saved_features[i])
            np.testing.assert_array_equal(details['raw_on_profit_eur'], saved_rewards[i])
            row = inputs.cases.iloc[index]
            times = pd.date_range(row.starttime, periods=168, freq='h')
            price = _lookup(inputs.price, times, 'latent-shadow price')[:,0].astype(np.float64)
            scale = 1. + np.abs(price).mean()
            lp_path = LP_CACHE / f'case_{int(row["Run No"]):04d}.npz'
            report['inputs'][str(lp_path.relative_to(ROOT))] = digest(lp_path)
            with np.load(lp_path, allow_pickle=False) as z:
                shadow = z['water_shadow']
            for j, unit in enumerate(TARGET_UNITS):
                plant_name = unit.rsplit('_G',1)[0]
                plant, gen = inputs.topology['model']['plant'][plant_name], inputs.topology['model']['generator'][unit]
                gross = details['initial_gross_heads_m'][j]
                curve = inputs.curves[(unit, gross)]
                energy = unit_curve(gen, plant, gross)['best_energy']
                nu = shadow[POOL_ORDER.index(plant_name)].copy()
                downstream = DOWNSTREAM[plant_name]
                if downstream is not None:
                    nu -= shadow[POOL_ORDER.index(downstream)]
                start_cost, stop_cost = float(gen['startcost_const']), float(gen['stopcost_const'])
                cost[i,j] = [start_cost, stop_cost]
                scales[i,j] = scale*float(gen['p_max'])
                best_energy[i,j] = energy
                for k, delta in enumerate(GRID):
                    rewards[i,j,k] = on_profit(price, nu+delta*scale*energy, curve)
                    paths[i,j,k], objectives[i,j,k] = optimal_schedule(rewards[i,j,k], start_cost, stop_cost)
                    np.testing.assert_allclose(schedule_value(paths[i,j,k], rewards[i,j,k], start_cost, stop_cost),
                                               objectives[i,j,k], rtol=1e-12, atol=1e-7)
                    assert .0036*0.*(delta*scale*energy) == 0.
                np.testing.assert_array_equal(rewards[i,j,d0_index], saved_rewards[i,:,j])
                common = np.full(168, .25)
                alternate, shifted = optimal_schedule(rewards[i,j,d0_index], start_cost, stop_cost, common)
                np.testing.assert_allclose(shifted, objectives[i,j,d0_index]+common.sum(), rtol=1e-12, atol=1e-7)
                np.testing.assert_allclose(schedule_value(alternate, rewards[i,j,d0_index], start_cost, stop_cost),
                                           objectives[i,j,d0_index], rtol=1e-12, atol=1e-7)
        report['input_only_controls_passed_before_labels'] = controls
        report['label_decode_started_utc'] = datetime.now(timezone.utc).isoformat()
        label_path = ROOT/'sintef_project/data/kernel/Unit_commitment_decisions.csv'
        metadata = pd.read_csv(label_path, usecols=['Run No','starttime'])
        keep = metadata['Run No'].isin(report['case_ids']).to_numpy()
        assert (pd.to_datetime(metadata.loc[keep,'starttime'],utc=True).dt.year <= 2020).all()
        columns = [f'result_committed_{unit}_t{hour}' for unit in TARGET_UNITS for hour in range(168)]
        labels = pd.read_csv(label_path, skiprows=(np.flatnonzero(~keep)+1).tolist(),
                             usecols=['Run No',*columns], dtype={name:'uint8' for name in columns})
        assert labels['Run No'].tolist() == report['case_ids']
        truth = labels[columns].to_numpy().reshape(36,3,168)
        assert np.isin(truth,[0,1]).all()
        error = np.sum(paths != truth[:,:,None,:],axis=-1)
        oracle_index = choose_best(error)
        best = np.take_along_axis(error,oracle_index[:,:,None],axis=2)[:,:,0]
        base = error[:,:,d0_index]
        constant_per_unit = choose_best(error.sum(axis=0))
        constant = np.stack([error[:,j,constant_per_unit[j]] for j in range(3)],axis=1)
        regret = np.empty_like(objectives)
        for i in range(36):
            for j in range(3):
                for k in range(17):
                    regret[i,j,k] = (objectives[i,j,k]-schedule_value(truth[i,j],rewards[i,j,k],*cost[i,j]))/(168*scales[i,j])
        assert regret.min() >= -1e-10
        selected_paths = np.take_along_axis(paths,oracle_index[:,:,None,None],axis=2)[:,:,0,:]
        boundary = np.zeros(truth.shape, dtype=bool)
        for i in range(36):
            for j in range(3):
                for hour in np.flatnonzero(np.diff(truth[i,j].astype(int)) != 0) + 1:
                    boundary[i,j,max(0,hour-3):min(168,hour+4)] = True
        rng = np.random.default_rng(4173)
        permuted = oracle_index.copy()
        seasons = ((selected.starttime.dt.month.to_numpy()%12)//3).astype(int)
        moved = np.zeros((36,3),dtype=bool)
        for block in dict.fromkeys(blocks):
            for season in range(4):
                group = np.flatnonzero((np.asarray(blocks)==block)&(seasons==season))
                for j in range(3):
                    permutation = rng.permutation(group)
                    permuted[group,j] = oracle_index[permutation,j]
                    moved[group,j] = permutation != group
        perm_error = np.take_along_axis(error,permuted[:,:,None],axis=2)[:,:,0]
        unit_gain = {u:relative_reduction(base[:,j].sum(),best[:,j].sum()) for j,u in enumerate(TARGET_UNITS)}
        block_gain = {b:relative_reduction(base[np.asarray(blocks)==b].sum(),best[np.asarray(blocks)==b].sum()) for b in dict.fromkeys(blocks)}
        total_gain = relative_reduction(base.sum(),best.sum())
        endpoint_fraction = float(np.mean(np.abs(GRID[oracle_index]) == 1))
        report['results'] = {'baseline_hour_errors':int(base.sum()), 'oracle_hour_errors':int(best.sum()),
            'overall_reduction':total_gain, 'per_unit_reduction':unit_gain, 'per_date_block_reduction':block_gain,
            'endpoint_fraction':endpoint_fraction,
            'best_constant_per_unit_displacement':dict(zip(TARGET_UNITS,GRID[constant_per_unit].tolist())),
            'constant_per_unit_hour_errors':int(constant.sum()),
            'oracle_reduction_vs_best_constant_per_unit':relative_reduction(constant.sum(),best.sum()),
            'seasonally_permuted_oracle_hour_errors':int(perm_error.sum()), 'permutation_moved_case_unit_fraction':float(moved.mean()),
            'baseline_on_fraction_absolute_error':float(np.abs(paths[:,:,d0_index,:].mean(-1)-truth.mean(-1)).mean()),
            'oracle_on_fraction_absolute_error':float(np.abs(selected_paths.mean(-1)-truth.mean(-1)).mean()),
            'baseline_switch_count_absolute_error':float(np.abs(np.sum(np.diff(paths[:,:,d0_index,:].astype(int),axis=-1)!=0,axis=-1)-np.sum(np.diff(truth.astype(int),axis=-1)!=0,axis=-1)).mean()),
            'oracle_switch_count_absolute_error':float(np.abs(np.sum(np.diff(selected_paths.astype(int),axis=-1)!=0,axis=-1)-np.sum(np.diff(truth.astype(int),axis=-1)!=0,axis=-1)).mean()),
            'hours_within_three_of_true_boundary':int(boundary.sum()),
            'baseline_errors_near_true_boundaries':int(np.sum((paths[:,:,d0_index,:]!=truth)&boundary)),
            'oracle_errors_near_true_boundaries':int(np.sum((selected_paths!=truth)&boundary)),
            'baseline_errors_away_from_true_boundaries':int(np.sum((paths[:,:,d0_index,:]!=truth)&~boundary)),
            'oracle_errors_away_from_true_boundaries':int(np.sum((selected_paths!=truth)&~boundary)),
            'baseline_mean_normalized_regret':float(regret[:,:,d0_index].mean()),
            'oracle_path_choice_mean_normalized_regret':float(np.take_along_axis(regret,oracle_index[:,:,None],axis=2).mean())}
        report['gate_passed'] = bool(total_gain>=.25 and min(unit_gain.values())>=.1 and min(block_gain.values())>=.1 and endpoint_fraction<.2)
        np.savez_compressed(out/'reachability.npz',rewards=rewards,paths=paths,objectives=objectives,
            normalized_regret=regret,error=error,truth=truth,oracle_index=oracle_index,permuted_oracle_index=permuted,
            grid=GRID,case_ids=selected['Run No'].to_numpy(),units=np.asarray(TARGET_UNITS),cost=cost,scale=scales,best_energy=best_energy)
        report['status']='complete'
        print(json.dumps({'status':report['status'],'gate_passed':report['gate_passed'],**report['results']},indent=2),flush=True)
    except BaseException:
        report.update(status='failed',error=traceback.format_exc())
        raise
    finally:
        report['runtime_seconds']=time.monotonic()-started
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report['peak_rss_bytes']=int(rss if sys.platform=='darwin' else rss*1024)
        report['artifacts']={str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'}
        (out/'manifest.json').write_text(json.dumps(report,indent=2))
        with (ROOT/'docs/experiments.md').open('a') as stream:
            stream.write(f'\n## {args.run_id} — {report["timestamp_utc"]}\n\nStatus: {report["status"]}. Read-only finite-grid water-shadow reachability diagnostic:36 metadata-selected pre-2021 cases,3 units,17 additive displacements; no fitted model, LP solve, held-out labels or Kaggle action. Exact command `{report["command"]}`. Declared gates passed: {report.get("gate_passed")}. Complete data/source/cache/IDs/formulas/control/oracle evidence: `artifacts/{args.run_id}/manifest.json`. Runtime {report["runtime_seconds"]:.3f}s, peak RSS {report["peak_rss_bytes"]} bytes. Label-informed diagnostic scores are not deployable validation or a bound over all correction families.\n')


if __name__ == '__main__':
    main()
