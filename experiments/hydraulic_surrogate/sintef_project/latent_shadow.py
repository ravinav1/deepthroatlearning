"""Small predictive mapping over the frozen finite water-shadow discrepancy grid."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

from sintef_project.features import _lookup
from sintef_project.lp_features import DOWNSTREAM, POOL_ORDER, UNIT_ORDER, unit_curve
from sintef_project.nonlinear_pq import NonlinearPQInputs, TARGET_UNITS, on_profit
from sintef_project.physics import POOLS
from scripts.diagnose_latent_shadow_reachability import GRID, optimal_schedule

FEATURES = ['intake_initial_fill', 'pool_initial_fill', 'direct_weekly_natural_inflow',
            'relative_terminal_local_water', 'mean_normalized_local_shadow',
            'std_normalized_local_shadow', 'lp_zero_discharge_fraction', 'inverse_monetary_scale']
PRIOR = -(GRID ** 2) / (2 * .25 ** 2)
RIDGE = .01
TEMPERATURE_HOURS = 2.


def partition_indices(cases: pd.DataFrame, partition: str) -> np.ndarray:
    """Return fixed dates with the six training/screen boundary starts purged."""
    dates = pd.to_datetime(cases.starttime, utc=True)
    first, last = {'train': ('2015-01-01', '2018-12-25'),
                   'screen': ('2019-01-01', '2020-12-25')}[partition]
    return np.flatnonzero((dates >= pd.Timestamp(first, tz='UTC')) & (dates <= pd.Timestamp(last, tz='UTC')))


def validate_partition_metadata(case_ids: list[int], case_dates: list[str],
                                metadata: pd.DataFrame, partition: str) -> None:
    """Bind exact membership and order to raw metadata, not only endpoint dates."""
    expected=metadata.iloc[partition_indices(metadata,partition)]
    dates=pd.to_datetime(expected.starttime,utc=True).astype(str).tolist()
    if case_ids!=expected['Run No'].tolist() or case_dates!=dates:
        raise ValueError('Target cache case IDs/dates differ from the exact raw-metadata partition')


def read_selected_labels(path: Path, case_ids: list[int]) -> np.ndarray:
    """Decode only named pre-2021 rows and three target generators, once per partition."""
    if len(set(case_ids)) != len(case_ids):
        raise ValueError('Duplicate selected case IDs')
    metadata = pd.read_csv(path, usecols=['Run No', 'starttime'])
    selected = metadata['Run No'].isin(case_ids).to_numpy()
    if metadata.loc[selected, 'Run No'].tolist() != case_ids:
        raise ValueError('Selected training IDs missing, duplicated, or unordered')
    if (pd.to_datetime(metadata.loc[selected, 'starttime'], utc=True).dt.year > 2020).any():
        raise ValueError('2021/2022 labels are forbidden')
    columns = [f'result_committed_{u}_t{h}' for u in TARGET_UNITS for h in range(168)]
    labels = pd.read_csv(path, skiprows=(np.flatnonzero(~selected)+1).tolist(),
                         usecols=['Run No', *columns], dtype={c: 'uint8' for c in columns})
    if labels['Run No'].tolist() != case_ids or not np.isin(labels[columns].to_numpy(), [0,1]).all():
        raise ValueError('Invalid selected binary labels or order')
    return labels[columns].to_numpy().reshape(len(case_ids), 3, 168)


def soft_targets(errors: np.ndarray) -> np.ndarray:
    """Retain finite-grid inverse ambiguity with a fixed zero-centered prior."""
    errors = np.asarray(errors, dtype=np.float64)
    if errors.shape[-1] != 17 or not np.isfinite(errors).all() or errors.min() < 0:
        raise ValueError('Invalid mismatch vector')
    return softmax(-(errors-errors.min(axis=-1, keepdims=True))/TEMPERATURE_HOURS+PRIOR, axis=-1)


def objective_gradient(theta: np.ndarray, design: np.ndarray,
                       target: np.ndarray) -> tuple[float, np.ndarray]:
    """Convex soft-target cross entropy plus ridge, including every intercept."""
    weights = theta.reshape(17, design.shape[1])
    score = design @ weights.T + PRIOR
    log_probability = score-logsumexp(score, axis=1, keepdims=True)
    probability = np.exp(log_probability)
    loss = -np.mean(np.sum(target*log_probability, axis=1)) + RIDGE*np.sum(weights**2)
    gradient = (probability-target).T @ design/len(design) + 2*RIDGE*weights
    return float(loss), gradient.ravel()


def normalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply training-only normalization with declared clipping and constant columns."""
    return np.clip((np.asarray(x)-mean)/std, -5., 5.)


def fit_mapping(x: np.ndarray, target: np.ndarray, constant: bool = False) -> dict:
    """Fit one fixed 17-state map; no validation input, early stopping or restarts."""
    x, target = np.asarray(x, dtype=float), np.asarray(target, dtype=float)
    if x.shape != (len(target), 8) or target.shape[1] != 17 or not np.isfinite(x).all():
        raise ValueError('Unexpected latent mapping feature/target shape')
    np.testing.assert_allclose(target.sum(axis=1), 1., rtol=0, atol=1e-12)
    mean, std = x.mean(axis=0), x.std(axis=0)
    std = np.where(std == 0, 1., std)
    design = np.ones((len(x), 1)) if constant else np.column_stack([np.ones(len(x)), normalize(x,mean,std)])
    result = minimize(objective_gradient, np.zeros(17*design.shape[1]), args=(design,target), jac=True,
                      method='L-BFGS-B', options={'maxiter':500,'ftol':1e-12,'gtol':1e-8})
    if not result.success or not np.isfinite(result.fun):
        raise ValueError(f'Latent mapping optimizer failed: {result.message}')
    return {'weights':result.x.reshape(17,-1).tolist(), 'mean':mean.tolist(), 'std':std.tolist(),
            'constant':constant, 'objective':float(result.fun), 'iterations':int(result.nit),
            'optimizer_success':bool(result.success), 'message':str(result.message),
            'gradient_max_abs':float(np.abs(result.jac).max())}


def predict_distribution(x: np.ndarray, model: dict) -> np.ndarray:
    """Predict grid probabilities from input-only features and locked coefficients."""
    design = np.ones((len(x),1)) if model['constant'] else np.column_stack([
        np.ones(len(x)), normalize(x,np.asarray(model['mean']),np.asarray(model['std']))])
    return softmax(design @ np.asarray(model['weights']).T+PRIOR,axis=1)


def schedule_mixture(distribution: np.ndarray, paths: np.ndarray) -> np.ndarray:
    """Marginalize the fixed grid schedules independently within each case."""
    if distribution.shape != paths.shape[:2] or paths.shape[1:] != (17,168):
        raise ValueError('Grid/DP shapes differ')
    result = np.einsum('nk,nkt->nt',distribution,paths)
    if not np.isfinite(result).all() or result.min() < -1e-12 or result.max() > 1+1e-12:
        raise ValueError('Invalid marginal probability')
    return np.clip(result,0.,1.)


def uncertainty_summary(distribution: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return central90% grid endpoints, without claiming intervening targets are equal."""
    cumulative = np.cumsum(distribution,axis=-1)
    return GRID[np.argmax(cumulative>=.05,axis=-1)], GRID[np.argmax(cumulative>=.95,axis=-1)]


class ShadowInputs(NonlinearPQInputs):
    """Extend frozen input-only nonlinear PQ data with three selected legal series."""

    def __init__(self, root: Path, lp_cache: Path):
        super().__init__(root,lp_cache)
        self.direct = [POOLS[u.rsplit('_G',1)[0]][0] for u in TARGET_UNITS]
        self.extra = {}
        for name,filename in [('inflow','Historical_inflow_1958_2025.csv'),('water','Synthetic_water_value_2015_2024.csv')]:
            frame = pd.read_csv(self.root/'sintef_project/data/kernel'/filename,usecols=['date',*self.direct])
            frame.index = pd.to_datetime(frame.pop('date'),utc=True)
            if frame.index.has_duplicates:
                raise ValueError('Duplicate known-input times')
            self.extra[name] = frame[self.direct].astype(np.float32)

    def grid_case(self, index: int) -> dict:
        """Build eight features and every prescribed path without consulting labels."""
        old_features, details = self.one_case(index)
        row = self.cases.iloc[index]
        times = pd.date_range(row.starttime,periods=168,freq='h')
        price = _lookup(self.price,times,'shadow price')[:,0].astype(np.float64)
        inflow = _lookup(self.extra['inflow'],times,'shadow natural inflow').astype(np.float64)
        water = _lookup(self.extra['water'],pd.DatetimeIndex([row.starttime+pd.Timedelta(days=7)]),'terminal water')[0]
        volume = self.volume.loc[row.starttime]
        scale = 1.+np.abs(price).mean()
        with np.load(self.lp_cache/f'case_{int(row["Run No"]):04d}.npz',allow_pickle=False) as z:
            shadow,flow = z['water_shadow'],z['flow']
        features, rewards, paths, objectives, scales = [],[],[],[],[]
        for j,unit in enumerate(TARGET_UNITS):
            plant_name = unit.rsplit('_G',1)[0]
            pool = POOLS[plant_name]
            generator = self.topology['model']['generator'][unit]
            plant = self.topology['model']['plant'][plant_name]
            gross = details['initial_gross_heads_m'][j]
            curve = self.curves[(unit,gross)]
            energy = unit_curve(generator,plant,gross)['best_energy']
            nu = shadow[POOL_ORDER.index(plant_name)].copy()
            if DOWNSTREAM[plant_name] is not None:
                nu -= shadow[POOL_ORDER.index(DOWNSTREAM[plant_name])]
            capacity = self.topology['model']['reservoir'][pool[0]]['max_vol']
            pool_capacity = sum(self.topology['model']['reservoir'][r]['max_vol'] for r in pool)
            features.append([float(volume[pool[0]])/capacity, sum(float(volume[r]) for r in pool)/pool_capacity,
                .0036*inflow[:,j].sum()/capacity, float(water[j])/scale, float(nu.mean())/(scale*energy),
                float(nu.std())/(scale*energy), float(np.mean(flow[UNIT_ORDER.index(unit)]==0)), 1./scale])
            unit_rewards,unit_paths,unit_objectives = [],[],[]
            for displacement in GRID:
                reward = on_profit(price,nu+displacement*scale*energy,curve)
                path,value = optimal_schedule(reward,float(generator['startcost_const']),float(generator['stopcost_const']))
                unit_rewards.append(reward); unit_paths.append(path); unit_objectives.append(value)
            rewards.append(unit_rewards); paths.append(unit_paths); objectives.append(unit_objectives)
            scales.append(scale*float(generator['p_max']))
        result = {'features':np.asarray(features),'rewards':np.asarray(rewards),'paths':np.asarray(paths),
                  'objectives':np.asarray(objectives),'scale':np.asarray(scales),'old_features':old_features}
        np.testing.assert_array_equal(result['rewards'][:,8,:].T,details['raw_on_profit_eur'])
        if not np.isfinite(result['features']).all():
            raise ValueError('Nonfinite legal features')
        return result
