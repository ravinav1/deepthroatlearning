"""Nonlinear on-state profit and switching features from frozen LP water duals."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
import yaml

from sintef_project.features import _lookup, _read_timeseries
from sintef_project.lp_features import DOWNSTREAM, POOL_ORDER
from sintef_project.physics import POOLS, switching_advantage

TARGET_UNITS = ['Haukeli_G1', 'Lio_G1', 'Hogga_G1']
COLUMNS = [f'nlpq_{kind}_{unit}' for unit in TARGET_UNITS for kind in ['profit', 'switch']]
VERSION = 'nonlinear_pq_dual_switching_v1'


def digest(path: Path) -> str:
    """Hash an input file in bounded chunks."""
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def original_power(q: np.ndarray, generator: dict, plant: dict,
                   gross_head: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate original unhulled PQ with eight generator-efficiency iterations."""
    q = np.asarray(q, dtype=np.float64)
    curves = sorted(generator['turb_eff_curves'], key=lambda curve: curve['ref'])
    references = np.asarray([curve['ref'] for curve in curves], dtype=np.float64)
    loss = sum(float(v) for v in plant['main_loss']) + float(plant['penstock_loss'][generator['penstock'] - 1])
    head = gross_head - loss * q ** 2
    efficiencies = np.stack([np.interp(q, curve['x'], curve['y']) for curve in curves]) / 100
    turbine = np.asarray([np.interp(h, references, efficiencies[:, j]) for j, h in enumerate(head)])
    power = .00981 * q * head * turbine
    for _ in range(8):
        power = .00981 * q * head * turbine * np.interp(
            power, generator['gen_eff_curve']['x'], generator['gen_eff_curve']['y']) / 100
    minimum = np.interp(head, references, [min(curve['x']) for curve in curves])
    maximum = np.interp(head, references, [max(curve['x']) for curve in curves])
    return power, head, minimum, maximum


def physical_on_curve(generator: dict, plant: dict, gross_head: float) -> dict:
    """Keep physical on-points and explicit operating-boundary roots, never an off point."""
    upper = max(max(curve['x']) for curve in generator['turb_eff_curves'])
    breakpoints = np.concatenate([np.asarray(curve['x']) for curve in generator['turb_eff_curves']])
    initial = np.unique(np.concatenate([np.linspace(0., upper, 257), breakpoints]))

    def constraints(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        power, head, lower, higher = original_power(q, generator, plant, gross_head)
        return np.column_stack([head, q - lower, higher - q,
                                power - float(generator['p_min']), float(generator['p_max']) - power]), power

    values, _ = constraints(initial)
    roots = []
    for column in range(values.shape[1]):
        for j in np.flatnonzero(values[:-1, column] * values[1:, column] < 0):
            roots.append(brentq(lambda q: float(constraints(np.asarray([q]))[0][0, column]),
                                initial[j], initial[j + 1], xtol=1e-12, rtol=1e-14))
    points = np.unique(np.concatenate([initial, roots]))
    values, power = constraints(points)
    valid = (points > 0) & (values[:, 0] > 0) & (values[:, 1:] >= -1e-8).all(axis=1)
    if not valid.any():
        raise ValueError(f'No physical on-domain at gross head {gross_head}')
    qmin, qmax = float(points[valid].min()), float(points[valid].max())
    # Retain original efficiency knots and all boundary roots, then fill the on-domain.
    points = np.unique(np.concatenate([points[valid], np.linspace(qmin, qmax, 257), [qmin, qmax]]))
    values, power = constraints(points)
    valid = (points > 0) & (values[:, 0] > 0) & (values[:, 1:] >= -1e-8).all(axis=1)
    points, power = points[valid], power[valid]
    if not len(points) or not np.isfinite(power).all():
        raise ValueError('Invalid nonlinear physical on-domain')
    return {'q_m3s': points, 'power_mw': power, 'qmin_m3s': qmin, 'qmax_m3s': qmax,
            'gross_head_m': float(gross_head)}


def on_profit(price: np.ndarray, local_water_shadow: np.ndarray, curve: dict) -> np.ndarray:
    """Maximize hourly on-profit in EUR; water dual is EUR/Mm3 and q is m3/s."""
    price = np.asarray(price, dtype=np.float64)
    local_water_shadow = np.asarray(local_water_shadow, dtype=np.float64)
    if price.ndim != 1 or price.shape != local_water_shadow.shape:
        raise ValueError('Expected aligned hourly price and local water-shadow vectors')
    reward = price[:, None] * curve['power_mw'][None, :] - local_water_shadow[:, None] * (.0036 * curve['q_m3s'][None, :])
    if not np.isfinite(reward).all():
        raise ValueError('Nonfinite nonlinear operating rewards')
    return reward.max(axis=1)  # Negative on-profit is preserved; off is a distinct DP state.


class NonlinearPQInputs:
    """Read only case metadata, initial storage, known prices and frozen input-only duals."""

    def __init__(self, root: Path, lp_cache: Path):
        self.root, self.lp_cache = Path(root), Path(lp_cache)
        project = self.root / 'sintef_project'
        self.topology_path = project / 'data/extended/Tokke_Vinje_topology.yaml'
        self.topology = yaml.safe_load(self.topology_path.read_text())
        train = pd.read_csv(project / 'data/kernel/Unit_commitment_decisions.csv', usecols=['Run No', 'starttime'])
        self.cases = pd.concat([train, pd.read_csv(project / 'prediction_mapping.csv')], ignore_index=True)
        self.cases.starttime = pd.to_datetime(self.cases.starttime, utc=True)
        self.volume = _read_timeseries(project / 'data/kernel/Historical_volume_2015_2024.csv')
        self.price = _read_timeseries(project / 'data/kernel/Historical_day_ahead_price_2015_2025.csv')
        self.lp_manifest = json.loads((self.lp_cache / 'manifest.json').read_text())
        if self.lp_manifest['status'] != 'complete':
            raise ValueError('Frozen LP cache incomplete')
        for relative, sha in self.lp_manifest['data'].items():
            if digest(self.root / relative) != sha:
                raise ValueError(f'Frozen input changed: {relative}')
        for relative, sha in self.lp_manifest['source'].items():
            if digest(self.lp_cache / 'source' / relative) != sha:
                raise ValueError(f'Frozen LP source snapshot changed: {relative}')
        if self.cases['Run No'].tolist() != self.lp_manifest['selected_case_ids']:
            raise ValueError('Case IDs differ from frozen LP cache')
        self.outlet_reservoirs = {edge['from']: edge['to'] for edge in self.topology['connections']
                                 if edge['from_type'] == 'plant' and edge['to_type'] == 'reservoir'}
        self.curves = {}

    def one_case(self, index: int) -> tuple[np.ndarray, dict]:
        """Return six dimensionless features and raw rewards for one independent case."""
        row = self.cases.iloc[index]
        times = pd.date_range(row.starttime, periods=168, freq='h')
        price = _lookup(self.price, times, 'nonlinear PQ price')[:, 0].astype(np.float64)
        scale = 1. + np.abs(price).mean()
        volume = self.volume.loc[row.starttime]
        model = self.topology['model']
        head = {name: float(np.interp(volume[name], r['vol_head']['x'], r['vol_head']['y']))
                for name, r in model['reservoir'].items()}
        path = self.lp_cache / f'case_{int(row["Run No"]):04d}.npz'
        if digest(path) != self.lp_manifest['artifacts'][path.name]:
            raise ValueError('Frozen per-case LP solution changed')
        with np.load(path, allow_pickle=False) as saved:
            water_shadow = saved['water_shadow']
        features, raw_rewards, gross_heads, curve_sizes = [], [], [], []
        for unit in TARGET_UNITS:
            plant_name = unit.rsplit('_G', 1)[0]
            plant, generator = model['plant'][plant_name], model['generator'][unit]
            outlet = max(plant['outlet_line'], head.get(self.outlet_reservoirs.get(plant_name), -np.inf))
            gross = max(head[POOLS[plant_name][0]] - outlet, 1.)
            key = (unit, gross)
            if key not in self.curves:
                self.curves[key] = physical_on_curve(generator, plant, gross)
            curve = self.curves[key]
            shadow = water_shadow[POOL_ORDER.index(plant_name)].copy()
            downstream = DOWNSTREAM[plant_name]
            if downstream is not None:
                shadow -= water_shadow[POOL_ORDER.index(downstream)]
            profit = on_profit(price, shadow, curve)
            advantage = switching_advantage(profit[None, :], float(generator['startcost_const']),
                                           float(generator['stopcost_const']))[0].astype(np.float64)
            divisor = scale * float(generator['p_max'])
            features.extend([profit / divisor, advantage / divisor])
            raw_rewards.append(profit)
            gross_heads.append(gross)
            curve_sizes.append(len(curve['q_m3s']))
        values = np.column_stack(features).astype(np.float32)
        if values.shape != (168, 6) or not np.isfinite(values).all():
            raise ValueError('Invalid nonlinear PQ feature matrix')
        return values, {'raw_on_profit_eur': np.column_stack(raw_rewards),
                        'initial_gross_heads_m': gross_heads, 'curve_sizes': curve_sizes,
                        'lp_case_sha256': digest(path), 'monetary_scale': scale}


def add_nonlinear_pq_features(x: pd.DataFrame, root: Path, cache_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Append the six frozen cache columns to an unchanged complete-case frame."""
    cache_dir = Path(cache_dir)
    manifest = json.loads((cache_dir / 'manifest.json').read_text())
    if manifest['status'] != 'complete' or manifest['version'] != VERSION:
        raise ValueError('Nonlinear PQ cache is not complete or has another version')
    for relative, sha in manifest['source'].items():
        if digest(Path(root) / relative) != sha:
            raise ValueError(f'Nonlinear PQ source changed: {relative}')
    for relative, sha in manifest['data'].items():
        if digest(Path(root) / relative) != sha:
            raise ValueError(f'Nonlinear PQ input changed: {relative}')
    feature_path = cache_dir / 'features.npz'
    if digest(feature_path) != manifest['artifacts']['features.npz']:
        raise ValueError('Nonlinear PQ cache bytes changed')
    with np.load(feature_path, allow_pickle=False) as saved:
        values, columns = saved['values'], saved['columns'].tolist()
        if saved['case_ids'].tolist() != manifest['case_ids']:
            raise ValueError('Cached nonlinear PQ case order changed')
    if columns != COLUMNS or values.shape != (len(x) // 168, 168, 6):
        raise ValueError('Nonlinear PQ schema mismatch')
    if not np.array_equal(x['horizon_hour'].to_numpy(), np.tile(np.arange(168), len(values))):
        raise ValueError('Expected complete case-major hour order')
    if any(name in x for name in columns):
        raise ValueError('Nonlinear PQ columns already present')
    frame = pd.DataFrame(values.reshape(-1, 6), index=x.index, columns=columns)
    return pd.concat([x, frame], axis=1), {'version': VERSION, 'columns': columns,
        'cache_directory': str(cache_dir), 'manifest_sha256': digest(cache_dir / 'manifest.json'),
        'feature_sha256': digest(feature_path), 'formula': manifest['formula']}
