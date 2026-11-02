"""Align existing input-only features into one common generator representation.

No labels, predictions, fitted statistics or new physical solutions enter this
adapter. Row order is original case-hour, then the supplied generator order.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd

UNIT_FAMILIES = [
    'lp_qmin_ratio', 'lp_power_fraction', 'lp_economic_margin',
    'lpjoint_qmin_ratio', 'lpjoint_power_fraction', 'lpjoint_economic_margin',
    'shop_state', 'shop_delta_qmin_ratio', 'shop_delta_power_fraction',
    'shop_delta_economic_margin',
    'lp_context_qmin_previous_1h', 'lp_context_qmin_next_1h',
    'lp_context_qmin_previous_3h', 'lp_context_qmin_next_3h',
    'lp_context_qmin_previous_6h_mean', 'lp_context_qmin_next_6h_mean',
    'lp_context_qmin_week_mean', 'lp_context_qmin_week_at_least_one_fraction',
    'lp_context_qmin_week_positive_fraction',
]
PLANT_FAMILIES = [
    'physics_head', 'physics_pool_fill', 'physics_supply_fraction',
    'physics_catchment_inflow_fraction', 'physics_dispatch_to_avoid_spill',
    'physics_profitable_remaining_fraction',
    'lp_plant_power_fraction', 'lp_storage_fill', 'lp_storage_min_slack',
    'lp_storage_max_slack', 'lpjoint_plant_power_fraction',
    'lpjoint_storage_fill', 'lpjoint_storage_min_slack', 'lpjoint_storage_max_slack',
    'norm_physics_switch_margin_minimum', 'norm_physics_switch_margin_maximum',
]
CALENDAR = [
    'horizon_hour', 'horizon_day', 'hour_of_day', 'hour_sin', 'hour_cos',
    'weekday', 'month', 'day_of_year', 'season_sin', 'season_cos',
    'price_week_zscore', 'price_week_rank',
]


def feature_specification(prefix_columns: list[str], units: list[str]) -> dict:
    """Declare a deterministic 115-column mapping from the original664 schema."""
    if len(prefix_columns) != 664 or len(set(prefix_columns)) != 664:
        raise ValueError('The exact664 prefix schema is required')
    if len(units) != 14 or len(set(units)) != 14:
        raise ValueError('Fourteen distinct ordered generators are required')
    unit_plants, positions = [], []
    for unit in units:
        plant, number = unit.rsplit('_G', 1)
        position = int(number)
        if position < 1:
            raise ValueError('Generator position must be positive')
        unit_plants.append(plant)
        positions.append(position)
    plants = list(dict.fromkeys(unit_plants))
    if len(plants) != 9:
        raise ValueError('Expected the original nine generating plants')
    common = CALENDAR + ['context_rank_remaining', 'monetary_scale_inverse'] + [
        name for name in prefix_columns
        if name.startswith('norm_price') and not name.startswith('norm_price_minus_water_')]
    global_context = ['shop_state_' + unit for unit in units] + [
        'lpjoint_storage_fill_' + plant for plant in plants]
    own = [[family + '_' + unit for family in UNIT_FAMILIES] +
           [family + '_' + plant for family in PLANT_FAMILIES]
           for unit, plant in zip(units, unit_plants)]
    required = common + global_context + [name for unit_names in own for name in unit_names]
    missing = sorted(set(required) - set(prefix_columns))
    if missing or len(common) != 53:
        raise ValueError(f'Unexpected common schema or missing source columns: {missing}')
    columns = common + ['own_' + name for name in UNIT_FAMILIES + PLANT_FAMILIES] + [
        'global_' + name for name in global_context] + [
        'unit_id', 'plant_id', 'unit_position', 'plant_unit_count']
    if len(columns) != len(set(columns)) or len(columns) != 115:
        raise ValueError('Canonical schema must contain115 distinct columns')
    counts = Counter(unit_plants)
    identities = [[j, plants.index(plant), position, counts[plant]]
                  for j, (plant, position) in enumerate(zip(unit_plants, positions))]
    return {'version': 'canonical_existing664_unit115_v1', 'prefix_columns': prefix_columns,
            'units': units, 'plants': plants, 'columns': columns,
            'common_sources': common, 'own_sources': own,
            'global_sources': global_context, 'identity_values': identities,
            'categorical_columns': ['unit_id', 'plant_id'],
            'row_order': 'case_hour_then_unit', 'dtype': '<f4',
            'source_cast': 'Existing numeric features cast directly to float32; no fitted normalization.'}


def canonical_block(frame: pd.DataFrame, rows: np.ndarray, specification: dict) -> np.ndarray:
    """Remap a bounded row block without accessing targets or any other case rows."""
    if frame.columns.tolist() != specification['prefix_columns']:
        raise ValueError('Source column identity or ordering changed')
    if rows.ndim != 1 or rows.dtype.kind not in 'iu' or not len(rows):
        raise ValueError('Expected nonempty integer case-hour row indices')
    if np.any(rows < 0) or np.any(rows >= len(frame)) or len(np.unique(rows)) != len(rows):
        raise ValueError('Invalid or duplicate case-hour rows')
    expected = feature_specification(specification['prefix_columns'], specification['units'])
    if specification != expected:
        raise ValueError('Canonical mapping changed')
    source = frame.iloc[rows].to_numpy(dtype=np.float32)
    if not np.isfinite(source).all():
        raise ValueError('Original feature inputs must be finite')
    indices = {name: j for j, name in enumerate(specification['prefix_columns'])}
    common = [indices[name] for name in specification['common_sources']]
    own = np.asarray([[indices[name] for name in names] for names in specification['own_sources']])
    shared = [indices[name] for name in specification['global_sources']]
    result = np.empty((len(rows), 14, 115), dtype=np.float32)
    result[:, :, :53] = source[:, None, common]
    result[:, :, 53:88] = source[:, own]
    result[:, :, 88:111] = source[:, None, shared]
    result[:, :, 111:] = np.asarray(specification['identity_values'], dtype=np.float32)
    return result.reshape(len(rows) * 14, 115)


def write_canonical_rows(frame: pd.DataFrame, rows: np.ndarray, path: Path,
                         specification: dict, block_rows: int = 2048) -> dict:
    """Stream the remapping to an immutable float32 NPY with per-block byte hashes."""
    if path.exists() or block_rows < 1:
        raise ValueError('Output exists or block size is invalid')
    if rows.ndim != 1 or rows.dtype.kind not in 'iu' or not len(rows):
        raise ValueError('Expected nonempty integer case-hour row indices')
    if len(np.unique(rows)) != len(rows):
        raise ValueError('Duplicate case-hour rows')
    output = np.lib.format.open_memmap(path, mode='w+', dtype='<f4', shape=(len(rows) * 14, 115))
    records = []
    for begin in range(0, len(rows), block_rows):
        block = canonical_block(frame, rows[begin:begin + block_rows], specification)
        output[begin * 14:begin * 14 + len(block)] = block
        stored = output[begin * 14:begin * 14 + len(block)]
        if stored.tobytes() != block.tobytes():
            raise ValueError('Written canonical feature bytes changed')
        records.append({'source_row_begin': begin, 'source_row_end': begin + len(block) // 14,
                        'float32_sha256': hashlib.sha256(block.tobytes()).hexdigest()})
    output.flush()
    del output
    with path.open('rb') as stream:
        checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'shape': [len(rows) * 14, 115], 'dtype': '<f4', 'sha256': checksum,
            'row_index_bytes_sha256': hashlib.sha256(rows.tobytes()).hexdigest(),
            'blocks': records, 'feature_specification': specification}
