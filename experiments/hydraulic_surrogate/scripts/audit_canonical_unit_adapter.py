"""Independent scalar audit of the canonical adapter using synthetic values only."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sintef_project.canonical_unit_features import canonical_block, feature_specification, write_canonical_rows


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    started = time.monotonic()
    output = ROOT / 'artifacts/canonical-unit-adapter-independent-audit-001'
    output.mkdir(exist_ok=False)
    metadata = json.loads((ROOT / 'docs/lgb_shop664_protocol.json').read_text())
    columns, units = metadata['feature_columns'], metadata['units']
    spec = feature_specification(columns, units)
    # These lists are an independently stated semantic contract, not adapter constants.
    unit_fields = ['lp_qmin_ratio', 'lp_power_fraction', 'lp_economic_margin',
                   'lpjoint_qmin_ratio', 'lpjoint_power_fraction', 'lpjoint_economic_margin',
                   'shop_state', 'shop_delta_qmin_ratio', 'shop_delta_power_fraction', 'shop_delta_economic_margin',
                   'lp_context_qmin_previous_1h', 'lp_context_qmin_next_1h', 'lp_context_qmin_previous_3h',
                   'lp_context_qmin_next_3h', 'lp_context_qmin_previous_6h_mean', 'lp_context_qmin_next_6h_mean',
                   'lp_context_qmin_week_mean', 'lp_context_qmin_week_at_least_one_fraction',
                   'lp_context_qmin_week_positive_fraction']
    plant_fields = ['physics_head', 'physics_pool_fill', 'physics_supply_fraction', 'physics_catchment_inflow_fraction',
                    'physics_dispatch_to_avoid_spill', 'physics_profitable_remaining_fraction', 'lp_plant_power_fraction',
                    'lp_storage_fill', 'lp_storage_min_slack', 'lp_storage_max_slack', 'lpjoint_plant_power_fraction',
                    'lpjoint_storage_fill', 'lpjoint_storage_min_slack', 'lpjoint_storage_max_slack',
                    'norm_physics_switch_margin_minimum', 'norm_physics_switch_margin_maximum']
    common = ['horizon_hour', 'horizon_day', 'hour_of_day', 'hour_sin', 'hour_cos', 'weekday', 'month', 'day_of_year',
              'season_sin', 'season_cos', 'price_week_zscore', 'price_week_rank', 'context_rank_remaining', 'monetary_scale_inverse']
    common += [name for name in columns if name.startswith('norm_price') and not name.startswith('norm_price_minus_water_')]
    plants = list(dict.fromkeys(unit.split('_G')[0] for unit in units))
    shared = ['shop_state_' + unit for unit in units] + ['lpjoint_storage_fill_' + plant for plant in plants]
    expected_names = common + ['own_' + name for name in unit_fields + plant_fields] + ['global_' + name for name in shared]
    expected_names += ['unit_id', 'plant_id', 'unit_position', 'plant_unit_count']
    assert len(common) == 53 and spec['columns'] == expected_names and len(expected_names) == 115
    rng = np.random.default_rng(9017)
    values = rng.normal(size=(11, 664))
    values[2, :] = -0.0
    frame = pd.DataFrame(values, columns=columns)
    rows = np.array([8, 2, 0, 10, 1, 4], dtype=np.int64)
    expected = []
    for source_row in rows:
        for index, unit in enumerate(units):
            plant, position = unit.split('_G')
            names = common + [name + '_' + unit for name in unit_fields] + [name + '_' + plant for name in plant_fields] + shared
            vector = [np.float32(frame.at[int(source_row), name]) for name in names]
            vector += [np.float32(index), np.float32(plants.index(plant)), np.float32(int(position)),
                       np.float32(sum(other.startswith(plant + '_G') for other in units))]
            expected.append(vector)
    expected = np.asarray(expected, dtype='<f4')
    actual = canonical_block(frame, rows, spec)
    assert actual.shape == (84, 115) and actual.dtype == expected.dtype and actual.tobytes() == expected.tobytes()
    for block_rows in [1, 2, 4, 100]:
        path = output / f'synthetic_block{block_rows}.npy'
        record = write_canonical_rows(frame, rows, path, spec, block_rows=block_rows)
        saved = np.load(path, allow_pickle=False)
        assert saved.tobytes() == expected.tobytes() and record['sha256'] == sha(path)
    # Row-hour then unit exactly matches C-order labels, including unsorted sampled hours.
    synthetic_labels = np.arange(11 * 14).reshape(11, 14)
    flattened = synthetic_labels[rows].reshape(-1)
    for k in range(len(actual)):
        assert flattened[k] == synthetic_labels[rows[k // 14], int(actual[k, -4])]
    # An unselected row is not read; selected nonfinite inputs are rejected.
    frame.iloc[3, 0] = np.nan
    assert canonical_block(frame, rows, spec).tobytes() == expected.tobytes()
    frame.iloc[rows[0], 0] = np.nan
    try:
        canonical_block(frame, rows, spec)
    except ValueError:
        pass
    else:
        raise AssertionError('Selected nonfinite input was not rejected')
    report = {'status': 'passed', 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'synthetic_only': True, 'competition_feature_or_target_values_read': False,
              'all_115_fields_all_14_units_scalar_oracle_bitwise': True, 'signed_zero_and_float64_cast_verified': True,
              'four_streaming_chunk_sizes_verified': True, 'row_hour_unit_label_alignment_verified': True,
              'chronology_assumption': 'Caller must pin unit/schema order and chronological case/sample scope; adapter itself reads only supplied rows and never targets.',
              'source': {name: sha(ROOT / name) for name in ['sintef_project/canonical_unit_features.py',
                        'tests/test_canonical_unit_features.py', 'scripts/audit_canonical_unit_adapter.py']},
              'metadata_protocol_sha256': sha(ROOT / 'docs/lgb_shop664_protocol.json'),
              'runtime_seconds': time.monotonic() - started}
    report['artifacts'] = {path.name: sha(path) for path in output.iterdir() if path.is_file()}
    (output / 'manifest.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'path': str(output.relative_to(ROOT)), 'sha256': sha(output / 'manifest.json'), **report}))


if __name__ == '__main__':
    main()
