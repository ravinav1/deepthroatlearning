"""Independent canonical115 audit with exact frozen projection rank inputs."""
from __future__ import annotations
import os
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[name] = '2'
import argparse
from datetime import datetime, timezone, timedelta
import gc
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import time
import traceback
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import screen_canonical_pooled_v2 as producer
from scripts.audit_pooled_pairwise_residual_result import auc, project

HELPERS = {'scripts/audit_pooled_pairwise_residual_result.py':
           '0d191dba63e8c199baac2828bf36adc2e5eecef7a31dd6724f11da29f100ab2a'}
UNIT_FIELDS = ['lp_qmin_ratio', 'lp_power_fraction', 'lp_economic_margin',
    'lpjoint_qmin_ratio', 'lpjoint_power_fraction', 'lpjoint_economic_margin',
    'shop_state', 'shop_delta_qmin_ratio', 'shop_delta_power_fraction', 'shop_delta_economic_margin',
    'lp_context_qmin_previous_1h', 'lp_context_qmin_next_1h', 'lp_context_qmin_previous_3h',
    'lp_context_qmin_next_3h', 'lp_context_qmin_previous_6h_mean', 'lp_context_qmin_next_6h_mean',
    'lp_context_qmin_week_mean', 'lp_context_qmin_week_at_least_one_fraction', 'lp_context_qmin_week_positive_fraction']
PLANT_FIELDS = ['physics_head', 'physics_pool_fill', 'physics_supply_fraction', 'physics_catchment_inflow_fraction',
    'physics_dispatch_to_avoid_spill', 'physics_profitable_remaining_fraction', 'lp_plant_power_fraction',
    'lp_storage_fill', 'lp_storage_min_slack', 'lp_storage_max_slack', 'lpjoint_plant_power_fraction',
    'lpjoint_storage_fill', 'lpjoint_storage_min_slack', 'lpjoint_storage_max_slack',
    'norm_physics_switch_margin_minimum', 'norm_physics_switch_margin_maximum']
CALENDAR = ['horizon_hour', 'horizon_day', 'hour_of_day', 'hour_sin', 'hour_cos', 'weekday',
    'month', 'day_of_year', 'season_sin', 'season_cos', 'price_week_zscore', 'price_week_rank',
    'context_rank_remaining', 'monetary_scale_inverse']
NATIVE = {'boosting': 'gbdt', 'objective': 'binary', 'device_type': 'cpu',
    'learning_rate': '0.06', 'num_leaves': '31', 'num_threads': '2', 'seed': '4173',
    'deterministic': '1', 'force_col_wise': '1', 'max_depth': '-1', 'min_data_in_leaf': '100',
    'min_sum_hessian_in_leaf': '0.001', 'bagging_fraction': '1', 'bagging_freq': '0',
    'feature_fraction': '0.9', 'early_stopping_round': '0', 'lambda_l1': '0', 'lambda_l2': '1',
    'min_gain_to_split': '0', 'max_bin': '255', 'linear_tree': '0',
    'categorical_feature': '111,112', 'min_data_per_group': '100', 'max_cat_threshold': '32',
    'cat_l2': '10', 'cat_smooth': '10', 'max_cat_to_onehot': '4',
    'is_unbalance': '0', 'scale_pos_weight': '1', 'sigmoid': '1', 'boost_from_average': '1',
    'bin_construct_sample_cnt': '200000'}
WEIGHTS = [0., .1, .25, .5]
ROUNDS = [700, 1400]


def fixed_arithmetic_projection(value: np.ndarray, units: list[str]) -> np.ndarray:
    """Independently state three-value PAV branches using the frozen operation order.

    Rank inputs require exact float64 tie preservation. Generic isotonic min/max
    means remain a separate semantic check, never alternate score inputs.
    """
    values = np.asarray(value, dtype=np.float64)
    assert values.ndim == 2 and values.shape[1] == len(units) == 14 and len(values) % 168 == 0
    assert len(set(units)) == 14 and len(values) > 0 and np.isfinite(values).all()
    assert 0 <= values.min() <= values.max() <= 1
    indices = [units.index(f'Vinje_G{i}') for i in [1, 2, 3]]
    first, middle, last = values[:, indices].T
    first_pair = (first + middle) / 2
    last_pair = (middle + last) / 2
    three = (first + middle + last) / 3
    merge_first = first < middle
    merge_last = ~merge_first & (middle < last)
    all_from_first = merge_first & (first_pair < last)
    all_from_last = merge_last & (first < last_pair)
    merge_three = all_from_first | all_from_last
    first_only = merge_first & ~merge_three
    last_only = merge_last & ~merge_three
    result = values.copy()
    result[:, indices[0]] = np.select([merge_three, first_only], [three, first_pair], default=first)
    result[:, indices[1]] = np.select([merge_three, first_only, last_only], [three, first_pair, last_pair], default=middle)
    result[:, indices[2]] = np.select([merge_three, last_only], [three, last_pair], default=last)
    return np.clip(result, 0, 1)


def sha(path: Path) -> str:
    """Bounded-memory file hash."""
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def safe(base: Path, relative: str) -> Path:
    """Require a canonical relative path without symlink aliases."""
    part = Path(relative)
    assert not part.is_absolute() and part.as_posix() == relative and '..' not in part.parts
    result = base
    for component in part.parts:
        result = result / component
        assert not result.is_symlink()
    assert result.resolve().is_relative_to(base.resolve())
    return result


def independent_block(frame: pd.DataFrame, rows: np.ndarray, units: list[str]) -> np.ndarray:
    """State the semantic mapping independently, then fill one bounded block per unit."""
    plants = list(dict.fromkeys(u.rsplit('_G', 1)[0] for u in units))
    common = CALENDAR + [n for n in frame.columns if n.startswith('norm_price') and not n.startswith('norm_price_minus_water_')]
    shared = ['shop_state_' + u for u in units] + ['lpjoint_storage_fill_' + p for p in plants]
    assert len(common) == 53 and len(plants) == 9 and len(shared) == 23
    result = np.empty((len(rows), 14, 115), dtype='<f4')
    for index, unit in enumerate(units):
        plant, position = unit.rsplit('_G', 1)
        columns = common + [n + '_' + unit for n in UNIT_FIELDS] + [n + '_' + plant for n in PLANT_FIELDS] + shared
        values = frame.iloc[rows][columns].to_numpy(dtype=np.float32)
        assert values.shape == (len(rows), 111) and np.isfinite(values).all()
        result[:, index, :111] = values
        result[:, index, 111:] = [index, plants.index(plant), int(position), sum(u.rsplit('_G', 1)[0] == plant for u in units)]
    return result.reshape(-1, 115)


def validate_exit(observation: dict, manifest: dict, expected_sha: str) -> None:
    """Bind external process exit to the exact finalized candidate."""
    assert observation['status'] == 'passed' and observation['version'] == 'canonical_pooled_process_exit_v1'
    assert observation['candidate_manifest_sha256'] == expected_sha
    assert observation['protocol_sha256'] == manifest['protocol_sha256']
    assert observation['observed_launch_command'] == manifest['command'] and observation['launch_observed'] is True
    assert observation['producer_pid'] > 0 and observation['process_exited'] is True
    assert observation['command'] == ['ps', '-p', str(observation['producer_pid']), '-o', 'pid=,stat=,command=']
    assert observation['returncode'] == 1 and not observation['stdout'].strip() and not observation['stderr'].strip()
    finish = datetime.fromisoformat(manifest['timestamp_utc']) + timedelta(seconds=manifest['runtime_seconds'])
    assert datetime.fromisoformat(observation['timestamp_utc']) >= finish


def native_model(path: Path, rounds: int, trajectory_rounds: int) -> lgb.Booster:
    """Check exact ordinary-BCE categorical recipe and native tree dimensions."""
    native = dict(re.findall(r'^\[([^:\]]+): ([^\]]*)\]$', path.read_text(), re.MULTILINE))
    for key, value in {**NATIVE, 'num_iterations': str(trajectory_rounds)}.items():
        assert native[key] == value, key
    model = lgb.Booster(model_file=str(path))
    assert model.num_trees() == rounds and model.num_feature() == 115
    assert model.feature_name() == [f'Column_{i}' for i in range(115)]
    return model


def replay(model: lgb.Booster, features: np.ndarray, expected: np.ndarray) -> None:
    """Replay all rows in bounded chunks against float64 saved values."""
    assert expected.dtype == np.float64 and expected.shape == (len(features),)
    for begin in range(0, len(features), 32768):
        actual = model.predict(np.asarray(features[begin:begin + 32768]), num_threads=2)
        assert actual.dtype == expected.dtype and actual.tobytes() == expected[begin:begin + len(actual)].tobytes()


def independent_scores(values: np.ndarray, truth: np.ndarray, dates: pd.Series) -> dict:
    """Independent rank-sum micro AUC for the fixed calendar partitions."""
    first = (dates < pd.Timestamp('2020-07-01', tz='UTC')).to_numpy()
    return {name: auc(truth[mask], values[mask]) for name, mask in {
        'full': np.ones(len(first), dtype=bool), 'first_half': first, 'second_half': ~first}.items()}


def decide(grid: dict, controls: dict, anchor: dict) -> dict:
    """Independently apply full-score selection and six strict comparison gates."""
    keys = [(r, w) for r in ROUNDS for w in WEIGHTS]
    assert set(grid) == {f'{r}:{w}' for r, w in keys}
    r, w = sorted(keys, key=lambda x: (-grid[f'{x[0]}:{x[1]}']['full'], x[0], x[1]))[0]
    score = grid[f'{r}:{w}']
    checks = {f'{scope}_beats_{label}': score[scope] > other[scope]
              for scope in ['full', 'first_half', 'second_half']
              for label, other in [('anchor', anchor), ('matched_canonical_control', controls[str(w)])]}
    return {'selected_rounds': r, 'selected_weight': w, 'selected_scores': score,
            'matched_control_scores': controls[str(w)], 'gate_checks': checks,
            'eligible_for_later_validation': bool(w > 0 and all(checks.values()))}


def compare_scores(actual: dict, saved: dict) -> None:
    """Allow only roundoff from independent rank-sum arithmetic."""
    assert set(actual) == set(saved)
    for key, value in actual.items():
        if isinstance(value, dict):
            compare_scores(value, saved[key])
        else:
            assert np.isfinite(value) and abs(value - saved[key]) <= 1e-13, key


def read_targets(metadata: pd.DataFrame, cases: np.ndarray, units: list[str]) -> np.ndarray:
    """Read only the explicit cases directly from the original target CSV."""
    selected = np.zeros(len(metadata), dtype=bool)
    selected[cases] = True
    columns = [f'result_committed_{unit}_t{hour}' for unit in units for hour in range(168)]
    saved = pd.read_csv(ROOT / 'sintef_project/data/kernel/Unit_commitment_decisions.csv',
        skiprows=(np.flatnonzero(~selected) + 1).tolist(), usecols=['Run No', *columns],
        dtype={name: 'uint8' for name in columns})
    assert saved['Run No'].tolist() == metadata.iloc[cases]['Run No'].tolist()
    truth = saved[columns].to_numpy().reshape(len(cases), 14, 168).transpose(0, 2, 1)
    assert np.isin(truth, [0, 1]).all()
    return truth


def main() -> None:
    """Audit an actual complete screen after external process-exit evidence."""
    ap = argparse.ArgumentParser()
    for name in ['protocol', 'candidate-run', 'training-audit', 'exit-observation']:
        ap.add_argument('--' + name, type=Path, required=True)
    for name in ['protocol-sha256', 'candidate-manifest-sha256', 'training-audit-sha256', 'exit-observation-sha256', 'run-id']:
        ap.add_argument('--' + name, required=True)
    a = ap.parse_args()
    assert Path(a.run_id).name == a.run_id and a.run_id not in ['', '.', '..']
    out = ROOT / 'artifacts' / a.run_id
    out.mkdir(exist_ok=False)
    start = time.monotonic()
    report = {'status': 'initializing', 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'command': [sys.executable, *sys.argv], 'training': False,
              'protocol_sha256': a.protocol_sha256, 'candidate_manifest_sha256': a.candidate_manifest_sha256}
    try:
        for name, expected in HELPERS.items(): assert sha(ROOT / name) == expected
        assert sha(a.protocol) == a.protocol_sha256 and sha(a.training_audit) == a.training_audit_sha256
        p = json.loads(a.protocol.read_text())
        audit = json.loads(a.training_audit.read_text())
        assert audit['status'] == 'passed' and audit['protocol_sha256'] == a.protocol_sha256
        original, loader_protocol, metadata, (train, screen, sample) = producer.check_closure(p)
        folder = a.candidate_run.resolve()
        assert sha(folder / 'manifest.json') == a.candidate_manifest_sha256
        m = json.loads((folder / 'manifest.json').read_text())
        assert m['status'] == 'complete' and m['protocol_sha256'] == a.protocol_sha256
        assert m['independent_audit_sha256'] == a.training_audit_sha256 and m['environment'] == p['environment']
        assert sha(a.exit_observation) == a.exit_observation_sha256
        validate_exit(json.loads(a.exit_observation.read_text()), m, a.candidate_manifest_sha256)
        for name, expected in m['artifacts'].items(): assert sha(safe(folder, name)) == expected, name
        actual_files = {q.relative_to(folder).as_posix() for q in folder.rglob('*') if q.is_file()}
        assert actual_files == set(m['artifacts']) | {'manifest.json', 'experiment_record.md'}
        assert sha(folder / 'protocol.json') == a.protocol_sha256
        for name, expected in p['source'].items(): assert sha(safe(folder, 'source/' + name)) == expected
        units = p['units']
        model_names = set(units) | {'pooled_700', 'pooled_1400'}
        assert set(m['models']) == model_names and m['all_14_controls_and_two_pooled_checkpoints_replayed']
        assert m['maximum_training_target_year'] == 2019 and m['no_2021_or_later_targets_decoded']
        for key in ['resources_before_prediction_seal', 'resources_after_scoring']:
            value = m[key]
            assert value['free_percent'] >= 20 and value['peak_rss_bytes'] <= p['resources']['observed_peak_limit_bytes']
            assert value['elapsed_seconds'] <= p['resources']['observed_wall_limit_seconds']
        assert m['peak_rss_bytes'] <= p['resources']['observed_peak_limit_bytes']
        assert m['training_ids'] == p['train_ids'] and m['screen_ids'] == p['screen_ids']
        assert m['sampled_hours'] == 122304 and m['pooled_training_rows'] == 1712256
        assert sha(folder / 'train_sample_2020.npy') == m['sample_sha256'] == p['sample']['sha256']
        dates = pd.to_datetime(metadata.starttime, utc=True)
        rng = np.random.default_rng(4173)
        for year in [2018, 2019, 2020]:
            cases = np.flatnonzero(dates + pd.Timedelta(hours=167) < pd.Timestamp(f'{year}-01-01', tz='UTC'))
            rows = (cases[:, None] * 168 + np.arange(168)).ravel()
            expected = np.sort(rng.choice(rows, int(.4 * len(rows)), replace=False))
        assert cases.tolist() == train.tolist() and expected.tobytes() == sample.tobytes()
        train_truth = read_targets(metadata, train, units)
        expected_y = train_truth[np.searchsorted(train, sample // 168), sample % 168]
        saved_y = np.load(folder / 'training_labels.npy', allow_pickle=False)
        assert saved_y.dtype == expected_y.dtype == np.uint8 and saved_y.shape == expected_y.shape
        assert saved_y.tobytes() == expected_y.tobytes()
        del train_truth, expected_y, saved_y
        seal_path = folder / 'prediction_seal.json'
        assert sha(seal_path) == m['prediction_seal_sha256']
        seal = json.loads(seal_path.read_text())
        assert seal['status'] == 'locked' and seal['protocol_sha256'] == a.protocol_sha256
        assert seal['models'] == m['models'] and seal['all_screen_prediction_replays_bitwise']
        assert seal['maximum_training_target_year'] == 2019
        assert m['timestamp_utc'] <= seal['timestamp_utc'] <= m['screen_labels_opened_utc']
        raw = {}
        for name in ['control', 'pooled_700', 'pooled_1400']:
            path = folder / f'raw_{name}.npy'
            assert seal['raw_predictions'][path.name] == sha(path)
            values = np.load(path, allow_pickle=False)
            assert values.dtype == np.float64 and values.shape == (360, 168, 14) and np.isfinite(values).all()
            assert 0 <= values.min() <= values.max() <= 1
            raw[name] = values
        report['memory_free_percent_before_build'] = producer.base.memory_free_percent()
        assert report['memory_free_percent_before_build'] >= 40
        loader = producer.import_path(ROOT / 'scripts/cuda_xgb664_loader.py', 'independent_canonical664_loader')
        x, _, loaded_metadata, _, feature_meta, _ = loader.build_verified_features(ROOT / original['stage'], 2019, loader_protocol)
        assert x.columns.tolist() == p['prefix_columns'] and x.shape == (3647 * 168, 664)
        assert loaded_metadata['Run No'].tolist() == metadata['Run No'].tolist()
        assert json.loads(json.dumps(feature_meta['feature_spec'])) == m['prefix_feature_specification']
        assert m['feature_specification'] == p['canonical_specification']
        screen_rows = (screen[:, None] * 168 + np.arange(168)).ravel()
        checked_blocks = {}
        for label, rows in [('train', sample), ('screen', screen_rows)]:
            path = folder / f'{label}_canonical.npy'
            saved = np.load(path, mmap_mode='r', allow_pickle=False)
            assert saved.dtype == np.dtype('<f4') and saved.shape == (len(rows) * 14, 115)
            blocks = []
            for begin in range(0, len(rows), 2048):
                actual = independent_block(x, rows[begin:begin + 2048], units)
                assert saved[begin * 14:begin * 14 + len(actual)].tobytes() == actual.tobytes()
                blocks.append({'source_row_begin': begin, 'source_row_end': begin + len(actual) // 14,
                               'float32_sha256': hashlib.sha256(actual.tobytes()).hexdigest()})
            record = m['feature_records'][label]
            assert record == {'shape': list(saved.shape), 'dtype': '<f4', 'sha256': sha(path),
                'row_index_bytes_sha256': hashlib.sha256(rows.tobytes()).hexdigest(),
                'blocks': blocks, 'feature_specification': p['canonical_specification']}
            checked_blocks[label] = len(blocks)
            del saved
        del x, loaded_metadata, feature_meta
        gc.collect()
        xv = np.load(folder / 'screen_canonical.npy', mmap_mode='r', allow_pickle=False)
        for index, unit in enumerate(units):
            path = folder / f'control_{unit}.txt'
            record = m['models'][unit]
            assert sha(path) == record['checkpoint_sha256'] and record['full_screen_replay_bitwise']
            assert record['trees'] == 350 and record['features'] == 115
            model = native_model(path, 350, 350)
            replay(model, xv[index::14], raw['control'][:, :, index].ravel())
            del model
        full = native_model(folder / 'pooled_1400.txt', 1400, 1400)
        short = native_model(folder / 'pooled_700.txt', 700, 1400)
        assert short.dump_model()['tree_info'] == full.dump_model()['tree_info'][:700]
        for rounds, model in [(700, short), (1400, full)]:
            record = m['models'][f'pooled_{rounds}']
            assert sha(folder / f'pooled_{rounds}.txt') == record['checkpoint_sha256']
            assert record['trees'] == rounds and record['features'] == 115
            assert record['full_screen_replay_bitwise'] and record['single_shared_1400_round_trajectory']
            replay(model, xv, raw[f'pooled_{rounds}'].ravel())
        del xv, full, short
        gc.collect()
        truth = read_targets(metadata, screen, units)
        ids = metadata.iloc[screen]['Run No'].to_numpy()
        projection_errors = []
        def checked_projection(value):
            exact = fixed_arithmetic_projection(value, units)
            generic = project(value, units)
            difference = float(np.max(np.abs(exact - generic)))
            assert difference <= 5e-16
            projection_errors.append(difference)
            return exact
        with np.load(ROOT / p['raw_anchor'], allow_pickle=False) as saved:
            assert 'labels' not in saved.files and saved['case_ids'].tolist() == ids.tolist() and saved['units'].tolist() == units
            anchor = checked_projection(saved['probability'].reshape(-1, 14)).reshape(truth.shape)
        with np.load(ROOT / p['projected_anchor'], allow_pickle=False) as saved:
            assert saved['case_ids'].tolist() == ids.tolist() and saved['units'].tolist() == units
            assert saved['labels'].dtype == truth.dtype and saved['labels'].tobytes() == truth.tobytes()
            assert saved['predictions'].dtype == anchor.dtype and saved['predictions'].tobytes() == anchor.tobytes()
        partitions = dates.iloc[screen]
        anchor_scores = independent_scores(anchor, truth, partitions)
        grid, controls = {}, {}
        for weight in WEIGHTS:
            controls[str(weight)] = independent_scores(checked_projection(((1-weight)*anchor + weight*raw['control']).reshape(-1, 14)).reshape(truth.shape), truth, partitions)
            for rounds in ROUNDS:
                values = checked_projection(((1-weight)*anchor + weight*raw[f'pooled_{rounds}']).reshape(-1, 14)).reshape(truth.shape)
                grid[f'{rounds}:{weight}'] = independent_scores(values, truth, partitions)
        decision = decide(grid, controls, anchor_scores)
        lock_path = folder / 'selection_2020.json'
        assert sha(lock_path) == m['selection_lock_sha256']
        lock = json.loads(lock_path.read_text())
        assert lock['protocol_sha256'] == a.protocol_sha256 and lock['prediction_seal_sha256'] == sha(seal_path)
        assert lock['version'] == 'canonical_unit115_screen_lock_v1' and lock['rounds'] == ROUNDS and lock['weights'] == WEIGHTS
        assert m['screen_labels_opened_utc'] <= lock['locked_utc']
        for record in [m, lock]:
            compare_scores(grid, record['grid' if record is m else 'scores'])
            compare_scores(controls, record['control_scores'])
            compare_scores(anchor_scores, record['anchor_scores'])
            for key, value in decision.items():
                if key in ['selected_scores', 'matched_control_scores']: compare_scores(value, record[key])
                else: assert value == record[key], key
        rounds, weight = decision['selected_rounds'], decision['selected_weight']
        expected = checked_projection(((1-weight)*anchor + weight*raw[f'pooled_{rounds}']).reshape(-1, 14)).reshape(truth.shape)
        with np.load(folder / 'selected_screen_2020.npz', allow_pickle=False) as saved:
            assert saved['case_ids'].tolist() == ids.tolist() and saved['units'].tolist() == units
            assert saved['labels'].dtype == truth.dtype and saved['labels'].shape == truth.shape and saved['labels'].tobytes() == truth.tobytes()
            assert saved['predictions'].dtype == np.float64 and saved['predictions'].shape == expected.shape
            error = float(np.max(np.abs(saved['predictions'] - expected)))
            assert saved['predictions'].tobytes() == expected.tobytes()
        report.update(status='passed', **decision, grid=grid, control_scores=controls, anchor_scores=anchor_scores,
            training_audit_sha256=a.training_audit_sha256, exit_observation_sha256=a.exit_observation_sha256,
            selection_lock_sha256=sha(lock_path), original_training_and_screen_targets_verified=True,
            all_canonical_feature_bytes_independently_reconstructed=True, feature_blocks=checked_blocks,
            all14_controls_and_two_pooled_checkpoints_replayed=True, actual700_native_tree_prefix_verified=True,
            prediction_seal_before_screen_labels_verified=True, independent_projection_max_error=error, generic_projection_semantic_max_error=max(projection_errors),
            exact_frozen_projection_rank_bytes_verified=True,
            producer_process_exited=True, future_year_or_submission_authorized=False)
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        shutil.copy2(__file__, out / Path(__file__).name)
        report.update(runtime_seconds=time.monotonic()-start, peak_rss_bytes=producer.base.peak_rss(),
                      environment=producer.base.environment(), source_sha256=sha(Path(__file__)), helper_sha256=HELPERS)
        (out / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
        print(report['status'], sha(out / 'manifest.json'), flush=True)


if __name__ == '__main__':
    main()
