"""Independent actual664 float32 bridge and CPU vector-checkpoint result audit.

Preparation does not read competition features. Bridge mode rebuilds them only
after an explicitly supplied completed candidate, approved protocol and exits.
Fit mode never loads newer native models into the original older XGBoost.
"""
from __future__ import annotations
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']: os.environ[key] = '2'
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import shutil
import sys
import time
import traceback
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.cpu_vector14_fold_common import read_protocol, review, validate_lock, verify_files, artifacts, sha


def native_config(config: dict, p: dict, rounds: int) -> None:
    """Verify fixedCPU shared-vector histogram recipe independently of producer checks."""
    learner = config['learner']; generic = learner['generic_param']; model = learner['learner_model_param']
    assert generic['device'] == 'cpu' and int(generic['nthread']) == 2 and int(generic['seed']) == 4173
    assert int(model['num_feature']) == 664 and int(model['num_target']) == 14
    assert learner['learner_train_param']['multi_strategy'] == 'multi_output_tree'
    assert learner['objective']['name'] == 'binary:logistic'
    gradient = learner['gradient_booster']; assert 'grow_quantile_histmaker' in json.dumps(gradient)
    assert int(gradient['gbtree_model_param']['num_trees']) == rounds
    assert int(gradient['gbtree_model_param']['num_parallel_tree']) == 1
    assert gradient['gbtree_train_param']['tree_method'] == 'hist' and gradient['gbtree_train_param']['process_type'] == 'default'
    tree = gradient['tree_train_param']; params = p['model_parameters']
    for name in ['max_depth', 'learning_rate', 'min_child_weight', 'reg_lambda', 'reg_alpha', 'max_bin',
                 'subsample', 'colsample_bytree', 'colsample_bylevel', 'colsample_bynode', 'max_delta_step']:
        assert abs(float(tree[name]) - params[name]) <= 1e-6, name
    assert tree['grow_policy'] == 'depthwise' and float(tree['gamma']) == 0
    assert tree['sampling_method'] == 'uniform' and int(tree['max_leaves']) == 0
    assert tree['interaction_constraints'] == '' and tree['monotone_constraints'] == '()'


def forest_prediction(saved: dict, x: np.ndarray, rounds: int, depth: int = 6) -> np.ndarray:
    """Independently evaluate native14-vector JSON trees with float32 arithmetic."""
    learner = saved['learner']; param = learner['learner_model_param']
    assert int(param['num_target']) == 14 and int(param['num_feature']) == 664
    assert learner['objective']['name'] == 'binary:logistic'
    forest = learner['gradient_booster']['model']; trees = forest['trees']
    assert len(trees) == int(forest['gbtree_model_param']['num_trees']) == rounds
    assert forest['tree_info'] == [0] * rounds and forest['iteration_indptr'] == list(range(rounds + 1))
    intercept = np.asarray(json.loads(param['base_score']), dtype=np.float32).reshape(-1)
    assert intercept.size in [1, 14] and np.isfinite(intercept).all() and np.all((intercept > 0) & (intercept < 1))
    margin = np.broadcast_to(-np.log(1 / intercept - 1), (len(x), 14)).astype(np.float32).copy()
    rows = np.arange(len(x))
    for expected_id, tree in enumerate(trees):
        assert tree['id'] == expected_id
        nodes = int(tree['tree_param']['num_nodes'])
        assert int(tree['tree_param']['size_leaf_vector']) == 14 and 1 <= nodes <= 2 ** (depth + 1) - 1
        left = np.asarray(tree['left_children'], dtype=np.int64); right = np.asarray(tree['right_children'], dtype=np.int64)
        features = np.asarray(tree['split_indices'], dtype=np.int64); thresholds = np.asarray(tree['split_conditions'], dtype=np.float32)
        assert all(len(v) == nodes for v in [left, right, features, thresholds])
        weights = np.asarray(tree['leaf_weights'], dtype=np.float32).reshape(-1, 14)
        assert len(weights) == int(np.sum(left < 0)) and np.isfinite(weights).all()
        internal = left >= 0
        assert np.all((left[internal] > np.flatnonzero(internal)) & (left[internal] < nodes))
        assert np.all((right[internal] > np.flatnonzero(internal)) & (right[internal] < nodes))
        assert np.all((features[internal] >= 0) & (features[internal] < 664)) and np.isfinite(thresholds[internal]).all()
        node = np.zeros(len(x), dtype=np.int64)
        for _ in range(depth):
            active = left[node] >= 0
            if not active.any(): break
            current = node[active]
            node[active] = np.where(x[rows[active], features[current]] < thresholds[current], left[current], right[current])
        assert np.all(left[node] < 0), 'Tree exceeds declared depth'
        index = right[node]; assert np.all((index >= 0) & (index < len(weights)))
        margin += weights[index]
    return 1 / (1 + np.exp(-margin))


def exit_evidence(observation: dict, candidate: dict, candidate_sha: str, required_pids: list[int], mode: str) -> None:
    """Bind actual external launch/exit observation to exact candidate and native PIDs."""
    assert observation['status'] == 'passed' and observation['version'] == 'cpu_vector14_process_exit_v1'
    assert observation['mode'] == mode and observation['candidate_manifest_sha256'] == candidate_sha
    assert observation['protocol_sha256'] == candidate['protocol_sha256']
    assert observation['observed_launch_command'] == candidate['command']
    assert observation['launch_observed'] is True and observation['coordinator_pid'] > 0
    assert observation['coordinator_pid'] not in required_pids
    expected = sorted(set(required_pids + [observation['coordinator_pid']]))
    assert observation['checked_pids'] == expected
    assert observation['command'] == ['ps', '-p', ','.join(map(str, expected)), '-o', 'pid=,stat=,command=']
    assert observation['returncode'] == 1 and not observation['stdout'].strip() and not observation['stderr'].strip()
    finish = datetime.fromisoformat(candidate['timestamp_utc']) + timedelta(seconds=candidate['runtime_seconds'])
    assert datetime.fromisoformat(observation['timestamp_utc']) >= finish


def bridge_audit(folder: Path, m: dict, p: dict, year: int) -> dict:
    """Rebuild actual664 once and compare every supplied model-input/target row byte."""
    from scripts.train_lgb_shop664 import check_closure, memory_free_percent
    import xgboost as old_xgb
    assert old_xgb.__version__ == '3.2.0'
    assert memory_free_percent() >= p['resources']['minimum_free_percent_before_build']
    original = json.loads((ROOT / p['original_local_protocol']).read_text())
    assert m['environment'] == original['environment']
    assert m['peak_rss_bytes'] <= p['resources']['maximum_build_peak_rss_bytes']
    loader_protocol, metadata, folds = check_closure(original)
    spec = importlib.util.spec_from_file_location('independent_cpu_vector664_loader', ROOT / 'scripts/cuda_xgb664_loader.py')
    loader = importlib.util.module_from_spec(spec); spec.loader.exec_module(loader)
    x, labels, train, _, meta, _ = loader.build_verified_features(ROOT / original['stage'], year, loader_protocol)
    assert x.shape == (3647 * 168, 664) and x.columns.tolist() == p['feature_columns']
    assert train['Run No'].tolist() == metadata['Run No'].tolist()
    split = folds[str(year)]; sample = split['sample']; validation = split['validation_rows']
    schema = json.loads((folder / 'schema.json').read_text())
    assert schema == {'feature_columns': p['feature_columns'], 'units': p['units'],
                      'train_ids': p['folds'][str(year)]['train_ids'], 'validation_ids': p['folds'][str(year)]['validation_ids']}
    assert sha(folder / 'train_sample.npy') == p['samples'][str(year)]['sha256']
    for name, expected in [('train_sample.npy', sample), ('validation_rows.npy', validation)]:
        actual = np.load(folder / name, allow_pickle=False)
        assert actual.dtype == expected.dtype and actual.shape == expected.shape and actual.tobytes() == expected.tobytes()
    for name, rows in [('train', sample), ('validation', validation)]:
        saved = np.load(folder / f'x_{name}.npy', mmap_mode='r', allow_pickle=False)
        assert saved.dtype == np.dtype('<f4') and saved.shape == (len(rows), 664)
        blocks = []
        for begin in range(0, len(rows), 2048):
            frame = x.iloc[rows[begin:begin + 2048]]
            actual = np.ascontiguousarray(frame.to_numpy(dtype=np.float32))
            assert np.isfinite(actual).all() and actual.tobytes() == saved[begin:begin + len(actual)].tobytes()
            native = old_xgb.DMatrix(frame, feature_names=p['feature_columns'], nthread=2).get_data()
            assert native.shape == actual.shape and native.nnz == actual.size
            assert np.array_equal(native.indptr, np.arange(len(actual) + 1) * 664)
            assert np.array_equal(native.indices, np.tile(np.arange(664), len(actual)))
            assert native.data.dtype == np.float32 and native.data.tobytes() == actual.tobytes()
            blocks.append({'begin': begin, 'end': begin + len(actual), 'float32_sha256': hashlib.sha256(actual.tobytes()).hexdigest()})
        record = m['records'][name]
        assert record['sha256'] == sha(folder / f'x_{name}.npy') and record['feature_columns'] == p['feature_columns']
        assert record['source_dtypes'] == [str(d) for d in x.dtypes]
        assert record['row_index_bytes_sha256'] == hashlib.sha256(rows.tobytes()).hexdigest()
        assert record['native']['blocks'] == blocks and record['native']['xgboost'] == '3.2.0'
    y = np.column_stack([labels[u] for u in p['units']]).astype(np.uint8)
    yt = np.load(folder / 'y_train.npy', allow_pickle=False)
    assert yt.dtype == np.uint8 and yt.shape == (len(sample), 14) and yt.tobytes() == y[sample].tobytes()
    with np.load(folder / 'validation_targets.npz', allow_pickle=False) as target:
        assert target['labels'].dtype == np.uint8 and target['labels'].shape == y[validation].shape
        assert target['labels'].tobytes() == y[validation].tobytes()
        assert target['case_ids'].tolist() == p['folds'][str(year)]['validation_ids'] and target['units'].tolist() == p['units']
    assert m['training_rows'] == len(sample) == p['sample_rows'][str(year)] and m['validation_rows'] == 61320
    assert m['feature_spec'] == meta['feature_spec'] and m['maximum_decoded_target_year'] == year
    return {'all_input_bridge_bytes_verified': True, 'original_targets_and_samples_verified': True,
            'all_664_features_rebuilt': True, 'no_model_fit_or_score': True}


def fit_audit(folder: Path, m: dict, p: dict, bridge_proof: dict, bridge_proof_sha: str) -> dict:
    """Verify bounded phases and independently replay every actual tree from JSON."""
    bridge = Path(m['bridge']); assert sha(bridge / 'manifest.json') == m['bridge_manifest_sha256']
    bm = json.loads((bridge / 'manifest.json').read_text()); verify_files(bridge, bm['artifacts'])
    assert bm['status'] == 'complete' and bm['year'] == m['year'] and bm['protocol_sha256'] == m['protocol_sha256']
    assert bm.get('selection_lock_sha256') == m['selection_lock_sha256']
    assert bridge_proof['status'] == 'passed' and bridge_proof['bridge_manifest_sha256'] == m['bridge_manifest_sha256']
    assert bridge_proof['protocol_sha256'] == m['protocol_sha256'] and bridge_proof['year'] == m['year']
    assert bridge_proof['all_input_bridge_bytes_verified'] and bridge_proof['original_targets_and_samples_verified']
    fit = json.loads((folder / 'fit_manifest.json').read_text()); replay = json.loads((folder / 'replay_manifest.json').read_text())
    assert sha(folder / 'fit_manifest.json') == m['fit_manifest_sha256'] == replay['fit_manifest_sha256']
    assert sha(folder / 'replay_manifest.json') == m['replay_manifest_sha256']
    expected_runtime = json.loads((ROOT / 'artifacts/cpu-vector14-timing-001/fit_manifest.json').read_text())['environment']
    assert fit['pid'] != replay['pid'] and replay['fit_pid'] == fit['pid']
    assert [item['phase'] for item in m['phases']] == ['fit', 'replay']
    for index, (name, phase) in enumerate([('fit', fit), ('replay', replay)]):
        assert phase['status'] == 'complete' and phase['phase'] == name and phase['environment'] == expected_runtime
        assert phase['source_sha256'] == p['source']['scripts/train_cpu_vector14_fold.py']
        assert phase['protocol_sha256'] == m['protocol_sha256'] and phase['bridge_manifest_sha256'] == m['bridge_manifest_sha256']
        assert phase['year'] == m['year'] and phase['rounds'] == m['rounds'] and not phase['validation_targets_decoded']
        assert phase['memory_free_percent_before_phase'] >= p['resources']['minimum_free_percent_before_fit']
        assert phase['peak_rss_bytes'] <= p['resources']['maximum_fit_peak_rss_bytes']
        assert phase['runtime_seconds'] <= p['resources'][name + '_timeout_seconds']
        assert m['phases'][index]['pid'] == phase['pid'] and m['phases'][index]['exitcode'] == 0
        verify_files(folder, phase['artifacts'])
        assert phase['validation_native_input_check']['blocks'] == bm['records']['validation']['native']['blocks']
        assert phase['validation_native_input_check']['xgboost'] == '3.4.1'
    assert fit['training_native_input_check']['blocks'] == bm['records']['train']['native']['blocks']
    assert fit['training_native_input_check']['xgboost'] == '3.4.1' and fit['parameters'] == p['model_parameters']
    assert fit['native_shared_vector_trees_verified'] and fit['actual_600_snapshot_equals_final_prefix']
    assert fit['all_sliced_and_iteration_range_predictions_bitwise'] and replay['all_saved_models_replayed_bitwise']
    assert m['all_models_replayed_bitwise'] and m['no_validation_target_decode'] and not m['validation_targets_decoded']
    expanded = json.loads((folder / 'expanded_parameters.json').read_text())
    for name, value in p['model_parameters'].items(): assert expanded[name] == value
    assert expanded['n_estimators'] == max(m['rounds']) and expanded['early_stopping_rounds'] is None
    xv = np.load(bridge / 'x_validation.npy', mmap_mode='r', allow_pickle=False); assert xv.shape == (61320, 664)
    checked = []
    full = json.loads((folder / f'model_{max(m["rounds"])}.json').read_text())
    full_forest = full['learner']['gradient_booster']['model']
    snapshot = json.loads((folder / 'trajectory_at_600.json').read_text())
    assert snapshot['learner']['learner_model_param'] == full['learner']['learner_model_param']
    assert snapshot['learner']['gradient_booster']['model']['trees'] == full_forest['trees'][:600]
    for rounds in m['rounds']:
        saved = full if rounds == max(m['rounds']) else json.loads((folder / f'model_{rounds}.json').read_text())
        assert saved['learner']['feature_names'] == p['feature_columns']
        assert saved['learner']['learner_model_param'] == full['learner']['learner_model_param']
        assert saved['learner']['gradient_booster']['model']['trees'] == full_forest['trees'][:rounds]
        native_config(json.loads((folder / f'fit_config_{rounds}.json').read_text()), p, max(m['rounds']))
        predicted = np.load(folder / f'predictions_{rounds}.npy', allow_pickle=False)
        assert predicted.dtype == np.float32 and predicted.shape == (61320, 14) and np.isfinite(predicted).all()
        assert 0 <= predicted.min() <= predicted.max() <= 1
        for extension in ['ubj', 'json']:
            replayed = np.load(folder / f'replay_{rounds}_{extension}.npy', allow_pickle=False)
            assert replayed.dtype == predicted.dtype and replayed.shape == predicted.shape and replayed.tobytes() == predicted.tobytes()
        error = 0.
        for begin in range(0, len(xv), 4096):
            independent = forest_prediction(saved, xv[begin:begin + 4096], rounds)
            error = max(error, float(np.max(np.abs(independent - predicted[begin:begin + len(independent)]))))
        assert error <= 3e-7, 'Independent JSON tree arithmetic mismatch'
        checked.append({'rounds': rounds, 'rows': 61320, 'outputs': 14, 'independent_json_max_error': error})
    return {'all_native_models_and_replays_verified': True, 'all_input_bridge_bytes_verified': True,
            'bridge_audit_sha256': bridge_proof_sha, 'models': checked,
            'newer_native_model_loaded_in_older_xgboost': False}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument('--mode', choices=['bridge', 'fit'], required=True)
    for name in ['protocol', 'candidate-run', 'training-audit', 'exit-observation']: ap.add_argument('--' + name, type=Path, required=True)
    for name in ['protocol-sha256', 'candidate-manifest-sha256', 'training-audit-sha256', 'exit-observation-sha256', 'run-id']: ap.add_argument('--' + name, required=True)
    for name in ['bridge-audit', 'selection-lock', 'lock-audit']: ap.add_argument('--' + name, type=Path)
    for name in ['bridge-audit-sha256', 'selection-lock-sha256', 'lock-audit-sha256']: ap.add_argument('--' + name)
    ap.add_argument('--year', type=int, choices=[2021, 2022], required=True); a = ap.parse_args()
    assert Path(a.run_id).name == a.run_id and a.run_id not in ['', '.', '..']
    out = ROOT / 'artifacts' / a.run_id; out.mkdir(exist_ok=False); start = time.monotonic()
    report = {'status': 'initializing', 'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'mode': a.mode,
              'year': a.year, 'command': [sys.executable, *sys.argv], 'training': False, 'protocol_sha256': a.protocol_sha256}
    try:
        p = read_protocol(a.protocol, a.protocol_sha256); review(a.training_audit, a.training_audit_sha256, a.protocol_sha256)
        lock = None
        if a.year == 2022: lock = validate_lock(a.selection_lock, a.selection_lock_sha256, p, a.protocol_sha256, a.lock_audit, a.lock_audit_sha256)
        else: assert a.selection_lock is None and a.lock_audit is None
        assert sha(a.candidate_run / 'manifest.json') == a.candidate_manifest_sha256
        m = json.loads((a.candidate_run / 'manifest.json').read_text()); assert m['status'] == 'complete'
        assert m['protocol_sha256'] == a.protocol_sha256 and m['year'] == a.year
        verify_files(a.candidate_run, m['artifacts'])
        assert sha(a.exit_observation) == a.exit_observation_sha256
        observation = json.loads(a.exit_observation.read_text())
        if lock is not None: assert m['selection_lock_sha256'] == a.selection_lock_sha256
        if a.mode == 'bridge':
            assert m['training'] is False and m['exact_original_sample_bytes'] and m['all_dataframe_native_model_inputs_bitwise']
            exit_evidence(observation, m, a.candidate_manifest_sha256, [], 'bridge')
            report.update(bridge_audit(a.candidate_run, m, p, a.year), bridge_manifest_sha256=a.candidate_manifest_sha256)
        else:
            assert m['rounds'] == ([600, 1800] if lock is None else [lock['selected_rounds']])
            assert m['selection_lock_sha256'] == (None if lock is None else a.selection_lock_sha256)
            phases = [json.loads((a.candidate_run / f'{phase}_manifest.json').read_text()) for phase in ['fit', 'replay']]
            exit_evidence(observation, m, a.candidate_manifest_sha256, [phase['pid'] for phase in phases], 'fit')
            assert a.bridge_audit and sha(a.bridge_audit) == a.bridge_audit_sha256
            bridge_proof = json.loads(a.bridge_audit.read_text())
            report.update(fit_audit(a.candidate_run, m, p, bridge_proof, a.bridge_audit_sha256), fit_manifest_sha256=a.candidate_manifest_sha256)
        report.update(status='passed', training_audit_sha256=a.training_audit_sha256,
                      exit_observation_sha256=a.exit_observation_sha256, all_required_processes_exited=True)
    except BaseException:
        report.update(status='failed', error=traceback.format_exc()); raise
    finally:
        shutil.copy2(__file__, out / Path(__file__).name)
        report.update(runtime_seconds=time.monotonic() - start, source_sha256=sha(Path(__file__)),
                      peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss), artifacts=artifacts(out))
        (out / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n'); print(report['status'], sha(out / 'manifest.json'), flush=True)


if __name__ == '__main__': main()
