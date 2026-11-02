"""Independently replay a completed LGB726 fold and its fixed chronological gate."""
from __future__ import annotations

import os
for _name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[_name] = '2'

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import time
import traceback

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_lgb_shop664_result import sha, auc, project
from scripts import train_lgb_river726 as runner

HELPERS = {
    'scripts/audit_lgb_shop664_result.py': '8603a85b3e1e251818b94744c597b84627ac78ee4e6e7cf4000442d10038ceb1',
    'scripts/audit_pooled_pairwise_residual_result.py': '0d191dba63e8c199baac2828bf36adc2e5eecef7a31dd6724f11da29f100ab2a',
}
WEIGHTS = [0., .1, .25, .5]
NATIVE = {
    'boosting': 'gbdt', 'objective': 'binary', 'device_type': 'cpu', 'num_iterations': '350',
    'learning_rate': '0.06', 'num_leaves': '31', 'num_threads': '2', 'seed': '4173',
    'deterministic': '1', 'force_col_wise': '1', 'max_depth': '-1', 'min_data_in_leaf': '100',
    'min_sum_hessian_in_leaf': '0.001', 'bagging_fraction': '1', 'bagging_freq': '0',
    'feature_fraction': '0.9', 'early_stopping_round': '0', 'lambda_l1': '0', 'lambda_l2': '1',
    'min_gain_to_split': '0', 'max_bin': '255', 'linear_tree': '0',
}


def prefix_fingerprint(frame: pd.DataFrame, names: list[str]) -> list[dict]:
    """Hash columns separately to verify exact prefix bytes without a large copy."""
    assert frame.columns[:len(names)].tolist() == names
    return [{'name': name, 'dtype': str(frame[name].dtype),
             'sha256': hashlib.sha256(frame[name].to_numpy().tobytes()).hexdigest()} for name in names]


def check_lock(lock: dict, protocol: dict, protocol_sha: str) -> None:
    """Independently validate a 2021 lock before any 2022 target is opened."""
    assert lock['version'] == 'lgb_river726_selection_lock_v1' and lock['selection_year'] == 2021
    assert lock['protocol_sha256'] == protocol_sha
    assert lock['trainer_sha256'] == protocol['source']['scripts/train_lgb_river726.py']
    assert lock['candidate_weights'] == protocol['candidate_weights'] == WEIGHTS
    scores = lock['scores_2021']
    assert set(scores) == {str(w) for w in WEIGHTS} and np.isfinite(list(scores.values())).all()
    chosen = max(WEIGHTS, key=lambda w: (scores[str(w)], -w))
    assert lock['selected_weight'] == chosen
    assert lock['selected_2021_micro_auc'] == scores[str(chosen)]
    assert lock['anchor_2021_micro_auc'] == scores['0.0']
    control = lock['control_2021_micro_auc']
    assert np.isfinite(control) and 0 <= control <= 1
    assert lock['eligible_for_2022'] == (chosen > 0 and scores[str(chosen)] > max(scores['0.0'], control))
    for kind in ['anchor', 'control']:
        assert lock[kind + '_prediction_sha256'] == protocol[kind]['2021']['prediction_sha256']


def verify_control(record: dict, year: int, expected_parameters: dict) -> None:
    """Independently require the actual raw664 control and its completed replay audit."""
    directory = f'artifacts/lgb-shop664-{year}-001'
    assert record['prediction'] == f'{directory}/validation_{year}.npz'
    assert record['manifest'] == f'{directory}/manifest.json'
    assert record['audit'] == f'artifacts/lgb-shop664-{year}-independent-audit-001/manifest.json'
    for kind in ['prediction', 'manifest', 'audit']:
        assert sha(ROOT / record[kind]) == record[kind + '_sha256']
    manifest = json.loads((ROOT / record['manifest']).read_text())
    audit = json.loads((ROOT / record['audit']).read_text())
    assert manifest['status'] == 'complete' and manifest['protocol_sha256'] == runner.BASE_SHA
    assert manifest['model_parameters'] == expected_parameters and manifest['completed_units'] == 14
    assert manifest['artifacts'][f'validation_{year}.npz'] == record['prediction_sha256']
    assert audit['status'] == 'passed' and audit['protocol_sha256'] == runner.BASE_SHA
    assert audit['candidate_manifest_sha256'] == record['manifest_sha256']
    assert audit['all14_native_checkpoints_replayed'] and audit['original_target_values_verified']


def main() -> None:
    """Audit only actual complete artifacts, after their model process has exited."""
    ap = argparse.ArgumentParser()
    for name in ['protocol', 'candidate-run', 'training-audit', 'selection-lock', 'lock-audit']:
        ap.add_argument('--' + name, type=Path, required=name in ['protocol', 'candidate-run', 'training-audit'])
    for name in ['protocol', 'candidate-manifest', 'training-audit', 'selection-lock', 'lock-audit']:
        ap.add_argument('--' + name + '-sha256', required=name in ['protocol', 'candidate-manifest', 'training-audit'])
    ap.add_argument('--year', type=int, choices=[2021, 2022], required=True)
    ap.add_argument('--run-id', required=True)
    args = ap.parse_args()
    assert Path(args.run_id).name == args.run_id and args.run_id not in ['', '.', '..']
    out = ROOT / 'artifacts' / args.run_id
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    report = {'status': 'initializing', 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'year': args.year, 'training': False, 'command': [sys.executable, *sys.argv],
              'environment': runner.base.environment()}
    try:
        for name, expected in HELPERS.items():
            assert sha(ROOT / name) == expected, name
        assert sha(args.protocol) == args.protocol_sha256
        p = json.loads(args.protocol.read_text())
        # This closure rejects an incomplete cache before any feature or target
        # array is decoded. The actual result audit additionally checks controls.
        original, loader_protocol, _, _ = runner.check_closure(p)
        for year in [2021, 2022]:
            verify_control(p['control'][str(year)], year, p['model_parameters'])
        assert sha(args.training_audit) == args.training_audit_sha256
        training_audit = json.loads(args.training_audit.read_text())
        assert training_audit['status'] == 'passed' and training_audit['protocol_sha256'] == args.protocol_sha256
        directory = args.candidate_run.resolve()
        assert sha(directory / 'manifest.json') == args.candidate_manifest_sha256
        m = json.loads((directory / 'manifest.json').read_text())
        assert m['status'] == 'complete' and m['training'] is True and m['year'] == args.year
        assert m['protocol_sha256'] == args.protocol_sha256
        assert m['independent_audit_sha256'] == args.training_audit_sha256
        assert m['completed_units'] == 14 and set(m['models']) == set(p['units'])
        assert m['completed_rounds_per_unit'] == 350 and m['checkpoint_replay_bitwise']
        assert m['features'] == p['feature_columns'] and len(m['features']) == 726
        assert m['units'] == p['units'] and m['model_parameters'] == p['model_parameters']
        assert m['maximum_decoded_target_year'] == args.year and m['environment'] == p['environment']
        assert sha(directory / 'protocol.json') == args.protocol_sha256
        for name, expected in m['artifacts'].items():
            assert sha(directory / name) == expected, name
        for name, expected in p['source'].items():
            assert sha(directory / 'source' / name) == expected, name
        seal_path = directory / 'fit_and_prediction_seal.json'
        assert sha(seal_path) == m['fit_and_prediction_seal_sha256']
        seal = json.loads(seal_path.read_text())
        assert seal['completed_units'] == 14 and set(seal['models']) == set(p['units'])
        assert seal['prediction_sha256'] == sha(directory / 'raw_predictions_before_selection.npy')
        assert m['timestamp_utc'] <= seal['timestamp_utc']
        if args.year == 2021:
            assert args.selection_lock is None and args.lock_audit is None
            lock_path = directory / 'selection_2021.json'
            assert sha(lock_path) == m['selection_lock_sha256']
            lock = json.loads(lock_path.read_text())
            weights = WEIGHTS
        else:
            assert args.selection_lock and args.lock_audit
            assert sha(args.selection_lock) == args.selection_lock_sha256 == m['selection_lock_sha256']
            assert sha(args.lock_audit) == args.lock_audit_sha256 == m['lock_audit_sha256']
            lock_path = args.selection_lock
            lock = json.loads(lock_path.read_text())
            audit = json.loads(args.lock_audit.read_text())
            assert audit['status'] == 'passed' and audit['eligible_for_2022'] is True
            assert audit['protocol_sha256'] == args.protocol_sha256
            assert audit['selection_lock_sha256'] == args.selection_lock_sha256
            assert audit['all14_native_checkpoints_replayed'] and audit['original_target_values_verified']
            assert lock['eligible_for_2022'] is True and lock['locked_utc'] <= m['timestamp_utc']
            weights = [0., lock['selected_weight']]
        check_lock(lock, p, args.protocol_sha256)
        if args.year == 2021:
            assert seal['timestamp_utc'] <= lock['locked_utc']
            assert lock['fit_and_prediction_seal_sha256'] == m['fit_and_prediction_seal_sha256']
            assert lock['raw_prediction_sha256'] == sha(directory / 'validation_2021.npz')

        metadata = pd.read_csv(ROOT / 'sintef_project/data/kernel/Unit_commitment_decisions.csv',
                               usecols=['Run No', 'starttime'])
        dates = pd.to_datetime(metadata.starttime, utc=True)
        rng = np.random.default_rng(4173)
        for year in [2021, 2022]:
            train_rows = np.flatnonzero(dates < pd.Timestamp(f'{year}-01-01', tz='UTC') - pd.Timedelta(days=6))
            hours = (train_rows[:, None] * 168 + np.arange(168)).ravel()
            expected_sample = np.sort(rng.choice(hours, int(len(hours) * .4), replace=False))
            if year == args.year:
                break
        sample_path = directory / f'train_sample_{args.year}.npy'
        sample = np.load(sample_path, allow_pickle=False)
        assert sample.dtype == expected_sample.dtype and sample.tobytes() == expected_sample.tobytes()
        assert sha(sample_path) == p['samples'][str(args.year)]['sha256'] == m['sample_sha256']
        assert len(sample) == m['sampled_training_rows'] == {2021: 146899, 2022: 171427}[args.year]
        assert metadata.iloc[train_rows]['Run No'].tolist() == m['train_ids'] == p['folds'][str(args.year)]['train_ids']
        selected = (dates >= pd.Timestamp(f'{args.year}-01-01', tz='UTC')) & (dates < pd.Timestamp(f'{args.year+1}-01-01', tz='UTC'))
        case_rows = np.flatnonzero(selected)
        validation = (case_rows[:, None] * 168 + np.arange(168)).ravel()
        ids, units = metadata.loc[selected, 'Run No'].to_numpy(), p['units']
        assert len(ids) == 365 and ids.tolist() == m['validation_ids'] == p['folds'][str(args.year)]['validation_ids']
        with np.load(directory / f'validation_{args.year}.npz', allow_pickle=False) as saved:
            raw, labels = saved['predictions'].copy(), saved['labels'].copy()
            assert saved['case_ids'].tolist() == ids.tolist() and saved['units'].tolist() == units
        assert raw.dtype == np.float32 and labels.dtype == np.uint8 and raw.shape == labels.shape == (61320, 14)
        assert np.isfinite(raw).all() and 0 <= raw.min() <= raw.max() <= 1 and np.isin(labels, [0, 1]).all()
        before = np.load(directory / 'raw_predictions_before_selection.npy', allow_pickle=False)
        assert before.dtype == raw.dtype and before.shape == raw.shape and before.tobytes() == raw.tobytes()
        columns = [f'result_committed_{unit}_t{hour}' for unit in units for hour in range(168)]
        target = pd.read_csv(ROOT / 'sintef_project/data/kernel/Unit_commitment_decisions.csv',
            skiprows=(np.flatnonzero(~selected.to_numpy()) + 1).tolist(), usecols=['Run No', *columns],
            dtype={column: 'uint8' for column in columns})
        assert target['Run No'].tolist() == ids.tolist()
        truth = target[columns].to_numpy().reshape(365, 14, 168).transpose(0, 2, 1).reshape(-1, 14)
        assert np.array_equal(labels, truth)

        report['memory_free_percent_before_build'] = runner.base.memory_free_percent()
        assert report['memory_free_percent_before_build'] >= 40, 'Large replay slot lacks declared headroom'
        loader = runner.import_path(ROOT / 'scripts/cuda_xgb664_loader.py', 'independent_river726_prefix')
        x, _, train, _, meta, _ = loader.build_verified_features(ROOT / original['stage'], args.year, loader_protocol)
        assert train['Run No'].tolist() == metadata['Run No'].tolist()
        prefix = prefix_fingerprint(x, original['feature_columns'])
        extension = runner.import_path(ROOT / 'sintef_project/hyljelihyl_river_features.py', 'independent_river726_extension')
        cache = p['cache']
        x, river_spec = extension.add_river_cost_features(x, ROOT, ROOT / cache['directory'],
            cache['manifest_sha256'], cache['protocol_sha256'], ROOT / cache['audit'], cache['audit_sha256'])
        assert prefix_fingerprint(x, original['feature_columns']) == prefix
        assert x.shape == (3647 * 168, 726) and x.columns.tolist() == p['feature_columns']
        meta['feature_spec']['river_cost'] = river_spec
        assert json.loads(json.dumps(meta['feature_spec'])) == m['feature_spec']
        xv = x.iloc[validation].copy()
        del x
        for index, unit in enumerate(units):
            checkpoint = directory / f'fold{args.year}_{unit}.txt'
            record = m['models'][unit]
            assert sha(checkpoint) == record['checkpoint_sha256'] == seal['models'][unit]
            native = dict(re.findall(r'^\[([^:\]]+): ([^\]]*)\]$', checkpoint.read_text(), re.MULTILINE))
            assert all(native[key] == value for key, value in NATIVE.items()), unit
            model = lgb.Booster(model_file=str(checkpoint))
            assert model.num_trees() == record['trees'] == 350 and model.num_feature() == record['features'] == 726
            assert model.feature_name() == [name.replace(' ', '_') for name in p['feature_columns']]
            prediction = model.predict(xv, num_threads=2)
            second = lgb.Booster(model_file=str(checkpoint)).predict(xv, num_threads=2)
            assert prediction.dtype == second.dtype == np.float64 and prediction.tobytes() == second.tobytes()
            assert prediction.astype(np.float32).tobytes() == raw[:, index].tobytes(), unit
            assert record['float64_checkpoint_replay_bitwise'] and record['replay_rows'] == 61320
            del model
        del xv

        references = {}
        for kind in ['anchor', 'control']:
            record = p[kind][str(args.year)]
            assert sha(ROOT / record['prediction']) == record['prediction_sha256']
            with np.load(ROOT / record['prediction'], allow_pickle=False) as saved:
                assert np.array_equal(saved['labels'], labels) and saved['case_ids'].tolist() == ids.tolist()
                assert saved['units'].tolist() == units
                references[kind] = saved['predictions'].astype(np.float64)
            assert references[kind].shape == labels.shape and np.isfinite(references[kind]).all()
            assert 0 <= references[kind].min() <= references[kind].max() <= 1
        assert np.array_equal(project(references['anchor'], units), references['anchor'])
        scores = {str(w): auc(labels, project((1-w) * references['anchor'] + w * raw.astype(np.float64), units)) for w in weights}
        assert set(scores) == set(m['scores'])
        for weight, value in scores.items():
            assert abs(value - m['scores'][weight]) < 1e-13
        chosen = max(weights, key=lambda w: (scores[str(w)], -w)) if args.year == 2021 else lock['selected_weight']
        assert chosen == lock['selected_weight'] == m['selected_weight']
        matched = auc(labels, project((1-chosen) * references['anchor'] + chosen * references['control'], units))
        eligible = chosen > 0 and scores[str(chosen)] > max(scores['0.0'], matched)
        assert abs(matched - m['control_micro_auc']) < 1e-13
        assert abs(scores[str(chosen)] - m['selected_micro_auc']) < 1e-13
        assert abs(scores['0.0'] - m['anchor_micro_auc']) < 1e-13
        assert abs(auc(labels, raw) - m['raw_micro_auc']) < 1e-13
        assert eligible == m['beats_anchor_and_control']
        if args.year == 2021:
            assert eligible == lock['eligible_for_2022']
            for weight, value in scores.items():
                assert abs(value - lock['scores_2021'][weight]) < 1e-13
            assert abs(matched - lock['control_2021_micro_auc']) < 1e-13
        else:
            assert eligible == m['accepted']
            assert m['meets_transfer_gate'] == (min(lock['selected_2021_micro_auc'], scores[str(chosen)]) >= .998)
        expected = project((1-chosen) * references['anchor'] + chosen * raw.astype(np.float64), units)
        with np.load(directory / f'selected_validation_{args.year}.npz', allow_pickle=False) as saved:
            assert saved['predictions'].dtype == np.float64 and saved['predictions'].shape == expected.shape
            assert np.array_equal(saved['labels'], labels) and saved['case_ids'].tolist() == ids.tolist()
            assert saved['units'].tolist() == units
            error = float(np.max(np.abs(saved['predictions'] - expected)))
            assert error <= 5e-16
        report.update(status='passed', protocol_sha256=args.protocol_sha256,
            candidate_manifest_sha256=args.candidate_manifest_sha256, training_audit_sha256=args.training_audit_sha256,
            selection_lock_sha256=sha(lock_path), eligible_for_2022=bool(eligible) if args.year == 2021 else None,
            accepted=bool(eligible) if args.year == 2022 else None, selected_weight=chosen,
            scores=scores, control_micro_auc=matched, all14_native_checkpoints_replayed=True,
            all14_independent_float64_replays_bitwise=True, original_target_values_verified=True,
            original664_prefix_bytes_verified=True, prefix_fingerprints=prefix,
            independent_projection_max_error=error, physical_cache_audit_sha256=cache['audit_sha256'],
            cache_manifest_sha256=cache['manifest_sha256'], cache_protocol_sha256=cache['protocol_sha256'],
            matched_raw664_control_audit_sha256=p['control'][str(args.year)]['audit_sha256'],
            source_review='Complete independently audited river cache, exact664 prefix, all14 native350-tree replays, original target values, independent rank AUC and isotonic projection; no fit or new weight.')
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        shutil.copy2(__file__, out / Path(__file__).name)
        report.update(runtime_seconds=time.monotonic() - started, peak_rss_bytes=runner.base.peak_rss(),
                      source_sha256=sha(Path(__file__)), independent_helper_sha256=HELPERS)
        (out / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({'status': report['status'], 'manifest_sha256': sha(out / 'manifest.json')}), flush=True)


if __name__ == '__main__':
    main()
