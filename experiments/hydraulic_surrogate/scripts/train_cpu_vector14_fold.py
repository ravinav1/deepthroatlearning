"""Prospective isolated CPU vector14 fit/replay with explicit resource limits."""
from __future__ import annotations
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']: os.environ[key] = '2'
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import re
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
import xgboost as xgb
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.cpu_vector14_fold_common import read_protocol, review, validate_lock, verify_files, artifacts, sha
from scripts.cpu_vector14_portability import verify_dmatrix_blocks
from scripts.pilot_cpu_vector14_timing import environment, validate_native


def rss() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def tree_prefix(long: dict, short: dict, count: int) -> None:
    """Verify saved600 actual trees/base score exactly equal the1800 prefix."""
    a, b = long['learner'], short['learner']
    assert a['learner_model_param'] == b['learner_model_param'] and a['objective'] == b['objective']
    af, bf = a['gradient_booster']['model'], b['gradient_booster']['model']
    assert len(bf['trees']) == count and af['trees'][:count] == bf['trees']
    assert af['tree_info'][:count] == bf['tree_info'] and af['iteration_indptr'][:count + 1] == bf['iteration_indptr']


def worker(a, p: dict, out: Path, bm: dict, rounds: list[int]) -> None:
    """Fit without decoding validation targets, or reload both native formats."""
    start = time.monotonic(); path = out / f'{a.phase}_manifest.json'; assert not path.exists()
    m = {'status': 'initializing', 'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'phase': a.phase,
         'pid': os.getpid(), 'year': a.year, 'rounds': rounds, 'protocol_sha256': a.protocol_sha256,
         'bridge_manifest_sha256': a.bridge_manifest_sha256, 'validation_targets_decoded': False}
    try:
        m['environment'] = environment()
        pressure = subprocess.run(['memory_pressure', '-Q'], capture_output=True, text=True, check=True)
        match = re.search(r'System-wide memory free percentage:\s*(\d+)%', pressure.stdout)
        assert match is not None, 'Cannot verify host memory headroom'
        m['memory_free_percent_before_phase'] = int(match.group(1))
        assert m['memory_free_percent_before_phase'] >= p['resources']['minimum_free_percent_before_fit']
        schema = json.loads((a.bridge / 'schema.json').read_text()); assert schema['feature_columns'] == p['feature_columns']
        xv = np.load(a.bridge / 'x_validation.npy', mmap_mode='r', allow_pickle=False)
        assert xv.shape == (61320, 664)
        m['validation_native_input_check'] = verify_dmatrix_blocks(xv, p['feature_columns'], xgb)
        assert m['validation_native_input_check']['blocks'] == bm['records']['validation']['native']['blocks']
        if a.phase == 'fit':
            xt = np.load(a.bridge / 'x_train.npy', mmap_mode='r', allow_pickle=False)
            yt = np.load(a.bridge / 'y_train.npy', allow_pickle=False)
            assert xt.shape == (p['sample_rows'][str(a.year)], 664) and yt.shape == (len(xt), 14)
            assert yt.dtype == np.uint8 and np.isin(yt, [0, 1]).all()
            m['training_native_input_check'] = verify_dmatrix_blocks(xt, p['feature_columns'], xgb)
            assert m['training_native_input_check']['blocks'] == bm['records']['train']['native']['blocks']
            assert rss() <= p['resources']['maximum_fit_peak_rss_bytes']
            class Guard(xgb.callback.TrainingCallback):
                def after_iteration(self, model, epoch, evals_log):
                    assert rss() <= p['resources']['maximum_fit_peak_rss_bytes'], 'OS peak RSS exceeded prospective cap'
                    assert time.monotonic() - start <= p['resources']['fit_timeout_seconds'], 'Worker time cap'
                    if epoch + 1 == 600:
                        model.save_model(out / 'trajectory_at_600.json')
                    if (epoch + 1) % 20 == 0:
                        record = {'rounds': epoch + 1, 'seconds': time.monotonic() - start, 'peak_rss_bytes': rss()}
                        (out / 'progress.json').write_text(json.dumps(record) + '\n'); print(json.dumps(record), flush=True)
                    return False
            model = xgb.XGBClassifier(n_estimators=max(rounds), callbacks=[Guard()], **p['model_parameters'])
            tick = time.monotonic(); model.fit(xt, yt); m['fit_seconds'] = time.monotonic() - tick
            booster = model.get_booster(); booster.feature_names = p['feature_columns']
            matrix = xgb.DMatrix(xv, feature_names=p['feature_columns'], nthread=2)
            for count in rounds:
                part = booster if count == max(rounds) else booster[:count]
                for extension in ['ubj', 'json']: part.save_model(out / f'model_{count}.{extension}')
                saved = json.loads((out / f'model_{count}.json').read_text())
                # A sliced booster has no training updater state; validate the original
                # trained config, and verify the sliced native structure separately.
                config = json.loads(booster.save_config())
                validate_native(booster, json.loads((out / f'model_{max(rounds)}.json').read_text()) if (out / f'model_{max(rounds)}.json').exists() else
                    json.loads(booster.save_raw(raw_format='json')), max(rounds))
                assert len(saved['learner']['gradient_booster']['model']['trees']) == count
                assert all(int(t['tree_param']['size_leaf_vector']) == 14 for t in saved['learner']['gradient_booster']['model']['trees'])
                prediction = part.predict(matrix)
                assert prediction.dtype == np.float32 and prediction.shape == (61320, 14) and np.isfinite(prediction).all()
                assert 0 <= prediction.min() <= prediction.max() <= 1
                original = booster.predict(matrix, iteration_range=(0, count))
                assert prediction.tobytes() == original.tobytes()
                np.save(out / f'predictions_{count}.npy', prediction, allow_pickle=False)
                (out / f'fit_config_{count}.json').write_text(json.dumps(config, indent=2) + '\n')
            full = json.loads((out / f'model_{max(rounds)}.json').read_text())
            if max(rounds) >= 600:
                snapshot = json.loads((out / 'trajectory_at_600.json').read_text()); tree_prefix(full, snapshot, 600)
            if rounds == [600, 1800]: tree_prefix(full, json.loads((out / 'model_600.json').read_text()), 600)
            (out / 'expanded_parameters.json').write_text(json.dumps({k: v for k, v in model.get_params().items() if k != 'callbacks'}, indent=2) + '\n')
            m.update(native_shared_vector_trees_verified=True, actual_600_snapshot_equals_final_prefix=True,
                     all_sliced_and_iteration_range_predictions_bitwise=True, parameters=p['model_parameters'])
        else:
            assert a.fit_manifest_sha256 and sha(out / 'fit_manifest.json') == a.fit_manifest_sha256
            fit = json.loads((out / 'fit_manifest.json').read_text()); assert fit['status'] == 'complete' and fit['pid'] != os.getpid()
            verify_files(out, fit['artifacts']); assert fit['environment'] == m['environment']
            matrix = xgb.DMatrix(xv, feature_names=p['feature_columns'], nthread=2)
            for count in rounds:
                expected = np.load(out / f'predictions_{count}.npy', allow_pickle=False)
                for extension in ['ubj', 'json']:
                    model = xgb.Booster(); model.load_model(out / f'model_{count}.{extension}')
                    model.set_param({'device': 'cpu', 'nthread': 2})
                    assert model.num_boosted_rounds() == count and model.num_features() == 664
                    actual = model.predict(matrix)
                    assert actual.dtype == expected.dtype and actual.tobytes() == expected.tobytes()
                    np.save(out / f'replay_{count}_{extension}.npy', actual, allow_pickle=False)
                    assert rss() <= p['resources']['maximum_fit_peak_rss_bytes']
            m.update(fit_manifest_sha256=a.fit_manifest_sha256, fit_pid=fit['pid'], all_saved_models_replayed_bitwise=True)
        m['status'] = 'complete'
    except BaseException:
        m.update(status='failed', error=traceback.format_exc()); raise
    finally:
        m.update(runtime_seconds=time.monotonic() - start, peak_rss_bytes=rss(), source_sha256=sha(Path(__file__)),
                 artifacts={q.name: sha(q) for q in out.iterdir() if q.is_file() and q.name.startswith(('model_', 'predictions_', 'replay_', 'fit_config_', 'trajectory_', 'expanded_'))})
        path.write_text(json.dumps(m, indent=2) + '\n')


def main() -> None:
    """Require explicit launch, reviewed recipe, actual bridge and conditional2022 lock."""
    ap = argparse.ArgumentParser()
    for name in ['protocol', 'bridge', 'independent-audit']: ap.add_argument('--' + name, type=Path, required=True)
    for name in ['protocol-sha256', 'bridge-manifest-sha256', 'independent-audit-sha256', 'run-id']: ap.add_argument('--' + name, required=True)
    for name in ['selection-lock', 'lock-audit']: ap.add_argument('--' + name, type=Path)
    for name in ['selection-lock-sha256', 'lock-audit-sha256', 'fit-manifest-sha256']: ap.add_argument('--' + name)
    ap.add_argument('--year', type=int, choices=[2021, 2022], required=True); ap.add_argument('--execute', action='store_true')
    ap.add_argument('--phase', choices=['fit', 'replay']); a = ap.parse_args()
    assert a.execute and Path(a.run_id).name == a.run_id and a.run_id not in ['', '.', '..']
    p = read_protocol(a.protocol, a.protocol_sha256); review(a.independent_audit, a.independent_audit_sha256, a.protocol_sha256)
    rounds = [600, 1800]; lock = None
    if a.year == 2022:
        lock = validate_lock(a.selection_lock, a.selection_lock_sha256, p, a.protocol_sha256, a.lock_audit, a.lock_audit_sha256)
        rounds = [lock['selected_rounds']]
    else: assert a.selection_lock is None and a.lock_audit is None
    assert sha(a.bridge / 'manifest.json') == a.bridge_manifest_sha256
    bm = json.loads((a.bridge / 'manifest.json').read_text())
    assert bm['status'] == 'complete' and bm['year'] == a.year and bm['protocol_sha256'] == a.protocol_sha256
    assert bm['exact_original_sample_bytes'] and bm['all_dataframe_native_model_inputs_bitwise']
    if lock is not None: assert bm['selection_lock_sha256'] == a.selection_lock_sha256
    verify_files(a.bridge, bm['artifacts'])
    out = ROOT / 'artifacts' / a.run_id
    if a.phase: worker(a, p, out, bm, rounds); return
    out.mkdir(exist_ok=False); start = time.monotonic()
    report = {'status': 'initializing', 'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'year': a.year,
              'command': [sys.executable, *sys.argv], 'protocol_sha256': a.protocol_sha256,
              'bridge': str(a.bridge), 'bridge_manifest_sha256': a.bridge_manifest_sha256, 'rounds': rounds,
              'selection_lock_sha256': a.selection_lock_sha256, 'validation_targets_decoded': False, 'phases': []}
    try:
        for phase in ['fit', 'replay']:
            command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--phase', phase]
            if phase == 'replay': command += ['--fit-manifest-sha256', sha(out / 'fit_manifest.json')]
            limit = p['resources']['fit_timeout_seconds'] if phase == 'fit' else p['resources']['replay_timeout_seconds']
            tick = time.monotonic()
            with (out / f'{phase}.log').open('w') as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                try:
                    while child.poll() is None:
                        if time.monotonic() - tick > limit: raise TimeoutError(phase + ' wall cap')
                        time.sleep(1)
                except BaseException:
                    child.terminate()
                    try: child.wait(timeout=5)
                    except subprocess.TimeoutExpired: child.kill(); child.wait()
                    raise
            report['phases'].append({'phase': phase, 'pid': child.pid, 'exitcode': child.returncode, 'seconds': time.monotonic() - tick})
            assert child.returncode == 0, phase + ' failed; no lower-capacity fallback'
            pm = json.loads((out / f'{phase}_manifest.json').read_text()); assert pm['status'] == 'complete'
            assert pm['peak_rss_bytes'] <= p['resources']['maximum_fit_peak_rss_bytes']
        report.update(status='complete', fit_manifest_sha256=sha(out / 'fit_manifest.json'), replay_manifest_sha256=sha(out / 'replay_manifest.json'),
                      all_models_replayed_bitwise=True, no_validation_target_decode=True)
    except BaseException:
        report.update(status='failed', error=traceback.format_exc()); raise
    finally:
        shutil.copy2(__file__, out / Path(__file__).name); shutil.copy2(a.protocol, out / 'protocol.json')
        report.update(runtime_seconds=time.monotonic() - start, artifacts=artifacts(out))
        (out / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n'); print(report['status'], sha(out / 'manifest.json'), flush=True)


if __name__ == '__main__': main()
