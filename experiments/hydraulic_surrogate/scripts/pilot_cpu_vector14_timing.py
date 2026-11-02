"""Bounded synthetic CPU timing and vector-tree prefix equivalence; no data fits."""
from __future__ import annotations
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[key] = '2'
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.smoke_cpu_vector14 import PACKAGES, ENVIRONMENT_SHA, PARAMETERS, require, sha

ROWS, FEATURES, TARGETS = 10000, 664, 14
ROUNDS = [64, 16]
MODEL_PARAMETERS = dict(PARAMETERS, max_depth=6)
MODEL_PARAMETERS.pop('n_estimators')


def environment() -> dict:
    """Verify the unchanged isolated wheel/native/runtime closure."""
    path = ROOT / 'artifacts/xgb341-cpu-environment-001/manifest.json'
    require(sha(path) == ENVIRONMENT_SHA, 'Environment manifest changed')
    prepared = json.loads(path.read_text())
    require(sys.version.split()[0] == '3.14.6' and platform.machine() == 'arm64', 'Pinned Python/architecture required')
    require(str(Path(sys.prefix).resolve()) == prepared['prefix'], 'Isolated environment changed')
    require(sha(Path(sys.executable).resolve()) == prepared['python_executable_sha256'], 'Python changed')
    for group in ['distribution_records', 'loaded_native_libraries']:
        for name, expected in prepared[group].items():
            require(sha(Path(name)) == expected, 'Installed runtime bytes changed: ' + name)
    require(sha(Path(xgb.build_info()['libxgboost'])) == prepared['xgboost_library_sha256'], 'Native XGB changed')
    packages = {name: importlib.metadata.version(name) for name in PACKAGES}
    require(packages == PACKAGES and xgb.build_info()['USE_CUDA'] is False, 'Package/CPU build changed')
    return {'python': sys.version, 'platform': platform.platform(), 'packages': packages,
            'build_info': xgb.build_info(), 'environment_manifest_sha256': ENVIRONMENT_SHA}


def validate_native(booster: xgb.Booster, saved: dict, rounds: int) -> dict:
    """Check actual shared-vector tree count, CPU recipe and all native tree depths."""
    config = json.loads(booster.save_config())
    learner = config['learner']; generic = learner['generic_param']
    require(generic['device'] == 'cpu' and int(generic['nthread']) == 2 and int(generic['seed']) == 4173,
            'CPU, thread or seed mismatch')
    require(learner['learner_train_param']['multi_strategy'] == 'multi_output_tree', 'Scalar strategy')
    require(learner['objective']['name'] == 'binary:logistic', 'Wrong objective')
    require('grow_quantile_histmaker' in json.dumps(learner['gradient_booster']), 'CPU histogram updater missing')
    params = learner['gradient_booster']['tree_train_param']
    for key in ['max_depth', 'learning_rate', 'min_child_weight', 'reg_lambda', 'reg_alpha',
                'max_bin', 'subsample', 'colsample_bytree', 'colsample_bylevel', 'colsample_bynode', 'max_delta_step']:
        require(abs(float(params[key]) - MODEL_PARAMETERS[key]) <= 1e-6, 'Native parameter mismatch: ' + key)
    require(params['grow_policy'] == 'depthwise', 'Growth policy changed')
    require(booster.num_features() == FEATURES and booster.num_boosted_rounds() == rounds, 'Actual dimensions differ')
    model = saved['learner']['gradient_booster']['model']
    require(int(model['gbtree_model_param']['num_trees']) == len(model['trees']) == rounds, 'Shared count differs')
    require(int(saved['learner']['learner_model_param']['num_target']) == TARGETS, 'Target count differs')
    for tree in model['trees']:
        require(int(tree['tree_param']['size_leaf_vector']) == TARGETS, 'Vector size differs')
        require(int(tree['tree_param']['num_nodes']) <= 127, 'Depth-6 node bound exceeded')
    return config


def compare_prefix(long: dict, short: dict) -> None:
    """Require every native first-16 tree payload and base-score to be identical."""
    a, b = long['learner'], short['learner']
    require(a['learner_model_param'] == b['learner_model_param'], 'Base score/model schema differs')
    require(a['objective'] == b['objective'], 'Objective differs')
    af, bf = a['gradient_booster']['model'], b['gradient_booster']['model']
    require(len(af['trees']) == 64 and len(bf['trees']) == 16, 'Prefix comparison counts differ')
    require(af['trees'][:16] == bf['trees'], 'Native first-16 tree payloads differ')
    require(af['tree_info'][:16] == bf['tree_info'], 'Tree groups differ')
    require(af['iteration_indptr'][:17] == bf['iteration_indptr'], 'Round boundaries differ')


def phase(name: str, out: Path, protocol: dict, fit_sha: str | None) -> None:
    """Run fixed fits or fresh-process replay, preserving failures and phase timings."""
    path = out / f'{name}_manifest.json'
    require(not path.exists(), 'Immutable phase already exists')
    tick = time.monotonic()
    report = {'status': 'initializing', 'phase': name, 'pid': os.getpid(), 'synthetic_only': True,
              'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'competition_data_read': False,
              'parameters': MODEL_PARAMETERS, 'rounds': ROUNDS, 'source_sha256': sha(Path(__file__))}
    try:
        report['environment'] = environment()
        rng = np.random.default_rng(4173)
        expected_x = rng.standard_normal((ROWS, FEATURES), dtype=np.float32)
        expected_y = (expected_x[:, :14] + .2 * expected_x[:, 14:28]
                      + .05 * rng.standard_normal((ROWS, TARGETS)) > 0).astype(np.uint8)
        if name == 'fit':
            x, y = expected_x, expected_y
            np.save(out / 'synthetic_x.npy', x, allow_pickle=False)
            np.save(out / 'synthetic_y.npy', y, allow_pickle=False)
            models, payloads, times = {}, {}, {}
            matrix = xgb.DMatrix(x, nthread=2)
            for count in ROUNDS:
                model = xgb.XGBClassifier(n_estimators=count, **MODEL_PARAMETERS)
                started = time.monotonic(); model.fit(x, y)
                times[str(count)] = time.monotonic() - started
                booster = model.get_booster(); models[count] = booster
                for extension in ['json', 'ubj']:
                    booster.save_model(out / f'model_{count}.{extension}')
                saved = json.loads((out / f'model_{count}.json').read_text()); payloads[count] = saved
                config = validate_native(booster, saved, count)
                (out / f'config_{count}.json').write_text(json.dumps(config, indent=2) + '\n')
                (out / f'parameters_{count}.json').write_text(json.dumps(model.get_params(), indent=2) + '\n')
                prediction = booster.predict(matrix)
                require(prediction.shape == (ROWS, TARGETS) and prediction.dtype == np.float32 and
                        np.isfinite(prediction).all() and 0 <= prediction.min() <= prediction.max() <= 1,
                        'Invalid prediction schema/range')
                np.save(out / f'predictions_{count}.npy', prediction, allow_pickle=False)
                print(json.dumps({'rounds': count, 'fit_seconds': times[str(count)]}), flush=True)
            started = time.monotonic()
            compare_prefix(payloads[64], payloads[16])
            prefix = models[64].predict(matrix, iteration_range=(0, 16))
            short = np.load(out / 'predictions_16.npy', allow_pickle=False)
            require(prefix.dtype == short.dtype and prefix.tobytes() == short.tobytes(), 'Prefix predictions differ')
            np.save(out / 'prefix_16_predictions.npy', prefix, allow_pickle=False)
            report.update(fit_seconds=times, prefix_compare_seconds=time.monotonic() - started,
                          prefix_native_trees_and_base_score_exact=True, prefix_predictions_bitwise=True)
        else:
            require(fit_sha is not None and sha(out / 'fit_manifest.json') == fit_sha, 'External fit hash required')
            fit = json.loads((out / 'fit_manifest.json').read_text())
            require(fit['status'] == 'complete' and fit['pid'] != os.getpid(), 'Separate completed fit process required')
            require(fit['environment'] == report['environment'], 'Runtime changed across processes')
            for file, expected in fit['artifacts'].items():
                require(sha(out / file) == expected, 'Fit artifact changed: ' + file)
            x = np.load(out / 'synthetic_x.npy', allow_pickle=False)
            y = np.load(out / 'synthetic_y.npy', allow_pickle=False)
            require(x.tobytes() == expected_x.tobytes() and y.tobytes() == expected_y.tobytes(), 'Synthetic regeneration differs')
            matrix = xgb.DMatrix(x, nthread=2); checks = []
            for count in ROUNDS:
                saved = json.loads((out / f'model_{count}.json').read_text())
                original = np.load(out / f'predictions_{count}.npy', allow_pickle=False)
                for extension in ['ubj', 'json']:
                    booster = xgb.Booster(); booster.load_model(out / f'model_{count}.{extension}')
                    booster.set_param({'device': 'cpu', 'nthread': 2})
                    require(booster.num_boosted_rounds() == count and booster.num_features() == FEATURES, 'Replay schema differs')
                    replay = booster.predict(matrix)
                    require(replay.dtype == original.dtype and replay.tobytes() == original.tobytes(), 'Serialized replay differs')
                    if count == 64:
                        prefix = booster.predict(matrix, iteration_range=(0, 16))
                        require(prefix.tobytes() == np.load(out / 'predictions_16.npy', allow_pickle=False).tobytes(), 'Replay prefix differs')
                    np.save(out / f'replay_{count}_{extension}.npy', replay, allow_pickle=False)
                    checks.append({'rounds': count, 'format': extension, 'bitwise_replay': True})
            compare_prefix(json.loads((out / 'model_64.json').read_text()), json.loads((out / 'model_16.json').read_text()))
            report.update(fit_pid=fit['pid'], fit_manifest_sha256=fit_sha, serialization_checks=checks,
                          synthetic_regeneration_bitwise=True, prefix_predictions_bitwise=True,
                          prefix_native_trees_and_base_score_exact=True)
        report['status'] = 'complete'
    except BaseException:
        report.update(status='failed', error=traceback.format_exc()); raise
    finally:
        report.update(runtime_seconds=time.monotonic() - tick,
                      peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                      artifacts={p.name: sha(p) for p in out.iterdir() if p.is_file() and
                                 (p.suffix in ['.npy', '.ubj'] or p.name.startswith(('model_', 'config_', 'parameters_')))})
        path.write_text(json.dumps(report, indent=2) + '\n')


def main() -> None:
    """Enforce one combined 120-second budget for both fits and fresh replay."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--protocol', type=Path, required=True); ap.add_argument('--protocol-sha256', required=True)
    ap.add_argument('--run-id', required=True); ap.add_argument('--phase', choices=['fit', 'replay'])
    ap.add_argument('--fit-manifest-sha256'); args = ap.parse_args()
    require(Path(args.run_id).name == args.run_id and args.run_id not in ['', '.', '..'], 'Invalid run ID')
    require(sha(args.protocol) == args.protocol_sha256, 'Protocol bytes changed')
    protocol = json.loads(args.protocol.read_text())
    for name, expected in protocol['source'].items():
        require(sha(ROOT / name) == expected, 'Pinned source changed: ' + name)
    require(protocol['parameters'] == MODEL_PARAMETERS and protocol['rounds'] == ROUNDS and
            protocol['shape'] == [ROWS, FEATURES] and protocol['target_shape'] == [ROWS, TARGETS] and
            protocol['combined_timeout_seconds'] == 120, 'Declared experiment changed')
    out = ROOT / 'artifacts' / args.run_id
    if args.phase:
        require(out.is_dir(), 'Parent must create output namespace')
        phase(args.phase, out, protocol, args.fit_manifest_sha256); return
    out.mkdir(exist_ok=False); start = time.monotonic()
    report = {'status': 'initializing', 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'command': [sys.executable, *sys.argv], 'protocol_sha256': args.protocol_sha256,
              'synthetic_only': True, 'competition_data_read': False, 'full_fold_authorized': False,
              'gpu_capability_tested': False, 'phases': [], 'combined_timeout_seconds': 120}
    try:
        environment()
        for name in ['fit', 'replay']:
            command = [sys.executable, str(Path(__file__).resolve()), '--protocol', str(args.protocol.resolve()),
                       '--protocol-sha256', args.protocol_sha256, '--run-id', args.run_id, '--phase', name]
            if name == 'replay': command += ['--fit-manifest-sha256', sha(out / 'fit_manifest.json')]
            tick = time.monotonic()
            with (out / f'{name}.log').open('w') as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                try:
                    process.wait(timeout=max(.01, 120 - (time.monotonic() - start)))
                except BaseException:
                    process.terminate()
                    try: process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill(); process.wait()
                    raise
            report['phases'].append({'phase': name, 'pid': process.pid, 'exitcode': process.returncode,
                                     'runtime_seconds': time.monotonic() - tick, 'command': command})
            require(process.returncode == 0, name + ' failed; see preserved log')
            result = json.loads((out / f'{name}_manifest.json').read_text())
            require(result['status'] == 'complete' and result['peak_rss_bytes'] <= 2 * 1024**3,
                    'Phase failed or post-exit OS memory threshold exceeded')
            for file, expected in result['artifacts'].items(): require(sha(out / file) == expected, 'Payload changed')
        fit = json.loads((out / 'fit_manifest.json').read_text()); replay = json.loads((out / 'replay_manifest.json').read_text())
        require(time.monotonic() - start <= 120, 'Combined phase budget exceeded')
        report.update(status='passed', fit_manifest_sha256=sha(out / 'fit_manifest.json'),
                      replay_manifest_sha256=sha(out / 'replay_manifest.json'), fit_seconds=fit['fit_seconds'],
                      prefix_compare_seconds=fit['prefix_compare_seconds'],
                      prefix_native_trees_and_base_score_exact=True, prefix_predictions_bitwise=True,
                      four_serialized_replays_bitwise=True,
                      maximum_worker_peak_rss_bytes=max(fit['peak_rss_bytes'], replay['peak_rss_bytes']),
                      memory_contract='Post-exit OS peak RSS <=2GiB; no live memory-enforcement claim.',
                      interpretation='Synthetic CPU timing and exact 16-of-64 prefix only; actual 600/1800 cost, prefix equivalence and competition accuracy remain unmeasured.')
    except BaseException:
        report.update(status='failed', error=traceback.format_exc()); raise
    finally:
        shutil.copy2(args.protocol, out / 'protocol.json'); shutil.copy2(__file__, out / Path(__file__).name)
        report.update(runtime_seconds=time.monotonic() - start, source_sha256=sha(Path(__file__)),
                      artifacts={p.name: sha(p) for p in out.iterdir() if p.is_file() and p.name != 'manifest.json'})
        (out / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({'status': report['status'], 'manifest_sha256': sha(out / 'manifest.json')}), flush=True)


if __name__ == '__main__': main()
