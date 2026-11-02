"""Bound and preserve the authorized isolated synthetic CPU vector-leaf smoke."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--protocol', type=Path, required=True)
    ap.add_argument('--protocol-sha256', required=True)
    ap.add_argument('--run-id', required=True)
    args = ap.parse_args()
    assert Path(args.run_id).name == args.run_id and args.run_id not in ['', '.', '..']
    out = ROOT / 'artifacts' / args.run_id
    out.mkdir(exist_ok=False)
    start = time.monotonic()
    report = {'status': 'initializing', 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'command': [sys.executable, *sys.argv], 'synthetic_only': True,
              'competition_data_read': False, 'gpu_capability_tested': False,
              'future_fold_authorized': False, 'phases': []}
    try:
        assert sha(args.protocol) == args.protocol_sha256
        p = json.loads(args.protocol.read_text())
        for name, expected in p['source'].items():
            assert sha(ROOT / name) == expected, name
        assert sha(ROOT / p['environment_manifest']) == p['environment_manifest_sha256']
        environment = json.loads((ROOT / p['environment_manifest']).read_text())
        assert str(Path(sys.prefix).resolve()) == environment['prefix']
        assert p['phase_timeout_seconds'] == 120 and p['maximum_phase_rss_bytes'] == 2 * 1024**3
        for phase in ['fit', 'replay']:
            command = [sys.executable, str(ROOT / 'scripts/smoke_cpu_vector14.py'),
                       '--phase', phase, '--directory', str(out)]
            if phase == 'replay':
                command += ['--fit-manifest-sha256', sha(out / 'fit_manifest.json')]
            tick = time.monotonic()
            maximum_rss = 0
            with (out / f'{phase}.log').open('w') as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                try:
                    while process.poll() is None:
                        elapsed = time.monotonic() - tick
                        assert elapsed <= 120, 'Synthetic phase timeout'
                        reading = subprocess.run(['ps', '-p', str(process.pid), '-o', 'rss='],
                                                 capture_output=True, text=True)
                        if reading.returncode == 0 and reading.stdout.strip():
                            rss = int(reading.stdout.strip()) * 1024
                            maximum_rss = max(maximum_rss, rss)
                            assert rss <= p['maximum_phase_rss_bytes'], 'Synthetic phase RSS cap'
                        time.sleep(.25)
                except BaseException:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise
            phase_record = {'phase': phase, 'command': command, 'pid': process.pid,
                            'exitcode': process.returncode, 'runtime_seconds': time.monotonic() - tick,
                            'maximum_sampled_rss_bytes': maximum_rss}
            report['phases'].append(phase_record)
            assert process.returncode == 0, phase + ' failed; see preserved log'
            manifest = json.loads((out / f'{phase}_manifest.json').read_text())
            assert manifest['status'] == 'complete' and manifest['parameters'] == p['model_parameters']
            assert manifest['source_sha256'] == p['source']['scripts/smoke_cpu_vector14.py']
            assert manifest['runtime_seconds'] <= 120 and manifest['peak_rss_bytes'] <= p['maximum_phase_rss_bytes']
            for name, expected in manifest['artifacts'].items():
                assert sha(out / name) == expected, name
            print(json.dumps({'phase': phase, 'status': 'complete', 'seconds': phase_record['runtime_seconds']}), flush=True)
        fit = json.loads((out / 'fit_manifest.json').read_text())
        replay = json.loads((out / 'replay_manifest.json').read_text())
        assert fit['pid'] != replay['pid'] and replay['fit_pid'] == fit['pid']
        assert replay['ubj_and_json_replays_bitwise'] and replay['exact_replay_rows'] == 2048
        # This previously reviewed parser is device-independent and does not
        # load XGBoost. It traverses the eight JSON vector trees independently.
        from scripts.verify_cuda_vector14_smoke_return import json_forest_prediction
        import numpy as np
        x = np.load(out / 'synthetic_x.npy', allow_pickle=False)
        y = np.load(out / 'synthetic_y.npy', allow_pickle=False)
        rng = np.random.default_rng(4173)
        expected_x = rng.standard_normal((2048, 664), dtype=np.float32)
        expected_y = (expected_x[:, :14] + .2 * expected_x[:, 14:28]
                      + .05 * rng.standard_normal((2048, 14)) > 0).astype(np.uint8)
        assert x.tobytes() == expected_x.tobytes() and y.tobytes() == expected_y.tobytes()
        prediction = np.load(out / 'fit_predictions.npy', allow_pickle=False)
        independent = json_forest_prediction(json.loads((out / 'model.json').read_text()), x)
        error = float(np.max(np.abs(independent - prediction)))
        assert error <= 3e-7
        report.update(status='passed', protocol_sha256=args.protocol_sha256,
                      environment_manifest_sha256=p['environment_manifest_sha256'],
                      fit_manifest_sha256=sha(out / 'fit_manifest.json'),
                      replay_manifest_sha256=sha(out / 'replay_manifest.json'),
                      actual_shared_tree_count=8, vector_leaf_size=14,
                      ubj_and_json_replays_bitwise=True, independent_json_prediction_max_error=error,
                      synthetic_regeneration_bitwise=True,
                      maximum_worker_peak_rss_bytes=max(fit['peak_rss_bytes'], replay['peak_rss_bytes']),
                      interpretation='CPU binary:logistic14-output vector-leaf capability and serialization only; no GPU, competition or full-fold performance claim.')
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        shutil.copy2(args.protocol, out / 'protocol.json')
        for name in ['scripts/run_cpu_vector14_smoke.py', 'scripts/smoke_cpu_vector14.py']:
            shutil.copy2(ROOT / name, out / Path(name).name)
        report.update(runtime_seconds=time.monotonic() - start, source_sha256=sha(Path(__file__)),
                      artifacts={path.name: sha(path) for path in out.iterdir()
                                 if path.is_file() and path.name != 'manifest.json'})
        (out / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({'status': report['status'], 'manifest_sha256': sha(out / 'manifest.json')}), flush=True)


if __name__ == '__main__':
    main()
