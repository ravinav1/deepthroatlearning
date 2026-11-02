"""Independent source/contract review using saved synthetic CPU vector artifacts."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCE = 'scripts/audit_cpu_vector14_fold.py'
SOURCE_SHA = 'e5c64878020eb325fc38d7afc8066a2f89a104a8666fb7e63bdc1d0889ea8340'
PROTOCOL = 'docs/cpu_vector14_fold_protocol_v2.json'
PROTOCOL_SHA = 'e4deebd4d7f9ac666c8da6c97e82f60f7ba55630816f2e7cf3e56008061ba121'
PREFLIGHT = 'artifacts/cpu-vector14-actual-auditor-preflight-003/manifest.json'
PREFLIGHT_SHA = '0574a6a530bc1e022839319aded3a9355533ed8f31feb482b3b4dc7a7843aa12'


def sha(path: Path) -> str:
    """Hash one immutable local file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Review the exact auditor without competition arrays or native model imports."""
    out = ROOT / 'artifacts/cpu-vector14-actual-auditor-independent-source-review-001'
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    report = {'status': 'initializing', 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'command': [sys.executable, *sys.argv], 'source_sha256': SOURCE_SHA,
              'protocol_sha256': PROTOCOL_SHA, 'preflight_manifest_sha256': PREFLIGHT_SHA,
              'training': False, 'competition_features_read': False,
              'competition_targets_decoded': False}
    try:
        for name, expected in [(SOURCE, SOURCE_SHA), (PROTOCOL, PROTOCOL_SHA), (PREFLIGHT, PREFLIGHT_SHA)]:
            assert sha(ROOT / name) == expected, name
        p = json.loads((ROOT / PROTOCOL).read_text())
        for group in ['source', 'inputs', 'data']:
            for name, expected in p[group].items():
                assert sha(ROOT / name) == expected, name
        prepared = json.loads((ROOT / PREFLIGHT).read_text())
        for name, expected in prepared['artifacts'].items():
            assert sha((ROOT / PREFLIGHT).parent / name) == expected
        from scripts.audit_cpu_vector14_fold import forest_prediction, native_config

        command = [sys.executable, '-m', 'unittest', 'tests.test_cpu_vector14_fold_audit']
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        (out / 'tests.log').write_text(result.stdout + result.stderr)
        assert result.returncode == 0
        report.update(tests_command=command, tests_exit_code=result.returncode)

        synthetic = ROOT / 'artifacts/cpu-vector14-timing-001'
        fit = json.loads((synthetic / 'fit_manifest.json').read_text())
        for name, expected in fit['artifacts'].items():
            assert sha(synthetic / name) == expected
        x = np.load(synthetic / 'synthetic_x.npy', allow_pickle=False)
        regenerated = np.random.default_rng(4173).standard_normal((10000, 664), dtype=np.float32)
        assert x.shape == regenerated.shape and x.dtype == regenerated.dtype and x.tobytes() == regenerated.tobytes()
        checked = []
        for rounds in [16, 64]:
            model = json.loads((synthetic / f'model_{rounds}.json').read_text())
            config = json.loads((synthetic / f'config_{rounds}.json').read_text())
            native_config(config, p, rounds)
            actual = forest_prediction(model, x, rounds)
            expected = np.load(synthetic / f'predictions_{rounds}.npy', allow_pickle=False)
            assert actual.dtype == expected.dtype and actual.shape == expected.shape and actual.tobytes() == expected.tobytes()
            checked.append({'rounds': rounds, 'rows': len(x), 'outputs': 14, 'independent_json_max_error': 0.0})
        rejected = []
        changes = [
            ('device', ['learner', 'generic_param', 'device'], 'cuda:0'),
            ('threads', ['learner', 'generic_param', 'nthread'], '3'),
            ('depth', ['learner', 'gradient_booster', 'tree_train_param', 'max_depth'], '7'),
            ('bins', ['learner', 'gradient_booster', 'tree_train_param', 'max_bin'], '256'),
            ('sampling', ['learner', 'gradient_booster', 'tree_train_param', 'sampling_method'], 'gradient_based'),
            ('constraints', ['learner', 'gradient_booster', 'tree_train_param', 'monotone_constraints'], '(1)'),
            ('strategy', ['learner', 'learner_train_param', 'multi_strategy'], 'one_output_per_tree'),
            ('parallel_tree', ['learner', 'gradient_booster', 'gbtree_model_param', 'num_parallel_tree'], '2'),
        ]
        for label, path, value in changes:
            altered = copy.deepcopy(config)
            node = altered
            for key in path[:-1]:
                node = node[key]
            node[path[-1]] = value
            try:
                native_config(altered, p, 64)
            except AssertionError:
                rejected.append(label)
            else:
                raise AssertionError('Changed native recipe accepted: ' + label)
        report.update(status='passed', synthetic_models_checked=checked,
                      native_config_corruptions_rejected=rejected,
                      original_timing_pilot_status_preserved=json.loads((synthetic / 'manifest.json').read_text())['status'],
                      review=[
                          'Exact original664 reconstruction, finite DataFrame/native dense CSR byte bridge, original purged sample and target dtype/shape/bytes.',
                          'Bridge recorded environment/build RSS and actual externally observed process exits are bound before acceptance.',
                          'Fit and replay are distinct exited PIDs in pinned isolated3.4.1; actual bridge protocol/year/lock and independent bridge proof are bound.',
                          'Exact600/1800 or audited locked-round scope, fixed native recipe, true600 trajectory prefix, both native-format replay arrays and independent float32 JSON arithmetic.',
                          'No validation targets decoded by the fit path; selector still requires actual independent fit proof before scores.',
                      ], limitations='Source approval only; no actual bridge, competition fit, fold score or runtime eligibility is established.')
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        for name in [SOURCE, 'tests/test_cpu_vector14_fold_audit.py', PROTOCOL]:
            dest = out / 'source' / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, dest)
        shutil.copy2(__file__, out / Path(__file__).name)
        report.update(runtime_seconds=time.monotonic() - started,
                      environment={'python': sys.version, 'platform': platform.platform(), 'numpy': np.__version__},
                      artifacts={str(path.relative_to(out)): sha(path) for path in out.rglob('*') if path.is_file()})
        (out / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
        print(report['status'], sha(out / 'manifest.json'), flush=True)


if __name__ == '__main__':
    main()
