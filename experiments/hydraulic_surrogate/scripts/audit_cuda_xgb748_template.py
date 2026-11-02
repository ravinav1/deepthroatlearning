"""Check unsealed integer-feature templates without opening partial cache data."""
from datetime import datetime, timezone
import argparse
import ast
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    """Hash a frozen source or small proof."""
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main() -> None:
    """Review source/test pins while keeping the actual cache-completion gate closed."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    assert Path(args.run_id).name == args.run_id and args.run_id not in ('', '.', '..')
    out = ROOT / 'artifacts' / args.run_id
    out.mkdir(exist_ok=False)
    start = time.monotonic()
    report = {'status': 'running', 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'command': [sys.executable, *sys.argv], 'training': False, 'solves': 0,
              'read_partial_cache_features': False, 'environment': {'python': sys.version, 'platform': platform.platform()}}
    try:
        preparation = ROOT / 'artifacts/cuda-xgb748-template-preparation-001'
        expected = '41f91c0f0b9e3d4224c6a6188f876b399b1a9e7906d7a23e846b9d99f97d57c4'
        assert sha(preparation / 'manifest.json') == expected
        m = json.loads((preparation / 'manifest.json').read_text())
        assert m['status'] == 'prepared_unsealed_waiting_for_complete_independently_audited_cache'
        assert not m['model_protocol_sealed'] and not m['dataset_packaged'] and not m['read_partial_cache_features']
        for name, value in m['source'].items():
            assert sha(ROOT / name) == value, name
            if name.endswith('.py'):
                compile(ast.parse((ROOT / name).read_text()), name, 'exec')
        assert sha(ROOT / 'docs/cuda_xgb664_protocol.json') == m['original664_protocol_sha256']
        assert sha(ROOT / 'docs/integer_water_budget_four_worker_protocol_v2.json') == m['expected_cache_protocol_sha256']
        assert sha(ROOT / 'docs/integer_water_budget_full_cache_protocol.json') == m['expected_cache_base_protocol_sha256']
        assert not (ROOT / 'docs/cuda_xgb748_protocol.json').exists()
        for branch in ('745', '748'):
            path = ROOT / f'docs/cuda_xgb{branch}_standalone_declaration.json'
            declaration = json.loads(path.read_text())
            assert declaration['formula'] == f'project_vinje(rawGPU{branch}),float64'
            assert declaration['automatic_promotion'] is False and declaration['final_export_enabled'] is False
        command = [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_cuda_xgb748.py', '-v']
        tests = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        (out / 'tests.log').write_text(tests.stdout + tests.stderr)
        assert tests.returncode == 0 and 'Ran 6 tests' in tests.stderr
        report.update(status='passed_template_only', preparation_manifest_sha256=expected, source=m['source'],
                      expected_cache_protocol_sha256=m['expected_cache_protocol_sha256'],
                      tests={'passed': 6, 'command': command, 'log_sha256': sha(out / 'tests.log')},
                      checks=['Source reviewed: original600/depth6/.05/GPU1/40percent samples and chronology retained.',
                              'Builder requires an externally hashed complete3647-case cache and independent full physical/replay/compatibility audit before feature access.',
                              'Every final raw artifact and case receipt checked; imported original636-case evidence byte-preserved with independently derived compatibility count.',
                              'Fourth supplement disjoint from original three; complete cache/protocol/audit and missing source dependencies explicitly included.',
                              'Existing frozen84-column formulas appended to exact664prefix; no partial-cache or relaxed fallback.',
                              'CurrentSHOP anchor, same-weight raw664 control, frozen2021grid and strict independently eligible2022 lock.',
                              'Separate standalone diagnostic declarations require both-year primary acceptance and verified actual final outputs; no additional weights.'],
                      scope='Unsealed source templates only. Actual complete cache, independently audited physical proof, package, real feature staging, hosted runtime and fresh CUDA smoke remain prerequisites. No748 model protocol or fit approved by this record.')
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        shutil.copy2(__file__, out / Path(__file__).name)
        report['audit_source_sha256'] = sha(Path(__file__))
        report['runtime_seconds'] = time.monotonic() - start
        (out / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({'status': report['status'], 'manifest_sha256': sha(out / 'manifest.json'),
                          'runtime_seconds': report['runtime_seconds']}), flush=True)


if __name__ == '__main__':
    main()
