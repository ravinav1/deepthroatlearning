"""Audit the frozen GPU XGBoost package and prepare explicit hosted cells."""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    """Hash a file without retaining large artifacts in memory."""
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    """Run only metadata/sample-array checks; never build features or fit models."""
    started = time.time()
    out = ROOT / 'artifacts/cuda-xgb-audit-recovery-001'
    out.mkdir(exist_ok=False)
    protocol_path = ROOT / 'docs/cuda_xgb_protocol.json'
    protocol = json.loads(protocol_path.read_text())
    package_dir = ROOT / 'artifacts/cuda-xgb-package-001'
    package = json.loads((package_dir / 'manifest.json').read_text())
    archive_path = package_dir / 'cuda-xgb-code.zip'
    checks = {}
    report = {
        'run_id': out.name,
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'command': 'UV_CACHE_DIR=/private/tmp/tdt4173-xgb-uv-cache uv run --offline --no-project --python .venv-ml/bin/python python scripts/audit_cuda_xgb_recovery.py',
        'environment': {'python': sys.version, 'platform': platform.platform(),
                        'machine': platform.machine(),
                        'packages': {name: importlib.metadata.version(name) for name in
                                     ['numpy', 'pandas', 'scikit-learn', 'xgboost']}},
        'seed': 4173,
        'data_scope': 'Hashes, chronological metadata, and saved sample arrays only; no full feature matrix or model fit.',
        'checks': checks,
        'prior_failed_command': {
            'command': 'uv run --offline --no-project --python .venv-ml/bin/python python -m unittest tests.test_cuda_xgb -v',
            'failure': 'Default /Users/heap/.cache/uv cache write denied; rerun used writable task-specific cache.'},
        'gpu_launched': False,
        'kaggle_submission': None,
    }
    try:
        assert digest(archive_path) == package['archive_sha256']
        assert digest(protocol_path) == package['protocol_sha256']
        checks['archive_and_protocol_sha256'] = True
        with zipfile.ZipFile(archive_path) as archive:
            manifest_bytes = archive.read('package_manifest.json')
            assert hashlib.sha256(manifest_bytes).hexdigest() == package['package_manifest_sha256']
            entries = json.loads(manifest_bytes)['files']
            assert set(archive.namelist()) == {item['path'] for item in entries} | {'package_manifest.json'}
            for item in entries:
                data = archive.read(item['path'])
                assert len(data) == item['size_bytes']
                assert hashlib.sha256(data).hexdigest() == item['sha256']
                assert data == (package_dir / 'payload' / item['path']).read_bytes()
        checks['all_package_members_exact'] = True
        sources = {
            'scripts/train_cuda_xgb.py': 'trainer_sha256',
            'scripts/train_cuda_tcn.py': 'shared_loader_sha256',
            'scripts/launch_cuda_xgb.py': 'launcher_sha256',
            'scripts/select_cuda_xgb_blend.py': 'local_selector_sha256',
            'sintef_project/postprocess.py': 'postprocess_sha256',
        }
        for relative, key in sources.items():
            assert digest(ROOT / relative) == protocol[key]
        checks['frozen_source_closure_matches'] = True
        anchor_dir = ROOT / 'artifacts' / protocol['anchor']['run_id']
        anchor = json.loads((anchor_dir / 'analysis.json').read_text())
        assert digest(anchor_dir / 'analysis.json') == protocol['anchor']['analysis_sha256']
        assert anchor['accepted'] is True
        assert anchor['selected_weights'] == protocol['anchor']['weights'] == [.1875, .5625, .25]
        for year in [2021, 2022]:
            name = f'selected_validation_{year}.npz'
            assert digest(anchor_dir / name) == anchor['artifacts'][name]
        checks['earlier_joint_anchor_and_arrays_unchanged'] = True
        sys.path.insert(0, str(ROOT))
        import numpy as np
        import pandas as pd
        from scripts.train_cuda_xgb import PARAMETERS, reconstruct_sample
        baseline_path = ROOT / 'artifacts/lgb-normalized-lp-001/manifest.json'
        baseline = json.loads(baseline_path.read_text())
        previous = json.loads((ROOT / 'artifacts/xgb-lp-specialists-001/manifest.json').read_text())
        assert digest(baseline_path) == protocol['baseline_manifest_sha256']
        assert PARAMETERS == protocol['model_parameters']
        assert {k: v for k, v in PARAMETERS.items() if k != 'device'} == {
            k: v for k, v in previous['model_parameters'].items() if k != 'device'}
        dates = pd.to_datetime(pd.read_csv(ROOT / 'sintef_project/data/kernel/Unit_commitment_decisions.csv',
                                          usecols=['starttime']).starttime, utc=True)
        checks['exact_samples'] = {}
        for year in [2021, 2022]:
            sample = reconstruct_sample(dates, year, baseline, np, pd)
            saved = baseline_path.parent / f'train_sample_{year}.npy'
            np.testing.assert_array_equal(sample, np.load(saved, allow_pickle=False))
            assert digest(saved) == protocol['sample_sha256'][str(year)]
            checks['exact_samples'][str(year)] = {'rows': len(sample), 'sha256': digest(saved)}
        checks['requested_parameters_match_cpu_specialists_except_cuda_device'] = True
        test = subprocess.run([sys.executable, '-m', 'unittest', 'tests.test_cuda_xgb', '-v'],
                              cwd=ROOT, capture_output=True, text=True)
        (out / 'fixture_tests.log').write_text(test.stdout + test.stderr)
        assert test.returncode == 0
        checks['four_cpu_fixtures_passed'] = True
        # Reuse the immutable code archive, rather than changing its protocol.
        encoded = base64.b64encode(archive_path.read_bytes()).decode('ascii')
        setup = f'''from pathlib import Path
import base64, hashlib, io, json, subprocess, sys, zipfile
XGB_ARCHIVE_SHA = {package['archive_sha256']!r}
XGB_ARCHIVE_BYTES = base64.b64decode({encoded!r})
assert hashlib.sha256(XGB_ARCHIVE_BYTES).hexdigest() == XGB_ARCHIVE_SHA
XGB_CODE_ROOT = Path('/kaggle/working/tdt4173-xgb-code-' + XGB_ARCHIVE_SHA[:12])
with zipfile.ZipFile(io.BytesIO(XGB_ARCHIVE_BYTES)) as archive:
    manifest_bytes = archive.read('package_manifest.json')
    assert hashlib.sha256(manifest_bytes).hexdigest() == {package['package_manifest_sha256']!r}
    entries = json.loads(manifest_bytes)['files']
    assert set(archive.namelist()) == {{entry['path'] for entry in entries}} | {{'package_manifest.json'}}
    for entry in entries:
        assert Path(entry['path']).name == entry['path']
        data = archive.read(entry['path'])
        assert len(data) == entry['size_bytes'] and hashlib.sha256(data).hexdigest() == entry['sha256']
    XGB_CODE_ROOT.mkdir(exist_ok=True)
    for name in archive.namelist():
        target = XGB_CODE_ROOT / name
        data = archive.read(name)
        if target.exists():
            assert target.read_bytes() == data
        else:
            with target.open('xb') as stream:
                stream.write(data)
XGB_PROTOCOL = XGB_CODE_ROOT / 'cuda_xgb_protocol.json'
assert hashlib.sha256(XGB_PROTOCOL.read_bytes()).hexdigest() == {package['protocol_sha256']!r}
XGB_BASE_COMMAND = [sys.executable, str(XGB_CODE_ROOT / 'launch_cuda_xgb.py'),
    '--venv', '/kaggle/working/tdt4173-cuda-system-env',
    '--runner', str(XGB_CODE_ROOT / 'train_cuda_xgb.py'), '--',
    '--protocol', str(XGB_PROTOCOL), '--shared-loader', str(XGB_CODE_ROOT / 'train_cuda_tcn.py')]
print(json.dumps({{'code_root': str(XGB_CODE_ROOT), 'protocol_sha256': hashlib.sha256(XGB_PROTOCOL.read_bytes()).hexdigest(),
                  'device': 'cuda:1', 'real_fit_started': False}}))
'''
        smoke = '''# Run only this synthetic cell first. Existing completed IDs must not be rerun.
XGB_SMOKE_RUN_ID = 'cuda-xgb-smoke-001'
subprocess.run(XGB_BASE_COMMAND + ['--mode', 'smoke', '--run-id', XGB_SMOKE_RUN_ID], check=True)
XGB_SMOKE_MANIFEST = Path('/kaggle/working/xgb-runs') / XGB_SMOKE_RUN_ID / 'manifest.json'
smoke_report = json.loads(XGB_SMOKE_MANIFEST.read_text())
assert smoke_report['status'] == 'complete'
print(json.dumps({'status': smoke_report['status'], 'manifest': str(XGB_SMOKE_MANIFEST),
                  'runtime_seconds': smoke_report['runtime_seconds'], 'device': smoke_report['cuda']['requested_device']}))
'''
        fit = '''# Explicit 2021 only. Run after inspecting successful XGB smoke output.
# This is the already observed private dataset's extracted root.
XGB_BUNDLE = Path('/kaggle/input/datasets/ravinav/tdt4173-private-hydraulic-training-inputs')
assert XGB_BUNDLE.is_dir()
assert hashlib.sha256((XGB_BUNDLE / 'bundle_manifest.json').read_bytes()).hexdigest() == '6160e83f641dd3f311a1ebd96b2b3a05914b99b810e98ec262576be892563469'
assert json.loads(XGB_SMOKE_MANIFEST.read_text())['status'] == 'complete'
subprocess.run(XGB_BASE_COMMAND + ['--mode', 'fit', '--year', '2021', '--run-id', 'cuda-xgb-2021-001',
    '--bundle', str(XGB_BUNDLE), '--smoke-manifest', str(XGB_SMOKE_MANIFEST)], check=True)
'''
        for name, source in [('setup_cell.py', setup), ('smoke_cell.py', smoke), ('fit_2021_cell.py', fit)]:
            compile(source, name, 'exec')
            (out / name).write_text(source)
        def cell(kind: str, source: str) -> dict:
            value = {'cell_type': kind, 'metadata': {}, 'source': source.splitlines(keepends=True)}
            if kind == 'code':
                value.update(execution_count=None, outputs=[])
            return value
        notebook = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {'kernelspec': {
            'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, 'cells': [
                cell('markdown', '# Private GPU XGBoost review\n\nKeep private. Setup and synthetic smoke only are enabled. Add these cells to the existing private notebook without replacing TCN work. No automatic 2022 command is provided.\n'),
                cell('code', setup), cell('code', smoke),
                cell('markdown', 'Run 2021 only after reviewing the successful smoke. Download all artifacts; select the frozen grid locally before any 2022 command.\n'),
                cell('code', 'RUN_2021_EXPLICITLY = False\nif RUN_2021_EXPLICITLY:\n' + ''.join('    ' + line + '\n' for line in fit.splitlines())),
            ]}
        (out / 'cuda_xgb_review.ipynb').write_text(json.dumps(notebook, indent=1))
        checks['generated_hosted_cells_compile'] = True
        report.update(status='passed', protocol_sha256=digest(protocol_path),
                      package_archive_sha256=digest(archive_path),
                      anchor_run=protocol['anchor']['run_id'],
                      source={relative: digest(ROOT / relative) for relative in sources})
    except BaseException:
        report['status'] = 'failed'
        raise
    finally:
        shutil.copy2(__file__, out / 'audit_script.py')
        report['runtime_seconds'] = time.time() - started
        report['artifacts'] = {p.name: digest(p) for p in out.iterdir() if p.is_file() and p.name != 'manifest.json'}
        (out / 'manifest.json').write_text(json.dumps(report, indent=2))
        print(json.dumps({'status': report['status'], 'manifest': str(out / 'manifest.json'),
                          'runtime_seconds': report['runtime_seconds']}))


if __name__ == '__main__':
    main()
