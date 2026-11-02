"""Independently audit the frozen608 GPU package without building features."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path, PurePosixPath
import platform
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_cuda_xgb608 import digest, require


def verify_archive(path: Path, manifest_name: str, files_key: str, expected_sha: str) -> dict:
    """Stream-check the exact allowlist, safe regular files, sizes and hashes."""
    require(digest(path) == expected_sha, 'Archive digest changed')
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        require(len(infos) == len({i.filename for i in infos}), 'Duplicate archive member')
        manifest = json.loads(archive.read(manifest_name))
        entries = manifest[files_key]
        require(set(archive.namelist()) == {e['path'] for e in entries} | {manifest_name}, 'Archive allowlist changed')
        for item in infos:
            p = PurePosixPath(item.filename)
            require(not p.is_absolute() and '..' not in p.parts and '\\' not in item.filename
                    and str(p) == item.filename and not item.is_dir()
                    and not stat.S_ISLNK(item.external_attr >> 16), 'Unsafe archive member')
        for entry in entries:
            item = archive.getinfo(entry['path'])
            require(item.file_size == entry['size_bytes'], 'Member size differs')
            sha = hashlib.sha256()
            with archive.open(item) as stream:
                for block in iter(lambda: stream.read(1 << 20), b''):
                    sha.update(block)
            require(sha.hexdigest() == entry['sha256'], 'Member bytes differ')
    return manifest


def main() -> None:
    """Verify immutable provenance, exact samples and loader/lock small fixtures."""
    out = ROOT / 'artifacts/cuda-xgb608-independent-audit-001'
    out.mkdir(exist_ok=False)
    start = time.time()
    report = {'run_id': out.name, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'command': shlex.join([sys.executable, *sys.argv]), 'status': 'initializing',
              'scope': 'File hashes, archive streaming, metadata/sample arrays and synthetic fixtures; no feature matrix, model fit, Torch or XGBoost import, GPU or upload',
              'environment': {'python': sys.version, 'platform': platform.platform(),
                   'packages': {p: importlib.metadata.version(p) for p in ['numpy', 'pandas']}}, 'checks': {}, 'kaggle': None}
    try:
        package_dir = ROOT / 'artifacts/cuda-xgb608-package-001'
        package = json.loads((package_dir / 'manifest.json').read_text())
        protocol_path = ROOT / 'docs/cuda_xgb608_protocol.json'
        require(digest(protocol_path) == package['protocol_sha256']
                == '789c2a6419c526079e18ea83ac5dd1143ef0b2c590e522b163ddf4e890a72401', 'Frozen608 protocol changed')
        protocol = json.loads(protocol_path.read_text())
        for path, expected in package['source'].items():
            require(digest(ROOT / path) == digest(package_dir / 'source' / path) == expected, f'Frozen source differs: {path}')
        for key, name in [('trainer_sha256', 'train_cuda_xgb608.py'), ('shared_loader_sha256', 'cuda_xgb608_loader.py'),
                          ('core_loader_sha256', 'train_cuda_tcn.py'), ('launcher_sha256', 'launch_cuda_xgb.py'),
                          ('local_selector_sha256', 'select_cuda_xgb608_blend.py')]:
            require(digest(ROOT / 'scripts' / name) == protocol[key], f'Protocol source differs: {key}')
        require(digest(ROOT / 'sintef_project/postprocess.py') == protocol['postprocess_sha256'], 'Fixed projection changed')
        base_dir = ROOT / 'artifacts/kaggle-neural-bundle-001'
        base_manifest_path = base_dir / 'payload/bundle_manifest.json'
        require(digest(base_dir / 'kaggle-neural-inputs.zip') == protocol['bundle_archive_sha256']
                and digest(base_manifest_path) == protocol['bundle_manifest_sha256'], 'Original sealed bundle changed')
        base = json.loads(base_manifest_path.read_text())
        extra = verify_archive(package_dir / 'cuda-xgb608-supplement.zip', 'bundle_manifest.json', 'payload_files',
                               protocol['supplement_archive_sha256'])
        require(digest(package_dir / 'supplement/bundle_manifest.json') == protocol['supplement_manifest_sha256'], 'Supplement manifest differs')
        require(extra == json.loads((package_dir / 'supplement/bundle_manifest.json').read_text()), 'Archive/supplement manifest differs')
        code = verify_archive(package_dir / 'cuda-xgb608-code.zip', 'package_manifest.json', 'files', package['code_sha256'])
        for entry in code['files']:
            require(digest(package_dir / 'code' / entry['path']) == entry['sha256'], 'Code payload differs')
        require(digest(package_dir / 'code/cuda_xgb608_protocol.json') == digest(protocol_path), 'Deployed protocol differs')
        base_files = {e['path']: e['sha256'] for e in base['payload_files']}
        extra_files = {e['path']: e['sha256'] for e in extra['payload_files']}
        require(not set(base_files).intersection(extra_files) and len(extra_files) == 8, 'Supplement overlays original inputs')
        union = {**base_files, **extra_files}
        for name, expected in protocol['feature_source_sha256'].items():
            require(union[name] == expected == digest(ROOT / name), 'Missing or altered feature dependency')
        reference_path = ROOT / 'artifacts/lgb-normalized-lp-context-joint-001/manifest.json'
        require(digest(reference_path) == protocol['baseline_manifest_sha256'], 'Frozen608 reference changed')
        reference = json.loads(reference_path.read_text())
        context = json.loads((ROOT / 'artifacts/lgb-normalized-lp-context-001/manifest.json').read_text())
        joint = json.loads((ROOT / 'artifacts/lp-joint-full-001/manifest.json').read_text())
        require(protocol['feature_columns'] == reference['features'] and len(reference['features']) == 608
                and reference['features'][:521] == context['features'] and reference['features'][521:] == joint['feature_names'],
                'Exact521+87 feature schema changed')
        for category in ['source', 'data']:
            for name, expected in joint[category].items():
                require(union[name] == expected == digest(ROOT / name), 'Joint cache verification dependency changed')
        require(extra_files['artifacts/lp-joint-full-001/features.npz'] == joint['artifacts']['features.npz'], 'Joint cache bytes differ')
        original = json.loads((ROOT / 'docs/cuda_xgb_protocol.json').read_text())
        require(protocol['model_parameters'] == original['model_parameters']
                and protocol['sample_sha256'] == original['sample_sha256'], 'Hyperparameters or sampled rows changed')
        anchor_dir = ROOT / 'artifacts/lp-context-joint-comparison-001'
        require(digest(anchor_dir / 'analysis.json') == protocol['anchor']['analysis_sha256'], 'Fixed608 anchor changed')
        anchor = json.loads((anchor_dir / 'analysis.json').read_text())
        require(anchor['accepted'] is True and anchor['selected_weights'] == protocol['anchor']['weights']
                == [.140625, .421875, .1875, .25], 'Anchor selection changed')
        for year in [2021, 2022]:
            path = anchor_dir / f'selected_validation_{year}.npz'
            require(digest(path) == anchor['artifacts'][path.name], 'Frozen anchor array changed')
        report['checks']['source_archive_schema_sample_anchor_closure'] = True
        # Existing five fixtures exercise collision/tampering, exact samples, cache-path scope and lock gating.
        test = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_cuda_xgb608.py'],
                              cwd=ROOT, capture_output=True, text=True)
        (out / 'fixtures.log').write_text(test.stdout + test.stderr)
        require(test.returncode == 0, '608 small fixtures failed')
        report['checks']['five_loader_sample_lock_fixtures'] = True
        import pandas as pd
        from scripts.cuda_xgb608_loader import core
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / 'targets.csv'
            target.write_text('Run No,starttime,total_calculation_time,target\n1,2020-01-01,0,1\n2,2021-01-01,0,0\n3,2022-01-01,0,INVALID_FUTURE_TARGET\n')
            calls = []
            def observed_read(path, **kwargs):
                calls.append(kwargs.copy())
                require(kwargs in [{'usecols': ['Run No', 'starttime']}, {'nrows': 2}], 'Future target decode attempted')
                return pd.read_csv(path, **kwargs)
            redacted = core.read_targets_through_year(target, 2021, observed_read)
            require(redacted.target.tolist() == [1, 0, 0] and len(calls) == 2, 'Future target prefix redaction failed')
        report['checks']['poisoned2022_target_not_decoded'] = True
        report.update(status='complete', approved_for_private_synthetic608_smoke_then_explicit2021=True,
                      protocol_sha256=digest(protocol_path), code_archive_sha256=package['code_sha256'],
                      supplement_archive_sha256=protocol['supplement_archive_sha256'],
                      caveat='Actual608 GPU runtime and feature construction unmeasured until hosted execution; external returned-artifact verifier still required before2021 selection.')
    except BaseException:
        report['status'] = 'failed'
        report['error'] = traceback.format_exc()
        raise
    finally:
        shutil.copy2(__file__, out / Path(__file__).name)
        report['source_sha256'] = digest(Path(__file__))
        report['runtime_seconds'] = time.time()-start
        (out / 'manifest.json').write_text(json.dumps(report, indent=2))
        with (ROOT / 'docs/experiments.md').open('a') as stream:
            stream.write(f'\n## {out.name} — {report["timestamp_utc"]}\n\nStatus:{report["status"]}. Command:`{report["command"]}`. Independent608GPU package/source/ZIP/schema/cache/sample/anchor audit; five supplied small fixtures plus poisoned future-target prefix check. No featurematrix, training, model import, GPU or upload. Exact hashes, source, environment, runtime and scope:`artifacts/{out.name}/manifest.json`. Frozen code/protocol/input archives unchanged. NoKaggle result.\n')
        print(json.dumps({k: report.get(k) for k in ['status', 'checks', 'runtime_seconds', 'approved_for_private_synthetic608_smoke_then_explicit2021']}), flush=True)


if __name__ == '__main__':
    main()
