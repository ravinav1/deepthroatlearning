"""Audit the derived private608 dataset and its exact-hash discovery helper."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_cuda_xgb608_package import digest, require, verify_archive


def main() -> None:
    """Verify the derived ZIP byte-for-byte against its already audited parent."""
    out = ROOT / 'artifacts/cuda-xgb608-dataset-independent-audit-001'
    out.mkdir(exist_ok=False)
    started = time.time()
    derived = ROOT / 'artifacts/cuda-xgb608-private-dataset-001'
    parent = ROOT / 'artifacts/cuda-xgb608-package-001'
    report = {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'command': shlex.join([sys.executable, *sys.argv]),
              'status': 'initializing', 'scope': 'Derived archive/source/hash-discovery fixtures only; no training, feature matrix or remote action', 'kaggle': None}
    try:
        expected = json.loads((derived / 'manifest.json').read_text())
        require(expected['archive_sha256'] == '480cacd6a83aef078104fd35ddfeb5c0569f0dd7d84158f63f82a1511d766a9a', 'Derived archive identity changed')
        dataset = verify_archive(derived / 'cuda-xgb608-private-dataset.zip', 'dataset_manifest.json', 'files', expected['archive_sha256'])
        require(digest(derived / 'payload/dataset_manifest.json') == expected['dataset_manifest_sha256']
                and dataset == json.loads((derived / 'payload/dataset_manifest.json').read_text()), 'Dataset manifest changed')
        require(dataset['source_package_manifest_sha256'] == digest(parent / 'manifest.json'), 'Audited parent identity differs')
        entries = {item['path']: item for item in dataset['files']}
        for namespace in ['code', 'supplement']:
            for source in (parent / namespace).rglob('*'):
                if not source.is_file() or (namespace == 'code' and source.name == 'package_manifest.json'):
                    continue
                name = source.relative_to(parent).as_posix()
                require(entries[name]['sha256'] == digest(source), f'Derived parent payload changed: {name}')
        helper = ROOT / 'scripts/launch_cuda_xgb608_dataset.py'
        require(digest(helper) == digest(derived / 'source/scripts/launch_cuda_xgb608_dataset.py')
                == entries['code/launch_cuda_xgb608_dataset.py']['sha256']
                == '8297e491bfe94fe7d78a8b171653a689c1494db201f59add13088f71f18b8891', 'Discovery helper changed')
        require(dataset['protocol_sha256'] == '789c2a6419c526079e18ea83ac5dd1143ef0b2c590e522b163ddf4e890a72401'
                and dataset['original_base_bundle_included'] is False, 'Derived protocol/base duplication changed')
        test = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_cuda_xgb608_discovery.py'],
                              cwd=ROOT, capture_output=True, text=True)
        (out / 'fixtures.log').write_text(test.stdout + test.stderr)
        require(test.returncode == 0, 'Discovery fixtures failed')
        report.update(status='complete', approved_for_private_upload=True, archive_sha256=expected['archive_sha256'],
                      helper_sha256=digest(helper), fixture_groups=2,
                      caveat='Root must verify destination privacy and observed input mount; synthetic608smoke still mandatory before any real fold.')
    finally:
        shutil.copy2(__file__, out / Path(__file__).name)
        report['runtime_seconds'] = time.time()-started
        (out / 'manifest.json').write_text(json.dumps(report, indent=2))
        with (ROOT / 'docs/experiments.md').open('a') as stream:
            stream.write(f'\n## {out.name} — {report["timestamp_utc"]}\n\nStatus:{report["status"]}. Command:`{report["command"]}`. Independent derived608PRIVATE ZIP/helper audit. Every inherited code/supplement payload byte matches previously auditedpackage001; unique exact-hash discovery helper and two fixtures verified. No featurematrix/training/remote action. Proof:`artifacts/{out.name}/manifest.json`. NoKaggle score.\n')
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
