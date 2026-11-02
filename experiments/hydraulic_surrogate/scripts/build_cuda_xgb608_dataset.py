"""Create one PRIVATE auto-extractable dataset archive from frozen608 inputs/code."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shlex
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_cuda_xgb608_package import zip_verified
from scripts.train_cuda_xgb608 import digest, require


def main() -> None:
    """Copy original package into disjoint code/supplement namespaces; no upload/run."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', default='cuda-xgb608-private-dataset-001')
    args = parser.parse_args()
    out = ROOT/'artifacts'/args.run_id
    out.mkdir(exist_ok=False)
    start = time.time()
    package = ROOT/'artifacts/cuda-xgb608-package-001'
    prior = json.loads((package/'manifest.json').read_text())
    require(prior['status'] == 'complete', 'Original608 package incomplete')
    for path, key in [(package/'cuda-xgb608-code.zip', 'code_sha256'),
                      (package/'cuda-xgb608-supplement.zip', 'supplement_sha256')]:
        require(digest(path) == prior[key], 'Frozen608 archive changed')
    payload = out/'payload'
    payload.mkdir()
    for dirname in ['code', 'supplement']:
        shutil.copytree(package/dirname, payload/dirname)
    launcher = ROOT/'scripts/launch_cuda_xgb608_dataset.py'
    shutil.copy2(launcher, payload/'code/launch_cuda_xgb608_dataset.py')
    # The original code manifest stays preserved in package001. This derived
    # code manifest includes the new path-independent launcher explicitly.
    code_files = [{'path': path.name, 'sha256': digest(path), 'size_bytes': path.stat().st_size}
                  for path in sorted((payload/'code').iterdir()) if path.name != 'package_manifest.json']
    (payload/'code/package_manifest.json').write_text(json.dumps({
        'files': code_files, 'private': True,
        'derived_from_manifest_sha256': digest(package/'code/package_manifest.json')}, indent=2))
    files = [{'path': path.relative_to(payload).as_posix(), 'sha256': digest(path), 'size_bytes': path.stat().st_size}
             for path in sorted(payload.rglob('*')) if path.is_file()]
    dataset = {'version': 'cuda_xgb608_private_dataset_v1',
               'created_utc': datetime.now(timezone.utc).isoformat(), 'required_visibility': 'PRIVATE',
               'files': files, 'source_package_manifest_sha256': digest(package/'manifest.json'),
               'protocol_sha256': prior['protocol_sha256'],
               'launcher_sha256': digest(launcher), 'original_base_bundle_included': False,
               'disjoint_namespaces': ['code', 'supplement'],
               'execution': 'No autoexecute. Locate launch_cuda_xgb608_dataset.py; explicitly request smoke first. Input manifests discovered by exact hashes.'}
    (payload/'dataset_manifest.json').write_text(json.dumps(dataset, indent=2))
    archive = out/'cuda-xgb608-private-dataset.zip'
    zip_verified(payload, archive)
    report = {'timestamp_utc': dataset['created_utc'], 'status': 'complete',
              'command': shlex.join([sys.executable, *sys.argv]), 'environment': {'python': sys.version, 'platform': platform.platform()},
              'archive': str(archive.relative_to(ROOT)), 'archive_sha256': digest(archive),
              'archive_size_bytes': archive.stat().st_size, 'dataset_manifest_sha256': digest(payload/'dataset_manifest.json'),
              'protocol_sha256': prior['protocol_sha256'], 'launcher_sha256': digest(launcher),
              'source_package_manifest_sha256': digest(package/'manifest.json'),
              'runtime_seconds': time.time()-start, 'upload': False, 'training': False, 'kaggle': None,
              'source': {name: digest(ROOT/name) for name in ['scripts/build_cuda_xgb608_dataset.py', 'scripts/launch_cuda_xgb608_dataset.py', 'tests/test_cuda_xgb608_discovery.py']}}
    for name in report['source']:
        target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    (out/'manifest.json').write_text(json.dumps(report,indent=2))
    with (ROOT/'docs/experiments.md').open('a') as stream:
        stream.write(f'\n## {args.run_id} — {report["timestamp_utc"]}\n\nStatus:complete. Command:`{report["command"]}`. Derived singlePRIVATE dataset archive with separatecode/supplement namespaces and exact-hash discovery launcher; original608package/original395bundle unchanged. No upload/launch. ZIPreadback everymemberverified. ArchiveSHA`{report["archive_sha256"]}`, bytes{report["archive_size_bytes"]}; protocolSHA`{report["protocol_sha256"]}`. Fullsources/environment/hashes:`artifacts/{args.run_id}/manifest.json`. Two small discoverytests pass (see test log); nofeaturematrix/model. NoKaggle score.\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
