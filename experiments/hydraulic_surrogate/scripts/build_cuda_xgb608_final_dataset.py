"""Wrap the prepared608 final code in a separate private auto-extracted dataset."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.train_cuda_xgb608_final import digest,require


def main()->None:
    """Build only a private code dataset; leave original code and data artifacts intact."""
    started=time.time();timestamp=datetime.now(timezone.utc).isoformat()
    source=ROOT/'artifacts/cuda-xgb608-final-package-001'
    parent=json.loads((source/'manifest.json').read_text())
    require(parent['status']=='complete' and parent['archive_sha256']=='a0fb18c6b30407d32951e94e951b9e3b62376d69e2e1517ce27cc0787336ed43',
            'Prepared final code package identity changed')
    require(digest(source/'cuda-xgb608-final-code.zip')==parent['archive_sha256'],'Original code archive changed')
    # Package preparation precedes independent review; no upload or fit is performed.
    out=ROOT/'artifacts/cuda-xgb608-final-private-dataset-001';out.mkdir(exist_ok=False)
    payload=out/'payload';payload.mkdir()
    shutil.copytree(source/'payload',payload/'code')
    require(digest(payload/'code/package_manifest.json')==parent['package_manifest_sha256'],'Code package manifest changed')
    for entry in parent['payload_files']:
        require(digest(payload/'code'/entry['path'])==entry['sha256'],'Code member changed')
    launcher=ROOT/'scripts/launch_cuda_xgb608_final_dataset.py'
    shutil.copy2(launcher,payload/launcher.name)
    (payload/'README.md').write_text('PRIVATE code-only dataset for the review-pending exact608 GPU final fit. Attach beside the existing private hydraulic inputs and608 supplement; preserve both. Run the exact hash-verified discovery launcher only when cuda:1 is free. The immutable final protocol and original validated dependencies are in code/. No model artifacts, credentials, environments, anchor predictions or raw data are included. The original successful608 smoke and full sample are required.\n')
    files={str(p.relative_to(payload)):digest(p) for p in sorted(payload.rglob('*')) if p.is_file()}
    (payload/'dataset_manifest.json').write_text(json.dumps({'private_only':True,'files':files,
        'original_package_manifest_sha256':digest(source/'manifest.json'),'independent_review':'Required separately before upload/launch',
        'protocol_sha256':parent['protocol_sha256'],'launcher_sha256':digest(launcher)},indent=2))
    archive=out/'cuda-xgb608-final-private-dataset.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as saved:
        for path in sorted(payload.rglob('*')):
            if path.is_file():saved.write(path,path.relative_to(payload).as_posix())
    with zipfile.ZipFile(archive) as saved:
        require(set(saved.namelist())==set(files)|{'dataset_manifest.json'},'Dataset namespace changed')
        for name,expected in files.items():require(hashlib.sha256(saved.read(name)).hexdigest()==expected,'Dataset ZIP readback changed')
    sources={}
    for relative in ['scripts/build_cuda_xgb608_final_dataset.py','scripts/launch_cuda_xgb608_final_dataset.py','tests/test_cuda_xgb608_final_discovery.py']:
        target=out/'source'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/relative,target);sources[relative]=digest(target)
    report={'timestamp_utc':timestamp,'status':'complete','source':sources,'archive_sha256':digest(archive),
            'archive_size_bytes':archive.stat().st_size,'dataset_manifest_sha256':digest(payload/'dataset_manifest.json'),
            'protocol_sha256':parent['protocol_sha256'],'launcher_sha256':digest(launcher),'runtime_seconds':time.time()-started,
            'original_package_manifest_sha256':digest(source/'manifest.json'),'private_only':True,'upload':False,'training':False,'kaggle':None}
    (out/'manifest.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    with (ROOT/'docs/experiments.md').open('a') as stream:
        stream.write(f'\n## cuda-xgb608-final-private-dataset-001 — {timestamp}\n\nPrepared separate PRIVATE code-only dataset for the prepared original608 final fit; no upload or training. Original final protocol/package bytes retained, code under disjoint code/ namespace; helper discovers only unique exact protocol/bundle hashes and verifies every package member. ZIP {report["archive_size_bytes"]} bytes, SHA `{report["archive_sha256"]}`. Original608 input dataset untouched. Full sources/readback proof: `artifacts/cuda-xgb608-final-private-dataset-001/manifest.json`.\n')


if __name__=='__main__':main()
