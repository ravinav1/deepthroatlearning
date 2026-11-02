"""Locate the private final608 code dataset by exact hashes and launch one final fit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

PROTOCOL_SHA256='a5939edb8ec7f46dbdb4c6edd0c06909ea09ed0ac029709b06945a5c6434683b'
PACKAGE_SHA256='f0c0722622f676a5f3a5601549e4e1cdf0b66ab328ba5150b2c5fa96b7198502'


def digest(path:Path)->str:
    """Hash a manifest or reviewed code file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_exact(root:Path,filename:str,expected:str)->Path:
    """Require exactly one regular manifest with the frozen identity."""
    matches=[p for p in root.rglob(filename) if p.is_file() and not p.is_symlink() and digest(p)==expected]
    if len(matches)!=1:
        raise ValueError(f'Expected one {filename} with SHA {expected}; found {len(matches)}')
    return matches[0].resolve()


def discover(root:Path)->dict:
    """Verify every code-package member and discover the original sealed input dataset."""
    protocol_path=find_exact(root,'cuda_xgb608_final_protocol.json',PROTOCOL_SHA256)
    code=protocol_path.parent
    package_path=code/'package_manifest.json'
    if package_path.is_symlink() or digest(package_path)!=PACKAGE_SHA256:
        raise ValueError('Final code package manifest changed')
    package=json.loads(package_path.read_text())
    for entry in package['files']:
        path=code/entry['path']
        if Path(entry['path']).name!=entry['path'] or not path.is_file() or path.is_symlink() \
                or path.stat().st_size!=entry['size_bytes'] or digest(path)!=entry['sha256']:
            raise ValueError('Final code package member changed: '+entry['path'])
    protocol=json.loads(protocol_path.read_text())
    validation_path=code/'cuda_xgb608_protocol.json'
    if digest(validation_path)!=protocol['validation_protocol_sha256']:
        raise ValueError('Original validation protocol changed')
    validation=json.loads(validation_path.read_text())
    bundle=find_exact(root,'bundle_manifest.json',validation['bundle_manifest_sha256']).parent
    supplement=find_exact(root,'bundle_manifest.json',validation['supplement_manifest_sha256']).parent
    return {'protocol':protocol_path,'code':code,'bundle':bundle,'supplement':supplement}


def main()->None:
    """Launch one explicit reviewed final command; dry-run prints without fitting."""
    parser=argparse.ArgumentParser()
    parser.add_argument('--input-root',type=Path,default=Path('/kaggle/input'))
    parser.add_argument('--smoke-manifest',type=Path,required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--venv',type=Path,default=Path('/kaggle/working/tdt4173-cuda-system-env'))
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    if '/' in args.run_id or args.run_id in ['', '.', '..']:
        raise ValueError('Invalid final run ID')
    paths=discover(args.input_root);code=paths['code']
    protocol=json.loads(paths['protocol'].read_text())
    if digest(args.smoke_manifest)!=protocol['smoke_manifest_sha256']:
        raise ValueError('Successful original GPU smoke differs')
    if (Path('/kaggle/working/xgb608-final-runs')/args.run_id).exists():
        raise ValueError('Final run ID already exists')
    command=[sys.executable,str(code/'launch_cuda_xgb.py'),'--venv',str(args.venv),
             '--runner',str(code/'train_cuda_xgb608_final.py'),'--','--protocol',str(paths['protocol']),
             '--validation-protocol',str(code/'cuda_xgb608_protocol.json'),
             '--validation-runner',str(code/'train_cuda_xgb608.py'),'--shared-loader',str(code/'cuda_xgb608_loader.py'),
             '--bundle',str(paths['bundle']),'--supplement',str(paths['supplement']),'--smoke-manifest',str(args.smoke_manifest),'--run-id',args.run_id]
    print(json.dumps({'located':{k:str(v) for k,v in paths.items()},'command':shlex.join(command),'dry_run':args.dry_run}),flush=True)
    if not args.dry_run:subprocess.run(command,check=True)


if __name__=='__main__':main()
