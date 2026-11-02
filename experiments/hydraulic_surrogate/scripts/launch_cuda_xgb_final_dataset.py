"""Locate the private final395 code dataset by exact hashes and launch one final fit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

PROTOCOL_SHA256='5033b6577b0142c8d6f38b290ac8119c45b65f0eb8b59a2d1b31b69debbf7a11'
PACKAGE_SHA256='3029627856850fca4058c744b72c6cc15c58e82ce4a091a681e973784d9a79a4'


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
    protocol_path=find_exact(root,'cuda_xgb_final_protocol.json',PROTOCOL_SHA256)
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
    validation_path=code/'cuda_xgb_protocol.json'
    if digest(validation_path)!=protocol['validation_protocol_sha256']:
        raise ValueError('Original validation protocol changed')
    validation=json.loads(validation_path.read_text())
    bundle=find_exact(root,'bundle_manifest.json',validation['bundle_manifest_sha256']).parent
    return {'protocol':protocol_path,'code':code,'bundle':bundle}


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
    if (Path('/kaggle/working/xgb-final-runs')/args.run_id).exists():
        raise ValueError('Final run ID already exists')
    command=[sys.executable,str(code/'launch_cuda_xgb.py'),'--venv',str(args.venv),
             '--runner',str(code/'train_cuda_xgb_final.py'),'--','--protocol',str(paths['protocol']),
             '--validation-protocol',str(code/'cuda_xgb_protocol.json'),
             '--validation-runner',str(code/'train_cuda_xgb.py'),'--shared-loader',str(code/'train_cuda_tcn.py'),
             '--bundle',str(paths['bundle']),'--smoke-manifest',str(args.smoke_manifest),'--run-id',args.run_id]
    print(json.dumps({'located':{k:str(v) for k,v in paths.items()},'command':shlex.join(command),'dry_run':args.dry_run}),flush=True)
    if not args.dry_run:subprocess.run(command,check=True)


if __name__=='__main__':main()
