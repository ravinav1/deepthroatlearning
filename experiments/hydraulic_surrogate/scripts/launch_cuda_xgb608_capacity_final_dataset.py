"""Discover exact private capacity-final code and launch a fresh smoke or final fit."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys


def digest(path: Path) -> str:
    """Hash one reviewed artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_exact(root: Path, filename: str, expected: str) -> Path:
    """Reject absent or ambiguous private input manifests."""
    paths = [p for p in root.rglob(filename) if p.is_file() and not p.is_symlink() and digest(p) == expected]
    if len(paths) != 1:
        raise ValueError(f'Expected one exact {filename}, found {len(paths)}')
    return paths[0].resolve()


def main() -> None:
    """Execute only an explicit fresh smoke or the independently accepted final recipe."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-root', type=Path, default=Path('/kaggle/input'))
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--package-sha256', required=True)
    parser.add_argument('--mode', choices=['smoke', 'final'], required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--smoke-manifest', type=Path)
    parser.add_argument('--smoke-manifest-sha256')
    parser.add_argument('--venv', type=Path, default=Path('/kaggle/working/tdt4173-cuda-system-env'))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id or args.run_id in ['', '.', '..']:
        raise ValueError('Invalid output ID')
    protocol_path = find_exact(args.input_root, 'cuda_xgb608_capacity_final_protocol.json', args.protocol_sha256)
    code = protocol_path.parent
    package_path = code / 'package_manifest.json'
    if package_path.is_symlink() or digest(package_path) != args.package_sha256:
        raise ValueError('Explicit final code package manifest differs')
    package = json.loads(package_path.read_text())
    for entry in package['files']:
        path = code / entry['path']
        if Path(entry['path']).name != entry['path'] or not path.is_file() or path.is_symlink() or path.stat().st_size != entry['size_bytes'] or digest(path) != entry['sha256']:
            raise ValueError('Final package member differs: ' + entry['path'])
    protocol = json.loads(protocol_path.read_text())
    if protocol['version'] != 'cuda_xgb608_capacity_final_v1' or digest(Path(__file__)) != protocol['dataset_launcher_sha256']:
        raise ValueError('Final protocol or launcher identity differs')
    validation_path = code / 'cuda_xgb608_capacity_protocol.json'
    if digest(validation_path) != protocol['validation_protocol_sha256']:
        raise ValueError('Accepted validation protocol differs')
    validation = json.loads(validation_path.read_text())
    command = [sys.executable, str(code / 'launch_cuda_xgb.py'), '--venv', str(args.venv)]
    if args.mode == 'smoke':
        if args.smoke_manifest is not None or args.smoke_manifest_sha256 is not None:
            raise ValueError('Fresh synthetic smoke does not take an existing smoke')
        command += ['--runner', str(code / 'train_cuda_xgb608_capacity.py'), '--', '--mode', 'smoke',
                    '--protocol', str(validation_path), '--shared-loader', str(code / 'cuda_xgb608_loader.py'), '--run-id', args.run_id]
    else:
        if args.smoke_manifest is None or not args.smoke_manifest_sha256 or digest(args.smoke_manifest) != args.smoke_manifest_sha256:
            raise ValueError('Final fit requires explicit fresh smoke identity')
        bundle = find_exact(args.input_root, 'bundle_manifest.json', validation['bundle_manifest_sha256']).parent
        supplement = find_exact(args.input_root, 'bundle_manifest.json', validation['supplement_manifest_sha256']).parent
        command += ['--runner', str(code / 'train_cuda_xgb608_capacity_final.py'), '--', '--protocol', str(protocol_path),
                    '--validation-protocol', str(validation_path), '--validation-runner', str(code / 'train_cuda_xgb608_capacity.py'),
                    '--shared-loader', str(code / 'cuda_xgb608_loader.py'), '--bundle', str(bundle), '--supplement', str(supplement),
                    '--smoke-manifest', str(args.smoke_manifest), '--smoke-manifest-sha256', args.smoke_manifest_sha256,
                    '--run-id', args.run_id]
    print(json.dumps({'command': shlex.join(command), 'dry_run': args.dry_run, 'mode': args.mode}), flush=True)
    if not args.dry_run:
        subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
