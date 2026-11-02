"""Locate a PRIVATE748 code/supplement dataset by hashes and launch one gated command."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys




def digest(path: Path) -> str:
    """Hash a small manifest or code file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_exact(root: Path, filename: str, expected: str) -> Path:
    """Choose only a unique regular file with the exact independently frozen hash."""
    matches = [path for path in root.rglob(filename)
               if path.is_file() and not path.is_symlink() and digest(path) == expected]
    if len(matches) != 1:
        raise ValueError(f'Expected one {filename} with SHA {expected}; found {len(matches)}')
    return matches[0].resolve()


def discover(root: Path, protocol_sha256: str, include_inputs: bool = False) -> dict:
    """Find code first; read only input manifests when a real fit is requested."""
    protocol_path = find_exact(root, 'cuda_xgb748_protocol.json', protocol_sha256)
    protocol = json.loads(protocol_path.read_text())
    code = protocol_path.parent
    for name, key in [('train_cuda_xgb748.py', 'trainer_sha256'),
                      ('cuda_xgb748_loader.py', 'shared_loader_sha256'),
                      ('cuda_xgb608_loader.py', 'prior_loader_sha256'),
                      ('cuda_xgb664_loader.py', 'prior664_loader_sha256'),
                      ('train_cuda_tcn.py', 'core_loader_sha256'),
                      ('launch_cuda_xgb.py', 'launcher_sha256')]:
        path = code/name
        if not path.is_file() or path.is_symlink() or digest(path) != protocol[key]:
            raise ValueError(f'Private code dependency differs: {name}')
    result = {'protocol': protocol_path, 'code': code, 'protocol_sha256': protocol_sha256}
    if include_inputs:
        result['bundle'] = find_exact(root, 'bundle_manifest.json', protocol['bundle_manifest_sha256']).parent
        result['supplement'] = find_exact(root, 'bundle_manifest.json', protocol['supplement_manifest_sha256']).parent
        result['shop_bundle'] = find_exact(root, 'bundle_manifest.json', protocol['shop_bundle_manifest_sha256']).parent
        result['integer_bundle'] = find_exact(root, 'bundle_manifest.json', protocol['integer_bundle_manifest_sha256']).parent
    return result


def main() -> None:
    """Launch only explicit smoke or one fold; dry-run prints a reviewable command."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--input-root', type=Path, default=Path('/kaggle/input'))
    parser.add_argument('--mode', choices=['smoke', 'fit'], required=True)
    parser.add_argument('--year', type=int, choices=[2021, 2022])
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--smoke-manifest', type=Path)
    parser.add_argument('--selection-lock', type=Path)
    parser.add_argument('--selection-lock-sha256')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.mode == 'smoke' and any(v is not None for v in [args.year, args.smoke_manifest, args.selection_lock, args.selection_lock_sha256]):
        raise ValueError('Synthetic smoke must not receive fold/lock inputs')
    if args.mode == 'fit' and (args.year is None or args.smoke_manifest is None):
        raise ValueError('Fit requires an explicit single year and successful smoke manifest')
    if args.year == 2022 and (args.selection_lock is None or args.selection_lock_sha256 is None):
        raise ValueError('2022 requires the locally reviewed2021 lock and exact SHA')
    if args.year == 2021 and (args.selection_lock is not None or args.selection_lock_sha256 is not None):
        raise ValueError('2021 must precede local selection')
    paths = discover(args.input_root, args.protocol_sha256, args.mode == 'fit')
    code = paths['code']
    command = [sys.executable, str(code/'launch_cuda_xgb.py'), '--runner', str(code/'train_cuda_xgb748.py'), '--',
               '--mode', args.mode, '--run-id', args.run_id, '--protocol', str(paths['protocol']),
               '--shared-loader', str(code/'cuda_xgb748_loader.py')]
    if args.mode == 'fit':
        command += ['--year', str(args.year), '--bundle', str(paths['bundle']), '--supplement', str(paths['supplement']),
                    '--shop-bundle', str(paths['shop_bundle']), '--integer-bundle', str(paths['integer_bundle']), '--smoke-manifest', str(args.smoke_manifest)]
    if args.year == 2022:
        command += ['--selection-lock', str(args.selection_lock), '--selection-lock-sha256', args.selection_lock_sha256]
    print(json.dumps({'located': {key: str(value) for key, value in paths.items()},
                      'command': shlex.join(command), 'dry_run': args.dry_run}), flush=True)
    if not args.dry_run:
        subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
