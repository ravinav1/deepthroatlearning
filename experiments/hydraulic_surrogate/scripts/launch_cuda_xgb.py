"""Launch the reviewed XGBoost runner through an offline uv hosted environment."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys


def main() -> None:
    """Reuse installed CUDA packages without installing or downloading anything."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--venv', type=Path, default=Path('/kaggle/working/.venv-tdt4173-xgb'))
    parser.add_argument('--runner', type=Path, required=True)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    uv = shutil.which('uv')
    if uv is None:
        raise RuntimeError('Preinstalled uv is unavailable; no package/network fallback is allowed')
    if not args.venv.exists():
        subprocess.run([uv, 'venv', '--offline', '--python', sys.executable,
                        '--system-site-packages', str(args.venv)], check=True)
    cfg = args.venv / 'pyvenv.cfg'
    if not cfg.is_file() or 'include-system-site-packages = true' not in cfg.read_text().lower():
        raise RuntimeError('Existing environment does not match the offline system-package protocol')
    forwarded = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    subprocess.run([str(args.venv / 'bin/python'), str(args.runner), *forwarded], check=True)


if __name__ == '__main__':
    main()
