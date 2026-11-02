"""Dispatch to the immutable, independently reviewed CUDA TCN selector.

The recorded 2021 lock binds the exact snapshot selector bytes. Keep that
implementation immutable when importing helpers or invoking its CLI here.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
REVIEWED_ROOT = WORKSPACE_ROOT / 'scripts/reviewed_tcn_selector_v1'
REVIEWED_FILES = {
    'scripts/select_cuda_tcn_blend.py': 'b14edb9f2e609038020202fd57c7c279b33835ba7c330b8608c8476bee51ad5f',
    'scripts/train_cuda_tcn.py': '1afae39884da9d210a8ef695fb0c99636536bc688ce79961d502bf4c4b743e57',
    'sintef_project/postprocess.py': 'ffbfa7974df4d0363f655bed00da1778aebe485988bdaef0309ac488f17c903b',
}


def load_reviewed_selector() -> ModuleType:
    """Load exact reviewed code without changing unrelated package resolution."""
    for relative, expected in REVIEWED_FILES.items():
        path = REVIEWED_ROOT / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Reviewed selector dependency changed: {path}')
        if relative != 'scripts/select_cuda_tcn_blend.py':
            local = WORKSPACE_ROOT / relative
            if hashlib.sha256(local.read_bytes()).hexdigest() != expected:
                raise ValueError(f'Frozen workspace dependency changed: {local}')
    os.environ.setdefault('TDT4173_WORKSPACE_ROOT', str(WORKSPACE_ROOT))
    previous_path = sys.path.copy()
    try:
        # Preload the same byte-verified dependencies from the workspace.
        # Otherwise the snapshot's minimal packages could shadow other modules
        # when unittest imports the remaining project test suite.
        sys.path.insert(0, str(WORKSPACE_ROOT))
        importlib.import_module('scripts.train_cuda_tcn')
        importlib.import_module('sintef_project.postprocess')
        path = REVIEWED_ROOT / 'scripts/select_cuda_tcn_blend.py'
        spec = importlib.util.spec_from_file_location('_tdt4173_reviewed_tcn_selector', path)
        if spec is None or spec.loader is None:
            raise RuntimeError('Unable to load the reviewed TCN selector')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = previous_path


reviewed = load_reviewed_selector()
if __name__ == '__main__':
    with reviewed.threadpool_limits(limits=2):
        reviewed.main()
else:
    # Expose the actual implementation module, preserving helper globals for
    # callers and synthetic tests that replace ROOT with a fixture directory.
    sys.modules[__name__] = reviewed
