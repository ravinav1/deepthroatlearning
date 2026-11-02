"""Verify separate original/supplement bundles and reproduce the frozen608 features."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

CORE_SHA256 = '1afae39884da9d210a8ef695fb0c99636536bc688ce79961d502bf4c4b743e57'


def load_core():
    """Import only the original byte-pinned staging/redaction helper beside this file."""
    import hashlib
    path = Path(__file__).with_name('train_cuda_tcn.py')
    if hashlib.sha256(path.read_bytes()).hexdigest() != CORE_SHA256:
        raise ValueError('Original portable staging helper changed')
    spec = importlib.util.spec_from_file_location('tdt4173_original_staging_core', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


core = load_core()
digest, require, safe_relative = core.digest, core.require, core.safe_relative


def merge_stages(base: Path, supplement: Path, destination: Path,
                 base_manifest: dict, supplement_manifest: dict) -> Path:
    """Make a separately verified union, rejecting every overlay of original inputs."""
    entries = {}
    for source, manifest in [(base, base_manifest), (supplement, supplement_manifest)]:
        for record in manifest['payload_files']:
            name = record['path']
            safe_relative(name)
            require(name not in entries, f'Supplement overlaps original input: {name}')
            entries[name] = (source / name, record)
    reuse = destination.exists() and any(destination.iterdir())
    if reuse:
        paths = list(destination.rglob('*'))
        require(not any(path.is_symlink() for path in paths), 'Merged stage contains symlinks')
        require({p.relative_to(destination).as_posix() for p in paths if p.is_file()} == set(entries),
                'Merged stage has missing/unlisted files')
    destination.mkdir(parents=True, exist_ok=True)
    for name, (source, record) in entries.items():
        require(source.is_file() and not source.is_symlink() and digest(source) == record['sha256']
                and source.stat().st_size == record['size_bytes'], 'Source stage changed before merging')
        target = destination / safe_relative(name)
        if not reuse:
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open('rb') as incoming, target.open('xb') as outgoing:
                shutil.copyfileobj(incoming, outgoing)
        require(digest(target) == record['sha256'] and target.stat().st_size == record['size_bytes'],
                f'Merged input hash/size mismatch: {name}')
    return destination.resolve()


def stage_inputs(bundle: Path, supplement: Path, destination: Path,
                 protocol: dict) -> tuple[dict, dict, Path]:
    """Stage unchanged original inputs and only missing608 inputs into separate roots."""
    require(protocol['core_loader_sha256'] == CORE_SHA256, 'Original core helper identity changed')
    base_root, extra_root = destination / 'base', destination / 'supplement'
    base = core.stage_bundle(bundle, base_root, protocol['bundle_manifest_sha256'], protocol['bundle_archive_sha256'])
    extra = core.stage_bundle(supplement, extra_root, protocol['supplement_manifest_sha256'], protocol['supplement_archive_sha256'])
    merged = merge_stages(base_root, extra_root, destination / 'merged', base, extra)
    return base, extra, merged


def portable_spec(spec: dict) -> dict:
    """Remove only the two documented local cache-directory paths for comparison."""
    copy = json.loads(json.dumps(spec))
    for key in ['lp', 'lp_joint']:
        copy[key].pop('cache_directory', None)
    return copy


def build_verified_features(stage: Path, year: int, protocol: dict) -> tuple:
    """Recreate exact521 context inputs, then append87 joint inputs with target redaction."""
    import numpy as np
    import pandas as pd
    for name, module in list(sys.modules.items()):
        if name.startswith('sintef_project') and getattr(module, '__file__', None):
            require(Path(module.__file__).resolve().is_relative_to(stage),
                    'Foreign feature modules loaded; use a fresh process')
    sys.path.insert(0, str(stage))
    from sintef_project.features import build_features
    from sintef_project.physics import add_physics_features
    from sintef_project.context_features import add_context_features
    from sintef_project.lp_features import add_lp_features
    from sintef_project.lp_context import add_lp_context_features
    from sintef_project.normalized import add_normalized_features
    from sintef_project.lp_joint_plant import add_lp_joint_features
    reference_path = stage / 'artifacts/lgb-normalized-lp-context-joint-001/manifest.json'
    require(digest(reference_path) == protocol['baseline_manifest_sha256'], 'Frozen608 reference manifest changed')
    reference = json.loads(reference_path.read_text())
    for relative, expected in protocol['feature_source_sha256'].items():
        require(digest(stage / relative) == expected == reference['source'][relative],
                f'Frozen608 feature source changed: {relative}')
    target = (stage / 'sintef_project/data/kernel/Unit_commitment_decisions.csv').resolve()
    original_read = pd.read_csv

    def guarded_read(path, *args, **kwargs):
        if isinstance(path, (str, Path)) and Path(path).resolve() == target:
            if kwargs.get('usecols') == ['Run No', 'starttime']:
                return original_read(path, *args, **kwargs)
            require(not args and not kwargs, 'Frozen full-target read signature changed')
            return core.read_targets_through_year(target, year, original_read)
        return original_read(path, *args, **kwargs)

    with patch.object(pd, 'read_csv', guarded_read):
        x, labels, train, mapping, meta = build_features(stage)
        x, physics = add_physics_features(x, stage)
        x, context = add_context_features(x, stage)
        x, lp = add_lp_features(x, stage)
        x, lp_context = add_lp_context_features(x, stage)
        x, normalized = add_normalized_features(x, stage)
        require(x.columns.tolist() == protocol['feature_columns'][:521], 'Exact521 context prefix changed')
        x, joint = add_lp_joint_features(x, stage)
    require(x.columns.tolist() == protocol['feature_columns'] == reference['features']
            and x.shape[1] == 608 and np.isfinite(x.to_numpy()).all(), 'Frozen608 schema or finite-value check failed')
    meta['feature_spec'].update(physics=physics, context=context, lp=lp, lp_context=lp_context,
                                normalized=normalized, lp_joint=joint)
    require(portable_spec(meta['feature_spec']) == portable_spec(reference['feature_spec']),
            'Feature formulas differ beyond portable cache paths')
    return x, labels, train, mapping, meta, reference
