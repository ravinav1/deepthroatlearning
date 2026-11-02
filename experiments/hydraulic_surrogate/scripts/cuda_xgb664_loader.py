"""Extend the frozen608 input stage with a disjoint input-only56-column cache."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil

PRIOR_LOADER_SHA256 = '4a3f7422989c49e830756bb9c660c2f6b601185be3989099616debb9a79aaf52'


def load_prior():
    """Import the byte-pinned608 loader without modifying its original source."""
    import hashlib
    path = Path(__file__).with_name('cuda_xgb608_loader.py')
    if hashlib.sha256(path.read_bytes()).hexdigest() != PRIOR_LOADER_SHA256:
        raise ValueError('Original608 loader changed')
    spec = importlib.util.spec_from_file_location('tdt4173_pinned608_loader', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prior = load_prior()
digest, require = prior.digest, prior.require


def stage_inputs(bundle: Path, supplement: Path, shop_bundle: Path, destination: Path,
                 protocol: dict) -> tuple:
    """Stage the unchanged original bundles plus an exclusively additive SHOP bundle."""
    require(protocol['prior_loader_sha256'] == PRIOR_LOADER_SHA256, '608 loader pin changed')
    base, extra, old_stage = prior.stage_inputs(bundle, supplement, destination / 'prior608', protocol)
    shop_root = destination / 'shop'
    shop = prior.core.stage_bundle(shop_bundle, shop_root, protocol['shop_bundle_manifest_sha256'],
                                   protocol['shop_bundle_archive_sha256'])
    old_manifest = {'payload_files': base['payload_files'] + extra['payload_files']}
    merged = prior.merge_stages(old_stage, shop_root, destination / 'merged664', old_manifest, shop)
    return base, extra, shop, merged


def build_verified_features(stage: Path, year: int, protocol: dict) -> tuple:
    """Build the exact608 prefix and append only byte-verified label-free SHOP inputs."""
    import numpy as np
    inherited = {**protocol, 'feature_columns': protocol['feature_columns'][:608]}
    x, labels, train, mapping, meta, baseline = prior.build_verified_features(stage, year, inherited)
    from sintef_project.lp_shop_features import add_shop_features
    cache = stage / protocol['shop_cache_directory']
    require(digest(cache / 'manifest.json') == protocol['shop_cache_manifest_sha256'], 'SHOP cache changed')
    require(digest(cache / 'features.npz') == protocol['shop_features_sha256'], 'SHOP feature bytes changed')
    require(x.columns.tolist() == protocol['feature_columns'][:608], 'Original608 prefix changed')
    x, shop_spec = add_shop_features(x, stage, cache)
    require(x.columns.tolist() == protocol['feature_columns'] and x.shape[1] == 664
            and np.isfinite(x.to_numpy()).all(), 'Invalid664 feature schema or values')
    meta['feature_spec']['lp_shop'] = shop_spec
    return x, labels, train, mapping, meta, baseline
