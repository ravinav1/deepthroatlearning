"""Append the existing directional LPv2 cache to the unchanged664-column stage."""
from __future__ import annotations

import importlib.util
from pathlib import Path

PRIOR664_LOADER_SHA256 = '7f36d9c2a9c823f0e8ca90f02a0556a88ce2adaca718751be5af9a1c5a2043ac'


def load_prior():
    """Import the immutable664 loader and its already pinned608 ancestor."""
    import hashlib
    path = Path(__file__).with_name('cuda_xgb664_loader.py')
    if hashlib.sha256(path.read_bytes()).hexdigest() != PRIOR664_LOADER_SHA256:
        raise ValueError('Original664 loader changed')
    spec = importlib.util.spec_from_file_location('tdt4173_pinned664_loader', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prior = load_prior()
digest, require = prior.digest, prior.require


def stage_inputs(bundle: Path, supplement: Path, shop_bundle: Path, v2_bundle: Path,
                 destination: Path, protocol: dict) -> tuple:
    """Stage all three original bundles plus a strictly disjoint LPv2 supplement."""
    require(protocol['prior664_loader_sha256'] == PRIOR664_LOADER_SHA256, '664 loader pin changed')
    base, extra, shop, old_stage = prior.stage_inputs(bundle, supplement, shop_bundle, destination / 'prior664', protocol)
    v2_root = destination / 'lp_v2'
    v2 = prior.prior.core.stage_bundle(v2_bundle, v2_root, protocol['v2_bundle_manifest_sha256'],
                                      protocol['v2_bundle_archive_sha256'])
    old_manifest = {'payload_files': base['payload_files'] + extra['payload_files'] + shop['payload_files']}
    merged = prior.prior.merge_stages(old_stage, v2_root, destination / 'merged745', old_manifest, v2)
    return base, extra, shop, v2, merged


def build_verified_features(stage: Path, year: int, protocol: dict) -> tuple:
    """Preserve every original feature value and append only81 pinned LPv2 values."""
    import numpy as np
    inherited = {**protocol, 'feature_columns': protocol['feature_columns'][:664]}
    x, labels, train, mapping, meta, baseline = prior.build_verified_features(stage, year, inherited)
    from sintef_project.lp_features_v2 import add_lp_features_v2
    cache = stage / protocol['v2_cache_directory']
    require(digest(cache / 'manifest.json') == protocol['v2_cache_manifest_sha256'], 'LPv2 cache changed')
    require(digest(cache / 'features.npz') == protocol['v2_features_sha256'], 'LPv2 feature bytes changed')
    require(x.columns.tolist() == protocol['feature_columns'][:664], 'Original664 prefix changed')
    result, v2_spec = add_lp_features_v2(x, stage, cache)
    require(result.iloc[:, :664].equals(x), 'Original664 values or dtypes changed')
    require(result.columns.tolist() == protocol['feature_columns'] and result.shape[1] == 745
            and np.isfinite(result.to_numpy()).all(), 'Invalid745 feature schema or values')
    meta['feature_spec']['lp_v2'] = v2_spec
    return result, labels, train, mapping, meta, baseline
