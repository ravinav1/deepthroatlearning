"""Append only the complete audited integer cache to the unchanged664-column stage."""
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


def stage_inputs(bundle: Path, supplement: Path, shop_bundle: Path, integer_bundle: Path,
                 destination: Path, protocol: dict) -> tuple:
    """Stage all three original bundles plus a strictly disjoint integer water-budget supplement."""
    require(protocol['prior664_loader_sha256'] == PRIOR664_LOADER_SHA256, '664 loader pin changed')
    base, extra, shop, old_stage = prior.stage_inputs(bundle, supplement, shop_bundle, destination / 'prior664', protocol)
    integer_root = destination / 'lp_integer'
    integer = prior.prior.core.stage_bundle(integer_bundle, integer_root, protocol['integer_bundle_manifest_sha256'],
                                      protocol['integer_bundle_archive_sha256'])
    old_manifest = {'payload_files': base['payload_files'] + extra['payload_files'] + shop['payload_files']}
    merged = prior.prior.merge_stages(old_stage, integer_root, destination / 'merged748', old_manifest, integer)
    return base, extra, shop, integer, merged


def build_verified_features(stage: Path, year: int, protocol: dict) -> tuple:
    """Preserve every original feature value and append only84 pinned integer water-budget values."""
    import numpy as np
    inherited = {**protocol, 'feature_columns': protocol['feature_columns'][:664]}
    x, labels, train, mapping, meta, baseline = prior.build_verified_features(stage, year, inherited)
    from sintef_project.integer_water_budget_features import add_integer_water_features
    cache = stage / protocol['integer_cache_directory']
    require(digest(cache / 'manifest.json') == protocol['integer_cache_manifest_sha256'], 'integer water-budget cache changed')
    require(digest(cache / 'features.npz') == protocol['integer_features_sha256'], 'integer water-budget feature bytes changed')
    require(x.columns.tolist() == protocol['feature_columns'][:664], 'Original664 prefix changed')
    audit_path = stage / protocol['integer_physical_audit_path']
    require(digest(audit_path) == protocol['integer_physical_audit_sha256'], 'Integer physical audit changed')
    import json
    audit = json.loads(audit_path.read_text())
    require(audit['status'] == 'passed' and audit['case_count'] == 3647
            and audit['cache_manifest_sha256'] == protocol['integer_cache_manifest_sha256']
            and audit['cache_protocol_sha256'] == protocol['integer_cache_protocol_sha256']
            and audit['features_sha256'] == protocol['integer_features_sha256']
            and audit['all_cases_physical_and_replay_verified'] and audit['compatible_receipts_verified']
            and audit['compatible_case_count'] == protocol['integer_compatible_case_count'], 'Integer physical proof incomplete')
    result, integer_spec = add_integer_water_features(x, stage, cache,
        protocol['integer_cache_manifest_sha256'], protocol['integer_cache_protocol_sha256'])
    integer_spec['physical_audit_sha256'] = protocol['integer_physical_audit_sha256']
    integer_spec['compatible_case_count'] = protocol['integer_compatible_case_count']
    require(result.iloc[:, :664].equals(x), 'Original664 values or dtypes changed')
    require(result.columns.tolist() == protocol['feature_columns'] and result.shape[1] == 748
            and np.isfinite(result.to_numpy()).all(), 'Invalid748 feature schema or values')
    meta['feature_spec']['lp_integer'] = integer_spec
    return result, labels, train, mapping, meta, baseline
