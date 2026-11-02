"""Freeze an additive84-column integer water-budget GPU ablation with a fourth private bundle."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_cuda_xgb748 import PARAMETERS, WEIGHTS, digest, require
from scripts.build_cuda_xgb608_package import zip_verified
from scripts.build_cuda_xgb664_package import reference_record
from sintef_project.integer_water_budget_features import COLUMNS, feature_specification
from scripts.continue_integer_water_budget_cache_v2 import check_compatibility, BASE_PROTOCOL_SHA

OLD_PROTOCOL_SHA = '4e09f0fe59973d7a9c2c5e5792feb30c29b8af04a67c27bafbe1386ae9da8182'
CACHE_PROTOCOL_SHA = '7571930990cadc542c7ad4f47e7f1195d7a239885c887ad45b051afd9b9377d9'
CODE_FILES = ['train_cuda_xgb748.py', 'cuda_xgb748_loader.py', 'cuda_xgb664_loader.py',
              'cuda_xgb608_loader.py', 'train_cuda_tcn.py', 'launch_cuda_xgb.py', 'launch_cuda_xgb748_dataset.py']
LOCAL_SOURCE = [*[f'scripts/{name}' for name in CODE_FILES], 'scripts/select_cuda_xgb748_blend.py',
                'scripts/verify_cuda_xgb748_return.py', 'scripts/build_cuda_xgb748_package.py',
                'scripts/build_cuda_xgb664_package.py', 'scripts/build_cuda_xgb608_package.py',
                'tests/test_cuda_xgb748.py', 'docs/cuda_xgb748.md', 'sintef_project/postprocess.py',
                'sintef_project/integer_water_budget_features.py', 'scripts/continue_integer_water_budget_cache_v2.py']


def require_complete_proof(manifest: dict, cache_protocol: dict, audit: dict,
                           manifest_sha: str, protocol_sha: str) -> None:
    """Reject partial, unproven or unaudited cache identities before feature access."""
    require(protocol_sha == CACHE_PROTOCOL_SHA and manifest['protocol_sha256'] == protocol_sha,
            'Unexpected frozen integer cache execution protocol')
    require(manifest['status'] == 'complete' and manifest['all_cases_proven_and_replayed']
            and not manifest['failures'] and len(manifest['case_ids']) == 3647
            and len(set(manifest['case_ids'])) == 3647
            and set(manifest['completed_cases']) == {str(x) for x in manifest['case_ids']},
            'Partial or unproven integer cache forbidden')
    require(manifest['case_ids'] == cache_protocol['case_ids'] and manifest['case_dates'] == cache_protocol['case_dates']
            and manifest['source'] == cache_protocol['source'] and manifest['data'] == cache_protocol['data']
            and manifest['feature_specification'] == cache_protocol['feature_specification'] == feature_specification(),
            'Integer physical recipe/source/data/IDs changed')
    require(audit['status'] == 'passed' and audit['cache_manifest_sha256'] == manifest_sha
            and audit['cache_protocol_sha256'] == protocol_sha
            and audit['features_sha256'] == manifest['features_sha256'] and audit['case_count'] == 3647
            and audit['all_cases_physical_and_replay_verified'] is True
            and audit['compatible_receipts_verified'] is True
            and isinstance(audit['compatible_case_count'], int) and 0 <= audit['compatible_case_count'] <= 3647,
            'Independent complete physical/replay/compatibility proof required')


def main() -> None:
    """Package existing features and frozen code only; perform no fitting or upload."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-id', default='cuda-xgb748-package-001')
    ap.add_argument('--cache', type=Path, required=True)
    ap.add_argument('--cache-manifest-sha256', required=True)
    ap.add_argument('--cache-audit', type=Path, required=True)
    ap.add_argument('--cache-audit-sha256', required=True)
    args = ap.parse_args()
    require(Path(args.run_id).name == args.run_id and args.run_id not in ['', '.', '..'], 'Invalid immutable run ID')
    out = ROOT / 'artifacts' / args.run_id
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    report = {'status': 'initializing', 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'command': [sys.executable, *sys.argv], 'training': False, 'upload': False,
              'kaggle_submission': None, 'environment': {'python': sys.version, 'platform': platform.platform(),
                'packages': {x: importlib.metadata.version(x) for x in ['numpy', 'pandas', 'scipy', 'scikit-learn']}}}
    try:
        old_path = ROOT / 'docs/cuda_xgb664_protocol.json'
        require(digest(old_path) == OLD_PROTOCOL_SHA, 'Original664 protocol changed')
        old = json.loads(old_path.read_text())
        require(old['model_parameters'] == PARAMETERS and old['candidate_weights'] == WEIGHTS, 'Model recipe changed')
        cache = args.cache.resolve()
        audit_path = args.cache_audit.resolve()
        require(cache.is_relative_to(ROOT) and audit_path.is_relative_to(ROOT), 'Portable cache/audit must belong to repository')
        require(digest(cache / 'manifest.json') == args.cache_manifest_sha256, 'External integer manifest hash differs')
        require(digest(cache / 'protocol.json') == CACHE_PROTOCOL_SHA, 'Frozen integer execution protocol changed')
        require(digest(audit_path) == args.cache_audit_sha256, 'External independent physical audit changed')
        m = json.loads((cache / 'manifest.json').read_text())
        cp = json.loads((cache / 'protocol.json').read_text())
        audit = json.loads(audit_path.read_text())
        require_complete_proof(m, cp, audit, args.cache_manifest_sha256, CACHE_PROTOCOL_SHA)
        base_path = ROOT / cp['compatible_base']['path']
        require(digest(base_path) == BASE_PROTOCOL_SHA, 'Compatible original mathematical protocol changed')
        check_compatibility(json.loads(base_path.read_text()), cp)
        require(m['prefix_columns'] == cp['prefix_columns'] == old['feature_columns'], 'Integer664 prefix differs')
        for group in ['source', 'data']:
            for name, expected in cp[group].items():
                require(digest(ROOT / name) == expected, 'Integer dependency changed: ' + name)
        for name, expected in m['artifacts'].items():
            require(digest(cache / name) == expected, 'Completed integer artifact changed: ' + name)
        require(digest(cache / 'features.npz') == m['features_sha256'] == m['artifacts']['features.npz'], 'Integer feature bytes changed')
        compatible_count = 0
        for case_id, expected in m['completed_cases'].items():
            folder = cache / 'cases' / f'case_{int(case_id):04d}'
            receipt_path = folder / 'complete.json'
            require(digest(receipt_path) == expected, 'Completed integer receipt changed')
            receipt = json.loads(receipt_path.read_text())
            require(receipt['status'] == 'complete' and receipt['protocol_sha256'] == CACHE_PROTOCOL_SHA, 'Integer case incomplete')
            for name, expected_sha in receipt['artifacts'].items():
                require(digest(folder / name) == expected_sha, 'Integer case payload changed')
            if 'compatibility' in receipt:
                compatible_count += 1
                compat = receipt['compatibility']
                original = folder / compat['original_complete_path']
                require(compat['original_protocol_sha256'] == BASE_PROTOCOL_SHA
                        and compat['original_manifest_sha256'] == m['reuse_manifest_sha256']
                        and digest(cache / 'reuse_manifest.json') == m['reuse_manifest_sha256']
                        and digest(original) == compat['original_complete_sha256']
                        and compat['all_original_payload_bytes_preserved'], 'Imported original receipt changed')
                old_receipt = json.loads(original.read_text())
                require(old_receipt['protocol_sha256'] == BASE_PROTOCOL_SHA and old_receipt['case_id'] == int(case_id), 'Original case identity changed')
                for name, expected_sha in old_receipt['artifacts'].items():
                    require(digest(folder / name) == expected_sha, 'Imported original payload bytes changed')
        require(compatible_count == audit['compatible_case_count'], 'Independent compatibility count differs')
        existing = {}
        bundle_paths = [ROOT / 'artifacts/kaggle-neural-bundle-001/payload/bundle_manifest.json',
                        ROOT / 'artifacts/cuda-xgb608-package-001/supplement/bundle_manifest.json',
                        ROOT / old['shop_bundle_manifest_path']]
        for path, expected in zip(bundle_paths, [old['bundle_manifest_sha256'], old['supplement_manifest_sha256'],
                                               old['shop_bundle_manifest_sha256']]):
            require(digest(path) == expected, 'Original bundle changed')
            for record in json.loads(path.read_text())['payload_files']:
                require(record['path'] not in existing, 'Original bundle collision')
                existing[record['path']] = record['sha256']
        needed = {**cp['source'], **cp['data'],
                  (cache / 'manifest.json').relative_to(ROOT).as_posix(): args.cache_manifest_sha256,
                  (cache / 'protocol.json').relative_to(ROOT).as_posix(): CACHE_PROTOCOL_SHA,
                  (cache / 'features.npz').relative_to(ROOT).as_posix(): m['features_sha256'],
                  audit_path.relative_to(ROOT).as_posix(): args.cache_audit_sha256,
                  cp['prefix_protocol']: cp['prefix_protocol_sha256'],
                  (ROOT / cp['continuous_cache'] / 'manifest.json').relative_to(ROOT).as_posix(): cp['continuous_cache_manifest_sha256'],
                  cp['compatible_base']['path']: BASE_PROTOCOL_SHA}
        for name in set(needed) & set(existing):
            require(needed[name] == existing[name], 'integer water-budget would overwrite an original dependency: ' + name)
        supplement = out / 'lp_integer'
        supplement.mkdir()
        records = []
        for name in sorted(set(needed) - set(existing)):
            target = supplement / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
            require(digest(target) == needed[name], 'Copy changed: ' + name)
            records.append({'path': name, 'sha256': digest(target), 'size_bytes': target.stat().st_size})
        (supplement / 'bundle_manifest.json').write_text(json.dumps({
            'version': 'cuda_xgb748_integer_water_budget_supplement_v1', 'created_utc': report['timestamp_utc'],
            'payload_files': records, 'contains_labels': False, 'contains_anchor_predictions': False,
            'purpose': 'PRIVATE existing input-only84-column integer water-budget cache plus missing verification sources'}, indent=2))
        supplement_zip = out / 'cuda-xgb748-lp-integer.zip'
        zip_verified(supplement, supplement_zip)
        p = json.loads(json.dumps(old))
        p.update(version='cuda_xgb_integer748_v1', created_utc=report['timestamp_utc'],
                 trainer_sha256=digest(ROOT / 'scripts/train_cuda_xgb748.py'),
                 shared_loader_sha256=digest(ROOT / 'scripts/cuda_xgb748_loader.py'),
                 prior664_loader_sha256=old['shared_loader_sha256'],
                 local_selector_sha256=digest(ROOT / 'scripts/select_cuda_xgb748_blend.py'),
                 return_verifier_sha256=digest(ROOT / 'scripts/verify_cuda_xgb748_return.py'),
                 dataset_launcher_sha256=digest(ROOT / 'scripts/launch_cuda_xgb748_dataset.py'),
                 integer_bundle_manifest_sha256=digest(supplement / 'bundle_manifest.json'),
                 integer_bundle_manifest_path=(supplement / 'bundle_manifest.json').relative_to(ROOT).as_posix(),
                 integer_bundle_archive_sha256=digest(supplement_zip),
                 integer_cache_directory=cache.relative_to(ROOT).as_posix(), integer_cache_manifest_sha256=args.cache_manifest_sha256,
                 integer_cache_protocol_sha256=CACHE_PROTOCOL_SHA,
                 integer_physical_audit_path=audit_path.relative_to(ROOT).as_posix(),
                 integer_physical_audit_sha256=args.cache_audit_sha256, integer_compatible_case_count=compatible_count,
                 integer_features_sha256=m['features_sha256'], feature_columns=old['feature_columns'] + COLUMNS,
                 preprocessing='Exact unchanged664 prefix plus84 complete independently audited integer water-budget features.',
                 selection='Global0/.1/.25/.5 grid on2021, lower ties; positive winner must beat acceptedSHOP664 identity and raw664 same-weight control.2022 only independently audited lock.',
                 fit_commands='Synthetic748 smoke; separate2021 then independently eligible locked2022.',
                 caveats='Reused development. Only complete independently physically audited integer cache; no partial or relaxed fallback. No public leaderboard tuning.')
        require(len(p['feature_columns']) == len(set(p['feature_columns'])) == 748, 'Feature collision/count changed')
        p['anchor'], p['control'] = {}, {}
        for year in [2021, 2022]:
            anchor = ROOT / f'artifacts/cuda-xgb664-select-{year}-001'
            am = json.loads((anchor / 'manifest.json').read_text())
            require(am['selected_weight'] == .25 and (year == 2021 or am['accepted']), 'SHOP anchor not accepted')
            p['anchor'][str(year)] = reference_record(anchor / 'manifest.json', anchor / f'selected_validation_{year}.npz')
            control = ROOT / f'artifacts/verify-cuda-xgb664-{year}-return-001/download/cuda-xgb664-{year}-001'
            p['control'][str(year)] = reference_record(control / 'manifest.json', control / f'validation_{year}.npz')
        p['feature_ablation'].update(original664_protocol_sha256=OLD_PROTOCOL_SHA, prefix=664, append=84,
                                    lp_integer=feature_specification(), original664_feature_ablation=old['feature_ablation'])
        protocol_path = ROOT / 'docs/cuda_xgb748_protocol.json'
        with protocol_path.open('x') as stream:
            json.dump(p, stream, indent=2)
        code = out / 'code'
        code.mkdir()
        for name in CODE_FILES:
            shutil.copy2(ROOT / 'scripts' / name, code / name)
        shutil.copy2(protocol_path, code / 'cuda_xgb748_protocol.json')
        shutil.copy2(ROOT / 'docs/cuda_xgb748.md', code / 'README.md')
        (code / 'package_manifest.json').write_text(json.dumps({'private': True,
            'files': [{'path': x.name, 'sha256': digest(x), 'size_bytes': x.stat().st_size} for x in sorted(code.iterdir())]}, indent=2))
        dataset = out / 'dataset'
        dataset.mkdir()
        shutil.copytree(code, dataset / 'code')
        shutil.copytree(supplement, dataset / 'lp_integer')
        (dataset / 'dataset_manifest.json').write_text(json.dumps({'required_visibility': 'PRIVATE',
            'protocol_sha256': digest(protocol_path), 'contains_original_bundles': False,
            'files': [{'path': x.relative_to(dataset).as_posix(), 'sha256': digest(x), 'size_bytes': x.stat().st_size}
                      for x in sorted(dataset.rglob('*')) if x.is_file()],
            'execution': 'Requires original three bundles. Explicit smoke,2021 and independent-lock-gated2022. No autoexecution.'}, indent=2))
        dataset_zip = out / 'cuda-xgb748-private-dataset.zip'
        zip_verified(dataset, dataset_zip)
        report.update(status='complete', protocol_sha256=digest(protocol_path), source={x: digest(ROOT / x) for x in LOCAL_SOURCE},
                      dataset_archive_sha256=digest(dataset_zip), dataset_archive_bytes=dataset_zip.stat().st_size,
                      integer_archive_sha256=digest(supplement_zip), added_files=records,
                      original_cache_artifacts_verified=len(m['artifacts']), needs_independent_audit=True)
        print(json.dumps({k: report[k] for k in ['status', 'protocol_sha256', 'dataset_archive_sha256', 'dataset_archive_bytes']}), flush=True)
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        for name in LOCAL_SOURCE:
            target = out / 'source' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        report['runtime_seconds'] = time.monotonic() - started
        report['artifacts'] = {x.relative_to(out).as_posix(): digest(x) for x in out.rglob('*') if x.is_file() and x.name != 'manifest.json'}
        (out / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
