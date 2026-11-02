"""Freeze an additive81-column LPv2 GPU ablation with a fourth private bundle."""
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
from scripts.train_cuda_xgb745 import PARAMETERS, WEIGHTS, digest, require
from scripts.build_cuda_xgb608_package import zip_verified
from scripts.build_cuda_xgb664_package import reference_record
from sintef_project.lp_features_v2 import math_specification_v2

OLD_PROTOCOL_SHA = '4e09f0fe59973d7a9c2c5e5792feb30c29b8af04a67c27bafbe1386ae9da8182'
V2_MANIFEST_SHA = 'ec9a628e1e5a46bd4ecbf6688178ffc95c91d4fa3c5a7af592fe5b98cf96ffc7'
V2_FEATURES_SHA = '5dc566d5dd2f2e07b5dc4e5ce8ec85430a45c79cf439b3ecbeff4e1a32709084'
CODE_FILES = ['train_cuda_xgb745.py', 'cuda_xgb745_loader.py', 'cuda_xgb664_loader.py',
              'cuda_xgb608_loader.py', 'train_cuda_tcn.py', 'launch_cuda_xgb.py', 'launch_cuda_xgb745_dataset.py']
LOCAL_SOURCE = [*[f'scripts/{name}' for name in CODE_FILES], 'scripts/select_cuda_xgb745_blend.py',
                'scripts/verify_cuda_xgb745_return.py', 'scripts/build_cuda_xgb745_package.py',
                'scripts/build_cuda_xgb664_package.py', 'scripts/build_cuda_xgb608_package.py',
                'tests/test_cuda_xgb745.py', 'docs/cuda_xgb745.md', 'sintef_project/postprocess.py']


def main() -> None:
    """Package existing features and frozen code only; perform no fitting or upload."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-id', default='cuda-xgb745-package-001')
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
        cache = ROOT / 'artifacts/lp-v2-full-001'
        require(digest(cache / 'manifest.json') == V2_MANIFEST_SHA, 'Existing LPv2 cache manifest changed')
        m = json.loads((cache / 'manifest.json').read_text())
        require(m['status'] == 'complete' and m['feature_shape'] == [3647, 168, 81]
                and m['math'] == json.loads(json.dumps(math_specification_v2()))
                and len(m['cases']) == 3647, 'LPv2 cache incomplete')
        require(all(c['status'] == 0 and all(c[k] == 0 for k in ['max_minimum_slack_mm3', 'max_capacity_slack_mm3',
                        'max_bypass_slack_mm3', 'max_vest_slack_mm3']) for c in m['cases']), 'LPv2 solve/slack proof changed')
        require(digest(cache / 'features.npz') == m['artifacts']['features.npz'] == V2_FEATURES_SHA,
                'Existing81-column cache changed')
        for group in ['source', 'data']:
            for name, expected in m[group].items():
                require(digest(ROOT / name) == expected, 'LPv2 dependency changed: ' + name)
        for name, expected in m['artifacts'].items():
            require(digest(cache / name) == expected, 'LPv2 original artifact changed: ' + name)
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
        needed = {**m['source'], **m['data'],
                  (cache / 'manifest.json').relative_to(ROOT).as_posix(): V2_MANIFEST_SHA,
                  (cache / 'features.npz').relative_to(ROOT).as_posix(): V2_FEATURES_SHA}
        for name in set(needed) & set(existing):
            require(needed[name] == existing[name], 'LPv2 would overwrite an original dependency: ' + name)
        supplement = out / 'lp_v2'
        supplement.mkdir()
        records = []
        for name in sorted(set(needed) - set(existing)):
            target = supplement / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
            require(digest(target) == needed[name], 'Copy changed: ' + name)
            records.append({'path': name, 'sha256': digest(target), 'size_bytes': target.stat().st_size})
        (supplement / 'bundle_manifest.json').write_text(json.dumps({
            'version': 'cuda_xgb745_directional_lp_v2_supplement_v1', 'created_utc': report['timestamp_utc'],
            'payload_files': records, 'contains_labels': False, 'contains_anchor_predictions': False,
            'purpose': 'PRIVATE existing input-only81-column LPv2 cache plus missing verification sources'}, indent=2))
        supplement_zip = out / 'cuda-xgb745-lp-v2.zip'
        zip_verified(supplement, supplement_zip)
        p = json.loads(json.dumps(old))
        p.update(version='cuda_xgb_lp745_v1', created_utc=report['timestamp_utc'],
                 trainer_sha256=digest(ROOT / 'scripts/train_cuda_xgb745.py'),
                 shared_loader_sha256=digest(ROOT / 'scripts/cuda_xgb745_loader.py'),
                 prior664_loader_sha256=old['shared_loader_sha256'],
                 local_selector_sha256=digest(ROOT / 'scripts/select_cuda_xgb745_blend.py'),
                 return_verifier_sha256=digest(ROOT / 'scripts/verify_cuda_xgb745_return.py'),
                 dataset_launcher_sha256=digest(ROOT / 'scripts/launch_cuda_xgb745_dataset.py'),
                 v2_bundle_manifest_sha256=digest(supplement / 'bundle_manifest.json'),
                 v2_bundle_manifest_path=(supplement / 'bundle_manifest.json').relative_to(ROOT).as_posix(),
                 v2_bundle_archive_sha256=digest(supplement_zip),
                 v2_cache_directory=cache.relative_to(ROOT).as_posix(), v2_cache_manifest_sha256=V2_MANIFEST_SHA,
                 v2_features_sha256=V2_FEATURES_SHA, feature_columns=old['feature_columns'] + m['feature_names'],
                 preprocessing='Exact unchanged664 prefix plus81 existing dimensionless directional Haukeli LPv2 features.',
                 selection='Global0/.1/.25/.5 grid on2021, lower ties; positive winner must beat acceptedSHOP664 identity and raw664 same-weight control.2022 only independently audited lock.',
                 fit_commands='Synthetic745 smoke; separate2021 then independently eligible locked2022.',
                 caveats='Reused development. Additive feature combination, not new physics or a repetition of the old385-feature replacement. No public leaderboard tuning.')
        require(len(p['feature_columns']) == len(set(p['feature_columns'])) == 745, 'Feature collision/count changed')
        p['anchor'], p['control'] = {}, {}
        for year in [2021, 2022]:
            anchor = ROOT / f'artifacts/cuda-xgb664-select-{year}-001'
            am = json.loads((anchor / 'manifest.json').read_text())
            require(am['selected_weight'] == .25 and (year == 2021 or am['accepted']), 'SHOP anchor not accepted')
            p['anchor'][str(year)] = reference_record(anchor / 'manifest.json', anchor / f'selected_validation_{year}.npz')
            control = ROOT / f'artifacts/verify-cuda-xgb664-{year}-return-001/download/cuda-xgb664-{year}-001'
            p['control'][str(year)] = reference_record(control / 'manifest.json', control / f'validation_{year}.npz')
        p['feature_ablation'].update(original664_protocol_sha256=OLD_PROTOCOL_SHA, prefix=664, append=81,
                                    lp_v2=m['math'], original664_feature_ablation=old['feature_ablation'])
        protocol_path = ROOT / 'docs/cuda_xgb745_protocol.json'
        with protocol_path.open('x') as stream:
            json.dump(p, stream, indent=2)
        code = out / 'code'
        code.mkdir()
        for name in CODE_FILES:
            shutil.copy2(ROOT / 'scripts' / name, code / name)
        shutil.copy2(protocol_path, code / 'cuda_xgb745_protocol.json')
        shutil.copy2(ROOT / 'docs/cuda_xgb745.md', code / 'README.md')
        (code / 'package_manifest.json').write_text(json.dumps({'private': True,
            'files': [{'path': x.name, 'sha256': digest(x), 'size_bytes': x.stat().st_size} for x in sorted(code.iterdir())]}, indent=2))
        dataset = out / 'dataset'
        dataset.mkdir()
        shutil.copytree(code, dataset / 'code')
        shutil.copytree(supplement, dataset / 'lp_v2')
        (dataset / 'dataset_manifest.json').write_text(json.dumps({'required_visibility': 'PRIVATE',
            'protocol_sha256': digest(protocol_path), 'contains_original_bundles': False,
            'files': [{'path': x.relative_to(dataset).as_posix(), 'sha256': digest(x), 'size_bytes': x.stat().st_size}
                      for x in sorted(dataset.rglob('*')) if x.is_file()],
            'execution': 'Requires original three bundles. Explicit smoke,2021 and independent-lock-gated2022. No autoexecution.'}, indent=2))
        dataset_zip = out / 'cuda-xgb745-private-dataset.zip'
        zip_verified(dataset, dataset_zip)
        report.update(status='complete', protocol_sha256=digest(protocol_path), source={x: digest(ROOT / x) for x in LOCAL_SOURCE},
                      dataset_archive_sha256=digest(dataset_zip), dataset_archive_bytes=dataset_zip.stat().st_size,
                      v2_archive_sha256=digest(supplement_zip), added_files=records,
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
