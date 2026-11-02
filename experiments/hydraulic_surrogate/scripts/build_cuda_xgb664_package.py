"""Freeze664 GPU ablation with a separate additive private SHOP input bundle."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import importlib.metadata
import os
from pathlib import Path
import platform
import shlex
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_cuda_xgb664 import PARAMETERS, WEIGHTS, digest, require
from scripts.build_cuda_xgb608_package import zip_verified
from sintef_project.lp_shop_features import COLUMNS, feature_specification

CODE_FILES = ['train_cuda_xgb664.py', 'cuda_xgb664_loader.py', 'cuda_xgb608_loader.py',
              'train_cuda_tcn.py', 'launch_cuda_xgb.py', 'launch_cuda_xgb664_dataset.py']
LOCAL_SOURCE = [*[f'scripts/{name}' for name in CODE_FILES], 'scripts/select_cuda_xgb664_blend.py',
                'scripts/verify_cuda_xgb664_return.py',
                'scripts/build_cuda_xgb664_package.py', 'tests/test_cuda_xgb664.py',
                'docs/cuda_xgb664.md', 'sintef_project/postprocess.py']


def reference_record(manifest_path: Path, prediction_path: Path) -> dict:
    """Pin existing prediction files without decoding any target or probability."""
    manifest = json.loads(manifest_path.read_text())
    require(manifest['status'] == 'complete', 'Reference incomplete')
    require(digest(prediction_path) == manifest['artifacts'][prediction_path.name], 'Reference prediction changed')
    return {'manifest': manifest_path.relative_to(ROOT).as_posix(), 'manifest_sha256': digest(manifest_path),
            'prediction': prediction_path.relative_to(ROOT).as_posix(), 'prediction_sha256': digest(prediction_path)}


def main() -> None:
    """Prepare a fully pinned private code/cache dataset; never upload or train."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', default='cuda-xgb664-package-001')
    parser.add_argument('--cache', type=Path, default=ROOT / 'artifacts/lp-shop-full-001')
    args = parser.parse_args()
    out = ROOT / 'artifacts' / args.run_id
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    report = {'run_id': args.run_id, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'command': shlex.join([sys.executable, *sys.argv]), 'status': 'initializing',
              'training': False, 'upload': False, 'kaggle_submission': None,
              'environment': {'python': sys.version, 'platform': platform.platform(), 'machine': platform.machine(), 'cpu_count': os.cpu_count(), 'packages': {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'PyYAML']}}}
    try:
        old_protocol_path = ROOT / 'docs/cuda_xgb608_protocol.json'
        old = json.loads(old_protocol_path.read_text())
        require(old['model_parameters'] == PARAMETERS and old['candidate_weights'] == WEIGHTS, '608 parameters changed')
        cache = args.cache.resolve()
        cache_manifest = json.loads((cache / 'manifest.json').read_text())
        require(cache_manifest['status'] == 'complete' and cache_manifest['all_replays_bitwise']
                and cache_manifest['diagnostic36_raw_agreement_bitwise'], 'SHOP cache not complete/replayed')
        require(cache_manifest['feature_specification'] == feature_specification()
                and cache_manifest['prefix_feature_columns'] == old['feature_columns'], 'SHOP feature definition differs')
        require(digest(cache / 'features.npz') == cache_manifest['artifacts']['features.npz'], 'SHOP features changed')
        for group in ['source', 'data']:
            for name, expected in cache_manifest[group].items():
                require(digest(ROOT / name) == expected, f'SHOP dependency changed: {name}')
        original_path = ROOT / 'artifacts/kaggle-neural-bundle-001/payload/bundle_manifest.json'
        extra_path = ROOT / 'artifacts/cuda-xgb608-package-001/supplement/bundle_manifest.json'
        require(digest(original_path) == old['bundle_manifest_sha256']
                and digest(extra_path) == old['supplement_manifest_sha256'], 'Original608 bundles changed')
        existing = {}
        for manifest_path in [original_path, extra_path]:
            for record in json.loads(manifest_path.read_text())['payload_files']:
                require(record['path'] not in existing, 'Original608 bundle collision')
                existing[record['path']] = record['sha256']
        all_needed = {**cache_manifest['source'], **cache_manifest['data'],
                      (cache / 'manifest.json').relative_to(ROOT).as_posix(): digest(cache / 'manifest.json'),
                      (cache / 'features.npz').relative_to(ROOT).as_posix(): digest(cache / 'features.npz')}
        for name, expected in all_needed.items():
            if name in existing:
                require(existing[name] == expected, f'SHOP overlays a different608 dependency: {name}')
        supplement = out / 'shop'
        supplement.mkdir()
        records = []
        for name in sorted(set(all_needed) - set(existing)):
            source, target = ROOT / name, supplement / name
            require(digest(source) == all_needed[name], 'SHOP source bytes changed during packaging')
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            records.append({'path': name, 'sha256': digest(target), 'size_bytes': target.stat().st_size})
        require(not set(existing).intersection(r['path'] for r in records), 'SHOP bundle not disjoint')
        shop_bundle_manifest = {'version': 'cuda_xgb664_shop_supplement_v1', 'created_utc': report['timestamp_utc'],
                                'payload_files': records, 'contains_labels': False, 'contains_anchor_predictions': False,
                                'purpose': 'PRIVATE label-free56column cache and missing verification dependencies'}
        (supplement / 'bundle_manifest.json').write_text(json.dumps(shop_bundle_manifest, indent=2))
        shop_zip = out / 'cuda-xgb664-shop.zip'
        zip_verified(supplement, shop_zip)
        protocol = json.loads(json.dumps(old))
        protocol.update(version='cuda_xgb_shop664_v1', created_utc=report['timestamp_utc'],
                        trainer_sha256=digest(ROOT / 'scripts/train_cuda_xgb664.py'),
                        shared_loader_sha256=digest(ROOT / 'scripts/cuda_xgb664_loader.py'),
                        prior_loader_sha256=old['shared_loader_sha256'],
                        local_selector_sha256=digest(ROOT / 'scripts/select_cuda_xgb664_blend.py'),
                        return_verifier_sha256=digest(ROOT / 'scripts/verify_cuda_xgb664_return.py'),
                        dataset_launcher_sha256=digest(ROOT / 'scripts/launch_cuda_xgb664_dataset.py'),
                        shop_bundle_manifest_sha256=digest(supplement / 'bundle_manifest.json'),
                        shop_bundle_manifest_path=(supplement / 'bundle_manifest.json').relative_to(ROOT).as_posix(),
                        shop_bundle_archive_sha256=digest(shop_zip),
                        shop_cache_directory=cache.relative_to(ROOT).as_posix(),
                        shop_cache_manifest_sha256=digest(cache / 'manifest.json'),
                        shop_features_sha256=digest(cache / 'features.npz'),
                        feature_columns=old['feature_columns'] + COLUMNS,
                        preprocessing='Unchanged608 prefix plus56 fixed input-only SHOP features; no learned scaling.',
                        selection='Global0/.1/.25/.5 grid on2021, lowerweight ties; beat accepted608GPU50% anchor AND raw608 sameweight control. Freeze2021 choice before2022; test only that choice.',
                        fit_commands='Synthetic664 smoke first; separate2021 then eligible2021-lock-gated2022.',
                        caveats='Reused chronological development years. No public leaderboard tuning. Only private original data and input-derived cache; no anchor probabilities uploaded.')
        protocol['anchor'], protocol['control'] = {}, {}
        for year in [2021, 2022]:
            anchor = ROOT / f'artifacts/cuda-xgb608-select-{year}-001'
            anchor_manifest = json.loads((anchor / 'manifest.json').read_text())
            require(anchor_manifest['selected_weight'] == .5 and anchor_manifest['status'] == 'complete', 'Accepted608 anchor changed')
            if year == 2022:
                require(anchor_manifest['accepted'] is True, '608 anchor was rejected')
            protocol['anchor'][str(year)] = reference_record(anchor / 'manifest.json', anchor / f'selected_validation_{year}.npz')
            control = ROOT / f'artifacts/verify-cuda-xgb608-{year}-return-001/download/cuda-xgb608-{year}-001'
            protocol['control'][str(year)] = reference_record(control / 'manifest.json', control / f'validation_{year}.npz')
        protocol['feature_ablation'] = {'original608_protocol_sha256': digest(old_protocol_path),
            'unchanged_hyperparameters': True, 'prefix': 608, 'append': 56, 'data_sha256': old['feature_ablation']['data_sha256'],
            'feature_spec': old['feature_ablation']['feature_spec'], 'shop': feature_specification()}
        protocol_path = ROOT / 'docs/cuda_xgb664_protocol.json'
        with protocol_path.open('x') as stream:
            json.dump(protocol, stream, indent=2)
        code = out / 'code'
        code.mkdir()
        for name in CODE_FILES:
            shutil.copy2(ROOT / 'scripts' / name, code / name)
        shutil.copy2(protocol_path, code / 'cuda_xgb664_protocol.json')
        shutil.copy2(ROOT / 'docs/cuda_xgb664.md', code / 'README.md')
        code_files = [{'path': p.name, 'sha256': digest(p), 'size_bytes': p.stat().st_size} for p in sorted(code.iterdir())]
        (code / 'package_manifest.json').write_text(json.dumps({'private': True, 'files': code_files}, indent=2))
        payload = out / 'dataset'
        payload.mkdir()
        shutil.copytree(code, payload / 'code')
        shutil.copytree(supplement, payload / 'shop')
        dataset_files = [{'path': p.relative_to(payload).as_posix(), 'sha256': digest(p), 'size_bytes': p.stat().st_size}
                         for p in sorted(payload.rglob('*')) if p.is_file()]
        (payload / 'dataset_manifest.json').write_text(json.dumps({'required_visibility': 'PRIVATE', 'files': dataset_files,
            'protocol_sha256': digest(protocol_path), 'contains_original608_bundle': False,
            'execution': 'Explicit protocolSHA required; smoke then2021 then locked2022. No autoexecution.'}, indent=2))
        dataset_zip = out / 'cuda-xgb664-private-dataset.zip'
        zip_verified(payload, dataset_zip)
        report.update(status='complete', protocol_sha256=digest(protocol_path),
                      shop_archive_sha256=digest(shop_zip), dataset_archive_sha256=digest(dataset_zip),
                      dataset_archive_bytes=dataset_zip.stat().st_size, source={name: digest(ROOT / name) for name in LOCAL_SOURCE},
                      added_files=records, needs_independent_audit=True)
        print(json.dumps({k: report[k] for k in ['status', 'protocol_sha256', 'dataset_archive_sha256', 'dataset_archive_bytes']}), flush=True)
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        report['runtime_seconds'] = time.monotonic() - started
        for name in LOCAL_SOURCE:
            path = ROOT / name
            if path.is_file():
                target = out / 'source' / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
        report['artifacts'] = {p.relative_to(out).as_posix(): digest(p) for p in out.rglob('*') if p.is_file() and p.name != 'manifest.json'}
        (out / 'manifest.json').write_text(json.dumps(report, indent=2))
        (out / 'experiment_record.md').write_text(f'\n## {args.run_id} — {report["timestamp_utc"]}\n\nStatus {report["status"]}. Command `{report["command"]}`. Separate PRIVATE664 code/cache dataset, unchanged600treeGPU parameters/sample, disjoint56feature supplement; no training/upload. Full source/data/cache/archive/environment/protocol pins in `artifacts/{args.run_id}/manifest.json`; runtime {report["runtime_seconds"]:.3f}s. Independent audit required before hosted use.\n')


if __name__ == '__main__':
    main()
