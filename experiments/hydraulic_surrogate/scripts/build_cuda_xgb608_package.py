"""Freeze a new608 GPU feature ablation and only missing private input files."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shlex
import shutil
import sys
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_cuda_xgb608 import PARAMETERS, WEIGHTS, digest, require

FEATURE_SOURCES = ['sintef_project/' + name + '.py' for name in [
    'features', 'physics', 'context_features', 'lp_features', 'lp_context', 'normalized',
    'lp_joint_plant', 'lp_head_update', 'lp_shared_loss_update']]
NEW_FILES = [
    'sintef_project/lp_context.py', 'sintef_project/lp_joint_plant.py',
    'sintef_project/lp_head_update.py', 'sintef_project/lp_shared_loss_update.py',
    'scripts/build_lp_joint_features.py', 'artifacts/lp-joint-full-001/manifest.json',
    'artifacts/lp-joint-full-001/features.npz',
    'artifacts/lgb-normalized-lp-context-joint-001/manifest.json',
]
SOURCE_FILES = ['scripts/train_cuda_xgb608.py', 'scripts/cuda_xgb608_loader.py',
    'scripts/select_cuda_xgb608_blend.py', 'scripts/build_cuda_xgb608_package.py',
    'scripts/launch_cuda_xgb.py', 'scripts/train_cuda_tcn.py', 'tests/test_cuda_xgb608.py',
    'docs/cuda_xgb608.md', *FEATURE_SOURCES]


def zip_verified(directory: Path, archive_path: Path) -> None:
    """Write a fresh archive and verify every member against its original bytes."""
    files = {p.relative_to(directory).as_posix(): p for p in directory.rglob('*') if p.is_file()}
    with zipfile.ZipFile(archive_path, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in sorted(files.items()):
            archive.write(path, name)
    with zipfile.ZipFile(archive_path) as archive:
        require(set(archive.namelist()) == set(files), 'Archive readback allowlist differs')
        for name, path in files.items():
            require(hashlib.sha256(archive.read(name)).hexdigest() == digest(path), 'Archive member readback differs')


def main() -> None:
    """Prepare code/supplement, never train, upload, or change the original bundle."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', default='cuda-xgb608-package-001')
    args = parser.parse_args()
    out = ROOT / 'artifacts' / args.run_id
    out.mkdir(exist_ok=False)
    started = time.time()
    report = {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'run_id': args.run_id,
              'command': shlex.join([sys.executable, *sys.argv]), 'status': 'initializing',
              'training': False, 'upload_performed': False, 'kaggle': None,
              'environment': {'python': sys.version, 'platform': platform.platform()}}
    try:
        bundle_dir = ROOT / 'artifacts/kaggle-neural-bundle-001'
        bundle_path = bundle_dir / 'payload/bundle_manifest.json'
        bundle = json.loads(bundle_path.read_text())
        original_hash = digest(bundle_dir / 'kaggle-neural-inputs.zip')
        original_files = {item['path']: item['sha256'] for item in bundle['payload_files']}
        baseline_path = ROOT / 'artifacts/lgb-normalized-lp-context-joint-001/manifest.json'
        baseline = json.loads(baseline_path.read_text())
        context = json.loads((ROOT / 'artifacts/lgb-normalized-lp-context-001/manifest.json').read_text())
        joint_path = ROOT / 'artifacts/lp-joint-full-001/manifest.json'
        joint = json.loads(joint_path.read_text())
        old = json.loads((ROOT / 'docs/cuda_xgb_protocol.json').read_text())
        anchor_path = ROOT / 'artifacts/lp-context-joint-comparison-001/analysis.json'
        anchor = json.loads(anchor_path.read_text())
        require(baseline['status'] == joint['status'] == anchor['status'] == 'complete' and anchor['accepted'] is True,
                '608 baseline/cache/anchor not complete and accepted')
        require(PARAMETERS == old['model_parameters'], 'XGB hyperparameters changed from395 branch')
        require(len(baseline['features']) == 608 and baseline['features'][:521] == context['features']
                and baseline['features'][521:] == joint['feature_names'], 'Exact521+87 feature ablation changed')
        require(anchor['selected_weights'] == [.140625, .421875, .1875, .25], 'Predeclared608-blend anchor changed')
        for year in ['2021', '2022']:
            require(baseline['artifacts'][f'train_sample_{year}.npy'] == old['sample_sha256'][year]
                    == digest(ROOT / f'artifacts/lgb-normalized-lp-context-joint-001/train_sample_{year}.npy'),
                    '608 baseline sample differs from original395 sample')
        for name in FEATURE_SOURCES:
            require(digest(ROOT / name) == baseline['source'][name], f'Feature source changed: {name}')
        for category in ['source', 'data']:
            for name, expected in joint[category].items():
                require(digest(ROOT / name) == expected, f'Joint cache dependency changed: {name}')
        require(digest(ROOT / 'artifacts/lp-joint-full-001/features.npz') == joint['artifacts']['features.npz'],
                'Joint feature cache changed')
        require(not set(NEW_FILES).intersection(original_files), 'Supplement must only add missing files')
        require(set(joint['source']).union(joint['data']) <= set(original_files).union(NEW_FILES),
                'Supplement is missing a joint-cache verification dependency')
        supplement = out / 'supplement'
        supplement.mkdir()
        entries = []
        for name in NEW_FILES:
            path = ROOT / name
            target = supplement / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            entries.append({'path': name, 'size_bytes': target.stat().st_size, 'sha256': digest(target)})
        extra_manifest = {'version': 'cuda_xgb608_supplement_v1', 'created_utc': report['timestamp_utc'],
                          'purpose': 'PRIVATE missing joint cache/source plus frozen608 reference; original bundle unchanged',
                          'original_bundle_manifest_sha256': digest(bundle_path), 'payload_files': entries,
                          'contains_anchor_predictions': False, 'contains_new_training_labels': False}
        (supplement / 'bundle_manifest.json').write_text(json.dumps(extra_manifest, indent=2))
        extra_zip = out / 'cuda-xgb608-supplement.zip'
        zip_verified(supplement, extra_zip)
        protocol = {
            'version': 'cuda_xgb_context_joint608_v1', 'created_utc': report['timestamp_utc'],
            'trainer_sha256': digest(ROOT / 'scripts/train_cuda_xgb608.py'),
            'shared_loader_sha256': digest(ROOT / 'scripts/cuda_xgb608_loader.py'),
            'core_loader_sha256': digest(ROOT / 'scripts/train_cuda_tcn.py'),
            'launcher_sha256': digest(ROOT / 'scripts/launch_cuda_xgb.py'),
            'local_selector_sha256': digest(ROOT / 'scripts/select_cuda_xgb608_blend.py'),
            'postprocess_sha256': digest(ROOT / 'sintef_project/postprocess.py'),
            'bundle_manifest_sha256': digest(bundle_path), 'bundle_archive_sha256': original_hash,
            'supplement_manifest_sha256': digest(supplement / 'bundle_manifest.json'),
            'supplement_archive_sha256': digest(extra_zip),
            'baseline_manifest_sha256': digest(baseline_path),
            'feature_source_sha256': {name: baseline['source'][name] for name in FEATURE_SOURCES},
            'model_parameters': PARAMETERS, 'seed': 4173, 'candidate_weights': WEIGHTS,
            'feature_columns': baseline['features'], 'units': list(baseline['folds']['2021']['unit_auc']),
            'anchor': {'run_id': 'lp-context-joint-comparison-001', 'analysis_sha256': digest(anchor_path),
                       'weights': anchor['selected_weights'],
                       'scores': {y: f['selected_micro_auc'] for y, f in anchor['folds'].items()}},
            'feature_ablation': {'original395_protocol_sha256': digest(ROOT / 'docs/cuda_xgb_protocol.json'),
                                 'unchanged_hyperparameters': True, 'context_prefix': 521, 'appended_joint': 87,
                                 'data_sha256': baseline['data'], 'feature_spec': baseline['feature_spec']},
            'folds': {y: {k: f[k] for k in ['train_ids', 'validation_ids', 'train_dates', 'validation_dates', 'purge_days', 'sampled_training_rows']}
                      for y, f in baseline['folds'].items()},
            'sample_sha256': {y: baseline['artifacts'][f'train_sample_{y}.npy'] for y in ['2021', '2022']},
            'sampling': old['sampling'],
            'preprocessing': 'Exact608 normalizedLP-context-joint features; no additional learned preprocessing; target rows after requestedyear redacted.',
            'fit_commands': 'Synthetic608 smoke first. Separate2021 and2022 commands. Improving nonzero2021 lock plus explicitSHA required before2022.',
            'selection': 'Fixed old608-blend anchor; project_vinje((1-w)*anchor_projected+w*rawXGB608) in float64. [0,.1,.25,.5] selected2021, tieslowerweight. Lock before2022; no2022 grid retuning.',
            'caveats': 'Chronological years repeatedly reused. Cross-platform bitwise feature equality not yet measured; exact source/data/schema/formula checks required. Private inputs only; no anchor probabilities uploaded.'}
        protocol_path = ROOT / 'docs/cuda_xgb608_protocol.json'
        with protocol_path.open('x') as stream:
            json.dump(protocol, stream, indent=2)
        payload = out / 'code'
        payload.mkdir()
        for name in ['train_cuda_xgb608.py', 'cuda_xgb608_loader.py', 'train_cuda_tcn.py', 'launch_cuda_xgb.py']:
            shutil.copy2(ROOT / 'scripts' / name, payload / name)
        shutil.copy2(protocol_path, payload / 'cuda_xgb608_protocol.json')
        shutil.copy2(ROOT / 'docs/cuda_xgb608.md', payload / 'README.md')
        code_files = [{'path': p.name, 'sha256': digest(p), 'size_bytes': p.stat().st_size} for p in sorted(payload.iterdir())]
        (payload / 'package_manifest.json').write_text(json.dumps({'files': code_files, 'private': True}, indent=2))
        code_zip = out / 'cuda-xgb608-code.zip'
        zip_verified(payload, code_zip)
        require(digest(bundle_dir / 'kaggle-neural-inputs.zip') == original_hash, 'Original bundle unexpectedly changed')
        report.update(status='complete', protocol_sha256=digest(protocol_path),
                      original_bundle_sha256=original_hash, supplement_sha256=digest(extra_zip),
                      supplement_manifest_sha256=digest(supplement / 'bundle_manifest.json'),
                      supplement_size_bytes=extra_zip.stat().st_size, code_sha256=digest(code_zip),
                      code_size_bytes=code_zip.stat().st_size, supplement_files=entries,
                      feature_schema_verified='608=exact521 context prefix+87 joint', needs_independent_audit=True)
        print(json.dumps({k: report[k] for k in ['status', 'protocol_sha256', 'supplement_sha256', 'supplement_size_bytes', 'code_sha256', 'code_size_bytes']}), flush=True)
    except BaseException:
        report['status'] = 'failed'
        report['error'] = traceback.format_exc()
        raise
    finally:
        report['source'] = {}
        for name in SOURCE_FILES:
            source = ROOT / name
            if source.is_file():
                destination = out / 'source' / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                report['source'][name] = digest(source)
        report['runtime_seconds'] = time.time() - started
        (out / 'manifest.json').write_text(json.dumps(report, indent=2))
        with (ROOT / 'docs/experiments.md').open('a') as stream:
            stream.write(f'\n## {args.run_id} — {report["timestamp_utc"]}\n\nStatus:{report["status"]}. Command:`{report["command"]}`. New fixed608GPU-XGB feature ablation package only; no featurematrix/model/GPU run or upload. Same600/depth6 hyperparameters and exact40%row samples as395branch; fixed current608-blend anchor. Original inputbundle preserved; separate minimal private jointcache/source supplement. Protocol/code/data/source/ZIP readback hashes and environment:`artifacts/{args.run_id}/manifest.json`. Runtime{report["runtime_seconds"]:.3f}s. Independent audit required before use. NoKaggle submission/score.\n')


if __name__ == '__main__':
    main()
