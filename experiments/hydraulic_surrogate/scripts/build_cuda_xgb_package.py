"""Freeze reviewed GPU-XGBoost code/protocol in a tiny input-only code package."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_cuda_xgb import PARAMETERS, WEIGHTS, digest


def main() -> None:
    """Package source and a protocol only; no raw-data copy or GPU execution."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', default='cuda-xgb-package-001')
    args = parser.parse_args()
    out = ROOT / 'artifacts' / args.run_id
    out.mkdir(exist_ok=False)
    started = time.time()
    source_files = ['scripts/train_cuda_xgb.py', 'scripts/launch_cuda_xgb.py',
                    'scripts/select_cuda_xgb_blend.py', 'scripts/train_cuda_tcn.py',
                    'scripts/build_cuda_xgb_package.py', 'tests/test_cuda_xgb.py', 'docs/cuda_xgb.md']
    baseline_path = ROOT / 'artifacts/lgb-normalized-lp-001/manifest.json'
    baseline = json.loads(baseline_path.read_text())
    previous_path = ROOT / 'artifacts/xgb-lp-specialists-001/manifest.json'
    previous = json.loads(previous_path.read_text())
    anchor_path = ROOT / 'artifacts/lp-joint-comparison-001/analysis.json'
    anchor = json.loads(anchor_path.read_text())
    bundle_dir = ROOT / 'artifacts/kaggle-neural-bundle-001'
    assert {k: v for k, v in PARAMETERS.items() if k != 'device'} == {k: v for k, v in previous['model_parameters'].items() if k != 'device'}
    assert digest(ROOT / 'scripts/train_cuda_tcn.py') == '1afae39884da9d210a8ef695fb0c99636536bc688ce79961d502bf4c4b743e57'
    assert anchor['accepted'] is True and anchor['selected_weights'] == [.1875, .5625, .25]
    protocol = {'version': 'cuda_xgb_full14_v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
                'trainer_sha256': digest(ROOT / 'scripts/train_cuda_xgb.py'),
                'shared_loader_sha256': digest(ROOT / 'scripts/train_cuda_tcn.py'),
                'launcher_sha256': digest(ROOT / 'scripts/launch_cuda_xgb.py'),
                'local_selector_sha256': digest(ROOT / 'scripts/select_cuda_xgb_blend.py'),
                'postprocess_sha256': digest(ROOT / 'sintef_project/postprocess.py'),
                'bundle_manifest_sha256': digest(bundle_dir / 'payload/bundle_manifest.json'),
                'bundle_archive_sha256': digest(bundle_dir / 'kaggle-neural-inputs.zip'),
                'model_parameters': PARAMETERS, 'seed': 4173, 'candidate_weights': WEIGHTS,
                'feature_columns': baseline['features'], 'units': list(baseline['folds']['2021']['unit_auc']),
                'anchor': {'run_id': 'lp-joint-comparison-001', 'analysis_sha256': digest(anchor_path),
                           'weights': anchor['selected_weights'],
                           'scores': {year: fold['selected_micro_auc'] for year, fold in anchor['folds'].items()}},
                'reference': {'run_id': 'xgb-lp-specialists-001', 'manifest_sha256': digest(previous_path),
                              'xgboost_version': previous['environment']['packages']['xgboost'], 'device': 'cpu'},
                'baseline_manifest_sha256': digest(baseline_path),
                'folds': {year: {key: fold[key] for key in ['train_ids', 'validation_ids', 'train_dates',
                                                           'validation_dates', 'purge_days', 'sampled_training_rows']}
                          for year, fold in baseline['folds'].items()},
                'sample_sha256': {year: baseline['artifacts'][f'train_sample_{year}.npy'] for year in ['2021', '2022']},
                'sampling': 'Reproduce original default_rng4173 choice calls sequentially2021 then2022; exactNPY SHA required, no fallback resampling.',
                'preprocessing': 'Existing395normalizedLP-v1 features only, no extra learned transform.',
                'fit_commands': 'One explicit year per command; matching synthetic CUDA smoke first;2022 requires local2021 lock and explicitSHA.',
                'selection': 'Local only:project_vinje((1-w)*frozenjoint_anchor_projected+w*rawGPU),float64; choose2021 highestmicroAUC,tieslowerweight; strictbothyearimprovement; noleaderboardtuning.',
                'runtime_observed_by_root': {'xgboost': '3.4.1', 'python': '3.13.15', 'numpy': '2.1.3',
                                             'pandas': '2.3.3', 'scikit-learn': '1.6.1', 'uv': '0.13.5',
                                             'gpu': '2xT4', 'xgboost_gpu_smoke': 'not yet performed'},
                'caveats': 'InstalledGPU/libraryversions differ fromCPUreference; noexactfit-equivalence claim. Bothchronologicalyears are reuseddevelopment evidence. Allactualversions/build/driver/nativeparams are captured.'}
    payload = out / 'payload'
    payload.mkdir()
    for name in ['train_cuda_xgb.py', 'launch_cuda_xgb.py', 'train_cuda_tcn.py']:
        shutil.copy2(ROOT / 'scripts' / name, payload / name)
    (payload / 'cuda_xgb_protocol.json').write_text(json.dumps(protocol, indent=2))
    shutil.copy2(payload / 'cuda_xgb_protocol.json', ROOT / 'docs/cuda_xgb_protocol.json')
    (payload / 'README.md').write_text('''# GPU XGBoost code package

No raw data, credentials or checkpoints are included. Use the existing sealed private input bundle separately. This package performs no installation or download. Run commands through the offline uv launcher, which reuses installed system CUDA packages. XGBoost is pinned to cuda:1; TCN uses cuda:0.

First run synthetic smoke only:

```
python launch_cuda_xgb.py --runner train_cuda_xgb.py -- --mode smoke --run-id cuda-xgb-smoke-001 --protocol cuda_xgb_protocol.json --shared-loader train_cuda_tcn.py
```

After a successful smoke, run one explicit2021 fit using the observed private input-bundle path:

```
python launch_cuda_xgb.py --runner train_cuda_xgb.py -- --mode fit --year 2021 --run-id cuda-xgb-2021-001 --protocol cuda_xgb_protocol.json --shared-loader train_cuda_tcn.py --bundle /kaggle/input/REPLACE_WITH_OBSERVED_PRIVATE_PATH/kaggle-neural-inputs.zip --smoke-manifest /kaggle/working/xgb-runs/cuda-xgb-smoke-001/manifest.json
```

Download the full2021 run directory. Root runs the local select_cuda_xgb_blend.py utility and obtains selection_2021.json plus itsSHA. Only then may a separate2022 fit command use --year2022, a newrun-id, --selection-lock with that uploaded local file, and --selection-lock-sha256 with its exact hash. The runner rejects a2022 command without this evidence. Never fill a placeholder with an invented path or hash. No command performs both folds, a final fit, leaderboard submission or repository transfer.
''')
    files = [{'path': path.name, 'size_bytes': path.stat().st_size, 'sha256': digest(path)}
             for path in sorted(payload.iterdir())]
    package_manifest = {'created_utc': protocol['created_utc'], 'purpose': 'Reviewedsource/protocol only; noGPUlaunch',
                        'files': files, 'raw_bundle_included': False}
    (payload / 'package_manifest.json').write_text(json.dumps(package_manifest, indent=2))
    archive_path = out / 'cuda-xgb-code.zip'
    with zipfile.ZipFile(archive_path, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(payload.iterdir()):
            archive.write(path, path.name)
    with zipfile.ZipFile(archive_path) as archive:
        assert set(archive.namelist()) == {item['path'] for item in files} | {'package_manifest.json'}
        for item in files:
            data = archive.read(item['path'])
            assert len(data) == item['size_bytes'] and hashlib.sha256(data).hexdigest() == item['sha256']
    sources = {}
    for relative in source_files:
        source = ROOT / relative
        destination = out / 'source' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        sources[relative] = digest(source)
    report = {'timestamp_utc': protocol['created_utc'], 'status': 'complete',
              'command': shlex.join([sys.executable, *sys.argv]), 'source': sources,
              'protocol_sha256': digest(payload / 'cuda_xgb_protocol.json'),
              'package_manifest_sha256': digest(payload / 'package_manifest.json'),
              'archive_sha256': digest(archive_path), 'archive_size_bytes': archive_path.stat().st_size,
              'payload_files': files, 'runtime_seconds': time.time() - started,
              'gpu_fit_or_smoke_launched': False, 'upload_performed': False, 'kaggle_submission': None}
    (out / 'manifest.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({key: report[key] for key in ['status', 'protocol_sha256', 'archive_sha256', 'archive_size_bytes']}), flush=True)
    with (ROOT / 'docs/experiments.md').open('a') as stream:
        stream.write(f'\n## {args.run_id} — {report["timestamp_utc"]}\n\nStatus:complete. Command:`{report["command"]}`. Fixed full14-unit GPU-XGB preparation only; no remote launch, upload, package install, full matrix or model fit. Reuses sealed24-file data bundle, freezes shared-loader/source/protocol hashes and originalsample-SHA checks. CUDA:1,600/depth6/fixedpriorparameters; local2021weight lock gates separate2022. Code-only archive{report["archive_size_bytes"]}bytes, SHA`{report["archive_sha256"]}`. ProtocolSHA`{report["protocol_sha256"]}`. Every ZIP member readback/hash checked. Allsource/provenance:`artifacts/{args.run_id}/manifest.json`. No Kaggle submission.\n')


if __name__ == '__main__':
    main()
