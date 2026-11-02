"""Select one fixed global664 weight against the accepted608 anchor and control."""
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

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_cuda_xgb664 import digest, require, validate_lock, require_gpu_config, WEIGHTS
from scripts.verify_cuda_xgb664_return import verify_return
from sintef_project.postprocess import project_vinje


def load_pinned_prediction(record: dict, labels: np.ndarray, ids: np.ndarray, units: list[str]) -> np.ndarray:
    """Verify manifest/prediction bytes and exact alignment before comparison."""
    manifest_path, prediction_path = ROOT / record['manifest'], ROOT / record['prediction']
    require(digest(manifest_path) == record['manifest_sha256'], 'Reference manifest changed')
    require(digest(prediction_path) == record['prediction_sha256'], 'Reference predictions changed')
    manifest = json.loads(manifest_path.read_text())
    require(manifest['status'] == 'complete', 'Reference run incomplete')
    require(manifest['artifacts'][prediction_path.name] == record['prediction_sha256'], 'Reference artifact pin differs')
    with np.load(prediction_path, allow_pickle=False) as saved:
        require(np.array_equal(saved['labels'], labels) and np.array_equal(saved['case_ids'], ids)
                and saved['units'].tolist() == units, 'Reference label/ID/unit alignment failed')
        values = saved['predictions'].astype(np.float64)
    require(values.shape == labels.shape and np.isfinite(values).all()
            and values.min() >= 0 and values.max() <= 1, 'Invalid reference probabilities')
    return values


def main() -> None:
    """Freeze2021 selection or evaluate only the locked choice on separately trained2022."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, choices=[2021, 2022], required=True)
    parser.add_argument('--gpu-run', type=Path, required=True)
    parser.add_argument('--gpu-manifest-sha256', required=True)
    parser.add_argument('--smoke-manifest', type=Path, required=True)
    parser.add_argument('--smoke-manifest-sha256', required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--selection-lock', type=Path)
    parser.add_argument('--selection-lock-sha256')
    args = parser.parse_args()
    require(Path(args.run_id).name == args.run_id and args.run_id not in ['', '.', '..'], 'Invalid run ID')
    out = ROOT / 'artifacts' / args.run_id
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    report = {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'year': args.year,
              'command': shlex.join([sys.executable, *sys.argv]), 'training': False,
              'environment': {'python': sys.version, 'platform': platform.platform(), 'machine': platform.machine(), 'cpu_count': os.cpu_count(), 'packages': {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'PyYAML']}},
              'kaggle_submission': None, 'status': 'initializing'}
    try:
        protocol_sha = digest(args.protocol)
        protocol = json.loads(args.protocol.read_text())
        require(protocol['version'] == 'cuda_xgb_shop664_v1' and protocol['candidate_weights'] == WEIGHTS,
                'Unexpected664 protocol')
        for path, key in [('scripts/train_cuda_xgb664.py', 'trainer_sha256'),
                          ('scripts/select_cuda_xgb664_blend.py', 'local_selector_sha256'),
                          ('scripts/verify_cuda_xgb664_return.py', 'return_verifier_sha256'),
                          ('sintef_project/postprocess.py', 'postprocess_sha256')]:
            require(digest(ROOT / path) == protocol[key], f'Source identity changed: {path}')
        report['protocol_sha256'] = protocol_sha
        lock = None
        if args.year == 2022:
            require(args.selection_lock is not None and args.selection_lock_sha256 is not None,
                    '2022 requires explicit2021 lock and hash')
            lock = validate_lock(args.selection_lock, args.selection_lock_sha256, protocol, protocol_sha)
            report['selection_lock_sha256'] = digest(args.selection_lock)
        else:
            require(args.selection_lock is None, '2021 choice must be new')
        directory = args.gpu_run.resolve()
        manifest_path = directory / 'manifest.json'
        report['independent_return_verification'] = verify_return(
            directory, args.smoke_manifest, args.gpu_manifest_sha256, args.smoke_manifest_sha256,
            protocol, args.year, ROOT, protocol_sha)
        gpu = json.loads(manifest_path.read_text())
        require(gpu['status'] == 'complete' and gpu['mode'] == 'fit' and gpu['year'] == args.year,
                'Expected complete requestedGPU fold')
        require(gpu['protocol_sha256'] == protocol_sha and gpu['trainer_sha256'] == protocol['trainer_sha256']
                and gpu['shared_loader_sha256'] == protocol['shared_loader_sha256'], 'GPU source/protocol changed')
        require(gpu['features'] == protocol['feature_columns'] and gpu['units'] == protocol['units'], 'GPU schema changed')
        fold = protocol['folds'][str(args.year)]
        require(gpu['train_ids'] == fold['train_ids'] and gpu['validation_ids'] == fold['validation_ids'], 'GPU fold changed')
        require(gpu['completed_units'] == 14 and gpu['completed_rounds_per_unit'] == 600, 'Incomplete GPU models')
        require(gpu['sample_sha256'] == protocol['sample_sha256'][str(args.year)], 'GPU training sample changed')
        if lock is not None:
            require(gpu['selection_lock_sha256'] == report['selection_lock_sha256'], 'GPU2022 lock changed')
        for name, expected in gpu['artifacts'].items():
            require(digest(directory / name) == expected, f'GPU artifact changed: {name}')
        for unit in gpu['units']:
            require_gpu_config(json.loads((directory / f'config_{args.year}_{unit}.json').read_text()))
            require(gpu['models'][unit]['checkpoint_replay_max_error'] == 0
                    and gpu['models'][unit]['replay_rows'] == len(fold['validation_ids'])*168, 'MissingGPU replay')
        candidate_path = directory / f'validation_{args.year}.npz'
        with np.load(candidate_path, allow_pickle=False) as saved:
            raw, labels, ids, units = saved['predictions'].astype(np.float64), saved['labels'], saved['case_ids'], saved['units'].tolist()
        require(ids.tolist() == fold['validation_ids'] and units == protocol['units'], 'Prediction schema differs')
        require(raw.shape == labels.shape == (len(ids)*168, 14) and np.isfinite(raw).all()
                and raw.min() >= 0 and raw.max() <= 1, 'Invalid664 probabilities')
        require(abs(roc_auc_score(labels.ravel(), raw.ravel()) - gpu['raw_micro_auc']) < 1e-12, 'Raw AUC differs')
        anchor = load_pinned_prediction(protocol['anchor'][str(args.year)], labels, ids, units)
        control = load_pinned_prediction(protocol['control'][str(args.year)], labels, ids, units)
        weights = WEIGHTS if lock is None else sorted({0., lock['selected_weight']})
        scores = {str(w): float(roc_auc_score(labels.ravel(), project_vinje((1-w)*anchor+w*raw, units).ravel())) for w in weights}
        selected = max(WEIGHTS, key=lambda w: scores[str(w)]) if lock is None else lock['selected_weight']
        control_auc = float(roc_auc_score(labels.ravel(), project_vinje((1-selected)*anchor+selected*control, units).ravel()))
        eligible = selected > 0 and scores[str(selected)] > max(scores['0.0'], control_auc)
        chosen = project_vinje((1-selected)*anchor+selected*raw, units)
        np.savez_compressed(out / f'selected_validation_{args.year}.npz', predictions=chosen,
                            labels=labels, case_ids=ids, units=np.asarray(units))
        report.update(status='complete', scores=scores, selected_weight=selected,
                      selected_micro_auc=scores[str(selected)], anchor_micro_auc=scores['0.0'],
                      control_micro_auc=control_auc, beats_anchor_and_control=eligible,
                      raw_micro_auc=float(roc_auc_score(labels.ravel(), raw.ravel())),
                      gpu_manifest_sha256=digest(manifest_path), gpu_validation_sha256=digest(candidate_path),
                      selected_unit_auc={unit: float(roc_auc_score(labels[:, j], chosen[:, j])) for j, unit in enumerate(units)})
        if lock is None:
            lock = {'version': 'cuda_xgb664_2021_blend_lock_v1', 'selection_year': 2021,
                    'locked_utc': datetime.now(timezone.utc).isoformat(), 'training_protocol_sha256': protocol_sha,
                    'candidate_weights': WEIGHTS, 'selected_weight': selected, 'scores_2021': scores,
                    'selected_2021_micro_auc': scores[str(selected)], 'anchor_2021_micro_auc': scores['0.0'],
                    'control_2021_micro_auc': control_auc, 'eligible_for_2022': eligible, 'training_2021_complete': True,
                    'selector_sha256': digest(Path(__file__)), 'gpu_run_manifest_sha256': digest(manifest_path),
                    'gpu_validation_2021_sha256': digest(candidate_path),
                    'anchor_prediction_sha256': protocol['anchor']['2021']['prediction_sha256'],
                    'control_prediction_sha256': protocol['control']['2021']['prediction_sha256'],
                    'formula': 'project_vinje((1-w)*accepted608GPUblend_projected+w*raw664),float64',
                    'caveat': 'Reused chronological development years; no public leaderboard selection'}
            (out / 'selection_2021.json').write_text(json.dumps(lock, indent=2))
            report['selection_lock_sha256'] = digest(out / 'selection_2021.json')
        else:
            report.update(accepted=eligible, meets_transfer_gate=min(lock['selected_2021_micro_auc'], scores[str(selected)]) >= .998)
        print(json.dumps({k: report.get(k) for k in ['status', 'selected_weight', 'selected_micro_auc', 'anchor_micro_auc', 'control_micro_auc', 'beats_anchor_and_control']}), flush=True)
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        for name in ['scripts/select_cuda_xgb664_blend.py', 'scripts/train_cuda_xgb664.py',
                     'scripts/verify_cuda_xgb664_return.py', 'sintef_project/postprocess.py']:
            target = out / 'source' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        if args.protocol.is_file():
            shutil.copy2(args.protocol, out / 'protocol.json')
        report['runtime_seconds'] = time.monotonic() - started
        report['artifacts'] = {p.relative_to(out).as_posix(): digest(p) for p in out.rglob('*') if p.is_file() and p.name != 'manifest.json'}
        (out / 'manifest.json').write_text(json.dumps(report, indent=2))
        (out / 'experiment_record.md').write_text(f'\n## {args.run_id} — {report["timestamp_utc"]}\n\nStatus {report["status"]}; local664 selection, no model fit or submission. Command `{report["command"]}`. Fixed global weights0/.1/.25/.5 on2021; lock before2022. Selected {report.get("selected_weight")}; microAUC {report.get("selected_micro_auc")}, anchor {report.get("anchor_micro_auc")}, matched608 control {report.get("control_micro_auc")}. Full provenance `artifacts/{args.run_id}/manifest.json`; runtime {report["runtime_seconds"]:.3f}s.\n')


if __name__ == '__main__':
    main()
