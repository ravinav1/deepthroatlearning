"""Select the fixed GPU-XGBoost global blend locally, one returned fold at a time."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import shutil
import sys
import time
import traceback

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_cuda_xgb import digest, require, validate_lock, require_gpu_config, WEIGHTS
from sintef_project.postprocess import project_vinje


def require_anchor_array(path: Path, manifest: dict) -> None:
    """Verify frozen anchor bytes before any probability or label is decoded."""
    require(digest(path) == manifest['artifacts'][path.name], 'Frozen anchor prediction bytes changed')


def main() -> None:
    """Freeze a2021 choice, or evaluate that exact choice on separately run2022."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, choices=[2021, 2022], required=True)
    parser.add_argument('--gpu-run', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--selection-lock', type=Path)
    parser.add_argument('--selection-lock-sha256')
    args = parser.parse_args()
    out = ROOT / 'artifacts' / args.run_id
    out.mkdir(exist_ok=False)
    started = time.time()
    report = {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'year': args.year,
              'command': shlex.join([sys.executable, *sys.argv]), 'inputs': {},
              'training': 'none; local fixed-grid selection from returned GPU OOF',
              'kaggle_submission': None}
    try:
        protocol_sha = digest(args.protocol)
        protocol = json.loads(args.protocol.read_text())
        require(protocol['version'] == 'cuda_xgb_full14_v1' and protocol['candidate_weights'] == WEIGHTS,
                'Unexpected frozen GPU protocol')
        require(digest(ROOT / 'scripts/train_cuda_xgb.py') == protocol['trainer_sha256']
                and digest(ROOT / 'sintef_project/postprocess.py') == protocol['postprocess_sha256'],
                'Local protocol-validation or projection source changed')
        report['protocol_sha256'] = protocol_sha
        if args.year == 2022:
            require(args.selection_lock is not None and args.selection_lock_sha256 is not None,
                    '2022 requires the frozen2021 local lock and explicit hash')
            lock = validate_lock(args.selection_lock, args.selection_lock_sha256, protocol, protocol_sha)
            selected = lock['selected_weight']
            report['selection_lock_sha256'] = digest(args.selection_lock)
        else:
            require(args.selection_lock is None, '2021 selection must be new')
        directory = args.gpu_run.resolve()
        manifest_path = directory / 'manifest.json'
        gpu = json.loads(manifest_path.read_text())
        require(gpu['status'] == 'complete' and gpu['mode'] == 'fit' and gpu['year'] == args.year,
                'Expected complete requested GPU fold')
        require(gpu['protocol_sha256'] == protocol_sha and gpu['trainer_sha256'] == protocol['trainer_sha256'],
                'GPU runner or protocol changed')
        require(gpu['shared_loader_sha256'] == protocol['shared_loader_sha256'], 'Returned shared-loader provenance changed')
        require(gpu['features'] == protocol['feature_columns'], 'Returned feature schema changed')
        expected_fold = protocol['folds'][str(args.year)]
        require(gpu['train_ids'] == expected_fold['train_ids']
                and gpu['validation_ids'] == expected_fold['validation_ids'], 'Returned chronological fold IDs changed')
        require(gpu['completed_units'] == 14 and gpu['completed_rounds_per_unit'] == 600,
                'Incomplete GPU learner ensemble')
        if args.year == 2022:
            require(gpu['selection_lock_sha256'] == report['selection_lock_sha256'],
                    'Remote2022 run was not gated by this local2021 lock')
        for relative, expected in gpu['artifacts'].items():
            require(digest(directory / relative) == expected, f'Changed or missing GPU artifact: {relative}')
        for unit in gpu['units']:
            native_path = directory / f'config_{args.year}_{unit}.json'
            require_gpu_config(json.loads(native_path.read_text()))
            require(gpu['models'][unit]['checkpoint_replay_max_error'] == 0.
                    and gpu['models'][unit]['replay_rows'] == 61320, 'GPU replay proof missing')
        anchor_dir = ROOT / 'artifacts' / protocol['anchor']['run_id']
        anchor_manifest_path = anchor_dir / 'analysis.json'
        require(digest(anchor_manifest_path) == protocol['anchor']['analysis_sha256'], 'Frozen anchor changed')
        anchor_manifest = json.loads(anchor_manifest_path.read_text())
        require(anchor_manifest['status'] == 'complete' and anchor_manifest['accepted'] is True,
                'Anchor is not the accepted frozen recipe')
        candidate_path = directory / f'validation_{args.year}.npz'
        anchor_path = anchor_dir / f'selected_validation_{args.year}.npz'
        require_anchor_array(anchor_path, anchor_manifest)
        with np.load(candidate_path, allow_pickle=False) as saved:
            raw, labels, ids, units = saved['predictions'].astype(np.float64), saved['labels'], saved['case_ids'], saved['units'].tolist()
        require(ids.tolist() == expected_fold['validation_ids'], 'OOF case IDs differ from frozen protocol vector')
        with np.load(anchor_path, allow_pickle=False) as anchor:
            require(np.array_equal(labels, anchor['labels']) and np.array_equal(ids, anchor['case_ids'])
                    and units == anchor['units'].tolist() == protocol['units'], 'OOF label/ID/unit alignment failed')
            prior = anchor['predictions'].astype(np.float64)
        require(raw.shape == prior.shape == (len(ids) * 168, 14) and np.isfinite(raw).all()
                and raw.min() >= 0 and raw.max() <= 1, 'Invalid raw GPU probabilities')
        require(abs(float(roc_auc_score(labels.ravel(), raw.ravel())) - gpu['raw_micro_auc']) <= 1e-12,
                'Returned GPU raw AUC differs from independent local score')
        report['inputs'] = {str(path): digest(path) for path in
                            [manifest_path, candidate_path, anchor_manifest_path, anchor_path, args.protocol]}
        scores = {}
        for weight in (WEIGHTS if args.year == 2021 else sorted(set([0., selected]))):
            values = project_vinje((1-weight) * prior + weight * raw, units)
            scores[str(weight)] = float(roc_auc_score(labels.ravel(), values.ravel()))
        if args.year == 2021:
            selected = max(WEIGHTS, key=lambda weight: scores[str(weight)])
            lock = {'version': 'cuda_xgb_2021_blend_lock_v1', 'selection_year': 2021,
                    'locked_utc': datetime.now(timezone.utc).isoformat(),
                    'training_protocol_sha256': protocol_sha,
                    'anchor_run': protocol['anchor']['run_id'], 'anchor_sha256': protocol['anchor']['analysis_sha256'],
                    'candidate_weights': WEIGHTS, 'selected_weight': selected,
                    'training_2021_complete': True, 'gpu_run_manifest_sha256': digest(manifest_path),
                    'gpu_validation_2021_sha256': digest(candidate_path),
                    'scores_2021': scores, 'selected_2021_micro_auc': scores[str(selected)],
                    'anchor_2021_micro_auc': scores['0.0'],
                    'formula': 'project_vinje((1-w)*accepted_joint_anchor_projected+w*gpu_xgb_raw),float64',
                    'case_ids': ids.tolist(), 'units': units,
                    'reuse_caveat': 'Repeatedly reused chronological development years; no leaderboard-based tuning.'}
            with (out / 'selection_2021.json').open('x') as stream:
                json.dump(lock, stream, indent=2)
            report['selection_lock_sha256'] = digest(out / 'selection_2021.json')
        chosen = project_vinje((1-selected) * prior + selected * raw, units)
        np.savez_compressed(out / f'selected_validation_{args.year}.npz', predictions=chosen, labels=labels,
                            case_ids=ids, units=np.asarray(units, dtype=str))
        report.update(status='complete', selected_weight=selected, scores=scores,
                      raw_micro_auc=float(roc_auc_score(labels.ravel(), raw.ravel())),
                      selected_micro_auc=scores[str(selected)], anchor_micro_auc=scores['0.0'],
                      selected_unit_auc={unit: float(roc_auc_score(labels[:, j], chosen[:, j])) for j, unit in enumerate(units)})
        if args.year == 2022:
            report['accepted'] = selected > 0. and lock['selected_2021_micro_auc'] > lock['anchor_2021_micro_auc'] and scores[str(selected)] > scores['0.0']
            report['meets_transfer_gate'] = lock['selected_2021_micro_auc'] >= .998 and scores[str(selected)] >= .998
        print(json.dumps({name: report.get(name) for name in ['status', 'selected_weight', 'selected_micro_auc', 'anchor_micro_auc', 'accepted']}), flush=True)
    except BaseException:
        report['status'] = 'failed'
        report['error'] = traceback.format_exc()
        raise
    finally:
        for relative in ['scripts/select_cuda_xgb_blend.py', 'scripts/train_cuda_xgb.py', 'sintef_project/postprocess.py']:
            target = out / 'source' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)
        report['runtime_seconds'] = time.time() - started
        report['artifacts'] = {str(path.relative_to(out)): digest(path) for path in out.rglob('*')
                               if path.is_file() and path.name != 'manifest.json'}
        (out / 'manifest.json').write_text(json.dumps(report, indent=2))
        with (ROOT / 'docs/experiments.md').open('a') as stream:
            stream.write(f'\n## {args.run_id} — {report["timestamp_utc"]}\n\nStatus:{report["status"]}. Command:`{report["command"]}`. Local GPU-XGB selection/evaluation only, no fit. Fixed global weights0/.1/.25/.5 on2021, immutable lock before separately authorized2022 fit, strict both-year improvement against frozen joint anchor; no leaderboard retuning. Selected{report.get("selected_weight")}, microAUC{report.get("selected_micro_auc")}, anchor{report.get("anchor_micro_auc")}. Full hashes/schema/checkpoint-replay provenance:`artifacts/{args.run_id}/manifest.json`. No Kaggle submission.\n')


if __name__ == '__main__':
    main()
