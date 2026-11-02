"""Verify a returned CUDA TCN fold and freeze/evaluate its local scalar blend."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
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
from threadpoolctl import threadpool_limits

SNAPSHOT_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ['TDT4173_WORKSPACE_ROOT']).resolve()
sys.path.insert(0, str(SNAPSHOT_ROOT))
from scripts.train_cuda_tcn import digest, require, safe_relative, validate_selection_lock
from sintef_project.postprocess import DEFAULT_UNITS, project_vinje

WEIGHTS = [0., .1, .25, .5]
PROTOCOL_SHA256 = 'd55bee5f39fc50eb5dc9ed48b99f244b7cad75aff1c9bc91c10c3a29de8cdd88'
FORMULA = 'project_vinje((1-w)*accepted_joint_anchor_already_projected+w*raw_tcn), float64 arithmetic; no smoothing'


def verify_files(directory: Path, manifest: dict, year: int) -> dict:
    """Require every fixed runner output and hash before decoding numeric files."""
    names = {'train_cuda_tcn.py', 'protocol.json', 'normalization.npz',
             'epoch_training_case_orders.npy', f'tcn_{year}.pt',
             f'validation_{year}.npz', f'replay_predictions_{year}.npy'}
    require(set(manifest['artifacts']) == names, 'Returned artifact allowlist differs from fixed trainer')
    files = list(directory.rglob('*'))
    require(not any(path.is_symlink() for path in files), 'Returned directory contains symlinks')
    actual = {path.relative_to(directory).as_posix() for path in files if path.is_file()}
    require(actual == names | {'manifest.json'}, 'Returned directory allowlist differs')
    verified = {}
    for relative, expected in manifest['artifacts'].items():
        path = directory / safe_relative(relative)
        require(path.is_file() and not path.is_symlink(), f'Missing file or symlink: {relative}')
        require(digest(path) == expected, f'Returned artifact checksum mismatch: {relative}')
        verified[relative] = expected
    return verified


def verify_training_arrays(directory: Path, protocol: dict, year: int) -> dict:
    """Check saved normalizer scope and full-case coverage without real features."""
    expected_ids = np.asarray(protocol['folds'][str(year)]['train_ids'])
    features = protocol['feature_columns']
    with np.load(directory / 'normalization.npz', allow_pickle=False) as saved:
        require(set(saved.files) == {'mean', 'scale', 'columns', 'training_case_ids'}, 'Normalizer schema changed')
        mean, scale = saved['mean'], saved['scale']
        require(saved['columns'].tolist() == features, 'Normalizer feature order changed')
        require(np.array_equal(saved['training_case_ids'], expected_ids), 'Normalizer includes different training cases')
        require(mean.dtype == scale.dtype == np.float64 and mean.shape == scale.shape == (len(features),),
                'Normalizer must contain one float64 statistic per feature')
        require(np.isfinite(mean).all() and np.isfinite(scale).all() and (scale >= 1e-6).all(),
                'Invalid saved normalization statistics')
    orders = np.load(directory / 'epoch_training_case_orders.npy', allow_pickle=False)
    require(orders.shape == (24, len(expected_ids)) and np.issubdtype(orders.dtype, np.integer),
            'Expected 24 complete-case epoch orders')
    for order in orders:
        require(np.array_equal(np.sort(order), np.sort(expected_ids)), 'Epoch omits/repeats/adds training cases')
    return {'normalizer_dtype': 'float64', 'feature_count': len(features), 'epochs': 24,
            'cases_per_epoch': len(expected_ids), 'training_statistics_recomputed': False,
            'scope': 'Saved training ID/feature scope and finite statistics verified; no real feature matrix loaded locally'}


def verify_download(directory: Path, protocol: dict, protocol_sha: str,
                    year: int, baseline: dict, bundle: dict, lock: dict | None = None,
                    lock_sha: str | None = None) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Validate trusted source provenance, fold boundaries, OOF arrays and replay."""
    manifest = json.loads((directory / 'manifest.json').read_text())
    require(manifest['status'] == 'complete' and manifest['mode'] == 'fit' and manifest['year'] == year,
            'Expected complete requested CUDA training fold')
    require(manifest['seed'] == 4173 and manifest['completed_epochs'] == 24,
            'Fixed seed/24-epoch fit did not complete')
    require(manifest['protocol_sha256'] == protocol_sha
            and manifest['trainer_sha256'] == protocol['trainer_sha256'], 'Returned protocol or trainer changed')
    verify_files(directory, manifest, year)
    require(digest(directory / 'protocol.json') == protocol_sha
            and digest(directory / 'train_cuda_tcn.py') == protocol['trainer_sha256'], 'Copied protocol/source bytes changed')
    require(manifest['bundle'] == bundle, 'Returned data/source bundle lineage changed')
    require(manifest['baseline_reference_sha256'] == digest(ROOT / 'artifacts/lgb-normalized-lp-001/manifest.json'),
            'Frozen baseline reference changed')
    expected = protocol['folds'][str(year)]
    require(manifest['train_ids'] == expected['train_ids'] and manifest['validation_ids'] == expected['validation_ids'],
            'Purged chronological IDs changed')
    require(manifest['purge_days'] == 6 and manifest['all_training_case_hours'] is True
            and manifest['training_rows'] == 168 * len(expected['train_ids']), 'Training scope changed')
    require(manifest['early_stopping'] is None and manifest['checkpoint_selection'] == 'fixed final epoch24 only',
            'Unexpected validation-dependent stopping or checkpoint selection')
    if year == 2022:
        require(lock is not None and manifest.get('selection_lock_sha256') == lock_sha
                and manifest.get('selection_lock') == lock, 'Remote 2022 fit was not gated by this exact local lock')
    else:
        require('selection_lock' not in manifest, '2021 unexpectedly references a selection lock')
    def canonical_spec(spec: dict) -> dict:
        result = json.loads(json.dumps(spec))
        result.get('lp', {}).pop('cache_directory', None)
        return result
    require(canonical_spec(manifest['feature_spec']) == canonical_spec(baseline['feature_spec']),
            'Feature formulas differ beyond the documented portable cache path')
    cuda = manifest['cuda']
    require(cuda['selected_device'] == 'cuda:0' and cuda['cpu_fallback'] is False
            and cuda['deterministic_algorithms'] is True and cuda['float32'] is True
            and cuda['amp'] is False and cuda['compile'] is False
            and cuda['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8' and cuda['devices'], 'CUDA execution settings changed')
    require(manifest['environment']['venv']['system_site_packages'] is True
            and manifest['environment']['venv']['package_actions'] == 'none', 'Reviewed hosted environment changed')
    epochs = manifest['epochs']
    require(len(epochs) == 24, 'Missing epoch log')
    for i, epoch in enumerate(epochs):
        expected_lr = .0001 + (.001 - .0001) * (1 + math.cos(math.pi * i / 24)) / 2
        require(epoch['epoch'] == i + 1 and math.isfinite(epoch['training_bce'])
                and epoch['training_bce'] >= 0 and epoch['seconds'] >= 0
                and abs(epoch['learning_rate'] - expected_lr) <= 1e-12, 'Epoch/schedule log differs from fixed recipe')
    stats = verify_training_arrays(directory, protocol, year)
    with np.load(directory / f'validation_{year}.npz', allow_pickle=False) as saved:
        require(set(saved.files) == {'predictions', 'labels', 'case_ids', 'units'}, 'OOF schema changed')
        values, labels, ids, units = saved['predictions'], saved['labels'], saved['case_ids'], saved['units'].tolist()
    require(ids.tolist() == expected['validation_ids'] and units == manifest['units'] == list(DEFAULT_UNITS),
            'Returned OOF case or unit order changed')
    require(values.dtype == np.float32 and values.shape == labels.shape == (len(ids) * 168, 14)
            and np.isfinite(values).all() and values.min() >= 0 and values.max() <= 1
            and np.isin(labels, [0, 1]).all(), 'Invalid raw OOF probabilities or binary targets')
    replay = np.load(directory / f'replay_predictions_{year}.npy', allow_pickle=False)
    proof = manifest['checkpoint_replay']
    require(proof['fresh_model'] is True and proof['saved_float64_normalizer'] is True
            and proof['eval_inference_mode'] is True and proof['bitwise_equal'] is True
            and proof['max_abs_probability_difference'] == 0., 'Fresh checkpoint replay proof missing')
    require(replay.dtype == values.dtype and replay.shape == values.shape and replay.tobytes() == values.tobytes(),
            'Saved replay predictions differ from OOF bytes')
    require(manifest['checkpoint_sha256'] == manifest['artifacts'][f'tcn_{year}.pt']
            and manifest['validation_sha256'] == manifest['artifacts'][f'validation_{year}.npz']
            and proof['replay_predictions_sha256'] == manifest['artifacts'][f'replay_predictions_{year}.npy'],
            'Checkpoint/OOF/replay provenance hashes disagree')
    score = float(roc_auc_score(labels.ravel(), values.ravel()))
    require(abs(score - manifest['raw_micro_auc']) <= 2e-12, 'Independent local raw AUC differs from hosted score')
    for j, unit in enumerate(units):
        require(abs(float(roc_auc_score(labels[:, j], values[:, j])) - manifest['unit_auc'][unit]) <= 2e-12,
                f'Independent local unit AUC differs: {unit}')
    return {'manifest_sha256': digest(directory / 'manifest.json'), 'artifacts': manifest['artifacts'],
            'raw_micro_auc': score, 'training_arrays': stats, 'cuda': cuda,
            'hosted_environment': manifest['environment'], 'checkpoint_replay': proof,
            'local_checkpoint_inference': False,
            'cross_platform_feature_bitwise_equality': 'Not claimed; no local feature rebuild'}, values, labels, ids, units


def score_blends(prior: np.ndarray, raw: np.ndarray, labels: np.ndarray,
                 units: list[str], weights: list[float]) -> dict[str, float]:
    """Evaluate only requested fixed weights, keeping the whole 14-unit matrix."""
    require(np.array_equal(project_vinje(prior, units), prior), 'Accepted anchor is not already projected')
    return {str(weight): float(roc_auc_score(labels.ravel(),
             project_vinje((1 - weight) * prior.astype(np.float64) + weight * raw.astype(np.float64), units).ravel()))
            for weight in weights}


def main() -> None:
    """Select 2021 locally or evaluate only its locked weight on returned 2022."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, choices=[2021, 2022], required=True)
    parser.add_argument('--gpu-run', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--smoke-manifest', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=ROOT / 'docs/cuda_tcn_protocol.json')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--selection-lock', type=Path)
    parser.add_argument('--selection-lock-sha256')
    args = parser.parse_args()
    require('/' not in args.run_id and args.run_id not in ['', '.', '..'], 'Run ID must be one directory name')
    out = ROOT / 'artifacts' / args.run_id
    out.mkdir(exist_ok=False)
    started = time.time()
    report = {'run_id': args.run_id, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'year': args.year, 'command': shlex.join([sys.executable, *sys.argv]),
              'executed_snapshot_root': str(SNAPSHOT_ROOT), 'workspace_root': str(ROOT),
              'training': 'none; verify downloaded arrays/checkpoint hashes and select fixed scalar weights',
              'formula': FORMULA, 'kaggle_submission': None,
              'reuse_caveat': 'Chronological development years have been repeatedly reused; 2022 is an acceptance check, not an untouched holdout.',
              'environment': {'python': sys.version, 'platform': platform.platform(), 'threads': 2,
                              'packages': {p: importlib.metadata.version(p) for p in ['numpy', 'scikit-learn']}}}
    try:
        protocol_sha = digest(args.protocol)
        require(protocol_sha == PROTOCOL_SHA256, 'Frozen protocol SHA256 changed')
        protocol = json.loads(args.protocol.read_text())
        require(protocol['version'] == 'cuda_tcn_v2_joint_anchor' and protocol['blend_neural_weights'] == WEIGHTS
                and protocol['epochs'] == 24 and protocol['seed'] == 4173 and len(protocol['feature_columns']) == 395,
                'Unexpected frozen TCN protocol')
        require(digest(SNAPSHOT_ROOT / 'scripts/train_cuda_tcn.py') == digest(ROOT / 'scripts/train_cuda_tcn.py') == protocol['trainer_sha256'], 'Local frozen trainer changed')
        report['training_protocol_sha256'] = protocol_sha
        lock, lock_sha = None, None
        if args.year == 2022:
            require(args.selection_lock is not None and args.selection_lock_sha256 is not None,
                    '2022 requires the 2021 lock and explicit checksum')
            lock_sha = args.selection_lock_sha256
            lock = validate_selection_lock(args.selection_lock, lock_sha, protocol, protocol_sha)
            require(lock['formula'] == FORMULA, 'Locked blend formula changed')
            selected = lock['selected_neural_weight']
            require(lock['selector_sha256'] == digest(Path(__file__)), 'Selector changed after 2021 lock')
            require(set(lock['scores_2021']) == {str(w) for w in WEIGHTS}
                    and all(math.isfinite(x) for x in lock['scores_2021'].values()), 'Invalid frozen selection grid/scores')
            require(selected == max(WEIGHTS, key=lambda w: lock['scores_2021'][str(w)])
                    and lock['selected_2021_micro_auc'] == lock['scores_2021'][str(selected)]
                    and lock['anchor_2021_micro_auc'] == lock['scores_2021']['0.0'], 'Inconsistent frozen selection')
            report['selection_lock_sha256'] = lock_sha
        else:
            require(args.selection_lock is None and args.selection_lock_sha256 is None, '2021 selection must be new')
        bundle_path = ROOT / 'artifacts/kaggle-neural-bundle-001/payload/bundle_manifest.json'
        require(digest(bundle_path) == protocol['bundle_manifest_sha256'], 'Sealed bundle manifest changed')
        bundle = json.loads(bundle_path.read_text())
        bundle_files = {entry['path']: entry['sha256'] for entry in bundle['payload_files']}
        require(digest(SNAPSHOT_ROOT / 'sintef_project/postprocess.py') == digest(ROOT / 'sintef_project/postprocess.py') == bundle_files['sintef_project/postprocess.py'],
                'Fixed projection source changed')
        baseline_path = ROOT / 'artifacts/lgb-normalized-lp-001/manifest.json'
        require(digest(baseline_path) == bundle_files['artifacts/lgb-normalized-lp-001/manifest.json'],
                'Sealed baseline changed')
        baseline = json.loads(baseline_path.read_text())
        for path, expected in protocol['data'].items():
            require(digest(ROOT / path) == expected, f'Original data bytes changed: {path}')
        anchor_dir = ROOT / 'artifacts' / protocol['anchor']['run_id']
        anchor_path = anchor_dir / 'analysis.json'
        require(digest(anchor_path) == protocol['anchor']['analysis_sha256'], 'Fixed accepted joint anchor changed')
        anchor = json.loads(anchor_path.read_text())
        require(anchor['status'] == 'complete' and anchor['accepted'] is True, 'Anchor recipe was not accepted')
        directory = args.gpu_run.resolve()
        manifest_path = directory / 'manifest.json'
        require(digest(manifest_path) == args.manifest_sha256, 'Downloaded manifest differs from hosted SHA256')
        returned = json.loads(manifest_path.read_text())
        require(digest(args.smoke_manifest) == returned['smoke_manifest_sha256'], 'Original successful smoke hash differs')
        smoke = json.loads(args.smoke_manifest.read_text())
        require(smoke['status'] == 'complete' and smoke['mode'] == 'smoke'
                and smoke['protocol_sha256'] == protocol_sha and smoke['smoke']['synthetic_only'] is True
                and smoke['cuda'] == returned['cuda'], 'Original matching successful smoke not established')
        proof, raw, labels, ids, units = verify_download(directory, protocol, protocol_sha, args.year, baseline, bundle, lock, lock_sha)
        report['download_verification'] = proof
        prior_path = anchor_dir / f'selected_validation_{args.year}.npz'
        require(digest(prior_path) == anchor['artifacts'][prior_path.name], 'Anchor OOF bytes changed')
        with np.load(prior_path, allow_pickle=False) as prior_saved:
            require(np.array_equal(prior_saved['labels'], labels) and np.array_equal(prior_saved['case_ids'], ids)
                    and prior_saved['units'].tolist() == units, 'Anchor versus returned neural IDs/labels/units differ')
            prior = prior_saved['predictions']
        report['inputs'] = {str(path): digest(path) for path in
                           [args.protocol, bundle_path, baseline_path, anchor_path, prior_path, directory / 'manifest.json', args.smoke_manifest]}
        scores = score_blends(prior, raw, labels, units, WEIGHTS if args.year == 2021 else sorted(set([0., selected])))
        if args.year == 2021:
            selected = max(WEIGHTS, key=lambda weight: scores[str(weight)])
            lock = {'version': 'cuda_tcn_2021_blend_lock_v1', 'selection_year': 2021,
                    'locked_utc': datetime.now(timezone.utc).isoformat(), 'training_protocol_sha256': protocol_sha,
                    'anchor_run': protocol['anchor']['run_id'], 'anchor_sha256': protocol['anchor']['analysis_sha256'],
                    'candidate_neural_weights': WEIGHTS, 'selected_neural_weight': selected,
                    'training_2021_complete': True, 'neural_run_manifest_sha256': proof['manifest_sha256'],
                    'neural_validation_2021_sha256': proof['artifacts']['validation_2021.npz'],
                    'formula': FORMULA, 'scores_2021': scores, 'selected_2021_micro_auc': scores[str(selected)],
                    'anchor_2021_micro_auc': scores['0.0'], 'selector_sha256': digest(Path(__file__)),
                    'units': units, 'case_ids': ids.tolist(), 'reuse_caveat': report['reuse_caveat']}
            (out / 'selection_2021.json').write_text(json.dumps(lock, indent=2))
            report['selection_lock_sha256'] = digest(out / 'selection_2021.json')
            validate_selection_lock(out / 'selection_2021.json', report['selection_lock_sha256'], protocol, protocol_sha)
        chosen = project_vinje((1 - selected) * prior.astype(np.float64) + selected * raw.astype(np.float64), units)
        np.savez_compressed(out / f'selected_validation_{args.year}.npz', predictions=chosen, labels=labels,
                            case_ids=ids, units=np.asarray(units, dtype=str))
        report.update(status='complete', selected_neural_weight=selected, scores=scores,
                      selected_micro_auc=scores[str(selected)], anchor_micro_auc=scores['0.0'],
                      raw_micro_auc=proof['raw_micro_auc'],
                      selected_unit_auc={u: float(roc_auc_score(labels[:, j], chosen[:, j])) for j, u in enumerate(units)})
        if args.year == 2022:
            report['accepted'] = selected > 0 and lock['selected_2021_micro_auc'] > lock['anchor_2021_micro_auc'] and scores[str(selected)] > scores['0.0']
            report['accepted_candidate'] = f'tcn_weight_{selected}' if report['accepted'] else protocol['anchor']['run_id']
            report['meets_transfer_gate'] = lock['selected_2021_micro_auc'] >= .998 and scores[str(selected)] >= .998
        print(json.dumps({k: report.get(k) for k in ['status', 'selected_neural_weight', 'raw_micro_auc', 'selected_micro_auc', 'anchor_micro_auc', 'accepted', 'selection_lock_sha256']}), flush=True)
    except BaseException:
        report['status'] = 'failed'
        report['error'] = traceback.format_exc()
        raise
    finally:
        for relative in ['scripts/select_cuda_tcn_blend.py', 'scripts/train_cuda_tcn.py', 'sintef_project/postprocess.py']:
            target = out / 'source' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(SNAPSHOT_ROOT / relative, target)
        report['runtime_seconds'] = time.time() - started
        report['artifacts'] = {str(path.relative_to(out)): digest(path) for path in out.rglob('*')
                               if path.is_file() and path.name != 'manifest.json'}
        (out / 'manifest.json').write_text(json.dumps(report, indent=2))
        with (ROOT / 'docs/experiments.md').open('a') as stream:
            stream.write(f'\n## {args.run_id} — {report["timestamp_utc"]}\n\nStatus: {report["status"]}. Command: `{report["command"]}`. Local CUDA TCN download verification and fixed [0,.1,.25,.5] selection; no model fit. Frozen accepted joint anchor; 2021 lock precedes separate 2022 fit. Selected weight {report.get("selected_neural_weight")}, micro AUC {report.get("selected_micro_auc")}, anchor {report.get("anchor_micro_auc")}. Accepted: {report.get("accepted")}. Complete hashes, environment, split IDs and replay provenance: `artifacts/{args.run_id}/manifest.json`. Reused chronological development folds; no Kaggle submission or public-score tuning.\n')


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
