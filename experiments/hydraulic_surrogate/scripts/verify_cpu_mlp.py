"""Verify a completed CPU MLP ablation using small arrays and own checkpoints."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import platform
import shlex
import shutil
import sys
import time
import traceback

for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[variable] = '2'
import numpy as np
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sintef_project.mlp_helpers import portable_predict
from sintef_project.postprocess import project_vinje


def digest(path: Path) -> str:
    """Return a streaming SHA256 content digest."""
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    """Check saved mixtures, normalization metadata and two checkpoint formats."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-run', default='cpu-mlp-lp-001')
    parser.add_argument('--run-id', default='verify-cpu-mlp-lp-001')
    args = parser.parse_args()
    out = ROOT / 'artifacts' / args.run_id
    out.mkdir(exist_ok=False)
    started = time.time()
    proof = {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'source_run': args.source_run,
             'command': shlex.join([sys.executable, *sys.argv]), 'folds': {},
             'environment': {'platform': platform.platform(), 'python': sys.version,
                             'packages': {name: importlib.metadata.version(name)
                                          for name in ['numpy', 'scipy', 'scikit-learn', 'threadpoolctl']}},
             'scope': 'Small saved arrays, metadata and checkpoints only; no full feature matrix or fit. Full validation replay was performed during source run; this verifier additionally compares both loaded formats on64 fresh synthetic normalized rows.',
             'kaggle': None}
    try:
        directory = ROOT / 'artifacts' / args.source_run
        path = directory / 'manifest.json'
        manifest = json.loads(path.read_text())
        if manifest['status'] != 'complete':
            raise ValueError('Source run must complete before verification')
        proof['source_manifest_sha256'] = digest(path)
        checked = 0
        for base, hashes in [(directory, manifest['artifacts']), (ROOT, manifest['inputs']),
                             (ROOT, manifest['data']), (directory / 'source', manifest['source'])]:
            for relative, expected in hashes.items():
                if digest(base / relative) != expected:
                    raise ValueError(f'Changed registered input: {base / relative}')
                checked += 1
        for relative in ['sintef_project/mlp_helpers.py', 'sintef_project/postprocess.py']:
            if digest(ROOT / relative) != manifest['source'][relative]:
                raise ValueError(f'Changed verification dependency: {relative}')
        lock_path = directory / 'selection_2021.json'
        lock = json.loads(lock_path.read_text())
        if digest(lock_path) != manifest['selection_sha256'] or set(lock['folds']) != {'2021'}:
            raise ValueError('Selection lock changed or contains later fold')
        if digest(directory / 'protocol.json') != manifest['protocol_sha256']:
            raise ValueError('Protocol hash changed')
        selected = manifest['selected_weight']
        if (lock['selected_weight'] != selected
                or selected != max([0., .1, .25, .5], key=lambda w: lock['folds']['2021']['scores'][str(w)])):
            raise ValueError('Unexpected selected weight or tie handling')
        rng = np.random.default_rng(4173)
        probe = rng.standard_normal((64, 395), dtype=np.float32)
        for year in ['2021', '2022']:
            fold = manifest['folds'][year]
            sample = np.load(directory / f'train_sample_{year}.npy', allow_pickle=False)
            original = np.load(ROOT / 'artifacts' / manifest['base_run'] / f'train_sample_{year}.npy', allow_pickle=False)
            np.testing.assert_array_equal(sample, original)
            with np.load(directory / f'normalization_{year}.npz', allow_pickle=False) as stats:
                if (stats['mean'].shape != (395,) or stats['mean'].dtype != np.float64
                        or stats['standard_deviation'].dtype != np.float64 or stats['scale'].dtype != np.float64
                        or stats['features'].tolist() != manifest['features'] or stats['sampled_rows'].item() != len(sample)):
                    raise ValueError('Invalid saved normalization schema')
                np.testing.assert_array_equal(stats['scale'], np.where(stats['standard_deviation'] == 0., 1., stats['standard_deviation']))
                if not all(np.isfinite(stats[name]).all() for name in ['mean', 'standard_deviation', 'scale']):
                    raise ValueError('Nonfinite normalization parameters')
            checkpoint_path = directory / fold['checkpoint']['pickle']
            if digest(checkpoint_path) != fold['checkpoint']['pickle_sha256']:
                raise ValueError('Changed trusted pickle')
            with checkpoint_path.open('rb') as stream:
                model = pickle.load(stream)
            if json.loads(json.dumps(model.get_params())) != manifest['model_parameters']:
                raise ValueError('Expanded model settings differ')
            if model.n_iter_ != 24 or model.n_outputs_ != 14 or model.out_activation_ != 'logistic':
                raise ValueError('Unexpected fitted architecture or epochs')
            if [list(value.shape) for value in model.coefs_] != [[395, 128], [128, 64], [64, 14]]:
                raise ValueError('Unexpected layer dimensions')
            portable_path = directory / fold['checkpoint']['portable']
            with np.load(portable_path, allow_pickle=False) as saved:
                coefficients = [saved[f'coefficient_{i}'] for i in range(3)]
                intercepts = [saved[f'intercept_{i}'] for i in range(3)]
                for actual, expected in zip(coefficients + intercepts, model.coefs_ + model.intercepts_):
                    np.testing.assert_array_equal(actual, expected)
                if saved['features'].tolist() != manifest['features'] or saved['units'].tolist() != fold['units']:
                    raise ValueError('Portable feature or unit order differs')
            with threadpool_limits(2):
                np.testing.assert_array_equal(model.predict_proba(probe), portable_predict(probe, coefficients, intercepts))
            with np.load(directory / f'validation_{year}.npz', allow_pickle=False) as saved:
                labels, predictions, raw, ids, units = saved['labels'], saved['predictions'], saved['neural_predictions'], saved['case_ids'], saved['units'].tolist()
            anchor_path = ROOT / 'artifacts' / manifest['anchor_run'] / f'selected_validation_{year}.npz'
            with np.load(anchor_path, allow_pickle=False) as anchor:
                np.testing.assert_array_equal(labels, anchor['labels'])
                np.testing.assert_array_equal(ids, anchor['case_ids'])
                if units != anchor['units'].tolist():
                    raise ValueError('Anchor unit alignment mismatch')
                reconstructed = project_vinje((1-selected) * anchor['predictions'].astype(np.float64)
                                                + selected * raw.astype(np.float64), units)
            np.testing.assert_array_equal(predictions, reconstructed)
            if predictions.shape != (len(ids) * 168, 14) or not np.isfinite(predictions).all() or ((predictions < 0) | (predictions > 1)).any():
                raise ValueError('Invalid selected probabilities')
            score = float(roc_auc_score(labels.ravel(), predictions.ravel()))
            raw_score = float(roc_auc_score(labels.ravel(), raw.ravel()))
            if score != fold['selected']['micro_auc'] or raw_score != fold['raw_neural']['micro_auc']:
                raise ValueError('Independent AUC differs')
            proof['folds'][year] = {'selected_micro_auc': score, 'raw_neural_micro_auc': raw_score,
                                   'exact_selected_mixture': True, 'checkpoint_formats_exact': True,
                                   'independent_synthetic_replay_rows': 64,
                                   'source_full_validation_replay_rows': fold['checkpoint']['replayed_rows'],
                                   'normalization_metadata_valid': True, 'exact_sample_rows': True}
        accepted = selected > 0. and all(fold['selected']['micro_auc'] > fold['anchor']['micro_auc'] for fold in manifest['folds'].values())
        if manifest['accepted'] != accepted:
            raise ValueError('Acceptance differs from frozen rule')
        proof.update(status='passed', checked_hashes=checked, selected_weight=selected, accepted=accepted)
        print(json.dumps(proof, indent=2), flush=True)
    except BaseException:
        proof['status'] = 'failed'
        proof['error'] = traceback.format_exc()
        raise
    finally:
        shutil.copy2(__file__, out / 'source.py')
        proof['source_sha256'] = digest(Path(__file__))
        proof['runtime_seconds'] = time.time() - started
        (out / 'manifest.json').write_text(json.dumps(proof, indent=2))
        with (ROOT / 'docs/experiments.md').open('a') as stream:
            stream.write(f'\n## {args.run_id} — {proof["timestamp_utc"]}\n\nStatus:{proof["status"]}. Command:`{proof["command"]}`. Runtime{proof["runtime_seconds"]:.2f}s. Small-array CPU-MLP verification: source/data/input/model hashes, exact sample rows, normalization metadata, trusted pickle/portable parameter identity and64 synthetic-row replay, exact frozen probability mixture/labels/IDs/units/AUC, immutable2021 lock and fixed acceptance rule. Full validation checkpoint replay remains the source-run check; no large feature reconstruction or new fit here. Proof:`artifacts/{args.run_id}/manifest.json`. No Kaggle submission.\n')


if __name__ == '__main__':
    main()
