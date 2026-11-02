"""Final-fit or independently replay an accepted nonlinear-PQ specialist export."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import sys
import time
import traceback

for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[variable] = '2'

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_nonlinear_pq_specialists import BASE, CACHE, TARGET_UNITS, build_608, compose, validate_lock
from sintef_project.nonlinear_pq import add_nonlinear_pq_features, digest

ANCHOR_FINAL = ROOT / 'artifacts/submission-lp-context-joint-blend-001'
BASE_FINAL = ROOT / 'artifacts/lgb-normalized-lp-context-joint-final-001'
FOLD1 = ROOT / 'artifacts/nonlinear-pq-specialists-2021-001'
FOLD2 = ROOT / 'artifacts/nonlinear-pq-specialists-2022-001'


def verify_files(root: Path, hashes: dict) -> None:
    """Verify every registered file without allowing a changed source or artifact."""
    for name, sha in hashes.items():
        if digest(root / name) != sha:
            raise ValueError(f'Changed registered file: {root / name}')


def export_matrix(prediction: np.ndarray, template: pd.DataFrame, units: list[str],
                  path: Path) -> dict:
    """Write and read back the exact ordered competition schema and probabilities."""
    expected = ['Run No'] + [f'result_committed_{unit}_t{hour}' for unit in units for hour in range(168)]
    if template.columns.tolist() != expected or prediction.shape != (len(template) * 168, len(units)):
        raise ValueError('Competition row/column order changed')
    if not np.isfinite(prediction).all() or prediction.min() < 0 or prediction.max() > 1:
        raise ValueError('Invalid final probabilities')
    wide = prediction.reshape(len(template), 168, len(units)).transpose(0, 2, 1).reshape(len(template), -1)
    frame = pd.DataFrame(wide, columns=template.columns[1:])
    frame.insert(0, 'Run No', template['Run No'].to_numpy())
    frame.to_csv(path, index=False, float_format='%.9g')
    restored = pd.read_csv(path)
    if restored.columns.tolist() != expected or not restored['Run No'].equals(template['Run No']):
        raise ValueError('CSV readback schema/IDs changed')
    error = float(np.max(np.abs(restored.iloc[:, 1:].to_numpy() - wide)))
    if not np.isfinite(restored.iloc[:, 1:].to_numpy()).all() or error > 5.1e-10:
        raise ValueError('CSV roundtrip changed probabilities')
    return {'shape': list(frame.shape), 'readback_max_error': error,
            'submission_sha256': digest(path), 'probability_range': [float(prediction.min()), float(prediction.max())]}


def main() -> None:
    """Require both independently reviewed folds before any final fitting or export."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--inference-from', type=Path)
    parser.add_argument('--review-2021', type=Path, default=ROOT / 'artifacts/verify-nonlinear-pq-selection-2021-001/manifest.json')
    parser.add_argument('--review-2022', type=Path, default=ROOT / 'artifacts/verify-nonlinear-pq-selection-2022-001/manifest.json')
    parser.add_argument('--anchor-review', type=Path, default=ROOT / 'artifacts/verify-lp-context-joint-export-003/manifest.json')
    args = parser.parse_args()
    out = ROOT / 'artifacts' / args.run_id
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    report = {'run_id': args.run_id, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'command': shlex.join([sys.executable, *sys.argv]), 'status': 'initializing',
        'mode': 'replay' if args.inference_from else 'final_fit',
        'environment': {'python': sys.version, 'platform': platform.platform(), 'machine': platform.machine(),
                        'packages': {n: importlib.metadata.version(n) for n in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'lightgbm']}},
        'models': {}, 'inputs': {}, 'seed': 4173, 'early_stopping': None, 'kaggle_submission': None}
    try:
        protocol_path = ROOT / 'docs/nonlinear_pq_protocol.json'
        protocol = json.loads(protocol_path.read_text())
        protocol_sha = digest(protocol_path)
        verify_files(ROOT, protocol['source'])
        verify_files(ROOT, protocol['data'])
        lock_path = FOLD1 / 'selection_2021.json'
        lock_sha = digest(lock_path)
        lock = validate_lock(lock_path, lock_sha, protocol_sha)
        for year, directory, review_path in [(2021, FOLD1, args.review_2021), (2022, FOLD2, args.review_2022)]:
            manifest = json.loads((directory / 'manifest.json').read_text())
            review = json.loads(review_path.read_text())
            if not (manifest['status'] == 'complete' and manifest['beats_anchor_and_control'] is True
                    and review['status'] == 'passed' and review['eligible'] is True and review['year'] == year
                    and review['model_manifest_sha256'] == digest(directory / 'manifest.json')
                    and review['selection_lock_sha256'] == lock_sha == manifest['selection_lock_sha256']
                    and review['protocol_sha256'] == protocol_sha == manifest['protocol_sha256']
                    and review['selected_weight'] == lock['selected_weight'] == manifest['selected_weight']):
                raise ValueError(f'Incomplete or mismatched independent {year} acceptance review')
            if year == 2022 and manifest.get('accepted') is not True:
                raise ValueError('2022 rejected the selected recipe')
            verify_files(directory, manifest['artifacts'])
            report['inputs'][str(directory / 'manifest.json')] = digest(directory / 'manifest.json')
            report['inputs'][str(review_path)] = digest(review_path)
        fold2 = json.loads((FOLD2 / 'manifest.json').read_text())
        anchor_manifest = json.loads((ANCHOR_FINAL / 'manifest.json').read_text())
        anchor_review = json.loads(args.anchor_review.read_text())
        if not (anchor_review['status'] == 'complete' and anchor_review['bitwise_export_array_match'] is True
                and anchor_review['submission_sha256'] == anchor_manifest['submission_sha256']
                and anchor_manifest['selection_sha256'] == protocol['anchor_analysis_sha256']
                and anchor_manifest['weights'] == protocol['anchor_weights']):
            raise ValueError('Final anchor does not implement the frozen validated blend')
        verify_files(ROOT, anchor_review['inputs'])
        for replay in anchor_review['replays'].values():
            if replay['bitwise_array_match'] is not True:
                raise ValueError('An anchor component lacks exact checkpoint replay')
            verify_files(ROOT, replay['checkpoints'])
        verify_files(ANCHOR_FINAL, anchor_manifest['artifacts'])
        report['inputs'].update({str(ANCHOR_FINAL / 'manifest.json'): digest(ANCHOR_FINAL / 'manifest.json'),
                                str(args.anchor_review): digest(args.anchor_review), str(lock_path): lock_sha,
                                str(protocol_path): protocol_sha})
        baseline = json.loads((BASE / 'manifest.json').read_text())
        base_final = json.loads((BASE_FINAL / 'manifest.json').read_text())
        if digest(BASE / 'manifest.json') != protocol['base_manifest_sha256'] or base_final['status'] != 'complete':
            raise ValueError('Final sample donor or validation baseline changed')
        verify_files(BASE_FINAL, base_final['artifacts'])
        if base_final['features'] != baseline['features'] or base_final['feature_spec'] != baseline['feature_spec']:
            raise ValueError('Final sample donor uses another feature recipe')
        for name, sha in protocol['source'].items():
            target = out / 'source' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        target = out / 'source/scripts/finalize_nonlinear_pq.py'
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(__file__, target)
        report.update(protocol_sha256=protocol_sha, selection_lock_sha256=lock_sha,
                      data=protocol['data'], model_parameters=protocol['model_parameters'],
                      weight=lock['selected_weight'], targets=TARGET_UNITS,
                      validation_micro_auc={'2021': lock['selected_micro_auc'], '2022': fold2['selected_micro_auc']})
        original = None
        if args.inference_from:
            original = json.loads((args.inference_from / 'manifest.json').read_text())
            if original['status'] != 'complete' or original['mode'] != 'final_fit':
                raise ValueError('Replay requires a completed final fit')
            verify_files(args.inference_from, original['artifacts'])
            if original['protocol_sha256'] != protocol_sha or original['selection_lock_sha256'] != lock_sha:
                raise ValueError('Replay source protocol or selection changed')
            if original['finalizer_sha256'] != digest(Path(__file__)):
                raise ValueError('Replay finalizer source changed')
            report['inputs'][str(args.inference_from / 'manifest.json')] = digest(args.inference_from / 'manifest.json')
        x, labels, train, mapping, meta = build_608(2022, baseline)
        cache_manifest = json.loads((CACHE / 'manifest.json').read_text())
        if digest(CACHE / 'manifest.json') != protocol['cache_manifest_sha256'] or digest(CACHE / 'features.npz') != protocol['cache_features_sha256']:
            raise ValueError('Frozen nonlinear feature cache changed')
        all_cases = pd.concat([train, mapping], ignore_index=True)
        if all_cases['Run No'].tolist() != cache_manifest['case_ids'] or pd.to_datetime(all_cases.starttime, utc=True).astype(str).tolist() != cache_manifest['case_dates']:
            raise ValueError('Final feature case order differs from frozen cache')
        x, specification = add_nonlinear_pq_features(x, ROOT, CACHE)
        if x.columns.tolist() != protocol['prefix_columns'] + protocol['append_columns']:
            raise ValueError('Final614-column schema changed')
        report.update(features=x.columns.tolist(), feature_spec={**meta['feature_spec'], 'nonlinear_pq': specification})
        sample_path = BASE_FINAL / 'train_sample_final.npy'
        sample = np.load(sample_path, allow_pickle=False)
        nt = len(train) * 168
        if len(sample) != int(nt * .4) or len(np.unique(sample)) != len(sample) or sample.min() < 0 or sample.max() >= nt:
            raise ValueError('Final saved sample is not the original40% unique train rows')
        if digest(sample_path) != base_final['artifacts'][sample_path.name]:
            raise ValueError('Final sample bytes changed')
        np.save(out / 'train_sample_final.npy', sample)
        report['inputs'][str(sample_path)] = digest(sample_path)
        report['train_ids'] = train['Run No'].tolist()
        report['test_ids'] = mapping['Run No'].tolist()
        report['units'] = meta['units']
        indices = [meta['units'].index(unit) for unit in TARGET_UNITS]
        xt, test = (None, x.iloc[nt:]) if original else (x.iloc[sample], x.iloc[nt:])
        del x
        raw = np.empty((len(test), 3), dtype=np.float64)
        print('FINAL_FEATURES', test.shape, 'mode', report['mode'], flush=True)
        for j, unit in enumerate(TARGET_UNITS):
            tic = time.monotonic()
            checkpoint = out / f'final_{unit}.txt'
            if original:
                source_checkpoint = args.inference_from / checkpoint.name
                if digest(source_checkpoint) != original['models'][unit]['checkpoint_sha256']:
                    raise ValueError('Replay checkpoint bytes changed')
                shutil.copy2(source_checkpoint, checkpoint)
                model = lgb.Booster(model_file=str(checkpoint))
                raw[:, j] = model.predict(test, num_threads=2)
            else:
                model = lgb.LGBMClassifier(**protocol['model_parameters'])
                model.fit(xt, labels[unit][sample])
                raw[:, j] = model.predict_proba(test)[:, 1]
                model.booster_.save_model(checkpoint)
                restored = lgb.Booster(model_file=str(checkpoint)).predict(test, num_threads=2)
                np.testing.assert_array_equal(raw[:, j], restored)
            report['models'][unit] = {'checkpoint_sha256': digest(checkpoint),
                                      'seconds': time.monotonic() - tic, 'exact_checkpoint_replay': True}
            print('FINAL_MODEL', unit, report['models'][unit], flush=True)
            del model
        np.save(out / 'raw_test_predictions.npy', raw)
        anchor = np.load(ANCHOR_FINAL / 'test_predictions.npy', allow_pickle=False)
        prediction = compose(anchor, raw, indices, lock['selected_weight'], meta['units'])
        np.save(out / 'test_predictions.npy', prediction)
        template = pd.read_csv(ROOT / 'sintef_project/sample_submission.csv')
        if len(template) != 725 or not mapping['Run No'].equals(template['Run No']):
            raise ValueError('Final test ID mapping changed')
        report.update(export_matrix(prediction, template, meta['units'], out / 'submission.csv'))
        if original:
            np.testing.assert_array_equal(raw, np.load(args.inference_from / 'raw_test_predictions.npy', allow_pickle=False))
            np.testing.assert_array_equal(prediction, np.load(args.inference_from / 'test_predictions.npy', allow_pickle=False))
            if report['submission_sha256'] != original['submission_sha256']:
                raise ValueError('Independent replay CSV checksum differs')
            report['independent_replay_exact'] = True
        report.update(status='complete', finalizer_sha256=digest(Path(__file__)))
        print(json.dumps({k: report.get(k) for k in ['status', 'mode', 'submission_sha256', 'independent_replay_exact']}), flush=True)
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        report['runtime_seconds'] = time.monotonic() - started
        report['artifacts'] = {str(p.relative_to(out)): digest(p) for p in out.rglob('*') if p.is_file() and p.name != 'manifest.json'}
        (out / 'manifest.json').write_text(json.dumps(report, indent=2))
        with (ROOT / 'docs/experiments.md').open('a') as stream:
            stream.write(f'\n## {args.run_id} — {report["timestamp_utc"]}\n\nStatus: {report["status"]}; mode: {report["mode"]}. Command: `{report["command"]}`. Three frozen614-feature specialists, original full-data40% sample, seed4173,350trees, two threads; no early stopping. Requires both independent chronological reviews and unchanged anchor/control gates. Final target-unit blend weight {report.get("weight")}; validation {report.get("validation_micro_auc")}. Checkpoints, raw/blended probabilities, full source/data/cache/sample environment provenance and schema/readback: `artifacts/{args.run_id}/manifest.json`. Runtime {report["runtime_seconds"]:.3f}s. Submission SHA256 {report.get("submission_sha256")}. Kaggle ID/score/rank/observation remain null.\n')


if __name__ == '__main__':
    main()
