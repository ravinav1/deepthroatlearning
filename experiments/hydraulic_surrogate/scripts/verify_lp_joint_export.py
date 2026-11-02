"""Verify the frozen joint-LP export and all three saved checkpoint replays.

No model fit, checkpoint inference, LP solve, or real feature construction is
performed. The exporter helper validates recipes/checkpoint schemas; this script
independently reconstructs the float64 blend from the original saved arrays.
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime, timezone
import importlib.metadata
import hashlib
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
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.export_ensemble import digest, verify_hashes, validate_final, read_submission
from sintef_project.postprocess import project_vinje

VALIDATION_RUNS = ['lgb-normalized-lp-001', 'lgb-normalized-lp-context-001', 'lgb-normalized-lp-joint-001']
FINAL_RUNS = ['lgb-normalized-lp-final-001', 'lgb-normalized-lp-context-final-001', 'lgb-normalized-lp-joint-final-001']
REPLAY_RUNS = ['replay-lp-final-001', 'replay-lp-context-final-001', 'replay-lp-joint-final-001']
WEIGHTS = [.1875, .5625, .25]
SELECTION_RUN = 'lp-joint-comparison-001'
EXPORT_RUN = 'submission-lp-joint-blend-001'
CONTEXT_EXPORT_MANIFEST = 'artifacts/submission-lp-context-blend-001/manifest.json'
ORIGINAL_CONTEXT_MANIFEST_SHA256 = '4a89ef8bcad009a2e70e5e4ab97fb66a621d46f91846446de1630b0ba4174a64'
EMPTY_KAGGLE = {'submission_id': None, 'score': None, 'rank': None, 'observation_time': None}


def require(condition: bool, message: str) -> None:
    """Fail closed without relying on Python assert optimization settings."""
    if not condition:
        raise ValueError(message)


def reconstruct_float64(arrays: Sequence[np.ndarray], coefficients: Sequence[float],
                        units: Sequence[str]) -> np.ndarray:
    """Promote each source BEFORE multiplication, accumulate in given order, project."""
    require(bool(arrays) and len(arrays) == len(coefficients), 'Every weight needs one source array')
    require(np.isfinite(coefficients).all() and min(coefficients) >= 0 and sum(coefficients) == 1.,
            'Expected finite convex coefficients')
    shape = arrays[0].shape
    values = np.zeros(shape, dtype=np.float64)
    for source, weight in zip(arrays, coefficients):
        require(source.shape == shape, 'Source array shapes differ')
        promoted = np.asarray(source, dtype=np.float64)
        require(np.isfinite(promoted).all() and promoted.min() >= 0 and promoted.max() <= 1,
                'Invalid raw source probabilities')
        values += weight * promoted
    return project_vinje(values, units)


def bitwise_equal(left: np.ndarray, right: np.ndarray) -> bool:
    """Compare dtype, shape and value bytes, including signed zeros."""
    return left.dtype == right.dtype and left.shape == right.shape and left.tobytes(order='C') == right.tobytes(order='C')


def verify_observation_only_update(current: dict, expected_sha256: str) -> dict:
    """Prove a historical manifest differs only in its Kaggle observation object.

    Match the exporter's original indent2 JSON serialization exactly after
    replacing only the known initially-empty observation dictionary. Any change
    to artifacts, weights, hashes, timestamps, data or other fields fails.
    """
    require(isinstance(current.get('kaggle'), dict), 'Expected a Kaggle observation object')
    historical = current.copy()
    historical['kaggle'] = EMPTY_KAGGLE.copy()
    reconstructed_sha = hashlib.sha256(json.dumps(historical, indent=2).encode()).hexdigest()
    require(reconstructed_sha == expected_sha256, 'Manifest changed outside the permitted Kaggle observation object')
    return {'historical_sha256': expected_sha256, 'reconstructed_historical_sha256': reconstructed_sha,
            'historical_kaggle': EMPTY_KAGGLE.copy(), 'current_kaggle': current['kaggle'],
            'allowed_changed_field': 'kaggle', 'all_other_manifest_fields_byte_reconstruction_verified': True}


def main() -> None:
    """Verify only completed fixed artifacts and emit an independent proof manifest."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', default='verify-lp-joint-export-001')
    args = parser.parse_args()
    out = ROOT / 'artifacts' / args.run_id
    out.mkdir(exist_ok=False)
    started = time.time()
    report = {'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'command': shlex.join([sys.executable, *sys.argv]),
              'environment': {'python': sys.version, 'platform': platform.platform(), 'threads': 2,
                              'packages': {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'lightgbm']}},
              'training_or_inference': 'none; saved-array and hash verification only',
              'inputs': {}, 'replays': {}, 'caches': {}, 'kaggle': None}

    def verified_manifest(directory: Path, filename: str = 'manifest.json',
                          *, allow_context_observation_update: bool = False) -> dict:
        path = directory / filename
        manifest = json.loads(path.read_text())
        require(manifest.get('status') == 'complete', f'Incomplete source: {directory.name}/{filename}')
        verify_hashes(directory, manifest['artifacts'])
        for key in ['source', 'source_code']:
            if key in manifest:
                verify_hashes(directory / 'source', manifest[key])
        for key in ['data', 'inputs']:
            if key in manifest:
                hashes = manifest[key].copy()
                if allow_context_observation_update and key == 'inputs' and CONTEXT_EXPORT_MANIFEST in hashes:
                    old_hash = hashes.pop(CONTEXT_EXPORT_MANIFEST)
                    require(old_hash == ORIGINAL_CONTEXT_MANIFEST_SHA256, 'Unexpected historical context manifest reference')
                    observed_path = ROOT / CONTEXT_EXPORT_MANIFEST
                    observed_hash = digest(observed_path)
                    if observed_hash != old_hash:
                        update = verify_observation_only_update(json.loads(observed_path.read_text()), old_hash)
                        update.update(path=CONTEXT_EXPORT_MANIFEST, current_sha256=observed_hash)
                        report['context_observation_metadata_update'] = update
                    else:
                        verify_hashes(ROOT, {CONTEXT_EXPORT_MANIFEST: old_hash})
                verify_hashes(ROOT, hashes)
        report['inputs'][str(path.relative_to(ROOT))] = digest(path)
        return manifest

    try:
        selection_dir = ROOT / 'artifacts' / SELECTION_RUN
        selection = verified_manifest(selection_dir, 'analysis.json')
        verify_hashes(ROOT, selection['sources'])
        lock_path = selection_dir / 'selection_2021.json'
        locked = json.loads(lock_path.read_text())
        require(set(locked['folds']) == {'2021'}, 'Selection lock contains another fold')
        require(selection.get('accepted') is True, 'Joint candidate was not accepted')
        for key, expected in [('source_order', VALIDATION_RUNS), ('selected_weights', WEIGHTS),
                              ('selected_candidate', 'joint_weight_0.25')]:
            require(selection[key] == locked[key] == expected, f'Frozen selection mismatch: {key}')
        require(selection.get('accepted_candidate', selection['selected_candidate']) == selection['selected_candidate'],
                'Accepted candidate differs from2021 choice')
        require(selection['selection_sha256'] == digest(lock_path), '2021 lock hash changed')
        require(all('validation_2022' not in key and 'train_sample_2022' not in key for key in locked['sources']),
                '2021 lock records a2022 prediction or sample')
        verify_hashes(ROOT, locked['sources'])
        require(locked['source_code'] == selection['source_code'], 'Selection code changed after lock')
        require(digest(ROOT / 'sintef_project/postprocess.py') == selection['source_code']['sintef_project/postprocess.py'],
                'Probability projection changed')
        units = selection['folds']['2021']['units']
        require(units == selection['folds']['2022']['units'], 'Validation unit order differs by year')
        template = pd.read_csv(ROOT / 'sintef_project/sample_submission.csv')
        mapping = pd.read_csv(ROOT / 'sintef_project/prediction_mapping.csv')
        expected_columns = ['Run No'] + [f'result_committed_{unit}_t{hour}' for unit in units for hour in range(168)]
        require(template.shape == (725, 2353) and template.columns.tolist() == expected_columns,
                'Unexpected exact competition template')
        require(mapping['Run No'].equals(template['Run No']), 'Mapping ID order differs from template')
        expected_shape = (725 * 168, 14)
        raw_arrays = []
        provenances = {}
        manifests = {}
        for validation_run, final_run, replay_run in zip(VALIDATION_RUNS, FINAL_RUNS, REPLAY_RUNS):
            validation_dir = ROOT / 'artifacts' / validation_run
            final_dir = ROOT / 'artifacts' / final_run
            replay_dir = ROOT / 'artifacts' / replay_run
            validation = verified_manifest(validation_dir)
            final = verified_manifest(final_dir)
            replay = verified_manifest(replay_dir)
            manifests[final_run] = final
            require(replay['arguments']['years'] == [], 'Replay unexpectedly contains validation training')
            require((ROOT / replay['arguments']['inference_from']).resolve() == final_dir.resolve(),
                    'Replay references another full fit')
            for key in ['features', 'feature_spec', 'data', 'model_parameters']:
                require(replay[key] == final[key], f'Replay source mismatch: {replay_run}/{key}')
            origin = replay['model_origin']
            require((ROOT / origin['manifest']).resolve() == (final_dir / 'manifest.json').resolve(),
                    'Replay model-origin path differs')
            require(origin['manifest_sha256'] == digest(final_dir / 'manifest.json'), 'Replay origin hash changed')
            expected_checkpoints = {str((final_dir / f'final_{unit}.txt').relative_to(ROOT)): digest(final_dir / f'final_{unit}.txt')
                                    for unit in units}
            origin_checkpoints = {str((ROOT / path).resolve().relative_to(ROOT)): value
                                  for path, value in origin['checkpoints'].items()}
            require(origin_checkpoints == expected_checkpoints, 'Replay checkpoint set or hashes differ')
            raw = np.load(final_dir / 'test_predictions.npy', allow_pickle=False)
            repeated = np.load(replay_dir / 'test_predictions.npy', allow_pickle=False)
            require(raw.shape == expected_shape and raw.dtype == np.float32, 'Unexpected raw prediction shape/dtype')
            require(bitwise_equal(raw, repeated), f'Raw checkpoint replay differs: {final_run}')
            csv_hash = digest(final_dir / 'submission.csv')
            require(csv_hash == digest(replay_dir / 'submission.csv') == final['submission_sha256'] == replay['submission_sha256'],
                    'Raw replay CSV differs')
            validated, provenance = validate_final(final_dir, validation, validation['arguments']['feature_set'], template, units)
            require(np.array_equal(validated, raw.astype(np.float64)), 'Final validation helper returned different probabilities')
            provenances[validation_run] = provenance
            raw_arrays.append(raw)
            report['replays'][final_run] = {'replay_run': replay_run, 'bitwise_array_match': True,
                                          'raw_csv_sha256': csv_hash, 'shape': list(raw.shape), 'dtype': str(raw.dtype),
                                          'raw_max_abs_error': 0., 'checkpoints': expected_checkpoints,
                                          'replay_command': replay['command']}
            # Verify the input-only LP cache referenced by the saved feature
            # specification, without loading its feature matrix or solving it.
            for spec in final['feature_spec'].values():
                if not isinstance(spec, dict) or 'cache_directory' not in spec:
                    continue
                cache_dir = Path(spec['cache_directory'])
                require(cache_dir.is_relative_to(ROOT / 'artifacts'), 'Unexpected cache path')
                cache_key = str(cache_dir.relative_to(ROOT))
                if cache_key not in report['caches']:
                    cache = verified_manifest(cache_dir)
                    require(digest(cache_dir / 'manifest.json') == spec['manifest_sha256'], 'LP cache manifest changed')
                    require(digest(cache_dir / 'features.npz') == spec['features_sha256'], 'LP cache features changed')
                    require(cache['source'] == spec['source_verified'] and cache['data'] == spec['data_verified'],
                            'LP cache source/data fingerprint differs')
                    report['caches'][cache_key] = {'manifest_sha256': spec['manifest_sha256'],
                                                  'features_sha256': spec['features_sha256'],
                                                  'verified_artifacts': len(cache['artifacts'])}
        # Preserve and check the earlier independently recorded base/context
        # replay proofs as well as freshly comparing their saved arrays above.
        base_proof_path = ROOT / 'artifacts/replay-lp-final-001/comparison.json'
        base_proof = json.loads(base_proof_path.read_text())
        require(base_proof['bitwise_equal'] is True, 'Existing base replay proof did not pass')
        require(base_proof['reference'] == 'artifacts/lgb-normalized-lp-final-001', 'Existing base proof points elsewhere')
        require(base_proof['raw_csv_sha256'] == report['replays'][FINAL_RUNS[0]]['raw_csv_sha256'], 'Existing base proof hash differs')
        report['inputs'][str(base_proof_path.relative_to(ROOT))] = digest(base_proof_path)
        # The earlier export manifest legitimately acquired a Kaggle observation
        # after its proof. Verify every current artifact, and allow only that
        # exact metadata-field update in the historical proof's manifest hash.
        context_export_dir = ROOT / 'artifacts/submission-lp-context-blend-001'
        context_export = verified_manifest(context_export_dir)
        require(context_export['source_order'] == VALIDATION_RUNS[:2]
                and context_export['weights'] == [.25, .75], 'Earlier context blend recipe changed')
        context_rebuilt = reconstruct_float64(raw_arrays[:2], [.25, .75], units)
        require(bitwise_equal(context_rebuilt, np.load(context_export_dir / 'test_predictions.npy', allow_pickle=False)),
                'Earlier context blend no longer reconstructs')
        context_csv = read_submission(context_export_dir / 'submission.csv', template)
        context_wide = context_rebuilt.reshape(725, 168, 14).transpose(0, 2, 1).reshape(725, -1)
        context_error = float(np.max(np.abs(context_csv.iloc[:, 1:].to_numpy() - context_wide)))
        require(context_error <= 5.1e-10 and digest(context_export_dir / 'submission.csv') == context_export['submission_sha256'],
                'Earlier context export CSV changed')
        report['prior_context_reconstruction'] = {'bitwise_match': True, 'csv_readback_max_abs_error': context_error,
                                                  'submission_sha256': context_export['submission_sha256']}
        context_proof = verified_manifest(ROOT / 'artifacts/verify-lp-context-blend-002',
                                          allow_context_observation_update=True)
        require(context_proof['replay']['bitwise_equal'] is True, 'Existing context replay proof did not pass')
        require(context_proof['replay']['raw_csv_sha256'] == report['replays'][FINAL_RUNS[1]]['raw_csv_sha256'],
                'Existing context proof hash differs')
        comparison_proof_path = ROOT / 'artifacts/verify-lp-joint-comparison-001/manifest.json'
        comparison_proof = json.loads(comparison_proof_path.read_text())
        require(comparison_proof['status'] == 'passed' and comparison_proof['accepted'] is True,
                'Existing joint selection verification did not pass')
        require(comparison_proof['analysis_sha256'] == digest(selection_dir / 'analysis.json')
                and comparison_proof['selection_sha256'] == digest(lock_path)
                and comparison_proof['weights'] == WEIGHTS,
                'Existing joint selection proof references different inputs')
        require(comparison_proof['source_sha256'] == digest(ROOT / 'scripts/verify_lp_joint_comparison.py'),
                'Existing selection verifier code changed')
        report['inputs'][str(comparison_proof_path.relative_to(ROOT))] = digest(comparison_proof_path)
        export_dir = ROOT / 'artifacts' / EXPORT_RUN
        exported = verified_manifest(export_dir)
        require(exported['source_order'] == VALIDATION_RUNS and exported['weights'] == WEIGHTS, 'Exporter changed source order/weights')
        require(exported['selection_sha256'] == digest(selection_dir / 'analysis.json')
                and exported['selection_lock_sha256'] == digest(lock_path), 'Export used another selection/lock')
        require(exported['sources'] == provenances, 'Export source provenance differs from independently validated sources')
        reconstructed = reconstruct_float64(raw_arrays, WEIGHTS, units)
        saved = np.load(export_dir / 'test_predictions.npy', allow_pickle=False)
        require(saved.dtype == np.float64 and bitwise_equal(reconstructed, saved), 'Exported blend differs at value-byte level')
        frame = read_submission(export_dir / 'submission.csv', template)
        wide = reconstructed.reshape(725, 168, 14).transpose(0, 2, 1).reshape(725, -1)
        error = float(np.max(np.abs(frame.iloc[:, 1:].to_numpy() - wide)))
        require(error <= 5.1e-10, 'CSV readback exceeds declared serialization tolerance')
        require(exported['submission_sha256'] == digest(export_dir / 'submission.csv'), 'Export CSV hash differs from its manifest')
        require(exported['validation_micro_auc'] == {year: fold['selected_micro_auc'] for year, fold in selection['folds'].items()},
                'Export validation score claims differ from selection')
        report.update(status='complete', selection_sha256=digest(selection_dir / 'analysis.json'),
                      selection_lock_sha256=digest(lock_path), source_order=VALIDATION_RUNS, weights=WEIGHTS,
                      bitwise_export_array_match=True, output_shape=list(frame.shape), unit_order=units,
                      csv_readback_max_abs_error=error, submission_sha256=exported['submission_sha256'],
                      output_probability_range=[float(saved.min()), float(saved.max())],
                      validation_micro_auc=exported['validation_micro_auc'],
                      postprocessing='float64 promotion before weighting; accumulation in exporter source order; fixed Vinje projection once')
        print(json.dumps({key: report[key] for key in ['status', 'submission_sha256', 'weights', 'bitwise_export_array_match', 'csv_readback_max_abs_error']}), flush=True)
    except BaseException:
        report['status'] = 'failed'
        report['error'] = traceback.format_exc()
        raise
    finally:
        for relative in ['scripts/verify_lp_joint_export.py', 'scripts/export_ensemble.py',
                         'sintef_project/postprocess.py', 'tests/test_verify_lp_joint_export.py']:
            target = out / 'source' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)
        report['runtime_seconds'] = time.time() - started
        report['artifacts'] = {str(path.relative_to(out)): digest(path) for path in out.rglob('*') if path.is_file() and path.name != 'manifest.json'}
        (out / 'manifest.json').write_text(json.dumps(report, indent=2))
        with (ROOT / 'docs/experiments.md').open('a') as handle:
            handle.write(f'\n## {args.run_id} — {report["timestamp_utc"]}\n\nStatus: {report["status"]}. '
                         f'Command: `{report["command"]}`. No fit, inference, LP solve or feature table. '
                         'Verified frozen joint selection and source/cache/data/checkpoint/artifact hashes, '
                         'all three saved raw checkpoint replays and CSV bytes, existing replay/selection proofs, '
                         'float64 blend reconstruction and exact template/readback. '
                         f'Proof: `artifacts/{args.run_id}/manifest.json`. Runtime {report["runtime_seconds"]:.2f}s. No Kaggle submission.\n')


if __name__ == '__main__':
    main()
