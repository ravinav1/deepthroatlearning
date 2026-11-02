"""Export the frozen ensemble from saved test arrays with manifest verification.

This script performs no model fitting or inference. Checkpoint replay remains a
separate independent verification step. A legacy 50/50 research blend requires
an explicit --legacy-reference argument and has no inherited validation score.
"""
from __future__ import annotations
import argparse
import hashlib
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
from datetime import datetime, timezone

for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[variable] = '2'
import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sintef_project.postprocess import project_vinje

EXPECTED_SOURCES = ['lgb-normalized-context-001', 'lgb-normalized-001', 'lgb-physics-001']
EXPECTED_WEIGHTS = [.75, 0., .25]


def digest(path: Path) -> str:
    """Return a file's content SHA256."""
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def verify_hashes(base: Path, expected: dict[str, str]) -> None:
    """Fail on any absent or modified manifest-listed file."""
    for relative, expected_hash in expected.items():
        path = base / relative
        if not path.is_file() or digest(path) != expected_hash:
            raise ValueError(f'Integrity mismatch: {path}')


def read_submission(path: Path, template: pd.DataFrame) -> pd.DataFrame:
    """Validate full schema, exact ID order, and finite probabilities."""
    frame = pd.read_csv(path, float_precision='round_trip')
    if frame.columns.tolist() != template.columns.tolist() or frame.shape != template.shape:
        raise ValueError(f'Submission schema mismatch: {path}')
    if not frame['Run No'].equals(template['Run No']) or frame['Run No'].duplicated().any():
        raise ValueError(f'Submission ID order mismatch: {path}')
    values = frame.iloc[:, 1:].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError(f'Invalid submission probabilities: {path}')
    return frame


def case_major_probabilities(frame: pd.DataFrame, units: list[str]) -> np.ndarray:
    """Convert exact generator-major submission columns to case-hour-unit rows."""
    return frame.iloc[:, 1:].to_numpy().reshape(len(frame), len(units), 168).transpose(0, 2, 1).reshape(-1, len(units))


def validate_final(path: Path, validation: dict, expected_feature_set: str,
                   template: pd.DataFrame, units: list[str]) -> tuple[np.ndarray, dict]:
    """Verify source artifacts, selected recipe, feature sources, and output order."""
    manifest_path = path / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    args = manifest['arguments']
    if manifest.get('status') != 'complete' or not args['final'] or args['years'] or args['inference_from']:
        raise ValueError(f'Expected completed direct full fit with no validation years: {path}')
    if args['feature_set'] != expected_feature_set:
        raise ValueError('Final feature set does not match selected validation branch')
    if manifest['data'] != validation['data']:
        raise ValueError('Final and validation data fingerprints differ')
    verify_hashes(ROOT, manifest['data'])
    verify_hashes(path, manifest['artifacts'])
    verify_hashes(path / 'source', manifest['source'])
    needed_sources = ['sintef_project/features.py', 'sintef_project/physics.py']
    if expected_feature_set.startswith('normalized'):
        needed_sources += ['sintef_project/normalized.py']
    if expected_feature_set in ['normalized-context', 'normalized-lp', 'normalized-lp-context', 'normalized-lp-v2', 'normalized-lp-switching', 'normalized-lp-joint', 'normalized-lp-context-joint']:
        needed_sources += ['sintef_project/context_features.py']
    if expected_feature_set.startswith('normalized-lp'):
        needed_sources += ['sintef_project/lp_features.py']
    extra_modules = {'normalized-lp-context': 'lp_context', 'normalized-lp-v2': 'lp_features_v2',
                     'normalized-lp-switching': 'lp_switching', 'normalized-lp-joint': 'lp_joint_plant',
                     'normalized-lp-context-joint': 'lp_context'}
    if expected_feature_set in extra_modules:
        needed_sources += [f'sintef_project/{extra_modules[expected_feature_set]}.py']
    if expected_feature_set in ['normalized-lp-joint', 'normalized-lp-context-joint']:
        needed_sources += ['sintef_project/lp_joint_plant.py', 'sintef_project/lp_head_update.py', 'sintef_project/lp_shared_loss_update.py']
    for name in needed_sources:
        if manifest['source'][name] != validation['source'][name]:
            raise ValueError(f'Feature code changed between validation and final fit: {name}')
    if manifest['features'] != validation['features'] or manifest['feature_spec'] != validation['feature_spec']:
        raise ValueError('Feature order or formula specification changed')
    knobs = ['trees', 'leaves', 'learning_rate', 'min_child_samples', 'sample_fraction', 'seed', 'feature_set']
    for name in knobs:
        if args[name] != validation['arguments'][name]:
            raise ValueError(f'Training argument changed: {name}')
    # Expand defaults using the recorded, matching installed LightGBM version.
    if manifest['environment']['packages']['lightgbm'] != validation['environment']['packages']['lightgbm']:
        raise ValueError('LightGBM version differs between validation and full fit')
    if importlib.metadata.version('lightgbm') != manifest['environment']['packages']['lightgbm']:
        raise ValueError('Installed LightGBM differs from source environment')
    final_params = lgb.LGBMClassifier(**manifest['model_parameters']).get_params()
    val_params = lgb.LGBMClassifier(**validation['model_parameters']).get_params()
    for name in set(final_params) | set(val_params):
        if name != 'n_jobs' and final_params.get(name) != val_params.get(name):
            raise ValueError(f'Model parameter changed: {name}')
    checkpoints = {}
    for unit in units:
        checkpoint = path / f'final_{unit}.txt'
        if checkpoint.name not in manifest['artifacts']:
            raise ValueError(f'Unregistered checkpoint: {checkpoint}')
        model = lgb.Booster(model_file=str(checkpoint))
        serialized_features = [name.replace(' ', '_') for name in manifest['features']]
        if len(set(serialized_features)) != len(serialized_features):
            raise ValueError('Feature names collide after LightGBM whitespace normalization')
        if model.feature_name() != serialized_features or model.current_iteration() != args['trees']:
            raise ValueError(f'Checkpoint schema/iteration mismatch: {checkpoint}')
        checkpoints[checkpoint.name] = digest(checkpoint)
    p = np.load(path / 'test_predictions.npy', allow_pickle=False).astype(np.float64)
    if p.shape != (len(template) * 168, len(units)) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError(f'Invalid test probability matrix: {path}')
    source_csv = read_submission(path / 'submission.csv', template)
    source_matrix = case_major_probabilities(source_csv, units)
    maximum_csv_error = float(np.max(np.abs(source_matrix - p)))
    if maximum_csv_error > 5.1e-10:
        raise ValueError('Saved test rows/units disagree with registered submission CSV')
    if digest(path / 'submission.csv') != manifest['submission_sha256']:
        raise ValueError('Source submission hash differs from declared hash')
    return p, {
        'path': str(path.relative_to(ROOT)), 'manifest_sha256': digest(manifest_path),
        'test_predictions_sha256': digest(path / 'test_predictions.npy'),
        'checkpoints': checkpoints, 'source': manifest['source'], 'data': manifest['data'],
        'source_csv_max_absolute_roundtrip_error': maximum_csv_error,
        'validation_recipe_match': True, 'artifact_and_source_hashes_verified': len(manifest['artifacts']) + len(manifest['source']),
    }


def main() -> None:
    """Create a new verified export artifact; never overwrite an earlier export."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', default='submission-ensemble-001')
    parser.add_argument('--selection', type=Path, default=Path('artifacts/ensemble-001'))
    parser.add_argument('--context-final', type=Path, default=Path('artifacts/lgb-normalized-context-final-001'))
    parser.add_argument('--physics-final', type=Path, default=Path('artifacts/lgb-physics-final-001'))
    parser.add_argument('--legacy-reference', type=Path, help='Explicit opt-in only: export 50%% supplied legacy CSV +50%% new ensemble; no legacy validation is available')
    args = parser.parse_args()
    out = ROOT / 'artifacts' / args.run_id
    out.mkdir(exist_ok=False)
    started = time.time()
    report = {'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'command': shlex.join([sys.executable, *sys.argv]),
              'environment': {'python': sys.version, 'platform': platform.platform(),
                              'packages': {n: importlib.metadata.version(n) for n in ['numpy', 'pandas', 'lightgbm']}},
              'training': 'none; no model fitting or checkpoint inference',
              'check_scope': 'Manifest/artifact/data hashes, saved checkpoint schema, saved test arrays versus source CSV, frozen recipe, export readback. Independent checkpoint replay is separate and not claimed here.',
              'sources': {}, 'kaggle': {'submission_id': None, 'score': None, 'rank': None, 'observation_time': None}}
    try:
        selection_dir = ROOT / args.selection
        selection_path = selection_dir / 'analysis.json'
        selection = json.loads(selection_path.read_text())
        if selection.get('status') != 'complete' or selection['source_order'] != EXPECTED_SOURCES or selection['selected_weights'] != EXPECTED_WEIGHTS:
            raise ValueError('Unexpected ensemble selection recipe')
        verify_hashes(selection_dir, selection['artifacts'])
        verify_hashes(ROOT, selection['sources'])
        verify_hashes(selection_dir / 'source', selection['source_code'])
        locked_path = selection_dir / 'selection_2021.json'
        locked = json.loads(locked_path.read_text())
        if locked['selected_weights'] != EXPECTED_WEIGHTS or locked['selected_candidate'] != selection['selected_candidate'] or '2022' in locked['folds']:
            raise ValueError('2021 selection lock is inconsistent')
        if digest(ROOT / 'sintef_project/postprocess.py') != selection['source_code']['sintef_project/postprocess.py']:
            raise ValueError('Projection implementation changed since selection')
        report['selection'] = {'path': str(selection_path.relative_to(ROOT)), 'sha256': digest(selection_path),
                               'lock_sha256': digest(locked_path), 'weights': EXPECTED_WEIGHTS,
                               'source_order': EXPECTED_SOURCES, 'candidate': selection['selected_candidate'],
                               'validation_micro_auc': {y: f['selected_micro_auc'] for y, f in selection['folds'].items()}}
        template = pd.read_csv(ROOT / 'sintef_project/sample_submission.csv')
        mapping = pd.read_csv(ROOT / 'sintef_project/prediction_mapping.csv')
        train_ids = pd.read_csv(ROOT / 'sintef_project/data/kernel/Unit_commitment_decisions.csv', usecols=['Run No'])
        units = selection['folds']['2021']['units']
        expected_columns = ['Run No'] + [f'result_committed_{u}_t{h}' for u in units for h in range(168)]
        if template.columns.tolist() != expected_columns or len(units) != 14 or len(template) != 725:
            raise ValueError('Unexpected complete sample submission schema')
        if not mapping['Run No'].equals(template['Run No']) or len(mapping) != 725 or mapping['Run No'].duplicated().any():
            raise ValueError('Mapping order differs from sample submission')
        if set(train_ids['Run No']) & set(mapping['Run No']):
            raise ValueError('Training and test IDs overlap')
        for relative in ['scripts/export_ensemble.py', 'sintef_project/postprocess.py']:
            destination = out / 'source' / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, destination)
        context_val = json.loads((ROOT / 'artifacts/lgb-normalized-context-001/manifest.json').read_text())
        physics_val = json.loads((ROOT / 'artifacts/lgb-physics-001/manifest.json').read_text())
        context, report['sources']['context'] = validate_final(ROOT / args.context_final, context_val, 'normalized-context', template, units)
        physics, report['sources']['physics'] = validate_final(ROOT / args.physics_final, physics_val, 'physics', template, units)
        pred = project_vinje(.75 * context + .25 * physics, units)
        report['postprocessing'] = 'Vinje G1>=G2>=G3 L2 projection after 75/25 probability blend; no smoothing'
        if args.legacy_reference:
            legacy_path = ROOT / args.legacy_reference
            legacy = case_major_probabilities(read_submission(legacy_path, template), units)
            pred = .5 * legacy + .5 * pred
            report['legacy_research_blend'] = {'source': str(legacy_path), 'sha256': digest(legacy_path), 'weights': {'legacy': .5, 'new_projected_ensemble': .5},
                                              'validation': None, 'limitation': 'Legacy validation predictions unavailable; the 75/25 validation scores do not apply to this 50/50 research blend.'}
        else:
            report['legacy_research_blend'] = {'created': False, 'proposal': 'Optional explicit future experiment:50% legacy reference +50% new projected ensemble. No artifact generated or submitted.', 'legacy_validation_available': False}
        wide = pred.reshape(len(template), 168, len(units)).transpose(0, 2, 1).reshape(len(template), -1)
        submission = pd.DataFrame(wide, columns=template.columns[1:])
        submission.insert(0, 'Run No', template['Run No'].to_numpy())
        np.save(out / 'test_predictions.npy', pred)
        submission.to_csv(out / 'submission.csv', index=False, float_format='%.9g')
        loaded = read_submission(out / 'submission.csv', template)
        roundtrip_error = float(np.max(np.abs(loaded.iloc[:, 1:].to_numpy() - wide)))
        if roundtrip_error > 5.1e-10:
            raise ValueError(f'CSV readback changed probabilities excessively: {roundtrip_error}')
        report['submission_sha256'] = digest(out / 'submission.csv')
        report['schema'] = {'rows': len(template), 'columns': len(template.columns), 'label_columns': 2352,
                            'case_ids': template['Run No'].tolist(), 'units': units,
                            'min_probability': float(pred.min()), 'max_probability': float(pred.max()),
                            'csv_max_absolute_roundtrip_error': roundtrip_error}
        recipe = f'''# Frozen ensemble export\n\nSource recipe:75% `lgb-normalized-context-final-001` +25% `lgb-physics-final-001`, followed by Vinje probability projection. No smoothing. Weights were chosen on2021 and locked before2022 evaluation in `artifacts/ensemble-001/selection_2021.json`.\n\nExact export command:\n\n```sh\n{report['command']}\n```\n\nUse a new `--run-id` to independently regenerate without overwriting this artifact. This command reads saved raw test arrays and verifies source data, archived code, checkpoints, model settings, source submission order and hashes. It does not run the checkpoints; independent raw-data-to-checkpoint replay remains a separate verification.\n\nOutput:`submission.csv`,725rows ×2353columns, exact sample schema. SHA256:`{report['submission_sha256']}`. Readback max error:{roundtrip_error}.\n\nFull data-to-model commands, sampling rows, all hyperparameters, source snapshots and checkpoints are in the two source `manifest.json` files. Their hashes and every model checksum are recorded in this export's `manifest.json`.\n\nValidation belongs to the frozen75/25 recipe:{report['selection']['validation_micro_auc']}. These reused development folds remain below the0.998migration gate. No new leaderboard score is claimed.\n\nOptional future research proposal:50% separately verified legacy reference +50% this projected ensemble. No legacy validation predictions are available, so that blend cannot inherit the75/25 validation scores. It is not created unless `--legacy-reference PATH` is explicitly supplied in a separately authorized run.\n'''
        if args.legacy_reference:
            recipe += '\nThis particular run explicitly enabled the legacy research blend. Its validation score is unknown; the75/25 scores above describe only its new-model component.\n'
        (out / 'README.md').write_text(recipe)
        report['status'] = 'complete'
        print(json.dumps({'status': 'complete', 'submission': str(out / 'submission.csv'), 'sha256': report['submission_sha256'], 'max_csv_error': roundtrip_error}), flush=True)
    except BaseException:
        report['status'] = 'failed'
        report['error'] = traceback.format_exc()
        raise
    finally:
        report['runtime_seconds'] = time.time() - started
        report['artifacts'] = {str(p.relative_to(out)): digest(p) for p in out.rglob('*') if p.is_file() and p.name != 'manifest.json'}
        (out / 'manifest.json').write_text(json.dumps(report, indent=2))
        with (ROOT / 'docs/experiments.md').open('a') as f:
            f.write(f'\n## {args.run_id} — {report["timestamp_utc"]}\n\nStatus:{report["status"]}. Exact command:`{report["command"]}`. Runtime:{report["runtime_seconds"]:.1f}s.\n\n')
            f.write('Export-only hypothesis:reproduce frozen ensemble selection on full-fit saved test predictions. No fitting/inference/seeds/early stopping. Source training manifests carry data paths/checksums, environment, seeds, sampling and full model settings. Frozen weights75%normalized-context+25%physics; Vinje projection; no smoothing. Full source/code/data/checkpoint integrity and CSV readback checks. Independent checkpoint replay is separate.\n\n')
            if 'selection' in report:
                f.write(f'Reused validation for75/25 recipe:{report["selection"]["validation_micro_auc"]}. Legacy research mode:{bool(args.legacy_reference)}; legacy blend validation unavailable.\n\n')
            if 'error' in report:
                f.write(f'Failure:`{report["error"].splitlines()[-1]}`\n\n')
            f.write(f'Complete provenance:`artifacts/{args.run_id}/manifest.json`. Kaggle submission ID/score/rank/observation:none.\n')


if __name__ == '__main__':
    main()
