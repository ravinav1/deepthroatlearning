"""Replay the selected validation specialists, then fit their full-data recipe."""
from __future__ import annotations
import argparse
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

for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[name] = '2'
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.xgb_lp_specialists import (
    PARAMS, SPECIALISTS, FEATURE_MODULES, digest, verify,
    build_features, add_physics_features, add_context_features,
    add_lp_features, add_normalized_features,
)


def main() -> None:
    """Require exact 2021 replay before training three final CPU specialists."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', default='xgb-lp-specialists-final-001')
    parser.add_argument('--selection', default='xgb-lp-specialists-001')
    parser.add_argument('--baseline-final', default='lgb-normalized-lp-final-001')
    args = parser.parse_args()
    out = ROOT / 'artifacts' / args.run_id
    out.mkdir(exist_ok=False)
    started = time.time()
    report = {'run_id': args.run_id, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'command': shlex.join([sys.executable, *sys.argv]), 'model_parameters': PARAMS,
              'early_stopping': None, 'seed': 4173, 'units': SPECIALISTS,
              'environment': {'python': sys.version, 'platform': platform.platform(),
                              'cpu': platform.processor(), 'cpu_count': os.cpu_count(), 'threads': 2,
                              'packages': {n: importlib.metadata.version(n) for n in ['numpy', 'pandas', 'scikit-learn', 'xgboost', 'lightgbm', 'scipy', 'pyyaml']}},
              'source': {}, 'data': {}, 'inputs': {}, 'models': {},
              'kaggle': {'submission_id': None, 'score': None, 'rank': None, 'observation_time': None}}
    try:
        selection_dir = ROOT / 'artifacts' / args.selection
        selection_path = selection_dir / 'manifest.json'
        selection = json.loads(selection_path.read_text())
        baseline_dir = ROOT / 'artifacts' / args.baseline_final
        baseline_path = baseline_dir / 'manifest.json'
        baseline = json.loads(baseline_path.read_text())
        if selection.get('status') != 'complete' or baseline.get('status') != 'complete':
            raise ValueError('Validation specialists and baseline full fit must be complete')
        if selection['specialists'] != SPECIALISTS or selection['model_parameters'] != PARAMS or selection['selected_weight'] != .5:
            raise ValueError('Unexpected frozen specialist recipe')
        if baseline['arguments']['feature_set'] != 'normalized-lp' or not baseline['arguments']['final'] or baseline['arguments']['years']:
            raise ValueError('Expected completed full-data normalized-LP v1 baseline')
        for path, manifest in [(selection_dir, selection), (baseline_dir, baseline)]:
            verify(ROOT, manifest['data'])
            verify(path, manifest['artifacts'])
            verify(path / 'source', manifest['source'])
        verify(ROOT, selection['inputs'])
        if baseline['data'] != selection['data'] or baseline['features'] != selection['features'] or baseline['feature_spec'] != selection['feature_spec']:
            raise ValueError('Full-fit and selected validation data/features differ')
        for name in FEATURE_MODULES:
            if digest(ROOT / name) != selection['source'][name]:
                raise ValueError(f'Feature module changed since validation: {name}')
        for name in ['numpy', 'pandas', 'xgboost']:
            if report['environment']['packages'][name] != selection['environment']['packages'][name]:
                raise ValueError(f'Runtime package changed: {name}')
        report['data'] = baseline['data']
        report['inputs'] = {str(selection_path.relative_to(ROOT)): digest(selection_path),
                            str(baseline_path.relative_to(ROOT)): digest(baseline_path)}
        report['selection'] = {'run': args.selection, 'joint_replacement_weight': .5,
                               'scope': 'Replace only Haukeli,Lio,Hogga jointly in raw normalized-LP baseline; all other units unchanged',
                               'validation_micro_auc': {y: f['selected_micro_auc'] for y, f in selection['folds'].items()},
                               'caveat': 'Inherited reused-fold validation of the recipe, not a score on this full-fit training set or competition labels'}
        for name in [*FEATURE_MODULES, 'scripts/xgb_lp_specialists.py', 'scripts/xgb_lp_specialists_final.py']:
            report['source'][name] = digest(ROOT / name)
            destination = out / 'source' / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, destination)
        x, labels, train, mapping, metadata = build_features(ROOT)
        x, physics = add_physics_features(x, ROOT)
        x, context = add_context_features(x, ROOT)
        x, lp = add_lp_features(x, ROOT)
        x, normalized = add_normalized_features(x, ROOT)
        metadata['feature_spec'].update(physics=physics, context=context, lp=lp, normalized=normalized)
        if x.columns.tolist() != baseline['features'] or metadata['feature_spec'] != baseline['feature_spec']:
            raise ValueError('Reconstructed feature pipeline does not match frozen recipe')
        report['features'] = x.columns.tolist()
        report['feature_spec'] = metadata['feature_spec']
        report['shape'] = list(x.shape)
        units = metadata['units']
        indices = [units.index(unit) for unit in SPECIALISTS]
        dates = pd.to_datetime(train.starttime, utc=True)
        val_cases = np.flatnonzero((dates >= pd.Timestamp('2021-01-01', tz='UTC')) & (dates < pd.Timestamp('2022-01-01', tz='UTC')))
        va = (val_cases[:, None] * 168 + np.arange(168)).ravel()
        validation_path = selection_dir / 'validation_2021.npz'
        with np.load(validation_path, allow_pickle=False) as z:
            expected_raw = z['specialist_predictions']
            expected_blend = z['predictions']
            if z['specialist_units'].tolist() != SPECIALISTS or z['units'].tolist() != units or not np.array_equal(z['case_ids'], train.iloc[val_cases]['Run No']):
                raise ValueError('Validation replay row/unit order mismatch')
        replayed = np.empty(expected_raw.shape, dtype=np.float32)
        replay_checks = {}
        for j, unit in enumerate(SPECIALISTS):
            checkpoint = selection_dir / f'fold2021_{unit}.ubj'
            model = XGBClassifier(n_jobs=2)
            model.load_model(checkpoint)
            model.get_booster().set_param({'nthread': 2})
            if model.get_booster().feature_names != x.columns.tolist():
                raise ValueError('XGBoost checkpoint feature names differ')
            replayed[:, j] = model.predict_proba(x.iloc[va])[:, 1]
            replay_checks[unit] = {'checkpoint_sha256': digest(checkpoint),
                                   'exact_equal': bool(np.array_equal(replayed[:, j], expected_raw[:, j])),
                                   'max_absolute_difference': float(np.max(np.abs(replayed[:, j] - expected_raw[:, j])))}
            if not replay_checks[unit]['exact_equal']:
                raise ValueError(f'Validation checkpoint replay mismatch: {unit}')
        base_validation = ROOT / 'artifacts' / selection['baseline_run'] / 'validation_2021.npz'
        with np.load(base_validation, allow_pickle=False) as z:
            replay_blend = z['predictions'].astype(np.float64)
        replay_blend[:, indices] = .5 * replay_blend[:, indices] + .5 * replayed
        if not np.array_equal(replay_blend, expected_blend):
            raise ValueError('Whole-matrix selected validation blend replay mismatch')
        report['reliability'] = {'validation_year': 2021, 'unit_checkpoints': replay_checks,
                                  'selected_full_matrix_exact_equal': True,
                                  'validation_cases': len(val_cases), 'all_14_units': True,
                                  'validation_artifact_sha256': digest(validation_path)}
        print('REPLAY2021 exact:three saved XGBoost models and selected14-unit blend', flush=True)
        (out / 'replay_2021.json').write_text(json.dumps(report['reliability'], indent=2))
        sample_path = baseline_dir / 'train_sample_final.npy'
        sample = np.load(sample_path, allow_pickle=False)
        ntrain = len(train) * 168
        if sample.ndim != 1 or not np.issubdtype(sample.dtype, np.integer) or len(np.unique(sample)) != len(sample) or sample.min() < 0 or sample.max() >= ntrain:
            raise ValueError('Invalid final baseline training sample')
        if set(train['Run No']) & set(mapping['Run No']):
            raise ValueError('Training/test case IDs overlap')
        np.save(out / 'train_sample_final.npy', sample)
        report['inputs'][str(sample_path.relative_to(ROOT))] = digest(sample_path)
        report['training'] = {'case_ids': train['Run No'].tolist(), 'date_range': [str(dates.min()), str(dates.max())],
                              'rows': len(sample), 'available_case_hour_rows': ntrain,
                              'sample_rule': 'Exact stored train_sample_final.npy from completed normalized-LP baseline; no resampling',
                              'baseline_sample_fraction': baseline['arguments']['sample_fraction']}
        report['test'] = {'case_ids': mapping['Run No'].tolist(), 'units': SPECIALISTS,
                         'row_order': 'Prediction mapping case order, then hour0..167; columns Haukeli_G1,Lio_G1,Hogga_G1'}
        xt, xv = x.iloc[sample], x.iloc[ntrain:]
        pred = np.empty((len(xv), 3), dtype=np.float32)
        for j, unit in enumerate(SPECIALISTS):
            fit_start = time.time()
            model = XGBClassifier(**PARAMS)
            model.fit(xt, labels[unit][sample])
            pred[:, j] = model.predict_proba(xv)[:, 1]
            checkpoint = out / f'final_{unit}.ubj'
            model.save_model(checkpoint)
            (out / f'config_{unit}.json').write_text(model.get_booster().save_config())
            pd.Series(model.feature_importances_, index=x.columns).sort_values(ascending=False).to_csv(out / f'importance_{unit}.csv')
            report['models'][unit] = {'checkpoint': checkpoint.name, 'sha256': digest(checkpoint),
                                      'training_seconds': time.time() - fit_start, 'trees': PARAMS['n_estimators']}
            print('FINAL', unit, report['models'][unit]['training_seconds'], flush=True)
        if not np.isfinite(pred).all() or ((pred < 0) | (pred > 1)).any():
            raise ValueError('Invalid final specialist probabilities')
        np.save(out / 'test_predictions.npy', pred)
        np.savez_compressed(out / 'test_predictions.npz', predictions=pred, case_ids=mapping['Run No'].to_numpy(), units=SPECIALISTS)
        report['status'] = 'complete'
        with (ROOT / 'docs/xgb_lp_specialists.md').open('a') as f:
            f.write(f'\n## Replay and full fit:{args.run_id}\n\nAll three2021UBJ checkpoints reproduced saved probabilities exactly(maxdifference0); recomposing the locked50%three-unit replacement reproduced the saved14-unit validationmatrix exactly. Verification:`artifacts/{args.run_id}/replay_2021.json`.\n\n')
            f.write(f'Full fit uses all2922labelledcases and exactly{len(sample)}sampledcase-hourrows from `{args.baseline_final}/train_sample_final.npy`, unchanged600treeparams/seed4173/twothreads. Three finalUBJ files, raw725×168×3testprobabilities, case IDs and unitorder are in `artifacts/{args.run_id}/`. These are componentoutputs, not a completeKaggleCSV; no submission was made.\n')
        print('COMPLETE', args.run_id, pred.shape, flush=True)
    except BaseException:
        report['status'] = 'failed'
        report['error'] = traceback.format_exc()
        raise
    finally:
        report['runtime_seconds'] = time.time() - started
        report['artifacts'] = {str(p.relative_to(out)): digest(p) for p in out.rglob('*') if p.is_file() and p.name != 'manifest.json'}
        (out / 'manifest.json').write_text(json.dumps(report, indent=2))
        with (ROOT / 'docs/experiments.md').open('a') as f:
            f.write(f'\n## {args.run_id} — {report["timestamp_utc"]}\n\nStatus:{report["status"]}. Command:`{report["command"]}`. Runtime:{report["runtime_seconds"]:.1f}s.\n\n')
            f.write(f'Hypothesis:reproduce validated complementary3-unit XGBoost recipe for fullfit. Exact2021checkpoint/fullmatrix replay required before fitting. Parameters:{PARAMS}; noearlystopping. Samefull-fit storedsamplerows as{args.baseline_final}; all2922trainingcases eligible, no testlabels.\n\n')
            f.write(f'Inherited selectedrecipe validation:{report.get("selection",{}).get("validation_micro_auc")}; this is not a fullfit orKagglescore. Source/data/code/checkpoint/sample/predictionhashes, exactIDs, featureformulas and environment:`artifacts/{args.run_id}/manifest.json`.\n\n')
            if 'error' in report:
                f.write(f'Failure:`{report["error"].splitlines()[-1]}`\n\n')
            f.write('Kaggle submission ID/score/rank/observation:none. Component predictions only; no finalblendCSV created.\n')


if __name__ == '__main__':
    main()
