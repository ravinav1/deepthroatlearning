"""Fixed CPU multilabel MLP ablation using exact LP-v1 chronological samples."""
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
import subprocess
import sys
import time
import traceback
import warnings

for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[variable] = '2'
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sintef_project.features import build_features
from sintef_project.physics import add_physics_features
from sintef_project.context_features import add_context_features
from sintef_project.lp_features import add_lp_features
from sintef_project.normalized import add_normalized_features
from sintef_project.mlp_helpers import build_without_later_targets, fit_standardization, standardize, portable_predict
from sintef_project.postprocess import project_vinje

FEATURE_MODULES = ['sintef_project/' + name + '.py' for name in
                   ['features', 'physics', 'context_features', 'lp_features', 'normalized']]
WEIGHTS = [0., .1, .25, .5]
BASE_RUN = 'lgb-normalized-lp-001'
ANCHOR_RUN = 'lp-joint-comparison-001'
PARAMETERS = dict(hidden_layer_sizes=(128, 64), activation='relu', solver='adam',
                  alpha=.0001, batch_size=512, learning_rate_init=.001,
                  max_iter=24, shuffle=True, random_state=4173, tol=0.,
                  early_stopping=False, n_iter_no_change=1000)


def digest(path: Path) -> str:
    """Return a streaming SHA256 content digest."""
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            hasher.update(block)
    return hasher.hexdigest()


def verify(base: Path, hashes: dict[str, str]) -> None:
    """Check registered bytes without decoding any validation labels."""
    for relative, expected in hashes.items():
        if digest(base / relative) != expected:
            raise ValueError(f'Changed registered artifact: {base / relative}')


def create_json(path: Path, data: dict) -> None:
    """Persist a predeclared protocol or immutable selection once."""
    with path.open('x') as stream:
        json.dump(data, stream, indent=2)


def mix(anchor: np.ndarray, neural: np.ndarray, units: list[str], weight: float) -> np.ndarray:
    """Blend the already projected accepted anchor with raw neural probabilities."""
    return project_vinje((1. - weight) * anchor.astype(np.float64)
                         + weight * neural.astype(np.float64), units)


def metrics(labels: np.ndarray, predictions: np.ndarray, units: list[str]) -> dict:
    """Report the competition micro objective and descriptive unit AUCs."""
    return {'micro_auc': float(roc_auc_score(labels.ravel(), predictions.ravel())),
            'unit_auc': {unit: float(roc_auc_score(labels[:, j], predictions[:, j]))
                         for j, unit in enumerate(units)}}


def save_checkpoint(out: Path, year: int, model: MLPClassifier, inputs: np.ndarray,
                    expected: np.ndarray, features: list[str], units: list[str]) -> dict:
    """Save own trusted pickle and portable arrays, then replay both exactly."""
    path = out / f'fold{year}_mlp.pkl'
    with path.open('xb') as stream:
        pickle.dump(model, stream, protocol=5)
    pickle_hash = digest(path)
    # This is the just-written local trusted checkpoint, never an external pickle.
    if digest(path) != pickle_hash:
        raise ValueError('Checkpoint changed before replay')
    with path.open('rb') as stream:
        restored = pickle.load(stream)
    np.testing.assert_array_equal(restored.predict_proba(inputs), expected)
    portable_path = out / f'fold{year}_portable.npz'
    arrays = {f'coefficient_{i}': value for i, value in enumerate(model.coefs_)}
    arrays.update({f'intercept_{i}': value for i, value in enumerate(model.intercepts_)})
    arrays.update(features=np.asarray(features, dtype=str), units=np.asarray(units, dtype=str), classes=model.classes_,
                  hidden_activation=np.asarray('relu'), output_activation=np.asarray('logistic'))
    with portable_path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    with np.load(portable_path, allow_pickle=False) as saved:
        coefficients = [saved[f'coefficient_{i}'] for i in range(len(model.coefs_))]
        intercepts = [saved[f'intercept_{i}'] for i in range(len(model.intercepts_))]
        replay = portable_predict(inputs, coefficients, intercepts)
    np.testing.assert_array_equal(replay, expected)
    return {'pickle': path.name, 'pickle_sha256': pickle_hash,
            'portable': portable_path.name, 'portable_sha256': digest(portable_path),
            'pickle_replay_max_error': 0., 'portable_replay_max_error': 0.,
            'replayed_rows': len(inputs), 'output_activation': model.out_activation_,
            'parameters': sum(value.size for value in model.coefs_ + model.intercepts_),
            'parameter_dtypes': sorted({str(value.dtype) for value in model.coefs_ + model.intercepts_})}


def main() -> None:
    """Fit two fixed CPU networks, selecting one global weight on2021 only."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', default='cpu-mlp-lp-001')
    args = parser.parse_args()
    out = ROOT / 'artifacts' / args.run_id
    out.mkdir(exist_ok=False)
    started = time.time()
    report = {'run_id': args.run_id, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'command': shlex.join([sys.executable, *sys.argv]), 'status': 'running',
              'hypothesis': 'A shared multilabel neural network may provide complementary unit-commitment probabilities to the accepted joint/context/base tree blend.',
              'base_run': BASE_RUN, 'anchor_run': ANCHOR_RUN, 'weight_grid': WEIGHTS,
              'model_parameters': MLPClassifier(**PARAMETERS).get_params(), 'seed': 4173,
              'training': '24fixed epochs, same stored40%sampled LP-base training hours, no early stopping or validation-based epoch selection, two CPU/BLAS threads; no GPU',
              'normalization': 'Float64 mean and population std(ddof0) fitted only to sampled training rows. Constant columns use scale1. Standardize in float64, clip[-10,10], emitfloat32. Save per-fold statistics once.',
              'selection_rule': 'NN weights[0,.1,.25,.5], maximizing2021 projected wholematrix microAUC; exactties lower weight. Exclusive selection lock before2022 labels/NPZ. Only locked weight and zero evaluated2022. Accept only if selected nonzero improves both years versus fixed accepted jointblend anchor.',
              'mix_formula': 'project_vinje((1-w)*accepted_anchor_projected + w*neural_raw), float64 weighting; no temporal smoothing',
              'label_access': 'Feature builder gets real target rows only before2022; later zero sentinels are discarded.2022 evaluation targets decoded only after immutable2021 lock.',
              'reuse_caveat': 'Both years and anchor choices are reused development evidence, not independent untouched validation.',
              'environment': {'platform': platform.platform(), 'python': sys.version,
                              'cpu_count': os.cpu_count(), 'threads': 2,
                              'hardware': subprocess.run(['system_profiler', 'SPHardwareDataType'], capture_output=True, text=True).stdout.split('Serial Number')[0],
                              'packages': {name: importlib.metadata.version(name) for name in
                                           ['numpy', 'pandas', 'scipy', 'scikit-learn', 'threadpoolctl', 'lightgbm', 'pyyaml']}},
              'source': {}, 'data': {}, 'inputs': {}, 'folds': {},
              'kaggle': {'submission_id': None, 'score': None, 'rank': None, 'observation_time': None}}

    def persist() -> None:
        (out / 'manifest.json').write_text(json.dumps(report, indent=2))

    try:
        base = ROOT / 'artifacts' / BASE_RUN
        baseline = json.loads((base / 'manifest.json').read_text())
        anchor_dir = ROOT / 'artifacts' / ANCHOR_RUN
        anchor = json.loads((anchor_dir / 'analysis.json').read_text())
        if baseline.get('status') != 'complete' or baseline['arguments']['feature_set'] != 'normalized-lp':
            raise ValueError('Expected completed normalizedLP-v1 baseline')
        if (anchor.get('status') != 'complete' or anchor.get('accepted') is not True
                or anchor['selected_weights'] != [.1875, .5625, .25]):
            raise ValueError('Expected frozen accepted joint25% anchor')
        verify(ROOT, baseline['data'])
        verify(base, baseline['artifacts'])
        verify(base / 'source', baseline['source'])
        verify(anchor_dir, anchor['artifacts'])
        verify(ROOT, anchor['sources'])
        verify(anchor_dir / 'source', anchor['source_code'])
        if anchor['data'] != baseline['data']:
            raise ValueError('Anchor and baseline raw data differ')
        report['data'] = baseline['data']
        for name in FEATURE_MODULES:
            if digest(ROOT / name) != baseline['source'][name]:
                raise ValueError(f'Frozen baseline feature code changed: {name}')
        for name in [*FEATURE_MODULES, 'sintef_project/mlp_helpers.py', 'sintef_project/postprocess.py',
                     'scripts/cpu_mlp.py', 'tests/test_cpu_mlp.py', 'docs/cpu_mlp.md']:
            report['source'][name] = digest(ROOT / name)
            target = out / 'source' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        for path in [base / 'manifest.json', anchor_dir / 'analysis.json', anchor_dir / 'selection_2021.json']:
            report['inputs'][str(path.relative_to(ROOT))] = digest(path)
        shutil.copy2(base / 'manifest.json', out / 'baseline_manifest.json')
        shutil.copy2(anchor_dir / 'analysis.json', out / 'anchor_analysis.json')
        create_json(out / 'protocol.json', report)
        report['protocol_sha256'] = digest(out / 'protocol.json')
        persist()
        x, labels, train, _, meta = build_without_later_targets(ROOT, build_features)
        x, physics = add_physics_features(x, ROOT)
        x, context = add_context_features(x, ROOT)
        x, lp = add_lp_features(x, ROOT)
        x, normalized = add_normalized_features(x, ROOT)
        meta['feature_spec'].update(physics=physics, context=context, lp=lp, normalized=normalized)
        if x.columns.tolist() != baseline['features'] or meta['feature_spec'] != baseline['feature_spec'] or x.shape[1] != 395:
            raise ValueError('Baseline feature formulas/order/cache provenance changed')
        report.update(features=x.columns.tolist(), feature_spec=meta['feature_spec'], shape=list(x.shape))
        units = meta['units']
        dates = pd.to_datetime(train.starttime, utc=True)
        retained = int((dates < pd.Timestamp('2022-01-01', tz='UTC')).sum()) * 168
        training_y = np.column_stack([labels[unit][:retained] for unit in units]).astype(np.uint8)
        del labels
        selected = None
        print('FEATURES', x.shape, flush=True)
        for year in [2021, 2022]:
            if year == 2022:
                if digest(out / 'selection_2021.json') != report['selection_sha256']:
                    raise ValueError('Selection lock missing or changed before2022')
            begin, end = pd.Timestamp(f'{year}-01-01', tz='UTC'), pd.Timestamp(f'{year+1}-01-01', tz='UTC')
            train_cases = np.flatnonzero(dates < begin - pd.Timedelta(days=6))
            val_cases = np.flatnonzero((dates >= begin) & (dates < end))
            if not dates.iloc[train_cases].max() + pd.Timedelta(hours=167) < begin:
                raise ValueError('Training targets overlap validation')
            inherited = baseline['folds'][str(year)]
            if (train.iloc[train_cases]['Run No'].tolist() != inherited['train_ids']
                    or train.iloc[val_cases]['Run No'].tolist() != inherited['validation_ids']):
                raise ValueError('Chronological case IDs differ from baseline')
            sample_path = base / f'train_sample_{year}.npy'
            sample = np.load(sample_path, allow_pickle=False)
            if len(np.unique(sample)) != len(sample) or not np.isin(sample // 168, train_cases).all():
                raise ValueError('Invalid inherited sampled training rows')
            if sample.max() >= len(training_y):
                raise ValueError('Attempt to fit a withheld target sentinel')
            np.save(out / sample_path.name, sample)
            report['inputs'][str(sample_path.relative_to(ROOT))] = digest(sample_path)
            va = (val_cases[:, None] * 168 + np.arange(168)).ravel()
            validation_path = base / f'validation_{year}.npz'
            anchor_path = anchor_dir / f'selected_validation_{year}.npz'
            for path in [validation_path, anchor_path]:
                report['inputs'][str(path.relative_to(ROOT))] = digest(path)
            with np.load(validation_path, allow_pickle=False) as saved:
                y, ids = saved['labels'], saved['case_ids']
                if saved['units'].tolist() != units or not np.array_equal(ids, train.iloc[val_cases]['Run No']):
                    raise ValueError('Baseline validation unit/ID alignment failure')
            with np.load(anchor_path, allow_pickle=False) as saved:
                if not (np.array_equal(y, saved['labels']) and np.array_equal(ids, saved['case_ids'])
                        and units == saved['units'].tolist()):
                    raise ValueError('Anchor validation alignment failure')
                anchor_predictions = saved['predictions'].astype(np.float64)
            if year == 2021 and not np.array_equal(training_y[va], y):
                raise ValueError('Decoded pre2022 targets differ from baseline')
            x_train_raw = x.iloc[sample].to_numpy()
            mean, deviation, scale = fit_standardization(x_train_raw)
            stats_path = out / f'normalization_{year}.npz'
            with stats_path.open('xb') as stream:
                np.savez_compressed(stream, mean=mean, standard_deviation=deviation, scale=scale,
                                    features=np.asarray(x.columns.tolist(), dtype=str), sampled_rows=np.asarray(len(sample)))
            x_train = standardize(x_train_raw, mean, scale)
            del x_train_raw
            x_val = standardize(x.iloc[va].to_numpy(), mean, scale)
            if not np.isin(training_y[sample], [0, 1]).all():
                raise ValueError('Nonbinary training targets')
            model = MLPClassifier(**PARAMETERS)
            fit_started = time.time()
            with threadpool_limits(2), warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter('always')
                model.fit(x_train, training_y[sample])
                fit_seconds = time.time() - fit_started
                raw = model.predict_proba(x_val)
                if (model.n_iter_ != 24 or len(model.loss_curve_) != 24 or model.out_activation_ != 'logistic'
                        or raw.shape != y.shape or raw.dtype != np.float32 or not np.isfinite(raw).all()
                        or ((raw < 0) | (raw > 1)).any()):
                    raise ValueError('Unexpected fixed-epoch multilabel fit or output')
                checkpoint = save_checkpoint(out, year, model, x_val, raw, x.columns.tolist(), units)
                pools = threadpool_info()
            fold = {key: inherited[key] for key in ['train_ids', 'validation_ids', 'train_dates',
                                                     'validation_dates', 'purge_days', 'sampled_training_rows']}
            fold.update(units=units, normalization_sha256=digest(stats_path),
                        zero_variance_columns=int((deviation == 0.).sum()),
                        fit_seconds=fit_seconds, loss_curve=[float(value) for value in model.loss_curve_],
                        n_iter=int(model.n_iter_), checkpoint=checkpoint, threadpools=pools,
                        warnings=[str(item.message) for item in captured],
                        raw_neural=metrics(y, raw, units), anchor=metrics(y, anchor_predictions, units), scores={})
            report['folds'][str(year)] = fold
            for weight in (WEIGHTS if year == 2021 else sorted(set([0., selected]))):
                fold['scores'][str(weight)] = metrics(y, mix(anchor_predictions, raw, units, weight), units)['micro_auc']
            if year == 2021:
                selected = max(WEIGHTS, key=lambda weight: fold['scores'][str(weight)])
                report['selected_weight'] = selected
                report['selection_locked_utc'] = datetime.now(timezone.utc).isoformat()
                create_json(out / 'selection_2021.json', report)
                report['selection_sha256'] = digest(out / 'selection_2021.json')
                print('LOCKED2021', selected, fold['scores'], flush=True)
            chosen = mix(anchor_predictions, raw, units, selected)
            fold['selected'] = metrics(y, chosen, units)
            fold['delta_auc'] = fold['selected']['micro_auc'] - fold['anchor']['micro_auc']
            np.savez_compressed(out / f'validation_{year}.npz', predictions=chosen, labels=y,
                                case_ids=ids, units=units, neural_predictions=raw)
            persist()
            print('FOLD', year, 'raw_neural', fold['raw_neural']['micro_auc'],
                  'anchor', fold['anchor']['micro_auc'], 'selected', fold['selected']['micro_auc'], flush=True)
            del model, x_train, x_val
        report['accepted'] = selected > 0. and all(fold['delta_auc'] > 0. for fold in report['folds'].values())
        report['meets_transfer_gate'] = all(fold['selected']['micro_auc'] >= .998 for fold in report['folds'].values())
        report['status'] = 'complete'
    except BaseException:
        report['status'] = 'failed'
        report['error'] = traceback.format_exc()
        raise
    finally:
        report['runtime_seconds'] = time.time() - started
        report['artifacts'] = {str(path.relative_to(out)): digest(path) for path in out.rglob('*')
                               if path.is_file() and path.name != 'manifest.json'}
        persist()
        with (ROOT / 'docs/experiments.md').open('a') as stream:
            stream.write(f'\n## {args.run_id} — {report["timestamp_utc"]}\n\nStatus:{report["status"]}. Command:`{report["command"]}`. Runtime{report["runtime_seconds"]:.2f}s. Fixed CPU multilabelMLP128/64,14 logistic outputs,24epochs/Adam.001/batch512/alpha1e-4/seed4173/two threads. Exact395LP-v1 features and stored40% training samples; per-fold train-only float64 normalization thenclip[-10,10]/float32. NN jointweights0/.1/.25/.5 versus frozen acceptedjointblend, selected2021 before2022 target decoding; no early stopping or posthoc weights. Both folds are reused development evidence.\n')
            for year, fold in report['folds'].items():
                stream.write(f'\n- {year}: rawNN={fold["raw_neural"]["micro_auc"]}; anchor={fold["anchor"]["micro_auc"]}; selected={fold.get("selected", {}).get("micro_auc")}; weight={report.get("selected_weight")}.\n')
            stream.write(f'\nAll source/data/cache/normalization/sample/pickle/portable-array hashes, complete model parameters, warnings, hardware/software, exact split IDs/dates and checkpoint replay proofs:`artifacts/{args.run_id}/manifest.json`. Accepted:{report.get("accepted")}. No full-data fit or Kaggle submission; ID/score/rank/observation:none.\n')


if __name__ == '__main__':
    main()
