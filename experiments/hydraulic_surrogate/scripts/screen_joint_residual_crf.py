"""Pre-2021 forward-OOF screen of a fixed three-generator residual CRF."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import shlex
import shutil
import sys
import time
import traceback

for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[variable] = '1'

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_nonlinear_pq_specialists import build_608
from sintef_project.joint_residual_crf import fit_model, predict, objective_gradient, unary_objective
from sintef_project.nonlinear_pq import digest
from sintef_project.postprocess import project_vinje

UNITS = ['Vinje_G1', 'Vinje_G2', 'Vinje_G3']


def covariates(x: pd.DataFrame) -> np.ndarray:
    """Four fixed legal features: intercept, price change, mean margin, time left."""
    price = x['norm_price'].to_numpy().reshape(-1, 168)
    delta = np.diff(price, axis=1, prepend=price[:, :1])
    margin = x[[f'lpjoint_economic_margin_{u}' for u in UNITS]].to_numpy().mean(axis=1).reshape(-1, 168)
    remaining = np.broadcast_to((167 - np.arange(168)) / 167, price.shape)
    return np.stack([np.ones_like(price), delta, margin, remaining], axis=2).astype(np.float64)


def score(probability: np.ndarray, y: np.ndarray, z: np.ndarray, model: dict) -> dict:
    """Measure fixed candidate marginals and proper unprojected sequence NLL."""
    raw = predict(probability, z, model)
    projected = project_vinje(raw.reshape(-1, 3), UNITS).reshape(raw.shape)
    function = objective_gradient if model['kind'] == 'joint' else unary_objective
    nll, _ = function(np.asarray(model['theta']), probability, y, z, ridge=0)
    return {'raw': raw, 'projected': projected, 'sequence_nll_per_hour': nll,
            'pooled_auc': float(roc_auc_score(y.ravel(), projected.ravel())),
            'raw_pooled_auc': float(roc_auc_score(y.ravel(), raw.ravel())),
            'interior_auc': float(roc_auc_score(y[:, 12:-12].ravel(), projected[:, 12:-12].ravel())),
            'unit_auc': {u: float(roc_auc_score(y[:, :, j].ravel(), projected[:, :, j].ravel())) for j, u in enumerate(UNITS)}}


def main() -> None:
    """Produce nine leakage-free base models, lock residual fits, then screen2020."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--protocol-sha256', required=True)
    args = parser.parse_args()
    out = ROOT / 'artifacts' / args.run_id
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    report = {'run_id': args.run_id, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'command': shlex.join([sys.executable, *sys.argv]), 'status': 'initializing', 'seed': 4173,
              'kaggle': None, 'screen_labels_opened_utc': None, 'folds': {},
              'scope': 'Vinje3-only pre2021screen, not full14-unit or Kaggle evidence',
              'environment': {'python': sys.version, 'platform': platform.platform(), 'machine': platform.machine(),
                              'packages': {n: importlib.metadata.version(n) for n in
                                           ['numpy', 'pandas', 'scipy', 'scikit-learn', 'lightgbm']}}}
    try:
        path = ROOT / 'docs/joint_residual_crf_protocol.json'
        if digest(path) != args.protocol_sha256:
            raise ValueError('Explicit frozen protocol hash mismatch')
        protocol = json.loads(path.read_text())
        for group in ['source', 'data', 'inputs']:
            for name, expected in protocol[group].items():
                if digest(ROOT / name) != expected:
                    raise ValueError(f'Changed {group}: {name}')
        report.update(protocol_sha256=args.protocol_sha256, source=protocol['source'], data=protocol['data'],
                      inputs=protocol['inputs'], model_parameters=protocol['model_parameters'],
                      residual_parameters=protocol['residual_parameters'])
        for name in protocol['source']:
            target = out / 'source' / name
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT / name, target)
        baseline = json.loads((ROOT / 'artifacts/lgb-normalized-lp-context-joint-001/manifest.json').read_text())
        # Target loader masks all2020andlater label values; building features is input-only.
        x, labels, train, _, meta = build_608(2019, baseline)
        if meta['units'] != list(baseline['folds']['2021']['unit_auc']):
            raise ValueError('Changed generator order')
        report.update(feature_names=x.columns.tolist(), feature_spec=meta['feature_spec'])
        z = covariates(x)
        y = np.column_stack([labels[u] for u in UNITS]).astype(np.uint8)
        dates = pd.to_datetime(train.starttime, utc=True)
        rng = np.random.default_rng(4173)
        fold_arrays = {}
        for year in [2018, 2019, 2020]:
            begin = pd.Timestamp(f'{year}-01-01', tz='UTC')
            last = pd.Timestamp(f'{year}-12-31' if year == 2018 else f'{year}-12-25', tz='UTC')
            tr = np.flatnonzero(dates < begin - pd.Timedelta(days=6))
            va = np.flatnonzero((dates >= begin) & (dates <= last))
            if dates.iloc[tr].max() + pd.Timedelta(hours=167) >= begin:
                raise ValueError('OOF training horizon overlaps its predictions')
            rows = (tr[:, None] * 168 + np.arange(168)).ravel()
            sample = np.sort(rng.choice(rows, size=int(len(rows) * .4), replace=False))
            validation_rows = (va[:, None] * 168 + np.arange(168)).ravel()
            np.save(out / f'train_sample_{year}.npy', sample)
            xt, xv = x.iloc[sample], x.iloc[validation_rows]
            p = np.zeros((len(validation_rows), 3), dtype=np.float32)
            for j, unit in enumerate(UNITS):
                model = lgb.LGBMClassifier(**protocol['model_parameters'])
                model.fit(xt, y[sample, j])
                p[:, j] = model.predict_proba(xv)[:, 1]
                checkpoint = out / f'fold{year}_{unit}.txt'
                model.booster_.save_model(str(checkpoint))
                replay = lgb.Booster(model_file=str(checkpoint)).predict(xv, num_threads=2).astype(np.float32)
                np.testing.assert_array_equal(p[:, j], replay)
                print('OOF_MODEL', year, unit, 'replay_exact', flush=True)
            values = {'probability': p.reshape(len(va), 168, 3), 'covariates': z[va],
                      'case_ids': train.iloc[va]['Run No'].to_numpy(), 'dates': dates.iloc[va].astype(str).to_numpy(dtype=str)}
            if year < 2020:
                values['labels'] = y[validation_rows].reshape(len(va), 168, 3)
            np.savez_compressed(out / f'oof_{year}.npz', **values)
            fold_arrays[year] = values
            report['folds'][str(year)] = {'train_ids': train.iloc[tr]['Run No'].tolist(),
                                          'prediction_ids': values['case_ids'].tolist(), 'purge_days': 6,
                                          'train_dates': [str(dates.iloc[tr].min()), str(dates.iloc[tr].max())],
                                          'prediction_dates': [str(dates.iloc[va].min()), str(dates.iloc[va].max())],
                                          'sample_rows': len(sample), 'checkpoint_replay_exact': True}
            (out / 'manifest.json').write_text(json.dumps(report, indent=2))
        del x, y, labels, xt, xv
        p_train = np.concatenate([fold_arrays[year]['probability'] for year in [2018, 2019]])
        y_train = np.concatenate([fold_arrays[year]['labels'] for year in [2018, 2019]])
        z_train = np.concatenate([fold_arrays[year]['covariates'] for year in [2018, 2019]])
        mean, std = z_train[:, :, 1:].mean(axis=(0, 1)), z_train[:, :, 1:].std(axis=(0, 1))
        std = np.where(std == 0, 1, std)
        normalized_train = z_train.copy()
        normalized_train[:, :, 1:] = np.clip((z_train[:, :, 1:] - mean) / std, -5, 5)
        models = {'joint': fit_model(p_train, y_train, normalized_train, 'joint'),
                  'unary': fit_model(p_train, y_train, normalized_train, 'unary'),
                  'identity': {'kind': 'unary', 'theta': [0.] * 12}}
        (out / 'models.json').write_text(json.dumps({'models': models, 'mean': mean.tolist(), 'std': std.tolist()}, indent=2))
        report['residual_training_ids'] = np.concatenate([fold_arrays[year]['case_ids'] for year in [2018, 2019]]).tolist()
        fit_lock = {'status': 'locked', 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                    'protocol_sha256': args.protocol_sha256, 'training_ids': report['residual_training_ids'],
                    'maximum_residual_target_year': 2019, 'models_sha256': digest(out / 'models.json'),
                    'oof_artifacts': {f'oof_{year}.npz': digest(out / f'oof_{year}.npz') for year in [2018, 2019, 2020]}}
        (out / 'fit_lock.json').write_text(json.dumps(fit_lock, indent=2))
        report['fit_lock_sha256'] = digest(out / 'fit_lock.json')
        screen = fold_arrays[2020]
        screen_z = screen['covariates'].copy()
        screen_z[:, :, 1:] = np.clip((screen_z[:, :, 1:] - mean) / std, -5, 5)
        restored = json.loads((out / 'models.json').read_text())
        raw = {name: predict(screen['probability'], screen_z, model) for name, model in models.items()}
        for name in models:
            np.testing.assert_array_equal(raw[name], predict(screen['probability'], screen_z, restored['models'][name]))
        np.savez_compressed(out / 'screen_input_predictions.npz', **raw, case_ids=screen['case_ids'])
        report['screen_predictions_sealed_sha256'] = digest(out / 'screen_input_predictions.npz')
        report['screen_labels_opened_utc'] = datetime.now(timezone.utc).isoformat()
        selected = train['Run No'].isin(screen['case_ids']).to_numpy()
        columns = [f'result_committed_{u}_t{h}' for u in UNITS for h in range(168)]
        target = pd.read_csv(ROOT / 'sintef_project/data/kernel/Unit_commitment_decisions.csv',
                             skiprows=(np.flatnonzero(~selected) + 1).tolist(), usecols=['Run No', *columns],
                             dtype={name: 'uint8' for name in columns})
        np.testing.assert_array_equal(target['Run No'].to_numpy(), screen['case_ids'])
        y_screen = target[columns].to_numpy().reshape(len(target), 3, 168).transpose(0, 2, 1)
        if len(target) != 360 or not np.isin(y_screen, [0, 1]).all():
            raise ValueError('Unexpected2020screen targets')
        results = {name: score(screen['probability'], y_screen, screen_z, model) for name, model in models.items()}
        for name, result in results.items():
            np.savez_compressed(out / f'screen_{name}.npz', predictions=result.pop('projected'), raw_predictions=result.pop('raw'),
                                labels=y_screen, case_ids=screen['case_ids'], units=np.asarray(UNITS))
        passed = (results['joint']['pooled_auc'] > max(results[name]['pooled_auc'] for name in ['identity', 'unary']) and
                  results['joint']['sequence_nll_per_hour'] < min(results[name]['sequence_nll_per_hour'] for name in ['identity', 'unary']))
        report.update(status='complete', results=results, gate_passed=bool(passed),
                      screen_ids=screen['case_ids'].tolist(), units=UNITS,
                      model_summary={name: {k: v for k, v in model.items() if k != 'theta'} for name, model in models.items()})
        print(json.dumps({'results': results, 'gate_passed': passed}, indent=2), flush=True)
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        report['runtime_seconds'] = time.monotonic() - started
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report['peak_rss_bytes'] = int(rss if sys.platform == 'darwin' else rss * 1024)
        report['artifacts'] = {str(p.relative_to(out)): digest(p) for p in out.rglob('*')
                               if p.is_file() and p.name != 'manifest.json'}
        (out / 'manifest.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
