"""Fixed small logistic/spline probe of weak-unit economic transfer."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import time
import traceback
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from threadpoolctl import threadpool_limits
from sintef_project.features import build_features
from sintef_project.physics import add_physics_features

UNITS = ['Lio_G1', 'Haukeli_G1']
RESERVOIRS = {'Lio': ['Byrtevatn', 'Botnedalsvatn', 'Bandak'],
              'Haukeli': ['Vatjern', 'Langeidvatn', 'Vinjevatn']}
WEIGHTS = [0.0, 0.1, 0.25, 0.5, 1.0]
LOGISTIC = dict(C=0.1, solver='lbfgs', max_iter=400, tol=1e-5, random_state=4173)
SPLINE = dict(n_knots=4, degree=3, knots='quantile', extrapolation='linear', include_bias=False)


def checksum(path: Path) -> str:
    """Return a SHA256 file digest."""
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def economic_features(x: pd.DataFrame, plant: str) -> pd.DataFrame:
    """Derive scale-relative prices and interactions with known initial state."""
    local, upstream, downstream = RESERVOIRS[plant]
    scale = np.repeat(1 + np.abs(x.price.to_numpy().reshape(-1, 168)).mean(axis=1), 168)
    values = {}
    for name in ['price', 'price_week_mean', 'price_week_std', 'price_week_min', 'price_week_max',
                 'price_remaining_mean', f'water_value_{local}', f'water_value_{downstream}',
                 f'price_minus_water_{local}', f'physics_switch_margin_minimum_{plant}',
                 f'physics_switch_margin_maximum_{plant}']:
        values[f'normalized_{name}'] = x[name].to_numpy() / scale
    for name in ['price_week_rank', 'price_week_zscore', 'horizon_hour', 'season_sin', 'season_cos',
                 f'price_over_water_{local}', f'physics_head_{plant}', f'physics_pool_fill_{plant}',
                 f'physics_supply_fraction_{plant}', f'physics_catchment_inflow_fraction_{plant}',
                 f'physics_dispatch_to_avoid_spill_{plant}', f'physics_profitable_remaining_fraction_{plant}']:
        values[name] = x[name].to_numpy()
    for reservoir in RESERVOIRS[plant]:
        for prefix in ['fill_', 'headroom_after_inflow_']:
            values[prefix + reservoir] = x[prefix + reservoir].to_numpy()
    for name in ['min_volume_Byrtevatn', 'min_flow_b_Byrtevatn', 'min_flow_b_Vinjevatn', 'min_flow_b_Bandak']:
        values[name] = x[name].to_numpy()
    margin = values[f'normalized_price_minus_water_{local}']
    head = x[f'physics_head_{plant}'].to_numpy() / 100
    fill = x[f'physics_pool_fill_{plant}'].to_numpy()
    values['normalized_margin_times_head_100m'] = margin * head
    values['normalized_price_times_head_100m'] = values['normalized_price'] * head
    values['normalized_margin_times_pool_fill'] = margin * fill
    values['normalized_margin_times_remaining_fraction'] = margin * (168 - x.horizon_hour.to_numpy()) / 168
    values['local_to_downstream_water_value_ratio'] = x[f'water_value_{local}'].to_numpy() / (1 + x[f'water_value_{downstream}'].to_numpy())
    values['inverse_price_scale'] = 1 / scale
    result = pd.DataFrame(values, dtype=np.float64)
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError('Nonfinite economic features')
    return result


def report_metrics(y: np.ndarray, p: np.ndarray, units: list[str]) -> dict:
    """Score full-matrix micro AUC and the two targeted units."""
    return {'micro_auc': float(roc_auc_score(y.ravel(), p.ravel())),
            'units': {u: {'auc': float(roc_auc_score(y[:, units.index(u)], p[:, units.index(u)])),
                          'log_loss': float(log_loss(y[:, units.index(u)], p[:, units.index(u)], labels=[0, 1]))}
                      for u in UNITS}}


def main() -> None:
    """Fit predeclared models and freeze a blend before evaluating 2022."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', default='economic-probe-001')
    parser.add_argument('--base-run', type=Path, default=ROOT / 'artifacts/lgb-physics-001')
    args = parser.parse_args()
    out = ROOT / 'artifacts' / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    manifest = {
        'run_id': args.run_id, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'command': shlex.join([sys.executable, *sys.argv]), 'status': 'running',
        'hypothesis': 'Scale-relative logistic or additive cubic-spline economic thresholds may extrapolate better than trees for Lio and Haukeli',
        'units': UNITS, 'seed': 4173, 'train_sample_rows': 80000, 'threads': 2,
        'logistic_parameters': LogisticRegression(**LOGISTIC).get_params(),
        'spline_parameters': SplineTransformer(**SPLINE).get_params(),
        'preprocessing': 'StandardScaler fit only retained sampled training rows. Additive spline variant fits quantile knots only on same sampled training rows, then StandardScaler. Explicit economic/head/storage interactions precede both. No target features.',
        'feature_formulas': {'monetary_scale': '1+mean(abs(price)) over same known168-hour case',
                             'local_downstream_ratio': 'local terminal EUR/MWh/(1+downstream terminal EUR/MWh); statistical ratio, not physical opportunity-cost subtraction',
                             'interactions': 'normalized local margin times head/100, poolfill, or remaining_horizon/168; normalized price times head/100',
                             'reservoirs': RESERVOIRS},
        'early_stopping': 'None; lbfgs tolerance1e-5, max400 iterations, warnings recorded',
        'selection': 'Select independently per weak unit using highest2021 fullmatrix micro AUC with other13 units unchanged; candidate models linear/spline and probability weights0,.1,.25,.5,1. Freeze choices before2022 evaluation.',
        'environment': {'platform': platform.platform(), 'python': sys.version, 'cpu_count': os.cpu_count(),
                        'hardware': subprocess.run(['system_profiler', 'SPHardwareDataType'], capture_output=True, text=True).stdout.split('Serial Number')[0],
                        'packages': {p: importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','scipy','pyyaml','joblib','threadpoolctl']}},
        'data': {}, 'source': {}, 'folds': {},
        'base_manifest_sha256': checksum(args.base_run / 'manifest.json'),
        'kaggle': {'submission_id': None, 'score': None, 'rank': None, 'observation_time': None},
    }
    for path in sorted((ROOT / 'sintef_project').rglob('*')):
        if path.suffix in ['.csv','.yaml']:
            manifest['data'][str(path.relative_to(ROOT))] = checksum(path)
    for path in [Path(__file__), ROOT/'sintef_project/features.py', ROOT/'sintef_project/physics.py']:
        rel = path.relative_to(ROOT)
        manifest['source'][str(rel)] = checksum(path)
        destination = out/'source'/rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    try:
        x, labels, train, _, metadata = build_features(ROOT)
        x, physics_spec = add_physics_features(x, ROOT)
        tables = {unit: economic_features(x, unit.rsplit('_G',1)[0]) for unit in UNITS}
        manifest['feature_names'] = {u: list(frame) for u, frame in tables.items()}
        manifest['base_features'] = metadata['feature_spec']
        manifest['physics_spec'] = physics_spec
        price_scale = np.repeat(1 + np.abs(x.price.to_numpy().reshape(-1,168)).mean(axis=1), 168)
        del x
        units = metadata['units']
        chosen = {}
        for year in (2021, 2022):
            begin, end = pd.Timestamp(f'{year}-01-01',tz='UTC'), pd.Timestamp(f'{year+1}-01-01',tz='UTC')
            tc = np.flatnonzero(train.starttime < begin-pd.Timedelta(days=6))
            vc = np.flatnonzero((train.starttime >= begin)&(train.starttime < end))
            assert train.starttime.iloc[tc].max()+pd.Timedelta(hours=167)<begin
            tr = (tc[:,None]*168+np.arange(168)).ravel()
            va = (vc[:,None]*168+np.arange(168)).ravel()
            sample = np.sort(np.random.default_rng(4173+year).choice(tr, min(80000,len(tr)), replace=False))
            np.save(out/f'train_sample_{year}.npy',sample)
            base_path = args.base_run/f'validation_{year}.npz'
            base = np.load(base_path)
            if list(base['units']) != units or not np.array_equal(base['case_ids'],train.iloc[vc]['Run No']):
                raise ValueError('Base row/unit order differs')
            y, p = base['labels'], base['predictions']
            for unit in UNITS:
                np.testing.assert_array_equal(y[:,units.index(unit)], labels[unit][va])
            fold = {'train_ids': train.iloc[tc]['Run No'].tolist(), 'validation_ids': train.iloc[vc]['Run No'].tolist(),
                    'train_dates': [str(train.starttime.iloc[tc].min()), str(train.starttime.iloc[tc].max())],
                    'validation_dates': [str(train.starttime.iloc[vc].min()),str(train.starttime.iloc[vc].max())],
                    'purge_days': 6, 'sample_seed': 4173+year, 'base_file_sha256': checksum(base_path),
                    'baseline': report_metrics(y,p,units), 'models': {}, 'unit_blend_search_2021': {},
                    'train_price_scale_quantiles': np.quantile(price_scale[tr],[0,.1,.5,.9,1]).tolist(),
                    'validation_price_scale_quantiles': np.quantile(price_scale[va],[0,.1,.5,.9,1]).tolist()}
            model_predictions = {}
            for unit in UNITS:
                j = units.index(unit)
                unit_x = tables[unit]
                fold['models'][unit] = {}
                for kind in ['linear','spline']:
                    transforms = [SplineTransformer(**SPLINE)] if kind == 'spline' else []
                    model = make_pipeline(*transforms, StandardScaler(), LogisticRegression(**LOGISTIC))
                    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=2):
                        warnings.simplefilter('always')
                        model.fit(unit_x.iloc[sample],labels[unit][sample])
                        pred = model.predict_proba(unit_x.iloc[va])[:,1].astype(np.float32)
                    joblib.dump(model,out/f'{year}_{unit}_{kind}.joblib')
                    model_predictions[(unit,kind)] = pred
                    replacement = p.copy()
                    replacement[:,j] = pred
                    info = report_metrics(y,replacement,units)
                    info['warnings'] = [str(w.message) for w in caught]
                    info['n_iter'] = model[-1].n_iter_.tolist()
                    fold['models'][unit][kind] = info
                    print(year,unit,kind,json.dumps(info),flush=True)
                    if year == 2021:
                        fold['unit_blend_search_2021'].setdefault(unit,[])
                        for weight in WEIGHTS:
                            replacement[:,j] = p[:,j]*(1-weight)+pred*weight
                            fold['unit_blend_search_2021'][unit].append({'kind':kind,'weight':weight,'micro_auc':float(roc_auc_score(y.ravel(),replacement.ravel()))})
                if year == 2021:
                    chosen[unit] = max(fold['unit_blend_search_2021'][unit],key=lambda candidate:candidate['micro_auc'])
            combined = p.copy()
            for unit in UNITS:
                choice = chosen[unit]
                j = units.index(unit)
                combined[:,j] = p[:,j]*(1-choice['weight'])+model_predictions[(unit,choice['kind'])]*choice['weight']
            fold['frozen_combined'] = report_metrics(y,combined,units)
            fold['selection'] = chosen.copy()
            for unit in UNITS:
                j=units.index(unit)
                fold.setdefault('scale_regime_diagnostics',{})[unit] = {}
                cutoff = np.quantile(price_scale[tr],.9)
                for regime,mask in [('at_most_train_p90',price_scale[va]<=cutoff),('above_train_p90',price_scale[va]>cutoff)]:
                    if mask.any() and np.unique(y[mask,j]).size==2:
                        fold['scale_regime_diagnostics'][unit][regime] = {'rows':int(mask.sum()),'base_auc':float(roc_auc_score(y[mask,j],p[mask,j])), 'base_log_loss':float(log_loss(y[mask,j],p[mask,j],labels=[0,1])),'prevalence':float(y[mask,j].mean()),'mean_prediction':float(p[mask,j].mean())}
            if year == 2021:
                (out/'selection_2021.json').write_text(json.dumps(chosen,indent=2))
            np.savez_compressed(out/f'predictions_{year}.npz', **{u+'_'+k:v for (u,k),v in model_predictions.items()}, combined=combined, labels=y, case_ids=base['case_ids'], units=units)
            manifest['folds'][str(year)] = fold
            (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
            print('FOLD',year,fold['frozen_combined'],flush=True)
        manifest['status']='complete'
    except BaseException:
        manifest['status']='failed'
        manifest['error']=traceback.format_exc()
        raise
    finally:
        manifest['runtime_seconds']=time.time()-started
        manifest['artifacts']={str(p.relative_to(out)):checksum(p) for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'}
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':
    main()
