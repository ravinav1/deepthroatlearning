"""Pre-2021 full14 residual ranking screen with two matched BCE controls."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
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
from sklearn.metrics import roc_auc_score, log_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sintef_project.pooled_pairwise_residual import (
    PooledPairwiseObjective, PointwiseBCEObjective, apply_residual, base_logits,
)
from sintef_project.postprocess import project_vinje


def digest(path: Path) -> str:
    """Hash files without materializing large feature matrices."""
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def expand_features(features: np.ndarray, probability: np.ndarray) -> np.ndarray:
    """Repeat legal case-hour inputs for each unit, then append identity/logit."""
    n_units = probability.shape[-1]
    if features.ndim != 2 or features.shape[1] != 608 or probability.size != len(features) * n_units:
        raise ValueError('Expected aligned 608-column case-hour features and unit probabilities')
    if features.dtype != np.float32 or not np.isfinite(features).all():
        raise ValueError('Features must be finite float32')
    expanded = np.empty((probability.size, 608 + n_units + 1), dtype=np.float32)
    expanded[:, 608:608 + n_units] = 0
    for j in range(n_units):
        expanded[j::n_units, :608] = features
        expanded[j::n_units, 608 + j] = 1
    expanded[:, -1] = base_logits(probability).ravel()
    return expanded


def read_screen_labels(path: Path, allowed_ids: np.ndarray, units: list[str]) -> np.ndarray:
    """Decode only explicitly selected 2020 starts, rejecting later-label scopes."""
    metadata = pd.read_csv(path, usecols=['Run No', 'starttime'])
    selected = metadata['Run No'].isin(allowed_ids).to_numpy()
    np.testing.assert_array_equal(metadata.loc[selected, 'Run No'], allowed_ids)
    dates = pd.to_datetime(metadata.loc[selected, 'starttime'], utc=True)
    if not len(dates) or not ((dates >= pd.Timestamp('2020-01-01', tz='UTC')) &
                             (dates <= pd.Timestamp('2020-12-25', tz='UTC'))).all():
        raise ValueError('Screen target scope must be 2020 through December25')
    columns = [f'result_committed_{u}_t{h}' for u in units for h in range(168)]
    target = pd.read_csv(path, skiprows=(np.flatnonzero(~selected) + 1).tolist(),
                         usecols=['Run No', *columns], dtype={name: 'uint8' for name in columns})
    np.testing.assert_array_equal(target['Run No'], allowed_ids)
    values = target[columns].to_numpy().reshape(len(target), len(units), 168).transpose(0, 2, 1)
    if not np.isin(values, [0, 1]).all():
        raise ValueError('Invalid binary screen targets')
    return values


def evaluate(probability: np.ndarray, labels: np.ndarray, dates: np.ndarray,
             units: list[str]) -> dict:
    """Score the frozen final projection on full and prespecified half-year scopes."""
    projected = project_vinje(probability.reshape(-1, len(units)), units).reshape(probability.shape)
    first_half = pd.to_datetime(dates, utc=True) < pd.Timestamp('2020-07-01', tz='UTC')
    result = {'projected_auc': float(roc_auc_score(labels.ravel(), projected.ravel())),
              'raw_auc': float(roc_auc_score(labels.ravel(), probability.ravel())),
              'raw_bce': float(log_loss(labels.ravel(), probability.astype(np.float64).ravel())),
              'halves': {}, 'unit_auc': {u: float(roc_auc_score(labels[:, :, j].ravel(), projected[:, :, j].ravel()))
                                         for j, u in enumerate(units)}}
    for name, mask in [('first_half', first_half), ('second_half', ~first_half)]:
        result['halves'][name] = {'cases': int(mask.sum()),
                                  'projected_auc': float(roc_auc_score(labels[mask].ravel(), projected[mask].ravel()))}
    return result


def main() -> None:
    """Lock three trained residuals and all predictions before opening screen labels."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--oof-manifest-sha256', required=True)
    parser.add_argument('--oof-audit', type=Path, required=True)
    parser.add_argument('--oof-audit-sha256', required=True)
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id or args.run_id in ['', '.', '..']:
        raise ValueError('Invalid immutable output run ID')
    out = ROOT / 'artifacts' / args.run_id
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    report = {'run_id': args.run_id, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'command': shlex.join([sys.executable, *sys.argv]), 'status': 'initializing',
              'seed': 4173, 'scope': 'Single608LGB full14 early OOF mechanism screen; not accepted GPU ensemble',
              'screen_labels_opened_utc': None, 'kaggle_submission': None,
              'environment': {'python': sys.version, 'platform': platform.platform(), 'machine': platform.machine(),
                              'packages': {n: importlib.metadata.version(n) for n in
                                           ['numpy', 'pandas', 'scipy', 'scikit-learn', 'lightgbm']}}}
    try:
        protocol_path = ROOT / 'docs/pooled_pairwise_residual_protocol_v2.json'
        if digest(protocol_path) != args.protocol_sha256:
            raise ValueError('Explicit residual protocol hash differs')
        protocol = json.loads(protocol_path.read_text())
        for group in ['source', 'data', 'inputs']:
            for name, expected in protocol[group].items():
                if digest(ROOT / name) != expected:
                    raise ValueError(f'Changed {group}: {name}')
        oof_dir = ROOT / protocol['oof_directory']
        if digest(oof_dir / 'manifest.json') != args.oof_manifest_sha256 or digest(args.oof_audit) != args.oof_audit_sha256:
            raise ValueError('Explicit completed OOF or audit manifest hash differs')
        oof = json.loads((oof_dir / 'manifest.json').read_text())
        audit = json.loads(args.oof_audit.read_text())
        if (oof['status'] != 'complete' or oof['protocol_sha256'] != protocol['oof_protocol_sha256'] or
                audit['status'] != 'passed' or audit['oof_manifest_sha256'] != args.oof_manifest_sha256):
            raise ValueError('Require complete independently verified frozen OOF cache')
        if oof['screen_2020_labels_read'] or not oof['all_42_checkpoint_replays_exact']:
            raise ValueError('OOF cache label-sealing or replay failure')
        for name, expected in oof['artifacts'].items():
            if digest(oof_dir / name) != expected:
                raise ValueError(f'Changed OOF artifact: {name}')
        for name in protocol['source']:
            target = out / 'source' / name
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT / name, target)
        shutil.copy2(protocol_path, out / 'protocol.json')
        report.update(protocol_sha256=args.protocol_sha256, oof_manifest_sha256=args.oof_manifest_sha256,
                      oof_audit_sha256=args.oof_audit_sha256, source=protocol['source'], data=protocol['data'],
                      inputs=protocol['inputs'], model_parameters=protocol['model_parameters'],
                      objective_parameters=protocol['objective_parameters'])
        arrays = {}
        for year in [2018, 2019, 2020]:
            with np.load(oof_dir / f'oof_{year}.npz', allow_pickle=False) as saved:
                arrays[year] = {key: saved[key] for key in saved.files}
            np.testing.assert_array_equal(arrays[year]['case_ids'], protocol['case_ids'][str(year)])
            if arrays[year]['units'].tolist() != protocol['units'] or ('labels' in arrays[year]) != (year < 2020):
                raise ValueError('Changed OOF unit schema or screen-label separation')
        units = protocol['units']
        p_train = np.concatenate([arrays[year]['probability'] for year in [2018, 2019]])
        y_train = np.concatenate([arrays[year]['labels'] for year in [2018, 2019]]).ravel()
        initial = base_logits(p_train).ravel()
        report['training_ids'] = np.concatenate([arrays[year]['case_ids'] for year in [2018, 2019]]).tolist()
        if len(y_train) != 1_702_848 or len(report['training_ids']) != 724:
            raise ValueError('Unexpected full14 residual training scope')
        features = np.concatenate([np.load(oof_dir / f'features_{year}.npy', allow_pickle=False) for year in [2018, 2019]])
        expanded = expand_features(features, p_train)
        report['training_matrix_shape'] = list(expanded.shape)
        feature_names = [f'x{i:03d}' for i in range(expanded.shape[1])]
        data = lgb.Dataset(expanded, label=y_train, init_score=initial, feature_name=feature_names,
                           params={'max_bin': 255, 'data_random_seed': 4173, 'num_threads': 2,
                                   'min_data_in_leaf': 100}, free_raw_data=True)
        data.construct()
        del expanded, features
        gc.collect()
        models = {}
        states = {'pairwise': PooledPairwiseObjective(y_train, initial,
                           pairs_per_round=protocol['objective_parameters']['pairs_per_round'], seed=4173),
                  'bce': PointwiseBCEObjective(y_train, initial),
                  'balanced_bce': PointwiseBCEObjective(y_train, initial, class_balanced=True)}
        report['class_counts'] = {'positive': int(y_train.sum()), 'negative': int((y_train == 0).sum())}
        report['balanced_bce_weights'] = {str(value): float(states['balanced_bce'].weights[y_train == value][0]) for value in [0, 1]}
        for kind in protocol['fit_order']:
            state = states[kind]
            def objective(scores, dataset):
                return state(scores, dataset)
            def progress(environment):
                if (environment.iteration + 1) % 10 == 0:
                    print('RESIDUAL_ROUND', kind, environment.iteration + 1, state.trace[-1], flush=True)
            params = dict(protocol['model_parameters']); params['objective'] = objective
            models[kind] = lgb.train(params, data, num_boost_round=80, callbacks=[progress])
            if len(state.trace) != 80 or models[kind].current_iteration() != 80:
                raise ValueError('Fixed80-round residual fit did not complete')
            models[kind].save_model(str(out / f'{kind}.txt'))
            (out / f'{kind}_trace.json').write_text(json.dumps(state.trace, indent=2))
            print('RESIDUAL_FIT_COMPLETE', kind, flush=True)
        del data, states, y_train, initial, p_train
        gc.collect()
        lock = {'status': 'locked', 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                'protocol_sha256': args.protocol_sha256, 'oof_manifest_sha256': args.oof_manifest_sha256,
                'training_ids': report['training_ids'], 'maximum_target_year': 2019,
                'model_sha256': {kind: digest(out / f'{kind}.txt') for kind in models},
                'trace_sha256': {kind: digest(out / f'{kind}_trace.json') for kind in models}}
        (out / 'fit_lock.json').write_text(json.dumps(lock, indent=2))
        report['fit_lock_sha256'] = digest(out / 'fit_lock.json')
        screen = arrays[2020]
        p_screen = screen['probability']
        identity = apply_residual(p_screen, np.zeros_like(p_screen))
        if identity.dtype != p_screen.dtype or identity.tobytes() != p_screen.tobytes():
            raise ValueError('Zero residual is not a byte-exact identity')
        screen_features = np.load(oof_dir / 'features_2020.npy', mmap_mode='r', allow_pickle=False)
        raw_residual = {kind: np.empty(p_screen.shape, dtype=np.float64) for kind in models}
        restored = {kind: lgb.Booster(model_file=str(out / f'{kind}.txt')) for kind in models}
        for start in range(0, len(p_screen), 16):
            stop = min(len(p_screen), start + 16)
            chunk = expand_features(screen_features[start * 168:stop * 168], p_screen[start:stop])
            for kind in models:
                residual = models[kind].predict(chunk, raw_score=True, num_threads=2)
                replay = restored[kind].predict(chunk, raw_score=True, num_threads=2)
                np.testing.assert_array_equal(residual, replay)
                raw_residual[kind][start:stop] = residual.reshape(stop - start, 168, 14)
        raw = {'identity': identity, **{kind: apply_residual(p_screen, delta) for kind, delta in raw_residual.items()}}
        report['residual_summary'] = {kind: {'mean': float(delta.mean()), 'std': float(delta.std()),
                                            'min': float(delta.min()), 'max': float(delta.max())} for kind, delta in raw_residual.items()}
        np.savez_compressed(out / 'screen_input_predictions.npz', **raw, case_ids=screen['case_ids'], units=np.asarray(units))
        np.savez_compressed(out / 'screen_residuals.npz', **raw_residual, case_ids=screen['case_ids'])
        prediction_lock = {'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                           'fit_lock_sha256': report['fit_lock_sha256'],
                           'screen_predictions_sha256': digest(out / 'screen_input_predictions.npz'),
                           'screen_residuals_sha256': digest(out / 'screen_residuals.npz'),
                           'all_three_checkpoint_replays_exact': True, 'identity_bytes_exact': True}
        (out / 'prediction_lock.json').write_text(json.dumps(prediction_lock, indent=2))
        report['prediction_lock_sha256'] = digest(out / 'prediction_lock.json')
        report['screen_labels_opened_utc'] = datetime.now(timezone.utc).isoformat()
        y_screen = read_screen_labels(ROOT / 'sintef_project/data/kernel/Unit_commitment_decisions.csv', screen['case_ids'], units)
        if y_screen.shape != (360, 168, 14):
            raise ValueError('Unexpected screen shape')
        results = {name: evaluate(value, y_screen, screen['dates'], units) for name, value in raw.items()}
        for name, probability in raw.items():
            projected = project_vinje(probability.reshape(-1, 14), units).reshape(probability.shape)
            np.savez_compressed(out / f'screen_{name}.npz', predictions=projected, raw_predictions=probability,
                                labels=y_screen, case_ids=screen['case_ids'], units=np.asarray(units))
        candidate = results['pairwise']['projected_auc']
        identity_auc = results['identity']['projected_auc']
        relative_gain = (candidate - identity_auc) / (1 - identity_auc)
        controls = ['identity', 'bce', 'balanced_bce']
        checks = {'whole_year_beats_all_controls': all(candidate > results[name]['projected_auc'] for name in controls)}
        for half in ['first_half', 'second_half']:
            checks[half + '_beats_identity'] = (results['pairwise']['halves'][half]['projected_auc'] >
                                                results['identity']['halves'][half]['projected_auc'])
        report.update(status='complete', results=results, gate_checks=checks,
                      gate_passed=bool(all(checks.values())), relative_error_reduction=relative_gain,
                      substantial_gain=bool(relative_gain >= .05),
                      screen_ids=screen['case_ids'].tolist(), identity_bytes_exact=True, all_checkpoint_replays_exact=True)
        print(json.dumps({'results': results, 'gate_checks': checks, 'relative_error_reduction': relative_gain,
                          'substantial_gain': report['substantial_gain']}, indent=2), flush=True)
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        report['runtime_seconds'] = time.monotonic() - started
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report['peak_rss_bytes'] = int(rss if sys.platform == 'darwin' else rss * 1024)
        report['artifacts'] = {p.relative_to(out).as_posix(): digest(p) for p in out.rglob('*') if p.is_file() and p.name != 'manifest.json'}
        (out / 'manifest.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
