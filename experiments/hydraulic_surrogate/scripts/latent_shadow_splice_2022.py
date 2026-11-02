"""Confirm one independently reviewed 2021 latent splice on purged 2022."""
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

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.fit_latent_shadow import load_cache, predict_all
from scripts.latent_shadow_splice_2021 import check_frozen_config, fit_families, splice
from sintef_project.latent_shadow import ShadowInputs, TARGET_UNITS, schedule_mixture, soft_targets
from sintef_project.nonlinear_pq import digest

PREVIOUS = ROOT / 'artifacts/latent-shadow-splice-2021-001'
ANCHOR = ROOT / 'artifacts/lp-context-joint-comparison-001'
SELECTION_SHA = '7cea46df8d37773ed568dca0b588cec1f20bbb6a06f6d42a6719924f68c67232'


def read_new_training_labels(path: Path, case_ids: list[int], lock_path: Path, lock_sha: str) -> np.ndarray:
    """Decode only the fixed 365 new training rows after a passing 2021 lock."""
    if lock_sha != SELECTION_SHA or digest(lock_path) != lock_sha:
        raise ValueError('Exact passing 2021 selection lock required')
    lock = json.loads(lock_path.read_text())
    if not lock['eligible_for_2022'] or lock['year'] != 2021 or lock['selected_weight'] != .05:
        raise ValueError('Invalid locked 2021 selection')
    metadata = pd.read_csv(path, usecols=['Run No', 'starttime'])
    dates = pd.to_datetime(metadata.starttime, utc=True)
    expected = metadata.loc[(dates > pd.Timestamp('2020-12-25', tz='UTC')) &
                            (dates <= pd.Timestamp('2021-12-25', tz='UTC')), 'Run No'].tolist()
    if case_ids != expected or len(case_ids) != 365:
        raise ValueError('New training IDs must exactly match the purged 2022 extension')
    selected = metadata['Run No'].isin(case_ids).to_numpy()
    columns = [f'result_committed_{unit}_t{hour}' for unit in TARGET_UNITS for hour in range(168)]
    labels = pd.read_csv(path, skiprows=(np.flatnonzero(~selected) + 1).tolist(),
                         usecols=['Run No', *columns], dtype={c: 'uint8' for c in columns})
    if labels['Run No'].tolist() != case_ids or not np.isin(labels[columns].to_numpy(), [0, 1]).all():
        raise ValueError('Invalid selected training labels or ordering')
    return labels[columns].to_numpy().reshape(len(case_ids), 3, 168)


def check_protocol(expected_sha: str, review_path: Path, review_sha: str) -> tuple[dict, dict]:
    """Check source/data closure, earlier result and an explicit independent audit."""
    path = ROOT / 'docs/latent_shadow_splice_2022_protocol.json'
    if digest(path) != expected_sha or digest(review_path) != review_sha:
        raise ValueError('2022 protocol or independent result audit hash mismatch')
    protocol = json.loads(path.read_text())
    parent = check_frozen_config(protocol['parent_protocol_sha256'])
    for group in ['source', 'inputs']:
        for name, expected in protocol[group].items():
            if digest(ROOT / name) != expected:
                raise ValueError(f'Changed {group} dependency: {name}')
    previous = json.loads((PREVIOUS / 'manifest.json').read_text())
    for name, expected in previous['artifacts'].items():
        if digest(PREVIOUS / name) != expected:
            raise ValueError(f'Changed prior run artifact: {name}')
    lock = json.loads((PREVIOUS / 'selection_2021.json').read_text())
    if (digest(PREVIOUS / 'selection_2021.json') != SELECTION_SHA or not lock['eligible_for_2022']
            or protocol['selected_weight'] != lock['selected_weight'] or protocol['units'] != TARGET_UNITS):
        raise ValueError('2021 gate or frozen splice configuration mismatch')
    review = json.loads(review_path.read_text())
    if review['status'] != 'passed' or review.get('selection_lock_sha256') != SELECTION_SHA:
        raise ValueError('Audit must explicitly verify this passing selection lock')
    return protocol, parent


def main() -> None:
    """Fit through 2021-12-25, then score only the locked five-percent splice."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--selection-audit', type=Path, required=True)
    parser.add_argument('--selection-audit-sha256', required=True)
    args = parser.parse_args()
    out = ROOT / 'artifacts' / args.run_id
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    report = {'run_id': args.run_id, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'status': 'initializing', 'year': 2022, 'seed': 4173,
              'command': shlex.join([sys.executable, *sys.argv]), 'kaggle_submission': None,
              'environment': {'python': sys.version, 'platform': platform.platform(), 'machine': platform.machine(),
                              'packages': {name: importlib.metadata.version(name) for name in
                                           ['numpy', 'pandas', 'scipy', 'scikit-learn', 'PyYAML']}},
              'selection_lock_sha256': SELECTION_SHA, 'selection_audit_sha256': args.selection_audit_sha256,
              'selection_audit': str(args.selection_audit), 'validation_label_access_utc': None}
    try:
        protocol, parent = check_protocol(args.protocol_sha256, args.selection_audit, args.selection_audit_sha256)
        report.update(protocol_sha256=args.protocol_sha256, source=protocol['source'], data=parent['data'],
                      inputs=protocol['inputs'], selected_weight=protocol['selected_weight'])
        for name in protocol['source']:
            target = out / 'source' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        metadata_path = ROOT / 'sintef_project/data/kernel/Unit_commitment_decisions.csv'
        metadata = pd.read_csv(metadata_path, usecols=['Run No', 'starttime'])
        metadata.starttime = pd.to_datetime(metadata.starttime, utc=True)
        training = metadata.loc[metadata.starttime <= pd.Timestamp('2021-12-25', tz='UTC')]
        validation = metadata.loc[metadata.starttime.dt.year == 2022]
        if (training['Run No'].tolist() != list(range(1, 2552)) or len(validation) != 365 or
                training.starttime.max() + pd.Timedelta(days=7) != validation.starttime.min()):
            raise ValueError('Unexpected complete purged 2022 partition')
        earlier, _ = load_cache(ROOT / 'artifacts/latent-shadow-targets-train-001', 'train', parent['learnability_protocol_sha256'])
        later, _ = load_cache(ROOT / 'artifacts/latent-shadow-targets-screen-001', 'screen', parent['learnability_protocol_sha256'])
        x_list, target_list = [earlier['features'], later['features']], [earlier['target'], later['target']]
        id_list = [earlier['case_ids'], later['case_ids']]
        for case_id in range(1456, 1462):
            with np.load(PREVIOUS / f'train_gap_{case_id:04d}.npz', allow_pickle=False) as z:
                np.testing.assert_array_equal(soft_targets(np.sum(z['paths'] != z['labels'][:, None, :], axis=-1)), z['target'])
                x_list.append(z['features'][None]); target_list.append(z['target'][None]); id_list.append(np.asarray([case_id]))
        new_ids = list(range(2187, 2552))
        report['new_training_label_access_utc'] = datetime.now(timezone.utc).isoformat()
        new_labels = read_new_training_labels(metadata_path, new_ids, PREVIOUS / 'selection_2021.json', SELECTION_SHA)
        cache = ROOT / 'artifacts/nonlinear-pq-features-001'
        cm = json.loads((cache / 'manifest.json').read_text())
        if digest(cache / 'manifest.json') != parent['cache_manifest_sha256']:
            raise ValueError('Nonlinear cache manifest changed')
        for name in ['features.npz', 'raw_on_rewards.npz']:
            if digest(cache / name) != cm['artifacts'][name]:
                raise ValueError('Frozen nonlinear arrays changed')
        with np.load(cache / 'features.npz', allow_pickle=False) as z:
            saved_features, cached_ids = z['values'], z['case_ids']
        with np.load(cache / 'raw_on_rewards.npz', allow_pickle=False) as z:
            saved_rewards = z['rewards']
        index_of = {int(value): index for index, value in enumerate(cached_ids)}
        inputs = ShadowInputs(ROOT, ROOT / 'artifacts/lp-full-001')
        reused_inputs = 0
        for i, case_id in enumerate(new_ids):
            saved = PREVIOUS / f'validation_input_{case_id:04d}.npz'
            if saved.exists():
                with np.load(saved, allow_pickle=False) as z:
                    values = {name: z[name] for name in z.files if name != 'case_id'}
                reused_inputs += 1
            else:
                values = inputs.grid_case(index_of[case_id])
            index = index_of[case_id]
            np.testing.assert_array_equal(values['old_features'], saved_features[index])
            np.testing.assert_array_equal(values['rewards'][:, 8, :].T, saved_rewards[index])
            error = np.sum(values['paths'] != new_labels[i, :, None, :], axis=-1)
            target = soft_targets(error)
            np.savez_compressed(out / f'train_new_{case_id:04d}.npz', **values, labels=new_labels[i],
                                target=target, error=error, case_id=np.asarray(case_id))
            x_list.append(values['features'][None]); target_list.append(target[None]); id_list.append(np.asarray([case_id]))
        if reused_inputs != 359:
            raise ValueError('Expected359 reused2021 grids and6 new2020 purge grids')
        all_ids = np.concatenate(id_list)
        order = np.argsort(all_ids)
        x, target = np.concatenate(x_list)[order], np.concatenate(target_list)[order]
        np.testing.assert_array_equal(all_ids[order], training['Run No'].to_numpy())
        families, permutation = fit_families(x, target, pd.DatetimeIndex(training.starttime))
        np.save(out / 'permutation_indices.npy', permutation)
        (out / 'models.json').write_text(json.dumps(families, indent=2))
        restored = json.loads((out / 'models.json').read_text())
        train_predictions = {name: predict_all(x, family) for name, family in families.items()}
        for name in families:
            np.testing.assert_array_equal(train_predictions[name], predict_all(x, restored[name]))
        np.savez_compressed(out / 'training_distributions.npz', **train_predictions,
                            case_ids=training['Run No'].to_numpy(), units=np.asarray(TARGET_UNITS))
        report.update(train_ids=training['Run No'].tolist(), train_dates=training.starttime.astype(str).tolist(),
                      maximum_fitted_label_year=2021, new_target_cases=365, reused_targets=2186,
                      reused_new_target_input_grids=359, newly_generated_training_grids=6)
        fit_lock = {'kind': 'latent_shadow_splice_2022_fit_v1', 'status': 'locked', 'year': 2022,
                    'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'selection_lock_sha256': SELECTION_SHA,
                    'protocol_sha256': args.protocol_sha256, 'train_ids': report['train_ids'],
                    'train_dates': report['train_dates'], 'maximum_fitted_label_year': 2021,
                    'artifacts': {name: digest(out / name) for name in
                                  ['models.json', 'permutation_indices.npy', 'training_distributions.npz']}}
        (out / 'fit_lock.json').write_text(json.dumps(fit_lock, indent=2))
        report['fit_lock_sha256'] = digest(out / 'fit_lock.json')
        print('FIT_LOCK', report['fit_lock_sha256'], flush=True)
        validation_x, validation_paths = [], []
        for i, case_id in enumerate(validation['Run No']):
            index = index_of[int(case_id)]
            values = inputs.grid_case(index)
            np.testing.assert_array_equal(values['old_features'], saved_features[index])
            np.testing.assert_array_equal(values['rewards'][:, 8, :].T, saved_rewards[index])
            np.savez_compressed(out / f'validation_input_{case_id:04d}.npz', **values, case_id=np.asarray(case_id))
            validation_x.append(values['features']); validation_paths.append(values['paths'])
            if (i + 1) % 100 == 0:
                print('VALIDATION_INPUTS', i + 1, len(validation), flush=True)
        validation_x, validation_paths = np.stack(validation_x), np.stack(validation_paths)
        distributions = {name: predict_all(validation_x, family) for name, family in families.items()}
        for name in distributions:
            np.testing.assert_array_equal(distributions[name], predict_all(validation_x, restored[name]))
        raw = {name: np.stack([schedule_mixture(pi[:, j], validation_paths[:, j]) for j in range(3)], axis=2).reshape(-1, 3)
               for name, pi in distributions.items()}
        np.savez_compressed(out / 'raw_validation_2022.npz', **raw,
                            case_ids=validation['Run No'].to_numpy(), units=np.asarray(TARGET_UNITS))
        np.savez_compressed(out / 'validation_distributions_2022.npz', **distributions,
                            case_ids=validation['Run No'].to_numpy(), units=np.asarray(TARGET_UNITS))
        prediction_lock = {'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                           'protocol_sha256': args.protocol_sha256, 'fit_lock_sha256': report['fit_lock_sha256'],
                           'artifacts': {p.name: digest(p) for p in out.glob('validation_input_*.npz')}}
        prediction_lock['artifacts'].update({name: digest(out / name) for name in
                                             ['raw_validation_2022.npz', 'validation_distributions_2022.npz']})
        (out / 'prediction_lock.json').write_text(json.dumps(prediction_lock, indent=2))
        report['prediction_lock_sha256'] = digest(out / 'prediction_lock.json')
        report['validation_label_access_utc'] = datetime.now(timezone.utc).isoformat()
        with np.load(ANCHOR / 'selected_validation_2022.npz', allow_pickle=False) as z:
            anchor, y, ids, units = z['predictions'], z['labels'], z['case_ids'], z['units'].tolist()
        if (ids.tolist() != validation['Run No'].tolist() or units != parent['anchor_units'] or
                anchor.shape != y.shape or anchor.shape != (365 * 168, 14)):
            raise ValueError('2022 anchor labels/IDs/schema mismatch')
        weight = protocol['selected_weight']
        selected = splice(anchor, raw['real'], units, weight)
        scores = {name: float(roc_auc_score(y.ravel(), splice(anchor, p, units, weight).ravel()))
                  for name, p in raw.items()}
        anchor_score = float(roc_auc_score(y.ravel(), anchor.ravel()))
        passed = scores['real'] > max(anchor_score, scores['constant'], scores['permuted'])
        np.savez_compressed(out / 'selected_validation_2022.npz', predictions=selected, labels=y,
                            case_ids=ids, units=np.asarray(units))
        report.update(status='complete', selected_micro_auc=scores['real'], anchor_micro_auc=anchor_score,
                      control_micro_auc={name: scores[name] for name in ['constant', 'permuted']},
                      accepted=bool(passed), validation_ids=ids.tolist(), units=units,
                      raw_unit_auc={name: {u: float(roc_auc_score(y[:, units.index(u)], p[:, j]))
                                            for j, u in enumerate(TARGET_UNITS)} for name, p in raw.items()})
        print(json.dumps({key: report[key] for key in ['status', 'selected_weight', 'selected_micro_auc',
                                                     'anchor_micro_auc', 'control_micro_auc', 'accepted']}, indent=2), flush=True)
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
