"""Fit one frozen nonlinear-PQ specialist fold and lock selection before 2022."""
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
from unittest.mock import patch

for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[variable] = '2'

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_cuda_tcn import read_targets_through_year
from sintef_project.features import build_features
from sintef_project.physics import add_physics_features
from sintef_project.context_features import add_context_features
from sintef_project.lp_features import add_lp_features
from sintef_project.lp_context import add_lp_context_features
from sintef_project.normalized import add_normalized_features
from sintef_project.lp_joint_plant import add_lp_joint_features
from sintef_project.nonlinear_pq import TARGET_UNITS, COLUMNS, add_nonlinear_pq_features, digest
from sintef_project.postprocess import project_vinje

WEIGHTS = [0., .1, .25, .5]
BASE = ROOT / 'artifacts/lgb-normalized-lp-context-joint-001'
ANCHOR = ROOT / 'artifacts/lp-context-joint-comparison-001'
CACHE = ROOT / 'artifacts/nonlinear-pq-features-001'
DEPENDENCIES = ['sintef_project/' + name + '.py' for name in [
    'features', 'physics', 'context_features', 'lp_features', 'lp_context', 'normalized',
    'lp_joint_plant', 'lp_head_update', 'lp_shared_loss_update', 'nonlinear_pq', 'postprocess']]
SOURCES = [*DEPENDENCIES, 'scripts/train_nonlinear_pq_specialists.py',
           'scripts/train_cuda_tcn.py', 'tests/test_nonlinear_pq.py',
           'tests/test_nonlinear_pq_protocol.py', 'docs/nonlinear_pq_protocol.md']


def validate_lock(path: Path, sha: str, protocol_sha: str) -> dict:
    """Require exactly the completed eligible 2021 choice before 2022 work."""
    if not sha or digest(path) != sha:
        raise ValueError('Explicit 2021 selection-lock SHA mismatch')
    lock = json.loads(path.read_text())
    if lock['version'] != 'nonlinear_pq_selection_v1' or lock['protocol_sha256'] != protocol_sha:
        raise ValueError('Selection-lock schema or protocol changed')
    if lock['year'] != 2021 or lock['candidate_weights'] != WEIGHTS or lock['selected_weight'] not in WEIGHTS:
        raise ValueError('Selection year or grid changed')
    scores = lock['scores_2021']
    if set(scores) != {str(weight) for weight in WEIGHTS} or not all(
            isinstance(value, (float, int)) and np.isfinite(value) and 0 <= value <= 1
            for value in scores.values()):
        raise ValueError('2021 grid scores missing or invalid')
    chosen = max(WEIGHTS, key=lambda weight: scores[str(weight)])
    if lock['selected_weight'] != chosen or lock['selected_micro_auc'] != scores[str(chosen)]:
        raise ValueError('2021 choice is not the frozen stable grid maximum')
    for key in ['anchor_sha256', 'control_sha256', 'raw_prediction_sha256']:
        value = lock[key]
        if not isinstance(value, str) or len(value) != 64 or any(char not in '0123456789abcdef' for char in value):
            raise ValueError('Missing 2021 anchor/control/raw prediction provenance')
    if not lock['eligible_for_2022'] or not 0 < lock['selected_weight'] <= .5:
        raise ValueError('2021 did not beat anchor and matched control; no 2022 fitting')
    if lock['selected_micro_auc'] <= max(lock['anchor_micro_auc'], lock['control_micro_auc']):
        raise ValueError('2021 score evidence does not satisfy the frozen gate')
    return lock


def compose(anchor: np.ndarray, raw: np.ndarray, indices: list[int], weight: float,
            units: list[str]) -> np.ndarray:
    """Splice only target units, preserving every other anchor column."""
    result = anchor.astype(np.float64).copy()
    result[:, indices] = (1. - weight) * result[:, indices] + weight * raw
    return project_vinje(result, units)


def build_608(year: int, baseline: dict) -> tuple:
    """Rebuild exact frozen608 prefix while redacting all later target rows."""
    original = pd.read_csv
    target = (ROOT / 'sintef_project/data/kernel/Unit_commitment_decisions.csv').resolve()

    def guarded(path, *args, **kwargs):
        if isinstance(path, (str, Path)) and Path(path).resolve() == target:
            if kwargs.get('usecols') == ['Run No', 'starttime']:
                return original(path, *args, **kwargs)
            if args or kwargs:
                raise ValueError('Unexpected target-reader signature')
            return read_targets_through_year(target, year, original)
        return original(path, *args, **kwargs)

    with patch.object(pd, 'read_csv', guarded):
        x, labels, train, mapping, meta = build_features(ROOT)
        for name, transform in [('physics', add_physics_features), ('context', add_context_features),
                                ('lp', add_lp_features), ('lp_context', add_lp_context_features),
                                ('normalized', add_normalized_features), ('lp_joint', add_lp_joint_features)]:
            x, specification = transform(x, ROOT)
            meta['feature_spec'][name] = specification
    if x.columns.tolist() != baseline['features'] or meta['feature_spec'] != baseline['feature_spec']:
        raise ValueError('608-feature prefix formulas, order or provenance changed')
    return x, labels, train, mapping, meta


def freeze_protocol(path: Path) -> dict:
    """Freeze full dependency hashes, fixed parameters and acceptance before fitting."""
    baseline = json.loads((BASE / 'manifest.json').read_text())
    anchor = json.loads((ANCHOR / 'analysis.json').read_text())
    cache = json.loads((CACHE / 'manifest.json').read_text())
    if not (baseline['status'] == cache['status'] == anchor['status'] == 'complete' and anchor['accepted']):
        raise ValueError('Required immutable sources incomplete')
    for name in DEPENDENCIES:
        if name in baseline['source'] and digest(ROOT / name) != baseline['source'][name]:
            raise ValueError(f'Frozen 608-feature dependency changed: {name}')
    protocol = {
        'version': 'nonlinear_pq_specialists_v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'source': {name: digest(ROOT / name) for name in SOURCES}, 'data': baseline['data'],
        'base_manifest_sha256': digest(BASE / 'manifest.json'), 'anchor_analysis_sha256': digest(ANCHOR / 'analysis.json'),
        'cache_manifest_sha256': digest(CACHE / 'manifest.json'), 'cache_features_sha256': digest(CACHE / 'features.npz'),
        'targets': TARGET_UNITS, 'append_columns': COLUMNS, 'prefix_columns': baseline['features'],
        'model_parameters': {**baseline['model_parameters'], 'n_jobs': 2}, 'seed': 4173,
        'candidate_weights': WEIGHTS, 'anchor_weights': anchor['selected_weights'],
        'sample_sha256': {str(year): digest(BASE / f'train_sample_{year}.npy') for year in [2021, 2022]},
        'selection': 'Select2021maximum microAUC,ties lowerweight; require strict improvement against anchor and saved608 same-weight control. Freeze lock before2022. Apply only frozenweight2022; same strict gate.',
        'training': 'Three specialists only; exact baseline40%samples; six-daypurge; fixed350trees/noearlystopping; target rows after requestedyear are redacted.',
    }
    with path.open('x') as stream:
        json.dump(protocol, stream, indent=2)
    return protocol


def main() -> None:
    """Run one requested year; preserve every failure and immutable chosen weight."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, choices=[2021, 2022], required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--protocol', type=Path, default=ROOT / 'docs/nonlinear_pq_protocol.json')
    parser.add_argument('--selection-lock', type=Path)
    parser.add_argument('--selection-lock-sha256')
    args = parser.parse_args()
    out = ROOT / 'artifacts' / args.run_id
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    report = {'run_id': args.run_id, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'command': shlex.join([sys.executable, *sys.argv]), 'year': args.year, 'status': 'initializing',
        'environment': {'python': sys.version, 'platform': platform.platform(), 'machine': platform.machine(),
                        'packages': {n: importlib.metadata.version(n) for n in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'lightgbm', 'pyyaml']}},
        'seed': 4173, 'kaggle_submission': None, 'early_stopping': None, 'models': {}}
    try:
        if not args.protocol.exists():
            if args.year != 2021:
                raise ValueError('2021 must freeze the training protocol first')
            protocol = freeze_protocol(args.protocol)
        else:
            protocol = json.loads(args.protocol.read_text())
        protocol_sha = digest(args.protocol)
        report['protocol_sha256'] = protocol_sha
        for name, sha in protocol['source'].items():
            if digest(ROOT / name) != sha:
                raise ValueError(f'Frozen protocol source changed: {name}')
            target = out / 'source' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        for name, sha in protocol['data'].items():
            if digest(ROOT / name) != sha:
                raise ValueError(f'Raw data changed: {name}')
        for path, key in [(BASE / 'manifest.json', 'base_manifest_sha256'),
                          (ANCHOR / 'analysis.json', 'anchor_analysis_sha256'),
                          (CACHE / 'manifest.json', 'cache_manifest_sha256'),
                          (CACHE / 'features.npz', 'cache_features_sha256')]:
            if digest(path) != protocol[key]:
                raise ValueError(f'Frozen artifact changed: {path}')
        lock = None
        if args.year == 2022:
            if args.selection_lock is None:
                raise ValueError('2022 requires the local2021 lock and explicitSHA')
            lock = validate_lock(args.selection_lock, args.selection_lock_sha256, protocol_sha)
            report['selection_lock_sha256'] = digest(args.selection_lock)
        elif args.selection_lock is not None:
            raise ValueError('2021 cannot accept an existing weight choice')
        baseline = json.loads((BASE / 'manifest.json').read_text())
        anchor_manifest = json.loads((ANCHOR / 'analysis.json').read_text())
        x, labels, train, mapping, meta = build_608(args.year, baseline)
        cache_manifest = json.loads((CACHE / 'manifest.json').read_text())
        all_cases = pd.concat([train, mapping], ignore_index=True)
        if all_cases['Run No'].tolist() != cache_manifest['case_ids']:
            raise ValueError('Incoming608 frame and nonlinear cache case IDs/order differ')
        if pd.to_datetime(all_cases.starttime, utc=True).astype(str).tolist() != cache_manifest['case_dates']:
            raise ValueError('Incoming608 frame and nonlinear cache dates/order differ')
        x, nonlinear = add_nonlinear_pq_features(x, ROOT, CACHE)
        if x.columns.tolist() != protocol['prefix_columns'] + protocol['append_columns'] or x.shape[1] != 614:
            raise ValueError('Expected exact614 feature schema')
        report.update(features=x.columns.tolist(), feature_spec={**meta['feature_spec'], 'nonlinear_pq': nonlinear},
                      data=protocol['data'], model_parameters=protocol['model_parameters'])
        print('FEATURES', x.shape, flush=True)
        sample_path = BASE / f'train_sample_{args.year}.npy'
        if digest(sample_path) != protocol['sample_sha256'][str(args.year)]:
            raise ValueError('Exact training sample changed')
        sample = np.load(sample_path, allow_pickle=False)
        np.save(out / f'train_sample_{args.year}.npy', sample)
        dates = pd.to_datetime(train.starttime, utc=True)
        begin, end = pd.Timestamp(f'{args.year}-01-01', tz='UTC'), pd.Timestamp(f'{args.year+1}-01-01', tz='UTC')
        tr = np.flatnonzero(dates < begin - pd.Timedelta(days=6))
        va_cases = np.flatnonzero((dates >= begin) & (dates < end))
        inherited = baseline['folds'][str(args.year)]
        if train.iloc[tr]['Run No'].tolist() != inherited['train_ids'] or train.iloc[va_cases]['Run No'].tolist() != inherited['validation_ids']:
            raise ValueError('Chronological case IDs changed')
        if not np.isin(sample // 168, tr).all() or dates.iloc[tr].max() + pd.Timedelta(hours=167) >= begin:
            raise ValueError('Training purge/sample failed')
        va = (va_cases[:, None] * 168 + np.arange(168)).ravel()
        units = meta['units']
        indices = [units.index(unit) for unit in TARGET_UNITS]
        y = np.column_stack([labels[unit] for unit in units]).astype(np.uint8)
        del labels
        xt, xv = x.iloc[sample], x.iloc[va]
        del x
        raw = np.empty((len(va), 3), dtype=np.float64)
        report['fold'] = inherited
        (out / 'manifest.json').write_text(json.dumps(report, indent=2))
        for j, unit in enumerate(TARGET_UNITS):
            tic = time.monotonic()
            model = lgb.LGBMClassifier(**protocol['model_parameters'])
            model.fit(xt, y[sample, indices[j]])
            raw[:, j] = model.predict_proba(xv)[:, 1]
            path = out / f'fold{args.year}_{unit}.txt'
            model.booster_.save_model(path)
            replay = lgb.Booster(model_file=str(path)).predict(xv, num_threads=2)
            np.testing.assert_array_equal(raw[:, j], replay)
            report['models'][unit] = {'checkpoint_sha256': digest(path), 'seconds': time.monotonic() - tic,
                'unit_auc': float(roc_auc_score(y[va, indices[j]], raw[:, j])), 'exact_checkpoint_replay': True}
            (out / 'manifest.json').write_text(json.dumps(report, indent=2))
            print('MODEL', args.year, unit, report['models'][unit], flush=True)
            del model
        ids = train.iloc[va_cases]['Run No'].to_numpy()
        np.savez_compressed(out / f'raw_validation_{args.year}.npz', predictions=raw,
                            labels=y[va], case_ids=ids, units=np.asarray(units), targets=np.asarray(TARGET_UNITS))
        control_path, anchor_path = BASE / f'validation_{args.year}.npz', ANCHOR / f'selected_validation_{args.year}.npz'
        if digest(control_path) != baseline['artifacts'][control_path.name] or digest(anchor_path) != anchor_manifest['artifacts'][anchor_path.name]:
            raise ValueError('Frozen control/anchor array changed')
        with np.load(control_path, allow_pickle=False) as saved:
            control = saved['predictions'][:, indices].astype(np.float64)
            np.testing.assert_array_equal(saved['labels'], y[va])
            np.testing.assert_array_equal(saved['case_ids'], ids)
            if saved['units'].tolist() != units:
                raise ValueError('Control unit order changed')
        with np.load(anchor_path, allow_pickle=False) as saved:
            anchor = saved['predictions'].astype(np.float64)
            np.testing.assert_array_equal(saved['labels'], y[va])
            np.testing.assert_array_equal(saved['case_ids'], ids)
            if saved['units'].tolist() != units:
                raise ValueError('Anchor unit order changed')
        scores = {str(w): float(roc_auc_score(y[va].ravel(), compose(anchor, raw, indices, w, units).ravel()))
                  for w in (WEIGHTS if args.year == 2021 else [lock['selected_weight']])}
        weight = max(WEIGHTS, key=lambda w: scores[str(w)]) if lock is None else lock['selected_weight']
        selected = compose(anchor, raw, indices, weight, units)
        control_auc = float(roc_auc_score(y[va].ravel(), compose(anchor, control, indices, weight, units).ravel()))
        anchor_auc = float(roc_auc_score(y[va].ravel(), anchor.ravel()))
        eligible = bool(weight > 0 and scores[str(weight)] > max(anchor_auc, control_auc))
        report.update(scores=scores, selected_weight=weight, selected_micro_auc=scores[str(weight)],
                      control_micro_auc=control_auc, anchor_micro_auc=anchor_auc,
                      beats_anchor_and_control=eligible, status='complete')
        np.savez_compressed(out / f'selected_validation_{args.year}.npz', predictions=selected,
                            labels=y[va], case_ids=ids, units=np.asarray(units))
        if args.year == 2021:
            lock = {'version': 'nonlinear_pq_selection_v1', 'year': 2021,
                    'locked_utc': datetime.now(timezone.utc).isoformat(), 'protocol_sha256': protocol_sha,
                    'candidate_weights': WEIGHTS, 'selected_weight': weight, 'scores_2021': scores,
                    'selected_micro_auc': scores[str(weight)], 'anchor_micro_auc': anchor_auc,
                    'control_micro_auc': control_auc, 'eligible_for_2022': eligible,
                    'anchor_sha256': digest(anchor_path), 'control_sha256': digest(control_path),
                    'raw_prediction_sha256': digest(out / 'raw_validation_2021.npz')}
            with (out / 'selection_2021.json').open('x') as stream:
                json.dump(lock, stream, indent=2)
            report['selection_lock_sha256'] = digest(out / 'selection_2021.json')
        else:
            report['accepted'] = eligible
            report['meets_transfer_gate'] = min(lock['selected_micro_auc'], scores[str(weight)]) >= .998
        print(json.dumps({k: report.get(k) for k in ['status', 'selected_weight', 'selected_micro_auc', 'control_micro_auc', 'anchor_micro_auc', 'beats_anchor_and_control']}), flush=True)
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        report['runtime_seconds'] = time.monotonic() - started
        if args.protocol.exists():
            shutil.copy2(args.protocol, out / 'protocol.json')
        report['artifacts'] = {str(p.relative_to(out)): digest(p) for p in out.rglob('*') if p.is_file() and p.name != 'manifest.json'}
        (out / 'manifest.json').write_text(json.dumps(report, indent=2))
        with (ROOT / 'docs/experiments.md').open('a') as stream:
            stream.write(f'\n## {args.run_id} — {report["timestamp_utc"]}\n\nStatus:{report["status"]};year{args.year};614features,three350-tree specialists,exact40%row samples,six-daypurge,seed4173,two threads,noearlystopping. Command:`{report["command"]}`. Frozen global weight{report.get("selected_weight")};selectedAUC{report.get("selected_micro_auc")},anchor{report.get("anchor_micro_auc")},matchedcontrol{report.get("control_micro_auc")}. Beatsboth:{report.get("beats_anchor_and_control")}. Runtime{report["runtime_seconds"]:.3f}s. Fullrawdata/source/cache/model/environment/split/sample/protocol hashes and exactcheckpointreplay:`artifacts/{args.run_id}/manifest.json`. NoKaggle submission;reuseddevelopmentfold.\n')


if __name__ == '__main__':
    main()
