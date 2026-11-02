"""Wait for fixed LP capacity folds and lock a scalar blend on 2021 only."""
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
import numpy as np
from sklearn.metrics import roc_auc_score

from lp_candidate_compare import ROOT, digest, verify
sys.path.insert(0, str(ROOT))
from sintef_project.postprocess import project_vinje

SOURCES = ['lgb-normalized-lp-001', 'lgb-normalized-lp-context-001', 'lgb-normalized-lp-large-001']
GRID = [0., .25, .5, .75, 1.]


def name(weight: float) -> str:
    """Give each predeclared scalar candidate a stable name."""
    return 'context75_base25' if weight == 0 else f'lp_large_w{weight:g}'


def weights(weight: float) -> list[float]:
    """Expand the scalar candidate to the three raw source probabilities."""
    return [.25 * (1 - weight), .75 * (1 - weight), weight]


def await_fold(year: int, *, complete: bool) -> dict:
    """Poll only metadata, sleeping15s; never open another year's predictions."""
    directory = ROOT / 'artifacts' / SOURCES[2]
    while True:
        try:
            manifest = json.loads((directory / 'manifest.json').read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            manifest = {}
        if manifest.get('status') == 'failed':
            raise ValueError('Large LP training failed: ' + (manifest.get('error') or 'no error detail').splitlines()[-1])
        if (str(year) in manifest.get('folds', {}) and (directory / f'validation_{year}.npz').is_file()
                and (not complete or manifest.get('status') == 'complete')):
            return manifest
        time.sleep(15)


def compose(raw: list[np.ndarray], weight: float, units: list[str]) -> np.ndarray:
    """Mix all14 raw unit probabilities, then apply one fixed Vinje projection."""
    prediction = np.zeros_like(raw[0], dtype=np.float64)
    for value, coefficient in zip(raw, weights(weight)):
        prediction += coefficient * value
    return project_vinje(prediction, units)


def main() -> None:
    """Freeze2021 choice during training, then accept or reject it on2022."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', default='lp-large-blend-001')
    args = parser.parse_args()
    out = ROOT / 'artifacts' / args.run_id
    out.mkdir(exist_ok=False)
    started = time.time()
    report = {'run_id': args.run_id, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'command': shlex.join([sys.executable, *sys.argv]), 'source_order': SOURCES,
              'scalar_weight_grid': GRID, 'candidate_weights': {name(w): weights(w) for w in GRID},
              'selection_rule': 'Maximize2021 projected whole-matrix micro AUC; exactties favor lower large-model weight. Freeze before2022 prediction decoding. If selected2022 fails strict improvement over fixed prior, retain prior and set accepted:false; no alternate2022 weight selection.',
              'postprocessing': 'Convex raw probability blend followed by fixed Vinje projection; no smoothing',
              'prior': 'context75_base25 from lp-candidate-comparison-001, verified identically at scalarweight0',
              'reuse_bias': 'Both chronological folds have been reused for development;2022 is an acceptance check, not an untouched holdout.',
              'training': 'none; separate fixed model training is recorded in source manifests',
              'environment': {'python': sys.version, 'platform': platform.platform(), 'threads_cap': 2,
                              'packages': {package: importlib.metadata.version(package) for package in ['numpy', 'scikit-learn']}},
              'sources': {}, 'source_code': {}, 'data': {}, 'folds': {}, 'status': 'waiting_for_2021',
              'kaggle': {'submission_id': None, 'score': None, 'rank': None, 'observation_time': None}}
    try:
        for relative in ['scripts/compare_lp_large.py', 'scripts/lp_candidate_compare.py',
                         'sintef_project/postprocess.py', 'docs/lp_large_protocol.md']:
            target = out / 'source' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)
            report['source_code'][relative] = digest(target)
        prior_dir = ROOT / 'artifacts/lp-candidate-comparison-001'
        prior_manifest = json.loads((prior_dir / 'manifest.json').read_text())
        if prior_manifest.get('status') != 'complete' or prior_manifest['accepted_candidate'] != 'context75_base25':
            raise ValueError('Predeclared prior candidate is unavailable')
        verify(prior_dir, prior_manifest['artifacts'])
        verify(prior_dir / 'source', prior_manifest['source'])
        verify(ROOT, prior_manifest['inputs'])
        report['sources'][str((prior_dir / 'manifest.json').relative_to(ROOT))] = digest(prior_dir / 'manifest.json')
        frozen = []
        for run in SOURCES[:2]:
            directory = ROOT / 'artifacts' / run
            manifest = json.loads((directory / 'manifest.json').read_text())
            if manifest.get('status') != 'complete':
                raise ValueError('Prior component is incomplete')
            verify(directory, manifest['artifacts'])
            verify(directory / 'source', manifest['source'])
            frozen.append(manifest)
            report['sources'][str((directory / 'manifest.json').relative_to(ROOT))] = digest(directory / 'manifest.json')
        report['data'] = frozen[0]['data']
        if frozen[1]['data'] != report['data']:
            raise ValueError('Prior components use different data')
        verify(ROOT, report['data'])
        (out / 'manifest.json').write_text(json.dumps(report, indent=2))
        with (ROOT / 'docs/experiments.md').open('a') as handle:
            handle.write(f'\n## {args.run_id} queued — {report["timestamp_utc"]}\n\nCommand: `{report["command"]}`. '
                         'Predeclared all-unit large-LP weights0/.25/.5/.75/1 versus fixed context75/base25 prior, '
                         'one Vinje projection after raw blending. Waiting for2021 source fold; no2022 prediction file decoded. '
                         f'Protocol/source snapshot: `artifacts/{args.run_id}/manifest.json`.\n')
        chosen_cache, prior_cache = {}, {}
        selected_weight = None
        first_large = None
        for year in [2021, 2022]:
            large = await_fold(year, complete=year == 2022)
            large_dir = ROOT / 'artifacts' / SOURCES[2]
            expected = {'feature_set': 'normalized-lp', 'trees': 700, 'leaves': 63,
                        'learning_rate': .04, 'min_child_samples': 100, 'sample_fraction': .7,
                        'seed': 4173, 'threads': 2, 'years': [2021, 2022], 'final': False, 'inference_from': None}
            if any(large['arguments'][key] != value for key, value in expected.items()):
                raise ValueError('Large model deviated from the predeclared design')
            expected_parameters = frozen[0]['model_parameters'].copy()
            expected_parameters.update(n_estimators=700, num_leaves=63, learning_rate=.04, n_jobs=2)
            if large['model_parameters'] != expected_parameters:
                raise ValueError('Large model changed an undeclared model parameter')
            if (large['data'] != report['data'] or large['features'] != frozen[0]['features']
                    or large['feature_spec'] != frozen[0]['feature_spec'] or len(large['features']) != 395):
                raise ValueError('Large model features/data differ from LP-v1 baseline')
            verify(large_dir / 'source', large['source'])
            if year == 2021:
                first_large = large
                snapshot = out / 'large_manifest_at_selection.json'
                snapshot.write_text(json.dumps(large, indent=2))
                report['large_manifest_snapshot_sha256'] = digest(snapshot)
            else:
                if not (out / 'selection_2021.json').is_file():
                    raise ValueError('Missing2021 lock')
                for key in ['data', 'source', 'features', 'feature_spec', 'arguments', 'model_parameters']:
                    if large[key] != first_large[key]:
                        raise ValueError(f'Large model changed after2021 selection: {key}')
                if large['folds']['2021'] != first_large['folds']['2021']:
                    raise ValueError('2021 fold metadata changed')
                verify(large_dir, large['artifacts'])
                verify(ROOT, report['sources'])
                report['sources'][str((large_dir / 'manifest.json').relative_to(ROOT))] = digest(large_dir / 'manifest.json')
            raw, y, ids, units = [], None, None, None
            for run in [*SOURCES, 'lp-candidate-comparison-001']:
                directory = ROOT / 'artifacts' / run
                path = directory / f'validation_{year}.npz'
                report['sources'][str(path.relative_to(ROOT))] = digest(path)
                with np.load(path, allow_pickle=False) as z:
                    if y is None:
                        y, ids, units = z['labels'], z['case_ids'], z['units'].tolist()
                    elif not (np.array_equal(y, z['labels']) and np.array_equal(ids, z['case_ids']) and units == z['units'].tolist()):
                        raise ValueError(f'Misaligned labels/IDs/units: {run}')
                    p = z['predictions'].astype(np.float64)
                    if p.shape != y.shape or p.shape != (len(ids) * 168, len(units)) or not np.isfinite(p).all() or p.min() < 0 or p.max() > 1:
                        raise ValueError('Invalid probabilities')
                    if run == 'lp-candidate-comparison-001':
                        supplied_prior = p
                    else:
                        raw.append(p)
            prior = compose(raw, 0., units)
            if not np.array_equal(prior, supplied_prior):
                raise ValueError('Zero-weight prior does not reconstruct the accepted candidate exactly')
            if large['folds'][str(year)]['train_ids'] != frozen[0]['folds'][str(year)]['train_ids']:
                raise ValueError('Large-model eligible training cases differ from baseline')
            for path in [large_dir / f'train_sample_{year}.npy',
                         *[large_dir / f'fold{year}_{unit}.txt' for unit in units]]:
                report['sources'][str(path.relative_to(ROOT))] = digest(path)
            fold = {'case_ids': ids.tolist(), 'units': units, 'scores': {},
                    'large_raw_micro_auc': float(roc_auc_score(y.ravel(), raw[2].ravel())),
                    'baseline_raw_micro_auc': float(roc_auc_score(y.ravel(), raw[0].ravel())),
                    'large_projected_micro_auc': float(roc_auc_score(y.ravel(), project_vinje(raw[2], units).ravel()))}
            report['folds'][str(year)] = fold
            for weight in (GRID if year == 2021 else sorted(set([0., selected_weight]))):
                fold['scores'][name(weight)] = float(roc_auc_score(y.ravel(), compose(raw, weight, units).ravel()))
            if year == 2021:
                selected_weight = max(GRID, key=lambda weight: fold['scores'][name(weight)])
                report.update(selected_scalar_weight=selected_weight, selected_candidate=name(selected_weight),
                              selected_weights=weights(selected_weight), selection_locked_utc=datetime.now(timezone.utc).isoformat())
            chosen = compose(raw, selected_weight, units)
            fold['selected_micro_auc'] = fold['scores'][name(selected_weight)]
            fold['prior_micro_auc'] = fold['scores']['context75_base25']
            fold['selected_unit_auc'] = {unit: float(roc_auc_score(y[:, j], chosen[:, j])) for j, unit in enumerate(units)}
            if year == 2021:
                (out / 'selection_2021.json').write_text(json.dumps(report, indent=2))
                report['status'] = 'waiting_for_2022'
            chosen_cache[year] = (chosen, y, ids, units)
            prior_cache[year] = (prior, y, ids, units)
            np.savez_compressed(out / f'selected_validation_{year}.npz', predictions=chosen, labels=y, case_ids=ids, units=units)
            (out / 'manifest.json').write_text(json.dumps(report, indent=2))
            print('LOCKED', year, 'large_weight', selected_weight, 'micro_auc', fold['selected_micro_auc'], 'prior', fold['prior_micro_auc'], flush=True)
        accepted = selected_weight == 0 or report['folds']['2022']['selected_micro_auc'] > report['folds']['2022']['prior_micro_auc']
        report['accepted'] = bool(accepted)
        report['accepted_candidate'] = report['selected_candidate'] if accepted else 'context75_base25'
        report['acceptance_reason'] = 'Retain2021 choice: prior itself won or locked mixture improves2022.' if accepted else 'Locked2021 mixture failed2022 improvement; retain fixed prior without selecting another2022 weight.'
        for year in [2021, 2022]:
            p, y, ids, units = chosen_cache[year] if accepted else prior_cache[year]
            np.savez_compressed(out / f'validation_{year}.npz', predictions=p, labels=y, case_ids=ids, units=units)
            report['folds'][str(year)]['accepted_micro_auc'] = report['folds'][str(year)]['selected_micro_auc' if accepted else 'prior_micro_auc']
        report['meets_transfer_gate'] = all(fold['accepted_micro_auc'] >= .998 for fold in report['folds'].values())
        report['status'] = 'complete'
        print('COMPLETE', 'accepted', accepted, 'candidate', report['accepted_candidate'], flush=True)
    except BaseException:
        report['status'] = 'failed'
        report['error'] = traceback.format_exc()
        raise
    finally:
        report['runtime_seconds_including_wait'] = time.time() - started
        report['artifacts'] = {str(path.relative_to(out)): digest(path) for path in out.rglob('*') if path.is_file() and path.name != 'manifest.json'}
        (out / 'manifest.json').write_text(json.dumps(report, indent=2))
        with (ROOT / 'docs/experiments.md').open('a') as handle:
            handle.write(f'\n## {args.run_id} result — {datetime.now(timezone.utc).isoformat()}\n\n'
                         f'Status: {report["status"]}. Command: `{report["command"]}`. '
                         f'Selected: {report.get("selected_candidate")}; accepted: {report.get("accepted")}; '
                         f'accepted candidate: {report.get("accepted_candidate")}. '
                         'No fitting or Kaggle submission. Full provenance, scalar grid,2021 lock, raw/base comparisons '
                         f'and accepted/selected predictions: `artifacts/{args.run_id}/manifest.json`.\n')
            for year, fold in report['folds'].items():
                handle.write(f'\n- {year}: selected {fold.get("selected_micro_auc")}; prior {fold.get("prior_micro_auc")}; '
                             f'large raw {fold["large_raw_micro_auc"]}; baseline raw {fold["baseline_raw_micro_auc"]}.\n')
            if 'error' in report:
                handle.write(f'\nFailure: `{report["error"].splitlines()[-1]}`.\n')


if __name__ == '__main__':
    main()
