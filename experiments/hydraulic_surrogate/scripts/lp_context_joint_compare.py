"""Lock a fixed context-plus-joint blend on2021 before decoding2022 arrays."""
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
import sys
import time
import traceback

for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[variable] = '2'
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sintef_project.postprocess import project_vinje

WEIGHTS = [0., .25, .5, .75, 1.]
SOURCES = {'base': 'lgb-normalized-lp-001', 'context': 'lgb-normalized-lp-context-001',
           'joint': 'lgb-normalized-lp-joint-001',
           'combined': 'lgb-normalized-lp-context-joint-001'}
PRIOR = 'lp-joint-comparison-001'


def digest(path: Path) -> str:
    """Return a content SHA256 without interpreting array contents."""
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_new_json(path: Path, value: dict) -> None:
    """Create a protocol or selection lock exactly once."""
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)


def choose_weight(scores: dict[str, float]) -> float:
    """Maximize 2021 micro AUC; exact ties prefer the smaller combined weight."""
    return max(WEIGHTS, key=lambda weight: scores[str(weight)])


def blend(inputs: dict[str, np.ndarray], units: list[str], weight: float) -> np.ndarray:
    """Blend raw float64 probabilities, then apply the fixed Vinje projection."""
    result = np.zeros_like(inputs['base'], dtype=np.float64)
    for name, fraction in zip(['base', 'context', 'joint', 'combined'],
                              [.1875 * (1. - weight), .5625 * (1. - weight), .25 * (1. - weight), weight]):
        result += fraction * inputs[name]
    return project_vinje(result, units)


def verify_hashes(base: Path, hashes: dict[str, str]) -> None:
    """Verify every registered file, without decoding validation labels."""
    for relative, expected in hashes.items():
        path = base / relative
        if digest(path) != expected:
            raise ValueError(f'Content hash mismatch: {path}')


def verify_feature_schema(out: Path) -> dict:
    """Check feature order and unchanged formula/cache closures without a matrix."""
    manifests = {name: json.loads((ROOT / 'artifacts' / SOURCES[name] / 'manifest.json').read_text())
                 for name in ['context', 'joint', 'combined']}
    combined, context, joint = (manifests[name] for name in ['combined', 'context', 'joint'])
    context_modules = ['features', 'physics', 'context_features', 'lp_features', 'normalized', 'lp_context']
    joint_modules = ['lp_joint_plant', 'lp_head_update', 'lp_shared_loss_update', 'lp_features', 'features', 'physics']
    checks = {'shape_612696_by608': combined['shape'] == [612696, 608],
              'first521_names_exact_context': combined['features'][:521] == context['features'],
              'last87_names_exact_joint': combined['features'][521:] == [name for name in joint['features'] if name.startswith('lpjoint_')],
              'same_raw_data': combined['data'] == context['data'] == joint['data'],
              'same_context_feature_and_cache_spec': {k: v for k, v in combined['feature_spec'].items() if k != 'lp_joint'} == context['feature_spec'],
              'same_joint_feature_and_cache_spec': combined['feature_spec']['lp_joint'] == joint['feature_spec']['lp_joint'],
              'same_context_source_closure': all(combined['source'][f'sintef_project/{name}.py'] == context['source'][f'sintef_project/{name}.py'] for name in context_modules),
              'same_joint_source_closure': all(combined['source'][f'sintef_project/{name}.py'] == joint['source'][f'sintef_project/{name}.py'] for name in joint_modules)}
    if not all(checks.values()):
        raise ValueError(f'Combined feature schema/provenance changed: {checks}')
    snapshot = out / 'combined_manifest_at_declaration.json'
    snapshot.write_text(json.dumps(combined, indent=2))
    return {'checks': checks, 'snapshot_sha256': digest(snapshot), 'large_matrix_loaded': False,
            'value_preservation_evidence': 'Trainer constructs and normalizes the original521 columns with unchanged code/spec/cache, then appends87 dimensionless joint-cache columns. No direct large-matrix numeric comparison performed.'}


def load_year(year: int, inputs_record: dict) -> tuple:
    """Decode only the explicitly requested year's aligned prediction arrays."""
    predictions = {}
    labels = ids = units = None
    for name, run in SOURCES.items():
        path = ROOT / 'artifacts' / run / f'validation_{year}.npz'
        inputs_record[str(path.relative_to(ROOT))] = digest(path)
        with np.load(path, allow_pickle=False) as values:
            if labels is None:
                labels, ids, units = values['labels'], values['case_ids'], values['units'].tolist()
            elif not (np.array_equal(labels, values['labels']) and np.array_equal(ids, values['case_ids'])
                      and units == values['units'].tolist()):
                raise ValueError(f'Label/ID/unit alignment failure: {run}/{year}')
            p = values['predictions'].astype(np.float64)
            if (p.shape != labels.shape or p.shape != (len(ids) * 168, len(units))
                    or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any()):
                raise ValueError(f'Invalid prediction shape or probabilities: {run}/{year}')
            predictions[name] = p
    if not np.isin(labels, [0, 1]).all():
        raise ValueError('Nonbinary labels')
    prior_path = ROOT / 'artifacts' / PRIOR / f'selected_validation_{year}.npz'
    inputs_record[str(prior_path.relative_to(ROOT))] = digest(prior_path)
    with np.load(prior_path, allow_pickle=False) as prior:
        if not (np.array_equal(labels, prior['labels']) and np.array_equal(ids, prior['case_ids'])
                and units == prior['units'].tolist()
                and np.array_equal(blend(predictions, units, 0.), prior['predictions'])):
            raise ValueError('Declared current recipe differs from accepted comparison artifact')
    sample_path = ROOT / 'artifacts' / SOURCES['combined'] / f'train_sample_{year}.npy'
    sample = np.load(sample_path, allow_pickle=False)
    inputs_record[str(sample_path.relative_to(ROOT))] = digest(sample_path)
    for run in [SOURCES['base'], SOURCES['context'], SOURCES['joint']]:
        path = ROOT / 'artifacts' / run / f'train_sample_{year}.npy'
        inputs_record[str(path.relative_to(ROOT))] = digest(path)
        if not np.array_equal(sample, np.load(path, allow_pickle=False)):
            raise ValueError(f'Training sample mismatch: {run}/{year}')
    return predictions, labels, ids, units


def auc(y: np.ndarray, p: np.ndarray) -> float | None:
    """Score a slice, preserving undefined single-class AUC as null."""
    return float(roc_auc_score(y.ravel(), p.ravel())) if np.unique(y).size == 2 else None


def diagnostics(y: np.ndarray, p: np.ndarray, ids: np.ndarray, units: list[str]) -> dict:
    """Report per-unit, calendar-month and within-case lead-day micro AUC."""
    metadata = pd.read_csv(ROOT / 'sintef_project' / 'data' / 'kernel' / 'Unit_commitment_decisions.csv',
                           usecols=['Run No', 'starttime'])
    metadata = metadata.set_index('Run No')
    dates = pd.to_datetime(metadata.loc[ids, 'starttime'], utc=True)
    month = np.repeat(dates.dt.month.to_numpy(), 168)
    lead = np.tile(np.arange(168) // 24, len(ids))
    return {'micro_auc': auc(y, p),
            'unit_auc': {unit: auc(y[:, j], p[:, j]) for j, unit in enumerate(units)},
            'month_auc': {str(m): auc(y[month == m], p[month == m]) for m in range(1, 13)},
            'lead_day_auc': {str(d): auc(y[lead == d], p[lead == d]) for d in range(7)}}


def wait_for_year(year: int, deadline: float) -> dict:
    """Wait for an externally owned trainer; never decode predictions here."""
    run = ROOT / 'artifacts' / SOURCES['combined']
    while time.time() < deadline:
        try:
            manifest = json.loads((run / 'manifest.json').read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            manifest = {}
        if manifest.get('status') == 'failed':
            raise RuntimeError('Combined trainer failed; inspect its manifest')
        if str(year) in manifest.get('folds', {}) and (run / f'validation_{year}.npz').is_file():
            if year == 2021 or manifest.get('status') == 'complete':
                return manifest
        time.sleep(10)
    raise TimeoutError(f'Timeout waiting for combined {year} validation artifacts')


def main() -> None:
    """Create a protocol, select on 2021, then evaluate the immutable choice."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', default='lp-context-joint-comparison-001')
    parser.add_argument('--wait-seconds', type=int, default=7200)
    args = parser.parse_args()
    out = ROOT / 'artifacts' / args.run_id
    out.mkdir(exist_ok=False)
    start = time.time()
    report = {'run_id': args.run_id, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'command': shlex.join([sys.executable, *sys.argv]), 'source_runs': SOURCES,
              'source_order': list(SOURCES.values()),
              'prior': PRIOR, 'combined_weights': WEIGHTS,
              'formula': 'project_vinje(sum([.1875*(1-w)*base_raw,.5625*(1-w)*context_raw,.25*(1-w)*joint_raw,w*combined_raw])); promote sources to float64 and accumulate base,context,joint,combined into zeros in that order',
              'selection_rule': 'Maximize 2021 micro AUC; ties choose smaller w. Persist exclusive selection_2021.json before any 2022 NPZ is decoded. Accept nonzero w only if locked 2022 micro AUC strictly improves current accepted blend.',
              'validation_caveat': '2021 and 2022 are repeatedly reused chronological development folds, not fresh independent holdouts.',
              'training': 'None; external source runs use fixed 350-tree models, seed4173 and matched40% samples. No fitted calibration, smoothing or early stopping.',
              'environment': {'platform': platform.platform(), 'python': sys.version,
                              'packages': {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scikit-learn']}},
              'source': {}, 'inputs': {}, 'data': {}, 'folds': {},
              'kaggle': {'submission_id': None, 'score': None, 'rank': None, 'observation_time': None}}
    try:
        for relative in ['scripts/lp_context_joint_compare.py', 'sintef_project/postprocess.py', 'docs/lp_context_joint_comparison.md']:
            path = ROOT / relative
            report['source'][relative] = digest(path)
            target = out / 'source' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        report['feature_schema_verification'] = verify_feature_schema(out)
        write_new_json(out / 'protocol.json', report)
        report['protocol_sha256'] = digest(out / 'protocol.json')
        prior_manifest = ROOT / 'artifacts' / PRIOR / 'manifest.json'
        prior = json.loads(prior_manifest.read_text())
        if prior.get('status') != 'complete' or prior.get('accepted') is not True or prior.get('selected_weights') != [.1875, .5625, .25]:
            raise ValueError('Unexpected accepted prior')
        report['inputs'][str(prior_manifest.relative_to(ROOT))] = digest(prior_manifest)
        for name in ['base', 'context', 'joint']:
            directory = ROOT / 'artifacts' / SOURCES[name]
            manifest = json.loads((directory / 'manifest.json').read_text())
            if manifest.get('status') != 'complete':
                raise ValueError(f'Incomplete fixed source: {name}')
            verify_hashes(directory, manifest['artifacts'])
            verify_hashes(directory / 'source', manifest['source'])
            if report['data'] and report['data'] != manifest['data']:
                raise ValueError('Source raw data hashes disagree')
            report['data'] = manifest['data']
            report['inputs'][str((directory / 'manifest.json').relative_to(ROOT))] = digest(directory / 'manifest.json')
            shutil.copy2(directory / 'manifest.json', out / f'{name}_manifest.json')
        verify_hashes(ROOT, report['data'])
        for year in [2021, 2022]:
            combined_manifest = wait_for_year(year, start + args.wait_seconds)
            if combined_manifest['data'] != report['data']:
                raise ValueError('Combined raw data hashes differ')
            combined_path = ROOT / 'artifacts' / SOURCES['combined'] / 'manifest.json'
            snapshot = out / f'combined_manifest_at_{year}.json'
            snapshot.write_text(json.dumps(combined_manifest, indent=2))
            report[f'combined_manifest_at_{year}_sha256'] = digest(snapshot)
            predictions, y, ids, units = load_year(year, report['inputs'])
            if year == 2021:
                grid = {str(w): auc(y, blend(predictions, units, w)) for w in WEIGHTS}
                selected = choose_weight(grid)
                lock = {'run_id': args.run_id, 'locked_utc': datetime.now(timezone.utc).isoformat(),
                        'protocol_sha256': report['protocol_sha256'], 'selected_combined_weight': selected,
                        'source_order': list(SOURCES.values()),
                        'selected_weights': [.1875 * (1. - selected), .5625 * (1. - selected), .25 * (1. - selected), selected],
                        'selected_candidate': f'combined_weight_{selected:g}',
                        'source_code': dict(report['source']), 'sources': dict(report['inputs']),
                        'folds': {'2021': {'units': units, 'case_ids': ids.tolist(),
                                           'selected_micro_auc': grid[str(selected)]}},
                        'scores_2021': grid, 'inputs_2021': dict(report['inputs']),
                        'source': report['source'], 'combined_manifest_snapshot_sha256': digest(snapshot),
                        'case_ids': ids.tolist(), 'units': units,
                        'read_scope': 'Only2021 validation NPZ arrays decoded; no2022 prediction/label arrays read by this process.'}
                write_new_json(out / 'selection_2021.json', lock)
                report['selection_sha256'] = digest(out / 'selection_2021.json')
                report['selection_locked_utc'] = lock['locked_utc']
                report['selected_combined_weight'] = selected
                report['selected_weights'] = lock['selected_weights']
                report['selected_candidate'] = lock['selected_candidate']
                report['scores_2021'] = grid
                print('LOCKED 2021', selected, grid, flush=True)
            else:
                if digest(out / 'selection_2021.json') != report['selection_sha256']:
                    raise ValueError('Immutable selection lock changed')
                # No alternate 2022 grid is evaluated. Standalone models are fixed diagnostics only.
                verify_hashes(combined_path.parent, combined_manifest['artifacts'])
                verify_hashes(combined_path.parent / 'source', combined_manifest['source'])
                report['inputs'][str(combined_path.relative_to(ROOT))] = digest(combined_path)
            candidates = {'selected': blend(predictions, units, selected),
                          'current': blend(predictions, units, 0.),
                          'matched_base_raw': predictions['base'],
                          'matched_base_projected': project_vinje(predictions['base'], units),
                          'combined_raw': predictions['combined'],
                          'combined_projected': project_vinje(predictions['combined'], units)}
            report['folds'][str(year)] = {'case_ids': ids.tolist(), 'units': units,
                                         'selected_micro_auc': auc(y, candidates['selected']),
                                         'split': combined_manifest['folds'][str(year)],
                                         'diagnostics': {name: diagnostics(y, p, ids, units) for name, p in candidates.items()}}
            np.savez_compressed(out / f'selected_validation_{year}.npz', predictions=candidates['selected'], labels=y, case_ids=ids, units=units)
            print('SCORED', year, {name: value['micro_auc'] for name, value in report['folds'][str(year)]['diagnostics'].items()}, flush=True)
        final = report['folds']['2022']['diagnostics']
        accepted = selected > 0. and final['selected']['micro_auc'] > final['current']['micro_auc']
        report['accept_combined_candidate'] = accepted
        report['accepted'] = accepted
        report['accepted_combined_weight'] = selected if accepted else 0.
        report['acceptance_reason'] = ('2021-selected combined blend improves current recipe on2022.' if accepted else 'Retain current recipe; selected combined blend did not pass the strict2022 improvement gate.')
        report['meets_transfer_gate'] = all(f['diagnostics']['selected' if accepted else 'current']['micro_auc'] >= .998 for f in report['folds'].values())
        report['status'] = 'complete'
        print('COMPLETE accepted_combined', accepted, 'weight', report['accepted_combined_weight'], flush=True)
    except BaseException:
        report['status'] = 'failed'
        report['error'] = traceback.format_exc()
        raise
    finally:
        report['runtime_seconds'] = time.time() - start
        report['source_code'] = dict(report['source'])
        report['sources'] = dict(report['inputs'])
        report['artifacts'] = {str(path.relative_to(out)): digest(path) for path in out.rglob('*') if path.is_file() and path.name not in ['manifest.json', 'analysis.json']}
        (out / 'manifest.json').write_text(json.dumps(report, indent=2))
        (out / 'analysis.json').write_text(json.dumps(report, indent=2))
        with (ROOT / 'docs/experiments.md').open('a') as stream:
            stream.write(f'\n## {args.run_id} — {report["timestamp_utc"]}\n\nStatus: {report["status"]}. Command: `{report["command"]}`. Runtime including waiting for external training: {report["runtime_seconds"]:.2f}s.\n\n')
            stream.write('Hypothesis: retaining521LP-context features and appending87joint hydraulic features improves scheduling information. Fixed global combined weights [0,.25,.5,.75,1], mixed with the current18.75%base/56.25%context/25%joint recipe and followed by unchanged Vinje projection. Select2021, persist immutable lock before2022 arrays, then apply one fixed2022 acceptance gate; no alternate2022 grid. Both years are reused development folds. No model training, seed, early stopping or package change in this comparison.\n\n')
            for year, fold in report['folds'].items():
                stream.write(f'- {year}: ' + ', '.join(f'{name}={value["micro_auc"]:.12f}' for name, value in fold['diagnostics'].items()) + '\n')
            stream.write(f'\nDecision: {report.get("acceptance_reason", "incomplete")}. Full source/input/data/checkpoint-manifest hashes, selection timing, exact IDs, per-unit/month/lead-day diagnostics, environment and failure trace: `artifacts/{args.run_id}/manifest.json`. Kaggle submission ID/score/rank/observation: none.\n')


if __name__ == '__main__':
    main()
