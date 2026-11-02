"""Fixed eight-candidate LP comparison with a 2021 lock and 2022 acceptance gate."""
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

for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[name] = '2'
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sintef_project.postprocess import project_vinje

SOURCES = {'base': 'lgb-normalized-lp-001', 'context': 'lgb-normalized-lp-context-001',
           'v2': 'lgb-normalized-lp-v2-001', 'combined': 'vinje-count-combined-001'}
CANDIDATES = ['base', 'context', 'v2', 'base50_context50', 'context75_base25',
              'base_haukeli_v2_50', 'base_haukeli_v2_100', 'combined_specialist_count']
PRIOR = 'combined_specialist_count'
RECIPES = {
    'base': '100% raw LPv1 baseline', 'context': '100% raw LP temporal-context model',
    'v2': '100% raw LPv2 model', 'base50_context50': '50% base +50% context, all units',
    'context75_base25': '75% context +25% base, all units',
    'base_haukeli_v2_50': 'Base, replacing only Haukeli_G1 by50%base+50%v2',
    'base_haukeli_v2_100': 'Base, replacing only Haukeli_G1 by100%v2',
    'combined_specialist_count': 'Existing frozen50%XGB replacement forHaukeli/Lio/Hogga and50%Vinje-count replacement, without new tuning',
}


def digest(path: Path) -> str:
    """Return SHA256 content digest."""
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def verify(base: Path, hashes: dict) -> None:
    """Check every registered file without interpreting heldout predictions."""
    for relative, expected in hashes.items():
        path = base / relative
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f'Hash mismatch: {path}')


def candidate(name: str, predictions: dict[str, np.ndarray], units: list[str]) -> np.ndarray:
    """Compose the declared raw probabilities and apply common Vinje projection."""
    if name in ['base', 'context', 'v2']:
        values = predictions[name]
    elif name == 'base50_context50':
        values = .5 * predictions['base'] + .5 * predictions['context']
    elif name == 'context75_base25':
        values = .75 * predictions['context'] + .25 * predictions['base']
    elif name in ['base_haukeli_v2_50', 'base_haukeli_v2_100']:
        values = predictions['base'].copy()
        weight = .5 if name.endswith('_50') else 1.
        j = units.index('Haukeli_G1')
        values[:, j] = (1 - weight) * values[:, j] + weight * predictions['v2'][:, j]
    elif name == 'combined_specialist_count':
        values = predictions['combined']
    else:
        raise ValueError(f'Unknown predeclared candidate: {name}')
    return project_vinje(values, units)


def save_predictions(path: Path, values: tuple) -> None:
    """Save aligned probabilities, labels, IDs and unit names."""
    p, y, ids, units = values
    np.savez_compressed(path, predictions=p, labels=y, case_ids=ids, units=units)


def main() -> None:
    """Select solely on 2021, then accept or retain the fixed prior on 2022."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', default='lp-candidate-comparison-001')
    args = parser.parse_args()
    out = ROOT / 'artifacts' / args.run_id
    out.mkdir(exist_ok=False)
    started = time.time()
    report = {'run_id': args.run_id, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'command': shlex.join([sys.executable, *sys.argv]), 'sources': SOURCES,
              'candidates': CANDIDATES, 'recipes': RECIPES, 'prior_accepted_candidate': PRIOR,
              'projection': 'Fixed Vinje G1>=G2>=G3 L2 projection for every candidate; no smoothing',
              'selection_rule': 'Maximize projected2021 wholematrix microAUC; exactties in listedorder. Lock before decoding any2022 validation NPZ. If winner differs fromprior and fails to improve2022 strictly overprior, retainprior. Never select a different candidate using2022 scores.',
              'training': 'none; no fitting, early stopping or random seed',
              'reuse_bias': 'These chronological years and component choices have been reused extensively; 2022 acceptance is a development check, not an untouched holdout.',
              'environment': {'python': sys.version, 'platform': platform.platform(), 'threads_cap': 2,
                              'packages': {n: importlib.metadata.version(n) for n in ['numpy', 'scikit-learn']}},
              'source': {}, 'inputs': {}, 'data': {}, 'folds': {},
              'kaggle': {'submission_id': None, 'score': None, 'rank': None, 'observation_time': None}}
    selected_cache, prior_cache = {}, {}
    try:
        for relative in ['scripts/lp_candidate_compare.py', 'sintef_project/postprocess.py', 'docs/lp_candidate_comparison.md']:
            report['source'][relative] = digest(ROOT / relative)
            destination = out / 'source' / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, destination)
        for source in SOURCES.values():
            source_dir = ROOT / 'artifacts' / source
            path = source_dir / 'manifest.json'
            manifest = json.loads(path.read_text())
            if manifest.get('status') != 'complete':
                raise ValueError(f'Source is not complete: {source}')
            verify(source_dir, manifest['artifacts'])
            verify(source_dir / 'source', manifest['source'])
            verify(ROOT, manifest.get('inputs', {}))
            if report['data'] and report['data'] != manifest['data']:
                raise ValueError('Source data fingerprints differ')
            report['data'] = manifest['data']
            report['inputs'][str(path.relative_to(ROOT))] = digest(path)
            shutil.copy2(path, out / f'{source}_manifest.json')
        verify(ROOT, report['data'])
        for year in [2021, 2022]:
            inputs = {}
            y = ids = units = None
            for key, source in SOURCES.items():
                path = ROOT / 'artifacts' / source / f'validation_{year}.npz'
                report['inputs'][str(path.relative_to(ROOT))] = digest(path)
                with np.load(path, allow_pickle=False) as z:
                    if y is None:
                        y, ids, units = z['labels'], z['case_ids'], z['units'].tolist()
                    elif not (np.array_equal(y, z['labels']) and np.array_equal(ids, z['case_ids']) and units == z['units'].tolist()):
                        raise ValueError(f'Misaligned source labels/IDs/units: {source}/{year}')
                    p = z['predictions'].astype(np.float64)
                    if p.shape != y.shape or p.shape != (len(ids) * 168, len(units)) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
                        raise ValueError(f'Invalid probabilities: {source}/{year}')
                    inputs[key] = p
            fold = {'case_ids': ids.tolist(), 'units': units, 'scores': {}}
            report['folds'][str(year)] = fold
            for name in CANDIDATES:
                p = candidate(name, inputs, units)
                fold['scores'][name] = float(roc_auc_score(y.ravel(), p.ravel()))
            if year == 2021:
                selected = max(CANDIDATES, key=lambda name: fold['scores'][name])
                report['selected_candidate'] = selected
                report['selection_locked_utc'] = datetime.now(timezone.utc).isoformat()
                (out / 'selection_2021.json').write_text(json.dumps(report, indent=2))
            selected_predictions = candidate(selected, inputs, units)
            selected_cache[year] = (selected_predictions, y, ids, units)
            prior_cache[year] = (candidate(PRIOR, inputs, units), y, ids, units)
            save_predictions(out / f'selected_validation_{year}.npz', selected_cache[year])
            fold['selected_micro_auc'] = fold['scores'][selected]
            fold['prior_micro_auc'] = fold['scores'][PRIOR]
            fold['selected_unit_auc'] = {u: float(roc_auc_score(y[:, j], selected_predictions[:, j])) for j, u in enumerate(units)}
            print('SELECTED', year, selected, fold['selected_micro_auc'], 'prior', fold['prior_micro_auc'], flush=True)
        if selected == PRIOR:
            accepted, reason = PRIOR, 'The prior candidate itself won2021; retain it.'
        elif report['folds']['2022']['selected_micro_auc'] > report['folds']['2022']['prior_micro_auc']:
            accepted, reason = selected, 'The2021 winner also improves2022 over the fixedprior; accept this development candidate.'
        else:
            accepted, reason = PRIOR, 'The2021 winner failed the frozen2022 improvement gate; retain the fixedprior, without selecting any other2022 candidate.'
        report['accepted_candidate'], report['acceptance_reason'] = accepted, reason
        for year in [2021, 2022]:
            save_predictions(out / f'validation_{year}.npz', selected_cache[year] if accepted == selected else prior_cache[year])
            report['folds'][str(year)]['accepted_micro_auc'] = report['folds'][str(year)]['scores'][accepted]
        report['meets_transfer_gate'] = all(f['accepted_micro_auc'] >= .998 for f in report['folds'].values())
        report['status'] = 'complete'
        with (ROOT / 'docs/lp_candidate_comparison.md').open('a') as f:
            f.write(f'\n## Completed comparison:{args.run_id}\n\n| Candidate | 2021 microAUC | 2022 microAUC |\n|---|---:|---:|\n')
            for name in CANDIDATES:
                f.write(f'| {name} | {report["folds"]["2021"]["scores"][name]:.8f} | {report["folds"]["2022"]["scores"][name]:.8f} |\n')
            f.write(f'\n2021selected:**{selected}**. Accepted:**{accepted}**. {reason} Meets>=.998bothyears:{report["meets_transfer_gate"]}.\n\n')
            f.write(f'Provenance/fullgrid/selectionlock:`artifacts/{args.run_id}/manifest.json`. `selected_validation_YYYY.npz` preserves the2021winner even if rejected; `validation_YYYY.npz` contains the acceptedcandidate. No fullfit orKaggle submission was performed.\n')
        print('ACCEPTED', accepted, 'transfer_gate', report['meets_transfer_gate'], flush=True)
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
            f.write(f'Hypothesis:LP temporalcontext orLPv2 may add complementary input-only scheduling information. Eightfixed recipes:{RECIPES}. CommonVinjeprojection, nosmoothing. No model fits,seeds orearlystopping; source manifests preserve exact splits/modelsettings/features. Select2021only, lock before2022, accept only if2022 improves overfixedprior{PRIOR}; otherwise retainprior, never retuneusing2022. Reuseddevelopmentfolds.\n\n')
            for year, fold in report['folds'].items():
                f.write(f'- {year}: selected={fold.get("selected_micro_auc")}; prior={fold.get("prior_micro_auc")}; accepted={fold.get("accepted_micro_auc")}.\n')
            if 'error' in report:
                f.write(f'\nFailure:`{report["error"].splitlines()[-1]}`\n')
            f.write(f'\nComplete source/data/artifact hashes, IDs, units, source/model snapshots and environment:`artifacts/{args.run_id}/manifest.json`. Kaggle submission ID/score/rank/observation:none.\n')


if __name__ == '__main__':
    main()
