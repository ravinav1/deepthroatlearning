"""Compare a fixed small blend grid; choose on 2021 before reading 2022 labels."""
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

for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[variable] = '2'
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sintef_project.postprocess import project_vinje

SOURCES = ['lgb-normalized-context-001', 'lgb-normalized-001', 'lgb-physics-001']
CANDIDATES = [
    ('context', [1., 0., 0.]),
    ('normalized', [0., 1., 0.]),
    ('physics', [0., 0., 1.]),
    ('context75_normalized25', [.75, .25, 0.]),
    ('context50_normalized50', [.5, .5, 0.]),
    ('context75_physics25', [.75, 0., .25]),
]


def sha256(path: Path) -> str:
    """Return content digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Evaluate the predeclared grid without model fitting."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', default='ensemble-001')
    args = parser.parse_args()
    out = ROOT / 'artifacts' / args.run_id
    out.mkdir(exist_ok=False)
    start = time.time()
    report = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'command': shlex.join([sys.executable, *sys.argv]),
        'source_order': SOURCES,
        'candidate_weights': dict(CANDIDATES),
        'postprocessing': 'Fixed Vinje G1>=G2>=G3 L2 projection after blending; no smoothing. This transform was selected on 2021 in the earlier normalized/physics postprocess experiments.',
        'selection': 'Highest projected 2021 pooled micro AUC, exact ties resolved in listed order. Lock before loading any 2022 predictions or labels. 2022 grid scores are diagnostics only.',
        'seed': None, 'training': 'none; convex mixtures of saved out-of-time probabilities',
        'environment': {'python': sys.version, 'platform': platform.platform(),
                        'cpu_threads_cap': 2,
                        'packages': {name: importlib.metadata.version(name) for name in ['numpy', 'scikit-learn']}},
        'sources': {}, 'source_code': {}, 'folds': {},
        'kaggle': {'submission_id': None, 'score': None, 'rank': None, 'observation_time': None},
    }
    try:
        for relative in ['scripts/ensemble_compare.py', 'sintef_project/postprocess.py']:
            path = ROOT / relative
            dest = out / 'source' / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            report['source_code'][relative] = sha256(path)
        for source in SOURCES:
            manifest = ROOT / 'artifacts' / source / 'manifest.json'
            report['sources'][str(manifest.relative_to(ROOT))] = sha256(manifest)
            saved = json.loads(manifest.read_text())
            if saved.get('status') != 'complete':
                raise ValueError(f'{source} is not complete')
            for year in [2021, 2022]:
                if not (manifest.parent / f'validation_{year}.npz').is_file():
                    raise FileNotFoundError(f'{source} missing {year} validation')
        selected = None
        for year in [2021, 2022]:
            arrays = []
            y = ids = units = None
            for source in SOURCES:
                path = ROOT / 'artifacts' / source / f'validation_{year}.npz'
                report['sources'][str(path.relative_to(ROOT))] = sha256(path)
                with np.load(path) as z:
                    if y is None:
                        y, ids, units = z['labels'], z['case_ids'], z['units'].tolist()
                    elif not (np.array_equal(y, z['labels']) and np.array_equal(ids, z['case_ids']) and units == z['units'].tolist()):
                        raise ValueError(f'Misaligned labels, case IDs or units in {source}/{year}')
                    p = z['predictions'].astype(np.float64)
                    if p.shape != y.shape or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
                        raise ValueError('Invalid source probabilities')
                    arrays.append(p)
            fold = {'case_ids': ids.tolist(), 'units': units, 'scores': {}}
            report['folds'][str(year)] = fold
            for name, weights in CANDIDATES:
                raw = sum(weight * p for weight, p in zip(weights, arrays))
                pred = project_vinje(raw, units)
                fold['scores'][name] = {
                    'raw_micro_auc': float(roc_auc_score(y.ravel(), raw.ravel())),
                    'projected_micro_auc': float(roc_auc_score(y.ravel(), pred.ravel())),
                }
            if year == 2021:
                selected = max(CANDIDATES, key=lambda item: fold['scores'][item[0]]['projected_micro_auc'])
                report['selected_candidate'], report['selected_weights'] = selected
                report['selection_locked_utc'] = datetime.now(timezone.utc).isoformat()
                (out / 'selection_2021.json').write_text(json.dumps(report, indent=2))
            name, weights = selected
            pred = project_vinje(sum(w * p for w, p in zip(weights, arrays)), units)
            fold['selected_micro_auc'] = fold['scores'][name]['projected_micro_auc']
            fold['selected_unit_auc'] = {unit: float(roc_auc_score(y[:, i], pred[:, i])) for i, unit in enumerate(units)}
            np.savez_compressed(out / f'validation_{year}.npz', predictions=pred, labels=y, case_ids=ids, units=units)
            print(year, name, weights, fold['selected_micro_auc'], flush=True)
        report['status'] = 'complete'
        lines = [f'# Ensemble comparison — {report["timestamp_utc"]}', '',
                 'No models were fitted. Six configurations were declared in the source before running. All receive the same fixed Vinje projection; smoothing is disabled. Selection uses only 2021 and is locked before reading 2022. These are repeatedly reused development years, not untouched holdouts.', '',
                 '| Candidate | 2021 micro AUC | 2022 micro AUC |', '|---|---:|---:|']
        for name, weights in CANDIDATES:
            a = report['folds']['2021']['scores'][name]['projected_micro_auc']
            b = report['folds']['2022']['scores'][name]['projected_micro_auc']
            lines.append(f'| {name} | {a:.8f} | {b:.8f} |')
        lines.extend(['', f'2021-selected candidate: **{report["selected_candidate"]}**, weights {report["selected_weights"]} in order {SOURCES}.', '',
                      'The explicit migration threshold is at least 0.998 on both folds. Results below that threshold remain experimental; no leaderboard score is inferred.', '',
                      f'Complete provenance, raw/projected scores, source and artifact checksums: `artifacts/{args.run_id}/analysis.json`. The selected fold predictions are saved alongside it. No Kaggle submission was made by this experiment.'])
        (ROOT / 'docs/ensemble_comparison.md').write_text('\n'.join(lines) + '\n')
    except BaseException:
        report['status'] = 'failed'
        report['error'] = traceback.format_exc()
        raise
    finally:
        report['runtime_seconds'] = time.time() - start
        report['artifacts'] = {str(p.relative_to(out)): sha256(p) for p in out.rglob('*') if p.is_file() and p.name != 'analysis.json'}
        (out / 'analysis.json').write_text(json.dumps(report, indent=2))
        with (ROOT / 'docs/experiments.md').open('a') as f:
            f.write(f'\n## {args.run_id} — {report["timestamp_utc"]}\n\nStatus: {report["status"]}. Command: `{report["command"]}`. Runtime {report["runtime_seconds"]:.1f}s.\n\n')
            f.write('Hypothesis: normalized, normalized-context and physics models make complementary errors. Six fixed convex mixtures; common Vinje projection, no smoothing; no fitting/early stopping/seeds. Source models determine exact data, preprocessing, model settings and purged splits. All 168 hours and 14 units retained. Select by 2021 only and lock before reading 2022.\n\n')
            f.write(f'Sources: {SOURCES}. Weights: {dict(CANDIDATES)}. Selected: {report.get("selected_candidate")}.\n\n')
            for year, fold in report['folds'].items():
                f.write(f'- {year}: selected micro AUC {fold.get("selected_micro_auc")}.\n')
            if 'error' in report:
                f.write(f'\nFailure: `{report["error"].splitlines()[-1]}`\n')
            f.write(f'\nComplete environment/source/artifact provenance and score grid: `artifacts/{args.run_id}/analysis.json`. Kaggle submission ID/score/rank/observation: none.\n')


if __name__ == '__main__':
    main()
