"""Select case-local postprocessing on 2021; independently evaluate 2022."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import platform
import shlex
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sintef_project.postprocess import smooth_predictions, project_vinje


def digest(path: Path) -> str:
    """Return file SHA256."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def transform(p: np.ndarray, units: list[str], config: dict) -> np.ndarray:
    """Apply the fixed smoothing/projection recipe without labels."""
    result = smooth_predictions(p, config['window'], config['weight'])
    return project_vinje(result, units) if config['projection'] else result


def main() -> None:
    """Evaluate all transforms, selecting solely on the 2021 tuning fold."""
    parser = argparse.ArgumentParser()
    parser.add_argument('runs', nargs='+')
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    output = ROOT / 'artifacts' / args.run_id
    output.mkdir(exist_ok=False)
    start = time.time()
    report = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'command': shlex.join([sys.executable, *sys.argv]),
        'sources': {}, 'source_snapshots': {}, 'results': {},
        'selection_rule': 'Maximize 2021 micro AUC only. Break exact ties by input grid order; no-op first. Freeze config before reading 2022 labels. 2022 grid scores are diagnostic only.',
        'environment': {'python': sys.version, 'platform': platform.platform(),
                        'packages': {n: importlib.metadata.version(n) for n in ['numpy', 'scikit-learn']}},
        'seed': None, 'training': 'none; evaluate previously saved probabilities',
        'kaggle': {'submission_id': None, 'score': None, 'rank': None, 'observation_time': None},
    }
    configs = [{'window': window, 'weight': weight, 'projection': projection}
               for window in [3, 7, 13, 25] for weight in [0., .25, .5, 1.]
               for projection in [False, True]]
    report['grid'] = configs
    try:
        for relative in ['scripts/analyze_predictions.py', 'sintef_project/postprocess.py']:
            path = ROOT / relative
            dest = output / 'source' / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            report['source_snapshots'][relative] = digest(path)
        for run in args.runs:
            run_dir = ROOT / 'artifacts' / run
            for year in [2021, 2022]:
                if not (run_dir / f'validation_{year}.npz').is_file():
                    raise FileNotFoundError(f'{run} requires both 2021 and 2022 validation files')
            manifest = run_dir / 'manifest.json'
            report['sources'][str(manifest.relative_to(ROOT))] = digest(manifest)
            result = {'folds': {}}
            report['results'][run] = result
            selected = None
            for year in [2021, 2022]:
                path = run_dir / f'validation_{year}.npz'
                report['sources'][str(path.relative_to(ROOT))] = digest(path)
                with np.load(path) as data:
                    y, p, units = data['labels'], data['predictions'], data['units'].tolist()
                    case_ids = data['case_ids'].tolist()
                scores, cache = [], {}
                for config in configs:
                    key = (config['window'] if config['weight'] else 1,
                           config['weight'], config['projection'])
                    if key not in cache:
                        transformed = transform(p, units, config)
                        cache[key] = float(roc_auc_score(y.ravel(), transformed.ravel()))
                    scores.append(cache[key])
                fold = {'case_ids': case_ids, 'grid_micro_auc': scores,
                        'baseline_micro_auc': float(roc_auc_score(y.ravel(), p.ravel()))}
                result['folds'][str(year)] = fold
                if year == 2021:
                    selected = int(np.argmax(scores))
                    result['selected_grid_index'] = selected
                    result['selected_config'] = configs[selected]
                    result['selection_locked_utc'] = datetime.now(timezone.utc).isoformat()
                    (output / 'selection_2021.json').write_text(json.dumps(report, indent=2))
                fold['selected_micro_auc'] = scores[selected]
                chosen = transform(p, units, configs[selected])
                fold['selected_unit_auc'] = {u: float(roc_auc_score(y[:, i], chosen[:, i]))
                                             for i, u in enumerate(units)}
                np.savez_compressed(output / f'{run}_validation_{year}.npz',
                                    predictions=chosen, labels=y,
                                    case_ids=case_ids, units=units)
                print(run, year, configs[selected], 'baseline', fold['baseline_micro_auc'],
                      'selected', fold['selected_micro_auc'], flush=True)
        report['status'] = 'complete'
    except BaseException:
        report['status'] = 'failed'
        report['error'] = traceback.format_exc()
        raise
    finally:
        report['runtime_seconds'] = time.time() - start
        report['artifacts'] = {str(p.relative_to(output)): digest(p) for p in output.rglob('*')
                               if p.is_file() and p.name != 'analysis.json'}
        (output / 'analysis.json').write_text(json.dumps(report, indent=2))
        with (ROOT / 'docs/experiments.md').open('a') as f:
            f.write(f'\n## {args.run_id} — {report["timestamp_utc"]}\n\n')
            f.write(f'Status: {report["status"]}. Hypothesis: startup costs favor persistent within-case schedules; near-nested Vinje labels support an optional probability ordering projection. Exact command: `{report["command"]}`. Runtime {report["runtime_seconds"]:.1f}s.\n\n')
            f.write('No training, seeds or early stopping. Same case splits, input data and base models as source manifests. Centered edge-normalized probability means over windows 3,7,13,25; weights 0,.25,.5,1; optional L2 projection onto Vinje G1 >= G2 >= G3. Select solely on 2021; freeze before reading 2022 labels. 2022 grid scores are diagnostics, never selection criteria.\n\n')
            for run, result in report['results'].items():
                f.write(f'- {run}: selected config {result.get("selected_config")}. ')
                for year, fold in result['folds'].items():
                    f.write(f'{year} baseline={fold["baseline_micro_auc"]:.8f}, selected={fold["selected_micro_auc"]:.8f}. ')
                f.write('\n')
            if 'error' in report:
                f.write(f'\nFailure: `{report["error"].splitlines()[-1]}`\n')
            f.write(f'\nComplete grid, environment, source snapshots, source/artifact checksums and selected predictions: `artifacts/{args.run_id}/analysis.json`. Kaggle submission ID, score, rank and observation time: none.\n')


if __name__ == '__main__':
    main()
