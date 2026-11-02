"""Independently reconstruct a completed frozen context-plus-joint comparison."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shlex
import shutil
import sys
import time
import traceback

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sintef_project.postprocess import project_vinje


def digest(path: Path) -> str:
    """Return SHA256 of a saved artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Verify exact mixtures and provenance without changing any selection."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--comparison-run', default='lp-context-joint-comparison-001')
    parser.add_argument('--run-id', default='verify-lp-context-joint-comparison-001')
    args = parser.parse_args()
    out = ROOT / 'artifacts' / args.run_id
    out.mkdir(exist_ok=False)
    started = time.time()
    result = {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'folds': {},
              'command': shlex.join([sys.executable, *sys.argv]),
              'comparison_run': args.comparison_run, 'training': 'none',
              'environment': {'platform': platform.platform(), 'python': sys.version,
                              'packages': {name: importlib.metadata.version(name)
                                           for name in ['numpy', 'scikit-learn']}}}
    try:
        directory = ROOT / 'artifacts' / args.comparison_run
        path = directory / 'analysis.json'
        report = json.loads(path.read_text())
        if report['status'] != 'complete':
            raise ValueError('Comparison must finish before independent verification')
        result['analysis_sha256'] = digest(path)
        lock_path = directory / 'selection_2021.json'
        lock = json.loads(lock_path.read_text())
        if digest(lock_path) != report['selection_sha256']:
            raise ValueError('Changed selection lock')
        if set(lock['folds']) != {'2021'}:
            raise ValueError('Lock must contain only2021')
        for name in ['source_order', 'selected_weights', 'selected_candidate']:
            if lock[name] != report[name]:
                raise ValueError(f'Lock mismatch: {name}')
        if digest(directory / 'protocol.json') != lock['protocol_sha256']:
            raise ValueError('Changed protocol')
        count = 0
        for base, hashes in [(directory, report['artifacts']), (ROOT, report['sources']),
                             (directory / 'source', report['source_code']), (ROOT, report['data'])]:
            for relative, expected in hashes.items():
                if digest(base / relative) != expected:
                    raise ValueError(f'Changed input: {base / relative}')
                count += 1
        weights = report['selected_weights']
        selected = report['selected_combined_weight']
        if weights != [.1875 * (1-selected), .5625 * (1-selected), .25 * (1-selected), selected]:
            raise ValueError('Unexpected flattened recipe')
        grid = lock['scores_2021']
        if selected != max([0., .25, .5, .75, 1.], key=lambda w: grid[str(w)]):
            raise ValueError('Wrong2021 selector or tie handling')
        combined_path = ROOT / 'artifacts' / report['source_order'][-1] / 'validation_2022.npz'
        selection_timestamp = datetime.fromisoformat(lock['locked_utc']).timestamp()
        if selection_timestamp >= combined_path.stat().st_mtime:
            raise ValueError('Selection was not locked before combined2022 array was created')
        result['selection_precedes_combined_2022_file'] = True
        for year in [2021, 2022]:
            labels = ids = units = None
            prediction = None
            for run, weight in zip(report['source_order'], weights):
                with np.load(ROOT / 'artifacts' / run / f'validation_{year}.npz', allow_pickle=False) as source:
                    if labels is None:
                        labels, ids, units = source['labels'], source['case_ids'], source['units'].tolist()
                        prediction = np.zeros(labels.shape, dtype=np.float64)
                    elif not (np.array_equal(labels, source['labels']) and np.array_equal(ids, source['case_ids'])
                              and units == source['units'].tolist()):
                        raise ValueError('Source label, ID or unit mismatch')
                    prediction += weight * source['predictions'].astype(np.float64)
            prediction = project_vinje(prediction, units)
            with np.load(directory / f'selected_validation_{year}.npz', allow_pickle=False) as saved:
                if not (np.array_equal(prediction, saved['predictions']) and np.array_equal(labels, saved['labels'])
                        and np.array_equal(ids, saved['case_ids']) and units == saved['units'].tolist()):
                    raise ValueError('Reconstructed selected array differs')
            score = float(roc_auc_score(labels.ravel(), prediction.ravel()))
            if score != report['folds'][str(year)]['selected_micro_auc']:
                raise ValueError('Recorded selected score differs')
            result['folds'][str(year)] = {'micro_auc': score, 'exact_mixture': True,
                                          'shape': list(prediction.shape)}
        final = report['folds']['2022']['diagnostics']
        accepted = selected > 0. and final['selected']['micro_auc'] > final['current']['micro_auc']
        if report['accepted'] != accepted or report['accept_combined_candidate'] != accepted:
            raise ValueError('Recorded acceptance differs from fixed rule')
        result.update(status='passed', checked_hashes=count, accepted=accepted,
                      weights=weights, selection_sha256=digest(lock_path))
        print(json.dumps(result, indent=2), flush=True)
    except BaseException:
        result['status'] = 'failed'
        result['error'] = traceback.format_exc()
        raise
    finally:
        source = Path(__file__)
        shutil.copy2(source, out / source.name)
        result['source_sha256'] = digest(source)
        result['runtime_seconds'] = time.time() - started
        (out / 'manifest.json').write_text(json.dumps(result, indent=2))
        with (ROOT / 'docs/experiments.md').open('a') as stream:
            stream.write(f'\n## {args.run_id} — {result["timestamp_utc"]}\n\nStatus: {result["status"]}. Command: `{result["command"]}`. Runtime{result["runtime_seconds"]:.2f}s. Independent saved-array verification of source/input/data/artifact hashes, immutable2021 lock, exact flattened convex mixture/projection, labels/IDs/units, pooled AUC and fixed acceptance rule. Also verifies the lock timestamp precedes creation of the combined2022 NPZ. No fitting or changed weights. Proof/source snapshot:`artifacts/{args.run_id}/manifest.json`. Kaggle submission ID/score/rank/observation:none.\n')


if __name__ == '__main__':
    main()
