"""Export an already selected chronological blend from verified full-fit arrays."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import shutil
import sys
import time
import traceback

from export_ensemble import ROOT, digest, verify_hashes, validate_final, read_submission
import numpy as np
import pandas as pd
from sintef_project.postprocess import project_vinje


def main() -> None:
    """Verify the selection lock and every source recipe before a convex export."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--selection-run', required=True)
    parser.add_argument('--final-runs', nargs='+', required=True)
    args = parser.parse_args()
    out = ROOT / 'artifacts' / args.run_id
    out.mkdir(exist_ok=False)
    start = time.time()
    report = {'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'command': shlex.join([sys.executable, *sys.argv]), 'sources': {},
              'training': 'none; saved-array export, independent replay is separate',
              'kaggle': {'submission_id': None, 'score': None, 'rank': None, 'observation_time': None}}
    try:
        directory = ROOT / 'artifacts' / args.selection_run
        selection_path = directory / 'analysis.json'
        if not selection_path.exists():
            selection_path = directory / 'manifest.json'
        selection = json.loads(selection_path.read_text())
        locked = json.loads((directory / 'selection_2021.json').read_text())
        if 'source_order' not in selection:
            # The separate LP candidate experiment uses named recipes, including
            # unit-specific candidates. Only this accepted convex recipe is
            # supported here; never reinterpret an arbitrary recipe string.
            if (selection.get('accepted_candidate') != 'context75_base25'
                    or selection.get('selected_candidate') != 'context75_base25'
                    or locked.get('selected_candidate') != 'context75_base25'):
                raise ValueError('Unsupported or rejected named blend recipe')
            for record in [selection, locked]:
                record['source_order'] = [record['sources']['base'], record['sources']['context']]
                record['selected_weights'] = [.25, .75]
                record['source_code'] = record['source']
                record['sources'] = record['inputs']
        if selection['status'] != 'complete' or set(locked['folds']) != {'2021'}:
            raise ValueError('Selection was not completed and locked on 2021 alone')
        if ('accepted_candidate' in selection
                and selection['accepted_candidate'] != selection['selected_candidate']):
            raise ValueError('The frozen candidate failed its chronological acceptance gate')
        if selection.get('accepted') is False:
            raise ValueError('The selected candidate was explicitly rejected')
        weights = selection['selected_weights']
        sources = selection['source_order']
        if (weights != locked['selected_weights'] or sources != locked['source_order']
                or selection['selected_candidate'] != locked['selected_candidate']):
            raise ValueError('Selected recipe differs from frozen lock')
        if len(args.final_runs) != len(sources) or len(weights) != len(sources):
            raise ValueError('Every selected source needs its matching final fit')
        if not np.isfinite(weights).all() or min(weights) < 0 or not np.isclose(sum(weights), 1):
            raise ValueError('Expected finite convex probability weights')
        verify_hashes(directory, selection['artifacts'])
        verify_hashes(ROOT, selection['sources'])
        verify_hashes(directory / 'source', selection['source_code'])
        if digest(ROOT / 'sintef_project/postprocess.py') != selection['source_code']['sintef_project/postprocess.py']:
            raise ValueError('Selected probability projection changed')
        template = pd.read_csv(ROOT / 'sintef_project/sample_submission.csv')
        mapping = pd.read_csv(ROOT / 'sintef_project/prediction_mapping.csv')
        units = selection['folds']['2021']['units']
        if (template.shape != (725, 2353) or not mapping['Run No'].equals(template['Run No'])
                or template.columns.tolist() != ['Run No'] + [f'result_committed_{u}_t{h}' for u in units for h in range(168)]):
            raise ValueError('Unexpected ordered competition schema')
        pred = np.zeros((725 * 168, 14), dtype=np.float64)
        for source, final_run, weight in zip(sources, args.final_runs, weights):
            validation = json.loads((ROOT / 'artifacts' / source / 'manifest.json').read_text())
            p, provenance = validate_final(ROOT / 'artifacts' / final_run, validation,
                                           validation['arguments']['feature_set'], template, units)
            pred += weight * p
            report['sources'][source] = provenance
        pred = project_vinje(pred, units)
        wide = pred.reshape(725, 168, 14).transpose(0, 2, 1).reshape(725, -1)
        submission = pd.DataFrame(wide, columns=template.columns[1:])
        submission.insert(0, 'Run No', template['Run No'].to_numpy())
        np.save(out / 'test_predictions.npy', pred)
        submission.to_csv(out / 'submission.csv', index=False, float_format='%.9g')
        readback = read_submission(out / 'submission.csv', template)
        error = float(np.max(np.abs(readback.iloc[:, 1:].to_numpy() - wide)))
        if error > 5.1e-10:
            raise ValueError('Submission serialization changed probabilities')
        report.update(status='complete', weights=weights, source_order=sources,
                      postprocessing='Vinje projection only, after convex probability blending',
                      validation_micro_auc={year: fold['selected_micro_auc'] for year, fold in selection['folds'].items()},
                      selection_manifest=str(selection_path), selection_sha256=digest(selection_path),
                      selection_lock_sha256=digest(directory / 'selection_2021.json'),
                      submission_sha256=digest(out / 'submission.csv'), shape=[725, 2353],
                      probability_range=[float(pred.min()), float(pred.max())], readback_max_error=error)
        print(json.dumps({key: report[key] for key in ['status', 'weights', 'validation_micro_auc', 'submission_sha256']}), flush=True)
    except BaseException:
        report['status'] = 'failed'
        report['error'] = traceback.format_exc()
        raise
    finally:
        for name in ['scripts/export_selected_blend.py', 'scripts/export_ensemble.py', 'sintef_project/postprocess.py']:
            destination = out / 'source' / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, destination)
        report['runtime_seconds'] = time.time() - start
        report['artifacts'] = {str(p.relative_to(out)): digest(p) for p in out.rglob('*') if p.is_file() and p.name != 'manifest.json'}
        (out / 'manifest.json').write_text(json.dumps(report, indent=2))
        with (ROOT / 'docs/experiments.md').open('a') as f:
            f.write(f'\n## {args.run_id} — {report["timestamp_utc"]}\n\nStatus: {report["status"]}. Exact command: `{report["command"]}`. No fitting or checkpoint inference; verified selected full-fit recipes, all source/checkpoint/data/artifact hashes, schema and probability serialization. Independent replay is separate. Manifest: `artifacts/{args.run_id}/manifest.json`. Runtime {report["runtime_seconds"]:.2f}s. Kaggle ID/score/rank/observation: none at export.\n')


if __name__ == '__main__':
    main()
