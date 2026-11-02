"""Independently verify the completed frozen large-LP blend comparison."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sintef_project.postprocess import project_vinje


def digest(path: Path) -> str:
    """Hash a file in bounded chunks."""
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def verify_files(directory: Path, expected: dict) -> int:
    """Check every saved relative file hash and return the checked count."""
    for relative, sha in expected.items():
        assert digest(directory / relative) == sha, str(directory / relative)
    return len(expected)


def main() -> None:
    """Reconstruct only the predeclared locked recipe, without selecting weights."""
    start = time.time()
    out = ROOT / 'artifacts/verify-lp-large-recovery-001'
    out.mkdir(exist_ok=False)
    comparison_dir = ROOT / 'artifacts/lp-large-blend-001'
    comparison_path = comparison_dir / 'manifest.json'
    comparison = json.loads(comparison_path.read_text())
    lock_path = comparison_dir / 'selection_2021.json'
    lock = json.loads(lock_path.read_text())
    large_dir = ROOT / 'artifacts/lgb-normalized-lp-large-001'
    large = json.loads((large_dir / 'manifest.json').read_text())
    report = {
        'run_id': out.name, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'command': 'UV_CACHE_DIR=/private/tmp/tdt4173-xgb-uv-cache uv run --offline --no-project --python .venv-ml/bin/python python scripts/verify_lp_large_recovery.py',
        'environment': {'python': sys.version, 'platform': platform.platform(), 'machine': platform.machine(),
                        'packages': {name: importlib.metadata.version(name) for name in ['numpy', 'scikit-learn']}},
        'seed': 4173, 'training': 'none; arrays and provenance checks only',
        'anchor': 'lp-candidate-comparison-001, context75_base25; not changed to the newer joint recipe',
        'locked_weights_base_context_large': [.1875, .5625, .25],
        'comparison_manifest_sha256': digest(comparison_path), 'selection_lock_sha256': digest(lock_path),
        'large_training_manifest_sha256': digest(large_dir / 'manifest.json'),
        'data': large['data'], 'folds': {}, 'checked_files': {}, 'kaggle_submission': None,
    }
    try:
        assert comparison['status'] == large['status'] == 'complete'
        assert comparison['accepted'] is True
        assert lock['selected_scalar_weight'] == comparison['selected_scalar_weight'] == .25
        assert lock['selected_weights'] == comparison['selected_weights'] == [.1875, .5625, .25]
        assert lock['source_order'] == comparison['source_order'] == [
            'lgb-normalized-lp-001', 'lgb-normalized-lp-context-001', 'lgb-normalized-lp-large-001']
        assert lock['folds']['2021'] == {k: v for k, v in comparison['folds']['2021'].items() if k != 'accepted_micro_auc'}
        snapshot = json.loads((comparison_dir / 'large_manifest_at_selection.json').read_text())
        for key in ['data', 'source', 'features', 'feature_spec', 'arguments', 'model_parameters']:
            assert snapshot[key] == large[key]
        assert snapshot['folds']['2021'] == large['folds']['2021']
        assert '2022' not in snapshot['folds']
        report['checked_files']['comparison_artifacts'] = verify_files(comparison_dir, comparison['artifacts'])
        report['checked_files']['comparison_sources'] = verify_files(comparison_dir / 'source', comparison['source_code'])
        report['checked_files']['comparison_inputs'] = verify_files(ROOT, comparison['sources'])
        report['checked_files']['raw_data'] = verify_files(ROOT, large['data'])
        for run in comparison['source_order']:
            directory = ROOT / 'artifacts' / run
            manifest = json.loads((directory / 'manifest.json').read_text())
            report['checked_files'][run] = verify_files(directory, manifest['artifacts'])
            report['checked_files'][run + '_source'] = verify_files(directory / 'source', manifest['source'])
        for year in [2021, 2022]:
            arrays = []
            for run in comparison['source_order']:
                with np.load(ROOT / 'artifacts' / run / f'validation_{year}.npz', allow_pickle=False) as saved:
                    arrays.append({name: saved[name].copy() for name in ['predictions', 'labels', 'case_ids', 'units']})
            first = arrays[0]
            for values in arrays[1:]:
                for key in ['labels', 'case_ids', 'units']:
                    np.testing.assert_array_equal(values[key], first[key])
            units = first['units'].tolist()
            raw = [value['predictions'].astype(np.float64) for value in arrays]
            selected = np.zeros_like(raw[0], dtype=np.float64)
            selected += .1875 * raw[0]
            selected += .5625 * raw[1]
            selected += .25 * raw[2]
            selected = project_vinje(selected, units)
            prior = project_vinje(.25 * raw[0] + .75 * raw[1], units)
            assert selected.shape == first['labels'].shape == (len(first['case_ids']) * 168, 14)
            assert np.isfinite(selected).all() and selected.min() >= 0 and selected.max() <= 1
            for name in ['selected_validation', 'validation']:
                with np.load(comparison_dir / f'{name}_{year}.npz', allow_pickle=False) as saved:
                    np.testing.assert_array_equal(selected, saved['predictions'])
            with np.load(ROOT / 'artifacts/lp-candidate-comparison-001' / f'validation_{year}.npz', allow_pickle=False) as saved:
                np.testing.assert_array_equal(prior, saved['predictions'])
            selected_auc = float(roc_auc_score(first['labels'].ravel(), selected.ravel()))
            prior_auc = float(roc_auc_score(first['labels'].ravel(), prior.ravel()))
            assert selected_auc == comparison['folds'][str(year)]['selected_micro_auc']
            assert prior_auc == comparison['folds'][str(year)]['prior_micro_auc']
            assert selected_auc > prior_auc
            report['folds'][str(year)] = {
                'selected_micro_auc': selected_auc, 'prior_micro_auc': prior_auc,
                'improvement': selected_auc - prior_auc, 'case_ids': first['case_ids'].tolist(),
                'validation_rows': len(selected), 'exact_array_reconstruction': True,
            }
        report['meets_transfer_gate'] = all(fold['selected_micro_auc'] >= .998 for fold in report['folds'].values())
        report['comparison_to_later_candidates'] = {}
        for run in ['lp-joint-comparison-001', 'lp-context-joint-comparison-001']:
            other_path = ROOT / 'artifacts' / run / 'analysis.json'
            other = json.loads(other_path.read_text())
            report['comparison_to_later_candidates'][run] = {
                'analysis_sha256': digest(other_path),
                'scores': {year: fold['selected_micro_auc'] for year, fold in other['folds'].items()},
                'large_is_lower_both_years': all(report['folds'][year]['selected_micro_auc'] < fold['selected_micro_auc']
                                               for year, fold in other['folds'].items()),
            }
        report.update(status='passed', accepted_against_predeclared_anchor=True,
                      recommendation='Retain as completed accepted ablation, but do not final-fit as a replacement: newer fixed candidates score higher on both development years.')
    except BaseException:
        report['status'] = 'failed'
        raise
    finally:
        shutil.copy2(__file__, out / 'verify_script.py')
        report['runtime_seconds'] = time.time() - start
        report['artifacts'] = {p.name: digest(p) for p in out.iterdir() if p.is_file() and p.name != 'manifest.json'}
        (out / 'manifest.json').write_text(json.dumps(report, indent=2))
        print(json.dumps({'status': report['status'], 'manifest': str(out / 'manifest.json'),
                          'runtime_seconds': report['runtime_seconds']}))


if __name__ == '__main__':
    main()
