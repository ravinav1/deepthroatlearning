"""Describe remaining ranking errors in fixed accepted predictions; fit nothing."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from sklearn.metrics import roc_auc_score
from scripts.analyze_gpu608_pair_errors import pair_matrices, synthetic_check


def digest(path: Path) -> str:
    """Hash a recorded input or output in bounded chunks."""
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main() -> None:
    """Compare unchanged SHOP and prepared LGB blends without selecting a recipe."""
    started = time.monotonic()
    out = ROOT / 'artifacts/shop664-pair-error-diagnostic-001'
    out.mkdir(exist_ok=False)
    protocol_path = ROOT / 'docs/lgb_shop664_protocol.json'
    assert digest(protocol_path) == '882e8c3124ba834322c8f062a7544e9ca3e9512b385e8830db387c38f7c97d08'
    protocol = json.loads(protocol_path.read_text())
    synthetic_check()
    report = {
        'status': 'initializing', 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'command': [sys.executable, *sys.argv], 'training': False,
        'selection_or_prediction_changes': False, 'kaggle_submission': None,
        'environment': {'python': sys.version, 'platform': platform.platform(),
                        'packages': {n: importlib.metadata.version(n) for n in
                                     ['numpy', 'pandas', 'scikit-learn']}},
        'source': {str(p.relative_to(ROOT)): digest(p) for p in
                   [Path(__file__), ROOT / 'scripts/analyze_gpu608_pair_errors.py']},
        'inputs': {str(protocol_path.relative_to(ROOT)): digest(protocol_path)}, 'years': {},
        'caveat': 'Both years are reused development evidence. Endpoint allocation is descriptive; it does not authorize unit-specific splicing or calibration.'}
    rows = []
    try:
        for year in ['2021', '2022']:
            anchor = protocol['anchor'][year]
            for key in ['manifest', 'prediction']:
                path = ROOT / anchor[key]
                assert digest(path) == anchor[key + '_sha256']
                report['inputs'][str(path.relative_to(ROOT))] = digest(path)
            with np.load(ROOT / anchor['prediction'], allow_pickle=False) as z:
                labels, units, ids = z['labels'], z['units'].tolist(), z['case_ids']
                predictions = {'public_shop664': z['predictions']}
            assert labels.shape == (365 * 168, 14) and len(ids) == 365
            directory = ROOT / f'artifacts/lgb-shop664-{year}-001'
            manifest = json.loads((directory / 'manifest.json').read_text())
            audit_path = ROOT / f'artifacts/lgb-shop664-{year}-independent-audit-001/manifest.json'
            audit = json.loads(audit_path.read_text())
            assert audit['status'] == 'passed' and audit['candidate_manifest_sha256'] == digest(directory / 'manifest.json')
            assert manifest['status'] == 'complete' and manifest['selected_weight'] == .1
            selected_path = directory / f'selected_validation_{year}.npz'
            assert digest(selected_path) == manifest['artifacts'][selected_path.name]
            for path in [directory / 'manifest.json', selected_path, audit_path]:
                report['inputs'][str(path.relative_to(ROOT))] = digest(path)
            with np.load(selected_path, allow_pickle=False) as z:
                assert np.array_equal(z['labels'], labels) and np.array_equal(z['case_ids'], ids)
                assert z['units'].tolist() == units
                predictions['prepared_lgb664_blend'] = z['predictions']
            results = {}
            for name, p in predictions.items():
                assert p.shape == labels.shape and np.isfinite(p).all() and p.min() >= 0 and p.max() <= 1
                error, count = pair_matrices(labels, p)
                auc = float(roc_auc_score(labels.ravel(), p.ravel()))
                assert abs(1 - error.sum() / count.sum() - auc) < 1e-13
                endpoint = (error.sum(axis=0) + error.sum(axis=1)) / (2 * error.sum())
                unit_rows = []
                for j, unit in enumerate(units):
                    unit_auc = float(roc_auc_score(labels[:, j], p[:, j]))
                    assert abs(1 - error[j, j] / count[j, j] - unit_auc) < 1e-13
                    row = {'year': year, 'candidate': name, 'unit': unit,
                           'endpoint_error_share': float(endpoint[j]), 'within_unit_auc': unit_auc,
                           'mean_probability_minus_on_rate': float(p[:, j].mean() - labels[:, j].mean())}
                    unit_rows.append(row)
                    rows.append(row)
                assert abs(sum(r['endpoint_error_share'] for r in unit_rows) - 1) < 1e-12
                results[name] = {'micro_auc': auc, 'cross_unit_error_share': float(1 - np.trace(error) / error.sum()),
                                 'units': sorted(unit_rows, key=lambda r: -r['endpoint_error_share'])}
                np.savez_compressed(out / f'{name}_{year}.npz', error_pair_counts=error,
                                    pair_counts=count, units=np.asarray(units), case_ids=ids)
            assert abs(results['prepared_lgb664_blend']['micro_auc'] - manifest['selected_micro_auc']) < 1e-15
            report['years'][year] = results
        report['status'] = 'complete'
        (out / 'unit_summary.json').write_text(json.dumps(rows, indent=2) + '\n')
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        shutil.copy2(__file__, out / Path(__file__).name)
        report['runtime_seconds'] = time.monotonic() - started
        report['peak_rss_bytes'] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        report['artifacts'] = {p.name: digest(p) for p in out.iterdir() if p.name != 'manifest.json'}
        (out / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
        with (ROOT / 'docs/experiments.md').open('a') as stream:
            stream.write(f'\n\n## shop664-pair-error-diagnostic-001 — {report["timestamp_utc"]}\n\n'
                         f'Status {report["status"]}; read-only error decomposition of current SHOP664 and already-fixed LGB664 blend. '
                         f'Runtime {report["runtime_seconds"]:.3f}s; no fitting, new recipe selection or submission. '
                         'Exact commands, input/source/environment hashes, case IDs and directed pair matrices: '
                         '`artifacts/shop664-pair-error-diagnostic-001/manifest.json`. Both years are reused development data.\n')
        print(json.dumps({'status': report['status'], 'manifest_sha256': digest(out / 'manifest.json'),
                          'seconds': report['runtime_seconds']}), flush=True)


if __name__ == '__main__':
    main()
