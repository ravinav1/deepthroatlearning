"""Describe residual errors of the frozen accepted four-component blend."""
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
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    """Hash a frozen input file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Compute exact pair-error and Brier decompositions without fitting."""
    start = time.monotonic()
    out = ROOT / 'artifacts/current-blend-residuals-001'
    out.mkdir(exist_ok=False)
    source = ROOT / 'artifacts/lp-context-joint-comparison-001'
    manifest_path = source / 'analysis.json'
    origin = json.loads(manifest_path.read_text())
    assert origin['status'] == 'complete' and origin['accepted']
    report = {
        'run_id': out.name, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'command': 'UV_CACHE_DIR=/private/tmp/tdt4173-xgb-uv-cache uv run --offline --no-project --python .venv-ml/bin/python python scripts/analyze_current_blend_errors.py',
        'training': 'none; descriptive analysis of frozen validation arrays only',
        'seed': None, 'kaggle_submission': None,
        'reuse_caveat': 'Both years are reused development evidence. Label-defined diagnostics are not deployable features or a new candidate selection.',
        'environment': {'python': sys.version, 'platform': platform.platform(), 'machine': platform.machine(),
                        'packages': {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scipy', 'scikit-learn']}},
        'source_analysis_sha256': digest(manifest_path), 'data': origin['data'],
        'weights': origin['selected_weights'], 'inputs': {}, 'results': {},
        'formulas': {'pair_loss': 'Each misordered positive-negative pair contributes half its error to each endpoint; ties count half. Divide by total positive-negative pairs.',
                     'brier_bias': 'Mean squared case-mean residual divided by mean squared residual.',
                     'within_case_auc': 'Pair-count-weighted AUC restricted to same case and unit.'},
    }
    rows, case_rows = [], []
    for year in [2021, 2022]:
        path = source / f'selected_validation_{year}.npz'
        assert digest(path) == origin['artifacts'][path.name]
        report['inputs'][str(path.relative_to(ROOT))] = digest(path)
        with np.load(path, allow_pickle=False) as saved:
            p, y, ids, units = saved['predictions'], saved['labels'], saved['case_ids'], saved['units'].tolist()
        positive, negative = np.sort(p[y == 1]), np.sort(p[y == 0])
        total_pairs = len(positive) * len(negative)
        losses = np.empty(p.shape)
        values = p[y == 1]
        losses[y == 1] = len(negative) - .5 * (np.searchsorted(negative, values, 'left') + np.searchsorted(negative, values, 'right'))
        values = p[y == 0]
        losses[y == 0] = .5 * (np.searchsorted(positive, values, 'left') + np.searchsorted(positive, values, 'right'))
        losses /= 2 * total_pairs
        auc = float(roc_auc_score(y.ravel(), p.ravel()))
        np.testing.assert_allclose(losses.sum(), 1 - auc, rtol=0, atol=1e-13)
        yc, pc = y.reshape(len(ids), 168, 14), p.reshape(len(ids), 168, 14)
        case_loss = losses.reshape(len(ids), 168, 14).sum(1)
        error = yc - pc
        bias, brier = error.mean(1), (error ** 2).mean(1)
        np.testing.assert_allclose(brier, bias ** 2 + ((error - bias[:, None, :]) ** 2).mean(1), rtol=0, atol=1e-12)
        count = yc.sum(1)
        pairs = count * (168 - count)
        good = (rankdata(pc, axis=1, method='average') * yc).sum(1) - count * (count + 1) / 2
        for j, unit in enumerate(units):
            constant = (count[:, j] == 0) | (count[:, j] == 168)
            unit_loss = float(losses[:, j].sum())
            row = {'year': year, 'unit': unit, 'unit_auc': float(roc_auc_score(y[:, j], p[:, j])),
                   'micro_error_share': unit_loss / (1 - auc),
                   'constant_case_micro_error_share': float(case_loss[constant, j].sum() / unit_loss),
                   'brier_bias_fraction': float((bias[:, j] ** 2).sum() / brier[:, j].sum()),
                   'within_case_auc': float(good[:, j].sum() / pairs[:, j].sum())}
            rows.append(row)
            for i, case_id in enumerate(ids):
                case_rows.append({'year': year, 'case_id': int(case_id), 'unit': unit,
                                  'micro_error': float(case_loss[i, j]), 'bias': float(bias[i, j]),
                                  'truth_fraction': float(yc[i, :, j].mean()),
                                  'predicted_fraction': float(pc[i, :, j].mean())})
        report['results'][str(year)] = {'micro_auc': auc, 'case_ids': ids.tolist(),
            'units': {row['unit']: row for row in rows if row['year'] == year}}
    pd.DataFrame(rows).to_csv(out / 'unit_residuals.csv', index=False)
    pd.DataFrame(case_rows).to_csv(out / 'case_residuals.csv', index=False)
    shutil.copy2(__file__, out / 'source.py')
    report.update(status='complete', runtime_seconds=time.monotonic() - start)
    report['artifacts'] = {p.name: digest(p) for p in out.iterdir() if p.is_file()}
    (out / 'manifest.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({str(year): sorted([r for r in rows if r['year'] == year], key=lambda r:-r['micro_error_share'])[:5]
                      for year in [2021, 2022]}, indent=2))


if __name__ == '__main__':
    main()
