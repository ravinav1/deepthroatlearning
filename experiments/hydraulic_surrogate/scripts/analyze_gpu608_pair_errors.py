"""Exact descriptive micro-AUC error decomposition of the accepted GPU608 blend."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import shlex
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = 'gpu608-pair-error-diagnostic-001'
PROTOCOL_SHA = 'e9172028088440d8a46a5966946c63cb723606ea916833d3c81e5f67ccfae026'


def digest(path: Path) -> str:
    """Return the exact file checksum."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bad_pairs(positive: np.ndarray, sorted_negative: np.ndarray) -> float:
    """Count reversed pairs, assigning half a loss to exact ties."""
    return float(np.sum(len(sorted_negative) - .5 * (
        np.searchsorted(sorted_negative, positive, 'left') +
        np.searchsorted(sorted_negative, positive, 'right'))))


def pair_matrices(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Count every directed positive-unit/negative-unit pair and its error."""
    n_units = y.shape[1]
    positive = [p[y[:, j] == 1, j] for j in range(n_units)]
    negative = [np.sort(p[y[:, j] == 0, j]) for j in range(n_units)]
    error, count = np.zeros((n_units, n_units)), np.zeros((n_units, n_units))
    for i in range(n_units):
        for j in range(n_units):
            error[i, j] = bad_pairs(positive[i], negative[j])
            count[i, j] = len(positive[i]) * len(negative[j])
    return error, count


def endpoint_losses(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Allocate half of each global pair error to each of its two endpoints."""
    positive, negative = np.sort(p[y == 1]), np.sort(p[y == 0])
    loss = np.zeros(p.shape)
    values = p[y == 1]
    loss[y == 1] = len(negative) - .5 * (
        np.searchsorted(negative, values, 'left') + np.searchsorted(negative, values, 'right'))
    values = p[y == 0]
    loss[y == 0] = .5 * (
        np.searchsorted(positive, values, 'left') + np.searchsorted(positive, values, 'right'))
    return loss / (2 * len(positive) * len(negative))


def synthetic_check() -> None:
    """Check ties, empty classes within cases, and all four pair partitions."""
    y = np.array([[[1, 0], [0, 1]], [[1, 1], [1, 1]], [[0, 0], [1, 0]]])
    p = np.array([[[.3, .3], [.8, .8]], [[.5, .9], [.7, .4]], [[.1, .9], [.8, .2]]])
    exact_error, exact_count = np.zeros((2, 2)), np.zeros((2, 2))
    same_error, same_count = np.zeros((2, 2)), np.zeros((2, 2))
    for c, t, i in zip(*np.where(y == 1)):
        for d, s, j in zip(*np.where(y == 0)):
            value = float(p[c, t, i] < p[d, s, j]) + .5 * (p[c, t, i] == p[d, s, j])
            exact_error[i, j] += value
            exact_count[i, j] += 1
            if c == d:
                same_error[i, j] += value
                same_count[i, j] += 1
    actual_error, actual_count = pair_matrices(y.reshape(-1, 2), p.reshape(-1, 2))
    actual_same = [pair_matrices(y[c], p[c]) for c in range(len(y))]
    np.testing.assert_array_equal(actual_error, exact_error)
    np.testing.assert_array_equal(actual_count, exact_count)
    np.testing.assert_array_equal(sum(a[0] for a in actual_same), same_error)
    np.testing.assert_array_equal(sum(a[1] for a in actual_same), same_count)
    np.testing.assert_allclose(endpoint_losses(y, p).sum(), exact_error.sum() / exact_count.sum(), atol=1e-15)


def context_masks(y: np.ndarray) -> dict[str, np.ndarray]:
    """Describe true schedules; these masks must never become inference features."""
    switches = (np.diff(y.astype(int), axis=1) != 0).sum(axis=1)
    on_count = y.sum(axis=1)
    masks = {name: np.broadcast_to(mask[:, None, :], y.shape) for name, mask in {
        'constant_off': on_count == 0,
        'constant_on': on_count == y.shape[1],
        'one_transition': switches == 1,
        'two_to_four_transitions': (switches >= 2) & (switches <= 4),
        'five_or_more_transitions': switches >= 5,
    }.items()}
    near = np.zeros_like(y, dtype=bool)
    lengths = np.zeros_like(y, dtype=np.int16)
    for c in range(y.shape[0]):
        for j in range(y.shape[2]):
            boundaries = np.flatnonzero(np.diff(y[c, :, j].astype(int))) + 1
            for t in boundaries:
                near[c, max(0, t - 3):min(y.shape[1], t + 4), j] = True
            for a, b in zip([0, *boundaries], [*boundaries, y.shape[1]]):
                lengths[c, a:b, j] = b - a
    masks.update({'within_3h_of_true_switch': near, 'away_from_true_switch': ~near,
                  'true_episode_1to3h': lengths <= 3,
                  'true_episode_4to12h': (lengths >= 4) & (lengths <= 12),
                  'true_episode_13to48h': (lengths >= 13) & (lengths <= 48),
                  'true_episode_49plush': lengths >= 49})
    return masks


def main() -> None:
    """Write deterministic exact decomposition and readable findings without fitting."""
    started = time.monotonic()
    synthetic_check()
    out = ROOT / 'artifacts' / RUN_ID
    out.mkdir(exist_ok=False)
    protocol_path = ROOT / 'docs/cuda_xgb608_capacity_protocol.json'
    assert digest(protocol_path) == PROTOCOL_SHA
    protocol = json.loads(protocol_path.read_text())
    sources = [Path(__file__), ROOT / 'artifacts/pair-calibration-diagnostic-001/diagnostic.py',
               ROOT / 'scripts/analyze_latent_shadow_residuals.py']
    report = {'run_id': RUN_ID, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'command': shlex.join([sys.executable, *sys.argv]), 'status': 'initializing',
              'environment': {'python': sys.version, 'platform': platform.platform(),
                              'machine': platform.machine(),
                              'packages': {n: importlib.metadata.version(n) for n in ['numpy', 'pandas', 'scikit-learn']}},
              'source': {str(p.relative_to(ROOT)): digest(p) for p in sources},
              'inputs': {str(protocol_path.relative_to(ROOT)): PROTOCOL_SHA},
              'training': None, 'seed': None, 'hyperparameter_search': None, 'kaggle_submission': None,
              'prediction_recipe': 'Existing accepted fixed 50% projected608anchor/50% rawGPU608, followed by fixed Vinje projection; arrays unchanged.',
              'formula': 'Directed E[i,j] counts positive unit i below negative unit j, plus half ties. All counts divided by total pooled positive-negative pairs. Endpoint context assigns half of each error to each endpoint.',
              'caveats': ['Both years are reused development evidence, not untouched tests.',
                          'Same case means the same 168-hour forecast start, not the same calendar hour.',
                          'Truth-derived switches and episode lengths are diagnostic categories only; no inference features or fitted correction are produced.',
                          'Cross-unit errors do not imply calibration alone can fix them; the monotone-unit ceiling is a loose necessary bound.'],
              'tests': ['Brute-force tie/empty-case-class/four-partition synthetic check passed'], 'years': {}}
    pair_rows, context_rows, unit_rows = [], [], []
    for year in [2021, 2022]:
        anchor = protocol['anchor'][str(year)]
        for key in ['manifest', 'prediction']:
            path = ROOT / anchor[key]
            assert digest(path) == anchor[key + '_sha256']
            report['inputs'][anchor[key]] = digest(path)
        with np.load(ROOT / anchor['prediction'], allow_pickle=False) as z:
            p, y, ids, units = z['predictions'], z['labels'], z['case_ids'], z['units'].tolist()
        assert p.shape == y.shape == (len(ids) * 168, 14)
        assert len(ids) == 365 and np.all(np.isfinite(p)) and p.min() >= 0 and p.max() <= 1
        assert np.isin(y, [0, 1]).all()
        yc, pc = y.reshape(-1, 168, 14), p.reshape(-1, 168, 14)
        error, count = pair_matrices(y, p)
        case_error, case_count = np.zeros_like(error), np.zeros_like(count)
        for c in range(len(ids)):
            ce, cn = pair_matrices(yc[c], pc[c])
            case_error += ce
            case_count += cn
        different_error, different_count = error - case_error, count - case_count
        assert (different_error >= 0).all() and (different_count >= 0).all()
        total_error, total_count = float(error.sum()), float(count.sum())
        auc = float(roc_auc_score(y.ravel(), p.ravel()))
        np.testing.assert_allclose(total_error / total_count, 1 - auc, atol=1e-13, rtol=0)
        endpoint = endpoint_losses(y, p).reshape(-1, 168, 14)
        np.testing.assert_allclose(endpoint.sum(), total_error / total_count, atol=1e-13, rtol=0)
        for j in range(14):
            np.testing.assert_allclose(1 - error[j, j] / count[j, j], roc_auc_score(y[:, j], p[:, j]), atol=1e-13, rtol=0)
            np.testing.assert_allclose(endpoint[:, :, j].sum(), (error[j].sum() + error[:, j].sum()) / (2 * total_count), atol=1e-13, rtol=0)
        partitions = {}
        diagonal = np.eye(14, dtype=bool)
        for name, e, n in [('same_case', case_error, case_count), ('different_case', different_error, different_count)]:
            for kind, mask in [('same_unit', diagonal), ('different_unit', ~diagonal), ('all_units', np.ones((14, 14), dtype=bool))]:
                es, ns = float(e[mask].sum()), float(n[mask].sum())
                partitions[name + '_' + kind] = {'error_pair_count': es, 'pair_count': ns,
                    'pooled_error_contribution': es / total_count, 'fraction_of_all_errors': es / total_error,
                    'fraction_of_all_pairs': ns / total_count, 'conditional_pair_auc': 1 - es / ns}
        for i in range(14):
            for j in range(14):
                pair_rows.append({'year': year, 'positive_unit': units[i], 'negative_unit': units[j],
                    'pair_count': count[i, j], 'error_pair_count': error[i, j],
                    'fraction_of_all_errors': error[i, j] / total_error,
                    'conditional_pair_auc': 1 - error[i, j] / count[i, j],
                    'same_case_pair_count': case_count[i, j], 'same_case_error_pair_count': case_error[i, j]})
        masks = context_masks(yc)
        context = {}
        for name, mask in masks.items():
            context[name] = {'hours_fraction': float(mask.mean()),
                             'endpoint_error_share': float(endpoint[mask].sum() / endpoint.sum())}
            context_rows.append({'year': year, 'unit': 'ALL', 'group': name, **context[name]})
        unit_summary = []
        for j, unit in enumerate(units):
            unit_loss = float(endpoint[:, :, j].sum())
            row = {'year': year, 'unit': unit, 'auc': float(1 - error[j, j] / count[j, j]),
                   'mean_probability_bias': float((p[:, j] - y[:, j]).mean()),
                   'positive_endpoint_error_share': float(error[j].sum() / total_error),
                   'negative_endpoint_error_share': float(error[:, j].sum() / total_error),
                   'half_endpoint_error_share': float(unit_loss / endpoint.sum()),
                   'near_switch_share_within_unit': float(endpoint[:, :, j][masks['within_3h_of_true_switch'][:, :, j]].sum() / unit_loss)}
            unit_summary.append(row)
            unit_rows.append(row)
            for name, mask in masks.items():
                context_rows.append({'year': year, 'unit': unit, 'group': name,
                    'hours_fraction': float(mask[:, :, j].mean()),
                    'endpoint_error_share': float(endpoint[:, :, j][mask[:, :, j]].sum() / unit_loss)})
        top_indices = np.argsort(error.ravel())[::-1][:12]
        top = [{'positive_unit': units[k // 14], 'negative_unit': units[k % 14],
                'fraction_of_all_errors': float(error.ravel()[k] / total_error)} for k in top_indices]
        result = {'micro_auc': auc, 'pooled_pair_error': 1 - auc, 'case_count': len(ids),
                  'case_ids': ids.tolist(), 'total_pair_count': total_count, 'total_error_pair_count': total_error,
                  'same_unit_fraction_of_all_errors': float(np.trace(error) / total_error),
                  'cross_unit_fraction_of_all_errors': float(1 - np.trace(error) / total_error),
                  'strictly_monotone_unit_map_auc_ceiling': float(1 - np.trace(error) / total_count),
                  'partitions': partitions, 'top_directed_pairs': top,
                  'unit_summary': unit_summary, 'all_unit_context': context}
        report['years'][str(year)] = result
        np.savez_compressed(out / f'pair_errors_{year}.npz', units=np.asarray(units), case_ids=ids,
            error_pair_count=error, pair_count=count, same_case_error_pair_count=case_error,
            same_case_pair_count=case_count, endpoint_error=endpoint)
        print(year, json.dumps({k: result[k] for k in ['micro_auc', 'cross_unit_fraction_of_all_errors', 'strictly_monotone_unit_map_auc_ceiling']}), flush=True)
        print('CASE_PARTITIONS', year, json.dumps(partitions), flush=True)
    pd.DataFrame(pair_rows).to_csv(out / 'directed_unit_pairs.csv', index=False)
    pd.DataFrame(context_rows).to_csv(out / 'transition_context.csv', index=False)
    pd.DataFrame(unit_rows).to_csv(out / 'unit_error_endpoints.csv', index=False)
    report.update(status='complete', runtime_seconds=time.monotonic() - started,
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  artifacts={p.name: digest(p) for p in out.iterdir() if p.is_file()})
    (out / 'manifest.json').write_text(json.dumps(report, indent=2))
    print('MANIFEST', digest(out / 'manifest.json'), 'RUNTIME', report['runtime_seconds'], flush=True)


if __name__ == '__main__':
    main()
