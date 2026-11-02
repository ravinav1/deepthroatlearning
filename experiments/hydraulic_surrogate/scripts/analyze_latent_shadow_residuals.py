"""Describe rejected splice errors by true schedule structure without fitting."""
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
UNITS = ['Haukeli_G1', 'Lio_G1', 'Hogga_G1']


def digest(path: Path) -> str:
    """Hash a frozen dependency."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def endpoint_losses(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Attribute half each misordered/tied pair to each endpoint, normalized."""
    positive, negative = np.sort(p[y == 1]), np.sort(p[y == 0])
    losses = np.empty(p.shape)
    values = p[y == 1]
    losses[y == 1] = len(negative) - .5 * (np.searchsorted(negative, values, 'left') + np.searchsorted(negative, values, 'right'))
    values = p[y == 0]
    losses[y == 0] = .5 * (np.searchsorted(positive, values, 'left') + np.searchsorted(positive, values, 'right'))
    losses /= 2 * len(positive) * len(negative)
    np.testing.assert_allclose(losses.sum(), 1 - roc_auc_score(y.ravel(), p.ravel()), atol=1e-13, rtol=0)
    return losses


def main() -> None:
    """Retain exact residual allocation and label-defined descriptive categories."""
    out = ROOT / 'artifacts/latent-shadow-residual-diagnostic-001'
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    report = {'run_id': out.name, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'command': shlex.join([sys.executable, *sys.argv]), 'status': 'initializing',
              'source': {'scripts/analyze_latent_shadow_residuals.py': digest(Path(__file__))},
              'environment': {'python': sys.version, 'platform': platform.platform(), 'machine': platform.machine(),
                              'packages': {n: importlib.metadata.version(n) for n in ['numpy', 'pandas', 'scikit-learn']}},
              'seed': None, 'training': None, 'kaggle': None, 'inputs': {}, 'results': {},
              'caveat': 'Read-only reused-development error analysis. All truth-defined categories are diagnostic only, never inference features or new tuning.'}
    summaries = []
    for year in [2021, 2022]:
        anchor_path = ROOT / f'artifacts/lp-context-joint-comparison-001/selected_validation_{year}.npz'
        candidate_dir = ROOT / f'artifacts/latent-shadow-splice-{year}-001'
        manifest = json.loads((candidate_dir / 'manifest.json').read_text())
        report['inputs'][str(anchor_path.relative_to(ROOT))] = digest(anchor_path)
        report['inputs'][str((candidate_dir / 'manifest.json').relative_to(ROOT))] = digest(candidate_dir / 'manifest.json')
        for name in [f'selected_validation_{year}.npz', f'raw_validation_{year}.npz']:
            if digest(candidate_dir / name) != manifest['artifacts'][name]:
                raise ValueError('Changed saved candidate arrays')
        with np.load(anchor_path, allow_pickle=False) as z:
            p, y, ids, units = z['predictions'], z['labels'], z['case_ids'], z['units'].tolist()
        with np.load(candidate_dir / f'selected_validation_{year}.npz', allow_pickle=False) as z:
            selected = z['predictions']
            np.testing.assert_array_equal(y, z['labels']); np.testing.assert_array_equal(ids, z['case_ids'])
        with np.load(candidate_dir / f'raw_validation_{year}.npz', allow_pickle=False) as z:
            alternatives = {name: z[name] for name in ['real', 'constant', 'permuted']}
        variants = {'anchor': p, 'real': selected}
        indices = [units.index(u) for u in UNITS]
        for name in ['constant', 'permuted']:
            variants[name] = p.copy()
            variants[name][:, indices] = .95 * p[:, indices] + .05 * alternatives[name]
        losses = {name: endpoint_losses(y, probability).reshape(len(ids), 168, 14)
                  for name, probability in variants.items()}
        yc = y.reshape(len(ids), 168, 14)
        switches = np.sum(np.diff(yc.astype(int), axis=1) != 0, axis=1)
        on_count = yc.sum(axis=1)
        case_groups = {'constant_off': on_count == 0, 'constant_on': on_count == 168,
                       'one_transition': switches == 1, 'two_to_four_transitions': (switches >= 2) & (switches <= 4),
                       'five_or_more_transitions': switches >= 5}
        near = np.zeros_like(yc, dtype=bool)
        episode_length = np.zeros_like(yc, dtype=int)
        for c in range(len(ids)):
            for j in range(14):
                boundaries = np.flatnonzero(np.diff(yc[c, :, j].astype(int))) + 1
                for t in boundaries:
                    near[c, max(0, t - 3):min(168, t + 4), j] = True
                for a, b in zip([0, *boundaries], [*boundaries, 168]):
                    episode_length[c, a:b, j] = b - a
        masks = {name: np.broadcast_to(mask[:, None, :], yc.shape) for name, mask in case_groups.items()}
        masks.update({'within_3h_of_true_switch': near, 'away_from_true_switch': ~near,
                      'true_episode_1to3h': episode_length <= 3,
                      'true_episode_4to12h': (episode_length >= 4) & (episode_length <= 12),
                      'true_episode_13to48h': (episode_length >= 13) & (episode_length <= 48),
                      'true_episode_49plush': episode_length >= 49})
        per_unit = []
        for j, unit in enumerate(units):
            loss = float(losses['anchor'][:, :, j].sum())
            pc = p.reshape(len(ids), 168, 14)[:, :, j]
            row = {'unit': unit, 'anchor_error_share': loss / losses['anchor'].sum(),
                   'anchor_auc': float(roc_auc_score(y[:, j], p[:, j])),
                   'selected_auc': float(roc_auc_score(y[:, j], selected[:, j])),
                   'constant_case_fraction': float(((on_count[:, j] == 0) | (on_count[:, j] == 168)).mean()),
                   'mean_true_switches': float(switches[:, j].mean()),
                   'high_confidence_wrong_hours': int((((pc > .9) & (yc[:, :, j] == 0)) |
                                                      ((pc < .1) & (yc[:, :, j] == 1))).sum()),
                   'groups': {}}
            for group, mask in masks.items():
                m = mask[:, :, j]
                row['groups'][group] = {'hours': int(m.sum()), 'anchor_error_share_within_unit': float(losses['anchor'][:, :, j][m].sum() / loss),
                                        'pair_error': {name: float(values[:, :, j][m].sum()) for name, values in losses.items()},
                                        'anchor_threshold_errors': int(((pc > .5) != yc[:, :, j])[m].sum())}
                summaries.append({'year': year, 'unit': unit, 'group': group, **row['groups'][group],
                                  **{f'{name}_pair_error': float(values[:, :, j][m].sum()) for name, values in losses.items()}})
            per_unit.append(row)
        report['results'][str(year)] = {'auc': {name: float(1 - values.sum()) for name, values in losses.items()},
                                       'case_ids': ids.tolist(), 'units': per_unit,
                                       'all_unit_groups': {group: {'hours_fraction': float(mask.mean()),
                                           'error_share': float(losses['anchor'][mask].sum() / losses['anchor'].sum())} for group, mask in masks.items()}}
    pd.DataFrame([{k: v for k, v in row.items() if k != 'pair_error'} for row in summaries]).to_csv(out / 'groups.csv', index=False)
    report.update(status='complete', runtime_seconds=time.monotonic() - started,
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  artifacts={'groups.csv': digest(out / 'groups.csv')})
    (out / 'manifest.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({year: result['all_unit_groups'] for year, result in report['results'].items()}, indent=2))


if __name__ == '__main__':
    main()
