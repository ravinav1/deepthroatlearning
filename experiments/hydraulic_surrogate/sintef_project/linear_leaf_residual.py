"""Small independently fitted unit residuals with linear or constant leaves."""
from __future__ import annotations

from typing import Callable

import lightgbm as lgb
import numpy as np

from sintef_project.pooled_pairwise_residual import PointwiseBCEObjective, base_logits


def select_gain_features(gains: np.ndarray, count: int = 32) -> np.ndarray:
    """Choose descending pre-2018 gains, breaking ties by original column index."""
    values = np.asarray(gains, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('Feature gains must be a finite nonnegative vector')
    if not isinstance(count, int) or not 1 <= count <= len(values):
        raise ValueError('Invalid fixed feature count')
    return np.lexsort((np.arange(len(values)), -values))[:count]


def residual_sample(rows: int, fraction: float = .4, seed: int = 4173) -> np.ndarray:
    """Draw one sorted, shared case-hour sample without unit-specific RNG calls."""
    if rows < 1 or not 0 < fraction <= 1 or int(rows*fraction) < 1:
        raise ValueError('Invalid residual sample scope')
    return np.sort(np.random.default_rng(seed).choice(rows, size=int(rows*fraction), replace=False))


def fit_normalizer(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estimate column mean/std only from fitted training rows; keep missing as NaN."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not len(values) or np.isinf(values).any():
        raise ValueError('Expected nonempty numeric training rows without infinity')
    present = ~np.isnan(values)
    count = present.sum(axis=0)
    if np.any(count == 0):
        raise ValueError('An all-missing selected training feature cannot be normalized')
    mean = np.where(present, values, 0).sum(axis=0) / count
    centered = np.where(present, values-mean, 0)
    scale = np.sqrt((centered*centered).sum(axis=0)/count)
    scale = np.where(scale > 1e-12, scale, 1.)
    return mean, scale


def normalize(values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Apply the sealed training normalizer without clipping or imputing missing zeros."""
    values = np.asarray(values, dtype=np.float64)
    mean, scale = np.asarray(mean), np.asarray(scale)
    if values.ndim != 2 or mean.shape != scale.shape or mean.shape != (values.shape[1],):
        raise ValueError('Normalizer shape differs from selected numeric features')
    if np.isinf(values).any() or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('Invalid feature values or training normalizer')
    return ((values-mean)/scale).astype(np.float32)


def fit_unit(values: np.ndarray, labels: np.ndarray, probability: np.ndarray,
             parameters: dict, rounds: int, linear: bool,
             progress: Callable | None = None) -> tuple[lgb.Booster, list[dict]]:
    """Fit one BCE residual; assert that the first gradient sees the frozen offset."""
    labels = np.asarray(labels)
    if values.ndim != 2 or labels.shape != (len(values),) or probability.shape != labels.shape:
        raise ValueError('Unit rows, labels and probabilities are not aligned')
    initial = base_logits(probability)
    state = PointwiseBCEObjective(labels, initial)
    dataset_parameters = {key: parameters[key] for key in
                          ['num_threads', 'min_data_in_leaf', 'max_bin', 'data_random_seed']}
    dataset_parameters.update(linear_tree=linear, use_missing=True, zero_as_missing=False)
    data = lgb.Dataset(values, label=labels, init_score=initial,
                       params=dataset_parameters, free_raw_data=True)
    def objective(scores, dataset):
        return state(scores, dataset)
    params = dict(parameters)
    params.update(linear_tree=linear, objective=objective)
    model = lgb.train(params, data, num_boost_round=rounds,
                      callbacks=[] if progress is None else [progress])
    if model.current_iteration() != rounds or len(state.trace) != rounds:
        raise ValueError('The fixed residual tree count did not complete')
    return model, state.trace


def leaf_geometry(model: lgb.Booster) -> dict:
    """Count serialized linear coefficients; the first linear-tree booster is constant."""
    counts = []
    for tree in model.dump_model()['tree_info']:
        pending = [tree['tree_structure']]
        count = 0
        while pending:
            node = pending.pop()
            if 'split_index' in node:
                pending.extend([node['left_child'], node['right_child']])
            else:
                coefficients = np.asarray(node.get('leaf_coeff', []), dtype=np.float64)
                if not np.isfinite(coefficients).all():
                    raise ValueError('Non-finite serialized linear leaf coefficient')
                count += int(np.count_nonzero(coefficients))
        counts.append(count)
    return {'nonzero_linear_coefficients_by_tree': counts,
            'total_nonzero_linear_coefficients': sum(counts)}
