"""Deterministic probability transforms confined to each scheduling case."""
from __future__ import annotations

from collections.abc import Sequence
import numpy as np

HORIZON = 168
DEFAULT_UNITS = (
    "Hogga_G1", "Tokke_G1", "Tokke_G2", "Tokke_G3", "Tokke_G4",
    "Lio_G1", "Byrte_G1", "Vinje_G1", "Vinje_G2", "Vinje_G3",
    "Haukeli_G1", "Songa_G1", "Kjela_G1", "Vesle Kjela_G1",
)


def _validate(p: np.ndarray) -> np.ndarray:
    values = np.asarray(p, dtype=np.float64)
    if values.ndim != 2 or not values.size or len(values) % HORIZON:
        raise ValueError("Expected nonempty (n_cases * 168, n_units) probabilities")
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Probabilities must be finite and within [0, 1]")
    return values


def smooth_predictions(p: np.ndarray, window: int, weight: float) -> np.ndarray:
    """Blend probabilities with a centered, edge-normalized within-case mean.

    The odd-width window includes the current hour. Boundary windows use only
    available hours of that case, with their actual count as denominator.
    No adjacent-case data or labels are read. Input arrays are never mutated.
    """
    values = _validate(p)
    if isinstance(window, bool) or not isinstance(window, (int, np.integer)):
        raise ValueError("window must be an odd integer between 1 and 167")
    if window < 1 or window >= HORIZON or window % 2 != 1:
        raise ValueError("window must be an odd integer between 1 and 167")
    if not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("weight must be finite and within [0, 1]")
    if weight == 0 or window == 1:
        return values.copy()
    cube = values.reshape(-1, HORIZON, values.shape[1])
    prefix = np.concatenate([np.zeros_like(cube[:, :1]), cube.cumsum(axis=1)], axis=1)
    hours = np.arange(HORIZON)
    left = np.maximum(hours - window // 2, 0)
    right = np.minimum(hours + window // 2 + 1, HORIZON)
    local_mean = (prefix[:, right] - prefix[:, left]) / (right - left)[None, :, None]
    result = (1 - weight) * cube + weight * local_mean
    return np.clip(result.reshape(values.shape), 0, 1)


def project_vinje(p: np.ndarray, units: Sequence[str] = DEFAULT_UNITS) -> np.ndarray:
    """L2-project Vinje G1/G2/G3 probabilities onto their decreasing order.

    This empirical constraint is an optional ablation, not a physical certainty.
    Other generators are unchanged. The three-unit sum is preserved to floating
    precision; no probability is reassigned to a different generator.
    """
    values = _validate(p).copy()
    if len(units) != values.shape[1] or len(set(units)) != len(units):
        raise ValueError("Unique unit names must match probability columns")
    try:
        indices = [list(units).index(f"Vinje_G{i}") for i in (1, 2, 3)]
    except ValueError as error:
        raise ValueError("All three named Vinje generators are required") from error
    a, b, c = values[:, indices].T
    projected = np.column_stack([a, b, c])
    left_violation = a < b
    left_mean = (a + b) / 2
    right_violation = (~left_violation) & (b < c)
    right_mean = (b + c) / 2
    pool_all = (left_violation & (left_mean < c)) | (right_violation & (a < right_mean))
    pool_left = left_violation & ~pool_all
    pool_right = right_violation & ~pool_all
    projected[pool_left, :2] = left_mean[pool_left, None]
    projected[pool_right, 1:] = right_mean[pool_right, None]
    projected[pool_all] = ((a + b + c) / 3)[pool_all, None]
    values[:, indices] = projected
    return np.clip(values, 0, 1)
