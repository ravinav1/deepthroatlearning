"""Soft correction of a named probability-order violation, with no fitted state."""

from collections.abc import Sequence

import numpy as np


def soften_pair_order_violations(probabilities: np.ndarray, units: Sequence[str], alpha: float,
                                 first: str = "Tokke_G1", second: str = "Tokke_G2") -> np.ndarray:
    """Move only reversed pairs toward their mean; preserve all other values.

    The direction is an empirical label-priority hypothesis, not a physical
    impossibility. Alpha0 leaves everything unchanged; alpha1 pools violations.
    """
    values = np.asarray(probabilities)
    names = list(units)
    if values.ndim != 2 or values.shape[1] != len(names) or len(set(names)) != len(names):
        raise ValueError("Expected a row-by-unique-unit probability matrix")
    if not np.issubdtype(values.dtype, np.floating) or not np.isfinite(values).all():
        raise ValueError("Probabilities must be finite floating-point values")
    if (values < 0).any() or (values > 1).any() or not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("Probabilities and alpha must lie in [0,1]")
    if first == second or first not in names or second not in names:
        raise ValueError("Expected two distinct named units")
    output = values.copy()
    if alpha == 0:
        return output
    a, b = names.index(first), names.index(second)
    violation = values[:, a] < values[:, b]
    pair_mean = (values[violation, a] + values[violation, b]) * .5
    output[violation, a] = (1 - alpha) * values[violation, a] + alpha * pair_mean
    output[violation, b] = (1 - alpha) * values[violation, b] + alpha * pair_mean
    return output
