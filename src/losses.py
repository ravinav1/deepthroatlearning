"""Optional focal loss for LightGBM's raw binary scores."""

import numpy as np
from scipy.special import expit


def focal_loss(gamma=2.0):
    """Unweighted focal loss: -(1 - p_true)**gamma * log(p_true)."""
    def objective(labels, scores):
        sign = 2.0 * labels - 1.0
        p = expit(sign * scores)
        q = expit(-sign * scores)
        log_p = -np.logaddexp(0, -sign * scores)
        gradient = sign * q**gamma * (gamma * p * log_p - q)
        hessian = p * q**gamma * (
            q * (1 + 2 * gamma + gamma * log_p) - gamma**2 * p * log_p
        )
        # Focal loss can have negative curvature; use a positive floor for tree fitting.
        return gradient, np.maximum(hessian, 1e-6)
    return objective
