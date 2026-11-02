"""Convex residual CRF over three binary generators with exact marginals."""
from __future__ import annotations

from itertools import product
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

STATES = np.asarray(list(product([0., 1.], repeat=3)))
PAIRS = np.column_stack([STATES[:, a] * STATES[:, b] for a, b in [(0, 1), (0, 2), (1, 2)]])
STARTS = (1 - STATES[:, None, :]) * STATES[None, :, :]
STOPS = STATES[:, None, :] * (1 - STATES[None, :, :])
TRANSITIONS = np.concatenate([STARTS, STOPS], axis=2)
RIDGE = .01


def logits(probability: np.ndarray) -> np.ndarray:
    """Stabilize legal probabilities only for log-potential arithmetic."""
    p = np.asarray(probability, dtype=np.float64)
    if p.ndim != 3 or p.shape[2] != 3 or not np.isfinite(p).all() or p.min() < 0 or p.max() > 1:
        raise ValueError('Expected finite case/hour/three-unit probabilities')
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p) - np.log1p(-p)


def forward_backward(probability: np.ndarray, z: np.ndarray, theta: np.ndarray) -> tuple:
    """Return log partition, state marginals and expected residual statistics."""
    base = logits(probability)
    n, hours, _ = base.shape
    if z.shape != (n, hours, 4) or not np.isfinite(z).all() or np.shape(theta) != (36,):
        raise ValueError('Expected exactly four covariates and36 coefficients')
    pair_w, transition_w = theta[:12].reshape(3, 4), theta[12:].reshape(6, 4)
    state_energy = np.einsum('ntg,sg->nts', base, STATES) + np.einsum('ntd,gd,sg->nts', z, pair_w, PAIRS)
    transition_energy = np.einsum('ntd,gd,abg->ntab', z, transition_w, TRANSITIONS)
    alpha = np.empty((n, hours, 8)); beta = np.zeros_like(alpha)
    alpha[:, 0] = state_energy[:, 0]
    for t in range(1, hours):
        alpha[:, t] = state_energy[:, t] + logsumexp(alpha[:, t - 1, :, None] + transition_energy[:, t], axis=1)
    log_z = logsumexp(alpha[:, -1], axis=1)
    for t in range(hours - 2, -1, -1):
        beta[:, t] = logsumexp(transition_energy[:, t + 1] + state_energy[:, t + 1, None, :] + beta[:, t + 1, None, :], axis=2)
    marginal = np.exp(alpha + beta - log_z[:, None, None])
    expected_pair = np.einsum('nts,sg,ntd->gd', marginal, PAIRS, z)
    expected_transition = np.zeros((6, 4))
    for t in range(1, hours):
        edge = np.exp(alpha[:, t - 1, :, None] + transition_energy[:, t] +
                      state_energy[:, t, None, :] + beta[:, t, None, :] - log_z[:, None, None])
        expected_transition += np.einsum('nab,abg,nd->gd', edge, TRANSITIONS, z[:, t])
    return log_z, marginal, np.concatenate([expected_pair.ravel(), expected_transition.ravel()])


def observed_statistics(y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Aggregate exactly the same state/transition statistics on observed paths."""
    y = np.asarray(y, dtype=np.float64)
    if not np.isin(y, [0, 1]).all():
        raise ValueError('Targets must be binary')
    pair = np.stack([y[:, :, a] * y[:, :, b] for a, b in [(0, 1), (0, 2), (1, 2)]], axis=2)
    transition = np.concatenate([(1 - y[:, :-1]) * y[:, 1:], y[:, :-1] * (1 - y[:, 1:])], axis=2)
    return np.concatenate([np.einsum('ntg,ntd->gd', pair, z).ravel(),
                           np.einsum('ntg,ntd->gd', transition, z[:, 1:]).ravel()])


def objective_gradient(theta: np.ndarray, probability: np.ndarray, y: np.ndarray,
                       z: np.ndarray, ridge: float = RIDGE) -> tuple[float, np.ndarray]:
    """Compute per-hour conditional NLL plus ridge and its analytic gradient."""
    log_z, _, expected = forward_backward(probability, z, theta)
    observed = observed_statistics(y, z)
    n, hours, _ = y.shape
    score = np.sum(logits(probability) * y) + theta @ observed
    loss = (log_z.sum() - score) / (n * hours) + ridge * np.sum(theta ** 2)
    gradient = (expected - observed) / (n * hours) + 2 * ridge * theta
    return float(loss), gradient


def unary_objective(theta: np.ndarray, probability: np.ndarray, y: np.ndarray,
                    z: np.ndarray, ridge: float = RIDGE) -> tuple[float, np.ndarray]:
    """Matched contextual unary control, with twelve independent coefficients."""
    score = logits(probability) + np.einsum('ntd,gd->ntg', z, theta.reshape(3, 4))
    loss = np.mean(np.sum(np.logaddexp(0, score) - y * score, axis=2)) + ridge * np.sum(theta ** 2)
    gradient = np.einsum('ntg,ntd->gd', expit(score) - y, z) / (y.shape[0] * y.shape[1])
    return float(loss), gradient.ravel() + 2 * ridge * theta


def fit_model(probability: np.ndarray, y: np.ndarray, z: np.ndarray, kind: str) -> dict:
    """Fit a fixed convex residual model with no screen data or early stopping."""
    if kind not in ['joint', 'unary']:
        raise ValueError('Unknown model family')
    function, size = (objective_gradient, 36) if kind == 'joint' else (unary_objective, 12)
    result = minimize(function, np.zeros(size), args=(probability, y, z), jac=True, method='L-BFGS-B',
                      options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-6})
    if not result.success or not np.isfinite(result.fun):
        raise ValueError(f'Residual optimizer failed: {result.message}')
    return {'kind': kind, 'theta': result.x.tolist(), 'objective': float(result.fun),
            'iterations': int(result.nit), 'gradient_max_abs': float(np.abs(result.jac).max()),
            'success': bool(result.success), 'message': str(result.message)}


def predict(probability: np.ndarray, z: np.ndarray, model: dict) -> np.ndarray:
    """Return exact unmodified anchor at zero residual, otherwise model marginals."""
    theta = np.asarray(model['theta'])
    if np.count_nonzero(theta) == 0:
        return np.array(probability, copy=True)
    if model['kind'] == 'unary':
        return expit(logits(probability) + np.einsum('ntd,gd->ntg', z, theta.reshape(3, 4)))
    _, marginal, _ = forward_backward(probability, z, theta)
    return np.clip(marginal @ STATES, 0, 1)
