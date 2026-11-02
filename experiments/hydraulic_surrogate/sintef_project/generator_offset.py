"""Convex bounded generator offsets for pooled ranking and matched balanced BCE."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def group_indices(units: list[str]) -> np.ndarray:
    """Assign one group per unit except the three Vinje units share one bias."""
    if len(units) != 14 or len(set(units)) != 14 or sum(u.startswith('Vinje_') for u in units) != 3:
        raise ValueError('Expected fourteen distinct units including three Vinje units')
    names = ['Vinje' if u.startswith('Vinje_') else u for u in units]
    unique = list(dict.fromkeys(names))
    return np.asarray([unique.index(u) for u in names], dtype=np.int64)


def apply_offsets(probability: np.ndarray, offsets: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Apply strictly monotone odds scaling, preserving endpoints and zero bias."""
    p = np.asarray(probability)
    b = np.asarray(offsets, dtype=np.float64)[groups]
    if p.shape[-1] != len(groups) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Invalid aligned probabilities')
    if not np.isfinite(b).all() or np.max(np.abs(b)) > .5 + 1e-12:
        raise ValueError('Offsets must satisfy the frozen half-logit bound')
    if np.all(b == 0):
        return p.copy()
    p = p.astype(np.float64)
    e = np.exp(-np.abs(b))
    result = np.where(b >= 0, p / (p + (1-p)*e), p*e / (1-p + p*e))
    result[..., b == 0] = p[..., b == 0]
    return result


def pair_loss_gradient(offsets: np.ndarray, margins: np.ndarray, positive_groups: np.ndarray,
                       negative_groups: np.ndarray) -> tuple[float, np.ndarray]:
    """Exact mean logistic pair loss and full twelve-parameter gradient."""
    wrong_margin = margins + offsets[negative_groups] - offsets[positive_groups]
    error = expit(wrong_margin)
    gradient = (np.bincount(negative_groups, weights=error, minlength=len(offsets)) -
                np.bincount(positive_groups, weights=error, minlength=len(offsets))) / len(error)
    return float(np.logaddexp(0, wrong_margin).mean()), gradient


def balanced_loss_gradient(offsets: np.ndarray, logits: np.ndarray, labels: np.ndarray,
                           groups: np.ndarray, weights: np.ndarray) -> tuple[float, np.ndarray]:
    """Exact globally class-balanced mean BCE and group gradient."""
    scores = logits + offsets[groups]
    loss = np.mean(weights * (np.logaddexp(0, scores) - labels*scores))
    gradient = np.bincount(groups, weights=weights*(expit(scores)-labels), minlength=len(offsets))/len(labels)
    return float(loss), gradient


def ridge_loss_gradient_hessian(offsets: np.ndarray, strength: float = .01) -> tuple[float, np.ndarray, np.ndarray]:
    """Shrink toward zero with half lambda times mean squared group bias."""
    return (float(.5*strength*np.mean(offsets**2)), strength*offsets/len(offsets),
            np.eye(len(offsets))*strength/len(offsets))


def pair_hessian(offsets: np.ndarray, margins: np.ndarray, positive_groups: np.ndarray,
                 negative_groups: np.ndarray) -> np.ndarray:
    """Return the exact full group Hessian, including negative cross terms."""
    error = expit(margins + offsets[negative_groups] - offsets[positive_groups])
    weights = error*(1-error)/len(error)
    hessian = np.zeros((len(offsets),len(offsets)))
    np.add.at(hessian,(positive_groups,positive_groups),weights)
    np.add.at(hessian,(negative_groups,negative_groups),weights)
    np.add.at(hessian,(positive_groups,negative_groups),-weights)
    np.add.at(hessian,(negative_groups,positive_groups),-weights)
    return hessian


def balanced_hessian(offsets: np.ndarray, logits: np.ndarray, groups: np.ndarray,
                     weights: np.ndarray) -> np.ndarray:
    """Return the exact diagonal group Hessian for balanced BCE."""
    probability = expit(logits + offsets[groups])
    return np.diag(np.bincount(groups,weights=weights*probability*(1-probability),
                              minlength=len(offsets))/len(logits))


def fit_offsets(probability: np.ndarray, labels: np.ndarray, units: list[str], kind: str,
                pairs: int = 1_000_000, seed: int = 4173) -> tuple[np.ndarray, dict, dict]:
    """Fit one fixed convex model from zero, with no restarts or validation tuning."""
    p = np.asarray(probability, dtype=np.float64)
    y = np.asarray(labels).ravel()
    if p.shape != labels.shape or not np.isfinite(p).all() or np.any((p <= 0) | (p >= 1)):
        raise ValueError('Training probabilities must be aligned and strictly interior')
    if not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError('Both binary classes required')
    ug = group_indices(units)
    groups = np.tile(ug, p.size//len(ug))
    logits = (np.log(p)-np.log1p(-p)).ravel()
    evidence = {}
    if kind == 'pairwise':
        rng = np.random.default_rng(seed)
        positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        pos = positive[rng.integers(len(positive), size=pairs)]
        neg = negative[rng.integers(len(negative), size=pairs)]
        margins = logits[neg]-logits[pos]
        evidence = {'positive': pos, 'negative': neg}
        raw_objective = lambda b: pair_loss_gradient(b, margins, groups[pos], groups[neg])
        raw_hessian = lambda b: pair_hessian(b,margins,groups[pos],groups[neg])
    elif kind == 'balanced_bce':
        weights = np.where(y == 1, len(y)/(2*y.sum()), len(y)/(2*(len(y)-y.sum())))
        raw_objective = lambda b: balanced_loss_gradient(b, logits, y, groups, weights)
        raw_hessian = lambda b: balanced_hessian(b,logits,groups,weights)
    else:
        raise ValueError('Unknown frozen objective')
    def objective(b):
        loss, gradient = raw_objective(b)
        penalty, penalty_gradient, _ = ridge_loss_gradient_hessian(b)
        return loss + penalty, gradient + penalty_gradient
    result = minimize(objective, np.zeros(12), jac=True, method='L-BFGS-B', bounds=[(-.5,.5)]*12,
                      options={'maxiter':500, 'maxls':40, 'ftol':1e-14, 'gtol':1e-9})
    loss, gradient = objective(result.x)
    projected = gradient.copy()
    projected[(result.x <= -.5+1e-12) & (gradient > 0)] = 0
    projected[(result.x >= .5-1e-12) & (gradient < 0)] = 0
    trace = {'success':bool(result.success), 'message':str(result.message), 'iterations':int(result.nit),
             'evaluations':int(result.nfev), 'objective':loss, 'projected_gradient_max':float(np.max(np.abs(projected))),
             'offsets':result.x.tolist(), 'groups':ug.tolist(), 'ridge_strength':.01,
             'ridge_penalty':ridge_loss_gradient_hessian(result.x)[0],
             'hessian_minimum_eigenvalue':float(np.linalg.eigvalsh(raw_hessian(result.x)+ridge_loss_gradient_hessian(result.x)[2]).min())}
    if not result.success or trace['projected_gradient_max'] > 1e-7:
        raise RuntimeError('Frozen convex fit did not meet convergence gate: '+str(trace))
    return result.x, trace, evidence


def verify_ranks(before: np.ndarray, after: np.ndarray, units: list[str]) -> dict:
    """Check actual strict ordering and ties, plus the projected Vinje order."""
    for j in range(len(units)):
        x, y = before[...,j].ravel(), after[...,j].ravel()
        order = np.argsort(x, kind='stable')
        a, b = np.diff(x[order]), np.diff(y[order])
        if not np.all(b[a > 0] > 0) or not np.all(b[a == 0] == 0):
            raise ValueError('Actual within-unit ranks or ties changed')
    vinje = [units.index(f'Vinje_G{j}') for j in [1,2,3]]
    if np.any(np.diff(after[...,vinje], axis=-1) > 0):
        raise ValueError('Vinje order changed')
    return {'all_unit_ranks_and_ties_exact':True, 'vinje_order_preserved':True}
