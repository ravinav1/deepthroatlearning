"""Pooled RankNet residual derivatives with uniform global pair sampling."""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from scipy.special import expit


def base_logits(probability: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    """Convert frozen base probabilities to finite float64 offsets."""
    if not np.isfinite(epsilon) or not 0 < epsilon < .5:
        raise ValueError('Logit clipping epsilon must be between zero and one half')
    probability = np.asarray(probability, dtype=np.float64)
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError('Base probabilities must be finite and in [0, 1]')
    safe = np.where(probability == 0, epsilon, np.where(probability == 1, 1 - epsilon, probability))
    return np.log(safe) - np.log1p(-safe)


def apply_residual(probability: np.ndarray, residual: np.ndarray) -> np.ndarray:
    """Apply a logit residual, preserving an all-zero baseline byte for byte."""
    probability, residual = np.asarray(probability), np.asarray(residual, dtype=np.float64)
    if probability.shape != residual.shape or not np.isfinite(residual).all():
        raise ValueError('Residual must be finite and match the base probability shape')
    logits = base_logits(probability)
    if np.all(residual == 0):
        return probability.copy()
    result = expit(logits + residual)
    result[residual == 0] = probability[residual == 0]
    return result


def pair_derivatives(scores: np.ndarray, positive: np.ndarray, negative: np.ndarray,
                     scale: float) -> tuple[float, np.ndarray, np.ndarray]:
    """Return scaled pair loss, exact gradient, and the exact Hessian diagonal.

    Each pair has loss log(1+exp(s_negative-s_positive)). The full Hessian
    contains negative cross-endpoint entries; the returned diagonal alone is
    not the full Hessian. Twice this diagonal is a PSD majorizer.
    """
    scores = np.asarray(scores, dtype=np.float64)
    positive, negative = np.asarray(positive), np.asarray(negative)
    if scores.ndim != 1 or positive.ndim != 1 or positive.shape != negative.shape or not len(positive):
        raise ValueError('Expected flat scores and equal nonempty pair arrays')
    if not np.isfinite(scores).all() or not np.isfinite(scale) or scale <= 0:
        raise ValueError('Scores and positive scaling must be finite')
    if not np.issubdtype(positive.dtype, np.integer) or not np.issubdtype(negative.dtype, np.integer):
        raise ValueError('Pair endpoints must be integer indices')
    if min(positive.min(), negative.min()) < 0 or max(positive.max(), negative.max()) >= len(scores):
        raise ValueError('Pair endpoint outside score array')
    if np.any(positive == negative):
        raise ValueError('A positive and negative endpoint cannot be the same row')
    margin = scores[positive] - scores[negative]
    wrong_probability = expit(-margin)
    curvature = wrong_probability * (1 - wrong_probability)
    gradient = (np.bincount(negative, weights=wrong_probability, minlength=len(scores)) -
                np.bincount(positive, weights=wrong_probability, minlength=len(scores))) * scale
    diagonal = (np.bincount(positive, weights=curvature, minlength=len(scores)) +
                np.bincount(negative, weights=curvature, minlength=len(scores))) * scale
    loss = float(np.logaddexp(0, -margin).sum() * scale)
    return loss, gradient, diagonal


def bce_derivatives(scores: np.ndarray, labels: np.ndarray,
                    weights: np.ndarray | None = None) -> tuple[float, np.ndarray, np.ndarray]:
    """Return sum-BCE loss and exact independent-row derivatives."""
    scores, labels = np.asarray(scores, dtype=np.float64), np.asarray(labels)
    if scores.ndim != 1 or scores.shape != labels.shape or not np.isfinite(scores).all():
        raise ValueError('Expected aligned finite score and label vectors')
    if not np.isin(labels, [0, 1]).all():
        raise ValueError('BCE labels must be binary')
    if weights is None:
        weights = np.ones_like(scores)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != scores.shape or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError('BCE weights must be aligned, finite and positive')
    probability = expit(scores)
    loss = float(((np.logaddexp(0, scores) - labels * scores) * weights).sum())
    return loss, (probability - labels) * weights, probability * (1 - probability) * weights


class PooledPairwiseObjective:
    """Fresh deterministic uniform pairs across all output rows each round.

    The fixed N/(2M) scale matches expected balanced-population gradient and
    diagonal curvature to sum BCE at zero margin. LightGBM receives twice the
    diagonal plus a negligible floor, an explicitly conservative Hessian
    majorizer; its optimizer dynamics therefore differ from the BCE control.
    """

    def __init__(self, labels: np.ndarray, initial_scores: np.ndarray,
                 pairs_per_round: int = 1_000_000, seed: int = 4173) -> None:
        self.labels = np.asarray(labels, dtype=np.uint8)
        self.initial_scores = np.asarray(initial_scores, dtype=np.float64)
        if self.labels.ndim != 1 or self.labels.shape != self.initial_scores.shape:
            raise ValueError('Expected aligned initial score and label vectors')
        if not np.isin(labels, [0, 1]).all() or not np.isfinite(self.initial_scores).all():
            raise ValueError('Invalid labels or initial scores')
        self.positive = np.flatnonzero(self.labels == 1)
        self.negative = np.flatnonzero(self.labels == 0)
        if not len(self.positive) or not len(self.negative) or pairs_per_round <= 0:
            raise ValueError('Both classes and a positive pair count are required')
        self.pairs_per_round, self.seed = int(pairs_per_round), int(seed)
        self.trace: list[dict[str, Any]] = []

    def sample_pairs(self, round_index: int) -> tuple[np.ndarray, np.ndarray]:
        """Draw the product of uniform positive and negative populations."""
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, round_index]))
        positive = self.positive[rng.integers(len(self.positive), size=self.pairs_per_round)]
        negative = self.negative[rng.integers(len(self.negative), size=self.pairs_per_round)]
        return positive, negative

    def __call__(self, scores: np.ndarray, dataset: Any) -> tuple[np.ndarray, np.ndarray]:
        """Return LightGBM derivatives for raw scores including the frozen offset."""
        if not self.trace:
            np.testing.assert_array_equal(dataset.get_label(), self.labels)
            np.testing.assert_array_equal(scores, self.initial_scores)
        round_index = len(self.trace)
        positive, negative = self.sample_pairs(round_index)
        scale = len(scores) / (2 * self.pairs_per_round)
        loss, gradient, diagonal = pair_derivatives(scores, positive, negative, scale)
        hessian = 2 * diagonal + 1e-12
        hasher = hashlib.sha256()
        hasher.update(positive.astype('<i8', copy=False).tobytes())
        hasher.update(negative.astype('<i8', copy=False).tobytes())
        self.trace.append({'round': round_index + 1, 'pair_sha256': hasher.hexdigest(),
                           'pairs': self.pairs_per_round, 'scale': scale,
                           'loss_per_output': loss / len(scores),
                           'gradient_sum': float(gradient.sum()),
                           'mean_absolute_gradient': float(np.abs(gradient).mean()),
                           'mean_majorized_hessian': float(hessian.mean()),
                           'rows_touched': int(np.count_nonzero(diagonal))})
        return gradient, hessian


class PointwiseBCEObjective:
    """Matched residual control with exact BCE derivatives and the same offset."""

    def __init__(self, labels: np.ndarray, initial_scores: np.ndarray,
                 class_balanced: bool = False) -> None:
        if not np.isin(labels, [0, 1]).all():
            raise ValueError('BCE labels must be binary before conversion')
        self.labels = np.asarray(labels, dtype=np.uint8)
        self.initial_scores = np.asarray(initial_scores, dtype=np.float64)
        if self.labels.ndim != 1 or self.labels.shape != self.initial_scores.shape or not np.isfinite(self.initial_scores).all():
            raise ValueError('Expected aligned labels and finite initial scores')
        self.weights = np.ones_like(self.initial_scores)
        if class_balanced:
            n_positive, n_negative = int(self.labels.sum()), int((self.labels == 0).sum())
            if not n_positive or not n_negative:
                raise ValueError('Balanced BCE requires both classes')
            self.weights[self.labels == 1] = len(self.labels) / (2 * n_positive)
            self.weights[self.labels == 0] = len(self.labels) / (2 * n_negative)
        self.trace: list[dict[str, Any]] = []

    def __call__(self, scores: np.ndarray, dataset: Any) -> tuple[np.ndarray, np.ndarray]:
        """Return raw-score BCE derivatives; assert offset integration initially."""
        if not self.trace:
            np.testing.assert_array_equal(dataset.get_label(), self.labels)
            np.testing.assert_array_equal(scores, self.initial_scores)
        loss, gradient, hessian = bce_derivatives(scores, self.labels, self.weights)
        self.trace.append({'round': len(self.trace) + 1, 'loss_per_output': loss / len(scores),
                           'mean_absolute_gradient': float(np.abs(gradient).mean()),
                           'mean_hessian': float(hessian.mean())})
        return gradient, hessian + 1e-12
