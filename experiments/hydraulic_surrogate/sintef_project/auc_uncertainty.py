"""Paired case-block uncertainty for a fixed pooled ROC AUC comparison."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class WeightedAUC:
    """Cache score order and ties, then score repeated case-weight vectors."""

    labels: np.ndarray
    case_indices: np.ndarray
    group_starts: np.ndarray
    n_cases: int

    @classmethod
    def prepare(cls, labels: np.ndarray, predictions: np.ndarray,
                n_cases: int) -> "WeightedAUC":
        """Accept matching case-major arrays without altering any predictions."""
        y, p = np.asarray(labels).ravel(), np.asarray(predictions).ravel()
        if y.shape != p.shape or not y.size or n_cases < 1 or y.size % n_cases:
            raise ValueError("Aligned arrays and complete cases are required")
        if not np.isfinite(p).all() or not np.isin(y, [0, 1]).all():
            raise ValueError("Expected finite scores and binary labels")
        order = np.argsort(p, kind="stable")
        sorted_scores = p[order]
        starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
        return cls(y[order].astype(np.float64), order // (y.size // n_cases),
                   starts, n_cases)

    def score(self, weights: np.ndarray) -> float:
        """Compute exact weighted pair AUC with half credit for score ties."""
        weights = np.asarray(weights, dtype=np.float64)
        if (weights.shape != (self.n_cases,) or not np.isfinite(weights).all()
                or (weights < 0).any()):
            raise ValueError("Expected one finite nonnegative weight per case")
        row_weights = weights[self.case_indices]
        positive = np.add.reduceat(row_weights * self.labels, self.group_starts)
        negative = np.add.reduceat(row_weights * (1 - self.labels), self.group_starts)
        denominator = positive.sum() * negative.sum()
        if denominator == 0:
            raise ValueError("Both weighted classes must be present")
        pairs = (positive * (np.cumsum(negative) - .5 * negative)).sum()
        return float(pairs / denominator)


def circular_block_weights(rng: np.random.Generator, n_cases: int,
                           block_days: int) -> np.ndarray:
    """Resample consecutive daily cases in circular blocks, keeping year size."""
    if not 1 <= block_days <= n_cases:
        raise ValueError("Block length must be between one and the case count")
    starts = rng.integers(0, n_cases, size=(n_cases + block_days - 1) // block_days)
    cases = ((starts[:, None] + np.arange(block_days)) % n_cases).ravel()[:n_cases]
    return np.bincount(cases, minlength=n_cases)
