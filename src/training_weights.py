"""Optional case-age weights for tree and sequence training."""

import numpy as np

from features import HOURS


def recency_fit_weights(cases, generator_count, half_life, reference_date):
    if half_life is None:
        return {}, None
    assert half_life > 0, "Recency half-life must be positive."
    age_years = (reference_date - cases["starttime"]).dt.total_seconds().to_numpy() / (365.25 * 86400)
    case_weights = np.exp2(-age_years / half_life)
    case_weights /= case_weights.mean()
    # Feature rows are ordered by generator, then case, then hour.
    row_weights = np.tile(np.repeat(case_weights, HOURS), generator_count).astype("float32")
    summary = {
        "half_life_years": half_life, "reference_date": str(reference_date),
        "normalization": "mean_1", "minimum": float(case_weights.min()),
        "maximum": float(case_weights.max()), "mean": float(case_weights.mean()),
    }
    return {"sample_weight": row_weights}, summary


def recency_case_weights(cases, half_life, reference_date):
    """Give each complete scheduling week one mean-normalized age weight."""
    if half_life is None:
        return None, None
    assert half_life > 0, "Recency half-life must be positive."
    age_years = (reference_date - cases["starttime"]).dt.total_seconds().to_numpy() / (365.25 * 86400)
    weights = np.exp2(-age_years / half_life)
    weights /= weights.mean()
    summary = {
        "half_life_years": half_life, "reference_date": str(reference_date),
        "training_cases": len(cases), "normalization": "mean_1",
        "minimum": float(weights.min()), "maximum": float(weights.max()),
        "mean": float(weights.mean()),
    }
    return weights.astype("float32"), summary
