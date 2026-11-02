"""Optional input-only emphasis on hours near the local economic boundary."""

import numpy as np


MARGIN_COLUMN = "price_minus_water_value_week_scaled"


def boundary_fit_weights(features, config):
    options = config.get("boundary_weighting")
    if options is None:
        return {}, None
    assert config.get("use_economic_rescaling", False), "Boundary weights need economic rescaling."
    assert config.get("recency_half_life_years") is None, "Use boundary or recency weights, not both."
    threshold = options["margin_threshold"]
    multiplier = options["multiplier"]
    assert threshold > 0 and multiplier >= 1

    # Keep the feature row order: generator, then case, then hour. No labels enter
    # the rule, so ON and OFF examples receive the same input-dependent weight.
    margin = features[MARGIN_COLUMN].to_numpy(dtype="float64")
    near = np.abs(margin) < threshold
    weights = np.where(near, multiplier, 1.)
    raw_mean = float(weights.mean())
    weights = (weights / raw_mean).astype("float32")
    summary = {
        "method": "local_economic_boundary", "input_column": MARGIN_COLUMN,
        "margin_threshold": threshold, "multiplier": multiplier,
        "training_rows": len(weights), "boundary_rows": int(near.sum()),
        "boundary_fraction": float(near.mean()),
        "normalization": "training_row_mean_1", "raw_mean": raw_mean,
        "minimum": float(weights.min()), "maximum": float(weights.max()),
        "mean": float(weights.mean(dtype="float64")),
        "labels_used_for_weight_rule": False,
    }
    return {"sample_weight": weights}, summary
