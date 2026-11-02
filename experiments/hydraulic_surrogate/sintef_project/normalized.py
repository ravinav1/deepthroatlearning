"""Optional case-relative economics for transfer across electricity-price regimes.

Replace absolute monetary predictors with case-normalized versions rather than
duplicating them. Fixed startup costs mean the complete scheduling problem is
not invariant to monetary scaling; inverse price scale retains that information.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def add_normalized_features(x: pd.DataFrame, root: Path) -> tuple[pd.DataFrame, dict]:
    """Normalize monetary columns by each case's mean absolute hourly price + 1.

    ``root`` is accepted for the common feature-transform interface; no files,
    targets, or fitted statistics are read. All 168 prices are documented inputs
    to a case's optimization. The input frame remains unchanged.
    """
    del root
    if len(x) == 0 or len(x) % 168:
        raise ValueError("Expected nonempty complete 168-hour cases")
    expected_hours = np.tile(np.arange(168), len(x) // 168)
    if not np.array_equal(x["horizon_hour"].to_numpy(), expected_hours):
        raise ValueError("Expected case-major ordering with hours 0 through 167")
    price = x["price"].to_numpy(dtype=np.float64).reshape(-1, 168)
    if not np.isfinite(price).all():
        raise ValueError("Price must be finite")
    case_scale = np.abs(price).mean(axis=1) + 1.0
    scale = np.repeat(case_scale, 168)
    preserved_relative = {"price_week_rank", "price_week_zscore"}
    monetary = []
    for column in x:
        is_relative = column in preserved_relative or column.startswith("price_over_water_")
        is_monetary = column == "price" or column.startswith(("price_", "water_value_", "physics_switch_margin_"))
        if is_monetary and not is_relative:
            monetary.append(column)
    additions = {}
    for column in monetary:
        values = x[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite monetary feature: {column}")
        name = f"norm_{column}"
        if name in x:
            raise ValueError(f"Normalized column already exists: {name}")
        additions[name] = (values / scale).astype(np.float32)
    if "monetary_scale_inverse" in x:
        raise ValueError("Monetary normalization has already been applied")
    additions["monetary_scale_inverse"] = (1.0 / scale).astype(np.float32)
    retained = [column for column in x if column not in monetary]
    result = pd.concat([x[retained], pd.DataFrame(additions, index=x.index, copy=False)], axis=1)
    spec = {
        "scale_formula": "s(case) = 1 + mean(abs(price(case,h))) for h=0..167, in EUR/MWh",
        "normalized_formula": "norm_<original> = original / s(case)",
        "inverse_scale": "monetary_scale_inverse = 1/s(case); no raw scale feature",
        "replaced_columns": monetary,
        "normalized_columns": [f"norm_{column}" for column in monetary],
        "retained_columns": retained,
        "preserved_relative_features": [column for column in retained if column in preserved_relative or column.startswith("price_over_water_")],
        "row_order": "Complete 168-hour cases; each scale applies only to its own case",
        "fit_scope": "No fitting; per-case known optimizer inputs only; no label or filesystem access",
        "switching_costs": "Normalize previously computed physics switching margins after their fixed-cost DP. This is not exact economic scale invariance: startup/stop costs stay fixed; inverse scale allows the learner to recover their relative importance.",
        "stabilization": "The +1 EUR/MWh denominator offset prevents near-zero price instability. Existing price_over_water uses water+1 and price_week_zscore uses std+1; retain both as requested relative features, despite nonexact scale invariance.",
        "hypothesis": "Relative prices, water values and economic margins generalize across historical electricity-price scales more easily than unnormalized currency values",
    }
    return result, spec
