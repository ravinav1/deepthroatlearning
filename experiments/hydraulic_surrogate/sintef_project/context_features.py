"""Case-local price context adapted from the user's canonical feature design.

This transform runs on complete 168-hour cases before monetary normalization
and before training-row sampling. It reads no labels or external files.
"""
from pathlib import Path

import numpy as np
import pandas as pd

HORIZON = 168


def add_context_features(x: pd.DataFrame, root: Path) -> tuple[pd.DataFrame, dict]:
    """Add remaining-horizon midpoint rank and neighboring price means.

    Each previous/next mean excludes the current hour and uses at most 6, 12,
    or 24 available case-local hours. At an empty boundary, use the current
    price. The rank includes the current hour, gives ties their midpoint,
    and is named without ``price_`` so monetary normalization preserves it.
    ``root`` is accepted for the shared transform interface but is not read.
    """
    del root
    if "price" not in x or "horizon_hour" not in x:
        raise ValueError("Context requires raw price and horizon_hour before normalization")
    if not len(x) or len(x) % HORIZON:
        raise ValueError("Context requires complete 168-hour cases before row sampling")
    hours = np.arange(HORIZON)
    if not np.array_equal(x["horizon_hour"].to_numpy(), np.tile(hours, len(x) // HORIZON)):
        raise ValueError("Expected case-major hours 0 through 167")
    price = x["price"].to_numpy(dtype=np.float64).reshape(-1, HORIZON)
    if not np.isfinite(price).all():
        raise ValueError("Prices must be finite")

    remaining_rank = np.empty(price.shape, dtype=np.float32)
    for hour in range(HORIZON):
        suffix = price[:, hour:]
        current = price[:, hour:hour + 1]
        remaining_rank[:, hour] = (
            (suffix < current).sum(axis=1) + .5 * (suffix == current).sum(axis=1)
        ) / (HORIZON - hour)
    additions = {"context_rank_remaining": remaining_rank.ravel()}
    prefix = np.concatenate([np.zeros_like(price[:, :1]), price.cumsum(axis=1)], axis=1)
    for window in (6, 12, 24):
        left = np.maximum(hours - window, 0)
        right = np.minimum(hours + window + 1, HORIZON)
        for direction, sums, counts in (
            ("previous", prefix[:, hours] - prefix[:, left], hours - left),
            ("next", prefix[:, right] - prefix[:, hours + 1], right - hours - 1),
        ):
            mean = np.divide(sums, counts[None, :], out=price.copy(), where=counts[None, :] > 0)
            additions[f"price_{direction}_mean_{window}h"] = mean.astype(np.float32).ravel()
            additions[f"price_minus_{direction}_mean_{window}h"] = (price - mean).astype(np.float32).ravel()
    overlap = set(x.columns).intersection(additions)
    if overlap:
        raise ValueError(f"Context features already exist: {sorted(overlap)}")
    spec = {
        "source": "User main repository src/temporal_features.py and src/price_features.py; independent adapter preserving formulas",
        "rank": "context_rank_remaining=(number strictly lower + 0.5*number equal)/(168-hour), over current hour through167; flat suffix and final hour rank0.5",
        "rank_normalization": "Dimensionless context_rank_remaining intentionally lacks price_ prefix and must not be divided by monetary scale",
        "neighbor_mean": "For6/12/24h, previous=[max(0,h-window),h), next=(h,min(167,h+window)]; excludecurrent; divide by actual available count; empty neighborhoods use current price",
        "contrasts": "Current price minus each neighbor mean; all12 price_* context columns are monetary and should be normalized with the rest of the case prices",
        "fit_scope": "No fitting, labels, filesystem reads or cross-case aggregation; compute complete168-hour features before selecting training rows",
        "added_columns": list(additions),
    }
    result = pd.concat([x, pd.DataFrame(additions, index=x.index, copy=False)], axis=1)
    return result, spec
