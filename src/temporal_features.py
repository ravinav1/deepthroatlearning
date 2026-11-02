"""Case-local temporal summaries of supplied prices, never commitment labels."""

import numpy as np


HOURS = 168


def temporal_ranks(price):
    """Midpoint percentiles among the full week and current-to-end hours.

    Ties share (number lower + half the number tied) / number considered.
    A flat week or a one-hour suffix therefore has rank 0.5.
    """
    price = np.asarray(price, dtype=np.float64)
    assert price.ndim == 2 and price.shape[1] == HOURS
    week = np.empty(price.shape, dtype=np.float32)
    remaining = np.empty_like(week)
    for hour in range(HOURS):
        lower = price < price[:, hour:hour + 1]
        equal = price == price[:, hour:hour + 1]
        week[:, hour] = (lower.sum(axis=1) + .5 * equal.sum(axis=1)) / HOURS
        remaining[:, hour] = (
            lower[:, hour:].sum(axis=1) + .5 * equal[:, hour:].sum(axis=1)
        ) / (HOURS - hour)
    return {"temporal_rank_week": week, "temporal_rank_remaining": remaining}


def favourable_periods(price, water_value):
    """Summarize positive price-minus-terminal-water-value blocks within a week.

    This margin is an economic proxy, not a feasible operation rule. Durations
    are observed within the case, and boundary flags mark censored blocks.
    Distances are zero on favourable hours and 168 when no such hour exists
    in the relevant direction. Equality is not favourable.
    """
    price = np.asarray(price, dtype=np.float64)
    water_value = np.asarray(water_value, dtype=np.float64)
    assert price.ndim == 2 and price.shape[1] == HOURS
    assert water_value.shape == (len(price),)
    active = price > water_value[:, None]
    elapsed = np.zeros(price.shape, dtype=np.float32)
    remaining = np.zeros_like(elapsed)
    previous = np.full(price.shape, HOURS, dtype=np.float32)
    following = np.full_like(previous, HOURS)
    for hour in range(HOURS):
        elapsed[:, hour] = np.where(active[:, hour], 1 + (elapsed[:, hour - 1] if hour else 0), 0)
        previous[:, hour] = np.where(active[:, hour], 0, np.minimum(
            HOURS, previous[:, hour - 1] + 1 if hour else HOURS))
        reverse = HOURS - 1 - hour
        remaining[:, reverse] = np.where(active[:, reverse], 1 + (remaining[:, reverse + 1] if hour else 0), 0)
        following[:, reverse] = np.where(active[:, reverse], 0, np.minimum(
            HOURS, following[:, reverse + 1] + 1 if hour else HOURS))
    hours = np.arange(HOURS)
    return {
        "temporal_favourable_duration": np.where(active, elapsed + remaining - 1, 0).astype(np.float32),
        "temporal_favourable_elapsed": elapsed,
        "temporal_favourable_remaining": remaining,
        "temporal_favourable_previous_distance": previous,
        "temporal_favourable_next_distance": following,
        "temporal_favourable_left_censored": (active & (elapsed == hours + 1)).astype(np.float32),
        "temporal_favourable_right_censored": (active & (remaining == HOURS - hours)).astype(np.float32),
    }
