"""Narrow summaries of supplied weekly requirements and natural inflow.

These are planning inputs, not forecasts of optimized storage or discharge.
Both cumulative and remaining water sums include the current hour.
"""

import numpy as np
import pandas as pd


HOURS = 168
M3_S_HOUR_TO_MILLION_M3 = 3600 / 1_000_000


def _week_matrix(starts, timestamps, frame, column):
    """Read complete, consecutive hourly cases, retaining duplicate timestamps."""
    starts = pd.DatetimeIndex(starts).as_unit("ns")
    timestamps = pd.DatetimeIndex(timestamps).as_unit("ns")
    if not len(starts) or len(timestamps) != len(starts) * HOURS:
        raise ValueError("Scheduling context requires complete 168-hour cases.")
    expected = starts.asi8[:, None] + np.arange(HOURS) * pd.Timedelta(hours=1).value
    if not np.array_equal(timestamps.asi8.reshape(-1, HOURS), expected):
        raise ValueError("Scheduling context timestamps must follow each case start hourly.")
    values = frame.loc[timestamps, column].to_numpy(dtype="float64").reshape(-1, HOURS)
    if not np.isfinite(values).all():
        raise ValueError(f"Scheduling context contains non-finite {column} inputs.")
    return values


def shared_river_constraint_context(starts, timestamps, inputs):
    """Summarize the known r_Vest schedule without assigning releases to a plant.

    With no later change inside the provided week, time to change is censored
    at the horizon end and next-change delta is zero. The final hour therefore
    has time-to-change one, delta zero and remaining burden of one hour.
    """
    required = _week_matrix(starts, timestamps, inputs["min_flow"], "r_Vest_Vassdraget")
    water = required * M3_S_HOUR_TO_MILLION_M3
    remaining = water[:, ::-1].cumsum(axis=1, dtype="float64")[:, ::-1]
    time_to_change = np.empty_like(required)
    next_delta = np.zeros_like(required)
    next_hour = np.full(len(required), HOURS, dtype="int64")
    delta = np.zeros(len(required), dtype="float64")
    for hour in range(HOURS - 1, -1, -1):
        if hour < HOURS - 1:
            change = required[:, hour + 1] != required[:, hour]
            next_hour[change] = hour + 1
            delta[change] = required[change, hour + 1] - required[change, hour]
        time_to_change[:, hour] = next_hour - hour
        next_delta[:, hour] = delta
    return {
        "shared_river_remaining_required_release_million_m3": remaining.ravel(),
        "shared_river_hours_until_next_change_or_end": time_to_change.ravel(),
        "shared_river_next_change_delta_m3_s": next_delta.ravel(),
    }


def distinct_inflow_timing(starts, timestamps, inputs):
    """Expose only Staavatn and Venemo natural inflow timing to every generator."""
    result = {}
    for reservoir in ("Staavatn", "Venemo"):
        flow = _week_matrix(starts, timestamps, inputs["inflow"], reservoir)
        water = flow * M3_S_HOUR_TO_MILLION_M3
        result[f"distinct_natural_inflow_{reservoir}_m3_s"] = flow.ravel()
        result[f"distinct_cumulative_natural_inflow_{reservoir}_million_m3"] = (
            water.cumsum(axis=1, dtype="float64").ravel()
        )
        result[f"distinct_remaining_natural_inflow_{reservoir}_million_m3"] = (
            water[:, ::-1].cumsum(axis=1, dtype="float64")[:, ::-1].ravel()
        )
    return result
