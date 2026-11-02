"""Case-level reservoir history summaries restricted to each case's start.

This module has no filesystem access, label argument, model, or interpolation.
Water-balance residuals are covariates, never reconstructed generator output.
"""
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


def _utc_index(index: pd.Index, name: str, *, daily: bool) -> pd.DatetimeIndex:
    """Require unambiguous, unique, aligned timestamps without filling gaps."""
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None:
        raise ValueError(f'{name} must have a timezone-aware DatetimeIndex')
    result = index.tz_convert('UTC')
    if result.hasnans or result.has_duplicates:
        raise ValueError(f'{name} contains duplicate or missing timestamps')
    if not result.equals(result.floor('D' if daily else 'h')):
        raise ValueError(f'{name} must be aligned to UTC {"days" if daily else "hours"}')
    return result


def build_past_state_features(
    case_starts: pd.DatetimeIndex,
    volume: pd.DataFrame,
    inflow: pd.DataFrame,
    capacities: Mapping[str, float],
    *,
    reservoirs: Sequence[str] | None = None,
    lookback_days: Sequence[int] = (1, 3, 7),
) -> tuple[pd.DataFrame, dict]:
    """Return five finite float32 summaries per reservoir/lookback, one row/case.

    Volume is start-of-day storage in million m3. Inflow is hourly m3/s,
    timestamped at the beginning of its hour. For case start t and lookback d,
    use V(t), V(t-d), and inflow timestamps in [t-d,t), never inflow at t.
    Every expected hour must exist and be finite. Missing endpoints or hours
    produce an explicit flag and zero for the affected numeric summary; the
    withdrawal residual is zero if either component is missing. No partial
    window, interpolation, as-of join, or future backfill is permitted.

    Output preserves the supplied case order. It must be broadcast within each
    corresponding case only if integrated later; this function performs no
    168-hour expansion or fitting. Supplied frames are never modified.
    """
    starts = _utc_index(case_starts, 'case_starts', daily=True)
    volume_index = _utc_index(volume.index, 'volume', daily=True)
    inflow_index = _utc_index(inflow.index, 'inflow', daily=False)
    if not len(starts) or not len(volume) or not len(inflow):
        raise ValueError('Cases, volume, and inflow must be nonempty')
    names = list(volume.columns if reservoirs is None else reservoirs)
    windows = list(lookback_days)
    if not names or len(set(names)) != len(names):
        raise ValueError('Expected unique nonempty reservoir names')
    if volume.columns.has_duplicates or inflow.columns.has_duplicates:
        raise ValueError('Duplicate source columns')
    if (not windows or len(set(windows)) != len(windows)
            or any(isinstance(d, bool) or not isinstance(d, (int, np.integer)) or d <= 0 for d in windows)):
        raise ValueError('Expected unique positive integer lookback days')
    for name in names:
        if name not in volume or name not in inflow or name not in capacities:
            raise ValueError(f'Missing named reservoir input: {name}')
    capacity = np.array([capacities[name] for name in names], dtype=np.float64)
    if not np.isfinite(capacity).all() or (capacity <= 0).any():
        raise ValueError('Capacities must be finite and positive')
    storage = volume[names].copy()
    storage.index = volume_index
    flow = inflow[names].copy()
    flow.index = inflow_index
    flow = flow.where(np.isfinite(flow), np.nan)
    # Unique aligned UTC hours imply at most 24 observations on any UTC day.
    # A daily sum is valid only with 24 finite observations for that reservoir.
    day = inflow_index.normalize()
    daily_sum = flow.groupby(day).sum(min_count=24)
    daily_count = flow.groupby(day).count()
    daily_sum = daily_sum.where(daily_count == 24, np.nan) * .0036
    current = storage.reindex(starts).to_numpy(dtype=np.float64)
    additions = {}
    for days in windows:
        previous = storage.reindex(starts - pd.Timedelta(days=days)).to_numpy(dtype=np.float64)
        volume_missing = ~np.isfinite(current) | ~np.isfinite(previous)
        delta = np.divide(current - previous, capacity)
        delta = np.where(volume_missing, 0., delta)
        # All requested daily integrals end before case start. Days at t and
        # later may exist in the source but never enter a returned feature.
        query = pd.DatetimeIndex(np.concatenate([
            (starts - pd.Timedelta(days=offset)).as_unit('ns').asi8
            for offset in range(1, days + 1)
        ]), tz='UTC')
        daily = daily_sum.reindex(query).to_numpy(dtype=np.float64).reshape(days, len(starts), len(names))
        inflow_missing = ~np.isfinite(daily).all(axis=0)
        inflow_ratio = np.where(inflow_missing, 0., np.nansum(daily, axis=0) / capacity)
        withdrawal = np.where(volume_missing | inflow_missing, 0., inflow_ratio - delta)
        for column, name in enumerate(names):
            for label, array in [('delta_fill', delta), ('inflow_capacity', inflow_ratio),
                                 ('withdrawal_proxy', withdrawal),
                                 ('volume_missing', volume_missing), ('inflow_missing', inflow_missing)]:
                additions[f'past_state_{label}_{days}d_{name}'] = array[:, column].astype(np.float32)
    result = pd.DataFrame(additions, index=starts)
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError('History features overflowed finite float32 representation')
    spec = {
        'version': 'past_reservoir_state_v1', 'reservoirs': names,
        'lookback_days': windows, 'columns': list(result.columns),
        'row_order': 'supplied case_starts, one row per case; no hourly expansion',
        'volume_formula': '(V(t)-V(t-d))/capacity; exact endpoint timestamps only',
        'inflow_formula': 'sum hourly q(s)*0.0036/capacity for t-d<=s<t; require all 24*d hours finite',
        'withdrawal_formula': 'inflow_capacity-delta_fill; signed net withdrawal proxy, not generation',
        'missing': 'explicit volume_missing and inflow_missing flags; affected numeric term zero; withdrawal zero if either flag set',
        'boundary': 'volume timestamp <=case_start; inflow hour start <case_start; no interpolation or future backfill',
        'units': 'All fields dimensionless float32; missing flags are 0 or 1',
        'interpretation': 'Residual also reflects upstream releases, transfers, bypasses/spill, measurement error and simplified storage; cannot identify plant discharge or generator status.',
        'provenance': 'Caller must archive source volume/inflow/capacity hashes and case IDs; no files, labels or learned predictions are accessed by this pure transform.',
    }
    return result, spec
