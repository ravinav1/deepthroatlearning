"""Input-only temporal summaries of an already verified hydraulic LP cache.

The frozen LP itself is unchanged. These summaries expose its neighboring
discharge decisions to the supervised model so it can learn omitted switching
costs. They are predictors, not probabilities or asserted commitment rules.
"""
from pathlib import Path

import numpy as np
import pandas as pd

HORIZON = 168
UNITS = (
    "Hogga_G1", "Tokke_G1", "Tokke_G2", "Tokke_G3", "Tokke_G4",
    "Lio_G1", "Byrte_G1", "Vinje_G1", "Vinje_G2", "Vinje_G3",
    "Haukeli_G1", "Songa_G1", "Kjela_G1", "Vesle Kjela_G1",
)


def add_lp_context_features(x: pd.DataFrame, root: Path) -> tuple[pd.DataFrame, dict]:
    """Append nine case-local q/Qmin summaries for each of fourteen units.

    Requires raw or normalized features containing a verified LP cache and all
    168 hours of each case. Lags/leads clamp to that case's boundary. Six-hour
    means exclude the current hour and use the available-hour count; an empty
    neighborhood falls back to the current ratio. All added columns are float32
    and dimensionless. ``root`` is an unused common-interface parameter.
    """
    del root
    required = [f"lp_qmin_ratio_{unit}" for unit in UNITS]
    missing = [name for name in ["horizon_hour", *required] if name not in x]
    if missing:
        raise ValueError(f"Missing verified LP input columns: {missing}")
    if not len(x) or len(x) % HORIZON:
        raise ValueError("LP context requires complete 168-hour cases before training sampling")
    hours = np.arange(HORIZON)
    if not np.array_equal(x["horizon_hour"].to_numpy(), np.tile(hours, len(x) // HORIZON)):
        raise ValueError("LP context requires case-major hours 0 through 167")
    additions = {}
    previous_start = np.maximum(hours - 6, 0)
    next_end = np.minimum(hours + 7, HORIZON)
    for unit, column in zip(UNITS, required):
        q = x[column].to_numpy(dtype=np.float64).reshape(-1, HORIZON)
        if not np.isfinite(q).all():
            raise ValueError(f"Nonfinite LP discharge ratio: {unit}")
        for offset in (1, 3):
            additions[f"lp_context_qmin_previous_{offset}h_{unit}"] = q[:, np.maximum(hours - offset, 0)]
            additions[f"lp_context_qmin_next_{offset}h_{unit}"] = q[:, np.minimum(hours + offset, HORIZON - 1)]
        prefix = np.concatenate([np.zeros_like(q[:, :1]), q.cumsum(axis=1)], axis=1)
        for direction, sums, counts in (
            ("previous", prefix[:, hours] - prefix[:, previous_start], hours - previous_start),
            ("next", prefix[:, next_end] - prefix[:, hours + 1], next_end - hours - 1),
        ):
            additions[f"lp_context_qmin_{direction}_6h_mean_{unit}"] = np.divide(
                sums, counts[None, :], out=q.copy(), where=counts[None, :] > 0
            )
        additions[f"lp_context_qmin_week_mean_{unit}"] = np.repeat(q.mean(axis=1), HORIZON)
        additions[f"lp_context_qmin_week_at_least_one_fraction_{unit}"] = np.repeat((q >= 1).mean(axis=1), HORIZON)
        additions[f"lp_context_qmin_week_positive_fraction_{unit}"] = np.repeat((q > 0).mean(axis=1), HORIZON)
    overlap = set(additions).intersection(x.columns)
    if overlap:
        raise ValueError("LP context features have already been added")
    with np.errstate(over="ignore", invalid="ignore"):
        frame = pd.DataFrame({name: np.asarray(values, dtype=np.float32).reshape(-1)
                              for name, values in additions.items()}, index=x.index)
    if frame.shape != (len(x), 126) or not np.isfinite(frame.to_numpy()).all():
        raise ValueError("LP context must produce 126 finite float32 columns")
    spec = {
        "version": "lp_temporal_context_v1",
        "source_module": "sintef_project/lp_context.py",
        "input_columns": required,
        "units": list(UNITS),
        "added_columns": list(frame.columns),
        "source_provenance": "Derived only from lp_qmin_ratio_<unit> in the already verified LP cache. add_lp_features must verify cache source/data/artifact hashes before this transform. Trainer must archive this module and retain the upstream LP feature specification.",
        "lag_lead": "q(case,clip(hour+offset,0,167)), offset=-3,-1,+1,+3; never cross a case boundary",
        "neighbor_means": "Previous max(0,h-6)..h-1 and next h+1..min(167,h+6), excluding current; divide by available count; empty neighborhood=current q/Qmin",
        "week_summaries": "Mean q/Qmin, fraction q/Qmin>=1, fraction q/Qmin>0 over all168 hours, repeated per case; exact comparisons, no tolerance or clipping",
        "monetary_schema": "All126 added lp_context_* columns are dimensionless and must survive later economic normalization unchanged",
        "fit_scope": "No fitting, labels, filesystem reads, model prediction or LP solving; derive full-case context before training-row sampling",
        "interpretation": "LP relaxation omits unit switching/integer decisions. Neighbor and week activity summaries help the supervised learner model those corrections; q>=1/q>0 summaries are not commitment labels or hard feasibility rules.",
        "deferred": "No storage end-minus-current features in this bounded126-column ablation; their separate contribution is unmeasured",
    }
    return pd.concat([x, frame], axis=1), spec
