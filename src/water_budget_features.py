"""Reference water budgets using natural inflow and required direct releases."""

import numpy as np


def natural_water_budgets(starts, timestamps, inputs, topology):
    """Return case/hour features, excluding production, transfers and spill.

    Cumulative and remaining sums both include the current hour. Reference
    storage includes that hour's net inflow; it is not an actual storage forecast.
    """
    result = {}
    hours = len(timestamps) // len(starts)
    for name in inputs["volume"].columns:
        inflow = inputs["inflow"].loc[timestamps, name].to_numpy().reshape(-1, hours)
        # Only a reservoir's own constrained outlets belong in this budget.
        # A downstream river's requirement may be supplied by several reservoirs.
        outlets = sorted({
            link["to"] for link in topology["connections"]
            if link["from_type"] == "reservoir" and link["from"] == name
            and link["to_type"] == "river" and link["to"] in inputs["min_flow"].columns
        })
        required = np.zeros_like(inflow, dtype="float64")
        if outlets:
            required = inputs["min_flow"].loc[timestamps, outlets].sum(
                axis=1, skipna=False
            ).to_numpy().reshape(-1, hours)
        net = (inflow - required) * 3600 / 1_000_000
        cumulative = net.cumsum(axis=1, dtype="float64")
        remaining = net[:, ::-1].cumsum(axis=1, dtype="float64")[:, ::-1]
        initial = inputs["volume"].loc[starts, name].to_numpy()
        reference_volume = initial[:, None] + cumulative
        minimum = 0.0
        if name in inputs["min_volume"].columns:
            minimum = inputs["min_volume"].loc[timestamps, name].to_numpy().reshape(-1, hours)

        result[f"budget_cumulative_net_natural_inflow_{name}_million_m3"] = cumulative.ravel()
        result[f"budget_remaining_net_natural_inflow_{name}_million_m3"] = remaining.ravel()
        # Keep negative and over-capacity values as pressure signals, not hard rules.
        result[f"budget_reference_fill_fraction_{name}"] = (
            reference_volume / topology["model"]["reservoir"][name]["max_vol"]
        ).ravel()
        result[f"budget_reference_volume_minus_min_{name}_million_m3"] = (
            reference_volume - minimum
        ).ravel()
    return result
