"""Small summaries of reservoirs connected to each plant."""

import numpy as np
import pandas as pd


def connected_reservoirs(topology, plant, upstream=False):
    """All upstream reservoirs, or the first receiving reservoir on each outlet path."""
    pending = [("plant", plant)]
    visited = set()
    reservoirs = set()
    while pending:
        node = pending.pop()
        if node in visited:
            continue
        visited.add(node)
        if node[0] == "reservoir":
            reservoirs.add(node[1])
            if not upstream:
                continue
        for link in topology["connections"]:
            if link["from_type"] == "generator":
                continue
            source = (link["from_type"], link["from"])
            target = (link["to_type"], link["to"])
            if upstream and target == node:
                pending.append(source)
            elif not upstream and source == node:
                pending.append(target)
    return sorted(reservoirs)


def topology_features(plant, starts, timestamps, inputs, topology):
    result = {}
    for direction in ["upstream", "downstream"]:
        names = connected_reservoirs(topology, plant, upstream=direction == "upstream")
        volume = np.zeros(len(starts))
        fill = np.zeros(len(starts))
        water_value = np.zeros(len(starts))
        capacity = 0.0
        if names:
            capacity = sum(topology["model"]["reservoir"][name]["max_vol"] for name in names)
            volume = inputs["volume"].loc[starts, names].sum(axis=1, skipna=False).to_numpy()
            fill = volume / capacity
            # Arithmetic mean of supplied local values: a feature, not a hydraulic valuation.
            water_value = inputs["water_value"].loc[
                starts + pd.Timedelta(hours=168), names
            ].mean(axis=1, skipna=False).to_numpy()
        result[f"topology_{direction}_fill_fraction"] = fill
        result[f"topology_{direction}_water_value_mean"] = water_value
        if direction == "upstream":
            result["topology_upstream_volume_million_m3"] = volume
            inflow = inputs["inflow"].loc[timestamps, names].sum(axis=1, skipna=False).to_numpy()
            result["topology_upstream_week_inflow_million_m3"] = (
                inflow.reshape(-1, 168).sum(axis=1) * 3600 / 1_000_000
            )
        else:
            result["topology_has_downstream_reservoir"] = np.full(len(starts), bool(names))
            result["topology_downstream_free_capacity_million_m3"] = capacity - volume
    result["topology_upstream_minus_downstream_water_value"] = np.where(
        result["topology_has_downstream_reservoir"],
        result["topology_upstream_water_value_mean"] - result["topology_downstream_water_value_mean"],
        0.0,
    )
    return result
