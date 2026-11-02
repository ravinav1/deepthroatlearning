"""Optional water-context and approximate start/stop economics features."""

import numpy as np

from topology_features import connected_reservoirs


def upcoming_sum(values, window):
    """Sum current hour and following hours, clipped to the supplied case."""
    hour = np.arange(values.shape[1])
    end = np.minimum(hour + window, values.shape[1])
    cumulative = np.concatenate([np.zeros((len(values), 1)), values.cumsum(axis=1, dtype="float64")], axis=1)
    return cumulative[:, end] - cumulative[:, hour]


def water_context(plant, volume, capacity, local_inflow, timestamps, inputs, topology):
    result = {}
    local = local_inflow * 3600 / 1_000_000
    # These comparisons stay anchored to starting storage, not predicted storage.
    free_capacity = capacity - volume[:, None]
    upstream = connected_reservoirs(topology, plant, upstream=True)
    upstream_flow = inputs["inflow"].loc[timestamps, upstream].sum(axis=1, skipna=False).to_numpy().reshape(local.shape)
    upstream_volume = upstream_flow * 3600 / 1_000_000
    result["water_upstream_natural_inflow_m3_s"] = upstream_flow.ravel()
    result["water_upstream_cumulative_natural_inflow_million_m3"] = upstream_volume.cumsum(axis=1).ravel()
    result["water_upstream_remaining_natural_inflow_million_m3"] = upcoming_sum(upstream_volume, local.shape[1]).ravel()
    for window in [6, 12, 24]:
        incoming = upcoming_sum(local, window)
        result[f"water_local_natural_inflow_next_{window}h_million_m3"] = incoming.ravel()
        result[f"water_local_natural_inflow_next_{window}h_capacity_fraction"] = (incoming / capacity).ravel()
        result[f"water_initial_free_capacity_minus_next_{window}h_inflow_million_m3"] = (free_capacity - incoming).ravel()
        # Natural water entering upstream reservoirs is not an actual arrival forecast at this plant.
        result[f"water_upstream_natural_inflow_next_{window}h_million_m3"] = upcoming_sum(upstream_volume, window).ravel()
    return result


def economic_context(price, water_value, generator):
    result = {}
    cycle_cost = generator["startcost_const"] + generator["stopcost_const"]
    margin = price - water_value[:, None]
    for level in ["p_min", "p_nom"]:
        power = generator[level]
        result[f"economics_start_stop_cost_per_{level}"] = np.full(price.size, cycle_cost / power)
        for window in [6, 12, 24]:
            # EUR: constant reference MW * sum of hourly EUR/MWh margins - cycle cost.
            # This is a hypothetical continuous run, not SHOP's true objective or a hard rule.
            result[f"economics_{level}_net_margin_next_{window}h_eur"] = (
                power * upcoming_sum(margin, window) - cycle_cost
            ).ravel()
    return result
