"""Optional hydraulic and switching-cost features, without optimization labels.

The supplied EUR/MWh water values are treated as local energy values, consistent
with SHOP's energy_value_input semantics. They are not subtracted across plants.
These transparent approximations are features, not a replacement SHOP solver.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# Reservoirs connected without an intervening generator. The equivalent-value
# groups also match the exact shared water-value series in the supplied dataset.
POOLS = {
    "Hogga": ["Bandak"],
    "Tokke": ["Vinjevatn"],
    "Lio": ["Byrtevatn"],
    "Byrte": ["Botnedalsvatn"],
    "Vinje": ["Vaamarvatn", "Totak", "Venemo", "Hyljelihyl"],
    "Haukeli": ["Vatjern", "Langeidvatn"],
    "Songa": ["Songavatn", "Bitdalsvatn"],
    "Kjela": ["Foersvatn", "Bordalsvatn", "Langesae"],
    "Vesle Kjela": ["Kjelavatn", "Staavatn"],
}


def switching_advantage(profit: np.ndarray, start_cost: float, stop_cost: float) -> np.ndarray:
    """Return optimal objective(on at t) minus objective(off at t), for each t.

    Max-sum dynamic programming includes hourly operating profit and asymmetric
    switch costs. Initial and final states are free because no initial commitment
    is supplied. Water budget, cascade coupling and ramp constraints are omitted.
    Both future and past profits belong to the same known scheduling horizon.
    """
    profit = np.asarray(profit, dtype=np.float64)
    if profit.ndim != 2 or not np.isfinite(profit).all():
        raise ValueError("Profit must be a finite case-by-hour matrix")
    n_cases, horizon = profit.shape
    forward0 = np.zeros((n_cases, horizon), dtype=np.float64)
    forward1 = np.empty_like(forward0)
    forward1[:, 0] = profit[:, 0]
    for hour in range(1, horizon):
        forward0[:, hour] = np.maximum(forward0[:, hour - 1], forward1[:, hour - 1] - stop_cost)
        forward1[:, hour] = profit[:, hour] + np.maximum(forward1[:, hour - 1], forward0[:, hour - 1] - start_cost)
    backward0 = np.zeros(n_cases, dtype=np.float64)
    backward1 = np.zeros(n_cases, dtype=np.float64)
    advantage = np.empty_like(forward0)
    for hour in range(horizon - 1, -1, -1):
        advantage[:, hour] = forward1[:, hour] + backward1 - forward0[:, hour] - backward0
        previous0 = np.maximum(backward0, profit[:, hour] + backward1 - start_cost)
        previous1 = np.maximum(backward0 - stop_cost, profit[:, hour] + backward1)
        backward0, backward1 = previous0, previous1
    return advantage.astype(np.float32)


def add_physics_features(x: pd.DataFrame, root: Path) -> tuple[pd.DataFrame, dict]:
    """Enrich complete case-hour features with physical and economic proxies.

    Returns a new frame and a JSON-serializable specification. Does not mutate x.
    Expects the 168-hour case order emitted by features.build_features.
    """
    root = Path(root)
    if (root / "sintef_project").is_dir():
        root = root / "sintef_project"
    topology_path = root / "data" / "extended" / "Tokke_Vinje_topology.yaml"
    topology = yaml.safe_load(topology_path.read_text())
    model = topology["model"]
    if len(x) % 168 or not np.array_equal(x["horizon_hour"].to_numpy(), np.tile(np.arange(168), len(x) // 168)):
        raise ValueError("Expected complete 168-hour cases in case-major order")
    n_cases = len(x) // 168
    at_start = x.iloc[::168]
    price = x["price"].to_numpy().reshape(n_cases, 168)
    remaining_hours = 168 - np.arange(168)
    additions = {}
    plant_specs = {}
    incoming = {}
    downstream = {}
    for edge in topology["connections"]:
        incoming.setdefault(edge["to"], []).append(edge["from"])
        if edge["from_type"] == "plant" and edge["to_type"] == "reservoir":
            downstream[edge["from"]] = edge["to"]

    def upstream_reservoirs(plant: str) -> list[str]:
        visited = set()
        queue = list(incoming.get(plant, []))
        while queue:
            node = queue.pop()
            if node in visited:
                continue
            visited.add(node)
            queue.extend(incoming.get(node, []))
        return sorted(visited.intersection(model["reservoir"]))

    def head(reservoir: str) -> np.ndarray:
        curve = model["reservoir"][reservoir]["vol_head"]
        return np.interp(at_start[f"volume_{reservoir}"], curve["x"], curve["y"])

    def add(name: str, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float32)
        if values.shape == (n_cases,):
            values = np.repeat(values, 168)
        values = values.reshape(-1)
        if len(values) != len(x) or not np.isfinite(values).all():
            raise ValueError(f"Invalid physical feature: {name}")
        additions[name] = values

    for plant, pool in POOLS.items():
        upstream = pool[0]
        plant_data = model["plant"][plant]
        units = [u for u in model["generator"] if u.rsplit("_G", 1)[0] == plant]
        representative = model["generator"][units[0]]
        max_power = sum(model["generator"][u]["p_max"] for u in units)
        outlet = np.full(n_cases, plant_data["outlet_line"], dtype=np.float64)
        if plant in downstream:
            outlet = np.maximum(outlet, head(downstream[plant]))
        upstream_head = head(upstream)
        if plant == "Songa":
            upstream_head = (head("Songavatn") + head("Bitdalsvatn")) / 2
        gross_head = np.maximum(upstream_head - outlet, 1)
        curves = sorted(representative["turb_eff_curves"], key=lambda curve: curve["ref"])
        peak_turbine_eff = np.interp(gross_head, [c["ref"] for c in curves], [max(c["y"]) / 100 for c in curves])
        generator_eff = max(representative["gen_eff_curve"]["y"]) / 100
        # 1000 kg/m3 *9.81 m/s2 *1e6 m3 /3.6e9 J/MWh = 2.725.
        energy_equivalent = 2.725 * gross_head * peak_turbine_eff * generator_eff
        flow_capacity = max_power / (energy_equivalent * 0.0036)
        local_water = at_start[f"water_value_{upstream}"].to_numpy()
        margin = price - local_water[:, None]
        capacity = sum(model["reservoir"][r]["max_vol"] for r in pool)
        initial = sum(at_start[f"volume_{r}"].to_numpy() for r in pool)
        minimum = sum(at_start[f"min_volume_week_max_{r}"].to_numpy() if f"min_volume_week_max_{r}" in x else np.zeros(n_cases) for r in pool)
        remaining_inflow = sum(x[f"inflow_remaining_mm3_{r}"].to_numpy().reshape(n_cases, 168) for r in pool)
        available = np.maximum(initial - minimum, 0)[:, None] + remaining_inflow
        catchment = upstream_reservoirs(plant)
        catchment_flow = sum(x[f"inflow_{r}"].to_numpy().reshape(n_cases, 168) for r in catchment)
        add(f"physics_head_{plant}", gross_head)
        add(f"physics_pool_fill_{plant}", initial / capacity)
        add(f"physics_supply_fraction_{plant}", available * energy_equivalent[:, None] / (remaining_hours * max_power))
        add(f"physics_catchment_inflow_fraction_{plant}", catchment_flow / flow_capacity[:, None])
        weekly_inflow = remaining_inflow[:, 0]
        add(f"physics_dispatch_to_avoid_spill_{plant}", np.maximum(initial + weekly_inflow - capacity, 0) * energy_equivalent / (168 * max_power))
        add(f"physics_profitable_remaining_fraction_{plant}", np.cumsum((margin > 0)[:, ::-1], axis=1)[:, ::-1] / remaining_hours)
        for operating_point, power in (("minimum", representative["p_min"]), ("maximum", representative["p_max"])):
            advantage = switching_advantage(margin * power, representative["startcost_const"], representative["stopcost_const"])
            add(f"physics_switch_margin_{operating_point}_{plant}", advantage / power)
        plant_specs[plant] = {
            "pool": pool, "head_reference_reservoir": upstream,
            "downstream_reservoir": downstream.get(plant), "catchment": catchment,
            "representative_unit": units[0], "plant_pmax_mw": max_power,
            "switch_pmin_mw": representative["p_min"], "switch_pmax_mw": representative["p_max"],
            "start_cost": representative["startcost_const"], "stop_cost": representative["stopcost_const"],
        }
    spec = {
        "sources": ["sintef_project/Dataset_definitions_and_explanation.pdf",
                    "https://docs.shop.sintef.energy/examples/water_values/water_values.html",
                    "https://docs.shop.sintef.energy/examples/water_values/individual_water_values.html"],
        "water_value_semantics": "Local PDF states EUR/MWh. Interpret as local energy_value_input consistent with official SHOP docs; case-generation code unavailable. No upstream-downstream EUR/MWh subtraction.",
        "head": "Initial volume interpolated on YAML vol_head; upstream minus max(plant outlet_line, downstream reservoir head). Songa averages its two supply heads.",
        "energy_equivalent": "Approximation 2.725*head*peak turbine efficiency interpolated across head reference curves*peak generator efficiency, MWh/Mm3. Ignores hydraulic losses and joint-unit dispatch.",
        "supply_fraction": "(max(initial pool volume-max weekly min-volume,0)+remaining natural pool inflow)*energy_equivalent/(remaining_hours*plant_pmax); omits upstream generation and past dispatch",
        "catchment_inflow_fraction": "Sum natural hourly inflow over directed upstream reservoir graph divided by approximate full-plant discharge capacity; omits storage buffering and travel times",
        "spill": "max(initial pool volume+weekly natural pool inflow-pool max volume,0)*energy_equivalent/(168*plant_pmax)",
        "switching": "Max-sum binary-state DP at representative pmin and pmax; profit=(price-local terminal WV)*power; transition costs from YAML; free start/end states; feature=(best objective forcing on-best forcing off)/power. No water budgets or cascade constraints.",
        "columns": list(additions), "plants": plant_specs,
    }
    return pd.concat([x, pd.DataFrame(additions, index=x.index, copy=False)], axis=1, copy=False), spec
