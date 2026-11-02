"""Experimental single hydraulic head update; frozen LP v1/v2 stay unchanged."""

import warnings

import numpy as np
from scipy.optimize import linprog

from sintef_project.lp_features import HydraulicLP, POOL_ORDER, UNIT_ORDER, unit_curve


def single_reservoir_heads(data, index: int, result: dict) -> dict:
    """Estimate hourly mean head from exact single-reservoir LP storage states.

    Interpolate level at beginning/end storage, then average the two levels.
    This is a midpoint approximation, not an assertion of SHOP's discretization.
    No observed post-start volume or pooled-storage allocation is used.
    """
    start = data.cases.iloc[index].starttime
    volume = data.volume.loc[start]
    levels = {}
    initial_levels = {}
    for reservoir, pool in [("Bandak", "Hogga"), ("Vinjevatn", "Tokke")]:
        curve = data.model["reservoir"][reservoir]["vol_head"]
        endpoint = result["storage"][POOL_ORDER.index(pool)]
        boundary = np.r_[float(volume[reservoir]), endpoint]
        level = np.interp(boundary, curve["x"], curve["y"])
        levels[reservoir] = (level[:-1] + level[1:]) / 2
        initial_levels[reservoir] = level[0]
    hogga_outlet = float(data.model["plant"]["Hogga"]["outlet_line"])
    tokke_outlet = float(data.model["plant"]["Tokke"]["outlet_line"])
    return {"Hogga": levels["Bandak"] - hogga_outlet,
            "Tokke": levels["Vinjevatn"] - np.maximum(tokke_outlet, levels["Bandak"]),
            "initial_Hogga": initial_levels["Bandak"] - hogga_outlet,
            "initial_Tokke": initial_levels["Vinjevatn"] - max(tokke_outlet, initial_levels["Bandak"])}


def shared_loss_correction(data, result: dict, plant_name: str) -> dict[str, np.ndarray]:
    """Estimate omitted shared head loss at previous-iteration unit discharges.

    Return additional loss relative to each unit operating alone. Nonnegative
    generation flows share the plant main tunnel and their assigned penstock.
    This estimates a missing mechanism; it does not change a generation curve.
    """
    indices = [i for i, unit in enumerate(UNIT_ORDER) if unit.rsplit("_G", 1)[0] == plant_name]
    q = result["flow"] / .0036
    plant = data.model["plant"][plant_name]
    main_loss = sum(float(value) for value in plant["main_loss"])
    total_q = q[indices].sum(axis=0)
    corrections = {}
    for i in indices:
        unit = UNIT_ORDER[i]
        penstock = data.model["generator"][unit]["penstock"]
        group = [j for j in indices if data.model["generator"][UNIT_ORDER[j]]["penstock"] == penstock]
        group_q = q[group].sum(axis=0)
        coefficient = float(plant["penstock_loss"][penstock - 1])
        corrections[unit] = main_loss * (total_q ** 2 - q[i] ** 2) + coefficient * (group_q ** 2 - q[i] ** 2)
    return corrections


def hourly_segments(values: np.ndarray, units: int, segments: int, horizon: int) -> np.ndarray:
    """Validate/broadcast constant or hourly segment coefficients to G×K×T."""
    values = np.asarray(values, dtype=np.float64)
    if values.shape == (units, segments):
        values = np.broadcast_to(values[:, :, None], (units, segments, horizon)).copy()
    elif values.shape == (units, segments, horizon):
        values = values.copy()
    else:
        raise ValueError(f"Expected {(units, segments)} or {(units, segments, horizon)}, got {values.shape}")
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Segment coefficients must be finite and positive")
    return values


def update_case_head(data, index: int, case: dict, previous: dict,
                     plants: tuple[str, ...] = ("Hogga",)) -> tuple[dict, dict]:
    """Rebuild selected unit PQ curves at prior-LP hourly mean heads exactly once."""
    if not set(plants).issubset({"Hogga", "Tokke"}):
        raise ValueError("Only exact single-reservoir Hogga/Tokke head updates are supported")
    heads = single_reservoir_heads(data, index, previous)
    result = dict(case)
    g, k = case["energy"].shape
    horizon = len(case["price"])
    result["energy"] = hourly_segments(case["energy"], g, k, horizon)
    result["widths"] = hourly_segments(case["widths"], g, k, horizon)
    for name in ["qmin", "qmax", "pmax"]:
        result[name] = np.repeat(case[name][:, None], horizon, axis=1)
    for unit_index, unit in enumerate(UNIT_ORDER):
        plant_name = unit.rsplit("_G", 1)[0]
        if plant_name not in plants:
            continue
        for hour in range(horizon):
            curve = unit_curve(data.model["generator"][unit], data.model["plant"][plant_name],
                               float(heads[plant_name][hour]), segments=k)
            result["energy"][unit_index, :, hour] = curve["energy"]
            result["widths"][unit_index, :, hour] = curve["widths"]
            for name in ["qmin", "qmax", "pmax"]:
                result[name][unit_index, hour] = curve[name]
    # Keep the initial-head terminal conversion: it represents the supplied
    # terminal energy value definition, not the iterated production curve.
    return result, heads


class HourlyHeadLP(HydraulicLP):
    """Same sparse hydraulic network with hour-dependent PQ widths and slopes."""

    def solve(self, case: dict, time_limit: float = 20.) -> dict:
        network, t = self.network, self.horizon
        r, g = len(network.pools), len(network.units)
        energy = hourly_segments(case["energy"], g, self.segments, t)
        widths = hourly_segments(case["widths"], g, self.segments, t)
        if (np.diff(energy, axis=1) > 1e-6).any():
            raise ValueError("Production segment slopes must be nonincreasing each hour")
        negative = case["price"] < 0
        average_energy = (energy * widths).sum(axis=1) / widths.sum(axis=1)
        energy[:, :, negative] = average_energy[:, None, negative]
        scale = 1 + np.abs(case["price"]).mean()
        penalty = case.get("slack_penalty", 1e7 + 100 * np.max(np.abs(case["terminal_value"])))
        c = np.zeros(self.n_variables)
        c[self.q] = -energy * case["price"][None, None, :]
        c[self.v[:, -1]] = -case["terminal_value"]
        c[self.bypass] = case.get("bypass_cost", 10 / .0036)
        c[self.spill] = case.get("spill_cost", 1000 / .0036)
        for indices in [self.low_slack, self.high_slack, self.bypass_slack, self.vest_slack]:
            c[indices] = penalty
        bounds = np.column_stack([np.zeros(self.n_variables), np.full(self.n_variables, np.inf)])
        bounds[self.q.ravel(), 1] = widths.ravel()
        bounds[self.bypass.ravel(), 1] = case.get("bypass_capacity", 1000 * .0036)
        rhs = np.asarray(case["natural"], dtype=np.float64).copy()
        rhs[:, 0] += case["initial"]
        b_ub = np.concatenate([-case["minimum"].ravel(), np.repeat(case["capacity"], t),
                               -case["bypass_minimum"].ravel(), case["vest_natural"] - case["vest_minimum"]])
        options = {"threads": 1, "time_limit": time_limit, "dual_feasibility_tolerance": 1e-7, "primal_feasibility_tolerance": 1e-7}
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Unrecognized options detected.*")
            optimized = linprog(c / scale, A_eq=self.a_eq, b_eq=rhs.ravel(), A_ub=self.a_ub,
                                b_ub=b_ub, bounds=bounds, method="highs", options=options)
        if not optimized.success:
            raise RuntimeError(f"Hourly-head LP failed: {optimized.message}")
        solution = optimized.x
        flow = solution[self.q].sum(axis=1)
        power = (solution[self.q] * energy).sum(axis=1)
        sea = flow[network.unit_down < 0].sum() + solution[self.bypass][network.bypass_down < 0].sum() + solution[self.spill][network.spill_down < 0].sum()
        residual = float(np.max(np.abs(self.a_eq @ solution - rhs.ravel())))
        if residual > 1e-5:
            raise RuntimeError(f"Water balance residual {residual}")
        return {"flow": flow, "power": power, "storage": solution[self.v], "bypass": solution[self.bypass],
                "spill": solution[self.spill], "water_shadow": -optimized.eqlin.marginals.reshape(r, t) * scale,
                "minimum_slack": solution[self.low_slack], "maximum_slack": solution[self.high_slack],
                "bypass_slack": solution[self.bypass_slack], "vest_slack": solution[self.vest_slack],
                "objective": float(-optimized.fun * scale), "balance_residual": residual,
                "total_balance_error": float(solution[self.v][:, -1].sum() + sea - case["initial"].sum() - case["natural"].sum()),
                "iterations": int(optimized.nit), "negative_price_hours": int(negative.sum()), "status": int(optimized.status),
                "effective_segment_energy": energy}
