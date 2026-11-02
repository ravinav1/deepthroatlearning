"""Joint Tokke/Vinje finite-grid operating-point convex mixtures.

Each vertex uses simultaneous main-tunnel and penstock losses. Convex mixtures
represent relaxed time sharing, not exact hourly integer commitment or SHOP.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, hstack, vstack

from sintef_project.lp_features import HydraulicLP, UNIT_ORDER, math_specification, result_features
from sintef_project.lp_shared_loss_update import precise_conservation


# Predetermined once, before benchmark results. No grid-size selection.
COMMON_LOAD_GRID = np.arange(1, 129, dtype=np.float64) / 128


@dataclass
class JointCurve:
    """Feasible joint vertices and their exact unit allocations."""

    plant: str
    unit_indices: np.ndarray
    total_flow: np.ndarray  # Mm3 per one-hour interval
    total_power: np.ndarray  # MW = MWh per one-hour interval
    unit_flow: np.ndarray  # vertices × local units, Mm3
    unit_power: np.ndarray  # vertices × local units, MW
    unit_on: np.ndarray  # vertices × local units, binary before mixture
    upper_vertex: np.ndarray
    specification: dict


def simultaneous_points(generators: dict, plant: dict, gross_head: float,
                        units: list[str], flow_m3s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate grid points with true simultaneous losses and feasibility checks."""
    flow_m3s = np.asarray(flow_m3s, dtype=np.float64)
    if flow_m3s.ndim != 2 or flow_m3s.shape[1] != len(units) or (flow_m3s < 0).any():
        raise ValueError("Expected nonnegative point-by-unit discharges")
    n_points = len(flow_m3s)
    total = flow_m3s.sum(axis=1)
    main_loss = sum(float(value) for value in plant["main_loss"]) * total ** 2
    power = np.zeros_like(flow_m3s)
    head = np.zeros_like(flow_m3s)
    valid = np.ones(n_points, dtype=bool)
    penstocks = np.array([generators[unit]["penstock"] for unit in units])
    for j, unit in enumerate(units):
        generator = generators[unit]
        group_q = flow_m3s[:, penstocks == penstocks[j]].sum(axis=1)
        head[:, j] = gross_head - main_loss - float(plant["penstock_loss"][penstocks[j] - 1]) * group_q ** 2
        curves = sorted(generator["turb_eff_curves"], key=lambda curve: curve["ref"])
        references = np.array([curve["ref"] for curve in curves], dtype=float)
        efficiencies = np.stack([np.interp(flow_m3s[:, j], curve["x"], curve["y"]) for curve in curves]) / 100
        if len(references) == 1:
            turbine = efficiencies[0]
        else:
            lower = np.clip(np.searchsorted(references, head[:, j], side="right") - 1, 0, len(references) - 2)
            weight = np.clip((head[:, j] - references[lower]) / (references[lower + 1] - references[lower]), 0, 1)
            turbine = efficiencies[lower, np.arange(n_points)] * (1 - weight) + efficiencies[lower + 1, np.arange(n_points)] * weight
        hydraulic = .00981 * flow_m3s[:, j] * head[:, j] * turbine
        production = hydraulic.copy()
        for _ in range(8):
            production = hydraulic * np.interp(production, generator["gen_eff_curve"]["x"], generator["gen_eff_curve"]["y"]) / 100
        power[:, j] = production
        qmin = np.interp(head[:, j], references, [min(curve["x"]) for curve in curves])
        qmax = np.interp(head[:, j], references, [max(curve["x"]) for curve in curves])
        active = flow_m3s[:, j] > 1e-12
        valid &= ~active | ((head[:, j] > 0) & (flow_m3s[:, j] >= qmin - 1e-8)
                            & (flow_m3s[:, j] <= qmax + 1e-8)
                            & (production >= float(generator["p_min"]) - 1e-8)
                            & (production <= float(generator["p_max"]) + 1e-8))
    valid &= np.isfinite(power).all(axis=1) & np.isfinite(head).all(axis=1)
    return power, head, valid


def envelope_indices(flow: np.ndarray, power: np.ndarray, upper: bool) -> list[int]:
    """Return a deterministic upper/lower convex boundary of finite points."""
    order = np.lexsort((np.arange(len(flow)), -power if upper else power, flow))
    distinct = []
    for index in order:
        if distinct and abs(flow[index] - flow[distinct[-1]]) <= 1e-12:
            continue
        distinct.append(int(index))
    hull = []
    for index in distinct:
        while len(hull) >= 2:
            a, b = hull[-2:]
            first = (power[b] - power[a]) / (flow[b] - flow[a])
            second = (power[index] - power[b]) / (flow[index] - flow[b])
            remove = first <= second + 1e-10 if upper else first >= second - 1e-10
            if not remove:
                break
            hull.pop()
        hull.append(index)
    return hull


def build_joint_curve(data, index: int, plant_name: str) -> JointCurve:
    """Enumerate every active subset at the fixed128 common-discharge fractions."""
    if plant_name not in ["Tokke", "Vinje"]:
        raise ValueError("Joint ablation is limited to Tokke and Vinje")
    unit_indices = np.array([j for j, unit in enumerate(UNIT_ORDER) if unit.rsplit("_G", 1)[0] == plant_name])
    units = [UNIT_ORDER[j] for j in unit_indices]
    generators = data.model["generator"]
    plant = data.model["plant"][plant_name]
    initial = data.volume.loc[data.cases.iloc[index].starttime]
    upstream = {"Tokke": "Vinjevatn", "Vinje": "Vaamarvatn"}[plant_name]
    downstream = {"Tokke": "Bandak", "Vinje": "Vinjevatn"}[plant_name]

    def level(reservoir):
        curve = data.model["reservoir"][reservoir]["vol_head"]
        return float(np.interp(initial[reservoir], curve["x"], curve["y"]))

    gross_head = level(upstream) - max(float(plant["outlet_line"]), level(downstream))
    reference_qmax = np.array([max(max(curve["x"]) for curve in generators[unit]["turb_eff_curves"]) for unit in units])
    all_q, all_p, masks, fractions = [np.zeros((1, len(units)))], [np.zeros((1, len(units)))], [0], [0.]
    candidate_count = 0
    feasible_by_subset = {}
    for subset in range(1, 2 ** len(units)):
        active = np.array([bool(subset & (1 << j)) for j in range(len(units))])
        q = COMMON_LOAD_GRID[:, None] * reference_qmax[None, :] * active[None, :]
        p, _, feasible = simultaneous_points(generators, plant, gross_head, units, q)
        all_q.append(q[feasible])
        all_p.append(p[feasible])
        masks.extend([subset] * int(feasible.sum()))
        fractions.extend(COMMON_LOAD_GRID[feasible].tolist())
        feasible_by_subset[str(subset)] = int(feasible.sum())
        candidate_count += len(q)
    q, p = np.vstack(all_q), np.vstack(all_p)
    total_q, total_p = q.sum(axis=1), p.sum(axis=1)
    upper = envelope_indices(total_q, total_p, upper=True)
    lower = envelope_indices(total_q, total_p, upper=False)
    # The all-off origin is represented by unused convex-combination mass.
    vertices = sorted(set(upper + lower) - {0}, key=lambda i: (total_q[i], total_p[i], i))
    if not vertices:
        raise ValueError(f"No feasible operating vertices for {plant_name}")
    upper_set = set(upper)
    if (np.interp(total_q, total_q[upper], total_p[upper]) < total_p - 1e-7).any():
        raise ValueError("Upper envelope fails to dominate a candidate point")
    if (np.interp(total_q, total_q[lower], total_p[lower]) > total_p + 1e-7).any():
        raise ValueError("Lower envelope exceeds a candidate point")
    return JointCurve(plant_name, unit_indices, total_q[vertices] * .0036, total_p[vertices],
                      q[vertices] * .0036, p[vertices], (q[vertices] > 1e-12).astype(float),
                      np.array([i in upper_set for i in vertices]),
                      {"units": units, "initial_gross_head_m": gross_head, "reference_qmax_m3s": reference_qmax.tolist(),
                       "common_load_grid": COMMON_LOAD_GRID.tolist(), "candidate_count": candidate_count,
                       "feasible_count_including_off": len(q), "feasible_by_subset": feasible_by_subset,
                       "upper_vertices_including_off": len(upper), "lower_vertices_including_off": len(lower),
                       "retained_nonzero_vertices": len(vertices), "vertex_subsets": [masks[i] for i in vertices],
                       "vertex_load_fractions": [fractions[i] for i in vertices]})


class JointPlantLP(HydraulicLP):
    """Augment the frozen network with convex-mixture columns and exact routing."""

    def solve(self, case: dict, joint_curves: list[JointCurve], time_limit: float = 20.) -> dict:
        network, t = self.network, self.horizon
        g, r = len(network.units), len(network.pools)
        joint_units = np.concatenate([curve.unit_indices for curve in joint_curves])
        if len(np.unique(joint_units)) != len(joint_units) or network.vest_units[joint_units].any():
            raise ValueError("Joint units overlap or require unsupported Vest routing")
        mode_indices, cursor = [], self.n_variables
        rows, cols, values = [], [], []
        mix_rows, mix_cols = [], []
        for plant_index, curve in enumerate(joint_curves):
            size = len(curve.total_flow) * t
            indices = np.arange(cursor, cursor + size).reshape(-1, t)
            mode_indices.append(indices)
            cursor += size
            up = np.unique(network.unit_up[curve.unit_indices])
            down = np.unique(network.unit_down[curve.unit_indices])
            if len(up) != 1 or len(down) != 1:
                raise ValueError("Joint plant units must share upstream/downstream routing")
            for mode in range(len(curve.total_flow)):
                for hour in range(t):
                    column = indices[mode, hour] - self.n_variables
                    rows.append(up[0] * t + hour); cols.append(column); values.append(curve.total_flow[mode])
                    if down[0] >= 0:
                        rows.append(down[0] * t + hour); cols.append(column); values.append(-curve.total_flow[mode])
                    mix_rows.append(plant_index * t + hour); mix_cols.append(indices[mode, hour])
        added = cursor - self.n_variables
        balance_modes = coo_matrix((values, (rows, cols)), shape=(r * t, added)).tocsc()
        a_eq = hstack([self.a_eq, balance_modes], format="csc")
        mix = coo_matrix((np.ones(len(mix_rows)), (mix_rows, mix_cols)), shape=(len(joint_curves) * t, cursor)).tocsc()
        a_ub = vstack([hstack([self.a_ub, coo_matrix((self.a_ub.shape[0], added))]), mix], format="csc")
        energy = np.repeat(case["energy"][:, :, None], t, axis=2)
        negative = case["price"] < 0
        average_energy = (case["energy"] * case["widths"]).sum(axis=1) / case["widths"].sum(axis=1)
        energy[:, :, negative] = average_energy[:, None, None]
        scale = 1 + np.abs(case["price"]).mean()
        penalty = case.get("slack_penalty", 1e7 + 100 * np.max(np.abs(case["terminal_value"])))
        c = np.zeros(cursor)
        c[self.q] = -energy * case["price"][None, None, :]
        c[self.v[:, -1]] = -case["terminal_value"]
        c[self.bypass] = case.get("bypass_cost", 10 / .0036)
        c[self.spill] = case.get("spill_cost", 1000 / .0036)
        for indices in [self.low_slack, self.high_slack, self.bypass_slack, self.vest_slack]:
            c[indices] = penalty
        bounds = np.column_stack([np.zeros(cursor), np.full(cursor, np.inf)])
        bounds[self.q.ravel(), 1] = np.repeat(case["widths"][:, :, None], t, axis=2).ravel()
        bounds[self.q[joint_units].ravel(), 1] = 0
        bounds[self.bypass.ravel(), 1] = case.get("bypass_capacity", 1000 * .0036)
        for curve, indices in zip(joint_curves, mode_indices):
            c[indices] = -curve.total_power[:, None] * case["price"][None, :]
            bounds[indices.ravel(), 1] = 1
        rhs = np.asarray(case["natural"], dtype=np.float64).copy()
        rhs[:, 0] += case["initial"]
        b_ub = np.concatenate([-case["minimum"].ravel(), np.repeat(case["capacity"], t),
                               -case["bypass_minimum"].ravel(), case["vest_natural"] - case["vest_minimum"],
                               np.ones(len(joint_curves) * t)])
        options = {"threads": 1, "time_limit": time_limit, "dual_feasibility_tolerance": 1e-7, "primal_feasibility_tolerance": 1e-7}
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Unrecognized options detected.*")
            optimized = linprog(c / scale, A_eq=a_eq, b_eq=rhs.ravel(), A_ub=a_ub,
                                b_ub=b_ub, bounds=bounds, method="highs", options=options)
        if not optimized.success:
            raise RuntimeError(f"Joint-plant LP failed: {optimized.message}")
        solution = optimized.x
        flow = solution[self.q].sum(axis=1)
        power = (solution[self.q] * energy).sum(axis=1)
        on_fraction = np.zeros((g, t))
        output = {}
        max_mass = 0.
        for curve, indices in zip(joint_curves, mode_indices):
            weights = solution[indices]
            local_flow, local_power = curve.unit_flow.T @ weights, curve.unit_power.T @ weights
            flow[curve.unit_indices] += local_flow
            power[curve.unit_indices] += local_power
            on_fraction[curve.unit_indices] = curve.unit_on.T @ weights
            max_mass = max(max_mass, float(weights.sum(axis=0).max()))
            if not np.allclose(local_flow.sum(axis=0), curve.total_flow @ weights, atol=1e-10):
                raise ValueError("Joint unit flow allocations do not sum to plant flow")
            if not np.allclose(local_power.sum(axis=0), curve.total_power @ weights, atol=1e-10):
                raise ValueError("Joint unit power allocations do not sum to plant power")
            output[f"mode_weights_{curve.plant}"] = weights
        old_max = float(solution[self.q[joint_units]].max())
        if max_mass > 1 + 1e-8 or old_max > 1e-10 or on_fraction.max() > 1 + 1e-8:
            raise ValueError("Invalid convex mass or nonzero disabled independent-unit dispatch")
        output.update(flow=flow, power=power, storage=solution[self.v], bypass=solution[self.bypass],
                      spill=solution[self.spill], water_shadow=-optimized.eqlin.marginals.reshape(r, t) * scale,
                      minimum_slack=solution[self.low_slack], maximum_slack=solution[self.high_slack],
                      bypass_slack=solution[self.bypass_slack], vest_slack=solution[self.vest_slack],
                      joint_on_fraction=on_fraction, joint_unit_indices=joint_units, maximum_convex_mass=max_mass,
                      old_joint_q_max=old_max, objective=float(-optimized.fun * scale),
                      balance_residual=float(np.max(np.abs(a_eq @ solution - rhs.ravel()))),
                      iterations=int(optimized.nit), negative_price_hours=int(negative.sum()),
                      n_variables=cursor, status=int(optimized.status))
        output["total_balance_error"] = precise_conservation(case, output, network)
        if output["balance_residual"] > 1e-5 or abs(output["total_balance_error"]) > 1e-8:
            raise ValueError("Joint model water balance failed")
        return output


def result_joint_features(case: dict, result: dict, network) -> pd.DataFrame:
    """Export87 dimensionless predictors with joint on-fraction/count semantics."""
    frame = result_features(case, result, network)
    frame.columns = [name.replace("lp_", "lpjoint_", 1) for name in frame]
    for j in result["joint_unit_indices"]:
        frame[f"lpjoint_on_fraction_{network.units[j]}"] = result["joint_on_fraction"][j]
    for plant in ["Tokke", "Vinje"]:
        indices = [j for j, unit in enumerate(network.units) if unit.rsplit("_G", 1)[0] == plant]
        frame[f"lpjoint_active_unit_count_{plant}"] = result["joint_on_fraction"][indices].sum(axis=0)
    frame = frame.astype(np.float32)
    if len(frame.columns) != 87 or not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Invalid joint-plant feature schema or nonfinite values")
    return frame


def joint_math_specification() -> dict:
    """Return the frozen grid, mathematical formulation and feature assumptions."""
    return {"version": "joint_plant_v1", "base": math_specification(),
            "plants": ["Tokke", "Vinje"], "common_load_grid": COMMON_LOAD_GRID.tolist(),
            "operating_points": "All15/7 nonempty active-unit subsets at fixed fractions1/128..128/128 of each unit's maximum turbine-curve discharge; simultaneous main-tunnel/assigned-penstock flow squared losses; interpolate turbine/head efficiency and8 generator-efficiency iterations. Reject nonpositive head, interpolated turbine-flow bounds, Pmin/Pmax violations. Include off origin.",
            "hull": "Deterministic upper and lower convex hull vertices in plant total discharge/production, retaining each actual point's unit discharge, power and binary active mask. Keep both frontiers for negative-price behavior; no independent incremental segment decomposition of unit allocations.",
            "LP": "Nonnegative lambda(vertex,hour), sum(lambda)<=1 perplant/hour; unused mass=all-off. Route sum(lambda*Qtotal) exactly once along original plant route, maximize price*sum(lambda*Ptotal). Old independent Tokke/Vinje q variables have zero upper bound. All other7 plants, constraints and initial-head terminal conversions unchanged fromv1.",
            "allocation": "Unit flow/power/onfraction are convex-weighted vertex allocations. Verify allocations sum to plant Q/P, lambda mass<=1 and old independent dispatch=0. On fractions are continuous relaxed time-sharing features, not binary labels or calibrated probabilities.",
            "features": "78 v1-comparable features renamed lpjoint_ plus7 joint-unit on fractions and2 plant active-unit counts, all dimensionless. The q/Qmin and economic-margin denominators retain each unit's v1 initial-head solo curve to preserve a comparable coordinate; this is not a true unit marginal production factor for the joint hull.",
            "limitations": "Finite common-load grid restricts feasible operating points; convexification permits time-sharing across discrete subsets within an hour. This is neither exactMIP nor full nonlinearSHOP. No startup costs, iterative reservoir heads, head-dependent spill, individual within-pool storage or command-state reconstruction. Feasible polytope/allocations may be degenerate.",
            "solver": "scipy HiGHS one thread,20s percase, primal/dual tolerance1e-7; no training, seed or tuning"}


def add_lp_joint_features(x: pd.DataFrame, root: Path,
                          cache_dir: Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Append a complete strict-verified joint cache, replacing v1 in an ablation."""
    from sintef_project.lp_features import add_lp_features

    root = Path(root).resolve()
    repository = root.parent if root.name == "sintef_project" else root
    cache_dir = repository / "artifacts/lp-joint-full-001" if cache_dir is None else Path(cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = repository / cache_dir
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    if manifest["math"]["version"] != "joint_plant_v1":
        raise ValueError("Expected a joint-plant v1 cache")
    result, spec = add_lp_features(x, root, cache_dir)
    if len(spec["columns"]) != 87 or not all(column.startswith("lpjoint_") for column in spec["columns"]):
        raise ValueError("Unexpected joint-plant feature schema")
    spec["monetary_schema"] = "All87 lpjoint_* columns are dimensionless; preserve during normalization. Joint economic margins retain baseline unit-solo best-energy denominators as comparative predictors, not exact unit derivatives of joint production. On fractions/counts are physical convex allocations, not probabilities forced into labels."
    return result, spec
