"""Per-subset joint production hulls with fractional transition costs.

This input-only prototype uses free boundary states and a fixed finite grid.
It does not claim the unavailable competition SHOP command configuration.
"""

import warnings

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, hstack, vstack

from sintef_project.lp_features import HydraulicLP
from sintef_project.lp_joint_plant import (
    COMMON_LOAD_GRID, JointCurve, build_joint_curve, envelope_indices, simultaneous_points,
)
from sintef_project.lp_shared_loss_update import precise_conservation


def switching_specification() -> dict:
    """Return the predeclared prototype, including its boundary assumptions."""
    return {"version": "joint_subset_switching_v1", "grid": COMMON_LOAD_GRID.tolist(),
            "plants": ["Tokke", "Vinje"],
            "vertices": "Keep upper and lower Q/P hull separately for every nonempty active subset; do not include an off point within an active subset. Same physical128 common-load grid and feasibility checks as frozen joint v1.",
            "state": "u[j,t]=sum_v(unit_on[v,j]*lambda[v,t]) exactly. Nonnegative lambda mass<=1; unused mass is all-off. There is no independent unconstrained u variable.",
            "cost": "For t=1..167 add s[j,t]>=u[j,t]-u[j,t-1], d[j,t]>=u[j,t-1]-u[j,t], s,d>=0; subtract supplied startcost_const*s+stopcost_const*d. Cost numbers assumed in the same currency as market input, as in earlier physical priors.",
            "boundary": "No cost before hour0 or after hour167 because initial/end generator states are unavailable. Hour0 on-fraction still requires physical vertex operation and consumes water.",
            "controls": "Global-hull frozen joint cache; subset hull without costs; subset hull with costs. Fixed32 cases and all results reported; no tuning, threshold fitting, full cache or supervised model.",
            "frozen": "Other7 plants, initial heads, terminal water values, network, release penalties and constraints unchanged from joint v1.",
            "limitations": "Continuous fractional transition penalty, not integer startups. No within-hour transition charge, minimum up/down time or claimed exact SHOP semantics. Per-subset2D reduction preserves Q/P/on objectives, but not arbitrary unit power/discharge constraints. Unknown currency and commands remain assumptions.",
            "sources": ["https://docs.shop.sintef.energy/commands/set_linear_startup.html",
                        "https://docs.shop.sintef.energy/objects/generator/generator.html#initial-state"]}


def build_subset_joint_curve(data, index: int, plant_name: str) -> JointCurve:
    """Preserve all subset-specific Q/P frontiers without active zero-flow modes."""
    reference = build_joint_curve(data, index, plant_name)
    units = reference.specification["units"]
    qmax = np.asarray(reference.specification["reference_qmax_m3s"])
    gross_head = reference.specification["initial_gross_head_m"]
    flows, powers, upper_flags, subsets, fractions = [], [], [], [], []
    subset_counts = {}
    for subset in range(1, 2 ** len(units)):
        active = np.array([bool(subset & (1 << j)) for j in range(len(units))])
        q = COMMON_LOAD_GRID[:, None] * qmax[None, :] * active[None, :]
        p, _, feasible = simultaneous_points(data.model["generator"], data.model["plant"][plant_name],
                                              gross_head, units, q)
        q, p, load = q[feasible], p[feasible], COMMON_LOAD_GRID[feasible]
        if not len(q):
            subset_counts[str(subset)] = 0
            continue
        upper = envelope_indices(q.sum(axis=1), p.sum(axis=1), True)
        lower = envelope_indices(q.sum(axis=1), p.sum(axis=1), False)
        selected = sorted(set(upper + lower))
        upper_set = set(upper)
        flows.extend(q[selected])
        powers.extend(p[selected])
        upper_flags.extend(i in upper_set for i in selected)
        subsets.extend([subset] * len(selected))
        fractions.extend(load[selected].tolist())
        subset_counts[str(subset)] = len(selected)
    q, p = np.asarray(flows), np.asarray(powers)
    on = (q > 1e-12).astype(float)
    if not len(q) or not (on.sum(axis=1) > 0).all() or not (p.sum(axis=1) > 0).all():
        raise ValueError("An active subset must have positive physical operation")
    spec = dict(reference.specification)
    spec.update(subset_retained_vertices=subset_counts, vertex_subsets=subsets,
                vertex_load_fractions=fractions, retained_nonzero_vertices=len(q),
                hull_reduction="Separate upper/lower hull per active mask; no off origin in an active subset")
    return JointCurve(plant_name, reference.unit_indices, q.sum(axis=1) * .0036, p.sum(axis=1),
                      q * .0036, p, on, np.asarray(upper_flags), spec)


class SwitchingJointLP(HydraulicLP):
    """Optimize convex operation and adjacent-hour fractional switching together."""

    def solve(self, case: dict, curves: list[JointCurve], startcost: np.ndarray,
              stopcost: np.ndarray, time_limit: float = 20.) -> dict:
        """Solve one case with exact state-to-vertex links and free boundaries."""
        network, t = self.network, self.horizon
        g, r = len(network.units), len(network.pools)
        startcost, stopcost = np.asarray(startcost, float), np.asarray(stopcost, float)
        if startcost.shape != (g,) or stopcost.shape != (g,) or (startcost < 0).any() or (stopcost < 0).any():
            raise ValueError("Expected nonnegative per-unit transition costs")
        joint_units = np.concatenate([curve.unit_indices for curve in curves])
        if len(np.unique(joint_units)) != len(joint_units) or network.vest_units[joint_units].any():
            raise ValueError("Overlapping or unsupported joint units")
        cursor, mode_indices = self.n_variables, []
        for curve in curves:
            indices = np.arange(cursor, cursor + len(curve.total_flow) * t).reshape(-1, t)
            mode_indices.append(indices)
            cursor += indices.size
        starts = np.arange(cursor, cursor + len(joint_units) * (t - 1)).reshape(len(joint_units), t - 1)
        stops = starts + starts.size
        cursor += starts.size * 2
        eq_rows, eq_cols, eq_values = [], [], []
        ub_rows, ub_cols, ub_values = [], [], []
        local_unit = 0
        for plant_index, (curve, indices) in enumerate(zip(curves, mode_indices)):
            up, down = np.unique(network.unit_up[curve.unit_indices]), np.unique(network.unit_down[curve.unit_indices])
            if len(up) != 1 or len(down) != 1:
                raise ValueError("Joint units must share a route")
            for mode in range(len(curve.total_flow)):
                for hour in range(t):
                    column = indices[mode, hour]
                    eq_rows.append(up[0] * t + hour); eq_cols.append(column - self.n_variables); eq_values.append(curve.total_flow[mode])
                    if down[0] >= 0:
                        eq_rows.append(down[0] * t + hour); eq_cols.append(column - self.n_variables); eq_values.append(-curve.total_flow[mode])
                    ub_rows.append(plant_index * t + hour); ub_cols.append(column); ub_values.append(1.)
                for j in range(len(curve.unit_indices)):
                    if not curve.unit_on[mode, j]:
                        continue
                    for hour in range(1, t):
                        start_row = len(curves) * t + (local_unit + j) * (t - 1) + hour - 1
                        stop_row = start_row + starts.size
                        for row, sign in [(start_row, 1.), (stop_row, -1.)]:
                            ub_rows.extend([row, row]); ub_cols.extend([indices[mode, hour], indices[mode, hour - 1]]); ub_values.extend([sign, -sign])
            local_unit += len(curve.unit_indices)
        for k in range(len(joint_units)):
            for hour in range(t - 1):
                row = len(curves) * t + k * (t - 1) + hour
                ub_rows.extend([row, row + starts.size]); ub_cols.extend([starts[k, hour], stops[k, hour]]); ub_values.extend([-1., -1.])
        added = cursor - self.n_variables
        equality_added = coo_matrix((eq_values, (eq_rows, eq_cols)), shape=(r * t, added)).tocsc()
        a_eq = hstack([self.a_eq, equality_added], format="csc")
        inequalities = coo_matrix((ub_values, (ub_rows, ub_cols)), shape=(len(curves) * t + starts.size * 2, cursor)).tocsc()
        a_ub = vstack([hstack([self.a_ub, coo_matrix((self.a_ub.shape[0], added))]), inequalities], format="csc")
        energy = np.repeat(case["energy"][:, :, None], t, axis=2)
        negative = case["price"] < 0
        average = (case["energy"] * case["widths"]).sum(axis=1) / case["widths"].sum(axis=1)
        energy[:, :, negative] = average[:, None, None]
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
        for curve, indices in zip(curves, mode_indices):
            c[indices] = -curve.total_power[:, None] * case["price"][None, :]
            bounds[indices.ravel(), 1] = 1
        c[starts] = startcost[joint_units, None]
        c[stops] = stopcost[joint_units, None]
        rhs = np.asarray(case["natural"], dtype=np.float64).copy()
        rhs[:, 0] += case["initial"]
        b_ub = np.concatenate([-case["minimum"].ravel(), np.repeat(case["capacity"], t),
                               -case["bypass_minimum"].ravel(), case["vest_natural"] - case["vest_minimum"],
                               np.ones(len(curves) * t), np.zeros(starts.size * 2)])
        options = {"threads": 1, "time_limit": time_limit, "dual_feasibility_tolerance": 1e-7, "primal_feasibility_tolerance": 1e-7}
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Unrecognized options detected.*")
            optimized = linprog(c / scale, A_eq=a_eq, b_eq=rhs.ravel(), A_ub=a_ub, b_ub=b_ub,
                                bounds=bounds, method="highs", options=options)
        if not optimized.success:
            raise RuntimeError(f"Subset-switching LP failed: {optimized.message}")
        solution = optimized.x
        flow, power = solution[self.q].sum(axis=1), (solution[self.q] * energy).sum(axis=1)
        on_fraction = np.zeros((g, t))
        result = {}
        max_mass = 0.
        for curve, indices in zip(curves, mode_indices):
            weights = solution[indices]
            flow[curve.unit_indices] += curve.unit_flow.T @ weights
            power[curve.unit_indices] += curve.unit_power.T @ weights
            on_fraction[curve.unit_indices] = curve.unit_on.T @ weights
            max_mass = max(max_mass, float(weights.sum(axis=0).max()))
            np.testing.assert_allclose(flow[curve.unit_indices].sum(axis=0), curve.total_flow @ weights, atol=1e-10)
            np.testing.assert_allclose(power[curve.unit_indices].sum(axis=0), curve.total_power @ weights, atol=1e-8)
            result[f"mode_weights_{curve.plant}"] = weights
        delta = np.diff(on_fraction[joint_units], axis=1)
        switch_cost = float(np.sum(startcost[joint_units, None] * np.maximum(delta, 0))
                            + np.sum(stopcost[joint_units, None] * np.maximum(-delta, 0)))
        solver_switch_cost = float((c[starts] * solution[starts]).sum() + (c[stops] * solution[stops]).sum())
        if abs(switch_cost - solver_switch_cost) > 1e-5 or max_mass > 1 + 1e-8 or solution[self.q[joint_units]].max() > 1e-10:
            raise ValueError("Transition or convex allocation invariants failed")
        result.update(flow=flow, power=power, storage=solution[self.v], bypass=solution[self.bypass], spill=solution[self.spill],
                      water_shadow=-optimized.eqlin.marginals.reshape(r, t) * scale, minimum_slack=solution[self.low_slack],
                      maximum_slack=solution[self.high_slack], bypass_slack=solution[self.bypass_slack], vest_slack=solution[self.vest_slack],
                      joint_on_fraction=on_fraction, joint_unit_indices=joint_units, maximum_convex_mass=max_mass,
                      old_joint_q_max=float(solution[self.q[joint_units]].max()), objective=float(-optimized.fun * scale),
                      balance_residual=float(np.max(np.abs(a_eq @ solution - rhs.ravel()))),
                      iterations=int(optimized.nit), negative_price_hours=int(negative.sum()), n_variables=cursor, status=int(optimized.status),
                      startup_fraction=solution[starts], shutdown_fraction=solution[stops], switching_cost=switch_cost)
        result["total_balance_error"] = precise_conservation(case, result, network)
        if result["balance_residual"] > 1e-5 or abs(result["total_balance_error"]) > 1e-8:
            raise ValueError("Subset-switching water balance failed")
        return result
