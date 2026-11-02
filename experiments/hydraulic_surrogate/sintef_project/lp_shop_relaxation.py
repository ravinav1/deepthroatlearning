"""Fixed-head continuous generator-state ablation of the frozen hydraulic LP.

This tests a SHOP-inspired H1/H2 relaxation, not SHOP output or physical binary
commitment. The original LP, its input conversion and cached results are intact.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, hstack, vstack

from sintef_project.lp_features import HydraulicLP, Network


def effective_energy(case: dict, horizon: int) -> np.ndarray:
    """Reproduce the frozen LP's negative-price average conversion exactly."""
    energy = np.repeat(np.asarray(case["energy"])[:, :, None], horizon, axis=2)
    average = (case["energy"] * case["widths"]).sum(axis=1) / case["widths"].sum(axis=1)
    energy[:, :, case["price"] < 0] = average[:, None, None]
    return energy


def state_interval(case: dict, flow: np.ndarray, power: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return state bounds implied by both flow and power, without clipping."""
    lower = np.maximum(power / np.asarray(case["pmax"])[:, None],
                       flow / np.asarray(case["qmax"])[:, None])
    upper = np.minimum(1.0, np.minimum(power / np.asarray(case["pmin"])[:, None],
                                       flow / np.asarray(case["qmin"])[:, None]))
    return lower, upper


class ShopRelaxationLP:
    """Jointly optimize fractional generator states, water and transition costs.

    P is MW (equivalently MWh in one hour); Q is Mm3 per one-hour interval.
    Nominal YAML Pmin and base case-feasible Qmin/Qmax/Pmax are held fixed.
    The new state is bounded independently of the PQ segment decomposition.
    """

    def __init__(self, network: Network, horizon: int = 168, segments: int = 4):
        self.base = HydraulicLP(network, horizon, segments)
        self.network, self.horizon, self.segments = network, horizon, segments
        g, t = len(network.units), horizon
        if t < 1:
            raise ValueError("Horizon must be positive")
        cursor = self.base.n_variables

        def allocate(shape: tuple[int, int]) -> np.ndarray:
            nonlocal cursor
            result = np.arange(cursor, cursor + np.prod(shape)).reshape(shape)
            cursor += int(np.prod(shape))
            return result

        self.state = allocate((g, t))
        self.start = allocate((g, t - 1))
        self.stop = allocate((g, t - 1))
        self.n_variables = cursor
        self.a_eq = hstack([self.base.a_eq, coo_matrix((self.base.a_eq.shape[0],
                                                       cursor - self.base.n_variables))], format="csc")
        self.base_a_ub = hstack([self.base.a_ub, coo_matrix((self.base.a_ub.shape[0],
                                                           cursor - self.base.n_variables))], format="csc")
        rows, columns, values = [], [], []
        row = 0
        for unit in range(g):
            for hour in range(1, t):
                for col, val in [(self.state[unit, hour], 1), (self.state[unit, hour - 1], -1),
                                 (self.start[unit, hour - 1], -1)]:
                    rows.append(row); columns.append(col); values.append(val)
                row += 1
                for col, val in [(self.state[unit, hour - 1], 1), (self.state[unit, hour], -1),
                                 (self.stop[unit, hour - 1], -1)]:
                    rows.append(row); columns.append(col); values.append(val)
                row += 1
        self.transition_a_ub = coo_matrix((values, (rows, columns)), shape=(row, cursor)).tocsc()

    def solve(self, case: dict, transition_cost_scale: float = 1.0,
              time_limit: float = 20.0) -> dict:
        """Solve one fixed-head case, preserving raw states and feasibility proof."""
        base, network, t = self.base, self.network, self.horizon
        g, r = len(network.units), len(network.pools)
        for name in ["pmin", "pmax", "qmin", "qmax", "start_cost", "stop_cost"]:
            value = np.asarray(case[name], dtype=np.float64)
            if value.shape != (g,) or not np.isfinite(value).all():
                raise ValueError(f"Invalid unit parameter: {name}")
            if (value < 0).any() or (name not in ["start_cost", "stop_cost"] and (value == 0).any()):
                raise ValueError(f"Invalid parameter bounds: {name}")
        if transition_cost_scale not in (0.0, 1.0):
            raise ValueError("Only the predeclared zero-cost and supplied-cost arms are supported")
        if np.shape(case["energy"]) != (g, self.segments) or np.shape(case["widths"]) != (g, self.segments):
            raise ValueError("Only fixed initial-head segment arrays are supported")
        if not np.isfinite(case["energy"]).all() or not (np.asarray(case["widths"]) > 0).all():
            raise ValueError("Invalid segment inputs")
        np.testing.assert_allclose(np.sum(case["widths"], axis=1), case["qmax"], rtol=1e-12, atol=1e-12)
        energy = effective_energy(case, t)
        scale = 1 + np.abs(case["price"]).mean()
        penalty = case.get("slack_penalty", 1e7 + 100 * np.max(np.abs(case["terminal_value"])))
        c = np.zeros(self.n_variables)
        c[base.q] = -energy * case["price"][None, None, :]
        c[base.v[:, -1]] = -case["terminal_value"]
        c[base.bypass] = case.get("bypass_cost", 10 / 0.0036)
        c[base.spill] = case.get("spill_cost", 1000 / 0.0036)
        for index in (base.low_slack, base.high_slack, base.bypass_slack, base.vest_slack):
            c[index] = penalty
        c[self.start] = transition_cost_scale * np.asarray(case["start_cost"])[:, None]
        c[self.stop] = transition_cost_scale * np.asarray(case["stop_cost"])[:, None]
        bounds = np.column_stack([np.zeros(self.n_variables), np.full(self.n_variables, np.inf)])
        bounds[base.q.ravel(), 1] = np.repeat(case["widths"][:, :, None], t, axis=2).ravel()
        bounds[base.bypass.ravel(), 1] = case.get("bypass_capacity", 1000 * 0.0036)
        bounds[base.n_variables:, 1] = 1.0
        rhs = np.asarray(case["natural"], dtype=np.float64).copy()
        rhs[:, 0] += case["initial"]
        original_b_ub = np.concatenate([-case["minimum"].ravel(), np.repeat(case["capacity"], t),
                                        -case["bypass_minimum"].ravel(), case["vest_natural"] - case["vest_minimum"]])
        rows, columns, values = [], [], []
        row = 0
        for unit in range(g):
            for hour in range(t):
                # Lower P, upper P, lower Q, upper Q, respectively.
                for coefficients, state_coefficient in [
                    (-energy[unit, :, hour], case["pmin"][unit]),
                    (energy[unit, :, hour], -case["pmax"][unit]),
                    (-np.ones(self.segments), case["qmin"][unit]),
                    (np.ones(self.segments), -case["qmax"][unit]),
                ]:
                    for segment, coefficient in enumerate(coefficients):
                        rows.append(row); columns.append(base.q[unit, segment, hour]); values.append(coefficient)
                    rows.append(row); columns.append(self.state[unit, hour]); values.append(state_coefficient)
                    row += 1
        link_a_ub = coo_matrix((values, (rows, columns)), shape=(row, self.n_variables)).tocsc()
        a_ub = vstack([self.base_a_ub, link_a_ub, self.transition_a_ub], format="csc")
        b_ub = np.concatenate([original_b_ub, np.zeros(row + self.transition_a_ub.shape[0])])
        options = {"threads": 1, "time_limit": time_limit,
                   "dual_feasibility_tolerance": 1e-7, "primal_feasibility_tolerance": 1e-7}
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Unrecognized options detected.*")
            solved = linprog(c / scale, A_ub=a_ub, b_ub=b_ub, A_eq=self.a_eq,
                             b_eq=rhs.ravel(), bounds=bounds, method="highs", options=options)
        if not solved.success:
            raise RuntimeError(f"Linked-state LP failed: {solved.status}: {solved.message}")
        x = solved.x
        flow = x[base.q].sum(axis=1)
        power = (x[base.q] * energy).sum(axis=1)
        state = x[self.state]
        diff = np.diff(state, axis=1)
        reconstructed_cost = transition_cost_scale * float(
            np.sum(np.maximum(diff, 0) * np.asarray(case["start_cost"])[:, None])
            + np.sum(np.maximum(-diff, 0) * np.asarray(case["stop_cost"])[:, None]))
        raw_cost = float(c[self.start].ravel() @ x[self.start].ravel()
                         + c[self.stop].ravel() @ x[self.stop].ravel())
        np.testing.assert_allclose(raw_cost, reconstructed_cost, rtol=1e-9, atol=1e-5)
        equality_error = float(np.max(np.abs(self.a_eq @ x - rhs.ravel())))
        inequality_error = float(max(0.0, np.max(a_ub @ x - b_ub)))
        bound_error = float(max(0.0, np.max(bounds[:, 0] - x), np.max(x - bounds[:, 1])))
        if max(equality_error, inequality_error, bound_error) > 1e-5:
            raise RuntimeError("Linked-state LP feasibility check failed")
        lower, upper = state_interval(case, flow, power)
        if np.max(lower - state) > 1e-5 or np.max(state - upper) > 1e-5:
            raise RuntimeError("Linked-state interval check failed")
        sea = (flow[network.unit_down < 0].sum()
               + x[base.bypass][network.bypass_down < 0].sum()
               + x[base.spill][network.spill_down < 0].sum())
        balance = float(x[base.v][:, -1].sum() + sea
                        - np.sum(case["initial"], dtype=np.float64)
                        - np.sum(case["natural"], dtype=np.float64))
        if abs(balance) > 1e-5:
            raise RuntimeError("Linked-state LP aggregate water balance failed")
        objective = float(-solved.fun * scale)
        reconstructed_objective = float(
            np.sum(power * case["price"][None, :])
            + case["terminal_value"] @ x[base.v][:, -1]
            - case.get("bypass_cost", 10 / 0.0036) * x[base.bypass].sum()
            - case.get("spill_cost", 1000 / 0.0036) * x[base.spill].sum()
            - penalty * sum(x[index].sum() for index in
                            [base.low_slack, base.high_slack, base.bypass_slack, base.vest_slack])
            - reconstructed_cost)
        np.testing.assert_allclose(objective, reconstructed_objective, rtol=2e-11, atol=1e-5)
        return {"flow": flow, "power": power, "state": state,
                "raw_start": x[self.start], "raw_stop": x[self.stop],
                "state_lower": lower, "state_upper": upper,
                "segment_flow": x[base.q], "storage": x[base.v],
                "bypass": x[base.bypass], "spill": x[base.spill],
                "water_shadow": -solved.eqlin.marginals.reshape(r, t) * scale,
                "minimum_slack": x[base.low_slack], "maximum_slack": x[base.high_slack],
                "bypass_slack": x[base.bypass_slack], "vest_slack": x[base.vest_slack],
                "objective": objective, "reconstructed_objective": reconstructed_objective,
                "transition_cost": reconstructed_cost, "gross_objective": objective + reconstructed_cost,
                "balance_residual": equality_error, "inequality_residual": inequality_error,
                "bound_residual": bound_error, "total_balance_error": balance,
                "iterations": int(solved.nit), "status": int(solved.status),
                "negative_price_hours": int(np.sum(case["price"] < 0))}


def specification() -> dict:
    """Return the predeclared mechanism and explicit differences from SHOP."""
    return {
        "version": "shop_relaxed_state_fixed_head_v1",
        "states": "0<=u<=1; Pmin*u<=P<=Pmax*u; Qmin*u<=Q<=Qmax*u",
        "power": "P=sum(segment_Mm3_per_hour * MWh_per_Mm3), numerically MW for1h",
        "transition": "s>=u[t]-u[t-1], d>=u[t-1]-u[t], only t1..167; bounds0..1; supplied asymmetric constant costs",
        "boundaries": "No initial or final transition charge; no unknown initial state inferred",
        "minimums": "Pmin is supplied nominal YAML value; Qmin/Qmax/Pmax are frozen base initial-head values, not a reconstruction of SHOP's effective minimum algorithm",
        "arms": ["cached_base", "base_float64_rhs", "linked_zero_cost", "linked_supplied_cost"],
        "numerical_control": "New arms use float64 natural inflow before initial-volume RHS addition; fresh base on identical input isolates this from cached float32 RHS rounding",
        "unchanged": "Frozen v1 storage pools/routing/PQ/terminal conversion/slacks/negative-price average conversion; all14 units, fixed initial heads",
        "omissions": "No H3 feedback, head/reference-flow iteration, integer states, river-cost/tunnel reconstruction, unit priority, initial-state inference, labels or fitting",
        "source_scope": "Competition17.17 commands unknown; newer18.4i source and17.9.1 report motivate a falsifiable mechanism",
        "references": ["https://docs.shop.sintef.energy/commands/set_linear_startup.html",
                       "https://docs.shop.sintef.energy/commands/print_pqcurves.html",
                       "https://nva.sikt.no/registration/01a0f3805ef8-2645f514-8c93-4994-af5b-3f76bc65c52f"],
    }
