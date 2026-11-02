"""A documented, approximate hydraulic LP for features, never ground-truth labels.

Nine storage pools, separate turbine/bypass routes, fixed initial-head concave
generation curves, hourly water balances, and terminal water values are used.
This does not reproduce SHOP's iterative nonlinear or integer optimization.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
import yaml

from sintef_project.features import _read_timeseries, _lookup
from sintef_project.physics import POOLS


UNIT_ORDER = ["Hogga_G1", "Tokke_G1", "Tokke_G2", "Tokke_G3", "Tokke_G4", "Lio_G1", "Byrte_G1", "Vinje_G1", "Vinje_G2", "Vinje_G3", "Haukeli_G1", "Songa_G1", "Kjela_G1", "Vesle Kjela_G1"]
POOL_ORDER = list(POOLS)
DOWNSTREAM = {"Hogga": None, "Tokke": "Hogga", "Lio": "Hogga", "Byrte": "Lio", "Vinje": "Tokke", "Haukeli": "Tokke", "Songa": "Vinje", "Kjela": "Vinje", "Vesle Kjela": "Kjela"}
# (source pool, destination pool or sea, mandatory-flow column, reaches Vest)
BYPASSES = [("Hogga", None, "b_Bandak", False), ("Tokke", "Hogga", "b_Vinjevatn", True),
            ("Lio", "Hogga", "b_Byrtevatn", True), ("Byrte", "Hogga", None, True),
            ("Vinje", "Tokke", "b_Hyljelihyl", False), ("Vinje", "Hogga", None, True),
            ("Haukeli", "Tokke", None, False), ("Songa", "Vinje", None, False),
            ("Kjela", "Vinje", None, False), ("Vesle Kjela", "Vinje", "b_Kjelavatn", False)]


@dataclass
class Network:
    """Topology needed by the generic linear water-balance model."""

    pools: list[str]
    units: list[str]
    unit_up: np.ndarray
    unit_down: np.ndarray
    bypass_up: np.ndarray
    bypass_down: np.ndarray
    spill_down: np.ndarray
    vest_units: np.ndarray
    vest_bypasses: np.ndarray
    vest_spills: np.ndarray


class HydraulicLP:
    """Reuse sparse constraint structure across independently solved cases."""

    def __init__(self, network: Network, horizon: int = 168, segments: int = 4):
        self.network, self.horizon, self.segments = network, horizon, segments
        r, g, b, t, k = len(network.pools), len(network.units), len(network.bypass_up), horizon, segments
        cursor = 0

        def allocate(shape):
            nonlocal cursor
            result = np.arange(cursor, cursor + np.prod(shape)).reshape(shape)
            cursor += int(np.prod(shape))
            return result

        self.q = allocate((g, k, t))
        self.v = allocate((r, t))
        self.bypass = allocate((b, t))
        self.spill = allocate((r, t))
        self.low_slack = allocate((r, t))
        self.high_slack = allocate((r, t))
        self.bypass_slack = allocate((b, t))
        self.vest_slack = allocate((t,))
        self.n_variables = cursor
        rows, cols, vals = [], [], []

        def eq(row, col, value):
            rows.append(row); cols.append(int(col)); vals.append(value)

        for pool in range(r):
            for hour in range(t):
                row = pool * t + hour
                eq(row, self.v[pool, hour], 1)
                if hour:
                    eq(row, self.v[pool, hour - 1], -1)
                eq(row, self.spill[pool, hour], 1)
                if network.spill_down[pool] >= 0:
                    eq(network.spill_down[pool] * t + hour, self.spill[pool, hour], -1)
        for unit in range(g):
            for segment in range(k):
                for hour in range(t):
                    eq(network.unit_up[unit] * t + hour, self.q[unit, segment, hour], 1)
                    if network.unit_down[unit] >= 0:
                        eq(network.unit_down[unit] * t + hour, self.q[unit, segment, hour], -1)
        for edge in range(b):
            for hour in range(t):
                eq(network.bypass_up[edge] * t + hour, self.bypass[edge, hour], 1)
                if network.bypass_down[edge] >= 0:
                    eq(network.bypass_down[edge] * t + hour, self.bypass[edge, hour], -1)
        self.a_eq = coo_matrix((vals, (rows, cols)), shape=(r * t, cursor)).tocsc()
        rows, cols, vals = [], [], []
        next_row = 0

        def inequality(terms):
            nonlocal next_row
            for col, coefficient in terms:
                rows.append(next_row); cols.append(int(col)); vals.append(coefficient)
            next_row += 1

        for pool in range(r):
            for hour in range(t):
                inequality([(self.v[pool, hour], -1), (self.low_slack[pool, hour], -1)])
        for pool in range(r):
            for hour in range(t):
                inequality([(self.v[pool, hour], 1), (self.high_slack[pool, hour], -1)])
        for edge in range(b):
            for hour in range(t):
                inequality([(self.bypass[edge, hour], -1), (self.bypass_slack[edge, hour], -1)])
        for hour in range(t):
            terms = [(self.vest_slack[hour], -1)]
            terms += [(self.q[unit, segment, hour], -1) for unit in np.flatnonzero(network.vest_units) for segment in range(k)]
            terms += [(self.bypass[edge, hour], -1) for edge in np.flatnonzero(network.vest_bypasses)]
            terms += [(self.spill[pool, hour], -1) for pool in np.flatnonzero(network.vest_spills)]
            inequality(terms)
        self.a_ub = coo_matrix((vals, (rows, cols)), shape=(next_row, cursor)).tocsc()

    def solve(self, case: dict, time_limit: float = 20.0) -> dict:
        """Solve one case, returning physical flows, shadow prices and diagnostics."""
        network, t = self.network, self.horizon
        r, g, b = len(network.pools), len(network.units), len(network.bypass_up)
        scale = 1 + np.abs(case["price"]).mean()
        penalty = case.get("slack_penalty", 1e7 + 100 * np.max(np.abs(case["terminal_value"])))
        energy = np.repeat(case["energy"][:, :, None], t, axis=2)
        # Nonpositive revenue invalidates incremental-segment loading. At negative
        # prices use a single average conversion for all segments, preserving a
        # linear physical production/discharge relation rather than allowing free
        # water disposal through low-efficiency out-of-order segments.
        negative = case["price"] < 0
        average_energy = (case["energy"] * case["widths"]).sum(axis=1) / case["widths"].sum(axis=1)
        energy[:, :, negative] = average_energy[:, None, None]
        c = np.zeros(self.n_variables)
        c[self.q] = -energy * case["price"][None, None, :]
        c[self.v[:, -1]] = -case["terminal_value"]
        c[self.bypass] = case.get("bypass_cost", 10 / 0.0036)
        c[self.spill] = case.get("spill_cost", 1000 / 0.0036)
        for indices in (self.low_slack, self.high_slack, self.bypass_slack, self.vest_slack):
            c[indices] = penalty
        bounds = np.column_stack([np.zeros(self.n_variables), np.full(self.n_variables, np.inf)])
        bounds[self.q.ravel(), 1] = np.repeat(case["widths"][:, :, None], t, axis=2).ravel()
        bounds[self.bypass.ravel(), 1] = case.get("bypass_capacity", 1000 * 0.0036)
        rhs = case["natural"].copy()
        rhs[:, 0] += case["initial"]
        b_ub = np.concatenate([-case["minimum"].ravel(), np.repeat(case["capacity"], t),
                               -case["bypass_minimum"].ravel(), case["vest_natural"] - case["vest_minimum"]])
        options = {"threads": 1, "time_limit": time_limit, "dual_feasibility_tolerance": 1e-7, "primal_feasibility_tolerance": 1e-7}
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Unrecognized options detected.*")
            result = linprog(c / scale, A_ub=self.a_ub, b_ub=b_ub, A_eq=self.a_eq,
                             b_eq=rhs.ravel(), bounds=bounds, method="highs", options=options)
        if not result.success:
            raise RuntimeError(f"LP failed status={result.status}: {result.message}")
        solution = result.x
        flows = solution[self.q].sum(axis=1)
        power = (solution[self.q] * energy).sum(axis=1)
        water_shadow = -result.eqlin.marginals.reshape(r, t) * scale
        residual = float(np.max(np.abs(self.a_eq @ solution - rhs.ravel())))
        if residual > 1e-5:
            raise RuntimeError(f"Water balance residual {residual}")
        sea = flows[network.unit_down < 0].sum() + solution[self.bypass][network.bypass_down < 0].sum() + solution[self.spill][network.spill_down < 0].sum()
        total_balance_error = float(solution[self.v][:, -1].sum() + sea - case["initial"].sum() - case["natural"].sum())
        return {"flow": flows, "power": power, "storage": solution[self.v], "bypass": solution[self.bypass],
                "spill": solution[self.spill], "water_shadow": water_shadow, "minimum_slack": solution[self.low_slack],
                "maximum_slack": solution[self.high_slack], "bypass_slack": solution[self.bypass_slack],
                "vest_slack": solution[self.vest_slack], "objective": float(-result.fun * scale),
                "balance_residual": residual, "total_balance_error": total_balance_error,
                "iterations": int(result.nit), "negative_price_hours": int(negative.sum()), "status": int(result.status)}


def unit_curve(generator: dict, plant: dict, gross_head: float, segments: int = 4) -> dict:
    """Build a four-segment concave initial-head approximation for one unit."""
    curves = sorted(generator["turb_eff_curves"], key=lambda curve: curve["ref"])
    q_upper = max(max(curve["x"]) for curve in curves)
    q_grid = np.unique(np.concatenate([np.linspace(0, q_upper, 257), *[np.array(curve["x"]) for curve in curves]]))
    penstock = plant["penstock_loss"][generator["penstock"] - 1]
    loss = sum(float(value) for value in plant["main_loss"]) + float(penstock)
    effective_head = np.maximum(gross_head - loss * q_grid ** 2, 0.01)
    efficiencies = np.stack([np.interp(q_grid, curve["x"], curve["y"]) for curve in curves]) / 100
    turbine_efficiency = np.array([np.interp(h, [curve["ref"] for curve in curves], efficiencies[:, idx]) for idx, h in enumerate(effective_head)])
    p = 0.00981 * q_grid * effective_head * turbine_efficiency
    for _ in range(8):
        p = 0.00981 * q_grid * effective_head * turbine_efficiency * np.interp(p, generator["gen_eff_curve"]["x"], generator["gen_eff_curve"]["y"]) / 100
    maximum_index = int(np.argmax(p))
    q_grid, p = q_grid[:maximum_index + 1], p[:maximum_index + 1]
    if np.max(p) >= generator["p_max"]:
        qmax = float(np.interp(generator["p_max"], p, q_grid))
        mask = q_grid < qmax
        q_grid, p = np.append(q_grid[mask], qmax), np.append(p[mask], generator["p_max"])
    power_qmin = float(np.interp(generator["p_min"], p, q_grid))
    qmin = power_qmin
    for _ in range(8):
        limit_from_curve = float(np.interp(max(gross_head - loss * qmin ** 2, 0.01),
                                          [curve["ref"] for curve in curves],
                                          [min(curve["x"]) for curve in curves]))
        qmin = max(power_qmin, limit_from_curve)
    # Upper concave hull including the origin gives decreasing segment slopes.
    hull = []
    for q, power in zip(q_grid, p):
        while len(hull) >= 2:
            a, previous = hull[-2:]
            if (previous[1] - a[1]) / (previous[0] - a[0]) <= (power - previous[1]) / (q - previous[0]) + 1e-12:
                hull.pop()
            else:
                break
        hull.append((q, power))
    hull = np.array(hull)
    best_q = hull[1, 0]
    if best_q >= hull[-1, 0] - 1e-9:
        knots = np.linspace(0, hull[-1, 0], segments + 1)
    else:
        knots = np.concatenate([[0], np.linspace(best_q, hull[-1, 0], segments)])
    power = np.interp(knots, hull[:, 0], hull[:, 1])
    widths = np.diff(knots) * 0.0036
    energy = np.diff(power) / widths
    if not (widths > 0).all() or not np.isfinite(energy).all() or (np.diff(energy) > 1e-6).any():
        raise ValueError("Invalid concave generation curve")
    return {"widths": widths, "energy": energy, "qmin": qmin * 0.0036, "qmax": knots[-1] * 0.0036,
            "pmax": float(power[-1]), "best_energy": float(energy[0])}


class HydroCaseData:
    """Read only documented inputs and run/date metadata; no commitment labels."""

    def __init__(self, root: Path):
        root = Path(root)
        if (root / "sintef_project").is_dir():
            root = root / "sintef_project"
        self.root = root
        self.topology = yaml.safe_load((root / "data/extended/Tokke_Vinje_topology.yaml").read_text())
        self.model = self.topology["model"]
        train = pd.read_csv(root / "data/kernel/Unit_commitment_decisions.csv", usecols=["Run No", "starttime"])
        self.n_train = len(train)
        self.cases = pd.concat([train, pd.read_csv(root / "prediction_mapping.csv")], ignore_index=True)
        self.cases["starttime"] = pd.to_datetime(self.cases["starttime"], utc=True)
        self.volume = _read_timeseries(root / "data/kernel/Historical_volume_2015_2024.csv")
        self.water = _read_timeseries(root / "data/kernel/Synthetic_water_value_2015_2024.csv")
        self.price = _read_timeseries(root / "data/kernel/Historical_day_ahead_price_2015_2025.csv")
        self.inflow = _read_timeseries(root / "data/kernel/Historical_inflow_1958_2025.csv")
        self.minimum = _read_timeseries(root / "data/extended/Constraint_min_volume.csv")
        self.flow_minimum = _read_timeseries(root / "data/extended/Constraint_min_flow.csv")
        self.network = Network(POOL_ORDER, UNIT_ORDER,
            np.array([POOL_ORDER.index(unit.rsplit("_G", 1)[0]) for unit in UNIT_ORDER]),
            np.array([POOL_ORDER.index(DOWNSTREAM[unit.rsplit("_G", 1)[0]]) if DOWNSTREAM[unit.rsplit("_G", 1)[0]] else -1 for unit in UNIT_ORDER]),
            np.array([POOL_ORDER.index(edge[0]) for edge in BYPASSES]),
            np.array([POOL_ORDER.index(edge[1]) if edge[1] else -1 for edge in BYPASSES]),
            np.array([-1, 0, 0, 0, 0, 1, 4, 4, 4]),
            np.array([unit == "Lio_G1" for unit in UNIT_ORDER]),
            np.array([edge[3] for edge in BYPASSES]),
            np.array([False, True, True, True, True, False, False, False, False]))
        self.plant_downstream_reservoir = {edge["from"]: edge["to"] for edge in self.topology["connections"] if edge["from_type"] == "plant" and edge["to_type"] == "reservoir"}

    def case(self, index: int) -> dict:
        """Assemble a single legal case with units of Mm3, hours, MWh and EUR."""
        start = self.cases.iloc[index].starttime
        times = pd.date_range(start, periods=168, freq="h")
        volumes = self.volume.loc[start]
        water = self.water.loc[start + pd.Timedelta(days=7)]
        price = _lookup(self.price, times, "LP price")[:, 0].astype(np.float64)
        inflow = pd.DataFrame(_lookup(self.inflow, times, "LP inflow"), columns=self.inflow.columns)
        minimum = pd.DataFrame(_lookup(self.minimum, times, "LP min volume"), columns=self.minimum.columns)
        flow_minimum = pd.DataFrame(_lookup(self.flow_minimum, times, "LP min flow"), columns=self.flow_minimum.columns)
        head = {reservoir: float(np.interp(volumes[reservoir], data["vol_head"]["x"], data["vol_head"]["y"])) for reservoir, data in self.model["reservoir"].items()}
        curves = []
        for unit in UNIT_ORDER:
            plant_name = unit.rsplit("_G", 1)[0]
            plant = self.model["plant"][plant_name]
            pool = POOLS[plant_name]
            source_head = np.mean([head[reservoir] for reservoir in pool]) if plant_name == "Songa" else head[pool[0]]
            outlet = max(plant["outlet_line"], head.get(self.plant_downstream_reservoir.get(plant_name), -np.inf))
            curves.append(unit_curve(self.model["generator"][unit], plant, max(source_head - outlet, 1)))
        best_energy = np.array([max(curves[j]["best_energy"] for j, unit in enumerate(UNIT_ORDER) if unit.rsplit("_G", 1)[0] == pool) for pool in POOL_ORDER])
        global_values = {}

        def terminal(pool):
            if pool is None:
                return 0.0
            if pool not in global_values:
                global_values[pool] = float(water[POOLS[pool][0]]) * best_energy[POOL_ORDER.index(pool)] + terminal(DOWNSTREAM[pool])
            return global_values[pool]

        natural = np.stack([inflow[pool].sum(axis=1).to_numpy() * 0.0036 for pool in POOLS.values()])
        vest_natural = inflow["r_Vest_Vassdraget"].to_numpy() * 0.0036
        natural[POOL_ORDER.index("Hogga")] += vest_natural
        return {"price": price, "initial": np.array([sum(volumes[r] for r in pool) for pool in POOLS.values()]),
                "capacity": np.array([sum(self.model["reservoir"][r]["max_vol"] for r in pool) for pool in POOLS.values()]),
                "minimum": np.stack([sum((minimum[r].to_numpy() for r in pool if r in minimum), np.zeros(168)) for pool in POOLS.values()]),
                "natural": natural, "terminal_value": np.array([terminal(pool) for pool in POOL_ORDER]),
                "widths": np.stack([curve["widths"] for curve in curves]), "energy": np.stack([curve["energy"] for curve in curves]),
                "qmin": np.array([curve["qmin"] for curve in curves]), "qmax": np.array([curve["qmax"] for curve in curves]),
                "pmax": np.array([curve["pmax"] for curve in curves]),
                "bypass_minimum": np.stack([flow_minimum[edge[2]].to_numpy() * 0.0036 if edge[2] else np.zeros(168) for edge in BYPASSES]),
                "vest_natural": vest_natural, "vest_minimum": flow_minimum["r_Vest_Vassdraget"].to_numpy() * 0.0036}


def result_features(case: dict, result: dict, network: Network) -> pd.DataFrame:
    """Convert an LP solution to interpretable continuous unit-hour predictors."""
    features = {}
    scale = 1 + np.abs(case["price"]).mean()
    for j, unit in enumerate(network.units):
        upstream, downstream = network.unit_up[j], network.unit_down[j]
        local_shadow = result["water_shadow"][upstream] - (result["water_shadow"][downstream] if downstream >= 0 else 0)
        opportunity_cost = local_shadow / case["energy"][j, 0]
        features[f"lp_qmin_ratio_{unit}"] = result["flow"][j] / max(case["qmin"][j], 1e-8)
        features[f"lp_power_fraction_{unit}"] = result["power"][j] / max(case["pmax"][j], 1e-8)
        features[f"lp_economic_margin_{unit}"] = (case["price"] - opportunity_cost) / scale
    for pool, name in enumerate(network.pools):
        units = network.unit_up == pool
        features[f"lp_plant_power_fraction_{name}"] = result["power"][units].sum(axis=0) / case["pmax"][units].sum()
        features[f"lp_storage_fill_{name}"] = result["storage"][pool] / case["capacity"][pool]
        features[f"lp_storage_min_slack_{name}"] = result["minimum_slack"][pool] / case["capacity"][pool]
        features[f"lp_storage_max_slack_{name}"] = result["maximum_slack"][pool] / case["capacity"][pool]
    frame = pd.DataFrame(features, dtype=np.float32)
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Nonfinite LP features")
    return frame


def math_specification() -> dict:
    """Return the frozen approximation and provenance for cache manifests."""
    return {
        "version": "pool_lp_v1", "pools": POOLS, "units": UNIT_ORDER, "turbine_downstream": DOWNSTREAM, "bypass_routes": BYPASSES,
        "balance": "v[t]-v[t-1]+all_outflows-all_upstream_transfers=natural_inflow[t], Mm3; t0 RHS includes initialvolume. Storage stays nonnegative; no water-creation slack.",
        "objective": "Maximize hourly price*production+sum(global terminalWV*final storage)-bypass/spill costs-softconstraint penalties; divide entire objective by1+mean(abs(price)) for conditioning; rescale duals afterward.",
        "PQ": "Fixed initial head. Interpolate head from initial individual reservoir volume; outlet=max(outletline,downstream head); Songa mean of its2 heads. Unit-alone quadratic main+ownpenstock loss.257-point discharge grid plus efficiency knots; iterative generator-efficiency interpolation; enforce Pmax and physical curve Qmax; upperconcave hull;4segments beginning at best-efficiency point.",
        "negative_prices": "All incremental segments use the same average full-discharge energy conversion at negative-price hours, avoiding out-of-order low-efficiency disposal. This is a deliberate linear approximation.",
        "terminal": "Local CSV EUR/MWh at start+7d times maximum unit best-point MWh/Mm3 in plant plus downstream global terminalvalue recursively. No raw EUR/MWh upstream/downstream subtraction.",
        "constraints": "Hourly pool minimum is the sum of supplied individual minimum volumes; summed pool maximum storage. Separate minimum bypass flows. Vest minimum includes own natural inflow,Lio discharge and designatedbypass/spill routes. All minimum/maximum violations have penalized slack; physical water conservation exact.",
        "penalties": "bypass10/0.0036 EUR/Mm3,spill1000/0.0036 EUR/Mm3, bypassmax1000m3/s; soft slackpenalty=1e7+100*max(abs(globalterminalWV)) EUR/Mm3 perhour. YAMLcurrency treated consistent with priceEUR, exact case-generationcurrencycommands unavailable.",
        "omissions": "No integer commitment,startstop,minruntime,ramping,head trajectory iteration,shared-penstock unit interactions,tunnel flow limits,individual reservoir storage withinpools,convex river flow-cost tables,head-dependent spill curves,or exact SHOPcommands. Pooling is approximate.",
        "dual_sign": "With stated equality orientation and minimization solver, economic water value=-eqlin.marginals*objective_scale. Unit opportunitycost=(lambda_up-lambda_down)/best_energy.",
        "solver": "scipy.optimize.linprog methodhighs,threads1,time_limit20s,primal/dual feasibility1e-7; no randomness or learnedparameters",
        "source_links": ["https://docs.shop.sintef.energy/examples/pq_curves/pq_curves.html", "https://docs.shop.sintef.energy/examples/water_values/water_values.html", "https://docs.shop.sintef.energy/objects/generator/generator.html", "https://docs.scipy.org/doc/scipy/reference/optimize.linprog-highs.html"],
        "interpretation": "Features only. LPdischarge and the0.5Qmin heuristic are not claimed to match unknown competition solver configuration.",
    }


def add_lp_features(x: pd.DataFrame, root: Path, cache_dir: Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Append a complete verified cache in exact training-plus-test case order.

    Every exported LP predictor is dimensionless: q/Qmin, power/Pmax, economic
    margin divided by mean absolute case price+1, storage/capacity, and slack/
    capacity. No raw currency-valued predictor needs further normalization.
    """
    root = Path(root).resolve()
    repository = root.parent if root.name == "sintef_project" else root
    data_root = repository / "sintef_project"
    cache_dir = repository / "artifacts/lp-full-001" if cache_dir is None else Path(cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = repository / cache_dir
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "complete":
        raise ValueError("LP cache is not complete")

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    for category in ("data", "source"):
        for relative, expected in manifest[category].items():
            if digest(repository / relative) != expected:
                raise ValueError(f"LP cache {category} checksum mismatch: {relative}")
    features_path = cache_dir / "features.npz"
    if digest(features_path) != manifest["artifacts"]["features.npz"]:
        raise ValueError("LP cache feature artifact checksum mismatch")
    train = pd.read_csv(data_root / "data/kernel/Unit_commitment_decisions.csv", usecols=["Run No", "starttime"])
    mapping = pd.read_csv(data_root / "prediction_mapping.csv")
    cases = pd.concat([train, mapping], ignore_index=True)
    expected_ids = cases["Run No"].to_numpy()
    expected_dates = pd.to_datetime(cases["starttime"], utc=True).astype(str).tolist()
    expected_hours = np.tile(np.arange(168), len(cases))
    if len(x) != len(expected_hours) or not np.array_equal(x["horizon_hour"].to_numpy(), expected_hours):
        raise ValueError("Input features do not contain the complete case-hour order")
    cache = np.load(features_path, allow_pickle=False)
    if not np.array_equal(cache["case_ids"], expected_ids) or manifest["selected_case_dates"] != expected_dates:
        raise ValueError("LP cache case order or dates mismatch")
    columns = cache["columns"].tolist()
    values = cache["features"]
    if columns != manifest["feature_names"] or values.shape != (len(cases), 168, len(columns)):
        raise ValueError("LP cache shape or column order mismatch")
    if cache["units"].tolist() != UNIT_ORDER or not np.isfinite(values).all():
        raise ValueError("Invalid LP cache units or feature values")
    if set(columns).intersection(x.columns):
        raise ValueError("LP features have already been added")
    frame = pd.DataFrame(values.reshape(len(x), len(columns)), columns=columns, index=x.index, copy=False)
    spec = {"cache_directory": str(cache_dir), "manifest_sha256": digest(manifest_path),
            "features_sha256": digest(features_path), "source_verified": manifest["source"], "data_verified": manifest["data"],
            "shape": list(frame.shape), "columns": columns, "mathematics": manifest["math"],
            "monetary_schema": "All78 exported features are dimensionless. lp_economic_margin=(price-local dual opportunity cost)/(1+mean(abs(case price))); retain lp_* during later monetary normalization. Raw water dual EUR/Mm3 values exist only in diagnostic case archives."}
    return pd.concat([x, frame], axis=1), spec
