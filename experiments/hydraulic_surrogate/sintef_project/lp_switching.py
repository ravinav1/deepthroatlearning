"""Startup-cost smoothing of LP shadow-price margins, without new optimization.

This is a binary switching prior conditional on an already solved relaxed LP.
Its fixed hourly shadow prices are not reoptimized when commitment changes.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from sintef_project.lp_features import UNIT_ORDER
from sintef_project.physics import switching_advantage


def add_lp_switching_features(x: pd.DataFrame, root: Path,
                              lp_prefix: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Append 28 dimensionless per-unit switching advantages at nominal Pmin/Pmax.

    Supports either v1 ``lp`` or directed Haukeli v2 ``lp2`` margins. Call before
    monetary normalization, or after it when monetary_scale_inverse is present.
    No labels, case IDs, fitted statistics, or future reservoir volumes are read.
    """
    root = Path(root)
    if (root / "sintef_project").is_dir():
        root = root / "sintef_project"
    topology_path = root / "data/extended/Tokke_Vinje_topology.yaml"
    generator = yaml.safe_load(topology_path.read_text())["model"]["generator"]
    if len(x) == 0 or len(x) % 168:
        raise ValueError("Expected nonempty complete 168-hour cases")
    n_cases = len(x) // 168
    if not np.array_equal(x["horizon_hour"].to_numpy(), np.tile(np.arange(168), n_cases)):
        raise ValueError("Expected case-major hours 0 through 167")
    available = [prefix for prefix in ["lp", "lp2"]
                 if all(f"{prefix}_economic_margin_{unit}" in x for unit in UNIT_ORDER)]
    if lp_prefix is None:
        if len(available) != 1:
            raise ValueError("Specify lp_prefix when zero or multiple complete LP versions exist")
        lp_prefix = available[0]
    if lp_prefix not in available:
        raise ValueError(f"Missing complete LP margin columns for {lp_prefix}")
    if "price" in x:
        price = x["price"].to_numpy(dtype=np.float64).reshape(n_cases, 168)
        scale = 1 + np.abs(price).mean(axis=1)
    elif "monetary_scale_inverse" in x:
        inverse = x["monetary_scale_inverse"].to_numpy(dtype=np.float64).reshape(n_cases, 168)
        if not (inverse > 0).all() or not np.all(inverse == inverse[:, :1]):
            raise ValueError("Inverse monetary scale must be positive and constant within each case")
        scale = 1 / inverse[:, 0]
    else:
        raise ValueError("Need raw price or monetary_scale_inverse to recover fixed startup-cost scale")
    if not np.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError("Invalid monetary scale")
    additions, unit_parameters = {}, {}
    for unit in UNIT_ORDER:
        data = generator[unit]
        start, stop = float(data["startcost_const"]), float(data["stopcost_const"])
        if start < 0 or stop < 0:
            raise ValueError(f"Negative transition costs for {unit}")
        margin = x[f"{lp_prefix}_economic_margin_{unit}"].to_numpy(dtype=np.float64).reshape(n_cases, 168)
        if not np.isfinite(margin).all():
            raise ValueError(f"Nonfinite LP economic margin for {unit}")
        unit_parameters[unit] = {"start_cost": start, "stop_cost": stop}
        for point in ["min", "max"]:
            power = float(data[f"p_{point}"])
            if not np.isfinite(power) or power <= 0:
                raise ValueError(f"Invalid nominal p_{point} for {unit}")
            name = f"lpdp_switch_margin_{point}_{unit}"
            if name in x:
                raise ValueError("LP switching features have already been added")
            profit = margin * scale[:, None] * power
            advantage = switching_advantage(profit, start, stop)
            additions[name] = (advantage / (scale[:, None] * power)).ravel().astype(np.float32)
            unit_parameters[unit][f"p_{point}_mw"] = power
    frame = pd.DataFrame(additions, index=x.index, copy=False)
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Nonfinite LP switching features")
    source_path = Path(__file__).resolve()
    dp_path = source_path.with_name("physics.py")
    spec = {"version": "lp_dual_switching_v1", "lp_prefix": lp_prefix, "columns": frame.columns.tolist(),
            "formula": "margin=(price-(lambda_up-lambda_down)/best_energy)/(1+mean(abs(case price))). profit=margin*scale*nominal_power. DP advantage=(best objective forcing on-best forcing off)/(scale*nominal_power).",
            "scale": "Same case mean absolute price+1 as LP; if already normalized, recover from monetary_scale_inverse.",
            "unit_parameters": unit_parameters,
            "states": "Binary on/off; free initial and terminal state; asymmetric fixed startup/stop costs; 168 one-hour intervals; each case independent.",
            "units": "All28 outputs dimensionless. YAML nominal power is MW; one-hour profit is EUR when price/LP dual is EUR/MWh. YAML start/stop monetary values are assumed consistent with dataset EUR; original SHOP docs use currency-specific labels and case-generation conversion commands are unavailable.",
            "approximations": "Constant best-energy conversion makes LP dual opportunity cost an average EUR/MWh cost at both nominal Pmin/Pmax. Actual part-load efficiencies and case-feasible Pmax differ. Fixed water shadows do not respond to the changed dispatch; water budgets, head changes, cascade coupling, reserve constraints and MIP settings are not reoptimized. This is an auxiliary prior, not a feasible hydraulic schedule or SHOP reconstruction.",
            "provenance": "Only legal input-derived LP margins, per-case price scale and supplied unit parameters; no targets or trained parameters.",
            "topology_path": str(topology_path), "topology_sha256": hashlib.sha256(topology_path.read_bytes()).hexdigest(),
            "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "switching_dp_sha256": hashlib.sha256(dp_path.read_bytes()).hexdigest()}
    return pd.concat([x, frame], axis=1), spec
