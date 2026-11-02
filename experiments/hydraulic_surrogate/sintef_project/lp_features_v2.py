"""Isolated Haukeli directional-storage ablation of the frozen pool LP v1.

Only Haukeli's two reservoirs are separated. The other eight pools, initial-head
generation curves and economic approximations remain those of v1. This is an
input-only physical correction, not a reconstruction of SHOP or commitment.
"""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sintef_project.lp_features import (
    HydroCaseData, Network, POOL_ORDER, result_features, math_specification,
)


HAUKELI = POOL_ORDER.index("Haukeli")


def directed_haukeli_network(base: Network) -> Network:
    """Append upstream Langeidvatn storage and a one-way river into Vatjern."""
    upstream = len(base.pools)
    return replace(
        base,
        pools=[*base.pools, "Langeidvatn"],
        bypass_up=np.append(base.bypass_up, upstream),
        bypass_down=np.append(base.bypass_down, HAUKELI),
        spill_down=np.append(base.spill_down, HAUKELI),
        vest_bypasses=np.append(base.vest_bypasses, False),
        vest_spills=np.append(base.vest_spills, False),
    )


def split_haukeli_case(base: dict, initial: np.ndarray, natural: np.ndarray,
                       capacity: np.ndarray, minimum: np.ndarray) -> dict:
    """Replace pooled Haukeli with Vatjern and append upstream Langeidvatn.

    The two-row inputs are ordered Vatjern, Langeidvatn. Values are Mm3 or
    Mm3/hour. Both reservoirs retain v1's common terminal value, verified equal
    for these reservoirs in the supplied local water-value data. The additional
    river has zero cost to isolate directionality from the omitted nonlinear
    river-cost mechanism. Existing bypass costs/routes remain unchanged.
    """
    result = {key: np.array(value, copy=True) if isinstance(value, np.ndarray) else value
              for key, value in base.items()}
    for key, values in (("initial", initial), ("capacity", capacity),
                        ("natural", natural), ("minimum", minimum)):
        existing = np.asarray(result[key], dtype=np.float64).copy()
        existing[HAUKELI] = values[0]
        result[key] = np.concatenate([existing, np.asarray(values[1:2], dtype=np.float64)], axis=0)
    result["terminal_value"] = np.append(result["terminal_value"], result["terminal_value"][HAUKELI])
    horizon = len(result["price"])
    n_edges = len(result["bypass_minimum"])
    result["bypass_minimum"] = np.vstack([result["bypass_minimum"], np.zeros(horizon)])
    result["bypass_cost"] = np.concatenate([np.full(n_edges, 10 / 0.0036), [0.0]])[:, None]
    return result


class DirectedHaukeliData(HydroCaseData):
    """Use the frozen v1 legal inputs with Haukeli storage direction corrected."""

    def __init__(self, root: Path):
        super().__init__(root)
        self.base_network = self.network
        self.network = directed_haukeli_network(self.network)
        expected = [("Langeidvatn", "r_Langeidvatn"), ("r_Langeidvatn", "Vatjern"),
                    ("Langeidvatn", "s_Langeidvatn"), ("s_Langeidvatn", "Vatjern"),
                    ("Vatjern", "Haukeli")]
        connections = {(edge["from"], edge["to"]) for edge in self.topology["connections"]}
        if not set(expected).issubset(connections):
            raise ValueError("Haukeli topology differs from the reviewed directed routes")
        if not np.array_equal(self.water["Vatjern"].to_numpy(), self.water["Langeidvatn"].to_numpy()):
            raise ValueError("Haukeli reservoir terminal values differ; review conversion before use")

    def case(self, index: int) -> dict:
        """Split only initial storage, legal horizon inflow, capacity and minima."""
        case = super().case(index)
        start = self.cases.iloc[index].starttime
        times = pd.date_range(start, periods=168, freq="h")
        names = ["Vatjern", "Langeidvatn"]
        initial = self.volume.loc[start, names].to_numpy(dtype=np.float64)
        natural = self.inflow.loc[times, names].to_numpy(dtype=np.float64).T * 0.0036
        capacity = np.array([self.model["reservoir"][name]["max_vol"] for name in names])
        minimum = np.stack([self.minimum.loc[times, name].to_numpy(dtype=np.float64)
                            if name in self.minimum else np.zeros(168) for name in names])
        return split_haukeli_case(case, initial, natural, capacity, minimum)


def result_features_v2(case: dict, result: dict, data: DirectedHaukeliData) -> pd.DataFrame:
    """Return v1-comparable features plus separate Haukeli reservoir states."""
    combined_case, combined_result = dict(case), dict(result)
    for key in ("capacity",):
        values = case[key][:-1].copy()
        values[HAUKELI] += case[key][-1]
        combined_case[key] = values
    for key in ("storage", "minimum_slack", "maximum_slack"):
        values = result[key][:-1].copy()
        values[HAUKELI] += result[key][-1]
        combined_result[key] = values
    combined_result["water_shadow"] = result["water_shadow"][:-1]
    frame = result_features(combined_case, combined_result, data.base_network)
    frame.columns = [column.replace("lp_", "lp2_", 1) for column in frame]
    frame["lp2_haukeli_vatjern_fill"] = result["storage"][HAUKELI] / case["capacity"][HAUKELI]
    frame["lp2_haukeli_langeidvatn_fill"] = result["storage"][-1] / case["capacity"][-1]
    unit = data.network.units.index("Haukeli_G1")
    frame["lp2_haukeli_river_qmin_ratio"] = result["bypass"][-1] / case["qmin"][unit]
    return frame.astype(np.float32)


def math_specification_v2() -> dict:
    """Document the single intended mathematical change and remaining limits."""
    return {"version": "haukeli_directed_lp_v2", "base": math_specification(),
            "change": "Split Haukeli into Vatjern (0.3 Mm3) and Langeidvatn (32.1 Mm3); turbine and existing bypass/spill drain Vatjern. Add nonnegative river Langeidvatn->Vatjern, upper capacity 1000 m3/s, zero flow cost; upstream emergency spill also enters Vatjern.",
            "isolation": "Keep all other eight pooled reservoirs, PQ curves and economic assumptions from v1. Keep equal global terminal values for both Haukeli reservoirs; supplied local water-value columns are asserted identical. No reverse river flow and no aggregated Haukeli storage capacity.",
            "limitations": "The actual river's nonlinear flow-cost table, head-dependent spill, transit delays, changing heads and unknown SHOP commands remain omitted. A free forward river is an optimistic relaxation, not exact channel hydraulics. Other internal pools remain freely connected.",
            "numeric": "Split Haukeli balance inputs use float64; pre-existing other pool aggregations retain the v1 inputs. Solver converts all natural/initial balance arrays to float64 through the split routine.",
            "features": "78 dimensionless v1-comparable lp2_* features plus separate Vatjern/Langeidvatn fill and directional river flow divided by Haukeli Qmin. LP outputs are features, never forced commitment labels."}


def add_lp_features_v2(x: pd.DataFrame, root: Path,
                       cache_dir: Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Append the verified complete v2 cache using v1's frozen integrity checks.

    For a matched model ablation call this instead of ``add_lp_features``. All
    81 columns are dimensionless and remain unchanged by monetary normalization.
    Both versions can coexist by name, but that is a different, larger ablation.
    """
    from sintef_project.lp_features import add_lp_features

    root = Path(root).resolve()
    repository = root.parent if root.name == "sintef_project" else root
    cache_dir = repository / "artifacts/lp-v2-full-001" if cache_dir is None else Path(cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = repository / cache_dir
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    if manifest["math"]["version"] != "haukeli_directed_lp_v2":
        raise ValueError("Expected a directed Haukeli v2 cache")
    result, spec = add_lp_features(x, root, cache_dir)
    if len(spec["columns"]) != 81 or not all(column.startswith("lp2_") for column in spec["columns"]):
        raise ValueError("Unexpected Haukeli v2 feature schema")
    spec["monetary_schema"] = "All81 lp2_* columns are dimensionless; preserve during monetary normalization. Same economic margin normalization as v1 plus two individual Haukeli storage fractions and river flow/Qmin."
    return result, spec
