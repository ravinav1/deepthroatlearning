"""Compact dimensionless additions from the fixed relaxed-state LP cache."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sintef_project.lp_features import Network, UNIT_ORDER

KINDS = ["state", "delta_qmin_ratio", "delta_power_fraction", "delta_economic_margin"]
COLUMNS = [f"shop_{kind}_{unit}" for unit in UNIT_ORDER for kind in KINDS]
VERSION = "shop_relaxed_state_56_v1"


def digest(path: Path) -> str:
    """Return a file hash without decoding raw target values."""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def compute_features(case: dict, result: dict, baseline: dict, network: Network) -> np.ndarray:
    """Return168×56 input-only state/change predictors in frozen unit-major order."""
    if network.units != UNIT_ORDER:
        raise ValueError("Unexpected generator order")
    for name in ["state", "flow", "power"]:
        if np.shape(result[name]) != (14, 168) or not np.isfinite(result[name]).all():
            raise ValueError(f"Invalid relaxed-state result: {name}")
    if np.min(result["state"]) < -1e-5 or np.max(result["state"]) > 1+1e-5:
        raise ValueError("Relaxed state exceeds feasibility tolerance")
    scale = 1 + np.abs(case["price"]).mean()
    columns = []
    for j, _ in enumerate(UNIT_ORDER):
        up, down = network.unit_up[j], network.unit_down[j]
        new_nu = np.asarray(result["water_shadow"])[up].copy()
        old_nu = np.asarray(baseline["water_shadow"])[up].copy()
        if down >= 0:
            new_nu -= result["water_shadow"][down]
            old_nu -= baseline["water_shadow"][down]
        if min(case["qmin"][j], case["pmax"][j], case["energy"][j, 0], scale) <= 0:
            raise ValueError("Nonpositive feature conversion factor")
        columns.extend([np.clip(result["state"][j], 0, 1),
                        (result["flow"][j] - baseline["flow"][j]) / case["qmin"][j],
                        (result["power"][j] - baseline["power"][j]) / case["pmax"][j],
                        (old_nu - new_nu) / (case["energy"][j, 0] * scale)])
    values = np.column_stack(columns).astype(np.float32)
    if values.shape != (168, 56) or not np.isfinite(values).all():
        raise ValueError("Invalid compact relaxed-state features")
    return values


def feature_specification() -> dict:
    """Describe all appended columns; no post-result feature search is allowed."""
    return {"version": VERSION, "columns": COLUMNS, "dtype": "float32", "units": UNIT_ORDER,
            "state": "clip raw linked u to[0,1] only for numerical tolerance; raw u retained",
            "delta_qmin_ratio": "(newQ-cached_baseQ)/fixed_initial_Qmin; all Q in Mm3 per hour",
            "delta_power_fraction": "(newP-cached_baseP)/fixed_initial_case_feasible_Pmax",
            "delta_economic_margin": "(cached_base_local_dual-new_local_dual)/(initial_best_energy*(1+mean(abs(caseprice)))); local_dual=lambda_up-lambda_down, EUR/Mm3",
            "baseline": "Original frozen lp-full-001, preserving its numerical behavior and the existing608 feature prefix",
            "scope": "All14 units; only case inputs and input-derived LP solutions; no targets, fitted statistics, calendar/case-ID features, weights or threshold selection"}


def add_shop_features(x: pd.DataFrame, root: Path, cache_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Verify a completed cache, then append56 columns to the unchanged608 prefix."""
    root = Path(root).resolve()
    repository = root.parent if root.name == "sintef_project" else root
    cache = Path(cache_dir)
    if not cache.is_absolute():
        cache = repository / cache
    manifest = json.loads((cache / "manifest.json").read_text())
    if manifest["status"] != "complete" or manifest["feature_specification"] != feature_specification():
        raise ValueError("Incomplete or different relaxed-state cache")
    if x.columns.tolist() != manifest["prefix_feature_columns"] or len(x.columns) != 608:
        raise ValueError("Existing608 feature prefix differs")
    for group in ["source", "data"]:
        for name, expected in manifest[group].items():
            if digest(repository / name) != expected:
                raise ValueError(f"Relaxed-state cache {group} changed: {name}")
    if digest(repository / manifest["baseline_manifest_path"]) != manifest["baseline_manifest_sha256"]:
        raise ValueError("Frozen baseline cache manifest changed")
    path = cache / "features.npz"
    if digest(path) != manifest["artifacts"]["features.npz"]:
        raise ValueError("Relaxed-state cache feature bytes changed")
    project = repository / "sintef_project"
    train = pd.read_csv(project / "data/kernel/Unit_commitment_decisions.csv", usecols=["Run No", "starttime"])
    cases = pd.concat([train, pd.read_csv(project / "prediction_mapping.csv")], ignore_index=True)
    if len(x) != len(cases)*168 or not np.array_equal(x["horizon_hour"].to_numpy(), np.tile(np.arange(168), len(cases))):
        raise ValueError("Expected complete case-major rows in the608 prefix")
    with np.load(path, allow_pickle=False) as z:
        if (z["case_ids"].tolist() != cases["Run No"].tolist() or z["columns"].tolist() != COLUMNS
                or z["units"].tolist() != UNIT_ORDER or z["features"].shape != (len(cases), 168, 56)
                or manifest["case_dates"] != pd.to_datetime(cases.starttime, utc=True).astype(str).tolist()):
            raise ValueError("Relaxed-state cache ID/date/unit/column alignment failed")
        values = z["features"].reshape(len(x), 56).copy()
    if not np.isfinite(values).all() or set(COLUMNS).intersection(x.columns):
        raise ValueError("Invalid or duplicate appended features")
    additions = pd.DataFrame(values, columns=COLUMNS, index=x.index)
    output = pd.concat([x, additions], axis=1)
    for name in x.columns:
        if output[name].dtype != x[name].dtype or not np.array_equal(output[name].to_numpy(), x[name].to_numpy()):
            raise ValueError("Appending relaxed features changed original608 values")
    return output, {"version": VERSION, "cache_directory": str(cache),
                    "manifest_sha256": digest(cache / "manifest.json"), "features_sha256": digest(path),
                    "shape": list(additions.shape), "columns": COLUMNS, "mathematics": feature_specification(),
                    "source_verified": manifest["source"], "data_verified": manifest["data"]}
