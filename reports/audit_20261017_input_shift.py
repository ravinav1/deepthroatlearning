"""Describe supplied covariate shifts without labels, fitting, or significance tests."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import sys
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features import load_data, HOURS


def main():
    output = ROOT / "reports/audit_20261017_input_shift.json"
    assert not output.exists()
    labeled, _, _, inputs, topology = load_data()
    mapping = pd.read_csv(ROOT / "data/raw/prediction_mapping.csv")
    mapping["starttime"] = pd.to_datetime(mapping.starttime, utc=True)
    cases = pd.concat([labeled[["Run No", "starttime"]], mapping], ignore_index=True).sort_values("starttime")
    assert cases["Run No"].is_unique
    starts = pd.DatetimeIndex(cases.starttime)
    hours = pd.DatetimeIndex(np.repeat(starts.array, HOURS)) + pd.to_timedelta(np.tile(np.arange(HOURS), len(starts)), unit="h")
    price = inputs["price"].loc[hours].iloc[:, 0].to_numpy().reshape(-1, HOURS)
    scale = np.maximum(np.abs(price).mean(axis=1), 10)
    fields = {"week_price_mean": price.mean(axis=1), "week_price_min": price.min(axis=1),
        "week_price_max": price.max(axis=1), "week_price_std": price.std(axis=1),
        "week_negative_price_fraction": (price < 0).mean(axis=1), "log_week_price_scale": np.log(scale),
        "week_relative_price_std": price.std(axis=1) / scale,
        "week_relative_price_range": np.ptp(price, axis=1) / scale}
    for reservoir in inputs["volume"]:
        capacity = topology["model"]["reservoir"][reservoir]["max_vol"]
        volume = inputs["volume"].loc[starts, reservoir].to_numpy()
        water = inputs["water_value"].loc[starts + pd.Timedelta(hours=HOURS), reservoir].to_numpy()
        inflow = inputs["inflow"].loc[hours, reservoir].to_numpy().reshape(-1, HOURS).sum(axis=1) * 3600 / 1e6
        fields[f"initial_fill_{reservoir}"] = volume / capacity
        fields[f"terminal_water_value_{reservoir}"] = water
        fields[f"relative_terminal_water_value_{reservoir}"] = water / scale
        fields[f"week_natural_inflow_capacity_{reservoir}"] = inflow / capacity
    features = pd.DataFrame(fields, index=starts)
    assert np.isfinite(features.to_numpy()).all()
    comparisons = []
    for cutoff, target_year in ((2021, 2021), (2021, 2022), (2022, 2022), (2023, 2023), (2023, 2024)):
        boundary = pd.Timestamp(f"{cutoff}-01-01", tz="UTC")
        train_mask = starts + pd.Timedelta(hours=HOURS) <= boundary if cutoff < 2023 else starts.year < cutoff
        target_mask = starts.year == target_year
        train, target = features.loc[train_mask], features.loc[target_mask]
        rows = []
        for name in features:
            x, z = train[name].to_numpy(), target[name].to_numpy()
            q = np.quantile(x, [.05, .25, .5, .75, .95])
            iqr = q[3] - q[1]
            rows.append({"feature": name, "train_quantiles_05_25_50_75_95": q.tolist(),
                "target_quantiles_05_25_50_75_95": np.quantile(z, [.05, .25, .5, .75, .95]).tolist(),
                "target_below_train_min_fraction": float((z < x.min()).mean()),
                "target_above_train_max_fraction": float((z > x.max()).mean()),
                "wasserstein_over_train_iqr": float(wasserstein_distance(x, z) / iqr) if iqr > 1e-12 else None,
                "train_constant": bool(np.ptp(x) == 0), "target_constant": bool(np.ptp(z) == 0)})
        comparisons.append({"training_before_year": cutoff, "target_year": target_year,
            "training_cases": len(train), "target_cases": len(target), "features": rows})
        prices = next(row for row in rows if row["feature"] == "week_price_mean")
        print(json.dumps({"training_before_year": cutoff, "target_year": target_year,
            "training_cases": len(train), "target_cases": len(target), "weekly_price_mean": prices,
            "top_shifts": sorted((row for row in rows if row["wasserstein_over_train_iqr"] is not None),
                                 key=lambda row: row["wasserstein_over_train_iqr"], reverse=True)[:5]}, indent=2), flush=True)
    paths = [Path(__file__), ROOT / "src/features.py", ROOT / "configs/Tokke_Vinje_topology.yaml",
        ROOT / "data/raw/prediction_mapping.csv", *[ROOT / "data/raw/kernel" / n for n in
        ["Historical_day_ahead_price_2015_2025.csv", "Historical_volume_2015_2024.csv",
         "Synthetic_water_value_2015_2024.csv", "Historical_inflow_1958_2025.csv"]]]
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "comparisons": comparisons,
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "labels_used": False, "fitting_performed": False,
        "limitations": "Descriptive distributions of weekly exogenous summaries, not a full model-feature drift test. Overlapping weeks are dependent. No p-values, causal claims, density model or test-label inference. Training cohort for2023/24 is all2922 labeled cases; earlier cutoffs purge overlapping target weeks."}
    with output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
