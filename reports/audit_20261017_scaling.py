"""Read-only audit of the existing price-normalization floor; no fitting."""

import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

from pathlib import Path
import sys
import json
import hashlib
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features import load_data, HOURS
from round6_evaluate import load_cache
from round2_diagnostics import PairDiagnostics


def runs(mask):
    edge = np.diff(np.r_[False, mask, False].astype(int))
    sizes = np.flatnonzero(edge == -1) - np.flatnonzero(edge == 1)
    return {"consecutive_runs": int(len(sizes)), "longest_run_days": int(sizes.max()) if len(sizes) else 0}


def describe(values):
    values = np.asarray(values)
    if not len(values):
        return None
    return {"minimum": float(values.min()), "quantiles_05_25_50_75_95": np.quantile(values, [.05, .25, .5, .75, .95]).tolist(), "maximum": float(values.max())}


def main():
    output = ROOT / "reports/audit_20261017_scaling.json"
    assert not output.exists()
    labeled, _, _, inputs, topology = load_data()
    mapping = pd.read_csv(ROOT / "data/raw/prediction_mapping.csv")
    mapping["starttime"] = pd.to_datetime(mapping.starttime, utc=True)
    cases = pd.concat([labeled[["Run No", "starttime"]], mapping], ignore_index=True).sort_values("starttime")
    assert cases["Run No"].is_unique
    starts = pd.DatetimeIndex(cases.starttime)
    hours = pd.DatetimeIndex(np.repeat(starts.array, HOURS)) + pd.to_timedelta(np.tile(np.arange(HOURS), len(starts)), unit="h")
    prices = inputs["price"].loc[hours].iloc[:, 0].to_numpy().reshape(-1, HOURS)
    absolute_mean = np.abs(prices).mean(axis=1)
    scale = np.maximum(absolute_mean, 10.)
    floor_active = absolute_mean <= 10.
    assert np.isfinite(absolute_mean).all() and absolute_mean.min() > 0
    water = inputs["water_value"].loc[starts + pd.Timedelta(hours=HOURS), "Kjelavatn"].to_numpy()
    pressure = np.zeros(len(cases))
    for reservoir in ("Staavatn", "Kjelavatn"):
        capacity = topology["model"]["reservoir"][reservoir]["max_vol"]
        initial = inputs["volume"].loc[starts, reservoir].to_numpy()
        weekly_inflow = inputs["inflow"].loc[hours, reservoir].to_numpy().reshape(-1, HOURS).sum(axis=1) * 3600 / 1e6
        pressure += initial + weekly_inflow - capacity
    rare = (pressure > 0) & (prices.max(axis=1) < water)
    coverage, rare_rows = [], []
    for year in range(2015, 2025):
        selected = starts.year == year
        coverage.append({"year": year, "cases": int(selected.sum()),
            "floor_active_cases": int(floor_active[selected].sum()), "floor_active_fraction": float(floor_active[selected].mean()),
            **runs(floor_active[selected]), "mean_absolute_week_price": describe(absolute_mean[selected]),
            "unfloored_to_current_water_ratio_multiplier": describe((scale / absolute_mean)[selected]),
            "rare_joint_condition_cases": int(rare[selected].sum()), "rare_joint_condition_floor_active_cases": int((rare & floor_active & selected).sum()),
            "rare_joint_condition_runs": runs(rare[selected]),
            "floor_active_case_ids": cases.loc[selected & floor_active, "Run No"].astype(int).tolist()})
    for i in np.flatnonzero(rare):
        rare_rows.append({"case_id": int(cases.iloc[i]["Run No"]), "starttime": str(starts[i]),
            "mean_absolute_week_price": float(absolute_mean[i]), "actual_scale": float(scale[i]), "floor_active": bool(floor_active[i]),
            "terminal_water_value": float(water[i]), "normalized_water_value_current": float(water[i] / scale[i]),
            "normalized_water_value_without_floor": float(water[i] / absolute_mean[i]),
            "normalized_mean_margin_current": float((prices[i].mean() - water[i]) / scale[i]),
            "normalized_mean_margin_without_floor": float((prices[i].mean() - water[i]) / absolute_mean[i]),
            "gross_pressure_million_m3": float(pressure[i])})
    evaluations, sources = [], {}
    masks = {"price_floor_active": floor_active, "price_floor_inactive": ~floor_active}
    for year, mode in ((2021, "native"), (2022, "native"), (2022, "frozen")):
        path = ROOT / f"data/processed/round6_{mode}_{year}_baseline.npz"
        arrays, metadata = load_cache(path, year, mode)
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        selected = starts.year == year
        assert np.array_equal(cases.loc[selected, "Run No"], arrays["ids"])
        y = arrays["labels"].reshape(-1, 14, HOURS)
        p = (.5 * arrays["round3"] + .25 * arrays["old299"] + .25 * arrays["old673"]).reshape(y.shape)
        pairs = PairDiagnostics(y, p)
        incident = pairs.incident.reshape(y.shape)
        clipped = np.clip(p, 1e-12, 1 - 1e-12)
        bce = -(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))
        groups = []
        for name, mask in masks.items():
            choose = mask[selected]
            outputs = np.broadcast_to(choose[:, None, None], y.shape)
            row = {"regime": name, "cases": int(choose.sum()), **runs(choose),
                **pairs.subset(outputs), "mean_bce": float(bce[outputs].mean()) if choose.any() else None,
                "allocated_wrong_pair_share": float(incident[outputs].sum() / (2 * pairs.wrong_pairs)), "generators": []}
            for g in range(14):
                yy, pp = y[choose, g].ravel(), p[choose, g].ravel()
                row["generators"].append({"generator": metadata["columns"][g * HOURS].removeprefix("result_committed_").removesuffix("_t0"),
                    "micro_auc": float(roc_auc_score(yy, pp)) if len(np.unique(yy)) == 2 else None,
                    "mean_bce": float(bce[choose, g].mean()) if choose.any() else None,
                    "allocated_global_wrong_pair_share": float(incident[choose, g].sum() / (2 * pairs.wrong_pairs))})
            groups.append(row)
        evaluations.append({"year": year, "mode": mode, "overall_micro_auc": float(roc_auc_score(y.ravel(), p.ravel())),
            "overall_mean_bce": float(bce.mean()), "regimes": groups})
        print(json.dumps({"year": year, "mode": mode, "floor_active": {k: groups[0][k] for k in ("cases", "consecutive_runs", "micro_roc_auc", "mean_bce", "error_enrichment_vs_pair_availability")}}), flush=True)
    paths = [Path(__file__), ROOT / "src/tree_preprocessing.py", ROOT / "src/features.py", ROOT / "src/round2_diagnostics.py",
        ROOT / "configs/Tokke_Vinje_topology.yaml", ROOT / "data/raw/prediction_mapping.csv",
        *[ROOT / "data/raw/kernel" / name for name in ("Historical_day_ahead_price_2015_2025.csv", "Historical_volume_2015_2024.csv", "Synthetic_water_value_2015_2024.csv", "Historical_inflow_1958_2025.csv")]]
    sources.update({str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "fitting_performed": False, "test_labels_used": False,
        "threshold": "mean(abs(168 hourly prices)) <= 10 EUR/MWh, fixed by current implementation, not searched against labels",
        "input_coverage": coverage, "evaluations": evaluations, "rare_joint_condition_examples": rare_rows,
        "rare_joint_condition_definition": "Staavatn+Kjelavatn initial storage plus weekly natural inflow exceeds their combined capacity AND maximum weekly price is below Kjelavatn terminal water value. This previously inspected retrospective condition is descriptive, not a physical necessity or a newly independent test.",
        "source_sha256": sources,
        "interpretation_limits": "No counterfactual feature replacement, fitting, floor tuning or predictions changed. Conditional AUC can vary with case/label composition. Incident pair enrichment adjusts available pairs; endpoint allocations are bookkeeping, not causal blame. Consecutive daily-start weeks remain dependent. Current scaling retains absolute scale and economic information; alternative ratios change representation, not measured information. Near-zero absolute prices make unfloored ratios numerically large. Absolute startup costs can break economic scale invariance. Test input coverage alone cannot identify label error. The rare condition was discovered from historical errors and must not be treated as an untouched validation target."}
    with output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"coverage": [{k: r[k] for k in ("year", "cases", "floor_active_cases", "consecutive_runs", "rare_joint_condition_cases", "rare_joint_condition_floor_active_cases")} for r in coverage]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
