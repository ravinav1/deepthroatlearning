"""Describe errors by exogenous regimes; no fitting or prediction changes."""

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


def consecutive_runs(mask):
    padded = np.r_[False, mask, False].astype(int)
    lengths = np.flatnonzero(np.diff(padded) == -1) - np.flatnonzero(np.diff(padded) == 1)
    return {"runs": int(len(lengths)), "longest_run_days": int(lengths.max()) if len(lengths) else 0}


def main():
    output = ROOT / "reports/audit_20261017_regimes.json"
    assert not output.exists()
    labeled, _, _, inputs, _ = load_data()
    mapping = pd.read_csv(ROOT / "data/raw/prediction_mapping.csv")
    mapping["starttime"] = pd.to_datetime(mapping.starttime, utc=True)
    cases = pd.concat([labeled[["Run No", "starttime"]], mapping]).sort_values("starttime")
    starts = pd.DatetimeIndex(cases.starttime)
    hours = pd.DatetimeIndex(np.repeat(starts.array, HOURS)) + pd.to_timedelta(np.tile(np.arange(HOURS), len(starts)), unit="h")
    prices = inputs["price"].loc[hours].iloc[:, 0].to_numpy().reshape(-1, HOURS)
    relative_std = prices.std(axis=1) / np.maximum(np.abs(prices).mean(axis=1), 10)
    mean_price = prices.mean(axis=1)
    negative_share = (prices < 0).mean(axis=1)
    train = starts + pd.Timedelta(hours=HOURS) <= pd.Timestamp("2021-01-01", tz="UTC")
    q50, q90 = np.quantile(relative_std[train], [.5, .9])
    mean_max = mean_price[train].max()
    # Thresholds use pre-2021 inputs only, fixed before reading any error slices.
    masks = {
        "relative_volatility_at_or_below_train_median": relative_std <= q50,
        "relative_volatility_train_median_to_p90": (relative_std > q50) & (relative_std <= q90),
        "relative_volatility_above_train_p90": relative_std > q90,
        "any_negative_price": negative_share > 0,
        "weekly_mean_above_pre2021_training_max": mean_price > mean_max,
    }
    evidence = []
    for year in (2021, 2022, 2023, 2024):
        selected = starts.year == year
        evidence.append({"year": year, "cases": int(selected.sum()), "regimes": {
            key: {"cases": int(mask[selected].sum()), **consecutive_runs(mask[selected])}
            for key, mask in masks.items()}})
    evaluations = []
    sources = {}
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
        p_clip = np.clip(p, 1e-12, 1 - 1e-12)
        bce = -(y * np.log(p_clip) + (1 - y) * np.log1p(-p_clip))
        rows = []
        for name, mask in masks.items():
            choose = mask[selected]
            outputs = np.broadcast_to(choose[:, None, None], y.shape)
            row = {"regime": name, "cases": int(choose.sum()), **consecutive_runs(choose),
                **pairs.subset(outputs),
                "allocated_wrong_pair_share": float(incident[outputs].sum() / (2 * pairs.wrong_pairs)),
                "mean_bce": float(bce[outputs].mean()) if choose.any() else None,
                "generators": []}
            for g in range(14):
                yy, pp = y[choose, g].ravel(), p[choose, g].ravel()
                row["generators"].append({"generator": metadata["columns"][g * HOURS].removeprefix("result_committed_").removesuffix("_t0"),
                    "micro_auc": float(roc_auc_score(yy, pp)) if len(np.unique(yy)) == 2 else None,
                    "mean_bce": float(bce[choose, g].mean()) if choose.any() else None,
                    "allocated_global_wrong_pair_share": float(incident[choose, g].sum() / (2 * pairs.wrong_pairs))})
            rows.append(row)
        evaluations.append({"year": year, "mode": mode, "overall_micro_auc": float(roc_auc_score(y.ravel(), p.ravel())),
            "overall_mean_bce": float(bce.mean()), "regimes": rows})
        print(json.dumps({"year": year, "mode": mode, "regimes": [{k: row[k] for k in
            ("regime", "cases", "runs", "micro_roc_auc", "mean_bce", "error_enrichment_vs_pair_availability")} for row in rows]}, indent=2), flush=True)
    for path in (Path(__file__), ROOT / "data/raw/kernel/Historical_day_ahead_price_2015_2025.csv", ROOT / "src/round2_diagnostics.py"):
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = {"created_utc": datetime.now(timezone.utc).isoformat(),
        "thresholds_from_purged_pre2021_inputs": {"training_cases": int(train.sum()), "relative_std_median": float(q50), "relative_std_p90": float(q90), "maximum_weekly_mean": float(mean_max)},
        "input_regime_coverage": evidence, "evaluations": evaluations, "source_sha256": sources,
        "fitting_performed": False, "test_labels_used": False,
        "limitations": "Descriptive, reused development predictions. Consecutive runs are not statistically independent samples; overlapping case weeks remain dependent across short breaks. Regimes overlap except the three volatility bins. Conditional subgroup AUC is not directly comparable across different label/covariate distributions. No conclusion of covariate-shift assumptions or optimal weighting follows."}
    with output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
