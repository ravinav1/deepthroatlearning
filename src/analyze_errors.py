"""Inspect the 75/25 ensemble on saved chronological validation predictions."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from features import HOURS, ROOT


def metrics(y, p):
    y, p = y.ravel(), p.ravel()
    errors = (p >= 0.5) != y
    clipped = np.clip(p, 1e-7, 1 - 1e-7)
    return {
        "outputs": len(y), "on_fraction": float(y.mean()),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "error_at_0_5": float(errors.mean()),
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(clipped) + (1-y) * np.log1p(-clipped))),
    }


def main():
    columns = pd.read_csv(ROOT / "data/raw/sample_submission.csv", nrows=0).columns[1:].tolist()
    generators = list(dict.fromkeys(c.removeprefix("result_committed_").rsplit("_t", 1)[0] for c in columns))
    labels = pd.read_csv(ROOT / "data/raw/kernel/Unit_commitment_decisions.csv",
                         usecols=["Run No", "starttime", *columns]).set_index("Run No")
    test_ids = pd.read_csv(ROOT / "data/raw/prediction_mapping.csv", usecols=["Run No"])["Run No"]
    rows, hourly, examples, summary = [], [], [], {}
    for year in [2021, 2022]:
        xgb = pd.read_csv(ROOT / f"data/processed/xgboost_{year}_validation_predictions.csv").set_index("Run No")
        lgb = pd.read_csv(ROOT / f"data/processed/topology_{year}_validation_predictions.csv").set_index("Run No")
        assert xgb.index.equals(lgb.index) and xgb.columns.tolist() == lgb.columns.tolist() == columns
        assert not xgb.index.isin(test_ids).any()
        truth = labels.loc[xgb.index]
        starts = pd.to_datetime(truth.starttime, utc=True)
        assert (starts.dt.year == year).all()
        y = truth[columns].to_numpy().reshape(-1, len(generators), HOURS)
        p = (0.75 * xgb[columns] + 0.25 * lgb[columns]).to_numpy().reshape(y.shape)
        errors = (p >= 0.5) != y
        # A switch at t means y[t] differs from y[t-1]. No assumption at t=0.
        switches = np.zeros(y.shape, dtype=bool)
        switches[:, :, 1:] = y[:, :, 1:] != y[:, :, :-1]
        near = np.zeros(y.shape, dtype=bool)
        for delta in range(-2, 3):
            if delta < 0:
                near[:, :, :delta] |= switches[:, :, -delta:]
            elif delta > 0:
                near[:, :, delta:] |= switches[:, :, :-delta]
            else:
                near |= switches
        summary[str(year)] = {
            **metrics(y, p), "cases": len(y), "switches": int(switches.sum()),
            "near_switch_output_fraction": float(near.mean()),
            "near_switch_share_of_errors_at_0_5": float(errors[near].sum() / errors.sum()),
        }
        for g, generator in enumerate(generators):
            rows.append({"year": year, "group": "generator", "name": generator, **metrics(y[:, g], p[:, g])})
            rows.append({"year": year, "group": "generator_near_switch", "name": generator,
                         **metrics(y[:, g][near[:, g]], p[:, g][near[:, g]])})
        for day in range(7):
            hours = slice(day * 24, (day + 1) * 24)
            rows.append({"year": year, "group": "day", "name": str(day + 1), **metrics(y[:, :, hours], p[:, :, hours])})
        for name, mask in [("near_switch", near), ("away_from_switch", ~near)]:
            rows.append({"year": year, "group": "switch", "name": name, **metrics(y[mask], p[mask])})
        for hour in range(HOURS):
            hourly.append({"year": year, "hour": hour, **metrics(y[:, :, hour], p[:, :, hour])})
        # Most confident mistakes at the diagnostic threshold; inspect, don't train on them.
        wrong_confidence = np.where(errors, np.abs(p-y), -1)
        worst = np.argsort(wrong_confidence.ravel())[-10:][::-1]
        for flat in worst:
            case, g, hour = np.unravel_index(flat, y.shape)
            examples.append({"year": year, "Run No": int(xgb.index[case]), "generator": generators[g],
                             "horizon_hour": int(hour), "timestamp": str(starts.iloc[case] + pd.Timedelta(hours=int(hour))),
                             "label": int(y[case,g,hour]), "probability": float(p[case,g,hour]),
                             "near_switch": bool(near[case,g,hour])})
        print(year, json.dumps(summary[str(year)]), flush=True)

    groups = pd.DataFrame(rows)
    hours = pd.DataFrame(hourly)
    groups.to_csv(ROOT / "reports/error_analysis_groups.csv", index=False)
    hours.to_csv(ROOT / "reports/error_analysis_hours.csv", index=False)
    pd.DataFrame(examples).to_csv(ROOT / "reports/error_analysis_examples.csv", index=False)
    (ROOT / "reports/error_analysis.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    generator_rows = groups.loc[groups.group == "generator"]
    order = generator_rows.groupby("name").auc.mean().sort_values().index
    for year in [2021, 2022]:
        data = generator_rows.loc[generator_rows.year == year].set_index("name").loc[order]
        axes[0].plot(data.auc, range(len(order)), "o-", label=str(year))
        day = groups.loc[(groups.year == year) & (groups.group == "day")]
        axes[1].plot(day.name.astype(int), day.auc, "o-", label=str(year))
        switch = groups.loc[(groups.year == year) & (groups.group == "switch")].set_index("name")
        axes[2].plot([0, 1], switch.loc[["away_from_switch", "near_switch"], "error_at_0_5"], "o-", label=str(year))
    axes[0].set_yticks(range(len(order)), order)
    axes[0].set_xlabel("ROC-AUC within each generator")
    axes[1].set_xlabel("Day within scheduling week")
    axes[1].set_ylabel("Micro ROC-AUC within day")
    axes[2].set_xticks([0, 1], ["Away from switches", "Within ±2 hours"])
    axes[2].set_ylabel("Error fraction at probability threshold 0.5")
    for ax in axes:
        ax.grid(alpha=0.3); ax.legend()
    fig.suptitle("75% XGBoost + 25% LightGBM: chronological validation")
    fig.tight_layout()
    fig.savefig(ROOT / "reports/figures/error_analysis.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
