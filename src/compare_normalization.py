"""Compare stable scaling and weekly economic inputs on the existing year splits."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from features import HOURS, ROOT


def metrics(labels, probabilities, generators):
    return {
        "micro_roc_auc": float(roc_auc_score(labels.ravel(), probabilities.ravel())),
        "generator_auc": {
            generator: float(roc_auc_score(labels[:, i].ravel(), probabilities[:, i].ravel()))
            for i, generator in enumerate(generators)
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/normalization_experiment.json")
    parser.add_argument("--output", default="reports/normalization_comparison.json")
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text())
    output = ROOT / args.output
    assert not output.exists(), f"Report already exists: {output}"
    columns = pd.read_csv(ROOT / "data/raw/sample_submission.csv", nrows=0).columns[1:].tolist()
    generators = list(dict.fromkeys(column.removeprefix("result_committed_").rsplit("_t", 1)[0]
                                   for column in columns))
    truth = pd.read_csv(ROOT / "data/raw/kernel/Unit_commitment_decisions.csv",
                        usecols=["Run No", "starttime", *columns],
                        dtype={column: "uint8" for column in columns}).set_index("Run No")
    truth["starttime"] = pd.to_datetime(truth.starttime, utc=True)
    rows = []
    for year in config["years"]:
        cases = truth.loc[truth.starttime.dt.year == year].sort_values("starttime")
        labels = cases[columns].to_numpy().reshape(len(cases), len(generators), HOURS)

        def read(stem):
            frame = pd.read_csv(ROOT / f"data/processed/{stem}_validation_predictions.csv").set_index("Run No")
            assert frame.index.equals(cases.index) and frame.columns.tolist() == columns
            values = frame.to_numpy().reshape(labels.shape)
            assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
            return values

        profiles = ["price", "sampling", "regularized", "deep_regularized", "recency"]
        xgb = sum(read(f"xgboost_{profile}_{year}") for profile in profiles) / len(profiles)
        other_seeds = config["other_seeds"]
        other_cnn = sum(read(f"cnn_seed{seed}_{year}") for seed in other_seeds)
        year_rows = []
        for variant, pattern in config["profiles"].items():
            stem = pattern.format(year=year)
            values = read(stem)
            selected = json.loads((ROOT / f"reports/{stem}_metrics.json").read_text())
            single_blend = config["xgboost_weight"] * xgb + config["cnn_weight"] * values
            hybrid_cnn = (other_cnn + values) / (len(other_seeds) + 1)
            hybrid_blend = config["xgboost_weight"] * xgb + config["cnn_weight"] * hybrid_cnn
            year_rows.append({
                "year": year, "variant": variant, "source": stem,
                "validation_cases": len(cases), "best_epoch": selected["best_epoch"],
                "channels": len(selected["feature_names"]),
                "cnn": metrics(labels, values, generators),
                "single_cnn_80_20": metrics(labels, single_blend, generators),
                "five_cnn_80_20": metrics(labels, hybrid_blend, generators),
                "five_cnn_members": {**{f"cnn_seed{seed}_{year}": .2 for seed in other_seeds}, stem: .2},
            })
        references = {row["variant"]: row for row in year_rows}
        for row in year_rows:
            for key in ["cnn", "single_cnn_80_20", "five_cnn_80_20"]:
                row[f"{key}_delta_legacy"] = row[key]["micro_roc_auc"] - references["legacy"][key]["micro_roc_auc"]
                row[f"{key}_delta_stable"] = row[key]["micro_roc_auc"] - references["stable"][key]["micro_roc_auc"]
        rows.extend(year_rows)
    report = {"config": config, "results": rows,
              "note": "Historical development validation. Five-member comparisons replace only seed42; four original seeds and XGBoost weights are fixed. Both normalization representations change the input layer size."}
    with output.open("x") as file:
        file.write(json.dumps(report, indent=2) + "\n")
    table = [{"year": row["year"], "variant": row["variant"],
              "cnn": row["cnn"]["micro_roc_auc"],
              "single_80_20": row["single_cnn_80_20"]["micro_roc_auc"],
              "five_80_20": row["five_cnn_80_20"]["micro_roc_auc"]} for row in rows]
    print(pd.DataFrame(table).set_index(["variant", "year"]).unstack("year").round(6).to_string())


if __name__ == "__main__":
    main()
