"""Tune Haukeli's specialist contribution while keeping the other outputs fixed."""

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from features import HOURS, ROOT
from generator_specialist import output_targets
from plant_blend import blend, column_groups, metrics, read_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/haukeli_blend.json")
    parser.add_argument("--output", default="reports/haukeli_blend_comparison.json")
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text())
    output = ROOT / args.output
    assert not output.exists(), f"Report already exists: {output}"
    reference = json.loads((ROOT / config["baseline_weight_source"]).read_text())
    base_config = reference["config"]
    weights = reference["selected_plant_weights"]
    target = config["target_generator"]
    assert target == base_config["fixed_generator"]
    assert config["specialist_prefix"] == base_config["specialist_prefix"]
    columns = pd.read_csv(ROOT / "data/raw/sample_submission.csv", nrows=0).columns[1:].tolist()
    generators = list(dict.fromkeys(column.removeprefix("result_committed_").rsplit("_t", 1)[0]
                                   for column in columns))
    groups = column_groups(columns)
    selected_columns, _ = output_targets(columns, generators, target)
    indices = [columns.index(column) for column in selected_columns]
    other = [i for i in range(len(columns)) if i not in indices]
    truth = pd.read_csv(ROOT / "data/raw/kernel/Unit_commitment_decisions.csv",
                        usecols=["Run No", "starttime", *columns],
                        dtype={column: "uint8" for column in columns})
    truth["starttime"] = pd.to_datetime(truth.starttime, utc=True)
    test_ids = pd.read_csv(ROOT / "data/raw/prediction_mapping.csv")["Run No"]
    assert not truth["Run No"].isin(test_ids).any()

    def read_year(year):
        cases = truth.loc[truth.starttime.dt.year == year].sort_values("starttime").reset_index(drop=True)
        ids = cases["Run No"]
        xgb = sum(read_predictions(ROOT / f"data/processed/xgboost_{profile}_{year}_validation_predictions.csv",
                                  columns, ids) for profile in base_config["xgboost_profiles"]) / len(base_config["xgboost_profiles"])
        transformer = read_predictions(ROOT / f"data/processed/{base_config['transformer_prefix']}_{year}_validation_predictions.csv",
                                       columns, ids)
        specialist = read_predictions(ROOT / f"data/processed/{config['specialist_prefix']}_{year}_validation_predictions.csv",
                                      selected_columns, ids)
        weight = config["baseline_specialist_weight"]
        baseline = blend(xgb, transformer, groups, weights, indices,
                         (1 - weight) * xgb[:, indices] + weight * specialist)
        labels = cases[columns].to_numpy()
        expected = next(row for row in reference["results"]
                        if row["year"] == year and row["variant"] == "selected_plant")
        assert abs(roc_auc_score(labels.ravel(), baseline.ravel()) - expected["micro_roc_auc"]) < 1e-12
        return labels, xgb[:, indices], specialist, baseline

    def replace_target(baseline, xgb, specialist, weight):
        result = baseline.copy()
        result[:, indices] = (1 - weight) * xgb + weight * specialist
        assert np.array_equal(result[:, other], baseline[:, other])
        return result

    year = config["selection_year"]
    labels, xgb, specialist, baseline = read_year(year)
    selected_weight = config["baseline_specialist_weight"]
    best_auc = float(roc_auc_score(labels.ravel(), baseline.ravel()))
    trials = []
    for weight in config["candidate_specialist_weights"]:
        candidate = replace_target(baseline, xgb, specialist, weight)
        value = float(roc_auc_score(labels.ravel(), candidate.ravel()))
        trials.append({"specialist_weight": weight, "micro_roc_auc": value})
        if value > best_auc + 1e-12:
            selected_weight, best_auc = weight, value
    print(f"Frozen Haukeli specialist weight selected on {year}: {selected_weight}", flush=True)

    rows = []
    for year in (config["selection_year"], config["confirmation_year"]):
        # The weight is frozen before the confirmation year is evaluated.
        if year != config["selection_year"]:
            labels, xgb, specialist, baseline = read_year(year)
        candidate = replace_target(baseline, xgb, specialist, selected_weight)
        for name, values, weight in (("baseline", baseline, config["baseline_specialist_weight"]),
                                    ("selected_haukeli_weight", candidate, selected_weight)):
            result = metrics(labels, values, generators)
            baseline_auc = float(roc_auc_score(labels.ravel(), baseline.ravel()))
            rows.append({"year": year, "variant": name, "specialist_weight": weight,
                         "generator_routes": {target: "cnn_component"}, "validation_cases": len(labels),
                         "delta_vs_baseline": result["micro_roc_auc"] - baseline_auc, **result})
    report = {"config": config, "selected_specialist_weight": selected_weight,
              "selection_trials": trials, "results": rows}
    with output.open("x") as file:
        file.write(json.dumps(report, indent=2) + "\n")
    print(pd.DataFrame(rows).pivot(index="variant", columns="year", values="micro_roc_auc").round(6).to_string())


if __name__ == "__main__":
    main()
