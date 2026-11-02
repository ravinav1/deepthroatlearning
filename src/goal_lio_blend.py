"""Check whether the saved Lio CNN complements the current Transformer blend."""

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from features import HOURS, ROOT
from plant_blend import blend, column_groups, metrics, read_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/goal_lio_blend.json")
    parser.add_argument("--output", default="reports/goal_lio_blend_comparison.json")
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text())
    output = ROOT / args.output
    assert not output.exists(), f"Report already exists: {output}"
    reference = json.loads((ROOT / config["plant_weight_source"]).read_text())
    haukeli = json.loads((ROOT / config["baseline_haukeli_weight_source"]).read_text())
    base_config = reference["config"]
    weights = reference["selected_plant_weights"]
    assert weights["Lio"] == 0.5
    columns = pd.read_csv(ROOT / "data/raw/sample_submission.csv", nrows=0).columns[1:].tolist()
    generators = list(dict.fromkeys(column.removeprefix("result_committed_").rsplit("_t", 1)[0]
                                   for column in columns))
    groups = column_groups(columns)
    indices = [columns.index(f"result_committed_{config['target_generator']}_t{hour}")
               for hour in range(HOURS)]
    target_columns = [columns[i] for i in indices]
    other = [i for i in range(len(columns)) if i not in indices]
    fixed_indices = groups[base_config["fixed_generator"].rsplit("_G", 1)[0]]
    fixed_columns = [columns[i] for i in fixed_indices]
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
        fixed_cnn = read_predictions(ROOT / f"data/processed/{base_config['specialist_prefix']}_{year}_validation_predictions.csv",
                                     fixed_columns, ids)
        weight = haukeli["selected_specialist_weight"]
        fixed = (1 - weight) * xgb[:, fixed_indices] + weight * fixed_cnn
        baseline = blend(xgb, transformer, groups, weights, fixed_indices, fixed)
        labels = cases[columns].to_numpy()
        expected = next(row for row in haukeli["results"]
                        if row["year"] == year and row["variant"] == "selected_haukeli_weight")
        assert abs(roc_auc_score(labels.ravel(), baseline.ravel()) - expected["micro_roc_auc"]) < 1e-12
        cnn = read_predictions(ROOT / f"data/processed/{config['specialist_prefix']}_{year}_validation_predictions.csv",
                               target_columns, ids)
        return labels, xgb[:, indices], transformer[:, indices], cnn, baseline

    def candidate(baseline, xgb, transformer, cnn, fraction):
        result = baseline.copy()
        result[:, indices] = .5 * xgb + .5 * ((1 - fraction) * transformer + fraction * cnn)
        assert np.array_equal(result[:, other], baseline[:, other])
        return result

    labels, xgb, transformer, cnn, baseline = read_year(config["selection_year"])
    selected_fraction = 0.0
    best_auc = float(roc_auc_score(labels.ravel(), baseline.ravel()))
    trials = []
    for fraction in config["candidate_cnn_fractions"]:
        values = candidate(baseline, xgb, transformer, cnn, fraction)
        score = float(roc_auc_score(labels.ravel(), values.ravel()))
        trials.append({"cnn_fraction_of_neural_component": fraction, "micro_roc_auc": score,
                       "lio_auc": float(roc_auc_score(labels[:, indices].ravel(), values[:, indices].ravel()))})
        if score > best_auc + 1e-12:
            selected_fraction, best_auc = fraction, score
    print(f"Frozen Lio CNN fraction selected on {config['selection_year']}: {selected_fraction}", flush=True)

    rows = []
    for year in (config["selection_year"], config["confirmation_year"]):
        if year != config["selection_year"]:
            labels, xgb, transformer, cnn, baseline = read_year(year)
        baseline_auc = float(roc_auc_score(labels.ravel(), baseline.ravel()))
        selected = candidate(baseline, xgb, transformer, cnn, selected_fraction)
        for name, values in (("baseline", baseline), ("selected_lio_cnn", selected)):
            result = metrics(labels, values, generators)
            rows.append({"year": year, "variant": name, "validation_cases": len(labels),
                         "delta_vs_baseline": result["micro_roc_auc"] - baseline_auc, **result})
    report = {"config": config, "selected_cnn_fraction": selected_fraction,
              "selected_lio_weights": {"xgboost": .5, "transformer": .5 * (1 - selected_fraction),
                                       "specialist_cnn": .5 * selected_fraction},
              "selection_trials": trials, "unchanged_outputs_per_case": len(other),
              "test_case_ids_excluded_from_labels": True, "results": rows}
    with output.open("x") as file:
        file.write(json.dumps(report, indent=2) + "\n")
    print(pd.DataFrame(rows).pivot(index="variant", columns="year", values="micro_roc_auc").round(6).to_string())


if __name__ == "__main__":
    main()
