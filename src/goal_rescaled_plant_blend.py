"""Select coarse plant weights for the stronger rescaled tree on 2021 only."""

import argparse
import json

import pandas as pd
from sklearn.metrics import roc_auc_score

from features import ROOT
from plant_blend import blend, column_groups, metrics, read_predictions


EXPECTED_INITIAL = {2021: 0.9868456865125457, 2022: 0.9843832046746634}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/goal_rescaled_plant_blend.json")
    parser.add_argument("--output", default="reports/goal_rescaled_plant_blend_comparison.json")
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text())
    output = ROOT / args.output
    assert not output.exists(), f"Report already exists: {output}"
    reference = json.loads((ROOT / config["initial_plant_weight_source"]).read_text())
    haukeli = json.loads((ROOT / config["initial_haukeli_weight_source"]).read_text())
    old_weights = reference["selected_plant_weights"]
    initial_specialist_weight = config["initial_haukeli_cnn_weight"]
    assert initial_specialist_weight == haukeli["selected_specialist_weight"] == 0.3
    columns = pd.read_csv(ROOT / "data/raw/sample_submission.csv", nrows=0).columns[1:].tolist()
    generators = list(dict.fromkeys(column.removeprefix("result_committed_").rsplit("_t", 1)[0]
                                   for column in columns))
    groups = column_groups(columns)
    fixed_plant = config["specialist_generator"].rsplit("_G", 1)[0]
    fixed = groups[fixed_plant]
    fixed_columns = [columns[i] for i in fixed]
    plants = [plant for plant in groups if plant != fixed_plant]
    plant_order = [*plants, fixed_plant]
    assert set(plants) == set(old_weights)
    truth = pd.read_csv(ROOT / "data/raw/kernel/Unit_commitment_decisions.csv",
                        usecols=["Run No", "starttime", *columns],
                        dtype={column: "uint8" for column in columns})
    truth["starttime"] = pd.to_datetime(truth.starttime, utc=True)
    test_ids = pd.read_csv(ROOT / "data/raw/prediction_mapping.csv")["Run No"]
    assert not truth["Run No"].isin(test_ids).any()
    sources = []

    def read(prefix, year, names, ids):
        path = ROOT / f"data/processed/{prefix}_{year}_validation_predictions.csv"
        sources.append(str(path.relative_to(ROOT)))
        return read_predictions(path, names, ids)

    def read_year(year):
        cases = truth.loc[truth.starttime.dt.year == year].sort_values("starttime").reset_index(drop=True)
        ids = cases["Run No"]
        old_tree = sum(read(f"xgboost_{profile}", year, columns, ids)
                       for profile in config["old_xgboost_profiles"]) / len(config["old_xgboost_profiles"])
        tree = read(config["tree_prefix"], year, columns, ids)
        transformer = read(config["transformer_prefix"], year, columns, ids)
        specialist = read(config["specialist_prefix"], year, fixed_columns, ids)
        initial = blend(old_tree, transformer, groups, old_weights, fixed,
                        (1 - initial_specialist_weight) * old_tree[:, fixed] + initial_specialist_weight * specialist)
        labels = cases[columns].to_numpy()
        initial_auc = float(roc_auc_score(labels.ravel(), initial.ravel()))
        assert abs(initial_auc - EXPECTED_INITIAL[year]) < 1e-12
        return labels, tree, transformer, specialist, initial, initial_auc

    labels, tree, transformer, specialist, initial, initial_auc = read_year(config["selection_year"])

    def probabilities(weights, cnn_weight):
        fixed_values = (1 - cnn_weight) * tree[:, fixed] + cnn_weight * specialist
        return blend(tree, transformer, groups, weights, fixed, fixed_values)

    def auc(weights, cnn_weight):
        return float(roc_auc_score(labels.ravel(), probabilities(weights, cnn_weight).ravel()))

    uniform_weights = dict.fromkeys(plants, 0.0)
    best_auc = auc(uniform_weights, initial_specialist_weight)
    uniform_trials = []
    for weight in config["candidate_transformer_weights"]:
        candidate = dict.fromkeys(plants, weight)
        value = auc(candidate, initial_specialist_weight)
        uniform_trials.append({"transformer_weight": weight, "micro_roc_auc": value})
        if value > best_auc + 1e-12:
            best_auc, uniform_weights = value, candidate
    plant_weights = uniform_weights.copy()
    haukeli_weight = initial_specialist_weight
    plant_trials = []
    for plant in plant_order:
        selected = plant_weights.copy()
        selected_haukeli_weight = haukeli_weight
        for weight in config["candidate_transformer_weights"]:
            candidate = plant_weights.copy()
            candidate_haukeli_weight = haukeli_weight
            if plant == fixed_plant:
                candidate_haukeli_weight = weight
            else:
                candidate[plant] = weight
            value = auc(candidate, candidate_haukeli_weight)
            plant_trials.append({"plant": plant, "neural_weight": weight,
                                 "neural_component": "specialist_cnn" if plant == fixed_plant else "transformer",
                                 "micro_roc_auc": value})
            if value > best_auc + 1e-12:
                best_auc, selected = value, candidate
                selected_haukeli_weight = candidate_haukeli_weight
        plant_weights = selected
        haukeli_weight = selected_haukeli_weight
    print(f"Frozen weights selected on {config['selection_year']}: {plant_weights}; Haukeli CNN={haukeli_weight}", flush=True)

    rows = []
    for year in (config["selection_year"], config["confirmation_year"]):
        if year != config["selection_year"]:
            labels, tree, transformer, specialist, initial, initial_auc = read_year(year)
        variants = {"baseline": initial, "rescaled_tree_standalone": tree,
                    "frozen_old_plant_weights": probabilities(old_weights, initial_specialist_weight),
                    "selected_uniform": probabilities(uniform_weights, initial_specialist_weight),
                    "selected_plant": probabilities(plant_weights, haukeli_weight)}
        for name, values in variants.items():
            result = metrics(labels, values, generators)
            rows.append({"year": year, "variant": name, "validation_cases": len(labels),
                         "delta_vs_initial_reference": result["micro_roc_auc"] - initial_auc,
                    "haukeli_cnn_weight": None if name == "rescaled_tree_standalone" else
                         haukeli_weight if name == "selected_plant" else initial_specialist_weight, **result})
    report = {
        "config": config, "plant_order": plant_order,
        "initial_reference_weights": old_weights,
        "initial_haukeli_cnn_weight": initial_specialist_weight,
        "selected_uniform_weights": uniform_weights, "selected_plant_weights": plant_weights,
        "selected_haukeli_cnn_weight": haukeli_weight,
        "uniform_trials": uniform_trials, "plant_trials": plant_trials,
        "prediction_sources": list(dict.fromkeys(sources)), "test_case_ids_excluded_from_labels": True,
        "variant_definitions": {
            "baseline": "Original five-tree ensemble with refined plant weights and Haukeli 70/30.",
            "rescaled_tree_standalone": "New single tree for all generators; no specialist or Transformer.",
            "frozen_old_plant_weights": "New single tree with the previously selected plant weights.",
            "selected_uniform": "New tree with one shared non-Haukeli Transformer weight selected on 2021.",
            "selected_plant": "New tree with one coordinate pass of Transformer weights and Haukeli's CNN weight selected on 2021."
        },
        "note": config["note"], "results": rows,
    }
    with output.open("x") as file:
        file.write(json.dumps(report, indent=2) + "\n")
    print(pd.DataFrame(rows).pivot(index="variant", columns="year", values="micro_roc_auc")
          .round(6).to_string())


if __name__ == "__main__":
    main()
