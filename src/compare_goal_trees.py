"""Compare two fixed tree substitutions inside the current plant-specific blend."""

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from features import ROOT
from plant_blend import blend, column_groups, metrics, read_predictions


EXPECTED_BASELINE = {2021: 0.9868456865125457, 2022: 0.9843832046746634}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-prefix", default="goal_xgboost_rescaled")
    parser.add_argument("--years", nargs="+", type=int, choices=[2021, 2022], default=[2021, 2022])
    parser.add_argument("--output", default="reports/goal_tree_comparison.json")
    args = parser.parse_args()
    output = ROOT / args.output
    assert not output.exists(), f"Report already exists: {output}"
    reference = json.loads((ROOT / "reports/plant_blend_refined_comparison.json").read_text())
    haukeli = json.loads((ROOT / "reports/haukeli_blend_comparison.json").read_text())
    config = reference["config"]
    weights = reference["selected_plant_weights"]
    specialist_weight = haukeli["selected_specialist_weight"]
    assert specialist_weight == 0.3
    profiles = config["xgboost_profiles"]
    assert len(profiles) == 5 and profiles.count("price") == 1
    columns = pd.read_csv(ROOT / "data/raw/sample_submission.csv", nrows=0).columns[1:].tolist()
    generators = list(dict.fromkeys(column.removeprefix("result_committed_").rsplit("_t", 1)[0]
                                   for column in columns))
    groups = column_groups(columns)
    fixed = groups[config["fixed_generator"].rsplit("_G", 1)[0]]
    fixed_columns = [columns[i] for i in fixed]
    neural_only = [i for plant, weight in weights.items() if weight == 1.0 for i in groups[plant]]
    truth = pd.read_csv(ROOT / "data/raw/kernel/Unit_commitment_decisions.csv",
                        usecols=["Run No", "starttime", *columns],
                        dtype={column: "uint8" for column in columns})
    truth["starttime"] = pd.to_datetime(truth.starttime, utc=True)
    test_ids = pd.read_csv(ROOT / "data/raw/prediction_mapping.csv")["Run No"]
    assert not truth["Run No"].isin(test_ids).any()
    rows, sources = [], []

    def read(prefix, year, names, ids):
        path = ROOT / f"data/processed/{prefix}_{year}_validation_predictions.csv"
        sources.append(str(path.relative_to(ROOT)))
        return read_predictions(path, names, ids)

    for year in args.years:
        cases = truth.loc[truth.starttime.dt.year == year].sort_values("starttime").reset_index(drop=True)
        ids = cases["Run No"]
        labels = cases[columns].to_numpy()
        members = {profile: read(f"xgboost_{profile}", year, columns, ids) for profile in profiles}
        xgb = sum(members.values()) / len(profiles)
        transformer = read(config["transformer_prefix"], year, columns, ids)
        specialist = read(config["specialist_prefix"], year, fixed_columns, ids)

        def combined(trees):
            fixed_values = (1 - specialist_weight) * trees[:, fixed] + specialist_weight * specialist
            return blend(trees, transformer, groups, weights, fixed, fixed_values)

        baseline = combined(xgb)
        baseline_auc = float(roc_auc_score(labels.ravel(), baseline.ravel()))
        assert abs(baseline_auc - EXPECTED_BASELINE[year]) < 1e-12
        candidate = read(args.candidate_prefix, year, columns, ids)
        replacement = sum(candidate if profile == "price" else members[profile]
                          for profile in profiles) / len(profiles)
        variants = {"baseline": baseline, "old_price_standalone": members["price"],
                    "rescaled_standalone": candidate, "replace_price_member": combined(replacement),
                    "replace_full_tree_ensemble": combined(candidate)}
        for name, values in variants.items():
            if name.startswith("replace_"):
                assert np.array_equal(values[:, neural_only], baseline[:, neural_only])
            result = metrics(labels, values, generators)
            rows.append({"year": year, "variant": name, "validation_cases": len(labels),
                         "delta_vs_baseline": result["micro_roc_auc"] - baseline_auc, **result})

    report = {
        "candidate_prefix": args.candidate_prefix, "years": args.years,
        "plant_weights": weights, "haukeli_specialist_weight": specialist_weight,
        "xgboost_profiles": profiles, "prediction_sources": list(dict.fromkeys(sources)),
        "test_case_ids_excluded_from_labels": True,
        "variant_definitions": {
            "baseline": "Five original tree members, refined plant Transformer weights, Haukeli 70/30 specialist blend.",
            "old_price_standalone": "Original unscaled price-profile XGBoost predictions, without blending.",
            "rescaled_standalone": "New case-rescaled XGBoost predictions, without blending.",
            "replace_price_member": "Replace only the price member in the equal five-model tree ensemble.",
            "replace_full_tree_ensemble": "Replace the complete five-model tree ensemble with the new single model."
        },
        "note": "Two substitutions declared before candidate fits finished. All plant neural weights and Haukeli's 70/30 weights stay fixed; Haukeli's tree contribution changes with the substituted tree predictions. Both years are reused historical development data. No weight optimization, training, submission export, or competition test labels.",
        "results": rows,
    }
    with output.open("x") as file:
        file.write(json.dumps(report, indent=2) + "\n")
    print(pd.DataFrame(rows).pivot(index="variant", columns="year", values="micro_roc_auc")
          .round(6).to_string())


if __name__ == "__main__":
    main()
