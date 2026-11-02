"""Compare fixed sequence-model blends against the current historical baseline."""

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
    parser.add_argument("--candidate-prefix", action="append", required=True,
                        help="Saved prediction prefix; repeat to compare several candidates separately.")
    parser.add_argument("--years", nargs="+", type=int, choices=[2021, 2022], default=[2021, 2022])
    parser.add_argument("--reference", choices=["initial", "rescaled"], default="initial")
    parser.add_argument("--output", default="reports/goal_sequence_comparison.json")
    args = parser.parse_args()
    output = ROOT / args.output
    assert not output.exists(), f"Report already exists: {output}"
    reference = json.loads((ROOT / "reports/plant_blend_refined_comparison.json").read_text())
    haukeli = json.loads((ROOT / "reports/haukeli_blend_comparison.json").read_text())
    config = reference["config"]
    weights = reference["selected_plant_weights"]
    expected = EXPECTED_BASELINE
    specialist_weight = haukeli["selected_specialist_weight"]
    if args.reference == "rescaled":
        newer = json.loads((ROOT / "reports/goal_rescaled_plant_blend_comparison.json").read_text())
        weights = newer["selected_plant_weights"]
        specialist_weight = newer["selected_haukeli_cnn_weight"]
        expected = {row["year"]: row["micro_roc_auc"] for row in newer["results"]
                    if row["variant"] == "selected_plant"}
    columns = pd.read_csv(ROOT / "data/raw/sample_submission.csv", nrows=0).columns[1:].tolist()
    generators = list(dict.fromkeys(column.removeprefix("result_committed_").rsplit("_t", 1)[0]
                                   for column in columns))
    groups = column_groups(columns)
    fixed = groups[config["fixed_generator"].rsplit("_G", 1)[0]]
    fixed_columns = [columns[i] for i in fixed]
    lio = groups["Lio"]
    other = [i for i in range(len(columns)) if i not in lio]
    lio_weight = weights["Lio"]
    truth = pd.read_csv(ROOT / "data/raw/kernel/Unit_commitment_decisions.csv",
                        usecols=["Run No", "starttime", *columns],
                        dtype={column: "uint8" for column in columns})
    truth["starttime"] = pd.to_datetime(truth.starttime, utc=True)
    test_ids = pd.read_csv(ROOT / "data/raw/prediction_mapping.csv")["Run No"]
    assert not truth["Run No"].isin(test_ids).any()
    rows = []
    sources = []

    def read(prefix, year, names, ids):
        path = ROOT / f"data/processed/{prefix}_{year}_validation_predictions.csv"
        sources.append(str(path.relative_to(ROOT)))
        return read_predictions(path, names, ids)

    for year in args.years:
        cases = truth.loc[truth.starttime.dt.year == year].sort_values("starttime").reset_index(drop=True)
        ids = cases["Run No"]
        labels = cases[columns].to_numpy()
        if args.reference == "rescaled":
            xgb = read("goal_xgboost_rescaled", year, columns, ids)
        else:
            xgb = sum(read(f"xgboost_{profile}", year, columns, ids)
                      for profile in config["xgboost_profiles"]) / len(config["xgboost_profiles"])
        transformer = read(config["transformer_prefix"], year, columns, ids)
        specialist = read(config["specialist_prefix"], year, fixed_columns, ids)
        weight = specialist_weight
        fixed_values = (1 - weight) * xgb[:, fixed] + weight * specialist
        baseline = blend(xgb, transformer, groups, weights, fixed, fixed_values)
        baseline_auc = float(roc_auc_score(labels.ravel(), baseline.ravel()))
        assert abs(baseline_auc - expected[year]) < 1e-12
        rows.append({"year": year, "candidate_prefix": None, "variant": "baseline",
                     "validation_cases": len(labels), "delta_vs_baseline": 0.0,
                     **metrics(labels, baseline, generators)})

        for prefix in args.candidate_prefix:
            candidate = read(prefix, year, columns, ids)
            full = blend(xgb, candidate, groups, weights, fixed, fixed_values)
            neural_mean = .5 * transformer + .5 * candidate
            full_mean = blend(xgb, neural_mean, groups, weights, fixed, fixed_values)
            lio_only = baseline.copy()
            lio_only[:, lio] = (1 - lio_weight) * xgb[:, lio] + lio_weight * candidate[:, lio]
            lio_mean = baseline.copy()
            lio_mean[:, lio] = (1 - lio_weight) * xgb[:, lio] + lio_weight * neural_mean[:, lio]
            variants = {"standalone": candidate, "full_neural_replacement": full,
                        "full_neural_mean": full_mean, "lio_neural_replacement": lio_only,
                        "lio_neural_mean": lio_mean}
            for name, values in variants.items():
                if name != "standalone":
                    assert np.array_equal(values[:, fixed], baseline[:, fixed])
                if name.startswith("lio_"):
                    assert np.array_equal(values[:, other], baseline[:, other])
                result = metrics(labels, values, generators)
                rows.append({"year": year, "candidate_prefix": prefix, "variant": name,
                             "validation_cases": len(labels),
                             "delta_vs_baseline": result["micro_roc_auc"] - baseline_auc, **result})

    report = {
        "candidate_prefixes": args.candidate_prefix, "years": args.years, "reference": args.reference,
        "plant_weights": weights, "haukeli_specialist_weight": weight,
        "prediction_sources": list(dict.fromkeys(sources)),
        "test_case_ids_excluded_from_labels": True,
        "variant_definitions": {
            "standalone": "Candidate probabilities for all 14 generators; no ensemble.",
            "full_neural_replacement": "Replace Transformer with candidate; keep plant tree weights and Haukeli fixed.",
            "full_neural_mean": "Fixed 50/50 Transformer/candidate mean inside each plant's neural component; Haukeli fixed.",
            "lio_neural_replacement": "Lio only: replace its neural component, retaining the reference tree/neural shares; other 13 generators fixed.",
            "lio_neural_mean": "Lio only: fixed 50/50 mean inside its neural component; reference tree share and other 13 generators fixed."
        },
        "note": "Fixed variants declared before new model fits finished. No weight optimization, grid, or automatic candidate selection. Both historical years are reused development data. No model training or submission export.",
        "results": rows,
    }
    with output.open("x") as file:
        file.write(json.dumps(report, indent=2) + "\n")
    summary = pd.DataFrame(rows).fillna({"candidate_prefix": "current"})
    print(summary.pivot(index=["candidate_prefix", "variant"], columns="year", values="micro_roc_auc")
          .round(6).to_string())


if __name__ == "__main__":
    main()
