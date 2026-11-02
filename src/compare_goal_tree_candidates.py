"""Compare new trees with fixed weights selected for the rescaled XGBoost."""

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from features import ROOT
from plant_blend import blend, column_groups, metrics, read_predictions


INITIAL_REFERENCE_AUC = {2021: 0.9868456865125457, 2022: 0.9843832046746634}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-prefix", action="append", required=True,
                        help="Saved prediction prefix; repeat to compare candidates separately.")
    parser.add_argument("--years", nargs="+", type=int, choices=[2021, 2022], default=[2021, 2022])
    parser.add_argument("--output", default="reports/goal_tree_candidates_comparison.json")
    args = parser.parse_args()
    output = ROOT / args.output
    assert not output.exists(), f"Report already exists: {output}"
    reference_path = ROOT / "reports/goal_rescaled_plant_blend_comparison.json"
    reference = json.loads(reference_path.read_text())
    config = reference["config"]
    weights = reference["selected_plant_weights"]
    cnn_weight = reference["selected_haukeli_cnn_weight"]
    columns = pd.read_csv(ROOT / "data/raw/sample_submission.csv", nrows=0).columns[1:].tolist()
    generators = list(dict.fromkeys(column.removeprefix("result_committed_").rsplit("_t", 1)[0]
                                   for column in columns))
    groups = column_groups(columns)
    fixed = groups[config["specialist_generator"].rsplit("_G", 1)[0]]
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
        current_tree = read(config["tree_prefix"], year, columns, ids)
        transformer = read(config["transformer_prefix"], year, columns, ids)
        specialist = read(config["specialist_prefix"], year, fixed_columns, ids) if cnn_weight else None

        def combined(trees):
            fixed_values = trees[:, fixed].copy()
            if cnn_weight:
                fixed_values = (1 - cnn_weight) * fixed_values + cnn_weight * specialist
            return blend(trees, transformer, groups, weights, fixed, fixed_values)

        baseline = combined(current_tree)
        baseline_auc = float(roc_auc_score(labels.ravel(), baseline.ravel()))
        expected = next(row for row in reference["results"]
                        if row["year"] == year and row["variant"] == "selected_plant")
        assert abs(baseline_auc - expected["micro_roc_auc"]) < 1e-12
        rows.append({"year": year, "candidate_prefix": None, "variant": "baseline",
                     "validation_cases": len(labels), "delta_vs_baseline": 0.0,
                     **metrics(labels, baseline, generators)})
        for prefix in args.candidate_prefix:
            candidate = read(prefix, year, columns, ids)
            variants = {"candidate_standalone": candidate,
                        "candidate_replace_tree_component": combined(candidate),
                        "candidate_equal_tree_mean": combined(.5 * current_tree + .5 * candidate)}
            for name, values in variants.items():
                if name != "candidate_standalone":
                    assert np.array_equal(values[:, neural_only], baseline[:, neural_only])
                result = metrics(labels, values, generators)
                rows.append({"year": year, "candidate_prefix": prefix, "variant": name,
                             "validation_cases": len(labels),
                             "delta_vs_baseline": result["micro_roc_auc"] - baseline_auc, **result})

    report = {
        "reference_source": str(reference_path.relative_to(ROOT)),
        "reference_variant": "selected_plant", "reference_tree_prefix": config["tree_prefix"],
        "candidate_prefixes": args.candidate_prefix, "years": args.years,
        "plant_weights": weights, "haukeli_cnn_weight": cnn_weight,
        "initial_reference_auc": INITIAL_REFERENCE_AUC,
        "prediction_sources": list(dict.fromkeys(sources)), "test_case_ids_excluded_from_labels": True,
        "variant_definitions": {
            "baseline": "Rescaled XGBoost with its 2021-selected plant weights and Haukeli CNN fraction.",
            "candidate_standalone": "Candidate tree probabilities for all outputs; no neural blend.",
            "candidate_replace_tree_component": "Replace the rescaled XGBoost with the candidate inside fixed new plant weights.",
            "candidate_equal_tree_mean": "Use a fixed 50/50 current/candidate tree mean inside the same fixed weights."
        },
        "note": "Primary reference is the stronger rescaled-tree selected-plant blend. Initial old-ensemble AUCs are context only. No weight fitting, candidate selection, model training, or submission export. Both historical years are reused development data; competition test labels are excluded.",
        "results": rows,
    }
    with output.open("x") as file:
        file.write(json.dumps(report, indent=2) + "\n")
    summary = pd.DataFrame(rows).fillna({"candidate_prefix": "current"})
    print(summary.pivot(index=["candidate_prefix", "variant"], columns="year", values="micro_roc_auc")
          .round(6).to_string())


if __name__ == "__main__":
    main()
