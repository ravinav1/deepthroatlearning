"""Compare component changes against the preserved 0.99053 ensemble."""

import argparse
import json

import numpy as np
import pandas as pd

from features import ROOT
from plant_blend import metrics, read_predictions
from round2_compare import combine, load_year


RECIPE = "configs/round2_submission_depth8_haukeli_weighted.json"
NEW_PREFIXES = {
    "xgb_depth8": "round2_xgb_depth8",
    "haukeli_specialist": "round2_xgb_haukeli",
    "transformer_weighted": "round2_transformer_task_weights",
}


def load_current_year(year):
    _, columns, cases, components = load_year(year)
    recipe = json.loads((ROOT / RECIPE).read_text())
    for name, prefix in NEW_PREFIXES.items():
        components[name] = read_predictions(
            ROOT / f"data/processed/{prefix}_{year}_validation_predictions.csv",
            columns, cases["Run No"],
        )
    baseline = combine(components, columns, recipe["plant_weights"])
    generators = list(dict.fromkeys(
        c.removeprefix("result_committed_").rsplit("_t", 1)[0] for c in columns
    ))
    expected = json.loads((ROOT / recipe["historical_report"]).read_text())
    expected_auc = next(row["micro_roc_auc"] for row in expected["results"]
                        if row["year"] == year and row["variant"] == recipe["historical_variant"])
    actual = metrics(cases[columns].to_numpy(), baseline, generators)["micro_roc_auc"]
    assert abs(actual - expected_auc) < 1e-12, "Current ensemble reconstruction differs."
    return recipe, columns, cases, generators, components, baseline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replace", action="append", default=[], metavar="COMPONENT=PREFIX")
    parser.add_argument("--years", nargs="+", type=int, default=[2021], choices=[2021, 2022])
    parser.add_argument("--output", required=True)
    parser.add_argument("--save-prefix")
    args = parser.parse_args()
    output = ROOT / args.output
    assert not output.exists(), f"Report already exists: {output}"
    replacements = dict(item.split("=", 1) for item in args.replace)
    assert len(replacements) == len(args.replace)
    rows = []
    for year in args.years:
        recipe, columns, cases, generators, components, baseline = load_current_year(year)
        assert set(replacements).issubset(components)
        changed, averaged = components.copy(), components.copy()
        for component, prefix in replacements.items():
            new = read_predictions(
                ROOT / f"data/processed/{prefix}_{year}_validation_predictions.csv",
                columns, cases["Run No"],
            )
            changed[component] = new
            averaged[component] = .5 * (new + components[component])
        variants = {"baseline": baseline}
        if replacements:
            variants.update(
                candidate_replace=combine(changed, columns, recipe["plant_weights"]),
                candidate_mean=combine(averaged, columns, recipe["plant_weights"]),
            )
        truth = cases[columns].to_numpy()
        baseline_metrics = metrics(truth, baseline, generators)
        after_selection = cases.starttime >= pd.Timestamp("2022-01-07", tz="UTC")
        for variant, probabilities in variants.items():
            values = metrics(truth, probabilities, generators)
            row = {"year": year, "variant": variant, "cases": len(cases), **values,
                   "delta_vs_baseline": values["micro_roc_auc"] - baseline_metrics["micro_roc_auc"]}
            if year == 2022:
                subset = metrics(truth[after_selection], probabilities[after_selection], generators)
                subset_baseline = metrics(truth[after_selection], baseline[after_selection], generators)
                row["after_2021_selection_horizon"] = {
                    "first_case_start": "2022-01-07", "cases": int(after_selection.sum()), **subset,
                    "delta_vs_baseline": subset["micro_roc_auc"] - subset_baseline["micro_roc_auc"],
                }
            rows.append(row)
            if args.save_prefix and variant != "baseline":
                frame = pd.DataFrame(probabilities, columns=columns)
                frame.insert(0, "Run No", cases["Run No"])
                path = ROOT / f"data/processed/{args.save_prefix}_{variant}_{year}_validation_predictions.csv"
                frame.to_csv(path, index=False, mode="x")
    report = {
        "baseline_config": RECIPE, "replacements": replacements,
        "variant_definitions": {
            "candidate_replace": "Replace the named contribution; retain every plant coefficient and other component.",
            "candidate_mean": "Average old/new predictions within each named contribution; retain all other coefficients.",
        },
        "validation_note": "Reused historical development years. Choose variants on 2021 before 2022 confirmation. Jan7+ removes six overlapping selection horizons, not prior data reuse.",
        "results": rows,
    }
    with output.open("x") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print(pd.DataFrame(rows).pivot(index="variant", columns="year", values="micro_roc_auc"))


if __name__ == "__main__":
    main()
