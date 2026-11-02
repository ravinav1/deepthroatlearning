"""Compare component changes against the actual current winning ensemble."""

import argparse
import json

import numpy as np
import pandas as pd

from features import ROOT
from plant_blend import column_groups, metrics, read_predictions


BASELINE_CONFIG = "configs/goal_submission_rescaled_tree_mean.json"
BASELINE_PREFIXES = {
    "xgb_rescaled": "goal_xgboost_rescaled",
    "lightgbm_rescaled": "goal_lightgbm_rescaled",
    "transformer": "transformer_rescaled",
}


def load_year(year):
    config = json.loads((ROOT / BASELINE_CONFIG).read_text())
    columns = pd.read_csv(ROOT / "data/raw/sample_submission.csv", nrows=0).columns[1:].tolist()
    truth = pd.read_csv(
        ROOT / "data/raw/kernel/Unit_commitment_decisions.csv",
        usecols=["Run No", "starttime", *columns],
        dtype={column: "uint8" for column in columns},
    )
    truth["starttime"] = pd.to_datetime(truth.starttime, utc=True)
    test_ids = pd.read_csv(ROOT / "data/raw/prediction_mapping.csv", usecols=["Run No"])["Run No"]
    assert not truth["Run No"].isin(test_ids).any()
    cases = truth.loc[truth.starttime.dt.year == year].sort_values("starttime").reset_index(drop=True)
    components = {
        name: read_predictions(
            ROOT / f"data/processed/{prefix}_{year}_validation_predictions.csv", columns, cases["Run No"]
        )
        for name, prefix in BASELINE_PREFIXES.items()
    }
    return config, columns, cases, components


def combine(components, columns, weights):
    groups = column_groups(columns)
    result = np.zeros_like(next(iter(components.values())), dtype=np.float64)
    assert weights.keys() == groups.keys()
    for plant, amounts in weights.items():
        assert abs(sum(amounts.values()) - 1) < 1e-12
        for name, weight in amounts.items():
            result[:, groups[plant]] += weight * components[name][:, groups[plant]]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replace", action="append", required=True, metavar="COMPONENT=PREFIX")
    parser.add_argument("--years", nargs="+", type=int, default=[2021, 2022], choices=[2021, 2022])
    parser.add_argument("--output", required=True)
    parser.add_argument("--save-prefix", help="Optionally save both candidate variants as wide prediction CSVs.")
    args = parser.parse_args()
    replacements = dict(item.split("=", 1) for item in args.replace)
    assert set(replacements).issubset(BASELINE_PREFIXES)
    assert len(replacements) == len(args.replace), "Replace each component once."
    output = ROOT / args.output
    assert not output.exists(), f"Report already exists: {output}"
    rows = []
    for year in args.years:
        config, columns, cases, original = load_year(year)
        generators = list(dict.fromkeys(c.removeprefix("result_committed_").rsplit("_t", 1)[0]
                                        for c in columns))
        labels = cases[columns].to_numpy()
        changed, averaged = original.copy(), original.copy()
        for component, prefix in replacements.items():
            new = read_predictions(ROOT / f"data/processed/{prefix}_{year}_validation_predictions.csv",
                                   columns, cases["Run No"])
            changed[component] = new
            averaged[component] = .5 * original[component] + .5 * new
            rows.append({"year": year, "variant": f"standalone_{component}",
                         **metrics(labels, new, generators)})
        variants = {
            "baseline": combine(original, columns, config["plant_weights"]),
            "candidate_replace": combine(changed, columns, config["plant_weights"]),
            "candidate_mean": combine(averaged, columns, config["plant_weights"]),
        }
        baseline = metrics(labels, variants["baseline"], generators)
        reference = json.loads((ROOT / config["historical_report"]).read_text())
        expected = next(r["micro_roc_auc"] for r in reference["results"]
                        if r["year"] == year and r["variant"] == config["historical_variant"])
        assert abs(baseline["micro_roc_auc"] - expected) < 1e-12
        for variant, probabilities in variants.items():
            result = metrics(labels, probabilities, generators)
            row = {"year": year, "variant": variant, "validation_cases": len(cases),
                   "delta_vs_baseline": result["micro_roc_auc"] - baseline["micro_roc_auc"], **result}
            if year == 2022:
                # The last2021 selection case covers Dec31 through Jan6.
                after_selection = cases.starttime >= pd.Timestamp("2022-01-07", tz="UTC")
                purged = metrics(labels[after_selection], probabilities[after_selection], generators)
                purged_baseline = metrics(labels[after_selection], variants["baseline"][after_selection], generators)
                row["after_2021_selection_horizon"] = {
                    "first_case_start": "2022-01-07", "cases": int(after_selection.sum()),
                    "excluded_overlapping_cases": int((~after_selection).sum()),
                    "delta_vs_baseline": purged["micro_roc_auc"] - purged_baseline["micro_roc_auc"],
                    **purged,
                }
            rows.append(row)
            if args.save_prefix and variant != "baseline":
                frame = pd.DataFrame(probabilities, columns=columns)
                frame.insert(0, "Run No", cases["Run No"])
                frame.to_csv(ROOT / f"data/processed/{args.save_prefix}_{variant}_{year}_validation_predictions.csv",
                             index=False, mode="x")
    report = {
        "baseline_config": BASELINE_CONFIG, "baseline_components": BASELINE_PREFIXES,
        "replacements": replacements, "plant_weights": config["plant_weights"],
        "variant_definitions": {
            "candidate_replace": "Replace only the named component(s), preserving every plant weight.",
            "candidate_mean": "Replace each named component by its fixed 50/50 old/new prediction mean.",
        },
        "validation_note": "Reused historical development years. Base-model early stopping may use evaluation-year labels. No competition labels, no blend-weight fitting. Additional2022summary excludes first6case starts whose horizons overlap2021selection cases.",
        "results": rows,
    }
    with output.open("x") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print(pd.DataFrame(rows).pivot(index="variant", columns="year", values="micro_roc_auc").round(8))


if __name__ == "__main__":
    main()
