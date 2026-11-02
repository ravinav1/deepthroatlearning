"""Compare predeclared combinations of three independent model changes."""

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from features import ROOT
from plant_blend import blend, column_groups, metrics, read_predictions


EXPECTED_BASELINE = {2021: 0.9868456865125457, 2022: 0.9843832046746634}
VARIANTS = {
    "baseline": (False, False, False),
    "lio_cnn_selected": (False, False, True),
    "tree_rescaled_only": (True, False, False),
    "tree_rescaled_plus_lio": (True, False, True),
    "sequence_mean_only": (False, True, False),
    "sequence_mean_plus_lio": (False, True, True),
    "tree_rescaled_plus_sequence_mean": (True, True, False),
    "all_three": (True, True, True),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", type=int, choices=[2021, 2022], default=[2021, 2022])
    parser.add_argument("--without-bilstm", action="store_true", help="Screen only tree and Lio changes.")
    parser.add_argument("--tree-recipe", choices=["full", "price_member"], default="full")
    parser.add_argument("--output", default="reports/goal_combined_comparison.json")
    args = parser.parse_args()
    output = ROOT / args.output
    assert not output.exists(), f"Report already exists: {output}"
    reference_path = ROOT / "reports/plant_blend_refined_comparison.json"
    haukeli_path = ROOT / "reports/haukeli_blend_comparison.json"
    lio_path = ROOT / "reports/goal_lio_blend_comparison.json"
    reference = json.loads(reference_path.read_text())
    haukeli = json.loads(haukeli_path.read_text())
    lio = json.loads(lio_path.read_text())
    config = reference["config"]
    weights = reference["selected_plant_weights"]
    assert haukeli["config"]["baseline_weight_source"] == str(reference_path.relative_to(ROOT))
    assert lio["config"]["plant_weight_source"] == str(reference_path.relative_to(ROOT))
    assert lio["config"]["baseline_haukeli_weight_source"] == str(haukeli_path.relative_to(ROOT))
    assert lio["config"]["selection_year"] == 2021 and lio["selected_cnn_fraction"] == .75
    assert lio["selected_lio_weights"] == {"xgboost": .5, "transformer": .125, "specialist_cnn": .375}
    assert weights["Lio"] == .5 and haukeli["selected_specialist_weight"] == .3
    columns = pd.read_csv(ROOT / "data/raw/sample_submission.csv", nrows=0).columns[1:].tolist()
    generators = list(dict.fromkeys(column.removeprefix("result_committed_").rsplit("_t", 1)[0]
                                   for column in columns))
    groups = column_groups(columns)
    fixed = groups["Haukeli"]
    lio_indices = groups["Lio"]
    other = [i for i in range(len(columns)) if i not in lio_indices]
    truth = pd.read_csv(ROOT / "data/raw/kernel/Unit_commitment_decisions.csv",
                        usecols=["Run No", "starttime", *columns],
                        dtype={column: "uint8" for column in columns})
    truth["starttime"] = pd.to_datetime(truth.starttime, utc=True)
    test_ids = pd.read_csv(ROOT / "data/raw/prediction_mapping.csv")["Run No"]
    assert not truth["Run No"].isin(test_ids).any()
    sources = []
    rows = []
    variants = {name: flags for name, flags in VARIANTS.items() if not (args.without_bilstm and flags[1])}

    def read(prefix, year, names, ids):
        path = ROOT / f"data/processed/{prefix}_{year}_validation_predictions.csv"
        sources.append(str(path.relative_to(ROOT)))
        return read_predictions(path, names, ids)

    for year in args.years:
        cases = truth.loc[truth.starttime.dt.year == year].sort_values("starttime").reset_index(drop=True)
        ids = cases["Run No"]
        labels = cases[columns].to_numpy()
        members = {profile: read(f"xgboost_{profile}", year, columns, ids)
                   for profile in config["xgboost_profiles"]}
        old_tree = sum(members.values()) / len(members)
        new_tree = read("goal_xgboost_rescaled", year, columns, ids)
        if args.tree_recipe == "price_member":
            new_tree = sum(new_tree if profile == "price" else values
                           for profile, values in members.items()) / len(members)
        transformer = read(config["transformer_prefix"], year, columns, ids)
        haukeli_cnn = read(config["specialist_prefix"], year, [columns[i] for i in fixed], ids)
        lio_cnn = read(lio["config"]["specialist_prefix"], year, [columns[i] for i in lio_indices], ids)
        neural_mean = transformer if args.without_bilstm else .5 * transformer + .5 * read("goal_bilstm_rescaled", year, columns, ids)
        baseline = blend(old_tree, transformer, groups, weights, fixed,
                         .7 * old_tree[:, fixed] + .3 * haukeli_cnn)
        baseline_auc = float(roc_auc_score(labels.ravel(), baseline.ravel()))
        assert abs(baseline_auc - EXPECTED_BASELINE[year]) < 1e-12
        for name, (replace_tree, mean_sequences, replace_lio) in variants.items():
            tree = new_tree if replace_tree else old_tree
            sequence = neural_mean if mean_sequences else transformer
            fixed_values = .7 * tree[:, fixed] + .3 * haukeli_cnn
            values = blend(tree, sequence, groups, weights, fixed, fixed_values)
            assert np.array_equal(values[:, fixed], fixed_values)
            if replace_lio:
                values[:, lio_indices] = (.5 * tree[:, lio_indices] + .125 * transformer[:, lio_indices]
                                         + .375 * lio_cnn)
            if not replace_tree:
                assert np.array_equal(values[:, fixed], baseline[:, fixed])
            if replace_lio and not replace_tree and not mean_sequences:
                assert np.array_equal(values[:, other], baseline[:, other])
            result = metrics(labels, values, generators)
            rows.append({"year": year, "variant": name, "validation_cases": len(labels),
                         "delta_vs_baseline": result["micro_roc_auc"] - baseline_auc, **result})

    report = {
        "years": args.years, "without_bilstm": args.without_bilstm, "tree_recipe": args.tree_recipe,
        "selection_sources": [str(path.relative_to(ROOT)) for path in (reference_path, haukeli_path, lio_path)],
        "plant_weights": weights, "haukeli_specialist_weight": .3,
        "lio_override_weights": lio["selected_lio_weights"],
        "variant_flags": {name: {"replace_tree": flags[0], "mean_sequences": flags[1], "replace_lio": flags[2]}
                          for name, flags in variants.items()},
        "prediction_sources": list(dict.fromkeys(sources)),
        "test_case_ids_excluded_from_labels": True,
        "note": "All combinations declared before the new tree's 2022 results. Frozen plant weights; sequence mean is 50/50. Lio override uses its 2021-selected CNN recipe and bypasses BiLSTM; Haukeli always uses 70% applicable tree and 30% specialist CNN. No optimization, automatic ranking or selection, training, or submission export. Both years are reused development data.",
        "results": rows,
    }
    with output.open("x") as file:
        file.write(json.dumps(report, indent=2) + "\n")
    print(pd.DataFrame(rows).pivot(index="variant", columns="year", values="micro_roc_auc").round(6).to_string())


if __name__ == "__main__":
    main()
