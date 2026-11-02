"""Compare the fixed equal three- and five-seed CNN means in the same 80/20 blend."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
HOURS = 168
PROFILES = ["price", "sampling", "regularized", "deep_regularized", "recency"]


def aucs(labels, predictions, generators):
    return {"micro_roc_auc": float(roc_auc_score(labels.ravel(), predictions.ravel())),
            "generator_auc": {g: float(roc_auc_score(labels[:, i].ravel(), predictions[:, i].ravel()))
                              for i, g in enumerate(generators)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/cnn_five_seeds.json"))
    parser.add_argument("--output", default=str(ROOT / "reports/cnn_five_seeds_comparison.json"))
    args = parser.parse_args()
    output = Path(args.output)
    assert not output.exists(), f"Comparison report already exists: {output}"
    config = json.loads(Path(args.config).read_text())
    assert config["seeds"] == [42, 7, 21, 67, 103] and config["years"] == [2021, 2022]
    assert config["xgboost_weight"] == .8 and config["cnn_weight"] == .2
    columns = pd.read_csv(ROOT / "data/raw/sample_submission.csv", nrows=0).columns[1:].tolist()
    generators = list(dict.fromkeys(c.removeprefix("result_committed_").rsplit("_t", 1)[0] for c in columns))
    assert len(generators) == 14
    assert columns == [f"result_committed_{g}_t{h}" for g in generators for h in range(HOURS)]
    truth = pd.read_csv(ROOT / "data/raw/kernel/Unit_commitment_decisions.csv",
                        usecols=["Run No", "starttime", *columns],
                        dtype={column: "uint8" for column in columns}).set_index("Run No")
    assert truth.index.is_unique and truth[columns].isin([0, 1]).all().all()
    truth["starttime"] = pd.to_datetime(truth["starttime"], utc=True)
    results, xgb_references = [], {}
    for year in config["years"]:
        expected = truth.loc[truth["starttime"].dt.year == year].sort_values("starttime")
        assert len(expected), f"No labelled cases for {year}"
        seeds = {seed: f"cnn_{year}" if seed == 42 else f"cnn_seed{seed}_{year}" for seed in config["seeds"]}
        xgb_names = [f"xgboost_{profile}_{year}" for profile in PROFILES]
        values = {}
        for name in [*xgb_names, *seeds.values()]:
            frame = pd.read_csv(ROOT / f"data/processed/{name}_validation_predictions.csv").set_index("Run No")
            assert frame.index.is_unique and frame.index.equals(expected.index), f"Case IDs/order differ: {name}"
            assert frame.columns.tolist() == columns, f"Prediction columns/order differ: {name}"
            array = frame.to_numpy()
            assert np.isfinite(array).all() and ((array >= 0) & (array <= 1)).all(), f"Invalid probabilities: {name}"
            values[name] = array.reshape(len(expected), len(generators), HOURS)
        labels = expected[columns].to_numpy().reshape(len(expected), len(generators), HOURS)
        ensemble = sum(values[name] for name in xgb_names) / len(xgb_names)
        xgb_references[str(year)] = {"member_weights": {name: .2 for name in xgb_names},
                                     **aucs(labels, ensemble, generators)}
        memberships = {"seed42": [42], "three_seed_mean": [42, 7, 21], "five_seed_mean": config["seeds"]}
        scores = {}
        for variant, members in memberships.items():
            cnn = sum(values[seeds[seed]] for seed in members) / len(members)
            scores[variant] = {"cnn": aucs(labels, cnn, generators),
                               "blend": aucs(labels, .8 * ensemble + .2 * cnn, generators)}
        for variant, members in memberships.items():
            row = {"year": year, "variant": variant, "seeds": members,
                   "validation_cases": len(expected),
                   "cnn_member_weights": {seeds[seed]: 1 / len(members) for seed in members},
                   "blend_member_weights": {**{name: .16 for name in xgb_names},
                                             **{seeds[seed]: .2 / len(members) for seed in members}}, **scores[variant]}
            for kind in ("cnn", "blend"):
                for reference in ("three_seed_mean", "seed42"):
                    row[f"{kind}_delta_{reference}"] = (scores[variant][kind]["micro_roc_auc"]
                                                        - scores[reference][kind]["micro_roc_auc"])
            results.append(row)
    report = {"config": config, "xgboost_references": xgb_references,
              "blend_weights": {"xgboost_ensemble": .8, "cnn": .2}, "results": results,
              "note": "All five seeds retained with equal weights. Unweighted micro AUC; historical validation is development data."}
    with output.open("x") as file:
        file.write(json.dumps(report, indent=2) + "\n")
    table = [{"variant": row["variant"], "year": row["year"], "cnn_auc": row["cnn"]["micro_roc_auc"],
              "blend_auc": row["blend"]["micro_roc_auc"], "delta_three": row["blend_delta_three_seed_mean"]}
             for row in results]
    print(pd.DataFrame(table).set_index(["variant", "year"]).unstack("year").round(6).to_string())


if __name__ == "__main__":
    main()
