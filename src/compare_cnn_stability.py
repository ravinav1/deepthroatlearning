"""Compare fixed CNN seed and weight averages in the unchanged 80/20 XGB blend."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from features import HOURS, ROOT


def aucs(labels, predictions, generators):
    return {
        "micro_roc_auc": float(roc_auc_score(labels.ravel(), predictions.ravel())),
        "generator_auc": {
            generator: float(roc_auc_score(labels[:, i].ravel(), predictions[:, i].ravel()))
            for i, generator in enumerate(generators)
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/cnn_stability.json"))
    parser.add_argument("--output", default=str(ROOT / "reports/cnn_stability_comparison.json"))
    args = parser.parse_args()
    output = Path(args.output)
    assert not output.exists(), f"Comparison report already exists: {output}"
    config = json.loads(Path(args.config).read_text())
    assert config["seeds"] == [42, 7, 21] and config["years"] == [2021, 2022]
    assert config["averaging_seed"] == 42
    assert config["xgboost_weight"] == .8 and config["cnn_weight"] == .2
    profiles = ["price", "sampling", "regularized", "deep_regularized", "recency"]
    columns = pd.read_csv(ROOT / "data/raw/sample_submission.csv", nrows=0).columns[1:].tolist()
    generators = list(dict.fromkeys(column.removeprefix("result_committed_").rsplit("_t", 1)[0]
                                   for column in columns))
    assert len(generators) == 14
    assert columns == [f"result_committed_{g}_t{h}" for g in generators for h in range(HOURS)]
    truth = pd.read_csv(ROOT / "data/raw/kernel/Unit_commitment_decisions.csv",
                        usecols=["Run No", "starttime", *columns],
                        dtype={column: "uint8" for column in columns}).set_index("Run No")
    assert truth.index.is_unique
    truth["starttime"] = pd.to_datetime(truth["starttime"], utc=True)
    assert truth[columns].isin([0, 1]).all().all(), "Truth must contain binary labels."
    rows, controls, xgb_sources = [], {}, {}
    for year in config["years"]:
        expected = truth.loc[truth["starttime"].dt.year == year].sort_values("starttime")
        assert len(expected), f"No labelled cases for {year}"
        seeds = {seed: f"cnn_{year}" if seed == 42 else f"cnn_seed{seed}_{year}"
                 for seed in config["seeds"]}
        raw, ema, swa = (f"cnn_averaging_{year}{suffix}" for suffix in ("", "_ema", "_swa"))
        xgb_names = [f"xgboost_{profile}_{year}" for profile in profiles]
        values = {}
        for name in [*xgb_names, *seeds.values(), raw, ema, swa]:
            frame = pd.read_csv(ROOT / f"data/processed/{name}_validation_predictions.csv").set_index("Run No")
            assert frame.index.is_unique and frame.index.equals(expected.index), f"Case IDs/order differ: {name}"
            assert frame.columns.tolist() == columns, f"Prediction columns/order differ: {name}"
            array = frame.to_numpy()
            assert np.isfinite(array).all(), f"Nonfinite probabilities: {name}"
            assert ((array >= 0) & (array <= 1)).all(), f"Invalid probabilities: {name}"
            values[name] = array.reshape(len(expected), len(generators), HOURS)
        difference = np.abs(values[raw] - values[seeds[42]])
        controls[str(year)] = {
            "baseline": seeds[42], "averaging_raw": raw,
            "exact_equal": bool(np.array_equal(values[raw], values[seeds[42]])),
            "different_outputs": int(np.count_nonzero(difference)),
            "maximum_absolute_difference": float(difference.max()),
        }
        assert controls[str(year)]["exact_equal"], f"Raw averaging control differs: {controls[str(year)]}"
        labels = expected[columns].to_numpy().reshape(len(expected), len(generators), HOURS)
        ensemble = sum(values[name] for name in xgb_names) / len(xgb_names)
        xgb_sources[str(year)] = {name: .2 for name in xgb_names}
        variants = {f"seed{seed}": {name: 1.0} for seed, name in seeds.items()}
        variants.update({"seed_mean": {name: 1 / len(seeds) for name in seeds.values()},
                         "ema_seed42": {ema: 1.0}, "swa_seed42": {swa: 1.0}})
        baseline_cnn = aucs(labels, values[seeds[42]], generators)["micro_roc_auc"]
        baseline_blend = aucs(labels, .8 * ensemble + .2 * values[seeds[42]], generators)["micro_roc_auc"]
        for variant, members in variants.items():
            cnn = sum(values[name] * weight for name, weight in members.items())
            cnn_metrics = aucs(labels, cnn, generators)
            blend_metrics = aucs(labels, .8 * ensemble + .2 * cnn, generators)
            rows.append({
                "year": year, "variant": variant, "validation_cases": len(expected),
                "seeds": config["seeds"] if variant == "seed_mean" else [
                    int(variant.removeprefix("seed")) if variant in ("seed42", "seed7", "seed21") else 42],
                "cnn_member_weights": members,
                "blend_member_weights": {**{name: .16 for name in xgb_names},
                                         **{name: .2 * weight for name, weight in members.items()}},
                "cnn": cnn_metrics, "blend": blend_metrics,
                "cnn_delta_baseline": cnn_metrics["micro_roc_auc"] - baseline_cnn,
                "blend_delta_baseline": blend_metrics["micro_roc_auc"] - baseline_blend,
            })
    report = {"config": config, "baseline_variant": "seed42", "xgboost_member_weights": xgb_sources,
              "blend_weights": {"xgboost_ensemble": .8, "cnn": .2},
              "raw_averaging_controls": controls, "results": rows,
              "note": "Unweighted micro AUC; historical validation is development data. Variants and blend weights were fixed before these runs."}
    with output.open("x") as file:
        file.write(json.dumps(report, indent=2) + "\n")
    table = [{"year": row["year"], "variant": row["variant"],
              "cnn_auc": row["cnn"]["micro_roc_auc"], "blend_auc": row["blend"]["micro_roc_auc"],
              "blend_delta": row["blend_delta_baseline"]} for row in rows]
    print(pd.DataFrame(table).set_index(["variant", "year"]).unstack("year").round(6).to_string())


if __name__ == "__main__":
    main()
