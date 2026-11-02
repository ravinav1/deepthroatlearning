"""Compare the two fixed EMA continuations against saved EMA and raw CNN references."""

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


def exact_prefix(reference, candidate, fields, through_epoch, description):
    for history in (reference, candidate):
        epochs = [row["epoch"] for row in history]
        assert epochs == sorted(set(epochs)), f"Invalid history epoch order: {description}"
    lookup = {row["epoch"]: row for row in candidate}
    prefix = [row for row in reference if row["epoch"] <= through_epoch]
    assert prefix, f"Empty history prefix: {description}"
    for row in prefix:
        assert row["epoch"] in lookup, f"Missing epoch {row['epoch']}: {description}"
        for field in fields:
            assert row[field] == lookup[row["epoch"]][field], \
                f"History differs at epoch {row['epoch']}, field {field}: {description}"
    return {"exact_equal": True, "matched_epochs": len(prefix),
            "through_epoch": through_epoch, "fields": fields}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/cnn_ema_continuation.json"))
    parser.add_argument("--output", default=str(ROOT / "reports/cnn_ema_continuation_comparison.json"))
    args = parser.parse_args()
    output = Path(args.output)
    assert not output.exists(), f"Comparison report already exists: {output}"
    config = json.loads(Path(args.config).read_text())
    assert set(config["branches"]) == {"long", "slow"} and config["years"] == [2021, 2022]
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
    results, controls, xgb_references = [], {}, {}
    for year in config["years"]:
        expected = truth.loc[truth["starttime"].dt.year == year].sort_values("starttime")
        assert len(expected), f"No labelled cases for {year}"
        raw_ref, ema_ref = f"cnn_{year}", f"cnn_averaging_{year}_ema"
        raw_names = {branch: f"{options['cnn_prefix']}_{year}" for branch, options in config["branches"].items()}
        variants = {"raw_reference": raw_ref, "ema_reference": ema_ref,
                    **{f"{branch}_ema": f"{name}_ema" for branch, name in raw_names.items()}}
        reports = {name: json.loads((ROOT / f"reports/{name}_metrics.json").read_text())
                   for name in [raw_ref, ema_ref, *raw_names.values(), *variants.values()]}
        baseline = reports[raw_ref]
        for name, report in reports.items():
            for field in ("feature_names", "training_cases", "validation_cases", "purged_cases",
                          "last_training_start", "first_validation_start", "last_validation_start"):
                assert report[field] == baseline[field], f"Report metadata differs: {name}, {field}"
        control = {}
        raw_fields, ema_fields = ["epoch", "training_loss", "validation_auc"], ["epoch", "validation_auc", "averaging_updates"]
        for branch, name in raw_names.items():
            averaged = reports[f"{name}_ema"]
            assert averaged["averaging_decay"] == config["branches"][branch]["ema_decay"]
            control[f"{branch}_raw_prefix"] = exact_prefix(
                baseline["history"], reports[name]["history"], raw_fields,
                baseline["history"][-1]["epoch"], f"{name} versus {raw_ref}")
            if averaged["averaging_decay"] == reports[ema_ref]["averaging_decay"]:
                control[f"{branch}_ema_prefix"] = exact_prefix(
                    reports[ema_ref]["history"], averaged["history"], ema_fields,
                    reports[ema_ref]["history"][-1]["epoch"], f"{name}_ema versus {ema_ref}")
        histories = [reports[name]["history"] for name in raw_names.values()]
        control["raw_branch_overlap"] = exact_prefix(
            histories[0], histories[1], raw_fields,
            min(history[-1]["epoch"] for history in histories), "long versus slow raw")
        controls[str(year)] = control
        xgb_names = [f"xgboost_{profile}_{year}" for profile in PROFILES]
        values = {}
        for name in [*xgb_names, *variants.values(), *raw_names.values()]:
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
        scores = {variant: {"cnn": aucs(labels, values[name], generators),
                            "blend": aucs(labels, .8 * ensemble + .2 * values[name], generators)}
                  for variant, name in variants.items()}
        for variant, name in variants.items():
            assert abs(scores[variant]["cnn"]["micro_roc_auc"] - reports[name]["cnn_micro_roc_auc"]) < 1e-10, \
                f"Saved AUC differs: {name}"
            row = {"year": year, "variant": variant, "cnn_member": name,
                   "validation_cases": len(expected), "best_epoch": reports[name]["best_epoch"],
                   "blend_member_weights": {**{member: .16 for member in xgb_names}, name: .2}, **scores[variant]}
            for kind in ("cnn", "blend"):
                for reference in ("ema_reference", "raw_reference"):
                    row[f"{kind}_delta_{reference}"] = (scores[variant][kind]["micro_roc_auc"]
                                                        - scores[reference][kind]["micro_roc_auc"])
            results.append(row)
    report = {"config": config, "history_controls": controls, "xgboost_references": xgb_references,
              "blend_weights": {"xgboost_ensemble": .8, "cnn": .2}, "results": results,
              "note": "Unweighted micro AUC. Historical validation is development data; both planned branches are retained."}
    with output.open("x") as file:
        file.write(json.dumps(report, indent=2) + "\n")
    table = [{"variant": row["variant"], "year": row["year"], "cnn_auc": row["cnn"]["micro_roc_auc"],
              "blend_auc": row["blend"]["micro_roc_auc"], "delta_ema": row["blend_delta_ema_reference"],
              "delta_raw": row["blend_delta_raw_reference"]} for row in results]
    print(pd.DataFrame(table).set_index(["variant", "year"]).unstack("year").round(6).to_string())


if __name__ == "__main__":
    main()
