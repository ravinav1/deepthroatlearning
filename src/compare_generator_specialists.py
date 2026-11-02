"""Compare generator specialists against the fixed rescaled 80/20 baseline."""

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from features import HOURS, ROOT
from generator_specialist import output_targets


def metrics(labels, probabilities, generators):
    return {
        "micro_roc_auc": float(roc_auc_score(labels.ravel(), probabilities.ravel())),
        "generator_auc": {
            generator: float(roc_auc_score(labels[:, i].ravel(), probabilities[:, i].ravel()))
            for i, generator in enumerate(generators)
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/generator_specialists.json")
    parser.add_argument("--output", default="reports/generator_specialists_comparison.json")
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text())
    output = ROOT / args.output
    assert not output.exists(), f"Report already exists: {output}"
    columns = pd.read_csv(ROOT / "data/raw/sample_submission.csv", nrows=0).columns[1:].tolist()
    generators = list(dict.fromkeys(column.removeprefix("result_committed_").rsplit("_t", 1)[0]
                                   for column in columns))
    truth = pd.read_csv(ROOT / "data/raw/kernel/Unit_commitment_decisions.csv",
                        usecols=["Run No", "starttime", *columns],
                        dtype={column: "uint8" for column in columns}).set_index("Run No")
    truth["starttime"] = pd.to_datetime(truth.starttime, utc=True)
    rows, target_rows = [], []
    for year in config["years"]:
        cases = truth.loc[truth.starttime.dt.year == year].sort_values("starttime")
        labels = cases[columns].to_numpy().reshape(len(cases), len(generators), HOURS)

        def read(stem, selected_columns):
            frame = pd.read_csv(ROOT / f"data/processed/{stem}_validation_predictions.csv").set_index("Run No")
            assert frame.index.is_unique and frame.index.equals(cases.index)
            assert frame.columns.tolist() == selected_columns
            values = frame.to_numpy()
            assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
            return values

        profiles = ["price", "sampling", "regularized", "deep_regularized", "recency"]
        xgb = (sum(read(f"xgboost_{profile}_{year}", columns) for profile in profiles)
               / len(profiles)).reshape(labels.shape)
        cnn = read(f"{config['baseline_prefix']}_{year}", columns).reshape(labels.shape)
        baseline = config["xgboost_weight"] * xgb + config["cnn_weight"] * cnn
        baseline_auc = metrics(labels, baseline, generators)["micro_roc_auc"]
        specialists = {}
        for generator, prefix in config["specialists"].items():
            selected_columns, _ = output_targets(columns, generators, generator)
            stem = f"{prefix}_{year}"
            report = json.loads((ROOT / f"reports/{stem}_metrics.json").read_text())
            assert report["target_generator"] == generator and report["output_columns"] == selected_columns
            values = read(stem, selected_columns)
            specialists[generator] = values
            i = generators.index(generator)
            target_rows.append({
                "year": year, "generator": generator, "source": stem,
                "best_epoch": report["best_epoch"],
                "baseline_cnn_auc": float(roc_auc_score(labels[:, i].ravel(), cnn[:, i].ravel())),
                "specialist_auc": float(roc_auc_score(labels[:, i].ravel(), values.ravel())),
                "baseline_blend_auc": float(roc_auc_score(labels[:, i].ravel(), baseline[:, i].ravel())),
            })
        deployments = {}
        for route in config["routes"]:
            for subset, selected_generators in config["subsets"].items():
                deployments[f"{subset}_{route}"] = {generator: route for generator in selected_generators}
        deployments.update(config["mixed_routes"])
        variants = {"baseline": (baseline, {})}
        for name, generator_routes in deployments.items():
            candidate = baseline.copy()
            for generator, route in generator_routes.items():
                assert route in config["routes"]
                i = generators.index(generator)
                specialist = specialists[generator]
                candidate[:, i] = (config["xgboost_weight"] * xgb[:, i]
                                   + config["cnn_weight"] * specialist
                                   if route == "cnn_component" else specialist)
            untouched = [i for i, generator in enumerate(generators) if generator not in generator_routes]
            assert np.array_equal(candidate[:, untouched], baseline[:, untouched])
            variants[name] = (candidate, generator_routes)
        for name, (probabilities, generator_routes) in variants.items():
            result = metrics(labels, probabilities, generators)
            rows.append({"year": year, "variant": name, "generator_routes": generator_routes,
                         "target_generators": list(generator_routes), "validation_cases": len(cases),
                         **result, "delta_vs_baseline": result["micro_roc_auc"] - baseline_auc})
    report = {"config": config, "target_results": target_rows, "results": rows,
              "note": "Historical development comparisons. Global AUC includes all 14 generators; specialist training selects epochs using only its target AUC. No test-case labels or predictions enter selection."}
    with output.open("x") as file:
        file.write(json.dumps(report, indent=2) + "\n")
    print(pd.DataFrame(rows).pivot(index="variant", columns="year", values="micro_roc_auc").round(6).to_string())
    print(pd.DataFrame(target_rows).set_index(["generator", "year"]).round(6).to_string())


if __name__ == "__main__":
    main()
