"""Choose plant blend weights on one historical year and check another."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from features import HOURS, ROOT


def column_groups(columns):
    groups = {}
    for i, column in enumerate(columns):
        generator = column.removeprefix("result_committed_").rsplit("_t", 1)[0]
        plant = generator.rsplit("_G", 1)[0]
        groups.setdefault(plant, []).append(i)
    return groups


def blend(xgb, transformer, groups, weights, fixed_columns, fixed_values):
    result = np.empty_like(xgb)
    for plant, indices in groups.items():
        if plant in weights:
            weight = weights[plant]
            result[:, indices] = (1 - weight) * xgb[:, indices] + weight * transformer[:, indices]
    result[:, fixed_columns] = fixed_values
    return result


def metrics(labels, probabilities, generators):
    labels = labels.reshape(-1, len(generators), HOURS)
    probabilities = probabilities.reshape(labels.shape)
    return {
        "micro_roc_auc": float(roc_auc_score(labels.ravel(), probabilities.ravel())),
        "generator_auc": {generator: float(roc_auc_score(labels[:, i].ravel(), probabilities[:, i].ravel()))
                          for i, generator in enumerate(generators)},
    }


def read_predictions(path, columns, ids):
    frame = pd.read_csv(path, float_precision="round_trip")
    assert frame.columns.tolist() == ["Run No", *columns]
    assert frame["Run No"].is_unique and frame["Run No"].equals(ids)
    values = frame[columns].to_numpy()
    assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
    return values


def compare(config, output, columns, groups, generators, reference_path=None):
    assert not output.exists(), f"Report already exists: {output}"
    truth = pd.read_csv(ROOT / "data/raw/kernel/Unit_commitment_decisions.csv",
                        usecols=["Run No", "starttime", *columns],
                        dtype={column: "uint8" for column in columns})
    truth["starttime"] = pd.to_datetime(truth.starttime, utc=True)
    test_ids = pd.read_csv(ROOT / "data/raw/prediction_mapping.csv")["Run No"]
    assert not truth["Run No"].isin(test_ids).any()
    fixed_columns = [columns.index(f"result_committed_{config['fixed_generator']}_t{hour}")
                     for hour in range(HOURS)]
    fixed_names = [columns[i] for i in fixed_columns]
    fixed_plant = config["fixed_generator"].rsplit("_G", 1)[0]
    assert groups[fixed_plant] == fixed_columns
    plants = [plant for plant in groups if plant != fixed_plant]
    baseline_weights = dict.fromkeys(plants, config["baseline_transformer_weight"])

    def read_year(year):
        cases = truth.loc[truth.starttime.dt.year == year].sort_values("starttime").reset_index(drop=True)
        ids = cases["Run No"]
        xgb = sum(read_predictions(ROOT / f"data/processed/xgboost_{profile}_{year}_validation_predictions.csv",
                                  columns, ids) for profile in config["xgboost_profiles"]) / len(config["xgboost_profiles"])
        transformer = read_predictions(ROOT / f"data/processed/{config['transformer_prefix']}_{year}_validation_predictions.csv",
                                       columns, ids)
        specialist = read_predictions(ROOT / f"data/processed/{config['specialist_prefix']}_{year}_validation_predictions.csv",
                                      fixed_names, ids)
        fixed_values = .8 * xgb[:, fixed_columns] + .2 * specialist
        return cases[columns].to_numpy(), xgb, transformer, fixed_values

    labels, xgb, transformer, fixed_values = read_year(config["selection_year"])

    def probabilities(weights):
        return blend(xgb, transformer, groups, weights, fixed_columns, fixed_values)

    def auc(weights):
        return float(roc_auc_score(labels.ravel(), probabilities(weights).ravel()))

    # First select one shared weight, then refine each plant once on the selection year.
    uniform_weights = baseline_weights.copy()
    best_auc = auc(uniform_weights)
    uniform_trials = []
    for weight in config["candidate_transformer_weights"]:
        candidate = dict.fromkeys(plants, weight)
        value = auc(candidate)
        uniform_trials.append({"transformer_weight": weight, "micro_roc_auc": value})
        if value > best_auc + 1e-12:
            best_auc, uniform_weights = value, candidate
    plant_weights = uniform_weights.copy()
    if config.get("initial_plant_weights_source"):
        initial = json.loads((ROOT / config["initial_plant_weights_source"]).read_text())
        assert initial["config"]["selection_year"] == config["selection_year"]
        plant_weights = initial["selected_plant_weights"].copy()
        assert plant_weights.keys() == baseline_weights.keys()
        best_auc = auc(plant_weights)
    plant_trials = []
    for plant in plants:
        selected = plant_weights.copy()
        for weight in config["candidate_transformer_weights"]:
            candidate = plant_weights.copy()
            candidate[plant] = weight
            value = auc(candidate)
            plant_trials.append({"plant": plant, "transformer_weight": weight, "micro_roc_auc": value})
            if value > best_auc + 1e-12:
                best_auc, selected = value, candidate
        plant_weights = selected
    print(f"Frozen weights selected on {config['selection_year']}: {plant_weights}", flush=True)

    variants = {"baseline_80_20": baseline_weights, "selected_uniform": uniform_weights,
                "selected_plant": plant_weights}
    if reference_path is not None:
        reference = json.loads(reference_path.read_text())
        for key in ("xgboost_profiles", "transformer_prefix", "fixed_generator", "specialist_prefix"):
            assert reference["config"][key] == config[key], f"Different components for {key}."
        reference_weights = reference["selected_plant_weights"]
        assert reference_weights.keys() == plant_weights.keys()
        variants["reference_plant"] = reference_weights
        variants["mean_plant"] = {plant: (reference_weights[plant] + plant_weights[plant]) / 2
                                  for plant in plants}

    rows = []
    for year in (config["selection_year"], config["confirmation_year"]):
        # The confirmation year's labels are read only after all weights are frozen.
        if year != config["selection_year"]:
            labels, xgb, transformer, fixed_values = read_year(year)
        for name, weights in variants.items():
            values = probabilities(weights)
            assert np.array_equal(values[:, fixed_columns], fixed_values)
            rows.append({"year": year, "variant": name, "validation_cases": len(labels),
                         **metrics(labels, values, generators)})
    report = {"config": config, "plant_order": plants, "selected_uniform_weights": uniform_weights,
              "selected_plant_weights": plant_weights, "uniform_trials": uniform_trials,
              "plant_trials": plant_trials, "variant_weights": variants, "results": rows}
    if reference_path is not None:
        report.update({"reference_source": str(reference_path), "evaluation_note": config["note"]})
    with output.open("x") as file:
        file.write(json.dumps(report, indent=2) + "\n")
    print(pd.DataFrame(rows).pivot(index="variant", columns="year", values="micro_roc_auc").round(6).to_string())


def make_submission(config, comparison_path, output, columns, groups, variant="selected_plant"):
    comparison = json.loads(comparison_path.read_text())
    assert comparison["config"] == config
    for year in (config["selection_year"], config["confirmation_year"]):
        rows = {row["variant"]: row for row in comparison["results"] if row["year"] == year}
        reference = "reference_plant" if "reference_source" in comparison else "baseline_80_20"
        assert rows[variant]["micro_roc_auc"] > rows[reference]["micro_roc_auc"], \
            "The candidate must improve both historical years against the reference blend."
    report_path = ROOT / f"reports/{output.stem}_run.json"
    for path in (output, report_path):
        assert not path.exists(), f"Artifact already exists: {path}"
    template = pd.read_csv(ROOT / "data/raw/sample_submission.csv")
    baseline = read_predictions(ROOT / config["baseline_submission"], columns, template["Run No"])
    xgb = read_predictions(ROOT / config["xgboost_submission"], columns, template["Run No"])
    transformer = read_predictions(ROOT / config["transformer_predictions"], columns, template["Run No"])
    values = baseline.copy()
    weights = comparison.get("variant_weights", {"selected_plant": comparison["selected_plant_weights"]})[variant]
    for plant, weight in weights.items():
        indices = groups[plant]
        values[:, indices] = (1 - weight) * xgb[:, indices] + weight * transformer[:, indices]
    assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
    fixed_plant = config["fixed_generator"].rsplit("_G", 1)[0]
    assert np.array_equal(values[:, groups[fixed_plant]], baseline[:, groups[fixed_plant]])
    submission = template.copy()
    submission[columns] = values
    submission.to_csv(output, index=False, mode="x")
    report = {"config": config, "selection_source": str(comparison_path),
              "selected_variant": variant, "selected_plant_weights": weights, "test_cases": len(template),
              "outputs_per_case": len(columns), "preserved_generator": config["fixed_generator"],
              "historical_results": comparison["results"], "submission": str(output),
              "note": comparison.get("evaluation_note", f"Weights selected on {config['selection_year']}, checked unchanged on {config['confirmation_year']}.")}
    report["note"] += " Saved model predictions only; no test labels or new model training."
    with report_path.open("x") as file:
        file.write(json.dumps(report, indent=2) + "\n")
    print(f"Saved {len(submission)} cases to {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/plant_blend.json")
    parser.add_argument("--make-submission", action="store_true")
    parser.add_argument("--comparison-report", default="reports/plant_blend_comparison.json")
    parser.add_argument("--reference-report", help="Also compare the existing plant weights and one fixed mean.")
    parser.add_argument("--variant", choices=["selected_plant", "mean_plant"], default="selected_plant")
    parser.add_argument("--output")
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text())
    columns = pd.read_csv(ROOT / "data/raw/sample_submission.csv", nrows=0).columns[1:].tolist()
    generators = list(dict.fromkeys(column.removeprefix("result_committed_").rsplit("_t", 1)[0]
                                   for column in columns))
    groups = column_groups(columns)
    default_output = "submissions/submission_xgb_transformer_plant_blend.csv" if args.make_submission else args.comparison_report
    output = ROOT / (args.output or default_output)
    if args.make_submission:
        make_submission(config, ROOT / args.comparison_report, output, columns, groups, args.variant)
    else:
        reference_path = ROOT / args.reference_report if args.reference_report else None
        compare(config, output, columns, groups, generators, reference_path)


if __name__ == "__main__":
    main()
