"""Select simple component temperatures on 2021; evaluate only that choice on 2022."""

import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from features import HOURS, ROOT
from plant_blend import column_groups, metrics
from round2_compare import BASELINE_CONFIG, BASELINE_PREFIXES, combine, load_year


TEMPERATURE_BOUNDS = (0.25, 4.0)
PROBABILITY_CLIP = 1e-12


def component_weights(columns, plant_weights, component):
    weights = np.zeros(len(columns), dtype=np.float64)
    for plant, indices in column_groups(columns).items():
        weights[indices] = plant_weights[plant].get(component, 0.0)
    return weights


def apply_temperature(probabilities, temperature):
    if temperature == 1.0:
        return probabilities.copy()
    # Exact endpoints stay endpoints; clipping below is only for finite fitting losses.
    return expit(logit(probabilities) / temperature)


def fit_temperature(probabilities, labels, column_weights):
    used = column_weights > 0
    original = probabilities[:, used]
    logits = logit(np.clip(original, PROBABILITY_CLIP, 1 - PROBABILITY_CLIP))
    targets = labels[:, used].astype(np.float64, copy=False)
    weights = column_weights[used]
    normalization = len(labels) * weights.sum()

    def objective(log_temperature):
        scaled = logits / np.exp(log_temperature)
        loss = np.logaddexp(0, scaled) - targets * scaled
        return float((loss * weights).sum() / normalization)

    result = minimize_scalar(objective, bounds=np.log(TEMPERATURE_BOUNDS),
                             method="bounded", options={"xatol": 1e-6, "maxiter": 100})
    assert result.success and np.isfinite(result.fun)
    temperature = float(np.exp(result.x))
    before, after = objective(0.0), objective(result.x)
    assert after <= before + 1e-10
    applied = np.clip(apply_temperature(original, temperature), PROBABILITY_CLIP, 1 - PROBABILITY_CLIP)
    applied_bce = float(((-targets * np.log(applied) - (1 - targets) * np.log1p(-applied)) * weights).sum() / normalization)
    return temperature, {
        "temperature": temperature, "weighted_bce_before": before, "weighted_bce_after": after,
        "weighted_bce_delta": after - before, "optimizer_evaluations": result.nfev,
        "applied_weighted_bce_after": applied_bce,
        "misclassified_exact_endpoints": int((((original == 1) & (targets == 0)) | ((original == 0) & (targets == 1))).sum()),
        "weighted_output_mass": float(normalization), "active_columns": int(used.sum()),
        "excluded_zero_weight_columns": int((~used).sum()),
        "fitting_clipped_outputs": int(((original < PROBABILITY_CLIP) | (original > 1 - PROBABILITY_CLIP)).sum()),
        "at_search_boundary": bool(temperature < TEMPERATURE_BOUNDS[0] + 1e-4 or
                                   temperature > TEMPERATURE_BOUNDS[1] - 1e-4),
    }


def transformed_components(components, temperatures, variant):
    selected = [] if variant == "original" else ["transformer"] if variant == "neural_only" else list(components)
    return {name: apply_temperature(values, temperatures[name]) if name in selected else values
            for name, values in components.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="reports/round2_temperature_comparison.json")
    args = parser.parse_args()
    output = ROOT / args.output
    assert not output.exists()
    config, columns, cases, components = load_year(2021)
    labels = cases[columns].to_numpy()
    generators = list(dict.fromkeys(c.removeprefix("result_committed_").rsplit("_t", 1)[0] for c in columns))
    temperatures, fits = {}, {}
    for name, probabilities in components.items():
        temperatures[name], fits[name] = fit_temperature(
            probabilities, labels, component_weights(columns, config["plant_weights"], name))
        calibrated = apply_temperature(probabilities, temperatures[name])
        fits[name]["component_auc_change_numerical_only"] = float(
            roc_auc_score(labels.ravel(), calibrated.ravel()) - roc_auc_score(labels.ravel(), probabilities.ravel()))

    selection_rows = []
    selected, best_auc = "original", -1.0
    for variant in ["original", "neural_only", "all_components"]:
        probabilities = combine(transformed_components(components, temperatures, variant), columns, config["plant_weights"])
        result = metrics(labels, probabilities, generators)
        selection_rows.append({"year": 2021, "variant": variant, "cases": len(cases), **result})
        if result["micro_roc_auc"] > best_auc + 1e-12:
            selected, best_auc = variant, result["micro_roc_auc"]
    reference = json.loads((ROOT / config["historical_report"]).read_text())
    expected = next(row["micro_roc_auc"] for row in reference["results"]
                    if row["year"] == 2021 and row["variant"] == config["historical_variant"])
    assert abs(selection_rows[0]["micro_roc_auc"] - expected) < 1e-12
    for row in selection_rows:
        row["delta_vs_original"] = row["micro_roc_auc"] - selection_rows[0]["micro_roc_auc"]
    selection_horizon_end = cases.starttime.max() + pd.Timedelta(hours=HOURS)
    print(f"Frozen 2021 choice: {selected}; temperatures: {temperatures}", flush=True)
    print(json.dumps([{key: row[key] for key in ["variant", "micro_roc_auc", "delta_vs_original"]}
                      for row in selection_rows]), flush=True)

    # Do not score unselected calibration variants on the confirmation year.
    confirmation_config, confirmation_columns, confirmation_cases, confirmation_components = load_year(2022)
    assert columns == confirmation_columns and config == confirmation_config
    truth = confirmation_cases[columns].to_numpy()
    confirmation = {"original": combine(confirmation_components, columns, config["plant_weights"])}
    if selected != "original":
        confirmation[selected] = combine(transformed_components(confirmation_components, temperatures, selected),
                                         columns, config["plant_weights"])
    confirmation_rows = []
    for subset, mask in [("all_2022", np.ones(len(confirmation_cases), dtype=bool)),
                         ("after_2021_selection_horizon", (confirmation_cases.starttime >= selection_horizon_end).to_numpy())]:
        original_auc = float(roc_auc_score(truth[mask].ravel(), confirmation["original"][mask].ravel()))
        for variant, probabilities in confirmation.items():
            result = metrics(truth[mask], probabilities[mask], generators)
            confirmation_rows.append({"year": 2022, "subset": subset, "variant": variant,
                                      "cases": int(mask.sum()), "delta_vs_original": result["micro_roc_auc"] - original_auc,
                                      **result})
    expected = next(row["micro_roc_auc"] for row in reference["results"]
                    if row["year"] == 2022 and row["variant"] == config["historical_variant"])
    assert abs(confirmation_rows[0]["micro_roc_auc"] - expected) < 1e-12
    source_paths = [BASELINE_CONFIG, "reports/generalization_calibration_comparison.json", *[
        f"data/processed/{prefix}_{year}_validation_predictions.csv"
        for prefix in BASELINE_PREFIXES.values() for year in [2021, 2022]
    ]]
    report = {
        "baseline_config": BASELINE_CONFIG, "plant_weights": config["plant_weights"],
        "prior_work": "Previous generalization_calibration_comparison fitted a shared logit slope plus 14 generator offsets on older unscaled XGBoost profiles. This test instead uses one positive scalar temperature per current rescaled XGBoost, LightGBM and Transformer component, with no offsets.",
        "temperature_bounds_predeclared": list(TEMPERATURE_BOUNDS), "probability_clip_for_finite_fit_loss": PROBABILITY_CLIP,
        "application": "sigmoid(logit(p)/T); exact 0 and 1 endpoints are preserved. T=1 returns original values unchanged.",
        "fit_objective": "Component BCE averaged with each output weighted by that component's frozen plant blend weight. Unused Transformer outputs have zero weight. This is component calibration, not direct ensemble BCE or AUC optimization.",
        "fit_year": 2021, "fit_cases": len(cases), "component_fits": fits,
        "selection_rule": "Evaluate original, neural_only and all_components final ensembles on 2021 micro ROC-AUC in that order; select only on improvement greater than 1e-12. Ties retain the simpler earlier variant. Temperatures and variant freeze before loading the 2022 prediction components.",
        "selected_variant": selected, "selected_temperatures": {
            name: temperatures[name] if selected == "all_components" or (selected == "neural_only" and name == "transformer") else 1.0
            for name in temperatures},
        "selection_results": selection_rows, "confirmation_results": confirmation_rows,
        "selection_horizon_end_exclusive": str(selection_horizon_end),
        "confirmation_overlap_starts_excluded_in_secondary_subset": int((confirmation_cases.starttime < selection_horizon_end).sum()),
        "unselected_2022_variants_evaluated": False,
        "source_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in source_paths},
        "limitations": [
            "Positive temperature preserves individual-component rankings mathematically. Finite precision can create ties; any tiny standalone AUC change is numerical, not learned discrimination. Ensemble rankings can change because the components' probability scales change.",
            "2021 temperature fitting and variant selection use the same already reused development year; its gains are in-sample calibration results.",
            "2022 has also been repeatedly used for development, and native component checkpoints were selected using 2022 labels. Freezing these new temperatures/variant does not make the complete ensemble independent of 2022.",
            "Removing six starts eliminates overlap with the 2021 calibration-label horizons but does not undo prior development reuse or native-model selection.",
            "Cached exact 0/1 probabilities have unrecoverable original logits. Finite BCE fitting clips probabilities to [1e-12,1-1e-12], while application preserves exact endpoints. Applied weighted BCE is separately reported.",
            "Only calibration scalars were fitted. No base-model training, competition-label access, submission creation or Kaggle probing occurred."
        ],
        "primary_source": {"title": "Guo et al. (2017), On Calibration of Modern Neural Networks",
                           "url": "https://proceedings.mlr.press/v70/guo17a.html",
                           "scope": "Supports temperature scaling as a simple post-hoc probability calibration technique. Binary plant-weighted fitting and blend-AUC selection here are project-specific adaptations; the paper does not establish hydropower AUC gains."},
    }
    with output.open("x") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print(json.dumps([{key: row[key] for key in ["subset", "variant", "micro_roc_auc", "delta_vs_original"]}
                      for row in confirmation_rows]), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
