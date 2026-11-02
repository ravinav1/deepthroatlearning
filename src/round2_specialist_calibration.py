"""Fit and select Lio specialist calibration using only 2021 prediction labels."""

import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from features import HOURS, ROOT
from plant_blend import column_groups, metrics, read_predictions
from round2_compare import BASELINE_CONFIG, BASELINE_PREFIXES, combine


SLOPE_BOUNDS = (0.25, 4.0)
INTERCEPT_BOUNDS = (-5.0, 5.0)
PROBABILITY_CLIP = 1e-12
SPECIALIST_PREFIX = "round2_xgb_lio"


def apply_platt(probabilities, slope, intercept):
    if slope == 1.0 and intercept == 0.0:
        return probabilities.copy()
    logits = logit(np.clip(probabilities, PROBABILITY_CLIP, 1 - PROBABILITY_CLIP))
    return expit(slope * logits + intercept)


def fit_platt(probabilities, labels):
    logits = logit(np.clip(probabilities.ravel(), PROBABILITY_CLIP, 1 - PROBABILITY_CLIP))
    targets = labels.ravel().astype(np.float64)

    def objective(parameters):
        values = parameters[0] * logits + parameters[1]
        error = expit(values) - targets
        loss = np.mean(np.logaddexp(0, values) - targets * values)
        gradient = np.array([np.mean(error * logits), np.mean(error)])
        return float(loss), gradient

    result = minimize(objective, [1.0, 0.0], jac=True, method="L-BFGS-B",
                      bounds=[SLOPE_BOUNDS, INTERCEPT_BOUNDS],
                      options={"maxiter": 200, "ftol": 1e-12, "gtol": 1e-9})
    assert result.success and np.isfinite(result.fun), result.message
    before = objective([1.0, 0.0])[0]
    assert result.fun <= before + 1e-10
    return {"slope": float(result.x[0]), "intercept": float(result.x[1])}, {
        "bce_before": before, "bce_after": float(result.fun),
        "optimizer_iterations": int(result.nit), "optimizer_evaluations": int(result.nfev),
        "gradient_at_solution": result.jac.tolist(),
        "clipped_prediction_count": int(((probabilities < PROBABILITY_CLIP) |
                                          (probabilities > 1 - PROBABILITY_CLIP)).sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prefix", default="round2_xgb_lio_platt")
    args = parser.parse_args()
    report_path = ROOT / f"reports/{args.output_prefix}_selection_2021.json"
    prediction_path = ROOT / f"data/processed/{args.output_prefix}_2021_validation_predictions.csv"
    assert not report_path.exists() and not prediction_path.exists()

    config = json.loads((ROOT / BASELINE_CONFIG).read_text())
    columns = pd.read_csv(ROOT / "data/raw/sample_submission.csv", nrows=0).columns[1:].tolist()
    # Read dates/IDs first, then parse labels for 2021 rows only. No 2022 labels
    # or prediction files are loaded by this selection program.
    truth_path = ROOT / "data/raw/kernel/Unit_commitment_decisions.csv"
    metadata = pd.read_csv(truth_path, usecols=["Run No", "starttime"])
    metadata["starttime"] = pd.to_datetime(metadata.starttime, utc=True)
    skip_rows = (np.flatnonzero(metadata.starttime.dt.year.to_numpy() != 2021) + 1).tolist()
    cases = pd.read_csv(truth_path, skiprows=skip_rows, usecols=["Run No", "starttime", *columns],
                        dtype={column: "uint8" for column in columns})
    cases["starttime"] = pd.to_datetime(cases.starttime, utc=True)
    cases = cases.sort_values("starttime").reset_index(drop=True)
    assert len(cases) == 365 and (cases.starttime.dt.year == 2021).all()
    test_ids = pd.read_csv(ROOT / "data/raw/prediction_mapping.csv", usecols=["Run No"])["Run No"]
    assert not cases["Run No"].isin(test_ids).any()
    paths = {name: f"data/processed/{prefix}_2021_validation_predictions.csv"
             for name, prefix in BASELINE_PREFIXES.items()}
    original = {name: read_predictions(ROOT / path, columns, cases["Run No"])
                for name, path in paths.items()}
    specialist_path = f"data/processed/{SPECIALIST_PREFIX}_2021_validation_predictions.csv"
    source_report_path = f"reports/{SPECIALIST_PREFIX}_2021_metrics.json"
    source_report = json.loads((ROOT / source_report_path).read_text())
    assert source_report["target_generator"] == "Lio_G1"
    assert pd.Timestamp(source_report["last_training_start"]) + pd.Timedelta(hours=HOURS) <= cases.starttime.min()
    specialist = read_predictions(ROOT / specialist_path, columns, cases["Run No"])
    indices = column_groups(columns)["Lio"]
    other = np.ones(len(columns), dtype=bool)
    other[indices] = False
    assert np.array_equal(specialist[:, other], original["xgb_rescaled"][:, other])
    labels = cases[columns].to_numpy()
    generators = list(dict.fromkeys(c.removeprefix("result_committed_").rsplit("_t", 1)[0]
                                    for c in columns))
    parameters, fit = fit_platt(specialist[:, indices], labels[:, indices])
    calibrated = specialist.copy()
    calibrated[:, indices] = apply_platt(specialist[:, indices], **parameters)
    baseline = combine(original, columns, config["plant_weights"])
    baseline_metrics = metrics(labels, baseline, generators)
    selection_rows = []
    selected, best_auc = "identity", -1.0
    variants = {"identity": specialist, "platt": calibrated}
    for name, predictions in variants.items():
        ensemble = combine({**original, "xgb_rescaled": predictions}, columns, config["plant_weights"])
        assert np.array_equal(ensemble[:, other], baseline[:, other])
        result = metrics(labels, ensemble, generators)
        result.update({"variant": name, "delta_vs_original_ensemble": result["micro_roc_auc"] - baseline_metrics["micro_roc_auc"],
                       "specialist_lio_auc": float(roc_auc_score(labels[:, indices].ravel(), predictions[:, indices].ravel()))})
        selection_rows.append(result)
        if result["micro_roc_auc"] > best_auc + 1e-12:
            selected, best_auc = name, result["micro_roc_auc"]
    selected_parameters = parameters if selected == "platt" else {"slope": 1.0, "intercept": 0.0}
    source_paths = [BASELINE_CONFIG, specialist_path, source_report_path, *paths.values()]
    report = {
        "fit_year": 2021, "fit_cases": len(cases), "target_generator": "Lio_G1",
        "specialist_prefix": SPECIALIST_PREFIX, "source_best_iteration": source_report["best_iteration"],
        "first_case_start": str(cases.starttime.min()), "last_case_start": str(cases.starttime.max()),
        "selection_horizon_end_exclusive": str(cases.starttime.max() + pd.Timedelta(hours=HOURS)),
        "fit_case_ids_sha256": hashlib.sha256(cases["Run No"].to_numpy(dtype="int64").tobytes()).hexdigest(),
        "fit_labels_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
        "slope_bounds_predeclared": list(SLOPE_BOUNDS), "intercept_bounds_predeclared": list(INTERCEPT_BOUNDS),
        "probability_clip": PROBABILITY_CLIP, "fit_objective": "Mean Lio specialist BCE over2021, no class balancing.",
        "transformation": "sigmoid(slope * logit(clip(p,1e-12,1-1e-12)) + intercept); exact identity returns original probabilities.",
        "candidate_parameters": parameters, "fit": fit,
        "selected_variant": selected, "selected_parameters": selected_parameters,
        "selection_rule": "Compare identity then fitted Platt on2021 full fixed ensemble micro-AUC; retain identity unless improvement exceeds1e-12.",
        "baseline_config": BASELINE_CONFIG, "plant_weights": config["plant_weights"],
        "changed_component": "Only Lio XGBoost share0.35; LightGBM0.35, Transformer0.30 and all other plants remain fixed.",
        "original_ensemble_metrics": baseline_metrics, "selection_results": selection_rows,
        "selected_component_predictions": str(prediction_path.relative_to(ROOT)),
        "source_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in source_paths},
        "confirmation_evaluated": False,
        "limitations": [
            "This hypothesis was proposed after seeing identity specialist2022 results; 2022 is reused development data, not a fresh holdout.",
            "Calibration fitting and variant selection share2021 labels, which also selected base-model checkpoints. Training records exclude2021 but this is not nested calibration validation.",
            "Positive slope preserves single-specialist ordering mathematically; clipping/finite precision may create ties. Full ensemble cross-generator and within-Lio rankings can change.",
            "No2022 prediction files or label values were loaded. Frozen confirmation requires a separate explicit step after this report is saved.",
            "No base model was fitted and no competition labels or submission outputs were used."
        ],
    }
    with report_path.open("x") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    frame = pd.DataFrame(variants[selected], columns=columns)
    frame.insert(0, "Run No", cases["Run No"].to_numpy())
    frame.to_csv(prediction_path, index=False, mode="x")
    print(json.dumps({"selected_variant": selected, "selected_parameters": selected_parameters,
                      "fit": fit, "original_auc": baseline_metrics["micro_roc_auc"],
                      "selection_results": [{key: row[key] for key in ["variant", "micro_roc_auc", "delta_vs_original_ensemble", "specialist_lio_auc"]}
                                            for row in selection_rows]}, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
