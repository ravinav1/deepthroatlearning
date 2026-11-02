"""Run from the project root: python src/train_baseline.py"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit

from features import HOURS, ROOT, build_features, flatten_labels, load_data, prediction_frame
from losses import focal_loss
from tree_preprocessing import rescale_tree_features

sys.path.insert(0, str(ROOT / "evaluation"))
from kaggle_metric import score


def training_case_hash(cases):
    """Identify the ordered training cohort without storing every case ID."""
    values = cases[["Run No", "starttime"]].to_csv(index=False).encode()
    return hashlib.sha256(values).hexdigest()


def prepare_continuation(config, features, cases):
    """Resume the same binary model/data; n_estimators remains a TOTAL cap."""
    params = config["model"].copy()
    overrides = config.get("continuation_parameter_overrides", {})
    assert set(overrides).issubset({"max_delta_step"}), "Only an explicit max_delta_step continuation override is supported."
    assert not overrides or config.get("init_model"), "Continuation overrides require an initial model."
    for key, value in overrides.items():
        assert np.isfinite(value) and value > 0 and params.get(key) == value, \
            f"The positive continuation override must match model.{key}."
    if not config.get("init_model"):
        return None, params, None
    model_path = ROOT / config["init_model"]
    report_path = ROOT / config["init_model_report"]
    source = json.loads(report_path.read_text())
    previous = source["config"]
    assert config.get("loss", "binary") == previous.get("loss", "binary") == "binary"
    assert previous.get("model_type", "lightgbm") == "lightgbm"
    for key in set(config) | set(previous):
        if key.startswith("use_"):
            assert config.get(key, False) == previous.get(key, False), f"Changed feature option: {key}"
    for key in ("validation_start", "validation_end", "recency_half_life_years"):
        assert config.get(key) == previous.get(key), f"Changed training setup: {key}"
    allowed_changes = {"n_estimators", *overrides}
    assert {k: v for k, v in params.items() if k not in allowed_changes} == {
        k: v for k, v in previous["model"].items() if k not in allowed_changes
    }, "Continuation must preserve model parameters except the total tree cap and explicitly allowed overrides."
    assert len(cases) == source["training_cases"], "Changed training cohort size."
    assert str(cases.starttime.max()) == source["last_training_start"], "Changed training period."
    if "first_training_start" in source:
        assert str(cases.starttime.min()) == source["first_training_start"]
    cohort_hash = training_case_hash(cases)
    if "training_case_ids_sha256" in source:
        assert cohort_hash == source["training_case_ids_sha256"], "Changed training case IDs/order."

    source_name = report_path.stem.removesuffix("_metrics").removesuffix("_run")
    expected_model = ROOT / source.get("model_path", f"models/lightgbm_{source_name}.txt")
    assert model_path.resolve() == expected_model.resolve(), "Model/report provenance mismatch."
    initial = lgb.Booster(model_file=str(model_path))
    names = features.columns.tolist()
    assert names == source.get("feature_names", source.get("features")) == initial.feature_name(), \
        "Continuation requires identical feature names and order."
    initial_rounds = initial.current_iteration()
    assert initial_rounds == source.get("best_iteration", source.get("trees")), \
        "Source report and model tree counts differ."
    target = params["n_estimators"]
    assert target > initial_rounds, "The requested total must exceed the saved model's tree count."
    params["n_estimators"] = target - initial_rounds
    provenance = {
        "source_model": str(model_path), "source_report": str(report_path),
        "source_model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "source_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "initial_rounds": initial_rounds, "requested_total_rounds": target,
        "requested_additional_rounds": params["n_estimators"],
        "training_case_ids_sha256": cohort_hash,
        "cohort_verification": ("stored_cohort_hash" if "training_case_ids_sha256" in source else
                                "legacy_report_split_dates_count_and_preserved_original_data"),
        "note": "Append new rounds to the preserved source model; the caller controls fixed-count fitting or early stopping.",
    }
    if overrides:
        provenance["parameter_overrides"] = {
            key: {"source_value": initial.params.get(key, previous["model"].get(key, 0.0)),
                  "new_value": value}
            for key, value in overrides.items()
        }
        provenance["parameter_phase_count"] = (source.get("continuation") or {}).get("parameter_phase_count", 1) + 1
        provenance["override_phase"] = {
            "source_rounds_preserved": initial_rounds, "first_new_tree": initial_rounds + 1,
            "requested_new_rounds": params["n_estimators"], "requested_last_tree": target,
            "maximum_absolute_new_leaf_output": params["learning_rate"] * overrides["max_delta_step"],
        }
    return initial, params, provenance


def main():
    parser = argparse.ArgumentParser(description="Train and validate the first LightGBM baseline.")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs/baseline.json"))
    args = parser.parse_args()
    with open(args.config) as file:
        config = json.load(file)
    run_name = Path(args.config).stem
    model_path = ROOT / f"models/lightgbm_{run_name}.txt"
    predictions_path = ROOT / f"data/processed/{run_name}_validation_predictions.csv"
    report_path = ROOT / f"reports/{run_name}_metrics.json"
    for path in (model_path, predictions_path, report_path):
        assert not path.exists(), f"Artifact already exists: {path}; choose a new config name."

    use_extended = config.get("use_extended", False)
    use_system_context = config.get("use_system_context", False)
    use_topology = config.get("use_topology", False)
    use_price_context = config.get("use_price_context", False)
    use_water_context = config.get("use_water_context", False)
    use_economic_context = config.get("use_economic_context", False)
    use_river_context = config.get("use_river_context", False)
    use_inflow_profiles = config.get("use_inflow_profiles", False)
    use_hydraulic_context = config.get("use_hydraulic_context", False)
    use_water_budget = config.get("use_water_budget", False)
    use_temporal_ranks = config.get("use_temporal_ranks", False)
    cases, label_columns, generators, inputs, topology = load_data(use_extended=use_extended)
    validation_start = pd.Timestamp(config["validation_start"], tz="UTC")
    ends = cases["starttime"] + pd.Timedelta(hours=HOURS)
    train = cases.loc[ends <= validation_start].copy()
    validation_mask = cases["starttime"] >= validation_start
    if "validation_end" in config:
        validation_end = pd.Timestamp(config["validation_end"], tz="UTC")
        validation_mask &= cases["starttime"] < validation_end
    validation = cases.loc[validation_mask].copy()
    purged = int(((cases["starttime"] < validation_start) & (ends > validation_start)).sum())
    assert len(train) and len(validation), "The split needs both training and validation cases."
    assert (train["starttime"] + pd.Timedelta(hours=HOURS)).max() <= validation["starttime"].min()
    test_ids = pd.read_csv(ROOT / "data/raw/prediction_mapping.csv", usecols=["Run No"])["Run No"]
    assert not cases["Run No"].isin(test_ids).any(), "Competition test cases must stay out of fitting and validation."
    print(f"Cases: {len(train)} training, {len(validation)} validation, {purged} purged", flush=True)

    truth = validation[["Run No", *label_columns]]
    y_train = flatten_labels(train, label_columns, generators)
    y_validation = flatten_labels(validation, label_columns, generators)

    # Each generator gets its training ON rate, with no validation labels used.
    on_rates = y_train.reshape(len(generators), -1).mean(axis=1)
    frequency_probabilities = np.repeat(on_rates, len(validation) * HOURS)
    frequency_score = score(truth, prediction_frame(
        frequency_probabilities, validation, label_columns, generators
    ))
    print(f"Generator-frequency micro ROC-AUC: {frequency_score:.6f}", flush=True)

    print(f"Building features (extended={use_extended}, system_context={use_system_context}, topology={use_topology})...", flush=True)
    x_train = build_features(train, generators, inputs, topology,
                             use_extended=use_extended, use_system_context=use_system_context,
                             use_topology=use_topology, use_price_context=use_price_context,
                                  use_water_context=use_water_context, use_economic_context=use_economic_context,
                                  use_river_context=use_river_context, use_inflow_profiles=use_inflow_profiles,
                                  use_hydraulic_context=use_hydraulic_context, use_water_budget=use_water_budget,
                                  use_temporal_ranks=use_temporal_ranks)
    x_validation = build_features(validation, generators, inputs, topology,
                                  use_extended=use_extended, use_system_context=use_system_context,
                                  use_topology=use_topology, use_price_context=use_price_context,
                                  use_water_context=use_water_context, use_economic_context=use_economic_context,
                                  use_river_context=use_river_context, use_inflow_profiles=use_inflow_profiles,
                                  use_hydraulic_context=use_hydraulic_context, use_water_budget=use_water_budget,
                                  use_temporal_ranks=use_temporal_ranks)
    if config.get("use_economic_rescaling", False):
        x_train = rescale_tree_features(x_train)
        x_validation = rescale_tree_features(x_validation)
    print(f"Training on {len(x_train):,} rows and {x_train.shape[1]} features...", flush=True)
    loss = config.get("loss", "binary")
    assert loss in ["binary", "focal"], "loss must be binary or focal."
    objective = focal_loss(config.get("focal_gamma", 2.0)) if loss == "focal" else "binary"
    print(f"Loss: {loss}", flush=True)
    initial, model_params, continuation = prepare_continuation(config, x_train, train)
    initial_score = None
    if initial is not None:
        initial_predictions = prediction_frame(
            expit(initial.predict(x_validation, raw_score=True)), validation, label_columns, generators
        )
        initial_score = score(truth, initial_predictions)
        source = json.loads((ROOT / config["init_model_report"]).read_text())
        assert abs(initial_score - source["lightgbm_micro_roc_auc"]) < 1e-10, \
            "Source model no longer reproduces its validation score."
        print(f"Continuing {continuation['initial_rounds']} trees to total cap "
              f"{continuation['requested_total_rounds']}; source AUC: {initial_score:.6f}", flush=True)
    model = lgb.LGBMClassifier(objective=objective, metric="auc", **model_params)
    evaluation_history = {}
    stopping_rounds = config.get("early_stopping_rounds", 100)
    callbacks = [lgb.log_evaluation(25), lgb.record_evaluation(evaluation_history)]
    if stopping_rounds is not None:
        callbacks.insert(0, lgb.early_stopping(stopping_rounds, first_metric_only=True))
    started = time.monotonic()
    model.fit(
        x_train, y_train,
        categorical_feature=["generator_id"],
        eval_X=x_validation,
        eval_y=y_validation,
        eval_metric="auc",
        callbacks=callbacks,
        init_model=initial,
    )
    training_seconds = time.monotonic() - started
    # Custom objectives return raw scores; convert them before saving probabilities.
    probabilities = expit(model.predict(x_validation, raw_score=True))
    validation_predictions = prediction_frame(probabilities, validation, label_columns, generators)
    model_score = score(truth, validation_predictions)
    selected_booster = model.booster_
    if stopping_rounds is None:
        assert selected_booster.current_iteration() == config["model"]["n_estimators"], \
            "Fixed-count training must reach the configured total number of trees."
    if initial is not None:
        retain_initial = stopping_rounds is not None and initial_score >= model_score
        continuation.update({
            "source_validation_auc": initial_score, "continued_validation_auc": model_score,
            "evaluated_total_rounds": continuation["initial_rounds"] +
                                      len(next(iter(evaluation_history.values()))["auc"]),
            "retained_initial_model": retain_initial,
        })
        if stopping_rounds is None:
            continuation["continued_fixed_total_rounds"] = selected_booster.current_iteration()
            continuation["note"] = "Continue to the fixed total; validation scores do not select rounds or retain the source model."
        else:
            continuation["continued_best_total_rounds"] = selected_booster.current_iteration()
        if retain_initial:
            selected_booster, validation_predictions, model_score = initial, initial_predictions, initial_score
        continuation["selected_total_rounds"] = selected_booster.current_iteration()
    print(f"LightGBM micro ROC-AUC: {model_score:.6f}", flush=True)

    training_auc_sample = None
    if config.get("record_training_auc", False):
        # Complete weeks from the same fixed subset used by the XGBoost diagnostic.
        sample = train.sample(n=min(256, len(train)), random_state=42).sort_values("starttime")
        positions = train.index.get_indexer(sample.index)
        case_rows = (positions[:, None] * HOURS + np.arange(HOURS)).ravel()
        rows = np.concatenate([g * len(train) * HOURS + case_rows for g in range(len(generators))])
        sample_probabilities = expit(selected_booster.predict(x_train.iloc[rows], raw_score=True))
        sample_score = score(sample[["Run No", *label_columns]], prediction_frame(
            sample_probabilities, sample, label_columns, generators
        ))
        training_auc_sample = {
            "micro_roc_auc": sample_score, "cases": len(sample),
            "outputs": len(rows), "seed": 42, "run_ids": sample["Run No"].tolist(),
        }
        print(f"Training sample ({len(sample)} cases) micro ROC-AUC: {sample_score:.6f}", flush=True)

    selected_booster.save_model(str(model_path))
    validation_predictions.to_csv(predictions_path, index=False, mode="x")
    metrics = {
        "config": config,
        "training_cases": len(train),
        "first_training_start": str(train["starttime"].min()),
        "training_case_ids_sha256": training_case_hash(train),
        "validation_cases": len(validation),
        "purged_cases": purged,
        "last_training_start": str(train["starttime"].max()),
        "first_validation_start": str(validation["starttime"].min()),
        "last_validation_start": str(validation["starttime"].max()),
        "frequency_micro_roc_auc": frequency_score,
        "lightgbm_micro_roc_auc": model_score,
        "best_iteration": selected_booster.current_iteration(),
        "iteration_selection": "fixed_by_config" if stopping_rounds is None else "validation_early_stopping",
        "training_seconds": training_seconds,
        "training_auc_sample": training_auc_sample,
        "feature_names": x_train.columns.tolist(),
        "model_path": str(model_path),
        "continuation": continuation,
    }
    with report_path.open("x") as file:
        file.write(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
