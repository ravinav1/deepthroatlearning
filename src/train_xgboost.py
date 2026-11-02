"""Run from the project root: python src/train_xgboost.py"""

import argparse
import json
import sys
import time
from pathlib import Path

import xgboost as xgb
import numpy as np
import pandas as pd

from features import HOURS, ROOT, build_features, flatten_labels, load_data, prediction_frame
from tree_preprocessing import rescale_tree_features
from training_weights import recency_fit_weights
from boundary_weights import boundary_fit_weights

sys.path.insert(0, str(ROOT / "evaluation"))
from kaggle_metric import score


def main():
    parser = argparse.ArgumentParser(description="Train and validate XGBoost with the shared features.")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs/xgboost_2022.json"))
    args = parser.parse_args()
    with open(args.config) as file:
        config = json.load(file)
    run_name = Path(args.config).stem
    model_path = ROOT / f"models/{run_name}.ubj"
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
    use_favourable_periods = config.get("use_favourable_periods", False)
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

    fit_kwargs, training_weight_summary = recency_fit_weights(
        train, len(generators), config.get("recency_half_life_years"), validation_start
    )

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
                             use_temporal_ranks=use_temporal_ranks, use_favourable_periods=use_favourable_periods)
    x_validation = build_features(validation, generators, inputs, topology,
                                  use_extended=use_extended, use_system_context=use_system_context,
                                  use_topology=use_topology, use_price_context=use_price_context,
                                  use_water_context=use_water_context, use_economic_context=use_economic_context,
                                  use_river_context=use_river_context, use_inflow_profiles=use_inflow_profiles,
                                  use_hydraulic_context=use_hydraulic_context, use_water_budget=use_water_budget,
                                  use_temporal_ranks=use_temporal_ranks, use_favourable_periods=use_favourable_periods)
    if config.get("use_economic_rescaling", False):
        x_train = rescale_tree_features(x_train)
        x_validation = rescale_tree_features(x_validation)
    if config.get("boundary_weighting") is not None:
        fit_kwargs, training_weight_summary = boundary_fit_weights(x_train, config)
    print(f"Training on {len(x_train):,} rows and {x_train.shape[1]} features...", flush=True)
    # Keep generator identities categorical, just as in the LightGBM run.
    for features in (x_train, x_validation):
        features["generator_id"] = pd.Categorical(
            features["generator_id"], categories=range(len(generators))
        )
    stopping_rounds = config.get("early_stopping_rounds", 100)
    callbacks = ([xgb.callback.EarlyStopping(rounds=stopping_rounds, save_best=True)]
                 if stopping_rounds else [])
    model = xgb.XGBClassifier(
        objective="binary:logistic", eval_metric="auc", enable_categorical=True,
        callbacks=callbacks,
        **config["model"],
    )
    started = time.monotonic()
    model.fit(x_train, y_train, eval_set=[(x_validation, y_validation)], verbose=25, **fit_kwargs)
    training_seconds = time.monotonic() - started
    probabilities = model.predict_proba(x_validation)[:, 1]
    validation_predictions = prediction_frame(probabilities, validation, label_columns, generators)
    model_score = score(truth, validation_predictions)
    print(f"XGBoost micro ROC-AUC: {model_score:.6f}", flush=True)

    training_auc_sample = None
    if config.get("record_training_auc", False):
        # Score complete weeks from a fixed subset; do not add a costly per-tree train metric.
        sample = train.sample(n=min(256, len(train)), random_state=42).sort_values("starttime")
        positions = train.index.get_indexer(sample.index)
        case_rows = (positions[:, None] * HOURS + np.arange(HOURS)).ravel()
        rows = np.concatenate([g * len(train) * HOURS + case_rows for g in range(len(generators))])
        sample_probabilities = model.predict_proba(x_train.iloc[rows])[:, 1]
        sample_score = score(sample[["Run No", *label_columns]], prediction_frame(
            sample_probabilities, sample, label_columns, generators
        ))
        training_auc_sample = {
            "micro_roc_auc": sample_score, "cases": len(sample),
            "outputs": len(rows), "seed": 42, "run_ids": sample["Run No"].tolist(),
        }
        print(f"Training sample ({len(sample)} cases) micro ROC-AUC: {sample_score:.6f}", flush=True)

    model.save_model(model_path)
    validation_predictions.to_csv(predictions_path, index=False, mode="x")
    metrics = {
        "config": config,
        "training_cases": len(train),
        "validation_cases": len(validation),
        "purged_cases": purged,
        "last_training_start": str(train["starttime"].max()),
        "first_validation_start": str(validation["starttime"].min()),
        "last_validation_start": str(validation["starttime"].max()),
        "frequency_micro_roc_auc": frequency_score,
        "xgboost_micro_roc_auc": model_score,
        "best_iteration": model.get_booster().num_boosted_rounds(),
        "iteration_selection": "validation_early_stopping" if stopping_rounds else "fixed_by_config",
        "training_seconds": training_seconds,
        "training_weight_summary": training_weight_summary,
        "training_auc_sample": training_auc_sample,
        "feature_names": x_train.columns.tolist(),
    }
    with report_path.open("x") as file:
        file.write(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
