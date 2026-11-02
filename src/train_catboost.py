"""Run from the project root: python src/train_catboost.py"""

import argparse
import json
import sys
from pathlib import Path

from catboost import CatBoostClassifier
import numpy as np
import pandas as pd

from features import HOURS, ROOT, build_features, flatten_labels, load_data, prediction_frame

sys.path.insert(0, str(ROOT / "evaluation"))
from kaggle_metric import score


def main():
    parser = argparse.ArgumentParser(description="Train and validate CatBoost with the shared features.")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs/catboost_2022.json"))
    args = parser.parse_args()
    with open(args.config) as file:
        config = json.load(file)
    run_name = Path(args.config).stem

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
                                  use_hydraulic_context=use_hydraulic_context, use_water_budget=use_water_budget)
    x_validation = build_features(validation, generators, inputs, topology,
                                  use_extended=use_extended, use_system_context=use_system_context,
                                  use_topology=use_topology, use_price_context=use_price_context,
                                  use_water_context=use_water_context, use_economic_context=use_economic_context,
                                  use_river_context=use_river_context, use_inflow_profiles=use_inflow_profiles,
                                  use_hydraulic_context=use_hydraulic_context, use_water_budget=use_water_budget)
    print(f"Training on {len(x_train):,} rows and {x_train.shape[1]} features...", flush=True)
    # CatBoost accepts integer category IDs, rather than float categories.
    for features in (x_train, x_validation):
        features["generator_id"] = features["generator_id"].astype("int32")
    model = CatBoostClassifier(
        loss_function="Logloss", eval_metric="AUC", allow_writing_files=False,
        **config["model"],
    )
    model.fit(
        x_train, y_train, cat_features=["generator_id"],
        eval_set=(x_validation, y_validation), use_best_model=True,
        early_stopping_rounds=config["early_stopping_rounds"], verbose=25,
    )
    probabilities = model.predict_proba(x_validation)[:, 1]
    validation_predictions = prediction_frame(probabilities, validation, label_columns, generators)
    model_score = score(truth, validation_predictions)
    print(f"CatBoost micro ROC-AUC: {model_score:.6f}", flush=True)

    model.save_model(ROOT / f"models/{run_name}.cbm")
    validation_predictions.to_csv(ROOT / f"data/processed/{run_name}_validation_predictions.csv", index=False)
    metrics = {
        "config": config,
        "training_cases": len(train),
        "validation_cases": len(validation),
        "purged_cases": purged,
        "last_training_start": str(train["starttime"].max()),
        "first_validation_start": str(validation["starttime"].min()),
        "last_validation_start": str(validation["starttime"].max()),
        "frequency_micro_roc_auc": frequency_score,
        "catboost_micro_roc_auc": model_score,
        "best_iteration": model.tree_count_,
        "feature_names": x_train.columns.tolist(),
    }
    (ROOT / f"reports/{run_name}_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
