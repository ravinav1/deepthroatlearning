"""Fit on all supplied training cases and create a competition CSV."""

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
from scipy.special import expit

from features import ROOT, build_features, flatten_labels, load_data, prediction_frame
from tree_preprocessing import rescale_tree_features
from training_weights import recency_fit_weights
from boundary_weights import boundary_fit_weights
from train_baseline import prepare_continuation, training_case_hash


def main():
    parser = argparse.ArgumentParser(description="Train the final BCE model and create a submission.")
    parser.add_argument("--config", default=str(ROOT / "configs/submission_topology.json"))
    parser.add_argument("--output", default=str(ROOT / "submissions/submission_topology.csv"))
    args = parser.parse_args()
    output = Path(args.output)
    with open(args.config) as file:
        config = json.load(file)
    assert config.get("loss", "binary") == "binary", "This submission uses the selected BCE model."
    name = Path(args.config).stem
    model_type = config.get("model_type", "lightgbm")
    assert config.get("boundary_weighting") is None or model_type == "xgboost", \
        "Boundary weighting is implemented for XGBoost only."
    assert not config.get("init_model") or model_type == "lightgbm", \
        "Saved-model continuation is implemented for LightGBM only."
    extension = {"xgboost": "ubj", "catboost": "cbm", "lightgbm": "txt"}[model_type]
    model_path = ROOT / f"models/{model_type}_{name}.{extension}"
    report_path = ROOT / f"reports/{name}_run.json"
    for path in (output, model_path, report_path):
        assert not path.exists(), f"Artifact already exists: {path}; choose a new config or output name."
    options = {name: config.get(name, False)
               for name in ["use_extended", "use_system_context", "use_topology", "use_price_context", "use_water_context", "use_economic_context", "use_river_context",
                            "use_inflow_profiles", "use_hydraulic_context", "use_water_budget",
                            "use_temporal_ranks", "use_favourable_periods"]}
    train, columns, generators, inputs, topology = load_data(use_extended=options["use_extended"])
    template = pd.read_csv(ROOT / "data/raw/sample_submission.csv")
    mapping = pd.read_csv(ROOT / "data/raw/prediction_mapping.csv")
    assert set(mapping["Run No"]) == set(template["Run No"])
    test = mapping.set_index("Run No").loc[template["Run No"]].reset_index()
    test["starttime"] = pd.to_datetime(test["starttime"], utc=True)
    assert not train["Run No"].isin(test["Run No"]).any(), "Test cases must not enter training."
    tree_limit = config["model"].get("n_estimators", config["model"].get("iterations"))
    print(f"Cases: {len(train)} training, {len(test)} test; trees: {tree_limit}", flush=True)

    x_train = build_features(train, generators, inputs, topology, **options)
    if config.get("use_economic_rescaling", False):
        x_train = rescale_tree_features(x_train)
    labels = flatten_labels(train, columns, generators)
    training_weight_summary = None
    continuation = None
    # Fixed tree count chosen using validation; no test labels or test-based early stopping.
    if model_type == "xgboost":
        x_train["generator_id"] = pd.Categorical(x_train["generator_id"], categories=range(len(generators)))
        model = xgb.XGBClassifier(objective="binary:logistic", enable_categorical=True, **config["model"])
        reference_date = train["starttime"].max().normalize() + pd.Timedelta(days=1)
        fit_kwargs, training_weight_summary = recency_fit_weights(
            train, len(generators), config.get("recency_half_life_years"), reference_date
        )
        if config.get("boundary_weighting") is not None:
            fit_kwargs, training_weight_summary = boundary_fit_weights(x_train, config)
        model.fit(x_train, labels, **fit_kwargs)
    elif model_type == "catboost":
        x_train["generator_id"] = x_train["generator_id"].astype("int32")
        model = CatBoostClassifier(loss_function="Logloss", allow_writing_files=False, **config["model"])
        model.fit(x_train, labels, cat_features=["generator_id"], verbose=100)
    else:
        assert model_type == "lightgbm"
        initial, model_params, continuation = prepare_continuation(config, x_train, train)
        model = lgb.LGBMClassifier(objective="binary", **model_params)
        model.fit(x_train, labels, categorical_feature=["generator_id"], init_model=initial)
        if continuation is not None:
            continuation["selected_total_rounds"] = model.booster_.current_iteration()
            assert continuation["selected_total_rounds"] == config["model"]["n_estimators"]
    del x_train
    print("Training complete; predicting test cases...", flush=True)
    x_test = build_features(test, generators, inputs, topology, **options)
    if config.get("use_economic_rescaling", False):
        x_test = rescale_tree_features(x_test)
    if model_type == "xgboost":
        x_test["generator_id"] = pd.Categorical(x_test["generator_id"], categories=range(len(generators)))
        probabilities = model.predict_proba(x_test)[:, 1]
    elif model_type == "catboost":
        x_test["generator_id"] = x_test["generator_id"].astype("int32")
        probabilities = model.predict_proba(x_test)[:, 1]
    else:
        probabilities = expit(model.predict(x_test, raw_score=True))
    submission = prediction_frame(probabilities, test, columns, generators)
    assert submission.columns.tolist() == template.columns.tolist()
    assert np.array_equal(submission["Run No"], template["Run No"])
    assert submission.shape == template.shape
    assert np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all()

    if model_type == "xgboost":
        model.save_model(model_path)
        trees = model.get_booster().num_boosted_rounds()
    elif model_type == "catboost":
        model.save_model(model_path)
        trees = model.tree_count_
    else:
        model.booster_.save_model(str(model_path))
        trees = model.booster_.current_iteration()
    submission.to_csv(output, index=False, mode="x")
    report = {
        "config": config,
        "training_cases": len(train),
        "training_case_ids_sha256": training_case_hash(train),
        "first_training_start": str(train["starttime"].min()),
        "last_training_start": str(train["starttime"].max()),
        "test_cases": len(test),
        "first_test_start": str(test["starttime"].min()),
        "last_test_start": str(test["starttime"].max()),
        "trees": trees,
        "training_weight_summary": training_weight_summary,
        "selection_source": config.get("selection_source"),
        "features": x_test.columns.tolist(),
        "submission": str(output),
        "model_path": str(model_path),
        "continuation": continuation,
    }
    with report_path.open("x") as file:
        file.write(json.dumps(report, indent=2) + "\n")
    print(f"Saved {submission.shape[0]} rows and {submission.shape[1]} columns to {output}", flush=True)


if __name__ == "__main__":
    main()
