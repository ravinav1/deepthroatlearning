"""Fit one generator's XGBoost, keeping the existing reservoir/price inputs.

Validation writes a full-width cached XGBoost prediction file with only the
target generator replaced. --full-fit instead writes 168 target probabilities
per competition case for later blending; it does not write a submission.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from features import HOURS, ROOT, build_features, flatten_labels, load_data, prediction_frame
from generator_specialist import output_targets
from plant_blend import read_predictions
from tree_preprocessing import rescale_tree_features


FEATURE_OPTIONS = (
    "use_extended", "use_system_context", "use_topology", "use_price_context",
    "use_water_context", "use_economic_context", "use_river_context",
    "use_inflow_profiles", "use_hydraulic_context", "use_water_budget",
)


def specialist_features(cases, target, inputs, topology, config):
    options = {name: config.get(name, False) for name in FEATURE_OPTIONS}
    result = build_features(cases, [target], inputs, topology, **options)
    if config.get("use_economic_rescaling", False):
        result = rescale_tree_features(result)
    # Generator identity is constant in a separate model; keep the same schema.
    result["generator_id"] = pd.Categorical(result["generator_id"], categories=[0])
    return result


def replace_target(reference, target_probabilities, columns, target_columns):
    indices = [columns.index(column) for column in target_columns]
    target = np.asarray(target_probabilities).reshape(len(reference), len(indices))
    assert np.isfinite(target).all() and ((target >= 0) & (target <= 1)).all()
    result = reference.copy()
    result[:, indices] = target
    untouched = [i for i in range(len(columns)) if i not in indices]
    assert np.array_equal(result[:, untouched], reference[:, untouched])
    return result


def frozen_selection(config, required):
    path = config.get("selection_source")
    if not path:
        assert not required, "Provide the 2021 specialist metrics as selection_source."
        return None
    path = ROOT / path
    source = json.loads(path.read_text())
    previous = source["config"]
    assert previous["target_generator"] == config["target_generator"]
    assert pd.Timestamp(previous["validation_start"]).year == 2021
    assert source["best_iteration"] == config["model"]["n_estimators"]
    for key in (*FEATURE_OPTIONS, "use_economic_rescaling"):
        assert previous.get(key, False) == config.get(key, False), f"Changed feature option: {key}"
    assert {k: v for k, v in previous["model"].items() if k != "n_estimators"} == {
        k: v for k, v in config["model"].items() if k != "n_estimators"
    }, "Confirmation/full fitting must preserve the selected specialist parameters."
    assert config.get("early_stopping_rounds") is None, "Use the frozen 2021 count without further stopping."
    return {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "selected_total_rounds": source["best_iteration"], "selection_year": 2021}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--full-fit", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    name = config_path.stem
    target = config["target_generator"]
    assert config.get("recency_half_life_years") is None and config.get("boundary_weighting") is None, \
        "This control changes only generator-specific capacity allocation."
    assert config["model"].get("n_jobs") == 4, "Keep the agreed four training threads."
    model_path = ROOT / f"models/{name}.ubj"
    report_path = ROOT / f"reports/{name}_{'run' if args.full_fit else 'metrics'}.json"
    prediction_path = ROOT / (f"data/processed/{name}_predictions.csv" if args.full_fit else
                              f"data/processed/{name}_validation_predictions.csv")
    for path in (model_path, report_path, prediction_path):
        assert not path.exists(), f"Artifact already exists: {path}"

    year = None if args.full_fit else pd.Timestamp(config["validation_start"]).year
    selection = frozen_selection(config, required=args.full_fit or year != 2021)
    cases, columns, generators, inputs, topology = load_data(use_extended=config.get("use_extended", False))
    target_columns, target_generators = output_targets(columns, generators, target)
    mapping = pd.read_csv(ROOT / "data/raw/prediction_mapping.csv")
    assert not cases["Run No"].isin(mapping["Run No"]).any(), "Competition cases must not enter training."
    if args.full_fit:
        train = cases.copy()
        template = pd.read_csv(ROOT / "data/raw/sample_submission.csv", usecols=["Run No"])
        predict_cases = mapping.set_index("Run No").loc[template["Run No"]].reset_index()
        predict_cases["starttime"] = pd.to_datetime(predict_cases.starttime, utc=True)
        purged = 0
    else:
        boundary = pd.Timestamp(config["validation_start"], tz="UTC")
        ends = cases.starttime + pd.Timedelta(hours=HOURS)
        train = cases.loc[ends <= boundary].copy()
        mask = cases.starttime >= boundary
        if config.get("validation_end"):
            mask &= cases.starttime < pd.Timestamp(config["validation_end"], tz="UTC")
        predict_cases = cases.loc[mask].copy().reset_index(drop=True)
        purged = int(((cases.starttime < boundary) & (ends > boundary)).sum())
        assert len(train) and len(predict_cases)
        assert (train.starttime + pd.Timedelta(hours=HOURS)).max() <= predict_cases.starttime.min()
    print(f"{target}: {len(train)} training cases, {len(predict_cases)} prediction cases, {purged} purged", flush=True)
    x_train = specialist_features(train, target, inputs, topology, config)
    y_train = flatten_labels(train, target_columns, target_generators)
    x_predict = specialist_features(predict_cases, target, inputs, topology, config)
    assert x_train.columns.tolist() == x_predict.columns.tolist()
    assert len(x_train) == len(train) * HOURS
    print(f"Training {len(x_train):,} rows, {x_train.shape[1]} unchanged features", flush=True)
    stopping = None if args.full_fit or selection is not None else config.get("early_stopping_rounds", 100)
    callbacks = [] if stopping is None else [xgb.callback.EarlyStopping(rounds=stopping, save_best=True)]
    model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="auc",
                              enable_categorical=True, callbacks=callbacks, **config["model"])
    fit_kwargs = {}
    if not args.full_fit:
        y_validation = flatten_labels(predict_cases, target_columns, target_generators)
        fit_kwargs = {"eval_set": [(x_predict, y_validation)], "verbose": 25}
    started = time.monotonic()
    model.fit(x_train, y_train, **fit_kwargs)
    training_seconds = time.monotonic() - started
    probabilities = model.predict_proba(x_predict)[:, 1]
    target_frame = prediction_frame(probabilities, predict_cases, target_columns, target_generators)
    trees = model.get_booster().num_boosted_rounds()
    if stopping is None:
        assert trees == config["model"]["n_estimators"]
    metrics = {
        "config": config, "target_generator": target, "output_generators": target_generators,
        "output_columns": target_columns,
        "training_cases": len(train), "training_rows": len(x_train),
        "training_case_ids_sha256": hashlib.sha256(train[["Run No", "starttime"]].to_csv(index=False).encode()).hexdigest(),
        "first_training_start": str(train.starttime.min()), "last_training_start": str(train.starttime.max()),
        "purged_cases": purged, "best_iteration": trees, "training_seconds": training_seconds,
        "iteration_selection": "fixed_by_config" if stopping is None else "target_generator_validation_early_stopping",
        "feature_names": x_train.columns.tolist(), "generator_id_encoding": "constant categorical 0",
        "selection_source": selection, "model_path": str(model_path.relative_to(ROOT)),
        "prediction_path": str(prediction_path.relative_to(ROOT)),
    }
    if config.get("record_training_auc", False):
        sample = train.sample(n=min(256, len(train)), random_state=42).sort_values("starttime")
        positions = train.index.get_indexer(sample.index)
        rows = (positions[:, None] * HOURS + np.arange(HOURS)).ravel()
        train_probability = model.predict_proba(x_train.iloc[rows])[:, 1]
        metrics["training_auc_sample"] = {
            "micro_roc_auc": float(roc_auc_score(y_train[rows], train_probability)),
            "cases": len(sample), "outputs": len(rows), "seed": 42, "run_ids": sample["Run No"].tolist(),
        }
    if args.full_fit:
        frame = target_frame
        metrics.update({"test_cases": len(predict_cases), "outputs_per_case": HOURS,
                        "note": "Partial target-generator predictions for later blending. All supplied labeled cases fitted at the frozen 2021 count; no competition labels or test-based stopping."})
    else:
        reference_path = ROOT / f"data/processed/{config['reference_tree_prefix']}_{year}_validation_predictions.csv"
        reference = read_predictions(reference_path, columns, predict_cases["Run No"])
        replacement = replace_target(reference, probabilities, columns, target_columns)
        frame = pd.DataFrame(replacement, columns=columns)
        frame.insert(0, "Run No", predict_cases["Run No"])
        indices = [columns.index(c) for c in target_columns]
        metrics.update({
            "validation_cases": len(predict_cases), "first_validation_start": str(predict_cases.starttime.min()),
            "last_validation_start": str(predict_cases.starttime.max()),
            "target_micro_roc_auc": float(roc_auc_score(y_validation, probabilities)),
            "reference_target_micro_roc_auc": float(roc_auc_score(y_validation, reference[:, indices].ravel())),
            "replacement_full_tree_micro_roc_auc": float(roc_auc_score(predict_cases[columns].to_numpy().ravel(), replacement.ravel())),
            "reference_prediction_path": str(reference_path.relative_to(ROOT)),
            "reference_prediction_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
            "untouched_generator_probabilities_identical": True,
            "note": "Only target-generator probabilities replace the current cached XGBoost component. All other generators are numerically unchanged. Evaluate the complete frozen ensemble with round2_compare.py. Historical years are reused development data.",
        })
        print(f"Target AUC {metrics['target_micro_roc_auc']:.6f}; current target {metrics['reference_target_micro_roc_auc']:.6f}", flush=True)
    model.save_model(model_path)
    frame.to_csv(prediction_path, index=False, mode="x")
    with report_path.open("x") as file:
        json.dump(metrics, file, indent=2)
        file.write("\n")
    print(f"Saved {prediction_path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
