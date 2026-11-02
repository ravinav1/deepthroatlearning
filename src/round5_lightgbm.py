"""Separate LightGBM classifiers per generator, with optional hourly sampling.

Train: --config configs/round5_lgb_separate_2021.json
Frozen inference: --source-report reports/<run>_metrics.json --predict-year 2022 --name round5_frozen_<run>_2022
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from features import ROOT, HOURS, build_features, load_data, prediction_frame
from tree_preprocessing import rescale_tree_features

sys.path.insert(0, str(ROOT / "evaluation"))
from kaggle_metric import score

OPTIONS = ("use_extended", "use_system_context", "use_topology", "use_price_context",
           "use_water_context", "use_economic_context", "use_river_context", "use_inflow_profiles",
           "use_hydraulic_context", "use_water_budget", "use_temporal_ranks", "use_favourable_periods")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generator_features(cases, generator, generator_id, inputs, topology, config):
    options = {key: config.get(key, False) for key in OPTIONS}
    features = build_features(cases, [generator], inputs, topology, **options)
    # build_features([one generator]) numbers it zero; retain the global ID.
    features["generator_id"] = np.float32(generator_id)
    if config.get("use_shared_river_constraint_timing") or config.get("use_distinct_inflow_timing"):
        from scheduling_context import shared_river_constraint_context, distinct_inflow_timing

        starts = pd.DatetimeIndex(cases.starttime)
        timestamps = pd.DatetimeIndex(np.repeat(starts.array, HOURS)) + pd.to_timedelta(
            np.tile(np.arange(HOURS), len(starts)), unit="h")
        context = {}
        if config.get("use_shared_river_constraint_timing"):
            context.update(shared_river_constraint_context(starts, timestamps, inputs))
        if config.get("use_distinct_inflow_timing"):
            context.update(distinct_inflow_timing(starts, timestamps, inputs))
        for column, values in context.items():
            features[column] = np.asarray(values, dtype=np.float32)
    if config.get("use_economic_rescaling", False):
        features = rescale_tree_features(features)
    return features


def training_rows(cases, stride, seed):
    # Compute complete weekly features first. Only labels/training rows are thinned.
    offsets = (cases["Run No"].to_numpy(dtype=np.int64) + seed) % stride
    keep = np.arange(HOURS)[None, :] % stride == offsets[:, None]
    return np.flatnonzero(keep.ravel())


def write_json(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--source-report")
    parser.add_argument("--predict-year", type=int)
    parser.add_argument("--name")
    args = parser.parse_args()
    frozen = args.source_report is not None
    assert bool(args.config) != frozen
    source = json.loads((ROOT / args.source_report).read_text()) if frozen else None
    config = source["config"] if frozen else json.loads(Path(args.config).read_text())
    name = args.name if frozen else Path(args.config).stem
    assert name and name.startswith(("round5_", "round10_")) and Path(name).name == name
    full = not frozen and config.get("full_fit", False)
    report_path = ROOT / "reports" / f"{name}_{'run' if full else 'metrics'}.json"
    predictions_path = ROOT / "data/processed" / f"{name}_{'predictions' if full else 'validation_predictions'}.csv"
    model_dir = ROOT / "models" / name
    assert not report_path.exists() and not predictions_path.exists()
    if not frozen:
        assert not model_dir.exists()
        model_dir.mkdir()

    all_cases, columns, generators, inputs, topology = load_data(use_extended=config.get("use_extended", False))
    if "trees_by_generator" in config:
        assert set(config["trees_by_generator"]) == set(generators)
        assert not config.get("early_stopping_rounds"), "Selected tree counts must remain fixed."
    test = pd.read_csv(ROOT / "data/raw/prediction_mapping.csv")
    test["starttime"] = pd.to_datetime(test.starttime, utc=True)
    assert all_cases["Run No"].is_unique and test["Run No"].is_unique
    assert not all_cases["Run No"].isin(test["Run No"]).any()
    if full:
        train = all_cases
        ids = pd.read_csv(ROOT / "data/raw/sample_submission.csv", usecols=["Run No"])["Run No"]
        assert ids.is_unique and set(ids) == set(test["Run No"])
        evaluation = test.set_index("Run No").loc[ids].reset_index()
        purged = 0
    else:
        start = pd.Timestamp(f"{args.predict_year}-01-01", tz="UTC") if frozen else pd.Timestamp(config["validation_start"], tz="UTC")
        end = pd.Timestamp(f"{args.predict_year + 1}-01-01", tz="UTC") if frozen else pd.Timestamp(config["validation_end"], tz="UTC")
        train_end = pd.Timestamp(config["validation_start"], tz="UTC")
        train = all_cases.loc[all_cases.starttime + pd.Timedelta(hours=HOURS) <= train_end].copy()
        evaluation = all_cases.loc[(all_cases.starttime >= start) & (all_cases.starttime < end)].copy()
        purged = int(((all_cases.starttime < train_end) & (all_cases.starttime + pd.Timedelta(hours=HOURS) > train_end)).sum())
        assert (train.starttime + pd.Timedelta(hours=HOURS)).max() <= evaluation.starttime.min()
        if frozen:
            assert pd.Timestamp(config["validation_end"], tz="UTC") <= evaluation.starttime.min()
    train = train.reset_index(drop=True)
    evaluation = evaluation.reset_index(drop=True)
    cohort_hash = hashlib.sha256(train[["Run No", "starttime"]].to_csv(index=False).encode()).hexdigest()
    if frozen:
        assert cohort_hash == source["training_case_ids_sha256"]
        assert source["generators"] == generators
    # An archived fitting deadline must not expire later saved-model inference.
    deadline = datetime.fromisoformat(config["deadline_utc"]).timestamp() if config.get("deadline_utc") and not frozen else float("inf")
    monotonic_deadline = time.monotonic() + max(0, deadline - time.time())

    def check_deadline(env=None):
        if time.monotonic() >= monotonic_deadline:
            raise TimeoutError("Round5 hard deadline reached; completed generator checkpoints remain saved.")

    check_deadline.order = 0
    check_deadline.before_iteration = True
    stride = config.get("hour_stride", 1)
    assert stride in (1, 4)
    seed = config["model"]["random_state"]
    rows = training_rows(train, stride, seed)
    print(f"{name}: {len(train)} train cases; {len(evaluation)} evaluation cases; {len(rows)} rows/generator; stride={stride}", flush=True)
    results, probabilities = {}, []
    started = time.monotonic()
    for index, generator in enumerate(generators):
        check_deadline()
        target_columns = columns[index * HOURS:(index + 1) * HOURS]
        assert all(c.startswith(f"result_committed_{generator}_t") for c in target_columns)
        x_evaluation = generator_features(evaluation, generator, index, inputs, topology, config)
        if frozen:
            saved = source["generator_models"][generator]
            assert saved["generator_id"] == index and saved["output_columns"] == target_columns
            model_path = ROOT / saved["model_path"]
            assert digest(model_path) == saved["model_sha256"]
            booster = lgb.Booster(model_file=str(model_path))
            assert booster.feature_name() == saved["feature_names"] == x_evaluation.columns.tolist()
            assert booster.current_iteration() == saved["trees"]
            seconds = 0.0
        else:
            x_train = generator_features(train, generator, index, inputs, topology, config)
            y_train = train[target_columns].to_numpy(dtype=np.uint8).ravel()
            params = config["model"].copy()
            params["n_estimators"] = config.get("trees_by_generator", {}).get(generator, params["n_estimators"])
            model = lgb.LGBMClassifier(objective="binary", metric="auc", **params)
            callbacks = [check_deadline]
            evaluation_args = {}
            if not full:
                evaluation_args = dict(eval_X=x_evaluation, eval_y=evaluation[target_columns].to_numpy(dtype=np.uint8).ravel(), eval_metric="auc")
                if config.get("early_stopping_rounds"):
                    callbacks.append(lgb.early_stopping(config["early_stopping_rounds"], first_metric_only=True, verbose=False))
            before = time.monotonic()
            model.fit(x_train.iloc[rows], y_train[rows], categorical_feature=["generator_id"], callbacks=callbacks, **evaluation_args)
            seconds = time.monotonic() - before
            booster = model.booster_
            if full or not config.get("early_stopping_rounds"):
                assert booster.current_iteration() == params["n_estimators"]
            model_path = model_dir / f"{generator.replace(' ', '_')}.txt"
            booster.save_model(str(model_path))
            del x_train, y_train, model
        prediction = booster.predict(x_evaluation, num_threads=1)
        check_deadline()
        assert prediction.shape == (len(evaluation) * HOURS,)
        assert np.isfinite(prediction).all() and ((prediction >= 0) & (prediction <= 1)).all()
        probabilities.append(prediction)
        results[generator] = {"model_path": str(model_path.relative_to(ROOT)), "model_sha256": digest(model_path),
                              "trees": booster.current_iteration(), "feature_names": booster.feature_name(),
                              "generator_id": index, "training_seconds": seconds, "training_rows": len(rows),
                              "output_columns": target_columns}
        if not full:
            results[generator]["auc"] = roc_auc_score(evaluation[target_columns].to_numpy().ravel(), prediction)
        print(f"{index + 1}/14 {generator}: {booster.current_iteration()} trees, {seconds:.1f}s, AUC={results[generator].get('auc', 'test')}", flush=True)
    prediction = prediction_frame(np.concatenate(probabilities), evaluation, columns, generators)
    check_deadline()
    assert prediction.shape == (len(evaluation), 2353)
    assert prediction["Run No"].equals(evaluation["Run No"])
    prediction.to_csv(predictions_path, index=False, mode="x")
    report = {"config": config, "training_cases": len(train), "training_case_ids_sha256": cohort_hash,
              "first_training_start": str(train.starttime.min()), "last_training_start": str(train.starttime.max()),
              "evaluation_cases": len(evaluation), "first_evaluation_start": str(evaluation.starttime.min()),
              "last_evaluation_start": str(evaluation.starttime.max()), "purged_cases": purged,
              "generator_models": results, "generators": generators, "prediction_path": str(predictions_path.relative_to(ROOT)),
              "prediction_sha256": digest(predictions_path), "elapsed_seconds": time.monotonic() - started,
              "competition_training_id_overlap": 0, "sampling_rule": "Full weekly inputs, then hour % stride == (Run No + seed) % stride; all evaluation hours retained.",
              "fitting_performed": not frozen, "source_report": args.source_report}
    if not full:
        report["lightgbm_micro_roc_auc"] = score(evaluation[["Run No", *columns]], prediction)
    write_json(report_path, report)
    print(f"Saved {predictions_path}; micro AUC={report.get('lightgbm_micro_roc_auc', 'test')}", flush=True)


if __name__ == "__main__":
    main()
