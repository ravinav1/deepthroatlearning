"""Verify the optional round4 neural-only mean-slot submission without fitting.

Replay 17 fixed competition cases from the saved Transformer and stored scaler
on one CPU thread. Independently verify all mean-slot and final blend values.
"""

import argparse
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import sys

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"


class RejectTreeLibraries(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"xgboost", "lightgbm", "catboost"}:
            raise AssertionError(f"Transformer verification must not import {fullname}.")


sys.meta_path.insert(0, RejectTreeLibraries())
import numpy as np
import pandas as pd
import torch

from consistency_regularization import normalized_generator_weights
from features import ROOT, load_data, prediction_frame
from temporal_cnn import predict, sequence_features
from weekly_transformer import WeeklyTransformer


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads((ROOT / path).read_text())


def frame(path, ids, columns, partial=False):
    result = pd.read_csv(ROOT / path, float_precision="round_trip")
    assert result.columns[0] == "Run No" and result.columns.is_unique
    assert result["Run No"].is_unique and result["Run No"].equals(ids)
    provided = result.columns[1:].tolist()
    assert (provided and set(provided).issubset(columns)) if partial else provided == columns
    values = result[provided].to_numpy()
    assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
    return result


def independent_blend(parts, columns, weights):
    values = np.zeros((len(next(iter(parts.values()))), len(columns)), dtype=np.float64)
    for index, column in enumerate(columns):
        plant = column.removeprefix("result_committed_").rsplit("_G", 1)[0]
        amounts = weights[plant]
        coefficients = np.asarray(list(amounts.values()), dtype=float)
        assert np.isfinite(coefficients).all() and (coefficients >= 0).all()
        assert abs(coefficients.sum() - 1) < 1e-12
        for name, weight in amounts.items():
            values[:, index] += weight * parts[name][column].to_numpy()
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--submission", required=True)
    parser.add_argument("--export-report", help="Default: reports/<submission stem>_run.json")
    parser.add_argument("--output", default="reports/round4_final_verification.json")
    parser.add_argument("--protection-snapshot", type=Path,
                        default=Path("/private/tmp/tdt4173-round4-protection.json"))
    args = parser.parse_args()
    output = ROOT / args.output
    assert not output.exists(), f"Report already exists: {output}"
    config_path = "configs/round4_transformer_lr_full.json"
    model_path = "models/round4_transformer_lr_full.pt"
    prediction_path = "data/processed/round4_transformer_lr_full_predictions.csv"
    slot_path = "data/processed/round4_transformer_lr_slot_full_predictions.csv"
    run_path = "reports/round4_transformer_lr_full_run.json"
    selection_path = "reports/round4_transformer_lr_2021_metrics.json"
    decision_path = "reports/round4_neural_selection_2021.json"
    native_config_path = "configs/round4_transformer_lr_2022.json"
    native_report_path = "reports/round4_transformer_lr_2022_metrics.json"
    original_config_path = "configs/round2_transformer_task_weights_2021.json"
    baseline_path = "configs/round3_submission_temporal_ranks.json"
    export_path = args.export_report or f"reports/{Path(args.submission).stem}_run.json"
    config, run, recipe, exported, selection, decision, native, native_report, original, baseline = map(
        read, (config_path, run_path, args.recipe, export_path, selection_path, decision_path,
               native_config_path, native_report_path, original_config_path, baseline_path)
    )
    selected = selection["config"]
    assert run["config"] == config and native_report["config"] == native
    assert config["selection_source"] == run["selection_source"] == selection_path
    assert native["iteration_selection_source"] == selection_path
    assert decision["selected_arm"] == "lr" and decision["selected_variant"] == "candidate_mean"
    assert decision["selected_epoch"] == selection["best_epoch"] == native_report["best_epoch"] == 39
    assert config["epochs"] == run["epochs"] == native["epochs"] == 39
    assert selected["validation_start"] == "2021-01-01" and selected["validation_end"] == "2022-01-01"
    assert selected["early_stopping_metric"] == "ensemble"
    assert native["validation_start"] == "2022-01-01" and native["validation_end"] == "2023-01-01"
    assert native["early_stopping_metric"] == "fixed" and "ensemble_monitor" not in native
    assert all(key not in config for key in ("validation_start", "validation_end", "ensemble_monitor", "lr_scheduler"))
    for item in (config, selected, native):
        assert item["model_type"] == "transformer" and item["device"] == "cpu"
        assert item["learning_rate"] == .0005 and item.get("weight_decay", 1e-4) == .0001
        for key in ("seed", "batch_size", "model", "plant_loss_weights", "stable_scaling"):
            assert item[key] == original[key], f"Original Transformer training setting changed: {key}"
        assert item["seed"] == 42 and item["batch_size"] == 32
        assert item.get("target_generator") is None and item.get("recency_half_life_years") is None
        assert item.get("weight_averaging") is None and "lr_scheduler" not in item
        assert all(item.get(key, 0.0) == 0.0 for key in ("rdrop_weight", "pairwise_weight", "transition_weight"))
        assert not item.get("use_temporal_ranks", False) and not item.get("use_favourable_periods", False)
    options = {key: config.get(key, False) for key in (
        "use_extended", "use_system_context", "use_topology", "use_price_context", "use_water_context",
        "use_economic_context", "use_river_context", "use_inflow_profiles", "use_hydraulic_context",
        "use_water_budget", "use_weekly_normalization", "use_economic_rescaling")}
    assert all(value == item.get(key, False) for key, value in options.items()
               for item in (selected, native, original))
    assert config["stable_scaling"] and options["use_economic_rescaling"]
    assert run["device"] == "cpu" and run["prediction_role"] == "ensemble_component"
    assert [row["epoch"] for row in run["history"]] == list(range(1, 40))
    protected = {path: sha(ROOT / path) for path in (
        config_path, model_path, prediction_path, slot_path, run_path, args.recipe, args.submission, export_path,
        selection_path, decision_path, native_config_path, native_report_path, original_config_path, baseline_path,
        recipe["historical_report"], "data/raw/sample_submission.csv", "data/raw/prediction_mapping.csv",
        "data/raw/kernel/Unit_commitment_decisions.csv", "src/features.py", "src/temporal_cnn.py",
        "src/weekly_transformer.py", "src/sequence_preprocessing.py", "src/consistency_regularization.py",
        "src/make_sequence_submission.py", "src/make_goal_submission.py",
    )}
    for path, expected in decision["source_hashes"].items():
        assert sha(ROOT / path) == expected
        protected[path] = expected
    for item in (config, native):
        for key in ("selection_source", "iteration_selection_source", "combined_selection_source"):
            if item.get(key):
                path = item[key]
                protected[path] = sha(ROOT / path)
                if item.get(key + "_sha256"):
                    assert protected[path] == item[key + "_sha256"]

    template = pd.read_csv(ROOT / "data/raw/sample_submission.csv", float_precision="round_trip")
    columns, ids = template.columns[1:].tolist(), template["Run No"]
    assert template.shape == (725, 2353) and len(columns) == 2352 and ids.is_unique
    submission = frame(args.submission, ids, columns)
    cached = frame(prediction_path, ids, columns)
    slot = frame(slot_path, ids, columns)
    train, label_columns, generators, inputs, topology = load_data(use_extended=options["use_extended"])
    mapping = pd.read_csv(ROOT / "data/raw/prediction_mapping.csv")
    assert train["Run No"].is_unique and mapping["Run No"].is_unique
    assert set(mapping["Run No"]) == set(ids)
    test = mapping.set_index("Run No").loc[ids].reset_index()
    test["starttime"] = pd.to_datetime(test.starttime, utc=True)
    assert label_columns == columns and len(generators) == 14
    assert len(train) == run["training_cases"] == 2922 and len(test) == run["test_cases"] == 725
    assert not train["Run No"].isin(ids).any()
    for prefix, cases in (("training", train), ("test", test)):
        assert str(cases.starttime.min()) == run[f"first_{prefix}_start"]
        assert str(cases.starttime.max()) == run[f"last_{prefix}_start"]
    assert str(train.starttime.min()) == "2015-01-01 00:00:00+00:00"
    assert str(train.starttime.max()) == "2022-12-31 00:00:00+00:00"
    assert str(test.starttime.min()) == "2023-01-01 00:00:00+00:00"
    assert str(test.starttime.max()) == "2024-12-25 00:00:00+00:00"
    cohort_hash = hashlib.sha256(train[["Run No", "starttime"]].to_csv(index=False).encode()).hexdigest()
    if "training_case_ids_sha256" in run:
        assert cohort_hash == run["training_case_ids_sha256"]
    assert (ROOT / run["model"]).resolve() == (ROOT / model_path).resolve()
    assert (ROOT / run["submission"]).resolve() == (ROOT / prediction_path).resolve()

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    checkpoint = torch.load(ROOT / model_path, map_location="cpu", weights_only=False)
    assert checkpoint["config"] == config and checkpoint["generators"] == generators
    names = checkpoint["feature_names"]
    assert names == run["feature_names"] == selection["feature_names"] == native_report["feature_names"]
    assert len(names) == 306
    normalized = normalized_generator_weights(config["plant_loss_weights"], generators, torch.device("cpu"))
    loss_weights = dict(zip(generators, normalized.tolist()))
    assert loss_weights == run["generator_loss_weights"] == selection["generator_loss_weights"] == native_report["generator_loss_weights"]
    mean, std = checkpoint["mean"].numpy(), checkpoint["std"].numpy()
    assert mean.shape == std.shape == (1, len(names), 1)
    assert np.isfinite(mean).all() and np.isfinite(std).all() and (std > 0).all()
    model = WeeklyTransformer(len(names), len(generators), **config["model"])
    model.load_state_dict(checkpoint["state_dict"])
    sample_indices = np.linspace(0, len(test) - 1, 17, dtype=int)
    sample = test.iloc[sample_indices].reset_index(drop=True)
    features, observed_names = sequence_features(sample, generators, inputs, topology, **options)
    assert observed_names == names
    tensor = torch.from_numpy((features - mean) / std)
    probabilities = predict(model, tensor, torch.device("cpu"), config["batch_size"])
    assert np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all()
    replay = prediction_frame(probabilities.transpose(1, 0, 2).ravel(), sample, columns, generators)[columns].to_numpy()
    saved = cached.set_index("Run No").loc[sample["Run No"], columns].to_numpy()
    difference = float(np.abs(replay - saved).max())
    assert difference <= 2e-6, f"Stored-scaler Transformer replay differs by {difference}."

    old_path = baseline["sources"]["transformer_weighted"]
    old = frame(old_path, ids, columns)
    expected_slot = .5 * (old[columns].to_numpy() + cached[columns].to_numpy())
    assert np.array_equal(expected_slot, slot[columns].to_numpy()), "Weighted Transformer slot is not the fixed 50/50 mean."
    expected_sources = baseline["sources"].copy()
    expected_sources["transformer_weighted"] = slot_path
    assert recipe["plant_weights"] == baseline["plant_weights"]
    assert recipe["sources"] == expected_sources, "Only transformer_weighted may change from the current recipe."
    parts = {name: frame(path, ids, columns, partial=True) for name, path in recipe["sources"].items()}
    recomputed = independent_blend(parts, columns, recipe["plant_weights"])
    assert np.array_equal(recomputed, submission[columns].to_numpy())
    assert exported["config"] == recipe and exported["config_sha256"] == sha(ROOT / args.recipe)
    assert exported["submission_sha256"] == sha(ROOT / args.submission)
    assert (ROOT / exported["submission"]).resolve() == (ROOT / args.submission).resolve()
    assert exported["historical_source"] == recipe["historical_report"]
    assert exported["historical_sha256"] == sha(ROOT / recipe["historical_report"])
    assert exported["template_sha256"] == sha(ROOT / "data/raw/sample_submission.csv")
    assert exported["test_cases"] == 725 and exported["outputs_per_case"] == 2352
    assert set(exported["sources"]) == set(recipe["sources"])
    for name, path in recipe["sources"].items():
        assert exported["sources"][name]["path"] == path
        assert exported["sources"][name]["sha256"] == sha(ROOT / path)
        assert exported["sources"][name]["outputs_per_case"] == len(parts[name].columns) - 1
        protected[path] = sha(ROOT / path)
    protected[old_path] = sha(ROOT / old_path)
    history = read(recipe["historical_report"])
    confirmed = []
    for year in (2021, 2022):
        baseline_rows = [r for r in history["results"] if r["year"] == year and r["variant"] == "baseline"]
        chosen_rows = [r for r in history["results"] if r["year"] == year and r["variant"] == recipe["historical_variant"]]
        assert len(baseline_rows) == len(chosen_rows) == 1
        before, after = baseline_rows[0]["micro_roc_auc"], chosen_rows[0]["micro_roc_auc"]
        assert np.isfinite([before, after]).all() and 0 <= before < after <= 1
        confirmed.append({"year": year, "baseline_micro_roc_auc": before,
                          "selected_micro_roc_auc": after, "delta_vs_baseline": after - before})
    assert exported["historical_results"] == confirmed
    snapshot = json.loads(args.protection_snapshot.read_text())
    assert snapshot and baseline_path in snapshot
    # Processed caches are outside the original-artifact snapshot. Check this
    # cache against the provenance saved when the incumbent was exported.
    incumbent_export_path = "reports/submission_round3_temporal_ranks_run.json"
    incumbent_source = read(incumbent_export_path)["sources"]["transformer_weighted"]
    assert incumbent_source["path"] == old_path
    assert incumbent_source["sha256"] == sha(ROOT / old_path)
    protected[incumbent_export_path] = sha(ROOT / incumbent_export_path)
    changed = [path for path, expected in snapshot.items()
               if not (ROOT / path).exists() or sha(ROOT / path) != expected]
    assert not changed, changed
    assert not any(name in sys.modules for name in ("xgboost", "lightgbm", "catboost"))
    assert torch.get_num_threads() == torch.get_num_interop_threads() == 1
    assert all(sha(ROOT / path) == expected for path, expected in protected.items())
    report = {
        "passed": True, "submission": args.submission, "submission_sha256": sha(ROOT / args.submission),
        "cases": 725, "outputs_per_case": 2352, "verified_probabilities": int(recomputed.size),
        "full_blend_max_absolute_difference": 0.0, "mean_slot_max_absolute_difference": 0.0,
        "mean_slot_coefficients": {"original_weighted_transformer": .5, "new_weighted_transformer": .5},
        "checkpoint_replay_indices": sample_indices.tolist(), "checkpoint_replay_cases": sample["Run No"].tolist(),
        "checkpoint_replay_max_absolute_difference": difference, "checkpoint_replay_tolerance": 2e-6,
        "epochs": 39, "learning_rate": .0005, "weight_decay": .0001, "feature_channels": len(names),
        "selected_2021_and_native_2022_configuration_match": True, "original_features_and_task_weights_unchanged": True,
        "scaler_refitted": False, "scaler_mean_sha256": hashlib.sha256(mean.tobytes()).hexdigest(),
        "scaler_std_sha256": hashlib.sha256(std.tobytes()).hexdigest(), "training_cases": 2922,
        "training_cohort_sha256": cohort_hash, "competition_training_id_overlap": 0,
        "original_plant_coefficients_unchanged": True, "changed_component_sources": ["transformer_weighted"],
        "historical_export_linkage": confirmed,
        "protected_originals": {"snapshot": str(args.protection_snapshot), "snapshot_sha256": sha(args.protection_snapshot),
                                "files_checked": len(snapshot), "changed": changed},
        "artifact_hashes": protected, "verification_script_sha256": sha(__file__), "device": "cpu", "cpu_threads": 1,
        "scope": "All raw/slot/final CSV probabilities checked; all mean-slot and final blend values independently reconstructed; 17 fixed-case Transformer replays with stored scaler. Historical result/hash linkage checked without recomputing historical metrics. No fitting, tree-library imports or Kaggle upload.",
    }
    with output.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({key: report[key] for key in (
        "passed", "submission", "verified_probabilities", "checkpoint_replay_max_absolute_difference")}, indent=2))


if __name__ == "__main__":
    main()
