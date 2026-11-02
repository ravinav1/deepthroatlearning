"""Evaluate an existing Transformer in a later year without fitting anything."""

import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from features import HOURS, ROOT, load_data, prediction_frame
from temporal_cnn import predict, sequence_features
from weekly_transformer import WeeklyTransformer


def metrics(labels, probabilities, generators):
    return {
        "micro_roc_auc": float(roc_auc_score(labels.ravel(), probabilities.ravel())),
        "generator_auc": {
            generator: float(roc_auc_score(labels[:, index].ravel(), probabilities[:, index].ravel()))
            for index, generator in enumerate(generators)
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="models/transformer_rescaled_2021.pt")
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument("--prefix", default="round2_frozen_transformer")
    args = parser.parse_args()
    checkpoint_path = ROOT / args.checkpoint
    source_name = checkpoint_path.stem
    source_report_path = ROOT / f"reports/{source_name}_metrics.json"
    prediction_path = ROOT / f"data/processed/{args.prefix}_{args.year}_validation_predictions.csv"
    report_path = ROOT / f"reports/{args.prefix}_{args.year}_metrics.json"
    assert not prediction_path.exists() and not report_path.exists(), "Choose an unused output prefix."
    source_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    source_report = json.loads(source_report_path.read_text())
    torch.set_num_threads(1)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    assert config == source_report["config"] and config["model_type"] == "transformer"
    options = {key: config.get(key, False) for key in [
        "use_extended", "use_system_context", "use_topology", "use_price_context",
        "use_water_context", "use_economic_context", "use_river_context",
        "use_inflow_profiles", "use_hydraulic_context", "use_water_budget",
        "use_weekly_normalization", "use_economic_rescaling",
    ]}
    cases, columns, generators, inputs, topology = load_data(use_extended=options["use_extended"])
    assert generators == checkpoint["generators"]
    cutoff = pd.Timestamp(config["validation_start"], tz="UTC")
    source_train = cases.loc[cases.starttime + pd.Timedelta(hours=HOURS) <= cutoff]
    evaluation = cases.loc[cases.starttime.dt.year == args.year].copy()
    assert len(source_train) == source_report["training_cases"] and len(evaluation)
    assert str(source_train.starttime.max()) == source_report["last_training_start"]
    assert source_train.starttime.max() + pd.Timedelta(hours=HOURS) <= evaluation.starttime.min()
    assert not source_train["Run No"].isin(evaluation["Run No"]).any()
    competition_ids = pd.read_csv(ROOT / "data/raw/prediction_mapping.csv", usecols=["Run No"])["Run No"]
    assert not cases["Run No"].isin(competition_ids).any()
    assert config.get("validation_end") is not None, "The source must have a bounded selection period."
    assert pd.Timestamp(config["validation_end"], tz="UTC") <= evaluation.starttime.min()

    started = time.perf_counter()
    features, names = sequence_features(evaluation, generators, inputs, topology, **options)
    assert names == checkpoint["feature_names"] == source_report["feature_names"]
    mean, std = checkpoint["mean"].numpy(), checkpoint["std"].numpy()
    features = torch.from_numpy((features - mean) / std)
    model = WeeklyTransformer(len(names), len(generators), **config["model"])
    model.load_state_dict(checkpoint["state_dict"])
    probabilities = predict(model, features, torch.device("cpu"), config["batch_size"])
    assert np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all()
    labels = evaluation[columns].to_numpy().reshape(-1, len(generators), HOURS)
    result = metrics(labels, probabilities, generators)
    predictions = prediction_frame(probabilities.transpose(1, 0, 2).ravel(), evaluation, columns, generators)
    assert predictions["Run No"].tolist() == evaluation["Run No"].tolist()
    assert predictions.columns.tolist() == ["Run No", *columns]
    inference_seconds = time.perf_counter() - started

    source_validation = cases.loc[
        (cases.starttime >= cutoff)
        & (cases.starttime < pd.Timestamp(config["validation_end"], tz="UTC"))
    ]
    source_prediction_path = ROOT / f"data/processed/{source_name}_validation_predictions.csv"
    source_predictions = pd.read_csv(source_prediction_path)
    assert source_predictions.columns.tolist() == ["Run No", *columns]
    assert source_predictions["Run No"].is_unique
    assert set(source_predictions["Run No"]) == set(source_validation["Run No"])
    source_values = source_predictions.set_index("Run No").loc[source_validation["Run No"], columns]
    source_metrics = metrics(
        source_validation[columns].to_numpy().reshape(-1, len(generators), HOURS),
        source_values.to_numpy().reshape(-1, len(generators), HOURS), generators,
    )
    assert abs(source_metrics["micro_roc_auc"] - source_report["transformer_micro_roc_auc"]) < 1e-12
    selection_horizon_end = source_validation.starttime.max() + pd.Timedelta(hours=HOURS)
    after_selection = (evaluation.starttime >= selection_horizon_end).to_numpy()
    after_selection_metrics = metrics(labels[after_selection], probabilities[after_selection], generators)
    assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == source_hash
    report = {
        "source_checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "source_checkpoint_sha256": source_hash,
        "source_report": str(source_report_path.relative_to(ROOT)),
        "source_best_epoch_selected_on_2021": source_report["best_epoch"],
        "source_training_cases": len(source_train),
        "last_source_training_start": str(source_train.starttime.max()),
        "last_source_training_horizon_end": str(source_train.starttime.max() + pd.Timedelta(hours=HOURS)),
        "source_validation_start": config["validation_start"],
        "source_validation_end": config["validation_end"],
        "last_source_selection_horizon_end": str(selection_horizon_end),
        "evaluation_cases_overlapping_source_selection_horizons": int((~after_selection).sum()),
        "evaluation_year": args.year,
        "evaluation_cases": len(evaluation),
        "first_evaluation_start": str(evaluation.starttime.min()),
        "last_evaluation_start": str(evaluation.starttime.max()),
        "device": "cpu", "cpu_threads": 1, "inference_seconds": inference_seconds,
        "new_fitting": False, "scaler_refitted": False,
        "competition_ids_excluded": True, "source_checkpoint_unchanged": True,
        "feature_channels": len(names),
        "source_validation": source_metrics,
        "frozen_later_year": result,
        "frozen_later_year_after_source_selection_horizon": {
            "cases": int(after_selection.sum()),
            "first_evaluation_start": str(evaluation.loc[after_selection, "starttime"].min()),
            **after_selection_metrics,
        },
        "delta_auc_later_year_vs_source_validation": result["micro_roc_auc"] - source_metrics["micro_roc_auc"],
        "generator_auc_delta_later_year_vs_source_validation": {
            generator: result["generator_auc"][generator] - source_metrics["generator_auc"][generator]
            for generator in generators
        },
        "prediction_path": str(prediction_path.relative_to(ROOT)),
        "note": "Inference only from the existing checkpoint and its stored training scaler. The 2022 curve does not select weights or epochs here. The first six 2022 case horizons overlap source checkpoint-selection horizons, so an additional subset starts after the last selection horizon. No evaluation case overlaps source training. Both years have been reused heavily for development; this is not an untouched holdout. Cross-year differences also reflect case difficulty and distribution shifts.",
    }
    predictions.to_csv(prediction_path, index=False, mode="x")
    with report_path.open("x") as file:
        file.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"source_validation": source_metrics["micro_roc_auc"],
                      "frozen_later_year": result["micro_roc_auc"],
                      "inference_seconds": inference_seconds, "report": str(report_path.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
