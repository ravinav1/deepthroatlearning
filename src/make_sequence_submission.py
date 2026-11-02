"""Fit the selected sequence model and predict the competition cases."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from features import HOURS, ROOT, load_data, prediction_frame
from temporal_cnn import TemporalCNN, predict, sequence_features
from weekly_transformer import WeeklyTransformer
from weekly_mixer import WeeklyMixer
from bidirectional_lstm import BidirectionalLSTM
from sequence_preprocessing import fit_sequence_scaler
from training_weights import recency_case_weights
from generator_specialist import output_targets
from consistency_regularization import normalized_generator_weights, sequence_training_loss


def selected_learning_rates(config, selection):
    """Freeze the validation run's epoch rates for the final all-labelled fit."""
    if "lr_scheduler" in config:
        assert config["lr_scheduler"] == selection["config"].get("lr_scheduler"), \
            "Scheduler must match the selected validation run."
    if "lr_scheduler" not in selection["config"]:
        return None
    rows = selection["history"][:config["epochs"]]
    assert len(rows) == config["epochs"] and all(
        row["epoch"] == epoch and "learning_rate" in row
        for epoch, row in enumerate(rows, 1)
    ), "The selected run must record a learning rate for every fitted epoch."
    rates = [row["learning_rate"] for row in rows]
    assert rates[0] == config["learning_rate"] and all(np.isfinite(rate) and rate > 0 for rate in rates), \
        "Recorded learning rates must be positive and start at the configured rate."
    return rates


def main():
    parser = argparse.ArgumentParser(description="Train the fixed-epoch sequence model and create a submission.")
    parser.add_argument("--config", default=str(ROOT / "configs/submission_cnn.json"))
    parser.add_argument("--output")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    rdrop_weight = config.get("rdrop_weight", 0.0)
    assert rdrop_weight >= 0, "R-Drop weight must be nonnegative."
    pairwise_weight = config.get("pairwise_weight", 0.0)
    pairwise_pairs = config.get("pairwise_pairs", 4096)
    assert pairwise_weight >= 0 and pairwise_pairs > 0, "Pairwise weight must be nonnegative and pair count positive."
    assert pairwise_weight == 0 or config.get("recency_half_life_years") is None, \
        "Sampled pairwise loss currently requires unweighted training cases."
    transition_weight = config.get("transition_weight", 0.0)
    assert transition_weight >= 0, "Transition weight must be nonnegative."
    name = Path(args.config).stem
    target_generator = config.get("target_generator")
    component_only = target_generator is not None or config.get("plant_loss_weights") is not None
    default_output = ROOT / (f"data/processed/{name}_predictions.csv" if component_only
                             else "submissions/submission_cnn.csv")
    output = Path(args.output) if args.output is not None else default_output
    if component_only:
        assert output.parent.resolve() == (ROOT / "data/processed").resolve(), \
            "Specialist and generator-weighted predictions are components; save them in data/processed."
    model_path = ROOT / f"models/{name}.pt"
    report_path = ROOT / f"reports/{name}_run.json"
    for path in (output, model_path, report_path):
        assert not path.exists(), f"Artifact already exists: {path}; choose a new config or output name."
    selection = json.loads((ROOT / config["selection_source"]).read_text())
    assert config["epochs"] == selection["best_epoch"], "Epoch count must match the selected validation run."
    assert target_generator == selection["config"].get("target_generator"), \
        "Target generator must match the selected sequence model."
    model_type = config.get("model_type", "cnn")
    model_classes = {"cnn": TemporalCNN, "transformer": WeeklyTransformer,
                     "bilstm": BidirectionalLSTM, "mixer": WeeklyMixer}
    assert model_type == selection["config"].get("model_type", "cnn") and model_type in model_classes, \
        "This exporter fits the selected CNN, Transformer, BiLSTM or Mixer architecture."
    for key in ["seed", "batch_size", "learning_rate", "model"]:
        assert config[key] == selection["config"][key], f"Selected sequence model differs for {key}."
    assert config.get("weight_decay", 1e-4) == selection["config"].get("weight_decay", 1e-4), \
        "Weight decay must match the selected sequence model."
    options = {key: config.get(key, False) for key in [
        "use_extended", "use_system_context", "use_topology", "use_price_context",
        "use_water_context", "use_economic_context", "use_river_context",
        "use_inflow_profiles", "use_hydraulic_context", "use_water_budget", "use_weekly_normalization",
        "use_economic_rescaling",
    ]}
    assert all(value == selection["config"].get(key, False) for key, value in options.items())
    assert config.get("stable_scaling", False) == selection["config"].get("stable_scaling", False), \
        "Scaler must match the selected sequence model."
    assert config.get("recency_half_life_years") == selection["config"].get("recency_half_life_years"), \
        "Recency weights must match the selected sequence model."
    assert rdrop_weight == selection["config"].get("rdrop_weight", 0.0), \
        "R-Drop weight must match the selected sequence model."
    assert pairwise_weight == selection["config"].get("pairwise_weight", 0.0), \
        "Pairwise weight must match the selected sequence model."
    assert pairwise_pairs == selection["config"].get("pairwise_pairs", 4096), \
        "Pair count must match the selected sequence model."
    assert transition_weight == selection["config"].get("transition_weight", 0.0), \
        "Transition weight must match the selected sequence model."
    assert config.get("plant_loss_weights") == selection["config"].get("plant_loss_weights"), \
        "Plant loss weights must match the selected sequence model."
    learning_rates = selected_learning_rates(config, selection)
    torch.manual_seed(config["seed"])
    torch.set_num_threads(4)
    device = torch.device(config.get("device", "mps" if torch.backends.mps.is_available() else "cpu"))
    train, columns, generators, inputs, topology = load_data(use_extended=options["use_extended"])
    output_columns, output_generators = output_targets(columns, generators, target_generator)
    generator_weights = normalized_generator_weights(config.get("plant_loss_weights"), output_generators, device)
    if generator_weights is not None:
        assert dict(zip(output_generators, generator_weights.cpu().tolist())) == selection["generator_loss_weights"], \
            "Normalized generator loss weights must match the selected output order."
    if target_generator is not None:
        assert selection["target_generator"] == target_generator
        assert selection["output_columns"] == output_columns
        assert selection["output_generators"] == output_generators
        assert selection["input_generators"] == generators
    template = pd.read_csv(ROOT / "data/raw/sample_submission.csv")
    if target_generator is not None:
        template = template[["Run No", *output_columns]]
    mapping = pd.read_csv(ROOT / "data/raw/prediction_mapping.csv")
    assert mapping["Run No"].is_unique and template["Run No"].is_unique
    assert set(mapping["Run No"]) == set(template["Run No"])
    test = mapping.set_index("Run No").loc[template["Run No"]].reset_index()
    test["starttime"] = pd.to_datetime(test["starttime"], utc=True)
    assert not train["Run No"].isin(test["Run No"]).any(), "Test cases must not enter training."
    print(f"Device: {device}; cases: {len(train)} training, {len(test)} test; epochs: {config['epochs']}", flush=True)

    x_train, names = sequence_features(train, generators, inputs, topology, **options)
    assert names == selection["feature_names"], "Input channels must match the selected sequence model."
    # Match validation preprocessing, fitted only on supplied training cases.
    mean, std = fit_sequence_scaler(x_train, stable=config.get("stable_scaling", False))
    x_train = torch.from_numpy((x_train - mean) / std)
    labels = train[output_columns].to_numpy(dtype="float32", copy=True).reshape(-1, len(output_generators), HOURS)
    y_train = torch.from_numpy(labels)
    case_weights, weight_summary = recency_case_weights(
        train, config.get("recency_half_life_years"), train.starttime.max() + pd.Timedelta(days=1)
    )
    training_tensors = (x_train, y_train)
    if case_weights is not None:
        training_tensors += (torch.from_numpy(case_weights),)
    loader = DataLoader(TensorDataset(*training_tensors), batch_size=config["batch_size"], shuffle=True)
    model = model_classes[model_type](len(names), len(output_generators), **config["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config.get("weight_decay", 1e-4))
    history = []
    started = time.monotonic()
    # Epoch count is fixed from historical validation; no test labels or early stopping.
    for epoch in range(1, config["epochs"] + 1):
        if learning_rates is not None:
            for group in optimizer.param_groups:
                group["lr"] = learning_rates[epoch - 1]
        model.train()
        total_loss = 0
        for batch in loader:
            x, y = batch[:2]
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            weights = batch[2].to(device) if case_weights is not None else None
            loss = sequence_training_loss(model, x, y, rdrop_weight, weights,
                                          pairwise_weight, pairwise_pairs, transition_weight, generator_weights)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(x)
        row = {"epoch": epoch, "training_loss": total_loss / len(train)}
        status = f"Epoch {epoch}: loss={total_loss / len(train):.5f}"
        if learning_rates is not None:
            row["learning_rate"] = learning_rates[epoch - 1]
            status += f", lr={learning_rates[epoch - 1]:g} (replayed)"
        history.append(row)
        print(status, flush=True)
    training_seconds = time.monotonic() - started
    del loader, x_train, y_train

    x_test, test_names = sequence_features(test, generators, inputs, topology, **options)
    assert names == test_names
    x_test = torch.from_numpy((x_test - mean) / std)
    probabilities = predict(model, x_test, device, config["batch_size"])
    submission = prediction_frame(probabilities.transpose(1, 0, 2).ravel(), test, output_columns, output_generators)
    assert submission.columns.tolist() == template.columns.tolist()
    assert np.array_equal(submission["Run No"], template["Run No"])
    assert submission.shape == template.shape
    assert np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all()

    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    checkpoint = {"state_dict": state, "config": config, "feature_names": names,
                  "generators": generators, "mean": torch.from_numpy(mean), "std": torch.from_numpy(std)}
    if target_generator is not None:
        checkpoint.update({"target_generator": target_generator, "output_generators": output_generators,
                           "output_columns": output_columns})
    with model_path.open("xb") as file:
        torch.save(checkpoint, file)
    submission.to_csv(output, index=False, mode="x")
    report = {"config": config, "selection_source": config["selection_source"],
              "epochs": config["epochs"], "device": str(device), "training_seconds": training_seconds,
              "training_cases": len(train), "first_training_start": str(train.starttime.min()),
              "last_training_start": str(train.starttime.max()), "test_cases": len(test),
              "first_test_start": str(test.starttime.min()), "last_test_start": str(test.starttime.max()),
              "feature_names": names, "history": history,
              "model": str(model_path), "submission": str(output)}
    if target_generator is not None:
        report.update({"target_generator": target_generator, "output_columns": output_columns,
                       "output_generators": output_generators, "input_generators": generators})
    if weight_summary is not None:
        report["recency_weights"] = weight_summary
    if generator_weights is not None:
        report["generator_loss_weights"] = dict(zip(output_generators, generator_weights.cpu().tolist()))
        report["prediction_role"] = "ensemble_component"
    with report_path.open("x") as file:
        file.write(json.dumps(report, indent=2) + "\n")
    print(f"Saved {submission.shape[0]} rows and {submission.shape[1]} columns to {output}", flush=True)


if __name__ == "__main__":
    main()
