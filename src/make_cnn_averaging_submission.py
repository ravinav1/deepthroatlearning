"""Fit one CNN trajectory and export its selected averaged checkpoints."""

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
from sequence_preprocessing import fit_sequence_scaler
from training_weights import recency_case_weights
from weight_averaging import WeightAveraging


def main():
    parser = argparse.ArgumentParser(description="Train selected CNN averages on supplied labels.")
    parser.add_argument("--config", default=str(ROOT / "configs/submission_cnn_averaging.json"))
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    assert config.get("target_generator") is None, "Use make_sequence_submission.py for generator specialists."
    name = Path(args.config).stem
    methods = list(config["selection_sources"])
    assert methods and set(methods) <= {"ema", "swa"}
    configured_methods = config["weight_averaging"].get("methods", ["ema", "swa"])
    assert set(methods) <= set(configured_methods), "Every selected method must be enabled."
    paths = {
        method: (ROOT / f"models/{name}_{method}.pt",
                 ROOT / f"data/processed/{name}_{method}_predictions.csv",
                 ROOT / f"reports/{name}_{method}_run.json")
        for method in methods
    }
    for member_paths in paths.values():
        for path in member_paths:
            assert not path.exists(), f"Artifact already exists: {path}; choose a new config name."
    options = {key: config.get(key, False) for key in [
        "use_extended", "use_system_context", "use_topology", "use_price_context",
        "use_water_context", "use_economic_context", "use_river_context",
        "use_inflow_profiles", "use_hydraulic_context", "use_water_budget", "use_weekly_normalization",
        "use_economic_rescaling",
    ]}
    sources = {method: json.loads((ROOT / config["selection_sources"][method]).read_text()) for method in methods}
    epochs = {method: sources[method]["best_epoch"] for method in methods}
    for method, source in sources.items():
        assert source["averaging_method"] == method
        assert source["config"]["weight_averaging"] == config["weight_averaging"]
        assert method in source["config"]["weight_averaging"].get("methods", ["ema", "swa"])
        assert not source["config"].get("lr_scheduler"), "This trajectory uses a fixed learning rate."
        for key in ["seed", "batch_size", "learning_rate", "model"]:
            assert source["config"][key] == config[key], f"Selected CNN differs for {key}."
        assert all(value == source["config"].get(key, False) for key, value in options.items())
        assert config.get("stable_scaling", False) == source["config"].get("stable_scaling", False), \
            "Scaler must match the selected CNN."
        assert config.get("recency_half_life_years") == source["config"].get("recency_half_life_years"), \
            "Recency weights must match the selected CNN."
        assert epochs[method] >= config["weight_averaging"][f"{method}_start_epoch"]
    trajectory_epochs = max(epochs.values())
    torch.manual_seed(config["seed"])
    torch.set_num_threads(4)
    device = torch.device("cpu")
    train, columns, generators, inputs, topology = load_data(use_extended=options["use_extended"])
    template = pd.read_csv(ROOT / "data/raw/sample_submission.csv")
    mapping = pd.read_csv(ROOT / "data/raw/prediction_mapping.csv")
    assert mapping["Run No"].is_unique and template["Run No"].is_unique
    assert set(mapping["Run No"]) == set(template["Run No"])
    test = mapping.set_index("Run No").loc[template["Run No"]].reset_index()
    test["starttime"] = pd.to_datetime(test.starttime, utc=True)
    assert not train["Run No"].isin(test["Run No"]).any(), "Test cases must not enter fitting."
    print(f"Cases: {len(train)} training, {len(test)} test; selected epochs: {epochs}", flush=True)

    features, names = sequence_features(train, generators, inputs, topology, **options)
    assert all(source["feature_names"] == names for source in sources.values())
    mean, std = fit_sequence_scaler(features, stable=config.get("stable_scaling", False))
    x_train = torch.from_numpy((features - mean) / std)
    del features
    labels = train[columns].to_numpy(dtype="float32", copy=True).reshape(-1, len(generators), HOURS)
    case_weights, weight_summary = recency_case_weights(
        train, config.get("recency_half_life_years"), train.starttime.max() + pd.Timedelta(days=1)
    )
    training_tensors = (x_train, torch.from_numpy(labels))
    if case_weights is not None:
        training_tensors += (torch.from_numpy(case_weights),)
    loader = DataLoader(TensorDataset(*training_tensors),
                        batch_size=config["batch_size"], shuffle=True)
    model = TemporalCNN(len(names), len(generators), **config["model"]).to(device)
    averaging = WeightAveraging(model, config["weight_averaging"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-4)
    loss_function = nn.BCEWithLogitsLoss(reduction="mean" if case_weights is None else "none")
    snapshots, metadata, history = {}, {}, []
    started = time.monotonic()
    # Budgets are fixed by historical validation; the test cases never select an epoch.
    for epoch in range(1, trajectory_epochs + 1):
        model.train()
        total_loss = 0
        for batch in loader:
            x, y = batch[:2]
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_function(model(x), y)
            if case_weights is not None:
                loss = (loss.mean(dim=(1, 2)) * batch[2].to(device)).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            averaging.update_step(epoch, model)
            total_loss += loss.item() * len(x)
        averaging.update_epoch(epoch, model)
        history.append({"epoch": epoch, "training_loss": total_loss / len(train)})
        for method in methods:
            if epoch == epochs[method]:
                averaged_model = averaging.active_models()[method]
                snapshots[method] = {key: value.detach().cpu().clone() for key, value in averaged_model.state_dict().items()}
                metadata[method] = averaging.metadata(method)
        print(f"Epoch {epoch}/{trajectory_epochs}: loss={total_loss / len(train):.5f}", flush=True)
    training_seconds = time.monotonic() - started
    del loader, x_train, labels

    features, test_names = sequence_features(test, generators, inputs, topology, **options)
    assert test_names == names
    x_test = torch.from_numpy((features - mean) / std)
    for method in methods:
        model.load_state_dict(snapshots[method])
        probabilities = predict(model, x_test, device, config["batch_size"])
        predictions = prediction_frame(probabilities.transpose(1, 0, 2).ravel(), test, columns, generators)
        assert predictions.columns.tolist() == template.columns.tolist()
        assert np.array_equal(predictions["Run No"], template["Run No"])
        assert predictions.shape == template.shape
        assert np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all()
        effective_config = {**config, "epochs": epochs[method], "averaging_method": method}
        model_path, prediction_path, report_path = paths[method]
        checkpoint = {"state_dict": snapshots[method], "config": effective_config, "feature_names": names,
                      "generators": generators, "mean": torch.from_numpy(mean), "std": torch.from_numpy(std),
                      "epochs": epochs[method], **metadata[method]}
        with model_path.open("xb") as file:
            torch.save(checkpoint, file)
        predictions.to_csv(prediction_path, index=False, mode="x")
        report = {"config": effective_config, "selection_source": config["selection_sources"][method],
                  "epochs": epochs[method], "trajectory_epochs": trajectory_epochs, **metadata[method],
                  "training_seconds": training_seconds, "training_cases": len(train),
                  "training_run_ids": train["Run No"].tolist(), "first_training_start": str(train.starttime.min()),
                  "last_training_start": str(train.starttime.max()), "test_cases": len(test),
                  "test_run_ids": test["Run No"].tolist(), "first_test_start": str(test.starttime.min()),
                  "last_test_start": str(test.starttime.max()), "feature_names": names,
                  "device": str(device), "threads": 4, "history": history[:epochs[method]],
                  "model": str(model_path), "predictions": str(prediction_path)}
        if weight_summary is not None:
            report["recency_weights"] = weight_summary
        with report_path.open("x") as file:
            file.write(json.dumps(report, indent=2) + "\n")
        print(f"Saved {method.upper()}: {len(predictions)} rows to {prediction_path}", flush=True)


if __name__ == "__main__":
    main()
