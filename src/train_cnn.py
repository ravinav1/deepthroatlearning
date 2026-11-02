"""Shared sequence training loop with the same chronological splits as the trees."""

import argparse
import json
import sys
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
from bidirectional_lstm import BidirectionalLSTM
from weekly_transformer import WeeklyTransformer
from weekly_mixer import WeeklyMixer
from weight_averaging import WeightAveraging
from generator_specialist import output_targets
from consistency_regularization import normalized_generator_weights, sequence_training_loss
from component_monitor import frozen_component_monitor

sys.path.insert(0, str(ROOT / "evaluation"))
from kaggle_metric import score


def frozen_ensemble_monitor(options, validation, columns):
    """Prepare fixed validation contributions, retaining the original tree mode."""
    mode = options.get("mode", "legacy_tree_mean")
    if mode == "replace_component":
        return frozen_component_monitor(options, validation, columns, ROOT)
    assert mode == "legacy_tree_mean", f"Unknown ensemble monitor mode: {mode}"
    weights = np.array([
        options["transformer_weights"][column.removeprefix("result_committed_").rsplit("_G", 1)[0]]
        for column in columns
    ])
    assert np.isfinite(weights).all() and ((weights >= 0) & (weights <= 1)).all()
    trees = []
    for name in options["tree_prediction_paths"]:
        frame = pd.read_csv(ROOT / name)
        assert frame.columns.tolist() == ["Run No", *columns], "Monitor columns must match model outputs."
        assert frame["Run No"].is_unique and set(frame["Run No"]) == set(validation["Run No"]), \
            "Monitor predictions must cover precisely these validation cases."
        values = frame.set_index("Run No").loc[validation["Run No"], columns].to_numpy()
        assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
        trees.append(values)
    assert len(trees) == 2, "The frozen reference uses an equal XGBoost/LightGBM mean."
    return (1 - weights) * (0.5 * trees[0] + 0.5 * trees[1]), weights


def main(default_config="cnn_2022.json"):
    parser = argparse.ArgumentParser(description="Train and validate a sequence model.")
    parser.add_argument("--config", default=str(ROOT / "configs" / default_config))
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
    averaging_options = config.get("weight_averaging")
    averaging_methods = averaging_options.get("methods", ["ema", "swa"]) if averaging_options is not None else []
    stopping_metric = config.get("early_stopping_metric", "raw")
    assert stopping_metric in ["raw", "ema", "ensemble", "fixed"]
    assert stopping_metric != "ema" or "ema" in averaging_methods, "EMA stopping needs EMA averaging."
    assert stopping_metric != "fixed" or ("lr_scheduler" not in config and averaging_options is None), \
        "Fixed-epoch confirmation uses the last raw checkpoint without a validation scheduler."
    monitor_options = config.get("ensemble_monitor")
    assert (monitor_options is not None) == (stopping_metric == "ensemble"), \
        "A frozen ensemble monitor requires ensemble checkpoint selection."
    assert monitor_options is None or averaging_options is None, "Ensemble monitoring uses the raw model."
    for variant in [name, *[f"{name}_{method}" for method in averaging_methods]]:
        for path in [ROOT / f"models/{variant}.pt",
                     ROOT / f"data/processed/{variant}_validation_predictions.csv",
                     ROOT / f"reports/{variant}_metrics.json"]:
            assert not path.exists(), f"Artifact already exists: {path}; choose a new config name."
    torch.manual_seed(config["seed"])
    torch.set_num_threads(4)
    device = torch.device(config.get("device", "mps" if torch.backends.mps.is_available() else "cpu"))
    options = {key: config.get(key, False) for key in [
        "use_extended", "use_system_context", "use_topology", "use_price_context",
        "use_water_context", "use_economic_context", "use_river_context",
        "use_inflow_profiles", "use_hydraulic_context", "use_water_budget", "use_weekly_normalization",
        "use_economic_rescaling",
    ]}
    cases, columns, generators, inputs, topology = load_data(use_extended=options["use_extended"])
    target_generator = config.get("target_generator")
    output_columns, output_generators = output_targets(columns, generators, target_generator)
    generator_weights = normalized_generator_weights(config.get("plant_loss_weights"), output_generators, device)
    boundary = pd.Timestamp(config["validation_start"], tz="UTC")
    ends = cases.starttime + pd.Timedelta(hours=HOURS)
    train = cases.loc[ends <= boundary].copy()
    mask = cases.starttime >= boundary
    if "validation_end" in config:
        mask &= cases.starttime < pd.Timestamp(config["validation_end"], tz="UTC")
    validation = cases.loc[mask].copy()
    assert len(train) and len(validation)
    assert (train.starttime + pd.Timedelta(hours=HOURS)).max() <= validation.starttime.min()
    test_ids = pd.read_csv(ROOT / "data/raw/prediction_mapping.csv", usecols=["Run No"])["Run No"]
    assert not cases["Run No"].isin(test_ids).any()
    purged = int(((cases.starttime < boundary) & (ends > boundary)).sum())
    print(f"Device: {device}; cases: {len(train)} train, {len(validation)} validation, {purged} purged", flush=True)
    if monitor_options is not None:
        frozen_predictions, neural_weights = frozen_ensemble_monitor(monitor_options, validation, output_columns)

    x_train, names = sequence_features(train, generators, inputs, topology, **options)
    x_validation, validation_names = sequence_features(validation, generators, inputs, topology, **options)
    assert names == validation_names
    # Fit scaling on training cases only, separately for each input channel.
    mean, std = fit_sequence_scaler(x_train, stable=config.get("stable_scaling", False))
    x_train = torch.from_numpy((x_train - mean) / std)
    x_validation = torch.from_numpy((x_validation - mean) / std)
    labels = train[output_columns].to_numpy(dtype="float32", copy=True).reshape(-1, len(output_generators), HOURS)
    y_train = torch.from_numpy(labels)
    case_weights, weight_summary = recency_case_weights(
        train, config.get("recency_half_life_years"), boundary
    )
    training_tensors = (x_train, y_train)
    if case_weights is not None:
        training_tensors += (torch.from_numpy(case_weights),)
    loader = DataLoader(TensorDataset(*training_tensors), batch_size=config["batch_size"], shuffle=True)
    model_type = config.get("model_type", "cnn")
    model_class = {"cnn": TemporalCNN, "bilstm": BidirectionalLSTM,
                   "transformer": WeeklyTransformer, "mixer": WeeklyMixer}[model_type]
    model = model_class(len(names), len(output_generators), **config["model"]).to(device)
    averaging = WeightAveraging(model, averaging_options) if averaging_options is not None else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config.get("weight_decay", 1e-4))
    scheduler = None
    if "lr_scheduler" in config:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", **config["lr_scheduler"]
        )
    print(f"Input: {len(names)} channels x {HOURS} hours; output: {len(output_generators)} x {HOURS}; parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    best_score, best_selection_score, best_epoch, history = -1, -1, 0, []
    averaged_results = {method: {"score": -1, "epoch": 0, "history": []}
                        for method in averaging_methods}
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        learning_rate = optimizer.param_groups[0]["lr"]
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
            if averaging is not None:
                averaging.update_step(epoch, model)
            total_loss += loss.item() * len(x)
        if averaging is not None:
            averaging.update_epoch(epoch, model)
        probabilities = predict(model, x_validation, device, config["batch_size"])
        # Sequence output is case -> generator -> hour; exporter expects generator first.
        predictions = prediction_frame(probabilities.transpose(1, 0, 2).ravel(), validation, output_columns, output_generators)
        auc = score(validation[["Run No", *output_columns]], predictions)
        row = {"epoch": epoch, "training_loss": total_loss / len(train), "validation_auc": auc}
        selection_score = auc
        if monitor_options is not None:
            ensemble_predictions = pd.DataFrame(
                frozen_predictions + neural_weights * predictions[output_columns].to_numpy(),
                columns=output_columns,
            )
            ensemble_predictions.insert(0, "Run No", predictions["Run No"].to_numpy())
            selection_score = score(validation[["Run No", *output_columns]], ensemble_predictions)
            row["ensemble_validation_auc"] = selection_score
        if scheduler is not None:
            row["learning_rate"] = learning_rate
        history.append(row)
        status = f"Epoch {epoch}: loss={total_loss / len(train):.5f}, validation AUC={auc:.6f}"
        if monitor_options is not None:
            status += f", frozen ensemble AUC={selection_score:.6f}"
        if scheduler is not None:
            status += f", lr={learning_rate:g}"
        print(status, flush=True)
        if stopping_metric == "fixed" or selection_score > best_selection_score:
            best_score, best_selection_score, best_epoch = auc, selection_score, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_predictions = predictions
        if averaging is not None:
            for method, averaged_model in averaging.active_models().items():
                probabilities = predict(averaged_model, x_validation, device, config["batch_size"])
                predictions = prediction_frame(probabilities.transpose(1, 0, 2).ravel(), validation, output_columns, output_generators)
                averaged_auc = score(validation[["Run No", *output_columns]], predictions)
                result = averaged_results[method]
                result["history"].append({"epoch": epoch, "validation_auc": averaged_auc,
                                          "averaging_updates": averaging.metadata(method)["averaging_updates"]})
                print(f"  {method.upper()} validation AUC={averaged_auc:.6f}", flush=True)
                if averaged_auc > result["score"]:
                    result.update({"score": averaged_auc, "epoch": epoch, "predictions": predictions,
                                   "state": {key: value.detach().cpu().clone() for key, value in averaged_model.state_dict().items()},
                                   "metadata": averaging.metadata(method)})
        if scheduler is not None:
            scheduler.step(selection_score)
        stopping_best_epoch = best_epoch if stopping_metric in ["raw", "ensemble", "fixed"] else averaged_results["ema"]["epoch"]
        if stopping_metric != "fixed" and stopping_best_epoch and epoch - stopping_best_epoch >= config["patience"]:
            print("Early stopping", flush=True)
            break

    report = {"config": config, "training_cases": len(train), "validation_cases": len(validation),
              "purged_cases": purged, "last_training_start": str(train.starttime.max()),
              "first_validation_start": str(validation.starttime.min()),
              "last_validation_start": str(validation.starttime.max()), "device": str(device),
              f"{model_type}_micro_roc_auc": best_score, "best_epoch": best_epoch, "feature_names": names,
              "history": history}
    if monitor_options is not None:
        standalone_best = max(history, key=lambda row: row["validation_auc"])
        monitor_note = "Cached tree predictions and fixed plant weights affect validation selection and stopping only; they are never model inputs or training targets."
        if monitor_options.get("mode") == "replace_component":
            monitor_note = "Cached component predictions and fixed plant weights affect validation selection and stopping only; they are never model inputs or training targets."
        report.update({"checkpoint_selection_metric": "frozen_ensemble_micro_roc_auc",
                       "best_ensemble_validation_auc": best_selection_score,
                       "standalone_best_epoch": standalone_best["epoch"],
                       "standalone_best_micro_roc_auc": standalone_best["validation_auc"],
                       "ensemble_monitor_note": monitor_note})
    if stopping_metric == "fixed":
        report["checkpoint_selection_metric"] = "fixed_by_config"
    if target_generator is not None:
        report.update({"target_generator": target_generator, "output_columns": output_columns,
                       "output_generators": output_generators, "input_generators": generators})
    if weight_summary is not None:
        report["recency_weights"] = weight_summary
    if generator_weights is not None:
        report["generator_loss_weights"] = dict(zip(output_generators, generator_weights.cpu().tolist()))
        report["prediction_role"] = "ensemble_component"
        report["standalone_auc_note"] = "Zero-weight heads receive no direct training loss; judge this component through the frozen ensemble."
    if "early_stopping_metric" in config:
        report.update({"training_epochs": epoch, "early_stopping_metric": stopping_metric,
                       "stopping_best_epoch": stopping_best_epoch,
                       "stop_reason": "fixed_by_config" if stopping_metric == "fixed" else
                           "patience" if epoch - stopping_best_epoch >= config["patience"] else "epoch_budget"})
    results = {"raw": {"state": best_state, "predictions": best_predictions, "report": report}}
    if averaging is not None:
        report["averaging_method"] = "raw"
        if stopping_metric == "raw":
            report["raw_stopping_epoch"] = epoch
        for method, result in averaged_results.items():
            if result["epoch"]:
                results[method] = {"state": result["state"], "predictions": result["predictions"],
                                   "report": {**report, **result["metadata"], "best_epoch": result["epoch"],
                                              f"{model_type}_micro_roc_auc": result["score"], "history": result["history"]}}
    for method, result in results.items():
        variant = name if method == "raw" else f"{name}_{method}"
        checkpoint = {"state_dict": result["state"], "config": config, "feature_names": names,
                      "generators": generators, "mean": torch.from_numpy(mean), "std": torch.from_numpy(std)}
        if target_generator is not None:
            checkpoint.update({"target_generator": target_generator, "output_generators": output_generators,
                               "output_columns": output_columns})
        if averaging is not None:
            checkpoint.update({"averaging_method": method, "best_epoch": result["report"]["best_epoch"]})
            if method != "raw":
                checkpoint.update(averaged_results[method]["metadata"])
        with (ROOT / f"models/{variant}.pt").open("xb") as file:
            torch.save(checkpoint, file)
        result["predictions"].to_csv(ROOT / f"data/processed/{variant}_validation_predictions.csv", index=False, mode="x")
        with (ROOT / f"reports/{variant}_metrics.json").open("x") as file:
            file.write(json.dumps(result["report"], indent=2) + "\n")
    label = "Selected" if monitor_options is not None or stopping_metric == "fixed" else "Best"
    print(f"{label} {model_type} micro ROC-AUC: {best_score:.6f} at epoch {best_epoch}", flush=True)
    if monitor_options is not None:
        print(f"Checkpoint selected by frozen ensemble micro ROC-AUC: {best_selection_score:.6f}", flush=True)


if __name__ == "__main__":
    main()
