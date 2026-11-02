"""Fixed-budget temporal base models for out-of-fold stacking."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from features import HOURS, ROOT, build_features, flatten_labels, load_data, prediction_frame
from training_weights import recency_fit_weights


FEATURE_FLAGS = [
    "use_extended", "use_system_context", "use_topology", "use_price_context",
    "use_water_context", "use_economic_context", "use_river_context",
    "use_inflow_profiles", "use_hydraulic_context", "use_water_budget",
]


def feature_options(config):
    return {key: config.get(key, False) for key in FEATURE_FLAGS}


def artifact_paths(name, extension):
    return (ROOT / f"models/{name}.{extension}",
            ROOT / f"data/processed/{name}_predictions.csv",
            ROOT / f"reports/{name}_run.json")


def matching_training(report, train, names):
    return (report.get("training_cases") == len(train)
            and report.get("last_training_start") == str(train.starttime.max())
            and report.get("feature_names", report.get("features")) == names)


def reuse_tree(profile, suffix, train, names, options, params, weight_summary):
    import xgboost as xgb

    if suffix == "final":
        source_name = "submission_xgboost_price" if profile == "price" else f"submission_xgb_{profile}"
        report_path = ROOT / f"reports/{source_name}_run.json"
        model_path = ROOT / f"models/xgboost_{source_name}.ubj"
    else:
        source_name = f"xgboost_{profile}_{suffix}"
        report_path = ROOT / f"reports/{source_name}_metrics.json"
        model_path = ROOT / f"models/{source_name}.ubj"
    if not report_path.is_file() or not model_path.is_file():
        return None, None
    report = json.loads(report_path.read_text())
    source = report["config"]
    source_params = {key: value for key, value in source["model"].items() if key != "n_estimators"}
    expected_params = {key: value for key, value in params.items() if key != "n_estimators"}
    if (not matching_training(report, train, names) or feature_options(source) != options
            or source_params != expected_params or report.get("training_weight_summary") != weight_summary):
        return None, None
    booster = xgb.Booster(params={"nthread": 4}, model_file=model_path)
    trees = params["n_estimators"]
    if booster.num_boosted_rounds() < trees or booster.feature_names != names:
        return None, None
    # Early stopping selected the old checkpoint, but does not change its first fixed trees.
    booster = booster[:trees]
    booster.set_attr(best_iteration=None, best_score=None)
    booster.set_param({"nthread": 4})
    return booster, {"source_model": str(model_path), "source_report": str(report_path), "fixed_trees": trees}


def reuse_cnn(suffix, train, names, mean, std, config, epochs, generators):
    import torch

    if suffix != "2022" or epochs != 20:
        return None, None
    report_path = ROOT / "reports/cnn_scheduler_2022_metrics.json"
    model_path = ROOT / "models/cnn_scheduler_2022.pt"
    if not report_path.is_file() or not model_path.is_file():
        return None, None
    report = json.loads(report_path.read_text())
    baseline = json.loads((ROOT / "reports/cnn_2022_metrics.json").read_text())
    history = report["history"][:epochs]
    if (not matching_training(report, train, names) or report["best_epoch"] != epochs
            or report["device"] != "cpu" or len(history) != epochs
            or feature_options(report["config"]) != feature_options(config)
            or any(report["config"][key] != config[key] for key in ["seed", "batch_size", "learning_rate", "model"])
            or any(row.get("learning_rate") != config["learning_rate"] for row in history)
            or any(row["training_loss"] != baseline["history"][i]["training_loss"] for i, row in enumerate(history))):
        return None, None
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    if (checkpoint["feature_names"] != names or checkpoint["generators"] != generators
            or not np.array_equal(checkpoint["mean"].numpy(), mean)
            or not np.array_equal(checkpoint["std"].numpy(), std)):
        return None, None
    return checkpoint, {"source_model": str(model_path), "source_report": str(report_path), "fixed_epochs": epochs}


def save_member(name, model, predictions, columns, template, report):
    values = predictions[columns].to_numpy()
    assert predictions.columns.tolist() == template.columns.tolist()
    assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
    model_path, prediction_path, report_path = artifact_paths(name, "pt" if isinstance(model, dict) else "ubj")
    # Reserve the model path exclusively before serializing it.
    with model_path.open("xb") as file:
        if isinstance(model, dict):
            import torch

            torch.save(model, file)
    if not isinstance(model, dict):
        model.save_model(model_path)
    predictions.to_csv(prediction_path, index=False, mode="x")
    report.update({"model": str(model_path), "predictions": str(prediction_path)})
    with report_path.open("x") as file:
        file.write(json.dumps(report, indent=2) + "\n")
    print(f"Saved {name}: {len(predictions)} prediction cases", flush=True)


def fit_cnn(train, prediction_cases, columns, generators, inputs, topology, config, epochs, suffix):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    from temporal_cnn import TemporalCNN, predict, sequence_features

    torch.manual_seed(config["seed"])
    torch.set_num_threads(4)
    device = torch.device("cpu")
    options = feature_options(config)
    features, names = sequence_features(train, generators, inputs, topology, **options)
    # Match the existing CNN scaler, using training cases only.
    mean = features.mean(axis=(0, 2), keepdims=True)
    std = features.std(axis=(0, 2), keepdims=True)
    std[std < 1e-6] = 1
    reused_checkpoint, reuse = reuse_cnn(suffix, train, names, mean, std, config, epochs, generators)
    x_train = torch.from_numpy((features - mean) / std)
    del features
    labels = train[columns].to_numpy(dtype="float32", copy=True).reshape(-1, len(generators), HOURS)
    loader = DataLoader(TensorDataset(x_train, torch.from_numpy(labels)),
                        batch_size=config["batch_size"], shuffle=True)
    model = TemporalCNN(len(names), len(generators), **config["model"]).to(device)
    if reused_checkpoint is not None:
        model.load_state_dict(reused_checkpoint["state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-4)
    loss_function = nn.BCEWithLogitsLoss()
    history = []
    started = time.monotonic()
    if reuse is None:
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = loss_function(model(x), y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item() * len(x)
            history.append({"epoch": epoch, "training_loss": total_loss / len(train)})
            print(f"CNN epoch {epoch}/{epochs}: loss={total_loss / len(train):.5f}", flush=True)
    training_seconds = time.monotonic() - started
    if reuse is not None:
        print(f"Reusing fixed epoch {epochs} from {reuse['source_model']}", flush=True)
        source_history = json.loads(Path(reuse["source_report"]).read_text())["history"][:epochs]
        history = [{"epoch": row["epoch"], "training_loss": row["training_loss"]} for row in source_history]
    del loader, x_train
    features, prediction_names = sequence_features(prediction_cases, generators, inputs, topology, **options)
    assert names == prediction_names
    probabilities = predict(model, torch.from_numpy((features - mean) / std), device, config["batch_size"])
    predictions = prediction_frame(probabilities.transpose(1, 0, 2).ravel(), prediction_cases, columns, generators)
    effective_config = {key: config[key] for key in ["seed", "batch_size", "learning_rate", "model"]}
    effective_config.update(options)
    effective_config["epochs"] = epochs
    checkpoint = {"state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                  "config": effective_config, "feature_names": names, "generators": generators,
                  "mean": torch.from_numpy(mean), "std": torch.from_numpy(std)}
    metadata = {"model_type": "cnn", "config": effective_config, "epochs": epochs,
                "device": str(device), "threads": 4, "feature_names": names,
                "training_seconds": training_seconds, "history": history,
                "fit_performed": reuse is None, "reuse": reuse}
    return checkpoint, predictions, metadata


def main():
    parser = argparse.ArgumentParser(description="Fit one temporal OOF fold or the final stacking bases.")
    parser.add_argument("--config", default=str(ROOT / "configs/stacking.json"))
    parser.add_argument("--cnn-only", action="store_true", help=argparse.SUPPRESS)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--year", type=int, choices=[2020, 2021, 2022])
    mode.add_argument("--final", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.year is not None:
        assert args.year in config["fold_years"]
    suffix = "final" if args.final else str(args.year)
    source_year = config["source_year"]
    sources = {profile: ROOT / f"reports/xgboost_{profile}_{source_year}_metrics.json"
               for profile in config["profiles"]}
    tree_configs = {profile: json.loads(path.read_text())["config"] for profile, path in sources.items()}
    cnn_source = ROOT / f"configs/cnn_{source_year}.json"
    cnn_config = json.loads(cnn_source.read_text())
    for profile in (["cnn"] if args.cnn_only else [*config["profiles"], "cnn"]):
        for path in artifact_paths(f"stacking_{profile}_{suffix}", "pt" if profile == "cnn" else "ubj"):
            assert not path.exists(), f"Artifact already exists: {path}; do not overwrite an existing run."
    options = feature_options(next(iter(tree_configs.values())))
    assert all(feature_options(source) == options for source in tree_configs.values())
    cases, columns, generators, inputs, topology = load_data(
        use_extended=options["use_extended"] or cnn_config.get("use_extended", False))
    template = pd.read_csv(ROOT / "data/raw/sample_submission.csv")
    mapping = pd.read_csv(ROOT / "data/raw/prediction_mapping.csv")
    assert not cases["Run No"].isin(mapping["Run No"]).any(), "Competition test IDs must not enter fitting."
    if args.final:
        assert mapping["Run No"].is_unique and template["Run No"].is_unique
        assert set(mapping["Run No"]) == set(template["Run No"])
        train = cases.copy()
        prediction_cases = mapping.set_index("Run No").loc[template["Run No"]].reset_index()
        prediction_cases["starttime"] = pd.to_datetime(prediction_cases.starttime, utc=True)
        reference_date = train.starttime.max().normalize() + pd.Timedelta(days=1)
        purged = 0
    else:
        reference_date = pd.Timestamp(f"{args.year}-01-01", tz="UTC")
        end = pd.Timestamp(f"{args.year + 1}-01-01", tz="UTC")
        ends = cases.starttime + pd.Timedelta(hours=HOURS)
        train = cases.loc[ends <= reference_date].copy()
        prediction_cases = cases.loc[(cases.starttime >= reference_date) & (cases.starttime < end)].copy()
        purged = int(((cases.starttime < reference_date) & (ends > reference_date)).sum())
        assert train.starttime.max() + pd.Timedelta(hours=HOURS) <= prediction_cases.starttime.min()
    assert len(train) and len(prediction_cases)
    assert not train["Run No"].isin(prediction_cases["Run No"]).any()
    common = {"stacking_config": config, "mode": "final" if args.final else "fold", "fold_year": args.year,
              "training_cases": len(train), "prediction_cases": len(prediction_cases), "purged_cases": purged,
              "first_training_start": str(train.starttime.min()), "last_training_start": str(train.starttime.max()),
              "last_training_end": str(train.starttime.max() + pd.Timedelta(hours=HOURS)),
              "first_prediction_start": str(prediction_cases.starttime.min()),
              "last_prediction_start": str(prediction_cases.starttime.max()),
              "training_run_ids": train["Run No"].tolist(), "prediction_run_ids": prediction_cases["Run No"].tolist(),
              "member_count": len(config["profiles"]) + 1}
    print(f"Stacking {suffix}: {len(train)} train, {len(prediction_cases)} prediction, {purged} purged", flush=True)
    if args.cnn_only:
        checkpoint, predictions, cnn_report = fit_cnn(
            train, prediction_cases, columns, generators, inputs, topology, cnn_config, config["cnn_epochs"], suffix)
        assert np.array_equal(predictions["Run No"], prediction_cases["Run No"])
        if args.final:
            assert np.array_equal(predictions["Run No"], template["Run No"])
        save_member(f"stacking_cnn_{suffix}", checkpoint, predictions, columns, template,
                    {**common, "source_config": str(cnn_source), **cnn_report})
        return

    import xgboost as xgb

    x_train = build_features(train, generators, inputs, topology, **options)
    x_prediction = build_features(prediction_cases, generators, inputs, topology, **options)
    assert x_train.columns.tolist() == x_prediction.columns.tolist()
    for frame in (x_train, x_prediction):
        frame["generator_id"] = pd.Categorical(frame["generator_id"], categories=range(len(generators)))
    prediction_matrix = xgb.DMatrix(x_prediction, enable_categorical=True, nthread=4)
    labels = flatten_labels(train, columns, generators)
    for profile, source in tree_configs.items():
        params = dict(source["model"], n_estimators=config["xgboost_trees"], n_jobs=4)
        fit_kwargs, weight_summary = recency_fit_weights(
            train, len(generators), source.get("recency_half_life_years") if profile == "recency" else None, reference_date)
        model, reuse = reuse_tree(profile, suffix, train, x_train.columns.tolist(), options, params, weight_summary)
        started = time.monotonic()
        if reuse is None:
            print(f"Fitting {profile}: {params['n_estimators']} fixed trees", flush=True)
            model = xgb.XGBClassifier(objective="binary:logistic", enable_categorical=True, **params)
            model.fit(x_train, labels, **fit_kwargs)
        else:
            print(f"Reusing {profile}: first {params['n_estimators']} trees from {reuse['source_model']}", flush=True)
        training_seconds = time.monotonic() - started
        probabilities = (model.predict(prediction_matrix, iteration_range=(0, params["n_estimators"]))
                         if reuse is not None else model.predict_proba(x_prediction)[:, 1])
        predictions = prediction_frame(probabilities, prediction_cases, columns, generators)
        assert np.array_equal(predictions["Run No"], prediction_cases["Run No"])
        if args.final:
            assert np.array_equal(predictions["Run No"], template["Run No"])
        trees = model.num_boosted_rounds() if reuse is not None else model.get_booster().num_boosted_rounds()
        assert trees == config["xgboost_trees"]
        report = {**common, "model_type": "xgboost", "source_report": str(sources[profile]),
                  "model_parameters": params, "feature_options": options, "feature_names": x_train.columns.tolist(),
                  "trees": trees, "training_weight_summary": weight_summary, "training_seconds": training_seconds,
                  "fit_performed": reuse is None, "reuse": reuse}
        save_member(f"stacking_{profile}_{suffix}", model, predictions, columns, template, report)
        del model, probabilities, predictions, fit_kwargs
    del x_train, x_prediction, prediction_matrix, labels, frame
    # Separate native runtimes: this Mac cannot safely mix XGBoost and Torch in one process.
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--config", str(Path(args.config).resolve()), "--cnn-only"]
    command.extend(["--final"] if args.final else ["--year", str(args.year)])
    subprocess.run(command, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
