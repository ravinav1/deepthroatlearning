"""Frozen Transformer inference in its own one-thread process using its stored scaler."""

import hashlib
import os
import sys

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import torch

from features import prediction_frame
from round3_frozen_evaluation import FEATURE_OPTIONS
from round4_frozen_checks import prepare
from temporal_cnn import predict, sequence_features
from weekly_transformer import WeeklyTransformer


def main():
    torch.set_num_threads(1)
    source, config, model_path, context, evidence, finish = prepare(
        "transformer", "round2_transformer_task_weights_2021", "round3_frozen_transformer_weighted_2022")
    _, _, columns, generators, inputs, topology = context
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    assert checkpoint["config"] == config and config["model_type"] == "transformer"
    assert checkpoint["generators"] == generators
    names = checkpoint["feature_names"]
    assert names == source["feature_names"]
    assert not config.get("use_temporal_ranks", False) and not config.get("use_favourable_periods", False)
    options = {key: config.get(key, False) for key in (*FEATURE_OPTIONS, "use_weekly_normalization", "use_economic_rescaling")}
    mean, std = checkpoint["mean"].numpy(), checkpoint["std"].numpy()
    assert np.isfinite(mean).all() and np.isfinite(std).all() and (std > 0).all()
    model = WeeklyTransformer(len(names), len(generators), **config["model"])
    model.load_state_dict(checkpoint["state_dict"])
    evidence.update({"feature_options": options, "feature_names": names, "selected_epoch": source["best_epoch"],
                     "scaler_mean_sha256": hashlib.sha256(mean.tobytes()).hexdigest(),
                     "scaler_std_sha256": hashlib.sha256(std.tobytes()).hexdigest(), "device": "cpu"})

    def predict_cases(subset):
        values, observed_names = sequence_features(subset, generators, inputs, topology, **options)
        assert observed_names == names
        tensor = torch.from_numpy((values - mean) / std)
        probabilities = predict(model, tensor, torch.device("cpu"), config["batch_size"])
        return prediction_frame(probabilities.transpose(1, 0, 2).ravel(), subset, columns, generators)

    assert "xgboost" not in sys.modules and "lightgbm" not in sys.modules
    finish(predict_cases, tolerance=2e-6)
    assert "xgboost" not in sys.modules and "lightgbm" not in sys.modules


if __name__ == "__main__":
    main()
