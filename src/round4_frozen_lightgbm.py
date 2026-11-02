"""Frozen LightGBM inference in its own one-thread process, without Torch/XGBoost."""

import os
import sys

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import lightgbm as lgb
from scipy.special import expit

from features import build_features, prediction_frame
from round3_frozen_evaluation import FEATURE_OPTIONS
from round4_frozen_checks import prepare
from tree_preprocessing import rescale_tree_features


def main():
    source, config, model_path, context, evidence, finish = prepare(
        "lightgbm", "goal_lightgbm_rescaled_2021", "round2_frozen_lgb_2022")
    _, _, columns, generators, inputs, topology = context
    model = lgb.Booster(model_file=str(model_path))
    names = model.feature_name()
    assert names == source["feature_names"]
    assert model.current_iteration() == source["best_iteration"]
    num_iteration = evidence["requested_num_iteration"] or model.current_iteration()
    assert num_iteration <= model.current_iteration()
    options = {key: config.get(key, False) for key in (*FEATURE_OPTIONS, "use_temporal_ranks", "use_favourable_periods")}
    evidence.update({"feature_options": options, "feature_names": names,
                     "saved_model_trees": model.current_iteration(), "trees": num_iteration})

    def predict_cases(subset, tree_count=None):
        values = build_features(subset, generators, inputs, topology, **options)
        if config.get("use_economic_rescaling", False):
            values = rescale_tree_features(values)
        assert values.columns.tolist() == names
        probabilities = expit(model.predict(values, raw_score=True, num_threads=1, num_iteration=tree_count))
        return prediction_frame(probabilities, subset, columns, generators)

    assert "torch" not in sys.modules and "xgboost" not in sys.modules
    finish(lambda subset: predict_cases(subset, num_iteration), tolerance=1e-10, predict_replay=predict_cases)
    assert "torch" not in sys.modules and "xgboost" not in sys.modules


if __name__ == "__main__":
    main()
