"""Read-only inference: apply the existing 2021-selected trees to 2022 cases."""

import hashlib
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.special import expit

from features import HOURS, ROOT, build_features, load_data, prediction_frame
from plant_blend import metrics, read_predictions
from tree_preprocessing import rescale_tree_features


SOURCES = {
    "xgb": ("goal_xgboost_rescaled_2021", "models/goal_xgboost_rescaled_2021.ubj"),
    "lgb": ("goal_lightgbm_rescaled_2021", "models/lightgbm_goal_lightgbm_rescaled_2021.txt"),
}
FEATURE_OPTIONS = [
    "use_extended", "use_system_context", "use_topology", "use_price_context",
    "use_water_context", "use_economic_context", "use_river_context",
    "use_inflow_profiles", "use_hydraulic_context", "use_water_budget",
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert "torch" not in sys.modules
    for name, (prefix, model_name) in SOURCES.items():
        output_prefix = f"round2_frozen_{name}_2022"
        prediction_path = ROOT / f"data/processed/{output_prefix}_validation_predictions.csv"
        output_report = ROOT / f"reports/{output_prefix}_metrics.json"
        assert not prediction_path.exists() and not output_report.exists()
        report_path = ROOT / f"reports/{prefix}_metrics.json"
        model_path = ROOT / model_name
        source_hash = sha256(model_path)
        source = json.loads(report_path.read_text())
        config = source["config"]
        options = {key: config.get(key, False) for key in FEATURE_OPTIONS}
        cases, columns, generators, inputs, topology = load_data(use_extended=options["use_extended"])
        test_ids = pd.read_csv(ROOT / "data/raw/prediction_mapping.csv", usecols=["Run No"])["Run No"]
        assert not cases["Run No"].isin(test_ids).any()
        boundary = pd.Timestamp(config["validation_start"], tz="UTC")
        selection_end = pd.Timestamp(config["validation_end"], tz="UTC")
        train = cases.loc[cases.starttime + pd.Timedelta(hours=HOURS) <= boundary]
        selection = cases.loc[(cases.starttime >= boundary) & (cases.starttime < selection_end)]
        assert len(train) == source["training_cases"]
        assert str(train.starttime.max()) == source["last_training_start"]
        assert len(selection) == source["validation_cases"]
        assert str(selection.starttime.min()) == source["first_validation_start"]
        assert str(selection.starttime.max()) == source["last_validation_start"]
        evaluation = cases.loc[cases.starttime.dt.year == 2022].reset_index(drop=True)
        assert not evaluation["Run No"].isin(train["Run No"]).any()
        assert (train.starttime + pd.Timedelta(hours=HOURS)).max() <= evaluation.starttime.min()
        selection_horizon_end = selection.starttime.max() + pd.Timedelta(hours=HOURS)
        purged_mask = evaluation.starttime >= selection_horizon_end
        assert len(evaluation) == 365 and int(purged_mask.sum()) == 359

        if name == "xgb":
            model = xgb.XGBClassifier(n_jobs=1, enable_categorical=True)
            model.load_model(model_path)
            model.set_params(n_jobs=1)
            model_names = model.get_booster().feature_names
            rounds = model.get_booster().num_boosted_rounds()
        else:
            model = lgb.Booster(model_file=str(model_path))
            model_names = model.feature_name()
            rounds = model.current_iteration()
        assert rounds == source["best_iteration"]

        def predict(subset):
            features = build_features(subset, generators, inputs, topology, **options)
            if config.get("use_economic_rescaling", False):
                features = rescale_tree_features(features)
            assert features.columns.tolist() == source["feature_names"] == model_names
            if name == "xgb":
                features["generator_id"] = pd.Categorical(features["generator_id"], categories=range(len(generators)))
                probabilities = model.predict_proba(features)[:, 1]
            else:
                probabilities = expit(model.predict(features, raw_score=True, num_threads=1))
            assert np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all()
            return prediction_frame(probabilities, subset, columns, generators)

        # Reproduce several saved source-year cases before generating the next year's predictions.
        check_cases = selection.iloc[[0, 60, 120, 180, 240, 300, 364]].reset_index(drop=True)
        check = predict(check_cases)
        reference = pd.read_csv(ROOT / f"data/processed/{prefix}_validation_predictions.csv", float_precision="round_trip")
        reference = reference.set_index("Run No").loc[check_cases["Run No"], columns].to_numpy()
        reference_difference = float(np.max(np.abs(check[columns].to_numpy() - reference)))
        # XGBoost's original float32 CSV uses shortest decimal representations.
        source_precision_equal = np.array_equal(check[columns].to_numpy().astype("float32"), reference.astype("float32"))
        assert source_precision_equal if name == "xgb" else reference_difference < 1e-10, \
            f"Source predictions changed: {reference_difference}"
        print(f"{name}: source predictions reproduced; predicting all 365 cases with one CPU thread.", flush=True)
        started = time.monotonic()
        predictions = predict(evaluation)
        prediction_seconds = time.monotonic() - started
        labels = evaluation[columns].to_numpy()
        values = predictions[columns].to_numpy()
        native_prefix = prefix.removesuffix("_2021") + "_2022"
        native = read_predictions(ROOT / f"data/processed/{native_prefix}_validation_predictions.csv", columns, evaluation["Run No"])
        rows = []
        for subset, mask in [("all_2022", np.ones(len(evaluation), dtype=bool)),
                             ("after_selection_horizon", purged_mask.to_numpy())]:
            for variant, probabilities in [("frozen_2021_model", values), ("native_2022_model", native)]:
                rows.append({"subset": subset, "variant": variant, "cases": int(mask.sum()),
                             **metrics(labels[mask], probabilities[mask], generators)})
        assert sha256(model_path) == source_hash
        assert "torch" not in sys.modules
        predictions.to_csv(prediction_path, index=False, mode="x")
        report = {
            "source_model": model_name, "source_model_sha256": source_hash,
            "source_report": str(report_path.relative_to(ROOT)), "source_report_sha256": sha256(report_path),
            "source_config": config, "model_rounds": rounds,
            "training_cases": len(train), "first_training_start": str(train.starttime.min()),
            "last_training_start": str(train.starttime.max()),
            "reconstructed_training_cohort_sha256": hashlib.sha256(train[["Run No", "starttime"]].to_csv(index=False).encode()).hexdigest(),
            "source_selection_cases": len(selection), "source_selection_start": str(selection.starttime.min()),
            "source_selection_last_start": str(selection.starttime.max()),
            "selection_horizon_end_exclusive": str(selection_horizon_end),
            "selection_overlap_cases": int((~purged_mask).sum()), "selection_purged_cases": int(purged_mask.sum()),
            "prediction_cases": len(evaluation), "feature_names": model_names,
            "reproduction_cases": check_cases["Run No"].tolist(), "source_prediction_max_absolute_difference": reference_difference,
            "source_prediction_equal_at_float32_precision": source_precision_equal,
            "prediction_seconds": prediction_seconds, "inference_threads": 1,
            "prediction_path": str(prediction_path.relative_to(ROOT)), "results": rows,
            "interpretation": "No fitting or 2022-label early stopping in this run. Existing source models were trained before 2021 and selected using 2021 labels. The first six 2022 case starts overlap those selection horizons; both all-case and January 7 onward metrics are reported. 2022 remains a previously reused development year, not a pristine holdout. Frozen versus native differences also include less training history and differing selected tree counts.",
        }
        with output_report.open("x") as file:
            json.dump(report, file, indent=2)
            file.write("\n")
        print(json.dumps([{key: row[key] for key in ["subset", "variant", "cases", "micro_roc_auc"]} for row in rows]), flush=True)


if __name__ == "__main__":
    main()
