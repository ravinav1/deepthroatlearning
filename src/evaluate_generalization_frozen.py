"""Evaluate fixed 2021 XGBoost artifacts on all 2022 cases without fitting."""

import json
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from features import HOURS, ROOT, build_features, load_data, prediction_frame

sys.path.insert(0, str(ROOT / "evaluation"))
from kaggle_metric import score


def main():
    profiles = ["price", "sampling", "regularized", "deep_regularized", "recency"]
    names = [f"xgboost_{profile}_2021" for profile in profiles]
    for name in names:
        for path in [ROOT / f"models/{name}.ubj", ROOT / f"reports/{name}_metrics.json"]:
            assert path.is_file(), f"Missing experiment artifact: {path}"
    reports = [json.loads((ROOT / f"reports/{name}_metrics.json").read_text()) for name in names]
    flags = ["use_extended", "use_system_context", "use_topology", "use_price_context",
             "use_water_context", "use_economic_context", "use_river_context",
             "use_inflow_profiles", "use_hydraulic_context", "use_water_budget"]
    options = {flag: reports[0]["config"].get(flag, False) for flag in flags}
    feature_names = reports[0]["feature_names"]
    assert len(feature_names) == 112, "Expected the current price-context feature set."
    for report in reports:
        assert {flag: report["config"].get(flag, False) for flag in flags} == options, "Feature flags differ."
        assert report["feature_names"] == feature_names, "Feature columns or order differ."
        assert pd.Timestamp(report["config"]["validation_start"], tz="UTC") == pd.Timestamp("2021-01-01", tz="UTC")

    cases, columns, generators, inputs, topology = load_data(use_extended=options["use_extended"])
    assert cases["Run No"].is_unique
    test_ids = pd.read_csv(ROOT / "data/raw/prediction_mapping.csv", usecols=["Run No"])["Run No"]
    assert not cases["Run No"].isin(test_ids).any(), "Competition test cases must be excluded."
    start, end = pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2023-01-01", tz="UTC")
    validation = cases.loc[(cases.starttime >= start) & (cases.starttime < end)].copy()
    expected_starts = pd.date_range(start, end, inclusive="left", freq="D")
    assert len(validation) == 365 and validation.starttime.tolist() == expected_starts.tolist(), "Expected all 365 daily 2022 cases."
    features = build_features(validation, generators, inputs, topology, **options)
    assert features.columns.tolist() == feature_names, "Built feature columns or order differ."
    features["generator_id"] = pd.Categorical(features["generator_id"], categories=range(len(generators)))
    truth = validation[["Run No", *columns]]
    labels = validation[columns].to_numpy().reshape(len(validation), len(generators), HOURS)

    rows = []
    ensemble_probabilities = np.zeros(len(features), dtype="float64")
    for profile, name, report in zip(profiles, names, reports):
        cutoff = pd.Timestamp(report["config"]["validation_start"], tz="UTC")
        training = cases.loc[cases.starttime + pd.Timedelta(hours=HOURS) <= cutoff]
        assert len(training) == report["training_cases"], "Source training count differs."
        assert training.starttime.max() == pd.Timestamp(report["last_training_start"]), "Source training end differs."
        assert training.starttime.max() + pd.Timedelta(hours=HOURS) <= cutoff
        model = xgb.XGBClassifier(n_jobs=4, enable_categorical=True)
        model.load_model(ROOT / f"models/{name}.ubj")
        model.set_params(n_jobs=4)
        assert model.get_booster().feature_names == feature_names, "Model feature columns or order differ."
        assert model.get_booster().num_boosted_rounds() == report["best_iteration"], "Artifact is not the reported best model."
        probabilities = model.predict_proba(features)[:, 1]
        assert np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all()
        ensemble_probabilities += probabilities
        predictions = prediction_frame(probabilities, validation, columns, generators)
        assert predictions.columns.tolist() == ["Run No", *columns]
        assert np.array_equal(predictions["Run No"].to_numpy(), validation["Run No"].to_numpy())
        values = predictions[columns].to_numpy().reshape(labels.shape)
        rows.append({
            "model": profile,
            "model_artifact": f"models/{name}.ubj",
            "source_metrics": f"reports/{name}_metrics.json",
            "training_cases": report["training_cases"],
            "first_training_start": str(training.starttime.min()),
            "last_training_start": report["last_training_start"],
            "training_cutoff": str(cutoff),
            "source_validation_start": report["first_validation_start"],
            "source_validation_end": report["last_validation_start"],
            "best_iteration": report["best_iteration"],
            "micro_roc_auc": float(score(truth, predictions)),
            "generator_auc": {
                generator: float(roc_auc_score(labels[:, i].ravel(), values[:, i].ravel()))
                for i, generator in enumerate(generators)
            },
        })

    ensemble = prediction_frame(ensemble_probabilities / len(profiles), validation, columns, generators)
    ensemble_values = ensemble[columns].to_numpy().reshape(labels.shape)
    rows.append({
        "model": "equal_weight_ensemble",
        "xgboost_member_weights": {profile: 1 / len(profiles) for profile in profiles},
        "micro_roc_auc": float(score(truth, ensemble)),
        "generator_auc": {
            generator: float(roc_auc_score(labels[:, i].ravel(), ensemble_values[:, i].ravel()))
            for i, generator in enumerate(generators)
        },
    })

    output = {
        "evaluation": "Frozen 2021 artifacts on 2022; no fitting or 2022 early stopping.",
        "note": "2022 has already been used for development and is not an untouched test set.",
        "validation_cases": len(validation),
        "first_validation_start": str(validation.starttime.min()),
        "last_validation_start": str(validation.starttime.max()),
        "feature_names": feature_names,
        "models": rows,
    }
    (ROOT / "reports/generalization_frozen_comparison.json").write_text(json.dumps(output, indent=2) + "\n")
    print(pd.DataFrame(rows)[["model", "micro_roc_auc"]].to_string(index=False))


if __name__ == "__main__":
    main()
