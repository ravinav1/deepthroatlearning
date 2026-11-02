"""Train count + unit-combination models for Tokke and Vinje."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from features import HOURS, ROOT, build_features, load_data
from plant_commitment import plant_labels, unit_probabilities

sys.path.insert(0, str(ROOT / "evaluation"))
from kaggle_metric import score


def fit_classifier(features, labels, validation_features, validation_labels, config):
    model = xgb.XGBClassifier(
        objective="multi:softprob", eval_metric="mlogloss",
        callbacks=[xgb.callback.EarlyStopping(rounds=config["early_stopping_rounds"], save_best=True)],
        **config["model"],
    )
    model.fit(features, labels, eval_set=[(validation_features, validation_labels)], verbose=100)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/plant_commitment_2022.json"))
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    name = Path(args.config).stem
    options = {key: value for key, value in config.items() if key.startswith("use_")}
    cases, columns, generators, inputs, topology = load_data(options["use_extended"])
    boundary = pd.Timestamp(config["validation_start"], tz="UTC")
    train = cases.loc[cases["starttime"] + pd.Timedelta(hours=HOURS) <= boundary]
    mask = cases["starttime"] >= boundary
    if "validation_end" in config:
        mask &= cases["starttime"] < pd.Timestamp(config["validation_end"], tz="UTC")
    validation = cases.loc[mask]
    test_ids = pd.read_csv(ROOT / "data/raw/prediction_mapping.csv")["Run No"]
    assert not cases["Run No"].isin(test_ids).any()
    assert (train["starttime"] + pd.Timedelta(hours=HOURS)).max() <= validation["starttime"].min()
    print(f"Cases: {len(train)} training, {len(validation)} validation; test excluded", flush=True)

    year = boundary.year
    base = pd.read_csv(ROOT / f"data/processed/xgboost_price_{year}_validation_predictions.csv")
    lightgbm = pd.read_csv(ROOT / f"data/processed/topology_{year}_validation_predictions.csv")
    for frame in [base, lightgbm]:
        assert np.array_equal(frame["Run No"], validation["Run No"])
        assert frame.columns.tolist() == ["Run No", *columns]
    current = base.copy()
    current[columns] = .75 * base[columns] + .25 * lightgbm[columns]
    proposed = current.copy()
    plant_metrics = {}

    for plant in config["plants"]:
        units = [g for g in generators if g.rsplit("_G", 1)[0] == plant]
        # Inputs are shared by the plant's units, so one representative avoids duplicates.
        x_train = build_features(train, units[:1], inputs, topology, **options).drop(columns="generator_id")
        x_valid = build_features(validation, units[:1], inputs, topology, **options).drop(columns="generator_id")
        y_train = plant_labels(train, units)
        y_valid = plant_labels(validation, units)
        counts = y_train.sum(axis=1).astype(int)
        valid_counts = y_valid.sum(axis=1).astype(int)
        print(f"{plant}: fitting count model ({x_train.shape[1]} features)", flush=True)
        count_model = fit_classifier(x_train, counts, x_valid, valid_counts, config)
        count_p = count_model.predict_proba(x_valid)

        powers = 2 ** np.arange(len(units))
        train_masks = y_train @ powers
        valid_masks = y_valid @ powers
        patterns, encoded = np.unique(train_masks, return_inverse=True)
        # Fit only training-observed combinations; unseen validation patterns are reported.
        valid_encoded = pd.Index(patterns).get_indexer(valid_masks)
        known = valid_encoded >= 0
        print(f"{plant}: fitting {len(patterns)} unit combinations", flush=True)
        pattern_model = fit_classifier(x_train, encoded, x_valid.loc[known], valid_encoded[known], config)
        pattern_p = pattern_model.predict_proba(x_valid)
        probabilities = unit_probabilities(count_p, pattern_p, patterns, len(units))
        assert np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1 + 1e-6)).all()
        np.testing.assert_allclose(probabilities.sum(axis=1), count_p @ np.arange(len(units) + 1), atol=1e-6)
        values = probabilities.reshape(len(validation), HOURS, len(units)).transpose(0, 2, 1).reshape(len(validation), -1)
        plant_columns = [f"result_committed_{g}_t{h}" for g in units for h in range(HOURS)]
        proposed[plant_columns] = values
        plant_metrics[plant] = {
            "generators": units, "patterns": patterns.tolist(), "feature_names": x_train.columns.tolist(),
            "count_trees": count_model.best_iteration + 1, "pattern_trees": pattern_model.best_iteration + 1,
            "unseen_validation_patterns": int((~known).sum()),
            "count_accuracy": float((count_p.argmax(axis=1) == valid_counts).mean()),
            "current_auc": float(roc_auc_score(y_valid.ravel(), current[plant_columns].to_numpy().reshape(len(validation), len(units), HOURS).transpose(0, 2, 1).ravel())),
            "proposed_auc": float(roc_auc_score(y_valid.ravel(), probabilities.ravel())),
        }
        count_model.save_model(ROOT / f"models/{name}_{plant}_count.ubj")
        pattern_model.save_model(ROOT / f"models/{name}_{plant}_patterns.ubj")
        print(f"{plant}: AUC {plant_metrics[plant]['current_auc']:.6f} -> {plant_metrics[plant]['proposed_auc']:.6f}", flush=True)

    truth = validation[["Run No", *columns]]
    half = current.copy()
    half[columns] = .5 * current[columns] + .5 * proposed[columns]
    scores = {"current": score(truth, current), "plant_models": score(truth, proposed), "half_blend": score(truth, half)}
    report = {"config": config, "training_cases": len(train), "validation_cases": len(validation),
              "last_training_start": str(train["starttime"].max()),
              "first_validation_start": str(validation["starttime"].min()),
              "last_validation_start": str(validation["starttime"].max()),
              "micro_roc_auc": scores, "plants": plant_metrics}
    (ROOT / f"reports/{name}_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    proposed.to_csv(ROOT / f"data/processed/{name}_validation_predictions.csv", index=False)
    half.to_csv(ROOT / f"data/processed/{name}_half_validation_predictions.csv", index=False)
    print(json.dumps(scores, indent=2), flush=True)


if __name__ == "__main__":
    main()
