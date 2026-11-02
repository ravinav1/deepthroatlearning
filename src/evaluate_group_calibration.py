"""Fit one fixed generator calibration on 2021 predictions and evaluate 2022."""

import json

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from features import HOURS, ROOT


def design_matrix(frame, generators):
    probabilities = np.clip(frame.to_numpy().ravel(), 1e-6, 1 - 1e-6)
    logits = np.log(probabilities / (1 - probabilities))
    # CSV rows are cases; their columns are generator-major, then hour-major.
    generator_ids = np.tile(np.repeat(np.arange(len(generators)), HOURS), len(frame))
    indicators = sparse.csr_matrix(
        (np.ones(len(probabilities)), (np.arange(len(probabilities)), generator_ids)),
        shape=(len(probabilities), len(generators)),
    )
    return sparse.hstack([sparse.csr_matrix(logits[:, None]), indicators], format="csr")


def main():
    profiles = ["price", "sampling", "regularized", "deep_regularized", "recency"]
    columns = pd.read_csv(ROOT / "data/raw/sample_submission.csv", nrows=0).columns[1:].tolist()
    generators = list(dict.fromkeys(column.removeprefix("result_committed_").rsplit("_t", 1)[0]
                                   for column in columns))
    assert len(generators) == 14
    assert columns == [f"result_committed_{generator}_t{hour}"
                       for generator in generators for hour in range(HOURS)]
    truth = pd.read_csv(ROOT / "data/raw/kernel/Unit_commitment_decisions.csv",
                        usecols=["Run No", "starttime", *columns],
                        dtype={column: "uint8" for column in columns}).set_index("Run No")
    truth["starttime"] = pd.to_datetime(truth["starttime"], utc=True)
    assert truth.index.is_unique, "Truth case IDs must be unique."

    models = {}
    labels = {}
    for year in [2021, 2022]:
        frames = {
            profile: pd.read_csv(ROOT / f"data/processed/xgboost_{profile}_{year}_validation_predictions.csv")
            .set_index("Run No")
            for profile in profiles
        }
        reference = frames["price"]
        expected = truth.loc[truth["starttime"].dt.year == year].sort_values("starttime").index
        assert reference.index.equals(expected), "Expected every chronological case in the selected year."
        for frame in frames.values():
            assert frame.index.is_unique and frame.index.equals(reference.index), "Prediction case IDs or order differ."
            assert frame.columns.tolist() == columns, "Prediction columns or order differ."
            values = frame.to_numpy()
            assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all(), "Invalid probabilities."
        models[year] = {"price": reference, "equal_weight_ensemble": sum(frames.values()) / len(profiles)}
        labels[year] = truth.loc[reference.index, columns].to_numpy().reshape(-1, len(generators), HOURS)

    rows = []
    for name in ["price", "equal_weight_ensemble"]:
        calibration = LogisticRegression(C=0.001, fit_intercept=False, max_iter=200)
        calibration.fit(design_matrix(models[2021][name], generators), labels[2021].ravel())
        assert calibration.coef_[0, 0] > 0, "The shared logit slope must be positive."
        fit_predictions = calibration.predict_proba(design_matrix(models[2021][name], generators))[:, 1]
        evaluation_predictions = calibration.predict_proba(design_matrix(models[2022][name], generators))[:, 1]
        raw = models[2022][name].to_numpy().reshape(labels[2022].shape)
        calibrated = evaluation_predictions.reshape(labels[2022].shape)
        raw_auc = float(roc_auc_score(labels[2022].ravel(), raw.ravel()))
        calibrated_auc = float(roc_auc_score(labels[2022].ravel(), calibrated.ravel()))
        row = {
            "model": name,
            "fit_year": 2021,
            "evaluation_year": 2022,
            "fit_cases": len(models[2021][name]),
            "evaluation_cases": len(models[2022][name]),
            "calibration_parameters": {"C": 0.001, "fit_intercept": False, "max_iter": 200,
                                       "probability_clip": [1e-6, 1 - 1e-6]},
            "shared_logit_slope": float(calibration.coef_[0, 0]),
            "generator_offsets": dict(zip(generators, calibration.coef_[0, 1:].tolist())),
            "iterations": int(calibration.n_iter_[0]),
            "raw_2021_micro_roc_auc": float(roc_auc_score(labels[2021].ravel(), models[2021][name].to_numpy().ravel())),
            "fit_2021_micro_roc_auc": float(roc_auc_score(labels[2021].ravel(), fit_predictions)),
            "fit_2021_note": (
                "Calibration fit AUC is in-sample. The base models' 2021 predictions were selected "
                "by early stopping on 2021 development labels; these are not untouched OOF predictions."
            ),
            "evaluation_2022_note": (
                "Calibration coefficients use only 2021 labels. The 2022 base models were selected "
                "by early stopping on 2022 development labels; this remains a development comparison."
            ),
            "raw_2022_micro_roc_auc": raw_auc,
            "calibrated_2022_micro_roc_auc": calibrated_auc,
            "calibration_auc_delta": calibrated_auc - raw_auc,
            "raw_2022_generator_auc": {
                generator: float(roc_auc_score(labels[2022][:, i].ravel(), raw[:, i].ravel()))
                for i, generator in enumerate(generators)
            },
            "calibrated_2022_generator_auc": {
                generator: float(roc_auc_score(labels[2022][:, i].ravel(), calibrated[:, i].ravel()))
                for i, generator in enumerate(generators)
            },
        }
        if name == "equal_weight_ensemble":
            row["xgboost_member_weights"] = {profile: 1 / len(profiles) for profile in profiles}
        rows.append(row)

    output = ROOT / "reports/generalization_calibration_comparison.json"
    output.write_text(json.dumps(rows, indent=2) + "\n")
    print(pd.DataFrame(rows).set_index("model")[["raw_2022_micro_roc_auc", "calibrated_2022_micro_roc_auc",
                                               "calibration_auc_delta"]].round(6).to_string())


if __name__ == "__main__":
    main()
