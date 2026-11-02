"""Compare saved XGBoost generalization experiments and fixed blends."""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from features import HOURS, ROOT


def main():
    profiles = ["price", "sampling", "regularized", "deep_regularized", "recency"]
    years = [2021, 2022]
    if (ROOT / "reports/xgboost_recency_2020_metrics.json").is_file():
        years.insert(0, 2020)
    # Finish every experiment before creating or replacing the derived report.
    for year in years:
        for profile in (["price", "recency"] if year == 2020 else profiles):
            name = f"xgboost_{profile}_{year}"
            for path in [ROOT / f"data/processed/{name}_validation_predictions.csv",
                         ROOT / f"reports/{name}_metrics.json"]:
                assert path.is_file(), f"Missing experiment artifact: {path}"

    diagnostics_path = ROOT / "reports/generalization_reference_diagnostics.json"
    reference_diagnostics = json.loads(diagnostics_path.read_text()) if diagnostics_path.is_file() else {}

    columns = pd.read_csv(ROOT / "data/raw/sample_submission.csv", nrows=0).columns[1:].tolist()
    truth = pd.read_csv(ROOT / "data/raw/kernel/Unit_commitment_decisions.csv",
                        usecols=["Run No", *columns],
                        dtype={column: "uint8" for column in columns}).set_index("Run No")
    assert truth.index.is_unique, "Truth case IDs must be unique."
    generators = list(dict.fromkeys(column.removeprefix("result_committed_").rsplit("_t", 1)[0]
                                   for column in columns))
    assert len(generators) == 14
    assert columns == [f"result_committed_{generator}_t{hour}"
                       for generator in generators for hour in range(HOURS)]

    rows = []
    for year in years:
        year_profiles = ["price", "recency"] if year == 2020 else profiles
        frames = {
            profile: pd.read_csv(ROOT / f"data/processed/xgboost_{profile}_{year}_validation_predictions.csv")
            .set_index("Run No")
            for profile in year_profiles
        }
        lightgbm = None
        plant = None
        if year != 2020:
            lightgbm = pd.read_csv(ROOT / f"data/processed/topology_{year}_validation_predictions.csv").set_index("Run No")
            plant = pd.read_csv(ROOT / f"data/processed/plant_commitment_{year}_validation_predictions.csv").set_index("Run No")
        reference = frames["price"]
        assert reference.index.is_unique and len(reference), "Validation case IDs must be unique and nonempty."
        assert reference.columns.tolist() == columns, "Prediction columns must match the submission order."
        for frame in [*frames.values(), *([lightgbm, plant] if year != 2020 else [])]:
            assert frame.index.equals(reference.index), "Validation case IDs or their order differ."
            assert frame.columns.tolist() == columns, "Prediction columns or their order differ."
            values = frame.to_numpy()
            assert np.isfinite(values).all(), "Predictions must be finite."
            # Plant marginalization has previously produced rounding just above 1.
            assert ((values >= 0) & (values <= 1 + 1e-6)).all(), "Invalid probabilities."

        labels = truth.loc[reference.index, columns].to_numpy().reshape(-1, len(generators), HOURS)
        if year != 2020:
            frames["equal_weight_ensemble"] = sum(frames.values()) / len(profiles)
        for profile, frame in frames.items():
            source_metrics = {}
            if profile in profiles:
                name = f"xgboost_{profile}_{year}"
                source_metrics = json.loads((ROOT / f"reports/{name}_metrics.json").read_text())
                assert source_metrics["validation_cases"] == len(reference), "Report validation count differs."
            if profile == "price" and str(year) in reference_diagnostics:
                source_metrics["training_auc_sample"] = reference_diagnostics[str(year)]
            sample = source_metrics.get("training_auc_sample")
            reference_sample = reference_diagnostics.get(str(year))
            if sample is not None and reference_sample is not None:
                assert sample["run_ids"] == reference_sample["run_ids"], "Training diagnostic cases differ."
            variants = [(profile, frame)]
            if year != 2020:
                blend = .75 * frame + .25 * lightgbm
                variants.extend([(f"{profile}_blend", blend),
                                 (f"{profile}_plant_blend", .5 * blend + .5 * plant)])
            for variant, predictions in variants:
                probabilities = predictions.to_numpy().reshape(labels.shape)
                row = {
                    "year": year,
                    "model": variant,
                    "micro_roc_auc": float(roc_auc_score(labels.ravel(), probabilities.ravel())),
                    "generator_auc": {
                        generator: float(roc_auc_score(labels[:, i].ravel(), probabilities[:, i].ravel()))
                        for i, generator in enumerate(generators)
                    },
                }
                if profile == "equal_weight_ensemble":
                    row["xgboost_member_weights"] = {member: 1 / len(profiles) for member in profiles}
                if variant == profile and source_metrics:
                    assert abs(row["micro_roc_auc"] - source_metrics["xgboost_micro_roc_auc"]) < 1e-10, "Saved AUC differs."
                    row["source_training_metrics"] = {
                        key: value for key, value in source_metrics.items() if key != "feature_names"
                    }
                    if sample is not None:
                        row["training_auc_sample_note"] = (
                            "In-sample AUC on sampled whole training cases, not full-training AUC "
                            "or an independent generalization estimate."
                        )
                        row["sample_training_minus_full_validation_auc"] = sample["micro_roc_auc"] - row["micro_roc_auc"]
                    if profile == "price" and str(year) in reference_diagnostics:
                        row["training_auc_sample_source"] = "reports/generalization_reference_diagnostics.json"
                rows.append(row)

    output = ROOT / "reports/generalization_experiment_comparison.json"
    output.write_text(json.dumps(rows, indent=2) + "\n")
    print(pd.DataFrame(rows).pivot(index="model", columns="year", values="micro_roc_auc").round(6).to_string())


if __name__ == "__main__":
    main()
