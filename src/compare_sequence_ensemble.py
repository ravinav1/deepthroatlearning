"""Compare the fixed five-XGBoost ensemble with a 20% CNN blend."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from features import HOURS, ROOT
from generator_specialist import output_targets


def main():
    parser = argparse.ArgumentParser(description="Compare CNN predictions in the fixed 80/20 blend.")
    parser.add_argument("--cnn-prefix", action="append", help="CNN experiment prefix; repeat to compare several.")
    parser.add_argument("--baseline-prefix", default="cnn")
    parser.add_argument("--average-with-baseline", help="Also compare one fixed 50/50 CNN average.")
    parser.add_argument("--fixed-specialist-prefix", help="Keep this specialist in the neural component of every 80/20 blend.")
    parser.add_argument("--output", default=str(ROOT / "reports/sequence_ensemble_comparison.json"))
    args = parser.parse_args()
    output = Path(args.output)
    assert not output.exists(), f"Report already exists: {output}; choose a new --output."
    experiments = args.cnn_prefix or ["cnn"]
    cnn_prefixes = list(dict.fromkeys([args.baseline_prefix, *experiments]))
    if args.average_with_baseline and args.average_with_baseline not in cnn_prefixes:
        cnn_prefixes.append(args.average_with_baseline)
    profiles = ["price", "sampling", "regularized", "deep_regularized", "recency"]
    columns = pd.read_csv(ROOT / "data/raw/sample_submission.csv", nrows=0).columns[1:].tolist()
    generators = list(dict.fromkeys(column.removeprefix("result_committed_").rsplit("_t", 1)[0]
                                   for column in columns))
    truth = pd.read_csv(ROOT / "data/raw/kernel/Unit_commitment_decisions.csv",
                        usecols=["Run No", *columns],
                        dtype={column: "uint8" for column in columns}).set_index("Run No")
    rows, specialist_reports = [], []
    for year in [2021, 2022]:
        frames = [pd.read_csv(ROOT / f"data/processed/xgboost_{profile}_{year}_validation_predictions.csv")
                  .set_index("Run No") for profile in profiles]
        cnn_frames = {
            prefix: pd.read_csv(ROOT / f"data/processed/{prefix}_{year}_validation_predictions.csv").set_index("Run No")
            for prefix in cnn_prefixes
        }
        reference = frames[0]
        for frame in [*frames, *cnn_frames.values()]:
            assert frame.index.is_unique, "Validation case IDs must be unique."
            assert frame.index.equals(reference.index), "Validation cases must match in order."
            assert frame.columns.tolist() == columns, "Prediction columns must match the submission."
            values = frame.to_numpy()
            assert np.isfinite(values).all(), "Predictions must be finite."
            assert ((values >= 0) & (values <= 1)).all(), "Predictions must be probabilities."
        labels = truth.loc[reference.index, columns].to_numpy().reshape(-1, len(generators), HOURS)
        ensemble = sum(frame.to_numpy() for frame in frames) / len(profiles)
        specialist_positions, specialist_values = [], None
        if args.fixed_specialist_prefix:
            stem = f"{args.fixed_specialist_prefix}_{year}"
            report = json.loads((ROOT / f"reports/{stem}_metrics.json").read_text())
            target = report["target_generator"]
            assert target in generators, "The specialist must target one known generator."
            selected_columns, _ = output_targets(columns, generators, target)
            assert report["output_columns"] == selected_columns, "Specialist output columns must match its generator."
            specialist = pd.read_csv(ROOT / f"data/processed/{stem}_validation_predictions.csv").set_index("Run No")
            assert specialist.index.is_unique and specialist.index.equals(reference.index), "Specialist validation cases must match in order."
            assert specialist.columns.tolist() == selected_columns, "Specialist prediction columns must match its report."
            specialist_values = specialist.to_numpy()
            assert np.isfinite(specialist_values).all(), "Specialist predictions must be finite."
            assert ((specialist_values >= 0) & (specialist_values <= 1)).all(), "Specialist predictions must be probabilities."
            specialist_positions = [columns.index(column) for column in selected_columns]
            specialist_reports.append({"year": year, "source": stem, "target_generator": target})

        def blend(neural):
            if specialist_values is not None:
                neural = neural.copy()
                neural[:, specialist_positions] = specialist_values
            return .8 * ensemble + .2 * neural

        variants = {"equal_weight_xgboost_ensemble": ensemble}
        for prefix, frame in cnn_frames.items():
            variants[prefix] = frame.to_numpy()
            variants[f"xgboost_{prefix}_80_20"] = blend(frame.to_numpy())
        if args.average_with_baseline:
            average = (cnn_frames[args.baseline_prefix].to_numpy()
                       + cnn_frames[args.average_with_baseline].to_numpy()) / 2
            name = f"{args.average_with_baseline}_mean_with_baseline"
            variants[name] = average
            variants[f"xgboost_{name}_80_20"] = blend(average)
        for model, predictions in variants.items():
            probabilities = predictions.reshape(labels.shape)
            rows.append({"year": year, "model": model,
                         "micro_roc_auc": float(roc_auc_score(labels.ravel(), probabilities.ravel())),
                         "generator_auc": {
                             generator: float(roc_auc_score(labels[:, i].ravel(), probabilities[:, i].ravel()))
                             for i, generator in enumerate(generators)},
                         "validation_cases": len(reference)})
    report = {"cnn_prefix": experiments[0] if len(experiments) == 1 else experiments,
              "baseline_prefix": args.baseline_prefix,
              "average_with_baseline": args.average_with_baseline,
              "fixed_specialist_prefix": args.fixed_specialist_prefix,
              "fixed_specialist_sources": specialist_reports,
              "xgboost_member_weights": {profile: .2 for profile in profiles},
              "blend_weights": {"xgboost_ensemble": .8, "cnn": .2},
              "results": rows}
    with output.open("x") as file:
        file.write(json.dumps(report, indent=2) + "\n")
    print(pd.DataFrame(rows).pivot(index="model", columns="year", values="micro_roc_auc").round(6).to_string())


if __name__ == "__main__":
    main()
