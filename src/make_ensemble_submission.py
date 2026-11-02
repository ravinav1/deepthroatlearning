"""Combine final-fit XGBoost submissions, with optional CNN probabilities."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from features import HOURS, ROOT


def main():
    parser = argparse.ArgumentParser(description="Create the five-XGBoost ensemble and calibrated variant.")
    parser.add_argument("--output", default=str(ROOT / "submissions/submission_xgb_ensemble.csv"))
    parser.add_argument("--calibrated-output", default=str(ROOT / "submissions/submission_xgb_ensemble_calibrated.csv"))
    parser.add_argument("--cnn", help="Optional final-fit CNN submission to include at 20%%.")
    parser.add_argument("--sequence-output", default=str(ROOT / "submissions/submission_xgb_cnn_80_20.csv"))
    args = parser.parse_args()
    output, calibrated_output = Path(args.output), Path(args.calibrated_output)
    report_path = ROOT / f"reports/{output.stem}_run.json"
    paths = [output, calibrated_output, report_path]
    if args.cnn:
        paths.append(Path(args.sequence_output))
    for path in paths:
        assert not path.exists(), f"Artifact already exists: {path}"

    template = pd.read_csv(ROOT / "data/raw/sample_submission.csv")
    columns = template.columns[1:].tolist()

    def read_probabilities(path):
        frame = pd.read_csv(path)
        assert frame.shape == template.shape and frame.columns.tolist() == template.columns.tolist()
        assert frame["Run No"].equals(template["Run No"])
        values = frame[columns].to_numpy()
        assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
        return values

    sources = [ROOT / "submissions/submission_xgboost_price.csv"] + [
        ROOT / f"submissions/submission_xgb_{profile}.csv"
        for profile in ["sampling", "regularized", "deep_regularized", "recency"]
    ]
    ensemble = sum(read_probabilities(path) for path in sources) / len(sources)
    calibration_rows = json.loads((ROOT / "reports/generalization_calibration_comparison.json").read_text())
    calibration = next(row for row in calibration_rows if row["model"] == "equal_weight_ensemble")
    generators = list(dict.fromkeys(column.removeprefix("result_committed_").rsplit("_t", 1)[0]
                                   for column in columns))
    assert calibration["shared_logit_slope"] > 0 and set(generators) == set(calibration["generator_offsets"])
    probabilities = np.clip(ensemble.reshape(len(template), len(generators), HOURS), 1e-6, 1 - 1e-6)
    offsets = np.array([calibration["generator_offsets"][generator] for generator in generators])
    calibrated = expit(calibration["shared_logit_slope"] * np.log(probabilities / (1 - probabilities))
                       + offsets[None, :, None]).reshape(ensemble.shape)

    outputs = [(output, ensemble, "equal_weight_xgboost"),
               (calibrated_output, calibrated, "calibrated_equal_weight_xgboost")]
    if args.cnn:
        cnn = read_probabilities(Path(args.cnn))
        outputs.append((Path(args.sequence_output), .8 * ensemble + .2 * cnn, "xgboost_cnn_80_20"))
    for path, values, _ in outputs:
        assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
        frame = pd.DataFrame(values, columns=columns)
        frame.insert(0, "Run No", template["Run No"].to_numpy())
        frame.to_csv(path, index=False, mode="x")
        print(f"Saved {len(frame)} cases to {path}", flush=True)

    report = {
        "xgboost_sources": [{"file": str(path), "weight": 1 / len(sources)} for path in sources],
        "calibration_source": "reports/generalization_calibration_comparison.json",
        "calibration_fit_year": calibration["fit_year"],
        "shared_logit_slope": calibration["shared_logit_slope"],
        "generator_offsets": calibration["generator_offsets"],
        "probability_clip": [1e-6, 1 - 1e-6],
        "test_cases": len(template),
        "outputs_per_case": len(columns),
        "outputs": [{"file": str(path), "variant": variant} for path, _, variant in outputs],
    }
    if args.cnn:
        report["sequence_blend"] = {"xgboost_ensemble": .8, "cnn": .2, "cnn_file": args.cnn}
    with report_path.open("x") as file:
        json.dump(report, file, indent=2)
        file.write("\n")


if __name__ == "__main__":
    main()
