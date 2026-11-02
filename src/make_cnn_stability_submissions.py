"""Combine the existing XGBoost ensemble with three CNN stability variants."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from features import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/cnn_stability.json"))
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    outputs = {method: ROOT / f"submissions/submission_xgb_cnn_{method}_80_20.csv"
               for method in ["seeds", "ema", "swa"]}
    report_path = ROOT / "reports/submission_cnn_stability_blends_run.json"
    for path in [*outputs.values(), report_path]:
        assert not path.exists(), f"Artifact already exists: {path}"
    template = pd.read_csv(ROOT / "data/raw/sample_submission.csv")
    columns = template.columns[1:].tolist()

    def read(path):
        frame = pd.read_csv(path)
        assert frame.shape == template.shape and frame.columns.tolist() == template.columns.tolist()
        assert frame["Run No"].equals(template["Run No"])
        values = frame[columns].to_numpy()
        assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
        return values

    xgb_path = ROOT / "submissions/submission_xgb_ensemble.csv"
    xgb = read(xgb_path)
    seed_paths = [ROOT / "submissions/submission_cnn.csv" if seed == 42 else
                  ROOT / f"data/processed/submission_cnn_seed{seed}_predictions.csv"
                  for seed in config["seeds"]]
    sources = {"seeds": seed_paths,
               "ema": [ROOT / "data/processed/submission_cnn_averaging_ema_predictions.csv"],
               "swa": [ROOT / "data/processed/submission_cnn_averaging_swa_predictions.csv"]}
    variants = {}
    for method, paths in sources.items():
        cnn = sum(read(path) for path in paths) / len(paths)
        probabilities = config["xgboost_weight"] * xgb + config["cnn_weight"] * cnn
        frame = pd.DataFrame(probabilities, columns=columns)
        frame.insert(0, "Run No", template["Run No"].to_numpy())
        frame.to_csv(outputs[method], index=False, mode="x")
        variants[method] = {"output": str(outputs[method]),
                            "cnn_sources": [{"file": str(path), "weight": 1 / len(paths)} for path in paths]}
        print(f"Saved {len(frame)} cases to {outputs[method]}", flush=True)
    report = {"config": config, "xgboost_source": str(xgb_path),
              "xgboost_weight": config["xgboost_weight"], "cnn_weight": config["cnn_weight"],
              "test_cases": len(template), "outputs_per_case": len(columns), "variants": variants}
    with report_path.open("x") as file:
        json.dump(report, file, indent=2)
        file.write("\n")


if __name__ == "__main__":
    main()
