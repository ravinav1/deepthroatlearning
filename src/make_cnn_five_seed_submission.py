"""Export one fixed 80/20 blend of the saved XGBoost and equal five-seed CNN means."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/cnn_five_seeds.json"))
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    assert config["seeds"] == [42, 7, 21, 67, 103]
    assert config["xgboost_weight"] == .8 and config["cnn_weight"] == .2
    output = ROOT / "submissions/submission_xgb_cnn_5seeds_80_20.csv"
    report_path = ROOT / "reports/submission_xgb_cnn_5seeds_80_20_run.json"
    for path in (output, report_path):
        assert not path.exists(), f"Artifact already exists: {path}"
    template = pd.read_csv(ROOT / "data/raw/sample_submission.csv")
    assert template.columns[0] == "Run No" and template["Run No"].is_unique and len(template)
    columns = template.columns[1:].tolist()

    def read(path):
        frame = pd.read_csv(path)
        assert frame.shape == template.shape and frame.columns.tolist() == template.columns.tolist(), \
            f"Template shape/header differs: {path}"
        assert frame["Run No"].is_unique and frame["Run No"].equals(template["Run No"]), f"Case IDs/order differ: {path}"
        values = frame[columns].to_numpy()
        assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all(), f"Invalid probabilities: {path}"
        return values

    xgb_path = ROOT / "submissions/submission_xgb_ensemble.csv"
    seed_paths = {seed: ROOT / "submissions/submission_cnn.csv" if seed == 42 else
                  ROOT / f"data/processed/submission_cnn_seed{seed}_predictions.csv" for seed in config["seeds"]}
    xgb = read(xgb_path)
    cnn = sum(read(path) for path in seed_paths.values()) / len(seed_paths)
    probabilities = config["xgboost_weight"] * xgb + config["cnn_weight"] * cnn
    assert np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all()
    frame = pd.DataFrame(probabilities, columns=columns)
    frame.insert(0, "Run No", template["Run No"].to_numpy())
    report = {"config": config, "output": str(output), "xgboost_source": str(xgb_path),
              "xgboost_weight": config["xgboost_weight"], "cnn_weight": config["cnn_weight"],
              "cnn_sources": [{"seed": seed, "file": str(path), "weight_within_cnn": 1 / len(seed_paths),
                               "weight_in_final_blend": config["cnn_weight"] / len(seed_paths)}
                              for seed, path in seed_paths.items()],
              "test_cases": len(template), "outputs_per_case": len(columns)}
    with output.open("x") as file:
        frame.to_csv(file, index=False)
    with report_path.open("x") as file:
        file.write(json.dumps(report, indent=2) + "\n")
    print(f"Saved {len(frame)} cases to {output}")


if __name__ == "__main__":
    main()
