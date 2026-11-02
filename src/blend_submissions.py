"""Blend two saved prediction files; default is 75% XGBoost and 25% LightGBM."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from features import HOURS, ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", "--xgboost", dest="first", default=str(ROOT / "submissions/submission_xgboost.csv"))
    parser.add_argument("--second", "--lightgbm", dest="second", default=str(ROOT / "submissions/submission_topology.csv"))
    parser.add_argument("--first-weight", type=float, default=0.75)
    parser.add_argument("--preserve-generator", help="Keep this generator's probabilities from --preserve-from.")
    parser.add_argument("--preserve-from", help="Existing submission containing the probabilities to preserve.")
    parser.add_argument("--output", default=str(ROOT / "submissions/submission_xgb_lgb_75_25.csv"))
    args = parser.parse_args()
    output = Path(args.output)
    report_path = ROOT / f"reports/{output.stem}_run.json"
    assert 0 <= args.first_weight <= 1
    assert bool(args.preserve_generator) == bool(args.preserve_from), "Use --preserve-generator and --preserve-from together."
    for path in (output, report_path):
        assert not path.exists(), f"Artifact already exists: {path}; choose another --output filename."

    first_path = Path(args.first)
    second_path = Path(args.second)
    first = pd.read_csv(first_path)
    second = pd.read_csv(second_path)
    preserved_path = Path(args.preserve_from) if args.preserve_from else None
    preserved = pd.read_csv(preserved_path, float_precision="round_trip") if preserved_path else None
    template = pd.read_csv(ROOT / "data/raw/sample_submission.csv")
    submissions = [first, second] + ([preserved] if preserved is not None else [])
    for submission in submissions:
        assert submission.shape == template.shape
        assert submission.columns.tolist() == template.columns.tolist()
        assert submission["Run No"].equals(template["Run No"])
        values = submission.drop(columns="Run No").to_numpy()
        assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()

    columns = template.columns.drop("Run No")
    blended = first.copy()
    blended[columns] = args.first_weight * first[columns] + (1 - args.first_weight) * second[columns]
    if preserved is not None:
        selected_columns = [f"result_committed_{args.preserve_generator}_t{hour}" for hour in range(HOURS)]
        assert all(column in columns for column in selected_columns), "Unknown generator to preserve."
        blended[selected_columns] = preserved[selected_columns]
    blended.to_csv(output, index=False, mode="x")
    report = {
        "sources": [{"file": str(first_path), "weight": args.first_weight},
                    {"file": str(second_path), "weight": 1 - args.first_weight}],
        "test_cases": len(blended),
        "outputs_per_case": len(columns),
        "submission": str(output),
    }
    if preserved_path:
        report["preserved_generator"] = args.preserve_generator
        report["preserved_source"] = str(preserved_path)
    with report_path.open("x") as file:
        file.write(json.dumps(report, indent=2) + "\n")
    print(f"Saved {len(blended)} cases to {output}")


if __name__ == "__main__":
    main()
