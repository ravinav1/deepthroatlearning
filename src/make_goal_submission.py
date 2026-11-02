"""Export a historically confirmed plant blend from saved component predictions."""

import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from features import ROOT
from plant_blend import column_groups


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = json.loads(config_path.read_text())
    output = ROOT / args.output
    report_path = ROOT / f"reports/{output.stem}_run.json"
    for path in (output, report_path):
        assert not path.exists(), f"Artifact already exists: {path}"

    historical_path = ROOT / config["historical_report"]
    historical = json.loads(historical_path.read_text())
    confirmed = []
    for year in (2021, 2022):
        baseline = [row for row in historical["results"] if row["year"] == year and row["variant"] == "baseline"]
        selected = [row for row in historical["results"]
                    if row["year"] == year and row["variant"] == config["historical_variant"]]
        assert len(baseline) == len(selected) == 1, f"Missing or ambiguous historical result for {year}"
        base_auc = float(baseline[0]["micro_roc_auc"])
        selected_auc = float(selected[0]["micro_roc_auc"])
        assert np.isfinite([base_auc, selected_auc]).all() and 0 <= base_auc < selected_auc <= 1, \
            f"Selected blend must improve the baseline in {year}"
        confirmed.append({"year": year, "baseline_micro_roc_auc": base_auc,
                          "selected_micro_roc_auc": selected_auc, "delta_vs_baseline": selected_auc - base_auc})

    template_path = ROOT / "data/raw/sample_submission.csv"
    template = pd.read_csv(template_path, float_precision="round_trip")
    columns = template.columns[1:].tolist()
    assert template.columns[0] == "Run No" and len(columns) == 2352 and template["Run No"].is_unique
    groups = column_groups(columns)
    assert config["plant_weights"].keys() == groups.keys(), "Recipe must cover every plant exactly"
    frames = {}
    sources = {}
    for name, relative_path in config["sources"].items():
        path = ROOT / relative_path
        frame = pd.read_csv(path, float_precision="round_trip")
        assert frame.columns.is_unique and frame.columns[0] == "Run No"
        assert frame["Run No"].is_unique and frame["Run No"].equals(template["Run No"])
        provided = frame.columns[1:].tolist()
        assert provided and set(provided).issubset(columns), f"Unexpected source columns: {name}"
        values = frame[provided].to_numpy()
        assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all(), f"Invalid probabilities: {name}"
        frames[name] = frame
        sources[name] = {"path": relative_path, "sha256": checksum(path), "outputs_per_case": len(provided)}

    probabilities = np.zeros((len(template), len(columns)), dtype=np.float64)
    for plant, weights in config["plant_weights"].items():
        assert weights and set(weights).issubset(frames), f"Missing prediction source for {plant}"
        amounts = np.asarray(list(weights.values()), dtype=float)
        assert np.isfinite(amounts).all() and (amounts >= 0).all() and abs(amounts.sum() - 1) < 1e-12, \
            f"Weights must be nonnegative and sum to one: {plant}"
        indices = groups[plant]
        names = [columns[i] for i in indices]
        for source, weight in weights.items():
            assert set(names).issubset(frames[source].columns), f"Source {source} lacks required columns for {plant}"
            probabilities[:, indices] += weight * frames[source][names].to_numpy()
    assert np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all()
    submission = template.copy()
    submission[columns] = probabilities
    submission.to_csv(output, index=False, mode="x")
    report = {
        "config": config, "config_sha256": checksum(config_path),
        "historical_source": config["historical_report"], "historical_sha256": checksum(historical_path),
        "historical_results": confirmed, "test_cases": len(template), "outputs_per_case": len(columns),
        "template_sha256": checksum(template_path), "sources": sources,
        "submission": str(output), "submission_sha256": checksum(output),
        "checks": {"unique_case_ids": True, "template_case_order": True, "template_column_order": True,
                   "all_plant_weights_sum_to_one": True, "finite_probabilities_in_unit_interval": True,
                   "improves_both_historical_years": True},
        "note": "Saved predictions only; no fitting, hard thresholds or weight optimization. Historical results use reused development years, not an untouched holdout. Kaggle score remains unknown.",
    }
    with report_path.open("x") as file:
        file.write(json.dumps(report, indent=2) + "\n")
    print(f"Saved {len(submission)} cases and {len(columns)} probabilities per case to {output}", flush=True)


if __name__ == "__main__":
    main()
