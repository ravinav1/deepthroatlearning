"""Replace selected generator probabilities using saved specialist predictions."""

import argparse
import json

import numpy as np
import pandas as pd

from features import ROOT
from generator_specialist import output_targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/submission_generator_specialists.json")
    parser.add_argument("--output", default="submissions/submission_xgb_cnn_haukeli_specialist_80_20.csv")
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text())
    output = ROOT / args.output
    report_path = ROOT / f"reports/{output.stem}_run.json"
    for path in (output, report_path):
        assert not path.exists(), f"Artifact already exists: {path}"
    template = pd.read_csv(ROOT / "data/raw/sample_submission.csv")
    columns = template.columns[1:].tolist()
    generators = list(dict.fromkeys(column.removeprefix("result_committed_").rsplit("_t", 1)[0]
                                   for column in columns))

    def read(path, selected_columns):
        frame = pd.read_csv(ROOT / path, float_precision="round_trip")
        assert frame.columns.tolist() == ["Run No", *selected_columns]
        assert frame["Run No"].is_unique and frame["Run No"].equals(template["Run No"])
        values = frame[selected_columns].to_numpy()
        assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
        return frame

    baseline = read(config["baseline"], columns)
    xgb = read(config["xgboost_source"], columns)
    result = baseline.copy()
    selection = json.loads((ROOT / config["selection_source"]).read_text())
    selected_rows = [row for row in selection["results"] if row["variant"] == config["selected_variant"]]
    routes = {generator: component["route"] for generator, component in config["components"].items()}
    assert selected_rows and all(row["generator_routes"] == routes for row in selected_rows)
    for generator, component in config["components"].items():
        target_columns, _ = output_targets(columns, generators, generator)
        source = json.loads((ROOT / component["report"]).read_text())
        assert source["target_generator"] == generator and source["output_columns"] == target_columns
        assert source["training_cases"] == 2922 and source["test_cases"] == len(template)
        predictions = read(component["predictions"], target_columns)
        route = component["route"]
        assert route in ["cnn_component", "direct"]
        values = predictions[target_columns].to_numpy()
        if route == "cnn_component":
            values = config["xgboost_weight"] * xgb[target_columns].to_numpy() + config["cnn_weight"] * values
        result[target_columns] = values
    untouched = [column for column in columns
                 if column.removeprefix("result_committed_").rsplit("_t", 1)[0] not in config["components"]]
    assert np.array_equal(result[untouched].to_numpy(), baseline[untouched].to_numpy())
    values = result[columns].to_numpy()
    assert result.shape == template.shape and np.isfinite(values).all()
    assert ((values >= 0) & (values <= 1)).all()
    result.to_csv(output, index=False, mode="x")
    report = {"config": config, "output": str(output), "test_cases": len(template),
              "outputs_per_case": len(columns), "untouched_outputs_per_case": len(untouched),
              "historical_selection": [{"year": row["year"], "micro_roc_auc": row["micro_roc_auc"],
                                        "delta_vs_baseline": row["delta_vs_baseline"]} for row in selected_rows]}
    with report_path.open("x") as file:
        file.write(json.dumps(report, indent=2) + "\n")
    print(f"Saved {len(result)} cases to {output}")


if __name__ == "__main__":
    main()
