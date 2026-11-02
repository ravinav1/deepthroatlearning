"""Verify frozen comparison source hashes and score with the supplied metric."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from features import ROOT


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    output = ROOT / "reports/round3_frozen_verification.json"
    assert not output.exists()
    comparison_path = ROOT / "reports/round3_frozen_ensemble_comparison.json"
    comparison = json.loads(comparison_path.read_text())
    checks = {}
    for path, expected in comparison["source_hashes"].items():
        actual = digest(ROOT / path)
        assert actual == expected, path
        checks[path] = actual
    for path in comparison["inference_reports"]:
        source = json.loads((ROOT / path).read_text())
        assert source["new_fitting"] is False and source["scaler_refitted"] is False
        assert source["cpu_threads"] == 1
        assert source["last_training_horizon_end"] == "2021-01-01 00:00:00+00:00"
        assert source["training_cases"] == 2186
        assert digest(ROOT / source["prediction_path"]) == source["prediction_sha256"]
        checks[path] = digest(ROOT / path)
        for artifact, expected in source["source_hashes"].items():
            assert digest(ROOT / artifact) == expected
            checks[artifact] = expected
    for name in ("xgb", "lgb", "transformer"):
        path = ROOT / f"reports/round2_frozen_{name}_2022_metrics.json"
        source = json.loads(path.read_text())
        model_key = "source_checkpoint" if name == "transformer" else "source_model"
        assert digest(ROOT / source[model_key]) == source[model_key + "_sha256"]
        checks[source[model_key]] = source[model_key + "_sha256"]
        if name != "transformer":
            assert digest(ROOT / source["source_report"]) == source["source_report_sha256"]
        checks[str(path.relative_to(ROOT))] = digest(path)
    columns = pd.read_csv(ROOT / "data/raw/sample_submission.csv", nrows=0).columns[1:].tolist()
    assert len(columns) == 2352
    truth = pd.read_csv(ROOT / "data/raw/kernel/Unit_commitment_decisions.csv", usecols=["Run No", "starttime", *columns])
    truth["starttime"] = pd.to_datetime(truth.starttime, utc=True)
    cases = truth.loc[truth.starttime.dt.year == 2022].sort_values("starttime").reset_index(drop=True)
    assert len(cases) == 365
    assert np.isin(cases[columns].to_numpy(), [0, 1]).all()
    spec = importlib.util.spec_from_file_location("provided_metric", ROOT / "evaluation/kaggle_metric.py")
    metric = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metric)
    results = []
    for variant in ("old_frozen", "latest_frozen"):
        path = ROOT / f"data/processed/round3_frozen_{variant}_2022_validation_predictions.csv"
        predictions = pd.read_csv(path, float_precision="round_trip")
        assert predictions.columns.tolist() == ["Run No", *columns]
        assert predictions["Run No"].equals(cases["Run No"])
        assert predictions["Run No"].is_unique
        values = predictions[columns].to_numpy()
        assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
        checks[str(path.relative_to(ROOT))] = digest(path)
        for subset in ("all", "jan7_onward"):
            mask = np.ones(len(cases), dtype=bool) if subset == "all" else (cases.starttime >= pd.Timestamp("2022-01-07", tz="UTC"))
            score = metric.score(cases.loc[mask, ["Run No", *columns]], predictions.loc[mask])
            expected = next(r["micro_roc_auc"] for r in comparison["results"]
                            if r["year"] == 2022 and r["variant"] == variant and r["subset"] == subset)
            assert score == expected, (score, expected)
            results.append({"variant": variant, "subset": subset, "cases": int(np.sum(mask)), "official_micro_roc_auc": score})
    report = {"passed": True, "comparison": str(comparison_path.relative_to(ROOT)),
              "comparison_sha256": digest(comparison_path), "checked_source_hashes": checks,
              "script_sha256": {"src/round3_frozen_evaluation.py": digest(ROOT / "src/round3_frozen_evaluation.py"),
                                "src/round3_frozen_verify.py": digest(__file__)},
              "results": results,
              "scope": "All new source checkpoints unchanged, existing frozen source model hashes match their prior reports, training boundary/count and saved-scaler inference asserted, exact full-width CSV IDs/order verified, provided metric reproduces all four frozen 2022 scores."}
    with output.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"passed": True, "checked_files": len(checks), "scores": results}, indent=2))


if __name__ == "__main__":
    main()
