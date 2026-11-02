"""Lightweight, independent accounting audit of a frozen joint-plant cache."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shlex
import sys
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from sintef_project.lp_features import HydroCaseData
from sintef_project.lp_joint_plant import build_joint_curve, simultaneous_points


def digest(path: Path) -> str:
    """Hash a file without constructing another full feature matrix."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    """Verify stored solutions, costs, input restrictions and source closure."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="lp-joint-audit-001")
    args = parser.parse_args()
    out = ROOT / "artifacts" / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    cache = ROOT / "artifacts/lp-joint-full-001"
    original = json.loads((cache / "manifest.json").read_text())
    ids = [1, 2008, 2193, 2461, 2653, 2690, 2923, 3647]
    report = {"run_id": args.run_id, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "status": "running", "command": shlex.join([sys.executable, *sys.argv]),
              "source": {str(Path(__file__).relative_to(ROOT)): digest(Path(__file__))},
              "cache_manifest_sha256": digest(cache / "manifest.json"),
              "selected_case_ids": ids, "cases": [], "source_verified": {}, "data_verified": {},
              "environment": {"platform": platform.platform(), "python": sys.version,
                              "packages": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scipy", "PyYAML"]}},
              "model_training": False, "LP_solves": 0, "labels_read": False,
              "validation_score": None, "kaggle": None}
    try:
        assert original["status"] == "complete"
        assert not any(name.endswith("train.py") for name in original["source"])
        for category in ["source", "data"]:
            for relative, expected in original[category].items():
                assert digest(ROOT / relative) == expected, relative
                report[category + "_verified"][relative] = expected
        assert digest(cache / "features.npz") == original["artifacts"]["features.npz"]
        report["feature_artifact_sha256"] = original["artifacts"]["features.npz"]
        real_reader = pd.read_csv
        requested = []

        def guarded_reader(path, *args, **kwargs):
            if Path(path).name == "Unit_commitment_decisions.csv":
                assert set(kwargs.get("usecols", [])) == {"Run No", "starttime"}
                requested.append(kwargs["usecols"])
                frame = real_reader(path, *args, **kwargs)
                # This extra illegal field must have no effect on case construction.
                frame["audit_poisoned_label"] = np.nan
                return frame
            return real_reader(path, *args, **kwargs)

        with patch("pandas.read_csv", guarded_reader):
            data = HydroCaseData(ROOT)
        assert requested
        report["label_reader_usecols"] = requested
        network = data.network
        for case_id in ids:
            index = int(np.flatnonzero(data.cases["Run No"].to_numpy() == case_id)[0])
            case = data.case(index)
            archive_path = cache / f"case_{case_id:04d}.npz"
            assert digest(archive_path) == original["artifacts"][archive_path.name]
            with np.load(archive_path, allow_pickle=False) as handle:
                result = {key: handle[key] for key in handle.files}
            # Independently accumulate every edge into every reservoir balance.
            change = np.asarray(case["natural"], dtype=np.float64).copy()
            for flow, up, down in [(result["flow"], network.unit_up, network.unit_down),
                                   (result["bypass"], network.bypass_up, network.bypass_down),
                                   (result["spill"], np.arange(len(network.pools)), network.spill_down)]:
                for k, (source, destination) in enumerate(zip(up, down)):
                    change[source] -= flow[k]
                    if destination >= 0:
                        change[destination] += flow[k]
            expected = case["initial"][:, None] + change.cumsum(axis=1)
            balance = float(np.max(np.abs(expected - result["storage"])))
            assert balance < 1e-8
            assert result["storage"].min() >= -1e-8
            assert result["old_joint_q_max"] == 0
            penalty = 1e7 + 100 * np.max(np.abs(case["terminal_value"]))
            objective = np.sum(case["price"][None, :] * result["power"])
            objective += case["terminal_value"] @ result["storage"][:, -1]
            objective -= 10 / .0036 * result["bypass"].sum() + 1000 / .0036 * result["spill"].sum()
            objective -= penalty * sum(result[key].sum() for key in ["minimum_slack", "maximum_slack", "bypass_slack", "vest_slack"])
            error = abs(float(objective - result["objective"]))
            assert error < 1e-5 * max(1., abs(float(objective)))
            record = {"case_id": case_id, "archive_sha256": digest(archive_path),
                      "balance_max_mm3": balance, "objective_absolute_difference": error,
                      "negative_price_hours": int((case["price"] < 0).sum()), "plants": {}}
            for plant in ["Tokke", "Vinje"]:
                curve = build_joint_curve(data, index, plant)
                weights = result[f"mode_weights_{plant}"]
                for field in ["total_flow", "total_power", "unit_flow", "unit_power", "unit_on", "unit_indices", "upper_vertex"]:
                    np.testing.assert_array_equal(getattr(curve, field), result[f"curve_{plant}_{field}"])
                assert weights.min() >= -1e-8 and weights.sum(axis=0).max() <= 1 + 1e-8
                np.testing.assert_allclose(curve.unit_flow.T @ weights, result["flow"][curve.unit_indices], atol=1e-10)
                np.testing.assert_allclose(curve.unit_power.T @ weights, result["power"][curve.unit_indices], atol=1e-8)
                on = curve.unit_on.T @ weights
                np.testing.assert_allclose(on, result["joint_on_fraction"][curve.unit_indices], atol=1e-10)
                for k, unit in enumerate(curve.specification["units"]):
                    params = data.model["generator"][unit]
                    power = result["power"][curve.unit_indices[k]]
                    assert (power >= params["p_min"] * on[k] - 1e-8).all()
                    assert (power <= params["p_max"] * on[k] + 1e-8).all()
                power, _, valid = simultaneous_points(data.model["generator"], data.model["plant"][plant],
                                                      curve.specification["initial_gross_head_m"],
                                                      curve.specification["units"], curve.unit_flow / .0036)
                assert valid.all()
                np.testing.assert_allclose(power, curve.unit_power, atol=1e-8)
                record["plants"][plant] = {"vertices": len(curve.total_flow),
                                           "maximum_weight_sum": float(weights.sum(axis=0).max()),
                                           "negative_hour_lower_weight": float(weights[~curve.upper_vertex][:, case["price"] < 0].sum())}
            # Poison every volume except the legal case-start value in memory.
            # No raw files are touched, and no solver is run.
            volume = data.volume
            data.volume = volume.copy()
            data.volume.loc[data.volume.index != data.cases.iloc[index].starttime] = np.nan
            poisoned = data.case(index)
            for key in case:
                np.testing.assert_array_equal(case[key], poisoned[key])
            for plant in ["Tokke", "Vinje"]:
                curve = build_joint_curve(data, index, plant)
                np.testing.assert_array_equal(curve.unit_power, result[f"curve_{plant}_unit_power"])
            data.volume = volume
            record["post_start_volume_poison_invariant"] = True
            report["cases"].append(record)
        trainer = ROOT / "artifacts/lgb-normalized-lp-joint-001/manifest.json"
        active = json.loads(trainer.read_text())
        report["trainer_manifest_observed_sha256"] = digest(trainer)
        report["trainer_shape"] = active["shape"]
        assert active["shape"] == [3647 * 168, 404]
        spec = active["feature_spec"]["lp_joint"]
        assert spec["manifest_sha256"] == report["cache_manifest_sha256"]
        assert spec["features_sha256"] == report["feature_artifact_sha256"]
        assert spec["source_verified"] == original["source"]
        assert spec["data_verified"] == original["data"]
        assert spec["columns"] == original["feature_names"]
        assert spec["mathematics"] == original["math"]
        report["trainer_joint_spec_verified"] = True
        report["status"] = "complete"
    except BaseException:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        raise
    finally:
        report["runtime_seconds"] = time.monotonic() - started
        (out / "manifest.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({key: report.get(key) for key in ["status", "runtime_seconds", "trainer_shape", "trainer_joint_spec_verified"]}))


if __name__ == "__main__":
    main()
