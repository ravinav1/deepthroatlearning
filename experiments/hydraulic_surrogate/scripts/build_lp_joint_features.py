"""Build the frozen joint Tokke/Vinje convex-mixture LP cache without labels."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from sintef_project.lp_features import HydroCaseData
from sintef_project.lp_joint_plant import JointPlantLP, build_joint_curve, result_joint_features, joint_math_specification


def checksum(path: Path) -> str:
    """Return SHA-256 of one complete file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Solve selected cases and persist features, raw solutions and audit metadata."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="lp-joint-full-001")
    parser.add_argument("--case-ids", nargs="+", type=int)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--time-limit", type=float, default=20.0)
    args = parser.parse_args()
    out = ROOT / "artifacts" / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = {"run_id": args.run_id, "timestamp_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
                "command": shlex.join([sys.executable, *sys.argv]), "math": joint_math_specification(), "arguments": vars(args),
                "environment": {"platform": platform.platform(), "python": sys.version, "cpu_count": os.cpu_count(),
                                "packages": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scipy", "PyYAML")}},
                "data": {}, "source": {}, "cases": [], "seeds": "None; deterministic LP; no training or evaluation labels read",
                "validation_score": None, "kaggle": {"submission_id": None, "score": None, "rank": None, "observation_time": None}}
    for path in sorted((ROOT / "sintef_project").rglob("*")):
        if path.suffix in (".csv", ".yaml"):
            manifest["data"][str(path.relative_to(ROOT))] = checksum(path)
    for path in (Path(__file__), ROOT / "sintef_project/lp_joint_plant.py", ROOT / "sintef_project/lp_head_update.py", ROOT / "sintef_project/lp_shared_loss_update.py", ROOT / "sintef_project/lp_features.py", ROOT / "sintef_project/features.py", ROOT / "sintef_project/physics.py"):
        relative = path.relative_to(ROOT)
        manifest["source"][str(relative)] = checksum(path)
        target = out / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    try:
        data = HydroCaseData(ROOT)
        solver = JointPlantLP(data.network)
        if args.case_ids:
            indices = [int(np.flatnonzero(data.cases["Run No"].to_numpy() == case_id)[0]) for case_id in args.case_ids]
        elif args.all:
            indices = list(range(len(data.cases)))
        else:
            indices = np.linspace(0, data.n_train - 1, 16, dtype=int).tolist()
        manifest["selected_case_ids"] = data.cases.iloc[indices]["Run No"].tolist()
        manifest["selected_case_dates"] = data.cases.iloc[indices].starttime.astype(str).tolist()
        manifest["matrix"] = {"note": "Base matrix only; extra convex-mixture variables and2*168 inequality rows added percase, exact variable counts in case records", "variables": solver.n_variables, "equalities": solver.a_eq.shape[0], "inequalities": solver.a_ub.shape[0], "eq_nonzero": solver.a_eq.nnz, "ub_nonzero": solver.a_ub.nnz}
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        feature_frames = []
        for index in indices:
            tick = time.monotonic()
            case = data.case(index)
            curves = [build_joint_curve(data, index, plant) for plant in ["Tokke", "Vinje"]]
            result = solver.solve(case, curves, time_limit=args.time_limit)
            features = result_joint_features(case, result, data.network)
            feature_frames.append(features.to_numpy())
            case_id = int(data.cases.iloc[index]["Run No"])
            record = {"run_no": case_id, "starttime": str(data.cases.iloc[index].starttime), "runtime_seconds": time.monotonic() - tick,
                      **{key: result[key] for key in ("objective", "iterations", "status", "balance_residual", "total_balance_error", "negative_price_hours", "n_variables", "maximum_convex_mass", "old_joint_q_max")},
                      "max_minimum_slack_mm3": float(result["minimum_slack"].max()), "max_capacity_slack_mm3": float(result["maximum_slack"].max()),
                      "max_bypass_slack_mm3": float(result["bypass_slack"].max()), "max_vest_slack_mm3": float(result["vest_slack"].max())}
            record["joint_vertices"] = {curve.plant: len(curve.total_flow) for curve in curves}
            record["feasible_grid_points"] = {curve.plant: curve.specification["feasible_count_including_off"] for curve in curves}
            manifest["cases"].append(record)
            curve_arrays = {}
            for curve in curves:
                for name in ["total_flow", "total_power", "unit_flow", "unit_power", "unit_on", "unit_indices", "upper_vertex"]:
                    curve_arrays[f"curve_{curve.plant}_{name}"] = getattr(curve, name)
            np.savez_compressed(out / f"case_{case_id:04d}.npz", **result, **curve_arrays)
            print(f"CASE {case_id} {record['runtime_seconds']:.3f}s balance={result['balance_residual']:.2e} minslack={record['max_minimum_slack_mm3']:.4g}", flush=True)
            (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        np.savez_compressed(out / "features.npz", features=np.stack(feature_frames), case_ids=data.cases.iloc[indices]["Run No"].to_numpy(), columns=np.array(features.columns, dtype=str), units=np.array(data.network.units, dtype=str))
        manifest["feature_names"] = list(features.columns)
        manifest["feature_shape"] = list(np.stack(feature_frames).shape)
        manifest["case_runtime_median"] = float(np.median([record["runtime_seconds"] for record in manifest["cases"]]))
        manifest["status"] = "complete"
    except BaseException:
        manifest["status"] = "failed"
        manifest["error"] = traceback.format_exc()
        raise
    finally:
        manifest["runtime_seconds"] = time.monotonic() - started
        manifest["artifacts"] = {str(path.relative_to(out)): checksum(path) for path in sorted(out.rglob("*")) if path.is_file() and path.name != "manifest.json"}
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print("STATUS", manifest["status"], "SECONDS", manifest["runtime_seconds"], flush=True)


if __name__ == "__main__":
    main()
