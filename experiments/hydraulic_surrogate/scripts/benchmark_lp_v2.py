"""Benchmark the isolated directional Haukeli storage correction, with provenance."""

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
import pandas as pd

from sintef_project.lp_features import HydraulicLP
from sintef_project.lp_features_v2 import DirectedHaukeliData, result_features_v2, math_specification_v2, HAUKELI


def digest(path: Path) -> str:
    """Return a SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Solve sixteen predefined cases before considering any full cache build."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--case-ids", nargs="+", type=int,
                        default=[1, 418, 835, 1252, 1669, 2086, 2503, 2922,
                                 2325, 2326, 2404, 2653, 2655, 2656, 2690, 2698])
    args = parser.parse_args()
    started = time.monotonic()
    out = ROOT / "artifacts" / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    manifest = {"run_id": args.run_id, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "status": "running", "command": shlex.join([sys.executable, *sys.argv]),
                "arguments": vars(args), "math": math_specification_v2(), "data": {}, "source": {},
                "environment": {"python": sys.version, "platform": platform.platform(), "cpu_count": os.cpu_count(),
                                "packages": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scipy", "pyyaml"]}},
                "cases": [], "selection": "Eight spread-out cases plus four wet and four dry Haukeli error cases selected by prior error analysis. Not a representative validation score or model selection result.",
                "seed": None, "solver": "HiGHS threads=1, time_limit=20, primal/dual tolerance=1e-7",
                "kaggle": {"submission_id": None, "score": None, "rank": None, "observation_time": None}}
    for path in sorted((ROOT / "sintef_project").rglob("*")):
        if path.suffix in [".csv", ".yaml"]:
            manifest["data"][str(path.relative_to(ROOT))] = digest(path)
    sources = [Path(__file__), ROOT / "sintef_project/lp_features_v2.py", ROOT / "sintef_project/lp_features.py",
               ROOT / "sintef_project/features.py", ROOT / "sintef_project/physics.py", ROOT / "tests/test_lp_features_v2.py"]
    for path in sources:
        relative = path.relative_to(ROOT)
        manifest["source"][str(relative)] = digest(path)
        target = out / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    try:
        data = DirectedHaukeliData(ROOT)
        solver = HydraulicLP(data.network)
        unit = data.network.units.index("Haukeli_G1")
        indices = [int(np.flatnonzero(data.cases["Run No"].to_numpy() == case_id)[0]) for case_id in args.case_ids]
        manifest["selected_case_ids"] = args.case_ids
        manifest["selected_case_dates"] = data.cases.iloc[indices].starttime.astype(str).tolist()
        manifest["matrix"] = {"variables": solver.n_variables, "equalities": solver.a_eq.shape[0], "inequalities": solver.a_ub.shape[0]}
        features = []
        for index in indices:
            tick = time.monotonic()
            case = data.case(index)
            result = solver.solve(case)
            frame = result_features_v2(case, result, data)
            features.append(frame.to_numpy())
            case_id = int(data.cases.iloc[index]["Run No"])
            prior_path = ROOT / "artifacts/lp-full-001" / f"case_{case_id:04d}.npz"
            prior = np.load(prior_path, allow_pickle=False)
            q = result["flow"][unit] / case["qmin"][unit]
            prior_q = prior["flow"][unit] / case["qmin"][unit]
            record = {"run_no": case_id, "runtime_seconds": time.monotonic() - tick,
                      "prior_sha256": digest(prior_path), "qmin_m3s": float(case["qmin"][unit] / .0036),
                      "prior_qmin_ratio": prior_q.tolist(), "qmin_ratio": q.tolist(),
                      "prior_nonzero_hours": int((prior_q > 1e-8).sum()), "nonzero_hours": int((q > 1e-8).sum()),
                      "river_volume_mm3": float(result["bypass"][-1].sum()),
                      "vatjern_end_mm3": float(result["storage"][HAUKELI, -1]),
                      "langeidvatn_end_mm3": float(result["storage"][-1, -1]),
                      **{key: result[key] for key in ["objective", "balance_residual", "total_balance_error", "iterations", "negative_price_hours"]},
                      "max_slack_mm3": float(max(result[key].max() for key in ["minimum_slack", "maximum_slack", "bypass_slack", "vest_slack"]))}
            manifest["cases"].append(record)
            np.savez_compressed(out / f"case_{case_id:04d}.npz", **result)
            print(f"CASE {case_id} {record['runtime_seconds']:.3f}s Haukeli nonzero {record['prior_nonzero_hours']}->{record['nonzero_hours']} balance={record['total_balance_error']:.3g}", flush=True)
            (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        np.savez_compressed(out / "features.npz", features=np.stack(features), case_ids=np.array(args.case_ids),
                            columns=np.array(frame.columns, dtype=str), units=np.array(data.network.units, dtype=str))
        manifest["feature_names"] = frame.columns.tolist()
        manifest["feature_shape"] = list(np.stack(features).shape)
        # Labels are read only after every LP solve/feature is complete, solely
        # to diagnose the preselected cases. They never enter the optimizer.
        columns = [f"result_committed_Haukeli_G1_t{hour}" for hour in range(168)]
        labels = pd.read_csv(ROOT / "sintef_project/data/kernel/Unit_commitment_decisions.csv", usecols=["Run No", *columns]).set_index("Run No")
        for record in manifest["cases"]:
            truth = labels.loc[record["run_no"], columns].to_numpy(dtype=np.uint8)
            record["true_on_hours"] = int(truth.sum())
            record["diagnostic_thresholds"] = {}
            for threshold in [.5, 1.0]:
                stats = {}
                for name, values in [("v1", record["prior_qmin_ratio"]), ("v2", record["qmin_ratio"])]:
                    predicted = np.array(values) >= threshold - 1e-8
                    stats[name] = {"on": int(predicted.sum()), "false_negative": int(((truth == 1) & ~predicted).sum()),
                                   "false_positive": int(((truth == 0) & predicted).sum())}
                record["diagnostic_thresholds"][str(threshold)] = stats
        manifest["case_runtime_median"] = float(np.median([case["runtime_seconds"] for case in manifest["cases"]]))
        manifest["status"] = "complete"
    except BaseException:
        manifest["status"] = "failed"
        manifest["error"] = traceback.format_exc()
        raise
    finally:
        manifest["runtime_seconds"] = time.monotonic() - started
        manifest["artifacts"] = {str(path.relative_to(out)): digest(path) for path in sorted(out.rglob("*")) if path.is_file() and path.name != "manifest.json"}
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print("STATUS", manifest["status"], "SECONDS", manifest["runtime_seconds"], flush=True)


if __name__ == "__main__":
    main()
