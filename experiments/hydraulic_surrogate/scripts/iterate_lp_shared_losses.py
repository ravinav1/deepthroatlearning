"""Four fixed damped shared-loss updates; consistency study, not iteration tuning."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
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
from sklearn.metrics import roc_auc_score

from sintef_project.lp_features import HydroCaseData, UNIT_ORDER
from sintef_project.lp_head_update import HourlyHeadLP, shared_loss_correction
from sintef_project.lp_shared_loss_update import update_case_shared_losses, precise_conservation


def digest(path):
    """Return SHA256."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    """Run exactly four updates with damping0.5 and immutable initial economics."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    started = time.monotonic()
    out = ROOT / "artifacts" / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    manifest = {"run_id": args.run_id, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "command": shlex.join([sys.executable, *sys.argv]), "status": "running", "source": {}, "data": {},
                "environment": {"python": sys.version, "platform": platform.platform(),
                                "packages": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scipy", "pyyaml", "scikit-learn"]}},
                "iterations": 4, "damping": .5, "seed": None,
                "method": "Reference flow starts at frozen v1 dispatch. In each iteration rebuild Tokke/Vinje curves at unchanged initial gross head minus shared additional losses evaluated at reference flow, solve once, then reference_flow=0.5*reference_flow+0.5*new_solution_flow. Keep initial-head terminal values fixed. No parameter fit or iteration selection.",
                "cases": [], "kaggle": {"submission_id": None, "score": None, "rank": None, "observation_time": None}}
    sources = [Path(__file__), ROOT / "sintef_project/lp_shared_loss_update.py", ROOT / "sintef_project/lp_head_update.py",
               ROOT / "sintef_project/lp_features.py", ROOT / "sintef_project/features.py", ROOT / "sintef_project/physics.py"]
    for source in sources:
        relative = source.relative_to(ROOT)
        manifest["source"][str(relative)] = digest(source)
        target = out / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    try:
        prior_dir = ROOT / "artifacts/lp-full-001"
        prior_manifest = json.loads((prior_dir / "manifest.json").read_text())
        manifest["prior_manifest_sha256"] = digest(prior_dir / "manifest.json")
        for category in ["data", "source"]:
            for relative, expected in prior_manifest[category].items():
                if digest(ROOT / relative) != expected:
                    raise ValueError(f"Frozen v1 {category} changed: {relative}")
        manifest["data"] = prior_manifest["data"]
        first_dir = ROOT / "artifacts/lp-shared-loss-001"
        first_manifest = json.loads((first_dir / "manifest.json").read_text())
        manifest["selection_manifest_sha256"] = digest(first_dir / "manifest.json")
        manifest["groups"] = first_manifest["groups"]
        ids = first_manifest["selected_case_ids"]
        manifest["selected_case_ids"] = ids
        manifest["selected_case_dates"] = first_manifest["selected_case_dates"]
        data = HydroCaseData(ROOT)
        lookup = dict(zip(data.cases["Run No"], range(len(data.cases))))
        solver = HourlyHeadLP(data.network)
        predictions = [[] for _ in range(5)]
        consistency_values = [[] for _ in range(4)]
        for case_id in ids:
            case = data.case(lookup[case_id])
            path = prior_dir / f"case_{case_id:04d}.npz"
            if digest(path) != prior_manifest["artifacts"][path.name]:
                raise ValueError("Prior case hash mismatch")
            previous = np.load(path, allow_pickle=False)
            reference = previous["flow"].copy()
            last_flow = reference.copy()
            last_on = reference >= case["qmin"][:, None] - 1e-10
            initial_ratio = reference / case["qmin"][:, None]
            predictions[0].append((initial_ratio / (1 + initial_ratio)).T)
            case_record = {"run_no": case_id, "prior_sha256": digest(path), "iterations": []}
            for iteration in range(4):
                tick = time.monotonic()
                changed, details = update_case_shared_losses(data, lookup[case_id], case, {"flow": reference})
                result = solver.solve(changed)
                balance = precise_conservation(case, result, data.network)
                if abs(balance) > 1e-8 or (result["power"] > changed["pmax"] + 1e-6).any():
                    raise ValueError("Conservation/power-cap failure")
                loss_after = {**shared_loss_correction(data, result, "Tokke"), **shared_loss_correction(data, result, "Vinje")}
                inconsistency = np.stack([np.abs(loss_after[unit] - detail["additional_loss"]) for unit, detail in details.items()])
                consistency_values[iteration].append(inconsistency)
                on = result["flow"] >= changed["qmin"] - 1e-10
                ratio = result["flow"] / changed["qmin"]
                predictions[iteration + 1].append((ratio / (1 + ratio)).T)
                record = {"iteration": iteration + 1, "runtime_seconds": time.monotonic() - tick,
                          "shared_loss_consistency_max_m": float(inconsistency.max()),
                          "shared_loss_consistency_mean_m": float(inconsistency.mean()),
                          "flow_change_max_m3s": float(np.max(np.abs(result["flow"] - last_flow)) / .0036),
                          "flow_change_mean_m3s": float(np.mean(np.abs(result["flow"] - last_flow)) / .0036),
                          "on_state_change_fraction": float((on != last_on).mean()),
                          "precise_conservation_mm3": balance, "balance_residual": result["balance_residual"],
                          "max_slack_mm3": float(max(result[key].max() for key in ["minimum_slack", "maximum_slack", "bypass_slack", "vest_slack"]))}
                if iteration == 0:
                    original = np.load(first_dir / f"updated_case_{case_id:04d}.npz", allow_pickle=False)
                    record["matches_previous_first_iteration_exactly"] = bool(np.array_equal(original["flow"], result["flow"]))
                    if not record["matches_previous_first_iteration_exactly"]:
                        raise ValueError("First iteration failed exact reproduction")
                case_record["iterations"].append(record)
                np.savez_compressed(out / f"case_{case_id:04d}_iteration_{iteration + 1}.npz", **result)
                reference = .5 * reference + .5 * result["flow"]
                last_flow, last_on = result["flow"], on
            manifest["cases"].append(case_record)
            print(f"CASE {case_id} max-loss-inconsistency " + " -> ".join(f"{r['shared_loss_consistency_max_m']:.3f}" for r in case_record["iterations"]), flush=True)
            (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        labels = pd.read_csv(ROOT / "sintef_project/data/kernel/Unit_commitment_decisions.csv").set_index("Run No")
        truth = np.stack([labels.loc[ids, [f"result_committed_{unit}_t{h}" for h in range(168)]].to_numpy(dtype=np.uint8) for unit in UNIT_ORDER], axis=2)
        predictions = np.stack([np.stack(values) for values in predictions])
        manifest["diagnostics"] = {}
        for group, positions in [("selected_2021", slice(0, 16)), ("spread_training_2015_2020", slice(16, 32))]:
            values = []
            for iteration in range(5):
                y, p = truth[positions], predictions[iteration, positions]
                values.append({"iteration": iteration, "micro_auc": float(roc_auc_score(y.ravel(), p.ravel())),
                               "unit_auc": {unit: float(roc_auc_score(y[:, :, j].ravel(), p[:, :, j].ravel()))
                                            if np.unique(y[:, :, j]).size == 2 else None for j, unit in enumerate(UNIT_ORDER)}})
            manifest["diagnostics"][group] = values
        manifest["consistency_summary"] = []
        for iteration, values in enumerate(consistency_values):
            array = np.stack(values)
            records = [case["iterations"][iteration] for case in manifest["cases"]]
            manifest["consistency_summary"].append({"iteration": iteration + 1, "max_m": float(array.max()),
                "mean_m": float(array.mean()), "q95_m": float(np.quantile(array, .95)),
                "mean_on_state_change_fraction": float(np.mean([row["on_state_change_fraction"] for row in records])),
                "mean_flow_change_m3s": float(np.mean([row["flow_change_mean_m3s"] for row in records]))})
        np.savez_compressed(out / "predictions.npz", predictions=predictions, labels=truth, case_ids=np.array(ids), units=np.array(UNIT_ORDER))
        manifest["status"] = "complete"
        print(json.dumps({"consistency": manifest["consistency_summary"], "micro_auc_trends": {group: [row["micro_auc"] for row in rows] for group, rows in manifest["diagnostics"].items()}}), flush=True)
    except BaseException:
        manifest["status"] = "failed"
        manifest["error"] = traceback.format_exc()
        raise
    finally:
        manifest["runtime_seconds"] = time.monotonic() - started
        manifest["artifacts"] = {str(path.relative_to(out)): digest(path) for path in out.rglob("*") if path.is_file() and path.name != "manifest.json"}
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
