"""Isolate fixed-flow shared hydraulic losses on 16 selected and 16 spread cases."""

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
    """Evaluate exactly one extra LP solve per case; never fit a model."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    started = time.monotonic()
    out = ROOT / "artifacts" / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    manifest = {"run_id": args.run_id, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "command": shlex.join([sys.executable, *sys.argv]), "status": "running", "data": {}, "source": {},
                "environment": {"python": sys.version, "platform": platform.platform(),
                                "packages": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scipy", "pyyaml", "scikit-learn"]}},
                "seed": None, "training": "None. No fitted parameters, selection of thresholds, early stopping or ensemble.",
                "method": "For Tokke/Vinje, extra loss=main_coefficient*(plant_total_q^2-own_q^2)+assigned_penstock_coefficient*(penstock_total_q^2-own_q^2). Use frozen v1 hourly flow. Rebuild PQ curve at initial gross head minus extra loss; unit_curve still applies own-flow quadratic losses. Keep initial-head terminal water values and all reservoir heads unchanged. Resolve network once.",
                "limitations": "Frozen additional losses are correct at prior flow only; they omit cross-flow marginal derivatives and simultaneous reoptimization of other units. Fixed-point consistency is measured, not assumed. This is not an exact nonlinear hydraulic model or SHOP reproduction.",
                "cases": [], "kaggle": {"submission_id": None, "score": None, "rank": None, "observation_time": None}}
    sources = [Path(__file__), ROOT / "sintef_project/lp_shared_loss_update.py", ROOT / "sintef_project/lp_head_update.py",
               ROOT / "sintef_project/lp_features.py", ROOT / "sintef_project/features.py", ROOT / "sintef_project/physics.py",
               ROOT / "tests/test_lp_shared_loss_update.py", ROOT / "tests/test_lp_head_update.py"]
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
        for group in ["data", "source"]:
            for path, expected in prior_manifest[group].items():
                if digest(ROOT / path) != expected:
                    raise ValueError(f"Frozen v1 {group} hash mismatch: {path}")
        manifest["data"] = prior_manifest["data"]
        selected_path = ROOT / "artifacts/lp-head-update-001/manifest.json"
        manifest["selection_source_sha256"] = digest(selected_path)
        selected = json.loads(selected_path.read_text())["selected_case_ids"]
        independent = np.linspace(1, 2186, 16, dtype=int).tolist()
        ids = selected + independent
        manifest["groups"] = {"selected_2021": selected, "spread_training_2015_2020": independent}
        manifest["selected_case_ids"] = ids
        manifest["selection"] = "Same16 selected2021 cases as rejected head-update experiment, plus16 evenly spaced IDs1..2186 independent of any labels/head/error values. Groups evaluated separately; neither is a complete holdout."
        data = HydroCaseData(ROOT)
        lookup = dict(zip(data.cases["Run No"], range(len(data.cases))))
        manifest["selected_case_dates"] = [str(data.cases.iloc[lookup[case_id]].starttime) for case_id in ids]
        solver = HourlyHeadLP(data.network)
        baseline, updated = [], []
        for case_id in ids:
            tick = time.monotonic()
            index = lookup[case_id]
            case = data.case(index)
            path = prior_dir / f"case_{case_id:04d}.npz"
            if digest(path) != prior_manifest["artifacts"][path.name]:
                raise ValueError(f"Prior archive mismatch for {case_id}")
            previous = np.load(path, allow_pickle=False)
            changed, details = update_case_shared_losses(data, index, case, previous)
            result = solver.solve(changed)
            physical_residual = precise_conservation(case, result, data.network)
            if abs(physical_residual) > 1e-8 or (result["power"] > changed["pmax"] + 1e-6).any():
                raise ValueError("Water conservation or hourly power cap violation")
            ratios = [previous["flow"] / case["qmin"][:, None], result["flow"] / changed["qmin"]]
            baseline.append((ratios[0] / (1 + ratios[0])).T)
            updated.append((ratios[1] / (1 + ratios[1])).T)
            after_losses = {**shared_loss_correction(data, result, "Tokke"), **shared_loss_correction(data, result, "Vinje")}
            consistency = {unit: float(np.max(np.abs(after_losses[unit] - detail["additional_loss"]))) for unit, detail in details.items()}
            record = {"run_no": case_id, "runtime_seconds": time.monotonic() - tick,
                      "prior_sha256": digest(path), "precise_conservation_mm3": physical_residual,
                      "shared_loss_consistency_max_error_m": consistency,
                      "unique_curves": {unit: detail["unique_curves"] for unit, detail in details.items()},
                      "maximum_extra_loss_m": {unit: float(detail["additional_loss"].max()) for unit, detail in details.items()},
                      **{key: result[key] for key in ["objective", "balance_residual", "total_balance_error", "iterations", "negative_price_hours"]},
                      "max_slack_mm3": float(max(result[key].max() for key in ["minimum_slack", "maximum_slack", "bypass_slack", "vest_slack"]))}
            manifest["cases"].append(record)
            np.savez_compressed(out / f"updated_case_{case_id:04d}.npz", **result)
            np.savez_compressed(out / f"hourly_curve_{case_id:04d}.npz", energy=changed["energy"], widths=changed["widths"], qmin=changed["qmin"], pmax=changed["pmax"])
            print(f"CASE {case_id} {record['runtime_seconds']:.3f}s shared-loss-consistency={max(consistency.values()):.3f}m conservation={physical_residual:.2e}", flush=True)
            (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        # Only after input-only optimization, load known training labels for the
        # fixed monotone diagnostic. No labels enter the solver or curve builder.
        labels = pd.read_csv(ROOT / "sintef_project/data/kernel/Unit_commitment_decisions.csv").set_index("Run No")
        truth = np.stack([labels.loc[ids, [f"result_committed_{unit}_t{h}" for h in range(168)]].to_numpy(dtype=np.uint8) for unit in UNIT_ORDER], axis=2)
        baseline, updated = np.stack(baseline), np.stack(updated)
        manifest["diagnostics"] = {}
        for group, positions in [("selected_2021", slice(0, 16)), ("spread_training_2015_2020", slice(16, 32))]:
            group_result = {}
            for name, values in [("baseline", baseline), ("updated", updated)]:
                group_truth, group_prediction = truth[positions], values[positions]
                group_result[name] = {"micro_auc": float(roc_auc_score(group_truth.ravel(), group_prediction.ravel())),
                    "unit_auc": {unit: float(roc_auc_score(group_truth[:, :, j].ravel(), group_prediction[:, :, j].ravel()))
                                 if np.unique(group_truth[:, :, j]).size == 2 else None for j, unit in enumerate(UNIT_ORDER)}}
            manifest["diagnostics"][group] = group_result
        np.savez_compressed(out / "predictions.npz", baseline=baseline, updated=updated, labels=truth, case_ids=np.array(ids), units=np.array(UNIT_ORDER))
        manifest["status"] = "complete"
        print(json.dumps(manifest["diagnostics"]), flush=True)
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
