"""Fixed32-case joint convex-mixture benchmark; no fitted model or grid tuning."""

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
from sintef_project.lp_joint_plant import COMMON_LOAD_GRID, JointPlantLP, build_joint_curve


def digest(path):
    """Return SHA256."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    """Solve the same selected2021 and independent2015–2020 groups once each."""
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
                "seed": None, "training": "None. Fixed128 common-load fractions, no grid/threshold/iteration selection, no early stopping or ensemble.",
                "common_load_grid": COMMON_LOAD_GRID.tolist(),
                "method": "Enumerate15 Tokke and7 Vinje nonempty active subsets. Discharge_j=common_fraction*max_turbine_curve_q_j for active units. Evaluate simultaneous main/assigned-penstock losses, turbine and generator efficiency; filter head/flow/Pmin/Pmax violations. Retain upper+lower convex-hull vertices, preserving unit allocations. Nonnegative lambda pervertex/hour, sum(lambda)<=1, unused mass is off. Disable old joint-unit variables; route total plant flow once. Leave other7plants and initial-head terminal conversion unchanged.",
                "limitations": "Finite common-load grid restricts feasible points; convex mixtures relax integer/same-hour operation by time sharing. Efficiency interpolation and fixed initial heads remain approximate. Vertex unit allocations can be degenerate; no exact MIP/SHOP claim. Lower hull supports negative-price cases without free water disposal.",
                "cases": [], "kaggle": {"submission_id": None, "score": None, "rank": None, "observation_time": None}}
    sources = [Path(__file__), ROOT / "sintef_project/lp_joint_plant.py", ROOT / "sintef_project/lp_shared_loss_update.py",
               ROOT / "sintef_project/lp_head_update.py", ROOT / "sintef_project/lp_features.py",
               ROOT / "sintef_project/features.py", ROOT / "sintef_project/physics.py", ROOT / "tests/test_lp_joint_plant.py"]
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
        selection_path = ROOT / "artifacts/lp-shared-loss-001/manifest.json"
        selection = json.loads(selection_path.read_text())
        manifest["selection_manifest_sha256"] = digest(selection_path)
        ids = selection["selected_case_ids"]
        manifest["groups"] = selection["groups"]
        manifest["selected_case_ids"] = ids
        manifest["selected_case_dates"] = selection["selected_case_dates"]
        data = HydroCaseData(ROOT)
        lookup = dict(zip(data.cases["Run No"], range(len(data.cases))))
        solver = JointPlantLP(data.network)
        baseline, updated, hybrid, baseline_counts, updated_counts = [], [], [], [], []
        plant_units = {name: [i for i, unit in enumerate(UNIT_ORDER) if unit.rsplit("_G", 1)[0] == name] for name in ["Tokke", "Vinje"]}
        for case_id in ids:
            tick = time.monotonic()
            index = lookup[case_id]
            case = data.case(index)
            path = prior_dir / f"case_{case_id:04d}.npz"
            if digest(path) != prior_manifest["artifacts"][path.name]:
                raise ValueError("Prior case hash mismatch")
            previous = np.load(path, allow_pickle=False)
            curves = [build_joint_curve(data, index, plant) for plant in ["Tokke", "Vinje"]]
            curve_seconds = time.monotonic() - tick
            result = solver.solve(case, curves)
            power_caps = case["pmax"].copy()
            for j in result["joint_unit_indices"]:
                power_caps[j] = data.model["generator"][UNIT_ORDER[j]]["p_max"]
            if (result["power"] > power_caps[:, None] + 1e-6).any():
                raise ValueError("Generator output exceeds allowed maximum")
            before, after = previous["flow"] / case["qmin"][:, None], result["flow"] / case["qmin"][:, None]
            baseline.append((before / (1 + before)).T)
            updated.append((after / (1 + after)).T)
            mixture_score = (after / (1 + after)).T
            mixture_score[:, result["joint_unit_indices"]] = result["joint_on_fraction"][result["joint_unit_indices"]].T
            hybrid.append(mixture_score)
            baseline_counts.append(np.stack([np.clip(before[indices], 0, 1).sum(axis=0) for indices in plant_units.values()], axis=1))
            updated_counts.append(np.stack([result["joint_on_fraction"][indices].sum(axis=0) for indices in plant_units.values()], axis=1))
            record = {"run_no": case_id, "runtime_seconds": time.monotonic() - tick,
                      "curve_seconds": curve_seconds, "prior_sha256": digest(path),
                      "curves": {curve.plant: curve.specification for curve in curves},
                      **{key: result[key] for key in ["objective", "balance_residual", "total_balance_error", "iterations", "negative_price_hours", "n_variables", "maximum_convex_mass", "old_joint_q_max"]},
                      "max_slack_mm3": float(max(result[key].max() for key in ["minimum_slack", "maximum_slack", "bypass_slack", "vest_slack"]))}
            manifest["cases"].append(record)
            np.savez_compressed(out / f"case_{case_id:04d}.npz", **result)
            for curve in curves:
                np.savez_compressed(out / f"curve_{case_id:04d}_{curve.plant}.npz", total_flow=curve.total_flow, total_power=curve.total_power,
                                    unit_flow=curve.unit_flow, unit_power=curve.unit_power, unit_on=curve.unit_on,
                                    upper_vertex=curve.upper_vertex, unit_indices=curve.unit_indices)
            print(f"CASE {case_id} {record['runtime_seconds']:.3f}s vertices=" + "/".join(str(len(curve.total_flow)) for curve in curves) + f" balance={result['total_balance_error']:.2e}", flush=True)
            (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        # Known labels are read only after all optimizer inputs/results exist.
        labels = pd.read_csv(ROOT / "sintef_project/data/kernel/Unit_commitment_decisions.csv").set_index("Run No")
        truth = np.stack([labels.loc[ids, [f"result_committed_{unit}_t{h}" for h in range(168)]].to_numpy(dtype=np.uint8) for unit in UNIT_ORDER], axis=2)
        baseline, updated, hybrid = np.stack(baseline), np.stack(updated), np.stack(hybrid)
        baseline_counts, updated_counts = np.stack(baseline_counts), np.stack(updated_counts)
        manifest["diagnostics"] = {}
        for group, positions in [("selected_2021", slice(0, 16)), ("spread_training_2015_2020", slice(16, 32))]:
            group_result = {}
            for name, values in [("baseline_q_proxy", baseline), ("joint_q_proxy", updated), ("joint_on_fraction_hybrid", hybrid)]:
                y, p = truth[positions], values[positions]
                group_result[name] = {"micro_auc": float(roc_auc_score(y.ravel(), p.ravel())),
                    "unit_auc": {unit: float(roc_auc_score(y[:, :, j].ravel(), p[:, :, j].ravel()))
                                 if np.unique(y[:, :, j]).size == 2 else None for j, unit in enumerate(UNIT_ORDER)}}
            group_result["count_mae"] = {}
            for j, (plant, indices) in enumerate(plant_units.items()):
                actual = truth[positions][:, :, indices].sum(axis=2)
                group_result["count_mae"][plant] = {"baseline_clipped_q_ratio": float(np.mean(np.abs(actual - baseline_counts[positions, :, j]))),
                                                   "joint_mixture_on_fraction": float(np.mean(np.abs(actual - updated_counts[positions, :, j])))}
            manifest["diagnostics"][group] = group_result
        manifest["diagnostic_interpretation"] = "Both discharge proxies use the same unchanged baseline initial-head Qmin denominator and fixed r/(1+r) mapping. The separately reported hybrid replaces only joint units with convex-mixture on fractions; its pooled score changes score scaling and is not calibrated or selected. Count MAE compares sum(clipped baseline q/Qmin) versus sum(joint on fractions), with no fitted threshold."
        np.savez_compressed(out / "predictions.npz", baseline=baseline, joint_q_proxy=updated, joint_on_hybrid=hybrid, labels=truth,
                            baseline_counts=baseline_counts, joint_counts=updated_counts, case_ids=np.array(ids), units=np.array(UNIT_ORDER))
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
