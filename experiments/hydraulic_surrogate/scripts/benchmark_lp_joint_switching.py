"""Fixed32-case per-subset-hull/start-stop-cost ablation, with all arms reported."""

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
from sintef_project.lp_joint_plant import JointPlantLP
from sintef_project.lp_joint_switching import SwitchingJointLP, build_subset_joint_curve, switching_specification


def digest(path: Path) -> str:
    """Return SHA256 for provenance."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Run the predeclared ablation without model training or score selection."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    out = ROOT / "artifacts" / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    prior_dir = ROOT / "artifacts/lp-joint-full-001"
    prior = json.loads((prior_dir / "manifest.json").read_text())
    selection_path = ROOT / "artifacts/lp-shared-loss-001/manifest.json"
    selection = json.loads(selection_path.read_text())
    ids = selection["selected_case_ids"]
    arms = ["frozen_global_hull", "subset_no_cost", "subset_switching"]
    manifest = {"run_id": args.run_id, "timestamp_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
                "command": shlex.join([sys.executable, *sys.argv]), "mathematics": switching_specification(),
                "selected_case_ids": ids, "selected_case_dates": selection["selected_case_dates"], "groups": selection["groups"],
                "selection_manifest_sha256": digest(selection_path), "prior_manifest_sha256": digest(prior_dir / "manifest.json"),
                "data": prior["data"], "source": {}, "arms": arms, "cases": [],
                "training": "None; no fitted thresholds, seeds, grid selection, early stopping or ensemble; labels read after all solves.",
                "environment": {"python": sys.version, "platform": platform.platform(),
                                "packages": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scipy", "PyYAML", "scikit-learn"]}},
                "kaggle": {"submission_id": None, "score": None, "rank": None, "observation_time": None}}
    sources = list(prior["source"]) + [str(Path(__file__).relative_to(ROOT)), "sintef_project/lp_joint_switching.py", "tests/test_lp_joint_switching.py"]
    for relative in sources:
        source = ROOT / relative
        manifest["source"][relative] = digest(source)
        target = out / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    try:
        for category in ["data", "source"]:
            for relative, expected in prior[category].items():
                assert digest(ROOT / relative) == expected, relative
        data = HydroCaseData(ROOT)
        lookup = dict(zip(data.cases["Run No"], range(len(data.cases))))
        free_solver, cost_solver = JointPlantLP(data.network), SwitchingJointLP(data.network)
        startcost = np.array([data.model["generator"][unit]["startcost_const"] for unit in UNIT_ORDER], float)
        stopcost = np.array([data.model["generator"][unit]["stopcost_const"] for unit in UNIT_ORDER], float)
        manifest["startcost"] = startcost.tolist()
        manifest["stopcost"] = stopcost.tolist()
        predictions = {arm: [] for arm in arms}
        hybrid = {arm: [] for arm in arms}
        counts = {arm: [] for arm in arms}
        plants = {plant: [j for j, unit in enumerate(UNIT_ORDER) if unit.rsplit("_G", 1)[0] == plant] for plant in ["Tokke", "Vinje"]}
        for case_id in ids:
            tick = time.monotonic()
            index = lookup[case_id]
            case = data.case(index)
            path = prior_dir / f"case_{case_id:04d}.npz"
            assert digest(path) == prior["artifacts"][path.name]
            with np.load(path, allow_pickle=False) as saved:
                results = {"frozen_global_hull": {k: saved[k] for k in saved.files}}
            curves = [build_subset_joint_curve(data, index, plant) for plant in plants]
            curve_seconds = time.monotonic() - tick
            results["subset_no_cost"] = free_solver.solve(case, curves)
            control_seconds = time.monotonic() - tick - curve_seconds
            results["subset_switching"] = cost_solver.solve(case, curves, startcost, stopcost)
            switching_seconds = time.monotonic() - tick - curve_seconds - control_seconds
            base_objective = float(results["frozen_global_hull"]["objective"])
            no_cost_objective = float(results["subset_no_cost"]["objective"])
            assert abs(no_cost_objective - base_objective) < max(.01, abs(base_objective) * 1e-8)
            assert float(results["subset_switching"]["objective"]) <= no_cost_objective + .01
            record = {"case_id": case_id, "runtime_seconds": time.monotonic() - tick, "curve_seconds": curve_seconds,
                      "control_seconds": control_seconds, "switching_seconds": switching_seconds,
                      "prior_archive_sha256": digest(path), "curves": {c.plant: c.specification for c in curves}, "arms": {}}
            for arm, result in results.items():
                ratio = result["flow"] / case["qmin"][:, None]
                proxy = (ratio / (1 + ratio)).T
                mixture = proxy.copy()
                mixture[:, result["joint_unit_indices"]] = result["joint_on_fraction"][result["joint_unit_indices"]].T
                predictions[arm].append(proxy)
                hybrid[arm].append(mixture)
                counts[arm].append(np.stack([result["joint_on_fraction"][indices].sum(axis=0) for indices in plants.values()], axis=1))
                for j in result["joint_unit_indices"]:
                    on = result["joint_on_fraction"][j]
                    params = data.model["generator"][UNIT_ORDER[j]]
                    assert (result["power"][j] >= params["p_min"] * on - 1e-6).all()
                    assert (result["power"][j] <= params["p_max"] * on + 1e-6).all()
                record["arms"][arm] = {key: float(result[key]) for key in ["objective", "balance_residual", "total_balance_error", "n_variables", "maximum_convex_mass", "old_joint_q_max"]}
                record["arms"][arm]["fractional_transition_total"] = float(np.abs(np.diff(result["joint_on_fraction"][result["joint_unit_indices"]], axis=1)).sum())
                record["arms"][arm]["max_slack_mm3"] = float(max(result[key].max() for key in ["minimum_slack", "maximum_slack", "bypass_slack", "vest_slack"]))
                record["arms"][arm]["switching_cost"] = float(result.get("switching_cost", 0.))
                if arm != "frozen_global_hull":
                    np.savez_compressed(out / f"case_{case_id:04d}_{arm}.npz", **result)
            for curve in curves:
                np.savez_compressed(out / f"curve_{case_id:04d}_{curve.plant}.npz", **{name: getattr(curve, name) for name in
                                    ["total_flow", "total_power", "unit_flow", "unit_power", "unit_on", "unit_indices", "upper_vertex"]})
            manifest["cases"].append(record)
            (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
            print(f"CASE {case_id} {record['runtime_seconds']:.3f}s vertices=" + "/".join(str(len(c.total_flow)) for c in curves), flush=True)
        labels = pd.read_csv(ROOT / "sintef_project/data/kernel/Unit_commitment_decisions.csv").set_index("Run No")
        truth = np.stack([labels.loc[ids, [f"result_committed_{unit}_t{hour}" for hour in range(168)]].to_numpy(dtype=np.uint8) for unit in UNIT_ORDER], axis=2)
        predictions = {arm: np.stack(values) for arm, values in predictions.items()}
        hybrid = {arm: np.stack(values) for arm, values in hybrid.items()}
        counts = {arm: np.stack(values) for arm, values in counts.items()}
        manifest["diagnostics"] = {}
        for name, positions in [("selected_2021", slice(0, 16)), ("spread_training_2015_2020", slice(16, 32))]:
            group = {}
            for arm in arms:
                y, p = truth[positions], predictions[arm][positions]
                group[arm] = {"q_proxy_micro_auc": float(roc_auc_score(y.ravel(), p.ravel())),
                              "on_hybrid_micro_auc": float(roc_auc_score(y.ravel(), hybrid[arm][positions].ravel())),
                              "unit_auc": {unit: float(roc_auc_score(y[:, :, j].ravel(), p[:, :, j].ravel()))
                                           if np.unique(y[:, :, j]).size == 2 else None for j, unit in enumerate(UNIT_ORDER)},
                              "count_mae": {plant: float(np.mean(np.abs(truth[positions][:, :, indices].sum(axis=2) - counts[arm][positions, :, k])))
                                            for k, (plant, indices) in enumerate(plants.items())}}
            manifest["diagnostics"][name] = group
        np.savez_compressed(out / "predictions.npz", **predictions, **{arm + "_hybrid": p for arm, p in hybrid.items()},
                            **{arm + "_counts": p for arm, p in counts.items()}, labels=truth, case_ids=np.asarray(ids), units=np.asarray(UNIT_ORDER))
        manifest["diagnostic_interpretation"] = "Fixed r/(1+r), r=q/baseline Qmin. Secondary hybrid replaces only joint-unit scores with physical on-fractions. All arms and weaker results retained; not calibrated probabilities or supervised/leaderboard performance."
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
