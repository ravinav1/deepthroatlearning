"""Measure 2021 head/shared-loss mechanisms and benchmark one Hogga LP update."""

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
from sintef_project.lp_head_update import (
    HourlyHeadLP, shared_loss_correction, single_reservoir_heads, update_case_head,
)


def digest(path):
    """Return SHA256."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def distribution(values):
    """Return finite descriptive quantiles without fitting any parameter."""
    values = np.asarray(values, dtype=float).ravel()
    return {"mean": float(values.mean()), "min": float(values.min()),
            "q50": float(np.quantile(values, .5)), "q90": float(np.quantile(values, .9)),
            "q95": float(np.quantile(values, .95)), "max": float(values.max())}


def main():
    """Use legal baseline-LP states, then inspect known labels diagnostically."""
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
                "seed": None, "training": "None. Existing 2021 model errors only select a diagnostic subset; no model fit or threshold selection.",
                "method": "Use baseline LP's single Bandak/Vinjevatn storage at each hour end plus legal initial volume. Average interpolated beginning/end levels. Rebuild only Hogga PQ curves and resolve full network once; unchanged initial-head terminal values and all other production curves. No pooled reservoir allocation, head iteration to convergence, startup/MIP or shared-loss correction.",
                "selection": "Eight spread 2021 cases, four greatest Hogga model Brier-error cases and four greatest Hogga max relative head-change cases, each added error/head case at least seven case starts from all selected IDs. Selected diagnostics are not representative validation.",
                "cases": [], "kaggle": {"submission_id": None, "score": None, "rank": None, "observation_time": None}}
    sources = [Path(__file__), ROOT / "sintef_project/lp_head_update.py", ROOT / "sintef_project/lp_features.py",
               ROOT / "sintef_project/features.py", ROOT / "sintef_project/physics.py", ROOT / "tests/test_lp_head_update.py"]
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
        model_dir = ROOT / "artifacts/lgb-normalized-lp-001"
        model_cache_path = model_dir / "validation_2021.npz"
        manifest["model_manifest_sha256"] = digest(model_dir / "manifest.json")
        manifest["model_prediction_sha256"] = digest(model_cache_path)
        validation = np.load(model_cache_path, allow_pickle=False)
        ids = validation["case_ids"]
        labels = validation["labels"].reshape(len(ids), 168, 14)
        prediction = validation["predictions"].reshape(len(ids), 168, 14)
        if validation["units"].tolist() != UNIT_ORDER:
            raise ValueError("Model unit order mismatch")
        data = HydroCaseData(ROOT)
        case_lookup = dict(zip(data.cases["Run No"], range(len(data.cases))))
        rows, head_arrays, losses = [], [], {"Tokke": [], "Vinje": []}
        manifest["prior_case_hashes"] = {}
        for position, case_id in enumerate(ids):
            path = prior_dir / f"case_{case_id:04d}.npz"
            if digest(path) != prior_manifest["artifacts"][path.name]:
                raise ValueError(f"Prior LP archive mismatch {case_id}")
            manifest["prior_case_hashes"][path.name] = digest(path)
            prior = np.load(path, allow_pickle=False)
            heads = single_reservoir_heads(data, case_lookup[int(case_id)], prior)
            change = heads["Hogga"] - heads["initial_Hogga"]
            head_arrays.append(np.stack([heads["Hogga"], heads["Tokke"], change]))
            row = {"run_no": int(case_id), "starttime": str(data.cases.iloc[case_lookup[int(case_id)]].starttime),
                   "hogga_initial_head_m": heads["initial_Hogga"],
                   "hogga_head_max_delta_m": float(np.max(np.abs(change))),
                   "hogga_head_max_relative_delta": float(np.max(np.abs(change)) / heads["initial_Hogga"]),
                   "hogga_model_brier": float(np.mean((prediction[position, :, 0] - labels[position, :, 0]) ** 2)),
                   "tokke_head_max_delta_m": float(np.max(np.abs(heads["Tokke"] - heads["initial_Tokke"]))),
                   "vinje_g3_model_brier": float(np.mean((prediction[position, :, 9] - labels[position, :, 9]) ** 2))}
            for plant in ["Tokke", "Vinje"]:
                correction = shared_loss_correction(data, prior, plant)
                active = [values[prior["flow"][UNIT_ORDER.index(unit)] > 1e-8] for unit, values in correction.items()]
                active = np.concatenate(active)
                if len(active):
                    losses[plant].append(active)
                    row[f"{plant.lower()}_active_shared_loss_mean_m"] = float(active.mean())
                    row[f"{plant.lower()}_active_shared_loss_max_m"] = float(active.max())
                else:
                    row[f"{plant.lower()}_active_shared_loss_mean_m"] = 0.
                    row[f"{plant.lower()}_active_shared_loss_max_m"] = 0.
            rows.append(row)
        table = pd.DataFrame(rows)
        table.to_csv(out / "head_case_summary_2021.csv", index=False)
        manifest["all_2021_ids"] = ids.tolist()
        manifest["head_summary"] = {column: distribution(table[column]) for column in ["hogga_initial_head_m", "hogga_head_max_delta_m", "hogga_head_max_relative_delta", "tokke_head_max_delta_m"]}
        manifest["shared_loss_summary_active_unit_hours"] = {plant: distribution(np.concatenate(values)) for plant, values in losses.items()}
        manifest["exploratory_correlations"] = {
            "hogga_head_change_vs_model_case_brier": float(table.hogga_head_max_relative_delta.corr(table.hogga_model_brier)),
            "vinje_shared_loss_vs_g3_case_brier": float(table.vinje_active_shared_loss_mean_m.corr(table.vinje_g3_model_brier))}
        selected = [int(ids[index]) for index in np.linspace(0, len(ids) - 1, 8, dtype=int)]
        for column in ["hogga_model_brier", "hogga_head_max_relative_delta"]:
            added = 0
            for case_id in table.sort_values(column, ascending=False).run_no:
                if min(abs(int(case_id) - chosen) for chosen in selected) >= 7:
                    selected.append(int(case_id))
                    added += 1
                    if added == 4:
                        break
        if len(selected) != 16:
            raise ValueError("Could not construct predeclared 16-case subset")
        manifest["selected_case_ids"] = selected
        solver = HourlyHeadLP(data.network)
        baseline_predictions, changed_predictions, selected_labels = [], [], []
        for case_id in selected:
            tick = time.monotonic()
            index = case_lookup[case_id]
            case = data.case(index)
            prior = np.load(prior_dir / f"case_{case_id:04d}.npz", allow_pickle=False)
            changed_case, heads = update_case_head(data, index, case, prior, plants=("Hogga",))
            result = solver.solve(changed_case)
            after_heads = single_reservoir_heads(data, index, result)
            before = prior["flow"] / case["qmin"][:, None]
            after = result["flow"] / changed_case["qmin"]
            baseline_predictions.append((before / (1 + before)).T)
            changed_predictions.append((after / (1 + after)).T)
            selected_labels.append(labels[np.flatnonzero(ids == case_id)[0]])
            if (result["power"] > changed_case["pmax"] + 1e-6).any():
                raise ValueError("Updated generation exceeded hourly feasible Pmax")
            record = {"run_no": case_id, "runtime_seconds": time.monotonic() - tick,
                      "head_used_min_m": float(heads["Hogga"].min()), "head_used_max_m": float(heads["Hogga"].max()),
                      "head_consistency_max_error_m": float(np.max(np.abs(after_heads["Hogga"] - heads["Hogga"]))),
                      "hogga_before_on_hours": int((before[0] >= 1 - 1e-8).sum()), "hogga_after_on_hours": int((after[0] >= 1 - 1e-8).sum()),
                      **{key: result[key] for key in ["objective", "balance_residual", "total_balance_error", "iterations", "negative_price_hours"]},
                      "max_slack_mm3": float(max(result[key].max() for key in ["minimum_slack", "maximum_slack", "bypass_slack", "vest_slack"]))}
            manifest["cases"].append(record)
            np.savez_compressed(out / f"updated_case_{case_id:04d}.npz", **result)
            np.savez_compressed(out / f"hourly_curve_{case_id:04d}.npz", energy=changed_case["energy"], widths=changed_case["widths"], qmin=changed_case["qmin"], pmax=changed_case["pmax"], head=heads["Hogga"])
            print(f"CASE {case_id} {record['runtime_seconds']:.3f}s Hogga {record['hogga_before_on_hours']}->{record['hogga_after_on_hours']} head-consistency={record['head_consistency_max_error_m']:.3f}m", flush=True)
        truth = np.stack(selected_labels)
        for name, values in [("baseline", baseline_predictions), ("updated", changed_predictions)]:
            values = np.stack(values)
            manifest[f"selected_{name}_diagnostic"] = {"micro_auc": float(roc_auc_score(truth.ravel(), values.ravel())),
                "unit_auc": {unit: float(roc_auc_score(truth[:, :, j].ravel(), values[:, :, j].ravel())) for j, unit in enumerate(UNIT_ORDER)}}
        np.savez_compressed(out / "selected_predictions.npz", baseline=np.stack(baseline_predictions), updated=np.stack(changed_predictions), labels=truth, case_ids=np.array(selected), units=np.array(UNIT_ORDER))
        manifest["status"] = "complete"
        print(json.dumps({key: manifest[key] for key in ["head_summary", "shared_loss_summary_active_unit_hours", "exploratory_correlations", "selected_baseline_diagnostic", "selected_updated_diagnostic"]}), flush=True)
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
