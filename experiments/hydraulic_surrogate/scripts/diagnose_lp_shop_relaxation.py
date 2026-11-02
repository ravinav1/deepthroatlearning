"""Reviewed, fixed pre-2021 relaxed-state LP diagnostic; never a full cache."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import shlex
import shutil
import sys
import time
import traceback

for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ[variable] = "1"

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SOURCE_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("TDT4173_WORKSPACE_ROOT", SOURCE_ROOT)).resolve()
sys.path.insert(0, str(SOURCE_ROOT))

from sintef_project.lp_features import HydroCaseData, HydraulicLP, UNIT_ORDER
from sintef_project.lp_shop_relaxation import ShopRelaxationLP, specification, state_interval

ARMS = ["cached_base", "base_float64_rhs", "linked_zero_cost", "linked_supplied_cost"]
WEAK_UNITS = ["Hogga_G1", "Lio_G1", "Haukeli_G1"]


def digest(path: Path) -> str:
    """Hash a file without decoding potentially unselected labels."""
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def select_cases(cases: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Choose 36 starts solely by dates; all windows end before 2021."""
    dates = pd.to_datetime(cases["starttime"], utc=True)
    indices, blocks = [], []
    for name, first, last in [("2015_2018", "2015-01-01", "2018-12-31"),
                              ("2019", "2019-01-01", "2019-12-31"),
                              ("2020", "2020-01-01", "2020-12-25")]:
        eligible = np.flatnonzero((dates >= pd.Timestamp(first, tz="UTC"))
                                  & (dates <= pd.Timestamp(last, tz="UTC")))
        chosen = eligible[np.rint(np.linspace(0, len(eligible) - 1, 12)).astype(int)]
        if len(np.unique(chosen)) != 12:
            raise ValueError("Insufficient metadata for the fixed date grid")
        indices.extend(chosen.tolist())
        blocks.extend([name] * 12)
    return np.asarray(indices), blocks


def auc(truth: np.ndarray, values: np.ndarray) -> float | None:
    """Compute pooled ranking AUC only when both target states occur."""
    return float(roc_auc_score(truth.ravel(), values.ravel())) if len(np.unique(truth)) == 2 else None


def metrics(truth: np.ndarray, qratio: np.ndarray, pratio: np.ndarray,
            blocks: list[str]) -> dict:
    """Score unchanged monotone proxies, without fitting a threshold or weight."""
    qproxy = np.maximum(qratio, 0) / (1 + np.maximum(qratio, 0))
    pproxy = np.maximum(pratio, 0) / (1 + np.maximum(pratio, 0))
    hard = pratio >= .5
    weak = [UNIT_ORDER.index(name) for name in WEAK_UNITS]
    result = {"micro_auc_qratio": auc(truth, qproxy), "micro_auc_pratio": auc(truth, pproxy),
              "weak3_micro_auc_qratio": auc(truth[:, weak], qproxy[:, weak]),
              "power_cutoff_hour_errors": int(np.sum(hard != truth)),
              "power_cutoff_error_fraction": float(np.mean(hard != truth)),
              "mean_absolute_case_unit_onfraction_error": float(np.abs(hard.mean(axis=-1) - truth.mean(axis=-1)).mean()),
              "mean_absolute_case_unit_switch_count_error": float(np.abs(
                  np.sum(np.diff(hard.astype(int), axis=-1) != 0, axis=-1)
                  - np.sum(np.diff(truth.astype(int), axis=-1) != 0, axis=-1)).mean()),
              "per_unit": {}, "per_date_block": {}}
    for unit, name in enumerate(UNIT_ORDER):
        result["per_unit"][name] = {"auc_qratio": auc(truth[:, unit], qproxy[:, unit]),
                                     "auc_pratio": auc(truth[:, unit], pproxy[:, unit]),
                                     "hour_errors": int(np.sum(hard[:, unit] != truth[:, unit]))}
    for name in dict.fromkeys(blocks):
        keep = np.asarray(blocks) == name
        result["per_date_block"][name] = {"auc_qratio": auc(truth[keep], qproxy[keep]),
                                          "hour_errors": int(np.sum(hard[keep] != truth[keep]))}
    return result


def main() -> None:
    """Verify the frozen protocol, solve input-only controls, then parse 36 labels."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id or args.run_id in ["", ".", ".."]:
        raise ValueError("Run ID must be a single directory name")
    out = ROOT / "artifacts" / args.run_id
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    report = {"run_id": args.run_id, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "status": "initializing", "command": shlex.join([sys.executable, *sys.argv]),
              "source_root": str(SOURCE_ROOT), "workspace_root": str(ROOT),
              "environment": {"python": sys.version, "platform": platform.platform(),
                              "machine": platform.machine(), "cpu_count": os.cpu_count(),
                              "packages": {name: importlib.metadata.version(name) for name in
                                           ["numpy", "pandas", "scipy", "scikit-learn", "PyYAML"]}},
              "training": False, "kaggle_submission": None, "seed": None,
              "labels_decoded_utc": None, "case_controls": [], "artifacts": {}}
    try:
        protocol_path = args.protocol if args.protocol.is_absolute() else ROOT / args.protocol
        if digest(protocol_path) != args.protocol_sha256:
            raise ValueError("Observed frozen protocol hash differs")
        protocol = json.loads(protocol_path.read_text())
        if protocol["specification"] != specification() or protocol["arms"] != ARMS:
            raise ValueError("Protocol and runtime mechanism differ")
        for name, expected in protocol["source"].items():
            path = SOURCE_ROOT / name
            if digest(path) != expected:
                raise ValueError(f"Reviewed source changed: {name}")
            target = out / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        for name, expected in protocol["data"].items():
            if digest(ROOT / name) != expected:
                raise ValueError(f"Frozen data changed: {name}")
        cache = ROOT / protocol["baseline_cache"]
        if digest(cache / "manifest.json") != protocol["baseline_manifest_sha256"]:
            raise ValueError("Baseline manifest changed")
        cm = json.loads((cache / "manifest.json").read_text())
        if cm["status"] != "complete" or cm["data"] != protocol["data"]:
            raise ValueError("Baseline data closure is incomplete")
        for name in ["sintef_project/lp_features.py", "sintef_project/features.py", "sintef_project/physics.py"]:
            if cm["source"][name] != protocol["source"][name]:
                raise ValueError("Baseline source differs from reviewed input dependency")
        inputs = HydroCaseData(ROOT)
        indices, blocks = select_cases(inputs.cases)
        chosen = inputs.cases.iloc[indices]
        if chosen["Run No"].tolist() != protocol["case_ids"] or blocks != protocol["date_blocks"]:
            raise ValueError("Metadata selection differs from the frozen protocol")
        if chosen.starttime.max() > pd.Timestamp("2020-12-25", tz="UTC"):
            raise ValueError("Diagnostic includes forbidden validation dates")
        report.update(protocol_sha256=args.protocol_sha256, protocol=protocol,
                      case_ids=protocol["case_ids"], case_dates=chosen.starttime.astype(str).tolist(),
                      date_blocks=blocks, status="solving_input_only")
        (out / "protocol_before_labels.json").write_text(json.dumps(report, indent=2) + "\n")
        base_solver = HydraulicLP(inputs.network)
        linked_solver = ShopRelaxationLP(inputs.network)
        qratios = {arm: [] for arm in ARMS}
        pratios = {arm: [] for arm in ARMS}
        states = {arm: [] for arm in ARMS[2:]}
        for index in indices:
            case_started = time.monotonic()
            case_id = int(inputs.cases.iloc[index]["Run No"])
            case = inputs.case(int(index))
            legacy_rhs = case["natural"].copy()
            legacy_rhs[:, 0] += case["initial"]
            case["natural"] = np.asarray(case["natural"], dtype=np.float64)
            reference_rhs = case["natural"].copy()
            reference_rhs[:, 0] += case["initial"]
            for name, field in [("pmin", "p_min"), ("start_cost", "startcost_const"), ("stop_cost", "stopcost_const")]:
                case[name] = np.asarray([inputs.model["generator"][unit][field] for unit in UNIT_ORDER], dtype=np.float64)
            cached_name = f"case_{case_id:04d}.npz"
            if digest(cache / cached_name) != cm["artifacts"][cached_name]:
                raise ValueError(f"Cached baseline case changed: {case_id}")
            with np.load(cache / cached_name, allow_pickle=False) as z:
                saved = {key: z[key].copy() for key in z}
            result = {"cached_base": saved, "base_float64_rhs": base_solver.solve(case),
                      "linked_zero_cost": linked_solver.solve(case, 0.),
                      "linked_supplied_cost": linked_solver.solve(case, 1.)}
            replay = ShopRelaxationLP(inputs.network).solve(case, 1.)
            for key in replay:
                np.testing.assert_array_equal(replay[key], result["linked_supplied_cost"][key], err_msg=f"Replay {case_id}: {key}")
            np.savez_compressed(out / f"case_{case_id:04d}_replay.npz", **replay)
            fresh, zero, cost = [result[arm] for arm in ARMS[1:]]
            lower, upper = state_interval(case, fresh["flow"], fresh["power"])
            lift_error = float(np.max(lower - upper))
            lift_feasible = lift_error <= 1e-8
            tolerance = 1e-5 + 2e-10 * max(1., abs(fresh["objective"]))
            if zero["objective"] > fresh["objective"] + tolerance:
                raise ValueError("Linked zero-cost objective exceeds its feasible-set superset")
            if lift_feasible and abs(zero["objective"] - fresh["objective"]) > tolerance:
                raise ValueError("Zero-cost control failed with a feasible baseline state lift")
            if cost["objective"] > zero["objective"] + tolerance or cost["gross_objective"] > zero["objective"] + tolerance:
                raise ValueError("Transition-cost objective control failed")
            for arm in ARMS:
                np.savez_compressed(out / f"case_{case_id:04d}_{arm}.npz", **result[arm])
                qratios[arm].append(result[arm]["flow"] / case["qmin"][:, None])
                pratios[arm].append(result[arm]["power"] / case["pmin"][:, None])
                if arm in states:
                    states[arm].append(result[arm]["state"])
            np.savez_compressed(out / f"case_{case_id:04d}_inputs.npz", **case)
            control = {"case_id": case_id, "state_lift_feasible": lift_feasible,
                       "state_lift_max_lower_minus_upper": lift_error,
                       "legacy_rhs_max_abs_difference": float(np.max(np.abs(legacy_rhs - reference_rhs))),
                       "cached_vs_fresh_objective_delta": float(fresh["objective"] - saved["objective"]),
                       "cached_vs_fresh_flow_max_abs_delta": float(np.max(np.abs(fresh["flow"] - saved["flow"]))),
                       "zero_vs_fresh_objective_delta": float(zero["objective"] - fresh["objective"]),
                       "cost_vs_zero_objective_delta": float(cost["objective"] - zero["objective"]),
                       "transition_cost": cost["transition_cost"], "fresh_replay_bitwise": True,
                       "max_new_arm_balance_error": max(abs(float(result[arm]["total_balance_error"])) for arm in ARMS[1:]),
                       "runtime_seconds": time.monotonic() - case_started}
            report["case_controls"].append(control)
            (out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(control), flush=True)
        report["input_only_controls_passed"] = True
        report["status"] = "decoding_selected_pre2021_labels"
        report["labels_decoded_utc"] = datetime.now(timezone.utc).isoformat()
        label_path = ROOT / "sintef_project/data/kernel/Unit_commitment_decisions.csv"
        metadata = pd.read_csv(label_path, usecols=["Run No", "starttime"])
        keep = metadata["Run No"].isin(protocol["case_ids"]).to_numpy()
        if len(metadata.loc[keep]) != 36 or (pd.to_datetime(metadata.loc[keep, "starttime"], utc=True).dt.year > 2020).any():
            raise ValueError("Selected label-row scope failed")
        columns = [f"result_committed_{unit}_t{hour}" for unit in UNIT_ORDER for hour in range(168)]
        labels = pd.read_csv(label_path, skiprows=(np.flatnonzero(~keep) + 1).tolist(),
                             usecols=["Run No", *columns], dtype={name: "uint8" for name in columns})
        if labels["Run No"].tolist() != protocol["case_ids"]:
            raise ValueError("Selected label order mismatch")
        truth = labels[columns].to_numpy().reshape(36, 14, 168)
        if not np.isin(truth, [0, 1]).all():
            raise ValueError("Nonbinary diagnostic target")
        results = {arm: metrics(truth, np.asarray(qratios[arm]), np.asarray(pratios[arm]), blocks) for arm in ARMS}
        candidate = results["linked_supplied_cost"]
        gates = {}
        for control in ARMS[:-1]:
            other = results[control]
            gates[f"auc_gain_at_least_001_vs_{control}"] = candidate["micro_auc_qratio"] >= other["micro_auc_qratio"] + .001
            gates[f"no_block_auc_loss_vs_{control}"] = all(candidate["per_date_block"][name]["auc_qratio"] >= other["per_date_block"][name]["auc_qratio"] for name in dict.fromkeys(blocks))
            gates[f"no_weak3_auc_loss_vs_{control}"] = candidate["weak3_micro_auc_qratio"] >= other["weak3_micro_auc_qratio"]
            gates[f"no_power_cutoff_mismatch_increase_vs_{control}"] = candidate["power_cutoff_hour_errors"] <= other["power_cutoff_hour_errors"]
        values = {"case_ids": np.asarray(protocol["case_ids"]), "units": np.asarray(UNIT_ORDER), "labels": truth}
        for arm in ARMS:
            values[f"qratio_{arm}"] = np.asarray(qratios[arm])
            values[f"pratio_{arm}"] = np.asarray(pratios[arm])
        for arm in states:
            values[f"state_{arm}"] = np.asarray(states[arm])
        np.savez_compressed(out / "diagnostic.npz", **values)
        report.update(results=results, screening_gates=gates, screen_passed=all(gates.values()),
                      decision="Separate review required before any expansion; no validation/model acceptance implied", status="complete")
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        (out / "failure.txt").write_text(report["error"])
        raise
    finally:
        report["runtime_seconds"] = time.monotonic() - started
        report["peak_rss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (1 if sys.platform == "darwin" else 1024)
        report["artifacts"] = {str(path.relative_to(out)): digest(path) for path in sorted(out.rglob("*"))
                               if path.is_file() and path.name != "manifest.json"}
        (out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
