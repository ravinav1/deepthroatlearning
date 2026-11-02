"""Build a label-free supplied-cost relaxed-state cache after explicit review."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
from unittest.mock import patch

for variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ[variable] = "1"

import numpy as np
import pandas as pd

SOURCE_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("TDT4173_WORKSPACE_ROOT", SOURCE_ROOT)).resolve()
sys.path.insert(0, str(SOURCE_ROOT))

from sintef_project.lp_features import HydroCaseData, UNIT_ORDER
from sintef_project.lp_shop_features import COLUMNS, compute_features, digest, feature_specification
from sintef_project.lp_shop_relaxation import ShopRelaxationLP, specification


def load_npz(path: Path) -> dict:
    """Read numeric arrays without pickle or executable deserialization."""
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key].copy() for key in z}


def verify_saved_replay(result: dict, replay: dict) -> None:
    """Require every raw solver array and scalar to repeat bitwise."""
    if set(result) != set(replay):
        raise ValueError("Saved replay field set differs")
    for key in result:
        np.testing.assert_array_equal(result[key], replay[key], err_msg=key)


def main() -> None:
    """Generate only frozen input-derived features, preserving all completed cases."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--reuse-cache", type=Path)
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id or args.run_id in ["", ".", ".."]:
        raise ValueError("Invalid single-component run ID")
    out = ROOT / "artifacts" / args.run_id
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    report = {"run_id": args.run_id, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "command": shlex.join([sys.executable, *sys.argv]), "status": "initializing",
              "training": False, "label_decode": False, "kaggle_submission": None,
              "source_root": str(SOURCE_ROOT), "workspace_root": str(ROOT),
              "case_files": {}, "case_records": [], "reused_diagnostic_ids": [], "reused_cache_ids": [],
              "environment": {"python": sys.version, "platform": platform.platform(),
                              "machine": platform.machine(), "cpu_count": os.cpu_count(),
                              "packages": {name: importlib.metadata.version(name) for name in
                                           ["numpy", "pandas", "scipy", "PyYAML"]}}}
    try:
        protocol_path = args.protocol if args.protocol.is_absolute() else ROOT / args.protocol
        if digest(protocol_path) != args.protocol_sha256:
            raise ValueError("Explicit reviewed cache protocol SHA differs")
        protocol = json.loads(protocol_path.read_text())
        if protocol["solver_specification"] != specification() or protocol["feature_specification"] != feature_specification():
            raise ValueError("Solver or feature definition differs from the reviewed protocol")
        for name, expected in protocol["source"].items():
            if digest(SOURCE_ROOT / name) != expected:
                raise ValueError(f"Frozen source changed: {name}")
            target = out / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(SOURCE_ROOT / name, target)
        for name, expected in protocol["data"].items():
            if digest(ROOT / name) != expected:
                raise ValueError(f"Frozen input bytes changed: {name}")
        baseline = ROOT / protocol["baseline_cache"]
        if digest(baseline / "manifest.json") != protocol["baseline_manifest_sha256"]:
            raise ValueError("Frozen baseline cache manifest changed")
        bm = json.loads((baseline / "manifest.json").read_text())
        reference = ROOT / protocol["diagnostic_run"]
        if digest(reference / "manifest.json") != protocol["diagnostic_manifest_sha256"]:
            raise ValueError("Verified36case diagnostic changed")
        dm = json.loads((reference / "manifest.json").read_text())
        audit_path = ROOT / protocol["diagnostic_audit"]
        if digest(audit_path) != protocol["diagnostic_audit_sha256"]:
            raise ValueError("Independent36case audit changed")
        audit = json.loads(audit_path.read_text())
        if dm["status"] != "complete" or not dm["screen_passed"] or audit["status"] != "complete" or not audit["all_stored_replays_bitwise"]:
            raise ValueError("Diagnostic and independent audit gates are incomplete")
        reuse = None
        if args.reuse_cache:
            reuse = json.loads((args.reuse_cache / "manifest.json").read_text())
            if reuse["status"] != "failed" or reuse["protocol_sha256"] != args.protocol_sha256:
                raise ValueError("Resume requires a failed cache with the exact same frozen protocol")
            report["reuse_manifest_sha256"] = digest(args.reuse_cache / "manifest.json")
        label_path = (ROOT / "sintef_project/data/kernel/Unit_commitment_decisions.csv").resolve()
        original_read = pd.read_csv

        def metadata_only_read(path, *positional, **kwargs):
            if isinstance(path, (str, Path)) and Path(path).resolve() == label_path:
                if positional or kwargs != {"usecols": ["Run No", "starttime"]}:
                    raise ValueError("Full feature cache may read commitment metadata only")
            return original_read(path, *positional, **kwargs)

        with patch.object(pd, "read_csv", metadata_only_read):
            inputs = HydroCaseData(ROOT)
            ids = inputs.cases["Run No"].tolist()
            dates = inputs.cases.starttime.astype(str).tolist()
            if ids != protocol["case_ids"] or dates != protocol["case_dates"] or len(ids) != 3647:
                raise ValueError("Full case metadata scope differs")
            report.update(protocol_sha256=args.protocol_sha256, source=protocol["source"], data=protocol["data"],
                          baseline_manifest_path=str((baseline / "manifest.json").relative_to(ROOT)),
                          baseline_manifest_sha256=protocol["baseline_manifest_sha256"],
                          diagnostic_manifest_sha256=protocol["diagnostic_manifest_sha256"],
                          diagnostic_audit_sha256=protocol["diagnostic_audit_sha256"],
                          feature_specification=feature_specification(), prefix_feature_columns=protocol["prefix_feature_columns"],
                          case_ids=ids, case_dates=dates, feature_shape=[3647, 168, 56], status="building_input_only")
            (out / "protocol_before_solves.json").write_text(json.dumps(report, indent=2) + "\n")
            features = np.empty((3647, 168, 56), dtype=np.float32)
            solver = ShopRelaxationLP(inputs.network)
            for index, case_id in enumerate(ids):
                tick = time.monotonic()
                case = inputs.case(index)
                case["natural"] = np.asarray(case["natural"], dtype=np.float64)
                for name, field in [("pmin", "p_min"), ("start_cost", "startcost_const"), ("stop_cost", "stopcost_const")]:
                    case[name] = np.asarray([inputs.model["generator"][unit][field] for unit in UNIT_ORDER], dtype=np.float64)
                baseline_name = f"case_{case_id:04d}.npz"
                if digest(baseline / baseline_name) != bm["artifacts"][baseline_name]:
                    raise ValueError(f"Original LP case changed: {case_id}")
                old = load_npz(baseline / baseline_name)
                names = {kind: f"case_{case_id:04d}_{kind}.npz" for kind in ["result", "replay", "inputs"]}
                if case_id in dm["case_ids"]:
                    reference_names = {"result": f"case_{case_id:04d}_linked_supplied_cost.npz",
                                       "replay": f"case_{case_id:04d}_replay.npz", "inputs": f"case_{case_id:04d}_inputs.npz"}
                    for kind, name in reference_names.items():
                        if digest(reference / name) != dm["artifacts"][name]:
                            raise ValueError("Retained36case raw result/replay/input changed")
                        shutil.copy2(reference / name, out / names[kind])
                    report["reused_diagnostic_ids"].append(case_id)
                elif reuse is not None and all(name in reuse["case_files"] for name in names.values()):
                    for name in names.values():
                        if digest(args.reuse_cache / name) != reuse["case_files"][name]:
                            raise ValueError("Retained failed-cache case changed")
                        shutil.copy2(args.reuse_cache / name, out / name)
                    report["reused_cache_ids"].append(case_id)
                else:
                    result = solver.solve(case, 1.)
                    np.savez_compressed(out / names["result"], **result)
                    replay = ShopRelaxationLP(inputs.network).solve(case, 1.)
                    np.savez_compressed(out / names["replay"], **replay)
                    np.savez_compressed(out / names["inputs"], **case)
                result = load_npz(out / names["result"])
                verify_saved_replay(result, load_npz(out / names["replay"]))
                saved_input = load_npz(out / names["inputs"])
                if set(saved_input) != set(case):
                    raise ValueError("Retained input field set differs")
                for key in case:
                    np.testing.assert_array_equal(saved_input[key], case[key], err_msg=f"Retained case {case_id}: {key}")
                features[index] = compute_features(case, result, old, inputs.network)
                for name in names.values():
                    report["case_files"][name] = digest(out / name)
                report["case_records"].append({"case_id": case_id,
                    "runtime_seconds": time.monotonic() - tick, "fresh_or_retained_replay_bitwise": True,
                    "balance_residual": float(result["balance_residual"]),
                    "total_balance_error": float(result["total_balance_error"]),
                    "slack_max": {key: float(np.max(result[key])) for key in
                                  ["minimum_slack", "maximum_slack", "bypass_slack", "vest_slack"]},
                    "transition_cost": float(result["transition_cost"])})
                if (index + 1) % 25 == 0 or index + 1 == len(ids):
                    (out / "progress.json").write_text(json.dumps({"completed": index+1, "total": len(ids),
                        "elapsed_seconds": time.monotonic()-started, "last_case_id": case_id}, indent=2) + "\n")
                    print(json.dumps({"completed": index+1, "total": len(ids), "seconds": round(time.monotonic()-started, 3)}), flush=True)
            np.savez_compressed(out / "features.npz", features=features, case_ids=np.asarray(ids),
                                columns=np.asarray(COLUMNS), units=np.asarray(UNIT_ORDER))
            with np.load(out / "features.npz", allow_pickle=False) as z:
                np.testing.assert_array_equal(z["features"], features)
                np.testing.assert_array_equal(z["case_ids"], ids)
            if report["reused_diagnostic_ids"] != dm["case_ids"]:
                raise ValueError("Full cache did not preserve all36reviewed cases")
            report.update(status="complete", all_replays_bitwise=True, diagnostic36_raw_agreement_bitwise=True,
                          label_reads="Only Run No/starttime metadata under rejecting pd.read_csv guard", features_finite=True)
    except BaseException:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        (out / "failure.txt").write_text(report["error"])
        raise
    finally:
        report["runtime_seconds"] = time.monotonic() - started
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["peak_rss_bytes"] = int(rss) * (1 if sys.platform == "darwin" else 1024)
        report["artifacts"] = {path.relative_to(out).as_posix(): digest(path) for path in sorted(out.rglob("*"))
                               if path.is_file() and path.name != "manifest.json"}
        (out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        with (ROOT / "docs/experiments.md").open("a") as handle:
            handle.write(f'\n\n## {args.run_id} — {report["timestamp_utc"]}\n\nStatus {report["status"]}; label-free compact56feature cache only, no fitted model. Command `{report["command"]}`. Exact source/data/protocol/cases/rawinputs/replays/features and failed-run reuse provenance in `artifacts/{args.run_id}/manifest.json`. Original608 prefix preserved for later addition; same supplied-cost solver,36verified cases reused, no target values decoded. Runtime {report["runtime_seconds"]:.3f}s, peakRSS {report["peak_rss_bytes"]}bytes. No validation, submission or transfer claim.\n')


if __name__ == "__main__":
    main()
