"""Independently reconstruct a frozen SHOP/fullsample blend and ranking score."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sintef_project.postprocess import project_vinje

PROTOCOL_SHA = "56c35e3a5d7d76e5be5631728e0bbd44b45a14b2228d9bbfd7e26e722e7fbba7"


def sha(path: Path) -> str:
    """Hash exact evidence bytes."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def independent_projection(values: np.ndarray, units: list[str]) -> np.ndarray:
    """Use the isotonic min-max characterization instead of producer branches."""
    indices = [units.index(f"Vinje_G{i}") for i in (1, 2, 3)]
    result = values.copy()
    x = values[:, indices]
    for i in range(3):
        result[:, indices[i]] = np.min(np.stack([
            np.max(np.stack([x[:, a:b + 1].mean(axis=1)
                             for b in range(i, 3)]), axis=0)
            for a in range(i + 1)]), axis=0)
    return result


def independent_auc(labels: np.ndarray, prediction: np.ndarray) -> float:
    """Compute Mann-Whitney AUC with average ranks for ties."""
    y, p = labels.ravel(), prediction.ravel()
    positive = y == 1
    count = int(positive.sum())
    return float((rankdata(p, method="average")[positive].sum()
                  - count * (count + 1) / 2) / (count * (len(y) - count)))


def main() -> None:
    """Audit only the requested year and retain reproducible proof or failure."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, choices=[2021, 2022], required=True)
    parser.add_argument("--producer", type=Path, required=True)
    parser.add_argument("--producer-sha256", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--prior-audit", type=Path)
    parser.add_argument("--prior-audit-sha256")
    args = parser.parse_args()
    assert Path(args.run_id).name == args.run_id and args.run_id not in ("", ".", "..")
    out = ROOT / "artifacts" / args.run_id
    out.mkdir(exist_ok=False)
    start = time.monotonic()
    report = {"status": "running", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "command": [sys.executable, *sys.argv], "year": args.year,
              "training": False, "protocol_sha256": PROTOCOL_SHA,
              "producer_manifest_sha256": args.producer_sha256,
              "environment": {"python": sys.version, "platform": platform.platform(),
                              "numpy": np.__version__}}
    try:
        protocol_path = ROOT / "docs/fullsample608_shop_reuse_protocol.json"
        assert sha(protocol_path) == PROTOCOL_SHA
        protocol = json.loads(protocol_path.read_text())
        for group in ("source", "inputs"):
            for name, expected in protocol[group].items():
                assert sha(ROOT / name) == expected, name
        assert sha(args.producer / "manifest.json") == args.producer_sha256
        producer = json.loads((args.producer / "manifest.json").read_text())
        assert producer["status"] == "complete" and producer["year"] == args.year
        assert producer["protocol_sha256"] == PROTOCOL_SHA
        for name, expected in producer["artifacts"].items():
            assert sha(args.producer / name) == expected, name

        lock_path = (args.producer if args.year == 2021 else
                     ROOT / "artifacts/fullsample608-shop-reuse-2021-001") / "selection_2021.json"
        lock = json.loads(lock_path.read_text())
        assert sha(lock_path) == producer["selection_lock_sha256"]
        assert lock["protocol_sha256"] == PROTOCOL_SHA and lock["candidate_weights"] == [0., .1, .25, .5]
        assert lock["selector_sha256"] == protocol["source"]["scripts/select_fullsample608_shop_reuse.py"]
        if args.year == 2022:
            assert args.prior_audit and sha(args.prior_audit) == args.prior_audit_sha256
            prior = json.loads(args.prior_audit.read_text())
            assert prior["status"] == "passed" and prior["eligible_for_2022"]
            assert prior["selection_lock_sha256"] == sha(lock_path)
            assert producer["lock_audit_sha256"] == args.prior_audit_sha256
            assert lock["eligible_for_2022"] and lock["selected_weight"] > 0
            report["prior_audit_sha256"] = args.prior_audit_sha256

        fold = protocol["folds"][str(args.year)]
        components, labels = {}, None
        for role, record in fold["components"].items():
            assert sha(ROOT / record["manifest"]) == record["manifest_sha256"]
            manifest = json.loads((ROOT / record["manifest"]).read_text())
            assert manifest["status"] == "complete"
            assert manifest["artifacts"][Path(record["prediction"]).name] == record["prediction_sha256"]
            with np.load(ROOT / record["prediction"], allow_pickle=False) as saved:
                assert saved["case_ids"].tolist() == fold["validation_ids"]
                assert saved["units"].tolist() == protocol["units"]
                assert saved["predictions"].shape == saved["labels"].shape == (61320, 14)
                if labels is None:
                    labels = saved["labels"].copy()
                else:
                    np.testing.assert_array_equal(labels, saved["labels"])
                components[role] = saved["predictions"].astype(np.float64)
                assert np.isfinite(components[role]).all()
                assert 0 <= components[role].min() <= components[role].max() <= 1
        assert np.isin(labels, [0, 1]).all()
        weights = [0., .1, .25, .5] if args.year == 2021 else [0., lock["selected_weight"]]
        predictions, scores = {}, {}
        projection_error = 0.
        for weight in weights:
            raw = (1 - weight) * components["anchor"] + weight * components["candidate"]
            exact = project_vinje(raw, protocol["units"])
            independent = independent_projection(raw, protocol["units"])
            projection_error = max(projection_error, float(np.max(np.abs(exact - independent))))
            np.testing.assert_allclose(exact, independent, rtol=0, atol=5e-16)
            predictions[weight] = exact
            scores[str(weight)] = independent_auc(labels, exact)
            assert abs(scores[str(weight)] - producer["scores"][str(weight)]) < 1e-14
        selected = (max(weights, key=lambda w: (scores[str(w)], -w))
                    if args.year == 2021 else lock["selected_weight"])
        assert selected == producer["selected_weight"] == lock["selected_weight"]
        matched = project_vinje((1 - selected) * components["anchor"]
                                + selected * components["control"], protocol["units"])
        control_auc = independent_auc(labels, matched)
        assert abs(control_auc - producer["control_micro_auc"]) < 1e-14
        eligible = bool(selected > 0 and scores[str(selected)] > max(scores["0.0"], control_auc))
        assert eligible == producer["beats_anchor_and_control"]
        if args.year == 2021:
            assert eligible == lock["eligible_for_2022"]
            for weight, score in scores.items():
                assert abs(score - lock["scores_2021"][weight]) < 1e-14
        else:
            assert eligible == producer["accepted"]
        with np.load(args.producer / f"selected_validation_{args.year}.npz", allow_pickle=False) as saved:
            assert saved["predictions"].dtype == np.float64
            assert saved["predictions"].tobytes() == predictions[selected].tobytes()
            np.testing.assert_array_equal(labels, saved["labels"])
            assert saved["case_ids"].tolist() == fold["validation_ids"]
            assert saved["units"].tolist() == protocol["units"]
        report.update(status="passed", selection_lock_sha256=sha(lock_path),
                      selected_weight=selected, independent_scores=scores,
                      independent_control_auc=control_auc, independent_projection_max_error=projection_error,
                      exact_selected_array_bytes=True,
                      eligible_for_2022=eligible if args.year == 2021 else None,
                      accepted=eligible if args.year == 2022 else None,
                      caveat="Reused chronological development, not untouched holdout or public score.")
    except BaseException:
        report.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        shutil.copy2(__file__, out / Path(__file__).name)
        report["audit_source_sha256"] = sha(Path(__file__))
        report["runtime_seconds"] = time.monotonic() - start
        (out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"status": report["status"], "manifest_sha256": sha(out / "manifest.json"),
                          "runtime_seconds": report["runtime_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
