"""Score the predeclared monotone LP discharge proxy; no fitting or tuning."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]


def checksum(path: Path) -> str:
    """Return SHA-256."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Evaluate independent 2021/2022 input-only LP scores against known labels."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="lp-v2-diagnostic-001")
    parser.add_argument("--cache", type=Path, default=ROOT / "artifacts/lp-v2-full-001")
    args = parser.parse_args()
    started = time.monotonic()
    out = ROOT / "artifacts" / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    cache_manifest = json.loads((args.cache / "manifest.json").read_text())
    if cache_manifest["status"] != "complete":
        raise ValueError("LP cache incomplete")
    cache_path = args.cache / "features.npz"
    if checksum(cache_path) != cache_manifest["artifacts"]["features.npz"]:
        raise ValueError("LP cache checksum mismatch")
    cache = np.load(cache_path)
    units = cache["units"].tolist()
    columns = cache["columns"].tolist()
    selected_columns = [columns.index(f"lp2_qmin_ratio_{unit}") for unit in units]
    ratio = cache["features"][:, :, selected_columns]
    # Fixed monotonic bounded mapping. No threshold or calibration is fitted;
    # pooled AUC is identical to ranking directly by q/Qmin.
    probabilities = ratio / (1 + ratio)
    path = ROOT / "sintef_project/data/kernel/Unit_commitment_decisions.csv"
    training = pd.read_csv(path)
    if not np.array_equal(cache["case_ids"][:len(training)], training["Run No"].to_numpy()):
        raise ValueError("Case IDs mismatch")
    dates = pd.to_datetime(training.starttime)
    labels = np.stack([training[[f"result_committed_{unit}_t{h}" for h in range(168)]].to_numpy(dtype=np.uint8) for unit in units], axis=2)
    manifest = {"run_id": args.run_id, "timestamp_utc": datetime.now(timezone.utc).isoformat(), "status": "complete",
                "command": shlex.join([sys.executable, *sys.argv]), "method": "p=(q/Qmin)/(1+q/Qmin), fixed monotone proxy; no trained parameters, clipping, fitted calibration, threshold tuning or labels in LP",
                "source_sha256": checksum(Path(__file__)), "label_path": str(path), "label_sha256": checksum(path),
                "cache_manifest_path": str(args.cache / "manifest.json"), "cache_manifest_sha256": checksum(args.cache / "manifest.json"),
                "cache_features_sha256": checksum(cache_path), "environment": cache_manifest["environment"], "seed": None,
                "folds": {}, "kaggle": {"submission_id": None, "score": None, "rank": None, "observation_time": None}}
    shutil.copy2(Path(__file__), out / Path(__file__).name)
    for year in (2021, 2022):
        positions = np.flatnonzero(dates.dt.year.to_numpy() == year)
        truth, prediction = labels[positions], probabilities[positions]
        fold = {"case_ids": training.iloc[positions]["Run No"].tolist(), "date_range": [str(dates.iloc[positions].min()), str(dates.iloc[positions].max())],
                "micro_auc": float(roc_auc_score(truth.ravel(), prediction.ravel())),
                "unit_auc": {unit: float(roc_auc_score(truth[:, :, j].ravel(), prediction[:, :, j].ravel())) for j, unit in enumerate(units)}}
        manifest["folds"][str(year)] = fold
        np.savez_compressed(out / f"validation_{year}.npz", predictions=prediction.reshape(-1, len(units)), labels=truth.reshape(-1, len(units)), case_ids=training.iloc[positions]["Run No"].to_numpy(), units=units)
        print(year, json.dumps(fold), flush=True)
    manifest["runtime_seconds"] = time.monotonic() - started
    manifest["artifacts"] = {p.name: checksum(p) for p in out.iterdir() if p.is_file()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
