"""Compose a supplied-only candidate submission from a recipe JSON and verify it.

Recipe format:
{
  "anchor_test": "models/hydraulic_surrogate/submission-lgb-shop664-anchor-blend-001/test_predictions.npy",
  "stages": [
    {"name": "l3b", "components": {"startup142": {"path": "...npy", "weight": 0.25}, "busbar144": {"path": "...npy", "weight": 0.1}}},
    {"name": "legacy", "components": {"legacy": {"csv_mean": ["data/processed/round10_legacy_so_seed299_predictions.csv", ...], "weight": 0.05}}}
  ],
  "tokke_projection": true
}
Each stage: out = project_vinje((1 - sum w) * previous + sum w * component), float64.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blend import project_tokke  # noqa: E402
from common import CLONE, HORIZON, UNITS, InputLedger, check_allowed  # noqa: E402
from sintef_project.postprocess import project_vinje  # noqa: E402


def sha(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def load_component(spec: dict, ids: np.ndarray, ledger: InputLedger) -> np.ndarray:
    if "path" in spec:
        arr = np.load(ledger.add(CLONE / spec["path"])).astype(np.float64)
    elif "npy_mean" in spec:
        arr = np.mean([np.load(ledger.add(CLONE / q)).astype(np.float64) for q in spec["npy_mean"]], axis=0)
    else:
        columns = [f"result_committed_{u}_t{h}" for u in UNITS for h in range(HORIZON)]
        arrays = []
        for p in spec["csv_mean"]:
            frame = pd.read_csv(ledger.add(CLONE / p)).set_index("Run No").loc[ids, columns]
            arrays.append(frame.to_numpy(dtype=np.float64).reshape(len(ids), len(UNITS), HORIZON).transpose(0, 2, 1).reshape(-1, len(UNITS)))
        arr = np.mean(arrays, axis=0)
    assert arr.shape == (len(ids) * HORIZON, len(UNITS)) and np.isfinite(arr).all() and arr.min() >= 0 and arr.max() <= 1
    return arr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    out = CLONE / "models/supplied_only/exports" / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    recipe = json.loads(Path(args.recipe).read_text())
    ledger = InputLedger()
    template = pd.read_csv(ledger.add(CLONE / "data/raw/sample_submission.csv"))
    ids = template["Run No"].to_numpy()
    assert template.columns[1:].tolist() == [f"result_committed_{u}_t{h}" for u in UNITS for h in range(HORIZON)]
    current = np.load(ledger.add(CLONE / recipe["anchor_test"])).astype(np.float64)
    assert current.shape == (len(ids) * HORIZON, len(UNITS))
    stage_log = []
    for stage in recipe["stages"]:
        total = sum(c["weight"] for c in stage["components"].values())
        mix = (1 - total) * current
        for name, spec in stage["components"].items():
            mix = mix + spec["weight"] * load_component(spec, ids, ledger)
        current = project_vinje(mix)
        stage_log.append({"name": stage["name"], "weights": {k: v["weight"] for k, v in stage["components"].items()}})
    if recipe.get("tokke_projection"):
        current = project_tokke(current)
    assert np.isfinite(current).all() and current.min() >= 0 and current.max() <= 1
    wide = current.reshape(len(ids), HORIZON, len(UNITS)).transpose(0, 2, 1).reshape(len(ids), -1)
    frame = pd.DataFrame(wide, columns=template.columns[1:]); frame.insert(0, "Run No", ids)
    csv = out / "submission.csv"
    frame.to_csv(csv, index=False, float_format="%.9g")
    reread = pd.read_csv(csv)
    assert reread.columns.tolist() == template.columns.tolist() and reread["Run No"].tolist() == ids.tolist()
    assert np.max(np.abs(reread.iloc[:, 1:].to_numpy() - wide)) <= 5.000001e-10
    np.save(out / "test_predictions.npy", current)
    ledger.write(out / "inputs.json")
    manifest = {"run_id": args.run_id, "timestamp_utc": datetime.now(timezone.utc).isoformat(), "recipe": recipe,
                "stages": stage_log, "submission_sha256": sha(csv), "rows": int(len(ids)), "columns": int(frame.shape[1]),
                "min": float(current.min()), "max": float(current.max()), "mean": float(current.mean())}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: manifest[k] for k in ("run_id", "submission_sha256", "rows", "columns", "stages")}, indent=1))


if __name__ == "__main__":
    main()
