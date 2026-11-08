"""Optimizer-only submission: rank unit-hours by the LP relaxation state, no fitted model.

Inputs are the two completed, label-free LP caches (supplied-only ancestry):
- lp-shop-full-001: fractional commitment u from the fixed-head LP with supplied
  start/stop costs (`shop_state_<unit>`).
- lp-full-001: base-LP discharge over minimum discharge (`lp_qmin_ratio_<unit>`).

Candidates are predeclared monotone transforms with no fitted parameters; the only
choice made on labels is which candidate to export, selected on 2021 and reported
unchanged on 2022. Nothing is trained, so no six-day purge applies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shlex
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments/hydraulic_surrogate"))
from blend import project_tokke  # noqa: E402
from common import CLONE, HORIZON, UNITS, InputLedger, micro_auc  # noqa: E402
from sintef_project.postprocess import project_vinje  # noqa: E402

SHOP_CACHE = CLONE / "models/hydraulic_surrogate/lp-shop-full-001"
BASE_CACHE = CLONE / "models/hydraulic_surrogate/lp-full-001"
LABELS = CLONE / "data/raw/kernel/Unit_commitment_decisions.csv"
TEMPLATE = CLONE / "data/raw/sample_submission.csv"
N_CASES, N_TRAIN = 3647, 2922
YEARS = (2021, 2022)


def sha(path: Path) -> str:
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_cache(directory: Path, prefix: str, ledger: InputLedger) -> tuple[np.ndarray, dict]:
    """Return (3647, 168, 14) float64 values for `<prefix>_<unit>` after verifying the cache."""
    manifest = json.loads((ledger.add(directory / "manifest.json")).read_text())
    assert manifest["status"] == "complete", directory
    path = ledger.add(directory / "features.npz")
    assert sha(path) == manifest["artifacts"]["features.npz"], f"feature bytes changed: {path}"
    with np.load(path, allow_pickle=False) as z:
        assert z["case_ids"].tolist() == list(range(1, N_CASES + 1)), directory
        assert z["units"].tolist() == UNITS, directory
        columns = z["columns"].tolist()
        index = [columns.index(f"{prefix}_{unit}") for unit in UNITS]
        values = z["features"][:, :, index].astype(np.float64)
    assert values.shape == (N_CASES, HORIZON, len(UNITS)) and np.isfinite(values).all()
    return values, {"directory": str(directory), "manifest_sha256": sha(directory / "manifest.json"),
                    "features_sha256": manifest["artifacts"]["features.npz"], "columns": [f"{prefix}_{u}" for u in UNITS]}


def candidates(state: np.ndarray, ratio: np.ndarray) -> dict[str, np.ndarray]:
    """Predeclared label-free scores, each (n_cases * 168, 14) in [0, 1]."""
    u = np.clip(state, 0.0, 1.0).reshape(-1, len(UNITS))
    q = np.maximum(ratio, 0.0).reshape(-1, len(UNITS))
    q = q / (1.0 + q)
    return {"state": u, "qproxy": q, "state_tiebreak": np.clip(u + 0.01 * q, 0.0, 1.0), "mean": 0.5 * u + 0.5 * q}


POSTPROCESS = {
    "none": lambda p: p,
    "vinje": project_vinje,
    "vinje_tokke": lambda p: project_tokke(project_vinje(p)),
}


def load_labels(ledger: InputLedger) -> tuple[pd.Series, np.ndarray]:
    """Labels as (n_train * 168, 14) in case-major, hour-major order; years per case."""
    columns = [f"result_committed_{u}_t{h}" for u in UNITS for h in range(HORIZON)]
    frame = pd.read_csv(ledger.add(LABELS), usecols=["Run No", "starttime", *columns], dtype={c: "uint8" for c in columns})
    assert frame["Run No"].tolist() == list(range(1, N_TRAIN + 1))
    years = pd.to_datetime(frame["starttime"], utc=True).dt.year
    labels = frame[columns].to_numpy().reshape(N_TRAIN, len(UNITS), HORIZON).transpose(0, 2, 1).reshape(-1, len(UNITS))
    return years, labels


def evaluate(scores: np.ndarray, labels: np.ndarray, years: pd.Series) -> dict:
    result = {}
    for year in YEARS:
        rows = np.repeat((years == year).to_numpy(), HORIZON)
        truth, prediction = labels[rows], scores[: N_TRAIN * HORIZON][rows]
        result[str(year)] = {"micro_auc": micro_auc(truth, prediction),
                             "unit_auc": {u: micro_auc(truth[:, j], prediction[:, j]) for j, u in enumerate(UNITS)},
                             "cases": int(rows.sum() // HORIZON)}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    started = time.monotonic()
    out = CLONE / "models/supplied_only/exports" / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    ledger = InputLedger()
    state, shop_meta = load_cache(SHOP_CACHE, "shop_state", ledger)
    ratio, base_meta = load_cache(BASE_CACHE, "lp_qmin_ratio", ledger)
    years, labels = load_labels(ledger)
    grid = {}
    for score_name, score in candidates(state, ratio).items():
        for post_name, post in POSTPROCESS.items():
            values = post(score)
            assert values.shape == score.shape and np.isfinite(values).all() and values.min() >= 0 and values.max() <= 1
            grid[f"{score_name}/{post_name}"] = evaluate(values, labels, years)
            print(f"{score_name:15s} {post_name:12s} 2021 {grid[f'{score_name}/{post_name}']['2021']['micro_auc']:.9f}  "
                  f"2022 {grid[f'{score_name}/{post_name}']['2022']['micro_auc']:.9f}", flush=True)
    # Sanity: the plain q/Qmin proxy must reproduce the recorded lp-diagnostic-002 numbers.
    reference = {"2021": 0.946964869, "2022": 0.952251082}
    for year, expected in reference.items():
        observed = grid["qproxy/none"][year]["micro_auc"]
        assert abs(observed - expected) < 1e-6, (year, observed, expected)
    selected = max(grid, key=lambda name: grid[name]["2021"]["micro_auc"])
    score_name, post_name = selected.split("/")
    final = POSTPROCESS[post_name](candidates(state, ratio)[score_name])
    # Export the 725 test cases in template order.
    template = pd.read_csv(ledger.add(TEMPLATE))
    ids = template["Run No"].to_numpy()
    assert template.columns[1:].tolist() == [f"result_committed_{u}_t{h}" for u in UNITS for h in range(HORIZON)]
    cube = final.reshape(N_CASES, HORIZON, len(UNITS))[ids - 1]
    wide = cube.transpose(0, 2, 1).reshape(len(ids), -1)
    frame = pd.DataFrame(wide, columns=template.columns[1:]); frame.insert(0, "Run No", ids)
    csv = out / "submission.csv"
    frame.to_csv(csv, index=False, float_format="%.9g")
    reread = pd.read_csv(csv)
    assert reread.shape == template.shape and reread.columns.tolist() == template.columns.tolist()
    assert reread["Run No"].tolist() == ids.tolist()
    assert np.isfinite(reread.iloc[:, 1:].to_numpy()).all()
    assert reread.iloc[:, 1:].to_numpy().min() >= 0 and reread.iloc[:, 1:].to_numpy().max() <= 1
    assert np.max(np.abs(reread.iloc[:, 1:].to_numpy() - wide)) <= 5.000001e-10
    np.save(out / "test_predictions.npy", cube.reshape(-1, len(UNITS)))
    ledger.write(out / "inputs.json")
    manifest = {"run_id": args.run_id, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "command": shlex.join([sys.executable, *sys.argv]), "training": False, "fitted_parameters": None,
                "method": "Monotone transforms of label-free LP cache columns; candidate chosen by 2021 micro AUC, "
                          "frozen, reported unchanged on 2022. No purge needed: nothing is trained.",
                "candidates": {"state": "u = shop_state clipped to [0,1]", "qproxy": "(q/Qmin)/(1+q/Qmin)",
                               "state_tiebreak": "clip(u + 0.01*qproxy, 0, 1)", "mean": "0.5*u + 0.5*qproxy"},
                "postprocess": list(POSTPROCESS), "selected": selected, "selection_rule": "argmax 2021 micro_auc",
                "selected_scores": grid[selected], "grid": grid, "shop_cache": shop_meta, "base_cache": base_meta,
                "submission_sha256": sha(csv), "rows": int(len(ids)), "columns": int(frame.shape[1]),
                "min": float(cube.min()), "max": float(cube.max()), "mean": float(cube.mean()),
                "environment": {"python": sys.version, "platform": platform.platform(),
                                "numpy": np.__version__, "pandas": pd.__version__},
                "kaggle": {"submission_id": None, "public_score": None, "rank": None, "observation_time": None},
                "runtime_seconds": time.monotonic() - started}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({"selected": selected, "scores": {y: grid[selected][y]["micro_auc"] for y in map(str, YEARS)},
                      "submission_sha256": manifest["submission_sha256"], "csv": str(csv)}, indent=1))


if __name__ == "__main__":
    main()
