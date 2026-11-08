"""L4: extra-seed canonical115 fits on the saved supplied-only feature matrices.

Reads the audited canonical115 matrices (features built from supplied inputs and
local LP caches only), fits the same 1,400-tree pooled LightGBM with a different
seed and row bagging, and saves raw validation/test predictions.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ARTIFACTS, OUT, REPORTS, InputLedger  # noqa: E402

PARAMS = {"boosting_type": "gbdt", "colsample_bytree": 0.9, "learning_rate": 0.06, "max_depth": -1,
          "min_child_samples": 100, "min_child_weight": 0.001, "min_split_gain": 0.0, "n_estimators": 1400,
          "num_leaves": 31, "objective": "binary", "reg_alpha": 0.0, "reg_lambda": 1.0,
          "subsample": 0.8, "subsample_freq": 1, "subsample_for_bin": 200000, "max_bin": 255,
          "deterministic": True, "force_col_wise": True, "verbosity": -1}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", choices=["2021", "2022", "final"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    run = ARTIFACTS / f"physics-trick-canonical115-{args.year}-001"
    out = OUT / f"seed_bag/canonical115_{args.year}_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    ledger = InputLedger()
    xt = np.load(ledger.add(run / "train_canonical.npy"), mmap_mode="r")
    y = np.load(ledger.add(run / "training_labels.npy")).ravel()
    xv = np.load(ledger.add(run / "prediction_canonical.npy"), mmap_mode="r")
    assert xt.shape[1] == 115 and xv.shape[1] == 115 and len(xt) == len(y)
    tick = time.monotonic()
    params = dict(PARAMS, random_state=args.seed, n_jobs=args.threads)
    model = lgb.LGBMClassifier(**params)
    model.fit(np.asarray(xt), y, categorical_feature=[111, 112])
    fit_seconds = time.monotonic() - tick
    model.booster_.save_model(str(out / "model.txt"))
    pred = np.empty(len(xv), dtype=np.float64)
    for b in range(0, len(xv), 65536):
        pred[b:b + 65536] = model.booster_.predict(np.asarray(xv[b:b + 65536]), num_threads=args.threads)
    pred = pred.reshape(-1, 14)
    np.save(out / "raw_predictions.npy", pred)
    ledger.write(out / "inputs.json")
    (out / "manifest.json").write_text(json.dumps({"year": args.year, "seed": args.seed, "params": params,
                                                   "fit_seconds": fit_seconds, "rows": int(len(y))}, indent=1))
    print(json.dumps({"year": args.year, "seed": args.seed, "fit_seconds": round(fit_seconds, 1)}))


if __name__ == "__main__":
    main()
