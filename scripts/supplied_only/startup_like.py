"""Startup-style pooled model: canonical115 + 16 supplied-topology start/stop economics fields.

Numerical inputs: the rebuilt 664-column supplied-only table (features664.npy), the supplied
topology, original training labels. No external numbers. Fit with the frozen pooled recipe
(1,400 trees, 31 leaves, lr 0.06, categorical unit/plant ids); blend on the 0.99652 anchor.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

CLONE = Path(__file__).resolve().parents[2]
ROOT = CLONE / "experiments/hydraulic_surrogate"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, REPORTS, UNITS, PLANTS, micro_auc  # noqa: E402
from blend import load_fold  # noqa: E402
from sintef_project.canonical_unit_features import feature_specification, canonical_block  # noqa: E402
from sintef_project.postprocess import project_vinje  # noqa: E402

PARAMS = {"boosting_type": "gbdt", "colsample_bytree": 0.9, "learning_rate": 0.06, "max_depth": -1,
          "min_child_samples": 100, "min_child_weight": 0.001, "min_split_gain": 0.0, "n_estimators": 1400,
          "num_leaves": 31, "objective": "binary", "reg_alpha": 0.0, "reg_lambda": 1.0, "subsample": 1.0,
          "subsample_freq": 0, "subsample_for_bin": 200000, "max_bin": 255, "deterministic": True,
          "force_col_wise": True, "verbosity": -1, "random_state": 4173}
WEIGHTS = [0, .1, .25, .5]
STARTUP_COLUMNS = ["su_pmin_frac", "su_pmin_scaled", "su_pmax_scaled", "su_start_cost_per_pmax", "su_cycle_cost_per_pmin",
                   "su_margin_min_minus_cycle", "su_margin_max_minus_cycle", "su_lp_margin_times_pmin", "su_lp_margin_times_pmax",
                   "su_recover_hours_pmin", "su_recover_hours_pmax", "su_margin_next6", "su_margin_prev6",
                   "su_margin_next24", "su_margin_prev24", "su_margin_remaining"]
CROSSCASE_COLUMNS = [f"cc_state_d{d:+d}" for d in (-3, -2, -1, 1, 2, 3)] + ["cc_state_mean", "cc_count"]


def crosscase_table(x: pd.DataFrame, cases: pd.DataFrame) -> np.ndarray:
    """(n_cases, 168, 14, 8): own-unit SHOP relaxed state of the case starting d days later/earlier at the same calendar hour."""
    n = len(cases)
    state = np.stack([x[f"shop_state_{u}"].to_numpy(dtype=np.float32).reshape(n, 168) for u in UNITS], axis=2)  # (n,168,14)
    starts = pd.to_datetime(cases["starttime"], utc=True)
    day = ((starts - starts.iloc[0]).dt.total_seconds() / 86400).round().astype(int).to_numpy()
    lookup = {d: i for i, d in enumerate(day)}
    out = np.empty((n, 168, 14, 8), dtype=np.float32)
    acc = np.zeros((n, 168, 14), dtype=np.float32); cnt = np.zeros((n, 168, 14), dtype=np.float32)
    hours = np.arange(168)
    for k, d in enumerate((-3, -2, -1, 1, 2, 3)):
        feat = state.copy()  # fallback: own state when the neighbour is unavailable
        for i in range(n):
            j = lookup.get(day[i] + d)
            if j is None:
                continue
            hp = hours - 24 * d  # neighbour's horizon hour for the same calendar hour
            ok = (hp >= 0) & (hp < 168)
            feat[i, ok] = state[j, hp[ok]]
            acc[i, ok] += state[j, hp[ok]]; cnt[i, ok] += 1
        out[:, :, :, k] = feat
    out[:, :, :, 6] = np.where(cnt > 0, acc / np.maximum(cnt, 1), state)
    out[:, :, :, 7] = cnt
    return out


FOLDS = {"2021": (2186, 2193, 2557), "2022": (2551, 2558, 2922)}  # last train id, first/last validation id


def topology_constants() -> dict:
    t = yaml.safe_load((ROOT / "sintef_project/data/extended/Tokke_Vinje_topology.yaml").read_text())["model"]["generator"]
    pmax_ref = max(float(t[u]["p_max"]) for u in UNITS)
    return {u: {"pmin": float(t[u]["p_min"]), "pmax": float(t[u]["p_max"]), "start": float(t[u]["startcost_const"]),
                "stop": float(t[u]["stopcost_const"]), "ref": pmax_ref} for u in UNITS}


def startup_fields(block: np.ndarray, columns: list[str], consts: dict) -> np.ndarray:
    """block: (n_hours*14, 115) canonical rows, unit-minor. Returns (n, 16) float32."""
    idx = {c: j for j, c in enumerate(columns)}
    n = len(block); units = np.tile(np.arange(14), n // 14)
    pmin = np.asarray([consts[u]["pmin"] for u in UNITS])[units]; pmax = np.asarray([consts[u]["pmax"] for u in UNITS])[units]
    start = np.asarray([consts[u]["start"] for u in UNITS])[units]; stop = np.asarray([consts[u]["stop"] for u in UNITS])[units]
    ref = consts[UNITS[0]]["ref"]
    inv_scale = block[:, idx["monetary_scale_inverse"]].astype(np.float64)
    lp_margin = block[:, idx["own_lp_economic_margin"]].astype(np.float64)
    m_min = block[:, idx["own_norm_physics_switch_margin_minimum"]].astype(np.float64)
    m_max = block[:, idx["own_norm_physics_switch_margin_maximum"]].astype(np.float64)
    price = block[:, idx["norm_price"]].astype(np.float64)
    def diff(name): return block[:, idx[name]].astype(np.float64) - price
    start_norm = start * inv_scale / pmax; cycle_pmin = (start + stop) * inv_scale / pmin
    eps = 1e-3
    out = np.column_stack([
        pmin / pmax, pmin / ref, pmax / ref, start_norm, cycle_pmin,
        m_min - cycle_pmin, m_max - start_norm,
        lp_margin * pmin / ref, lp_margin * pmax / ref,
        np.clip(start * inv_scale / (pmin * np.maximum(lp_margin, eps)), 0, 168),
        np.clip(start * inv_scale / (pmax * np.maximum(lp_margin, eps)), 0, 168),
        lp_margin + diff("norm_price_next_mean_6h"), lp_margin + diff("norm_price_previous_mean_6h"),
        lp_margin + diff("norm_price_next_mean_24h"), lp_margin + diff("norm_price_previous_mean_24h"),
        lp_margin + diff("norm_price_remaining_mean"),
    ]).astype(np.float32)
    assert out.shape == (n, len(STARTUP_COLUMNS)) and np.isfinite(out).all()
    return out


def build_rows(frame: pd.DataFrame, rows: np.ndarray, spec: dict, consts: dict, block_rows: int = 4096, cc: np.ndarray | None = None) -> np.ndarray:
    parts = []
    for b in range(0, len(rows), block_rows):
        r = rows[b:b + block_rows]
        block = canonical_block(frame, r, spec)
        pieces = [block, startup_fields(block, spec["columns"], consts)]
        if cc is not None:
            pieces.append(cc[r // 168, r % 168].reshape(len(r) * 14, cc.shape[-1]))
        parts.append(np.hstack(pieces))
    return np.vstack(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", choices=["2021", "2022", "final"], required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--selection-lock", type=Path)
    parser.add_argument("--tag", default="startup16")
    parser.add_argument("--seed", type=int, default=4173)
    parser.add_argument("--subsample", type=float, default=1.0)
    parser.add_argument("--volatility-weight", type=float, default=0.0, help="sample weight = 1 + k * norm_price_week_std (per training hour)")
    parser.add_argument("--crosscase", action="store_true", help="append neighbouring-case relaxed LP states for the same calendar hour")
    args = parser.parse_args()
    suffix = "" if args.seed == 4173 else f"_seed{args.seed}"
    out = OUT / f"{args.tag}_{args.year}{suffix}"; out.mkdir(parents=True, exist_ok=False)
    params = dict(PARAMS, random_state=args.seed, subsample=args.subsample, subsample_freq=(1 if args.subsample < 1 else 0))
    tick = time.monotonic()
    feat = OUT / "features664"
    columns = json.loads((feat / "columns.json").read_text())["columns"]
    x = pd.DataFrame(np.load(feat / "features664.npy", mmap_mode="r"), columns=columns)
    labels = np.load(feat / "labels_train.npy")
    cases = pd.read_csv(feat / "cases.csv")
    spec = feature_specification(columns, UNITS); consts = topology_constants()
    run_no = cases["Run No"].to_numpy()
    if args.year == "final":
        sample = np.load(ROOT / "artifacts/lgb-normalized-lp-context-joint-final-001/train_sample_final.npy")
        assert sample.max() < 2922 * 168
        pred_cases = np.where(run_no >= 2923)[0]
    else:
        last_train, first_val, last_val = FOLDS[args.year]
        train_hours = np.arange(last_train * 168)
        rng = np.random.default_rng(4173)
        sample = np.sort(rng.choice(train_hours, size=int(round(0.4 * len(train_hours))), replace=False))
        pred_cases = np.where((run_no >= first_val) & (run_no <= last_val))[0]
    pred_rows = (pred_cases[:, None] * 168 + np.arange(168)).ravel()
    cc = crosscase_table(x, cases) if args.crosscase else None
    extra = STARTUP_COLUMNS + (CROSSCASE_COLUMNS if args.crosscase else [])
    xt = build_rows(x, sample, spec, consts, cc=cc); y = labels[sample].ravel()
    assert len(xt) == len(y) and xt.shape[1] == 115 + len(extra)
    xv = build_rows(x, pred_rows, spec, consts, cc=cc)
    print(json.dumps({"train_rows": int(len(xt)), "pred_rows": int(len(xv)), "features_seconds": round(time.monotonic() - tick, 1)}), flush=True)
    model = lgb.LGBMClassifier(**dict(params, n_jobs=args.threads))
    sample_weight = None
    if args.volatility_weight:
        j = spec["columns"].index("norm_price_week_std")
        sample_weight = (1.0 + args.volatility_weight * xt[:, j].astype(np.float64)).astype(np.float32)
        sample_weight *= len(sample_weight) / sample_weight.sum()
    model.fit(xt, y, sample_weight=sample_weight, categorical_feature=[111, 112])
    model.booster_.save_model(str(out / "model.txt"))
    raw = model.booster_.predict(xv, num_threads=args.threads).reshape(-1, 14)
    np.save(out / "raw_predictions.npy", raw); np.save(out / "sample.npy", sample)
    report = {"year": args.year, "params": params, "columns": spec["columns"] + extra, "train_rows": int(len(xt)),
              "fit_seconds": time.monotonic() - tick, "sample_rule": "exact original final sample" if args.year == "final" else "rng(4173) 40% of training case-hours", "volatility_weight": args.volatility_weight, "crosscase": args.crosscase}
    if args.year != "final":
        fold = load_fold(int(args.year)); assert fold["case_ids"].tolist() == run_no[pred_cases].tolist()
        anchor, truth = fold["anchor"], fold["labels"]
        weights = WEIGHTS if args.selection_lock is None else [0, json.loads(args.selection_lock.read_text())["selected_weight"]]
        scores = {str(w): micro_auc(truth, project_vinje((1 - w) * anchor + w * raw)) for w in weights}
        pick = max(weights, key=lambda w: (scores[str(w)], -w)) if args.selection_lock is None else weights[1]
        report.update(scores=scores, selected_weight=pick, standalone_projected=micro_auc(truth, project_vinje(raw)),
                      eligible=bool(pick > 0 and scores[str(pick)] > scores["0"]))
        (out / f"selection_{args.year}.json").write_text(json.dumps({"selected_weight": pick, "scores": scores}, indent=1))
    (out / "manifest.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "columns"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
