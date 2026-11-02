"""Audit three generators against literal inputs and frozen LP/ML predictions."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sintef_project.features import _lookup
from sintef_project.lp_features import HydroCaseData, POOL_ORDER, UNIT_ORDER, DOWNSTREAM, unit_curve
from sintef_project.physics import POOLS

TARGETS = ["Haukeli_G1", "Lio_G1", "Hogga_G1"]


def digest(path: Path) -> str:
    """Hash an input or output without changing it."""
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def safe_mean(value: np.ndarray) -> float | None:
    """Return a finite mean or None for an empty diagnostic subgroup."""
    return float(np.mean(value)) if np.size(value) else None


def choose_distinct(frame: pd.DataFrame, metric: str, count: int = 3) -> list[int]:
    """Choose largest errors, with deterministic ties and seven-day separation."""
    chosen = []
    ordered = frame.sort_values([metric, "case_id"], ascending=[False, True])
    for row in ordered.itertuples():
        if getattr(row, metric) <= 0:
            continue
        if all(abs(row.case_id - other) >= 7 for other in chosen):
            chosen.append(int(row.case_id))
        if len(chosen) == count:
            break
    return chosen


def residual_main() -> None:
    """Decompose exact pooled AUC errors and case bias in frozen LP-feature models."""
    from scipy.stats import rankdata
    parser = argparse.ArgumentParser()
    parser.add_argument("--residual-only", action="store_true")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--ml-run", type=Path, default=ROOT / "artifacts/lgb-normalized-lp-001")
    parser.add_argument("--years", type=int, nargs="+", default=[2021, 2022])
    args = parser.parse_args()
    out = ROOT / "artifacts" / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = {"run_id": args.run_id, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running", "command": shlex.join([sys.executable, *sys.argv]),
        "hypothesis": "Determine whether remaining exact micro-AUC errors concentrate in a few generators and in case-level commitment state versus within-week switching.",
        "training": "None: descriptive error decomposition of fixed validation predictions", "seed": None,
        "early_stopping": None, "ensemble_weights": None,
        "reuse_bias": "Both years are reused development folds; label-defined subgroups are diagnostics and cannot be inputs at inference. No model or hyperparameter selected by these scores.",
        "formulas": {"pair_error": "For each positive score, count negative scores above plus half ties; for each negative, count positives below plus half ties. Divide each endpoint count by2*Npositive*Nnegative. Unit/case sums partition exactly1-microAUC.",
            "case_bias_brier": "For error e=y-p in one case/unit, mean(e^2)=mean(e)^2+mean((e-mean(e))^2). Sum each term across equal-size cases.",
            "within_week_auc": "Pooled pair-count-weighted AUC computed only within the same case and generator, excluding constant-label cases. This is descriptive and differs from competition microAUC.",
            "constant_case": "All168 labels equal0 or all168 equal1. Mixed cases contain both labels."},
        "environment": {"platform": platform.platform(), "python": sys.version, "cpu_count": os.cpu_count(),
            "hardware": subprocess.run(["system_profiler", "SPHardwareDataType"], capture_output=True, text=True).stdout.split("Serial Number")[0],
            "packages": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn"]}},
        "inputs": {}, "source": {}, "folds": {}, "results": {},
        "kaggle": {"submission_id": None, "score": None, "rank": None, "observation_time": None}}
    try:
        source = Path(__file__)
        shutil.copy2(source, out / source.name)
        manifest["source"][str(source.relative_to(ROOT))] = digest(source)
        origin_path = args.ml_run / "manifest.json"
        manifest["inputs"][str(origin_path)] = digest(origin_path)
        shutil.copy2(origin_path, out / "model_origin_manifest.json")
        origin = json.loads(origin_path.read_text())
        manifest.update(data=origin["data"], model_parameters=origin["model_parameters"],
                        training_arguments=origin["arguments"], feature_spec=origin["feature_spec"],
                        inherited_source_hashes=origin["source"])
        rows, case_rows = [], []
        for year in args.years:
            path = args.ml_run / f"validation_{year}.npz"
            manifest["inputs"][str(path)] = digest(path)
            with np.load(path) as z:
                y, p, units, ids = z["labels"], z["predictions"], z["units"].tolist(), z["case_ids"]
            if not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
                raise ValueError("Invalid validation probabilities")
            for unit in units:
                checkpoint = args.ml_run / f"fold{year}_{unit}.txt"
                manifest["inputs"][str(checkpoint)] = digest(checkpoint)
            manifest["folds"][str(year)] = origin["folds"][str(year)]
            assert ids.tolist() == origin["folds"][str(year)]["validation_ids"]
            positive, negative = np.sort(p[y == 1]), np.sort(p[y == 0])
            pairs = len(positive) * len(negative)
            losses = np.empty(p.shape, dtype=np.float64)
            values = p[y == 1]
            losses[y == 1] = len(negative) - .5 * (np.searchsorted(negative, values, side="left") + np.searchsorted(negative, values, side="right"))
            values = p[y == 0]
            losses[y == 0] = .5 * (np.searchsorted(positive, values, side="left") + np.searchsorted(positive, values, side="right"))
            losses /= 2 * pairs
            micro_auc = float(roc_auc_score(y.ravel(), p.ravel()))
            np.testing.assert_allclose(losses.sum(), 1 - micro_auc, atol=1e-13, rtol=0)
            yc, pc = y.reshape(len(ids), 168, len(units)), p.reshape(len(ids), 168, len(units)).astype(np.float64)
            error = yc - pc
            bias = error.mean(axis=1)
            brier = (error ** 2).mean(axis=1)
            centered = ((error - bias[:, None, :]) ** 2).mean(axis=1)
            np.testing.assert_allclose(brier, bias ** 2 + centered, atol=1e-12, rtol=0)
            positive_count = yc.sum(axis=1)
            constant = (positive_count == 0) | (positive_count == 168)
            local_pairs = positive_count * (168 - positive_count)
            ranks = rankdata(pc, axis=1, method="average")
            local_good_pairs = (ranks * yc).sum(axis=1) - positive_count * (positive_count + 1) / 2
            per_case_loss = losses.reshape(len(ids), 168, len(units)).sum(axis=1)
            errors = ((pc >= .5) != yc).sum(axis=1)
            for j, unit in enumerate(units):
                mask = constant[:, j]
                unit_loss = float(losses[:, j].sum())
                row = {"year": year, "unit": unit, "unit_auc": float(roc_auc_score(y[:, j], p[:, j])),
                    "micro_auc_error_contribution": unit_loss, "fraction_all_micro_error": unit_loss / (1 - micro_auc),
                    "constant_cases": int(mask.sum()), "all_off_cases": int((positive_count[:, j] == 0).sum()), "all_on_cases": int((positive_count[:, j] == 168).sum()),
                    "constant_case_micro_error_fraction": float(per_case_loss[mask, j].sum() / unit_loss),
                    "mean_case_fraction_abs_error": float(np.abs(bias[:, j]).mean()),
                    "brier": float(brier[:, j].mean()), "brier_case_bias_fraction": float((bias[:, j] ** 2).sum() / brier[:, j].sum()),
                    "within_week_auc_pair_weighted": float(local_good_pairs[:, j].sum() / local_pairs[:, j].sum()),
                    "classification_errors": int(errors[:, j].sum()), "constant_case_classification_error_fraction": float(errors[mask, j].sum() / errors[:, j].sum()),
                    "all_on_predicted_mean_below_half_cases": int(((positive_count[:, j] == 168) & (pc[:, :, j].mean(axis=1) < .5)).sum()),
                    "all_off_predicted_mean_above_half_cases": int(((positive_count[:, j] == 0) & (pc[:, :, j].mean(axis=1) > .5)).sum())}
                rows.append(row)
                for i, case_id in enumerate(ids):
                    case_rows.append({"year": year, "unit": unit, "case_id": int(case_id), "truth_fraction": float(yc[i, :, j].mean()),
                        "predicted_fraction": float(pc[i, :, j].mean()), "case_bias": float(bias[i, j]), "brier": float(brier[i, j]),
                        "centered_brier": float(centered[i, j]), "constant_case": bool(mask[i]), "micro_error_contribution": float(per_case_loss[i, j]),
                        "classification_errors": int(errors[i, j])})
            selected_rows = [r for r in rows if r["year"] == year]
            contribution = sum(r["fraction_all_micro_error"] for r in selected_rows if r["unit"] in TARGETS)
            manifest["results"][str(year)] = {"micro_auc": micro_auc, "total_pair_error": float(losses.sum()),
                "target_units_micro_error_fraction": contribution, "units": {row["unit"]: row for row in selected_rows}}
            print("RESIDUAL", year, "micro_auc", micro_auc, "HaukeliLioHogga_error_fraction", contribution, flush=True)
        pd.DataFrame(rows).to_csv(out / "unit_residuals.csv", index=False)
        pd.DataFrame(case_rows).to_csv(out / "case_residuals.csv", index=False)
        manifest["status"] = "complete"
    except BaseException:
        manifest["status"] = "failed"
        manifest["error"] = traceback.format_exc()
        raise
    finally:
        manifest["runtime_seconds"] = time.monotonic() - started
        manifest["artifacts"] = {str(path.relative_to(out)): digest(path) for path in sorted(out.rglob("*")) if path.is_file() and path.name != "manifest.json"}
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    """Analyze existing validation predictions; perform no fitting or LP solves."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="lp-error-analysis-001")
    parser.add_argument("--years", type=int, nargs="+", default=[2021, 2022])
    parser.add_argument("--cache", type=Path, default=ROOT / "artifacts/lp-full-001")
    parser.add_argument("--lp-run", type=Path, default=ROOT / "artifacts/lp-diagnostic-002")
    parser.add_argument("--ml-run", type=Path, default=ROOT / "artifacts/lgb-normalized-context-001")
    args = parser.parse_args()
    out = ROOT / "artifacts" / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = {
        "run_id": args.run_id, "timestamp_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
        "command": shlex.join([sys.executable, *sys.argv]), "hypothesis": "Trace Haukeli/Lio/Hogga errors to literal case inputs, directional storage aggregation, and fixed-head energy conversion before changing any model.",
        "analysis_only": True, "seed": None, "training": "none", "early_stopping": "none", "ensemble_weights": None,
        "targets": TARGETS, "years": args.years, "selection": "For each year/unit/model and FP/FN direction, choose largest incorrect probability mass per case, ties by Run No, at most3 examples separated by7 case-start days. All requested validation cases are summarized before case selection. LP fixed p=.5 means q=Qmin and is diagnostic, not calibrated probability.",
        "reuse_bias": "2021 and2022 are reused development folds. All case selections and associations are exploratory. They establish no new validation improvement or causal conclusion.",
        "feature_formulas": {
            "direct_intake_release_lower_bound_mm3": "max(sum168h(natural inflow into actual intake reservoir)*.0036 - max(intake_capacity-initial_intake_volume,0),0). This is required outgoing water, not necessarily turbine release; no upstream inflow assumed.",
            "price_margin": "literal hourly price minus literal upstream terminal WV at start+7days; normalized by1+mean(abs(case price)) only for comparison",
            "opportunity_cost": "(LP water shadow of source pool minus downstream pool shadow)/same fixed initial-head best-point energy factor used by LP",
            "false_negative_mass": "mean(y*(1-p))", "false_positive_mass": "mean((1-y)*p)",
            "physical_energy": "Recreate frozen unit_curve from legal initial volume and YAML; no solver and no future volume observations",
        },
        "environment": {"platform": platform.platform(), "python": sys.version, "cpu_count": os.cpu_count(),
            "hardware": subprocess.run(["system_profiler", "SPHardwareDataType"], capture_output=True, text=True).stdout.split("Serial Number")[0],
            "packages": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn", "pyyaml"]}},
        "data": {}, "source": {}, "inputs": {}, "folds": {}, "results": {},
        "kaggle": {"submission_id": None, "score": None, "rank": None, "observation_time": None},
        "references": ["https://docs.shop.sintef.energy/examples/pq_curves/pq_curves.html", "https://docs.shop.sintef.energy/objects/river/river.html", "https://docs.shop.sintef.energy/objects/plant/plant.html"],
        "preflight_failure": "An interactive read-only probe constructed nanosecond timestamps from pandas3 microsecond .asi8 values and failed before computing a diagnostic (missing1970 inflow timestamps). Corrected explicitly with DatetimeIndex.as_unit('ns'). No inputs, models, cache or predictions were modified.",
    }
    def record(path: Path, category: str = "inputs") -> None:
        relative = str(path.resolve().relative_to(ROOT))
        manifest[category][relative] = digest(path)
    def persist() -> None:
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        for relative in ["scripts/lp_error_analysis.py", "sintef_project/lp_features.py", "sintef_project/features.py", "sintef_project/physics.py", "docs/lp_features.md", "docs/physics_features.md"]:
            path = ROOT / relative
            record(path, "source")
            dest = out / "source" / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
        for path in sorted((ROOT / "sintef_project").rglob("*")):
            if path.suffix in [".csv", ".yaml"]:
                record(path, "data")
        origins = {}
        for name, directory in [("cache", args.cache), ("lp", args.lp_run), ("ml", args.ml_run)]:
            path = directory / "manifest.json"
            record(path)
            origins[name] = json.loads(path.read_text())
            shutil.copy2(path, out / f"{name}_manifest.json")
        manifest["lp_math"] = origins["cache"]["math"]
        manifest["ml_model_parameters"] = origins["ml"]["model_parameters"]
        manifest["ml_training_arguments"] = origins["ml"]["arguments"]
        for category in ["source", "data"]:
            for relative, expected in origins["cache"][category].items():
                if digest(ROOT / relative) != expected:
                    raise ValueError(f"Frozen LP {category} changed: {relative}")
        cache_path = args.cache / "features.npz"
        record(cache_path)
        if digest(cache_path) != origins["cache"]["artifacts"]["features.npz"]:
            raise ValueError("LP cache differs from frozen manifest")
        with np.load(cache_path) as stored:
            feature_names, cache_ids = stored["columns"].tolist(), stored["case_ids"]
            cache_features = stored["features"]
        data = HydroCaseData(ROOT)
        manifest["topology_audit"] = {
            "Haukeli_pools": {r: data.model["reservoir"][r] for r in POOLS["Haukeli"]},
            "Haukeli_river": data.model["river"]["r_Langeidvatn"],
            "Haukeli_connections": [e for e in data.topology["connections"] if e["from"] in ["Vatjern", "Langeidvatn", "Haukeli", "r_Langeidvatn"] or e["to"] in ["Vatjern", "Langeidvatn", "Haukeli"]],
            "constants": {u: {"p_min_mw": data.model["generator"][u]["p_min"], "p_max_mw": data.model["generator"][u]["p_max"], "start_cost": data.model["generator"][u]["startcost_const"], "stop_cost": data.model["generator"][u]["stopcost_const"]} for u in TARGETS},
        }
        persist()
        rows, hourly_by_year, curve_cache = [], {}, {}
        for year in args.years:
            for directory in [args.lp_run, args.ml_run]:
                record(directory / f"validation_{year}.npz")
            with np.load(args.lp_run / f"validation_{year}.npz") as lp, np.load(args.ml_run / f"validation_{year}.npz") as ml:
                for key in ["case_ids", "labels", "units"]:
                    np.testing.assert_array_equal(lp[key], ml[key])
                units, ids = lp["units"].tolist(), lp["case_ids"]
                truth = lp["labels"].reshape(-1, 168, len(units))
                lp_prediction = lp["predictions"].reshape(truth.shape)
                ml_prediction = ml["predictions"].reshape(truth.shape)
            assert units == UNIT_ORDER
            cases = data.cases.set_index("Run No").loc[ids]
            starts = pd.DatetimeIndex(cases.starttime)
            times = pd.DatetimeIndex((starts.as_unit("ns").asi8[:, None] + np.arange(168)[None, :] * 3600_000_000_000).ravel(), tz="UTC")
            prices = _lookup(data.price, times, "audit price").reshape(len(ids), 168)
            inflows = _lookup(data.inflow, times, "audit inflow").reshape(len(ids), 168, -1)
            volume = data.volume.loc[starts]
            water = data.water.loc[starts + pd.Timedelta(days=7)]
            cache_positions = pd.Index(cache_ids).get_indexer(ids)
            assert (cache_positions >= 0).all()
            features = cache_features[cache_positions]
            ratios = features[:, :, [feature_names.index(f"lp_qmin_ratio_{unit}") for unit in units]]
            np.testing.assert_array_equal(ratios / (1 + ratios), lp_prediction)
            manifest["folds"][str(year)] = {"case_ids": ids.tolist(), "dates": [str(starts.min()), str(starts.max())], "ml_training": origins["ml"]["folds"][str(year)], "hours_per_case": 168}
            storage = np.empty((len(ids), len(POOL_ORDER), 168))
            flow = np.empty((len(ids), len(units), 168))
            power = np.empty_like(flow)
            shadows = np.empty_like(storage)
            minimum_slack = np.empty_like(storage)
            for i, case_id in enumerate(ids):
                path = args.cache / f"case_{case_id:04}.npz"
                record(path)
                if digest(path) != origins["cache"]["artifacts"][path.name]:
                    raise ValueError(f"Changed LP case: {case_id}")
                with np.load(path) as result:
                    storage[i], flow[i], power[i], shadows[i], minimum_slack[i] = [result[key] for key in ["storage", "flow", "power", "water_shadow", "minimum_slack"]]
            target_hours = []
            for unit in TARGETS:
                j, plant = units.index(unit), unit.rsplit("_G", 1)[0]
                pi, pool = POOL_ORDER.index(plant), POOLS[plant]
                intake = pool[0]
                down = DOWNSTREAM[plant]
                down_i = POOL_ORDER.index(down) if down else None
                initial = volume[pool].sum(axis=1).to_numpy()
                capacity = sum(data.model["reservoir"][r]["max_vol"] for r in pool)
                intake_initial = volume[intake].to_numpy()
                intake_capacity = data.model["reservoir"][intake]["max_vol"]
                local_flow = inflows[:, :, data.inflow.columns.get_loc(intake)]
                intake_week = local_flow.sum(axis=1) * .0036
                release_lb = np.maximum(intake_week - np.maximum(intake_capacity - intake_initial, 0), 0)
                water_value = water[intake].to_numpy()
                raw_margin = prices - water_value[:, None]
                normalized_margin = raw_margin / (1 + np.abs(prices).mean(axis=1))[:, None]
                heads, energies, qmins, qmaxs = [], [], [], []
                for i in range(len(ids)):
                    head = float(np.interp(intake_initial[i], data.model["reservoir"][intake]["vol_head"]["x"], data.model["reservoir"][intake]["vol_head"]["y"]))
                    outlet = float(data.model["plant"][plant]["outlet_line"])
                    downstream_res = data.plant_downstream_reservoir.get(plant)
                    if downstream_res:
                        curve = data.model["reservoir"][downstream_res]["vol_head"]
                        outlet = max(outlet, float(np.interp(volume.iloc[i][downstream_res], curve["x"], curve["y"])))
                    head = max(head - outlet, 1)
                    key = (unit, head)
                    if key not in curve_cache:
                        curve_cache[key] = unit_curve(data.model["generator"][unit], data.model["plant"][plant], head)
                    curve = curve_cache[key]
                    heads.append(head); energies.append(curve["best_energy"]); qmins.append(curve["qmin"]); qmaxs.append(curve["qmax"])
                energy, qmin, qmax = map(np.asarray, [energies, qmins, qmaxs])
                opportunity = (shadows[:, pi] - (shadows[:, down_i] if down_i is not None else 0)) / energy[:, None]
                np.testing.assert_allclose(flow[:, j] / qmin[:, None], ratios[:, :, j], rtol=2e-6, atol=1e-6)
                np.testing.assert_allclose((prices - opportunity) / (1 + np.abs(prices).mean(axis=1))[:, None], features[:, :, feature_names.index(f"lp_economic_margin_{unit}")], rtol=2e-5, atol=1e-6)
                y, p, m = truth[:, :, j], lp_prediction[:, :, j], ml_prediction[:, :, j]
                groups = {"lp_true_positive": (y == 1) & (p >= .5), "lp_false_negative_zero": (y == 1) & (flow[:, j] <= 1e-10), "lp_false_negative_fractional": (y == 1) & (flow[:, j] > 1e-10) & (p < .5), "lp_false_positive": (y == 0) & (p >= .5), "lp_true_negative": (y == 0) & (p < .5)}
                group_stats = {}
                for name, mask in groups.items():
                    group_stats[name] = {"hours": int(mask.sum()), "price_mean": safe_mean(prices[mask]), "price_minus_local_WV_mean": safe_mean(raw_margin[mask]), "price_below_local_WV_fraction": safe_mean((raw_margin < 0)[mask]), "local_inflow_m3s_mean": safe_mean(local_flow[mask]), "case_direct_release_lb_positive_fraction": safe_mean(np.broadcast_to(release_lb[:, None] > 0, mask.shape)[mask]), "initial_pool_fill_mean": safe_mean(np.broadcast_to((initial / capacity)[:, None], mask.shape)[mask]), "lp_opportunity_cost_mean": safe_mean(opportunity[mask]), "ml_prediction_mean": safe_mean(m[mask])}
                result = {"lp_auc": float(roc_auc_score(y.ravel(), p.ravel())), "ml_auc": float(roc_auc_score(y.ravel(), m.ravel())), "on_hours": int(y.sum()), "lp_zero_hours": int((flow[:, j] <= 1e-10).sum()), "groups": group_stats, "direct_intake_release_positive_cases": int((release_lb > 0).sum()), "initial_head_range": [min(heads), max(heads)], "best_energy_range": [float(energy.min()), float(energy.max())], "qmin_m3s_range": [float(qmin.min() / .0036), float(qmin.max() / .0036)], "qmax_m3s_range": [float(qmax.min() / .0036), float(qmax.max() / .0036)]}
                manifest["results"].setdefault(str(year), {})[unit] = result
                for i, case_id in enumerate(ids):
                    row = {"year": year, "unit": unit, "case_id": int(case_id), "start": str(starts[i]), "truth_on_hours": int(y[i].sum()), "lp_on_hours": int((p[i] >= .5).sum()), "ml_on_hours": int((m[i] >= .5).sum()), "lp_fn_hours": int(((y[i] == 1) & (p[i] < .5)).sum()), "lp_fp_hours": int(((y[i] == 0) & (p[i] >= .5)).sum()), "ml_fn_hours": int(((y[i] == 1) & (m[i] < .5)).sum()), "ml_fp_hours": int(((y[i] == 0) & (m[i] >= .5)).sum()), "lp_fn_mass": float((y[i] * (1 - p[i])).mean()), "lp_fp_mass": float(((1 - y[i]) * p[i]).mean()), "ml_fn_mass": float((y[i] * (1 - m[i])).mean()), "ml_fp_mass": float(((1 - y[i]) * m[i]).mean()), "price_min": float(prices[i].min()), "price_mean": float(prices[i].mean()), "price_max": float(prices[i].max()), "local_water_value": float(water_value[i]), "lp_opportunity_mean": float(opportunity[i].mean()), "initial_pool_mm3": float(initial[i]), "pool_capacity_mm3": float(capacity), "initial_intake_mm3": float(intake_initial[i]), "intake_capacity_mm3": float(intake_capacity), "intake_week_inflow_mm3": float(intake_week[i]), "intake_release_lower_bound_mm3": float(release_lb[i]), "gross_head_m": float(heads[i]), "best_energy_mwh_per_mm3": float(energy[i]), "qmin_m3s": float(qmin[i] / .0036), "qmax_m3s": float(qmax[i] / .0036), "lp_discharge_mm3": float(flow[i, j].sum()), "lp_final_pool_mm3": float(storage[i, pi, -1]), "lp_minimum_slack_max_mm3": float(minimum_slack[i, pi].max())}
                    for reservoir in pool:
                        row[f"initial_{reservoir}_mm3"] = float(volume.iloc[i][reservoir])
                    rows.append(row)
                    target_hours.append(pd.DataFrame({"year": year, "case_id": int(case_id), "start": str(starts[i]), "unit": unit, "hour": np.arange(168), "truth": y[i], "lp_proxy": p[i], "ml_probability": m[i], "price": prices[i], "local_water_value": water_value[i], "local_price_margin": raw_margin[i], "local_price_margin_normalized": normalized_margin[i], "lp_opportunity_cost": opportunity[i], "lp_q_m3s": flow[i, j] / .0036, "qmin_m3s": qmin[i] / .0036, "lp_power_mw": power[i, j], "lp_storage_pool_mm3": storage[i, pi], "direct_intake_inflow_m3s": local_flow[i], "intake_release_lower_bound_mm3": release_lb[i]}))
                print(year, unit, "lp_auc", result["lp_auc"], "ml_auc", result["ml_auc"], "zero_FN", group_stats["lp_false_negative_zero"]["hours"], flush=True)
            hourly_by_year[year] = pd.concat(target_hours, ignore_index=True)
            persist()
        summary = pd.DataFrame(rows)
        summary.to_csv(out / "all_case_summary.csv", index=False)
        selections = []
        for year in args.years:
            for unit in TARGETS:
                subset = summary[(summary.year == year) & (summary.unit == unit)]
                for model in ["lp", "ml"]:
                    for direction in ["fn", "fp"]:
                        metric = f"{model}_{direction}_mass"
                        for case_id in choose_distinct(subset, metric):
                            selections.append({"year": year, "unit": unit, "model": model, "direction": direction, "case_id": case_id, "selection_metric": metric})
        selected = pd.DataFrame(selections).merge(summary, on=["year", "unit", "case_id"], validate="many_to_one")
        selected.to_csv(out / "selected_cases.csv", index=False)
        selected_keys = selected[["year", "unit", "case_id"]].drop_duplicates()
        all_hours = pd.concat(list(hourly_by_year.values()), ignore_index=True)
        all_hours.merge(selected_keys, on=["year", "unit", "case_id"], validate="many_to_one").to_csv(out / "selected_hourly.csv", index=False)
        manifest["selected_cases"] = selections
        manifest["status"] = "complete"
    except BaseException:
        manifest["status"] = "failed"
        manifest["error"] = traceback.format_exc()
        raise
    finally:
        manifest["runtime_seconds"] = time.monotonic() - started
        manifest["artifacts"] = {str(path.relative_to(out)): digest(path) for path in sorted(out.rglob("*")) if path.is_file() and path.name != "manifest.json"}
        persist()


if __name__ == "__main__":
    residual_main() if "--residual-only" in sys.argv else main()
