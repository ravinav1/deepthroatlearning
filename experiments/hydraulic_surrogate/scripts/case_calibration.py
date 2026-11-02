"""Purged case-fraction regression and independent hourly probability adjustment."""

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

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score

from sintef_project.features import build_features


MODEL_PARAMETERS = {
    "objective": "regression", "n_estimators": 200, "num_leaves": 15,
    "min_child_samples": 20, "learning_rate": 0.04, "random_state": 4173,
    "n_jobs": 2, "deterministic": True, "force_col_wise": True, "verbosity": -1,
}
CANDIDATES = [{"scope": "global", "weight": 0.0}] + [
    {"scope": scope, "weight": weight}
    for scope in ("global", "weak_units") for weight in (0.1, 0.25, 0.5)
]
WEAK_UNITS = ["Lio_G1", "Haukeli_G1"]


def checksum(path: Path) -> str:
    """Return a file SHA-256 digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def case_feature_table(root: Path) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, pd.DataFrame, dict]:
    """Collapse legal horizon inputs, with no commitment-derived predictors."""
    x, labels, train, mapping, metadata = build_features(root)
    n_cases = len(train) + len(mapping)
    case_x = x.iloc[::168].reset_index(drop=True).copy(deep=True)
    additions = {f"case_price_hour_{hour}": x["price"].to_numpy().reshape(n_cases, 168)[:, hour].copy()
                 for hour in range(168)}
    for column in x:
        if column.startswith("inflow_") and not column.startswith(("inflow_week_", "inflow_elapsed_", "inflow_remaining_")):
            matrix = x[column].to_numpy().reshape(n_cases, 168)
            additions[f"case_max_{column}"] = matrix.max(axis=1)
            additions[f"case_std_{column}"] = matrix.std(axis=1)
    case_x = pd.concat([case_x, pd.DataFrame(additions)], axis=1)
    case_y = np.column_stack([labels[unit].reshape(len(train), 168).mean(axis=1)
                              for unit in metadata["units"]]).astype(np.float32)
    metadata["case_feature_spec"] = {
        "first_hour": "All 231 base features at horizon hour0, including initial storage, terminal water value at start+7d, and complete-horizon price/inflow statistics",
        "hourly_prices": "Full 168-hour price vector in chronological order",
        "inflow_extras": "Maximum and population standard deviation of each of18 natural-inflow series over168 hours",
        "target": "Arithmetic mean of168 binary labels for each generator, computed only from retained training cases",
    }
    return case_x, case_y, train, mapping, metadata


def adjust_case_fraction(probability: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shift each case-unit's hourly logits until mean probability equals target."""
    probability = np.asarray(probability, dtype=np.float64)
    target = np.clip(np.asarray(target, dtype=np.float64), 1e-5, 1 - 1e-5)
    logits = logit(np.clip(probability, 1e-6, 1 - 1e-6))
    low = np.full(target.shape, -40.0)
    high = np.full(target.shape, 40.0)
    for _ in range(48):
        middle = (low + high) / 2
        current = expit(logits + middle[:, None, :]).mean(axis=1)
        low = np.where(current < target, middle, low)
        high = np.where(current >= target, middle, high)
    shifts = (low + high) / 2
    adjusted = expit(logits + shifts[:, None, :])
    if not np.allclose(adjusted.mean(axis=1), target, atol=1e-6):
        raise ValueError("Case mean adjustment did not converge")
    return adjusted.astype(np.float32), shifts.astype(np.float32)


def blend(original: np.ndarray, adjusted: np.ndarray, candidate: dict, units: list[str]) -> np.ndarray:
    """Apply one predeclared blend globally or to the two predeclared weak units."""
    weights = np.full(len(units), candidate["weight"], dtype=np.float32)
    if candidate["scope"] == "weak_units":
        weights = np.array([candidate["weight"] if unit in WEAK_UNITS else 0 for unit in units], dtype=np.float32)
    return original * (1 - weights) + adjusted * weights


def metrics(labels: np.ndarray, prediction: np.ndarray, units: list[str]) -> dict:
    """Return pooled AUC and unit-wise AUC with fraction mean absolute errors."""
    return {
        "micro_auc": float(roc_auc_score(labels.ravel(), prediction.ravel())),
        "unit_auc": {unit: float(roc_auc_score(labels[:, :, j].ravel(), prediction[:, :, j].ravel())) for j, unit in enumerate(units)},
        "case_fraction_mae": {unit: float(np.abs(labels[:, :, j].mean(axis=1) - prediction[:, :, j].mean(axis=1)).mean()) for j, unit in enumerate(units)},
    }


def main() -> None:
    """Select on2021, freeze the decision, then independently evaluate2022."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="case-calibration-001")
    parser.add_argument("--base-run", type=Path, default=ROOT / "artifacts/lgb-context-001")
    args = parser.parse_args()
    out = ROOT / "artifacts" / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    manifest = {
        "run_id": args.run_id, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join([sys.executable, *sys.argv]), "status": "running",
        "hypothesis": "A case-level commitment-fraction model can correct level errors in hourly forecasts while retaining within-case ranking",
        "model_parameters": MODEL_PARAMETERS,
        "effective_model_parameters": lgb.LGBMRegressor(**MODEL_PARAMETERS).get_params(),
        "early_stopping": "None; fixed200 rounds", "preprocessing": "No learned transforms; float32 features; unscaled; no labels in features",
        "candidates_predeclared": CANDIDATES, "weak_units_predeclared": WEAK_UNITS,
        "selection": "Highest2021 microAUC, ties favor first candidate (zero-adjustment baseline); evaluate single frozen candidate on2022",
        "adjustment": "Clip original hourly p to[1e-6,1-1e-6], regress target mean clipped[1e-5,1-1e-5], solve scalar case-unit logit offset by48 bisections over[-40,40]; blend probabilities (1-w)*original+w*adjusted",
        "environment": {"platform": platform.platform(), "python": sys.version, "cpu_count": os.cpu_count(),
                        "hardware": subprocess.run(["system_profiler", "SPHardwareDataType"], capture_output=True, text=True).stdout.split("Serial Number")[0],
                        "packages": {package: importlib.metadata.version(package) for package in ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm")}},
        "data": {}, "source": {}, "folds": {},
        "kaggle": {"submission_id": None, "score": None, "rank": None, "observation_time": None},
    }
    for path in sorted((ROOT / "sintef_project").rglob("*")):
        if path.suffix in (".csv", ".yaml"):
            manifest["data"][str(path.relative_to(ROOT))] = checksum(path)
    for path in (Path(__file__), ROOT / "sintef_project/features.py"):
        relative = path.relative_to(ROOT)
        manifest["source"][str(relative)] = checksum(path)
        destination = out / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    manifest["base_manifest_path"] = str(args.base_run / "manifest.json")
    manifest["base_manifest_sha256"] = checksum(args.base_run / "manifest.json")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    try:
        x, case_y, train, mapping, metadata = case_feature_table(ROOT)
        units = metadata["units"]
        manifest.update(feature_names=list(x), shape=list(x.shape), base_feature_spec=metadata["feature_spec"], case_feature_spec=metadata["case_feature_spec"], units=units)
        print("FEATURES", x.shape, flush=True)
        for year in (2021, 2022):
            begin = pd.Timestamp(f"{year}-01-01", tz="UTC")
            end = pd.Timestamp(f"{year + 1}-01-01", tz="UTC")
            retained = np.flatnonzero(train.starttime < begin - pd.Timedelta(days=6))
            validation = np.flatnonzero((train.starttime >= begin) & (train.starttime < end))
            assert train.starttime.iloc[retained].max() + pd.Timedelta(hours=167) < begin
            targets = np.empty((len(validation), len(units)), dtype=np.float32)
            for j, unit in enumerate(units):
                model = lgb.LGBMRegressor(**MODEL_PARAMETERS)
                model.fit(x.iloc[retained], case_y[retained, j])
                targets[:, j] = model.predict(x.iloc[validation])
                model.booster_.save_model(str(out / f"fold{year}_{unit}.txt"))
            base_file = args.base_run / f"validation_{year}.npz"
            base = np.load(base_file)
            if list(base["units"]) != units or not np.array_equal(base["case_ids"], train.iloc[validation]["Run No"].to_numpy()):
                raise ValueError("Base prediction case/unit order mismatch")
            original = base["predictions"].reshape(len(validation), 168, len(units))
            truth = base["labels"].reshape(len(validation), 168, len(units))
            np.testing.assert_allclose(truth.mean(axis=1), case_y[validation], atol=1e-7)
            adjusted, shifts = adjust_case_fraction(original, targets)
            fold = {
                "train_ids": train.iloc[retained]["Run No"].tolist(), "validation_ids": train.iloc[validation]["Run No"].tolist(),
                "train_dates": [str(train.starttime.iloc[retained].min()), str(train.starttime.iloc[retained].max())],
                "validation_dates": [str(train.starttime.iloc[validation].min()), str(train.starttime.iloc[validation].max())],
                "purged_case_count": 6, "base_prediction_path": str(base_file), "base_prediction_sha256": checksum(base_file),
                "baseline": metrics(truth, original, units),
                "regressor_fraction_mae": {unit: float(np.abs(case_y[validation, j] - np.clip(targets[:, j], 1e-5, 1 - 1e-5)).mean()) for j, unit in enumerate(units)},
            }
            if year == 2021:
                candidate_results = []
                for candidate in CANDIDATES:
                    result = blend(original, adjusted, candidate, units)
                    candidate_results.append({**candidate, "micro_auc": float(roc_auc_score(truth.ravel(), result.ravel()))})
                best_index = max(range(len(candidate_results)), key=lambda index: candidate_results[index]["micro_auc"])
                manifest["frozen_selection"] = CANDIDATES[best_index]
                manifest["selection_frozen_utc"] = datetime.now(timezone.utc).isoformat()
                fold["candidate_results"] = candidate_results
                print("CANDIDATES", json.dumps(candidate_results), flush=True)
                # Persist the frozen selection before opening2022 predictions.
                manifest["folds"][str(year)] = fold
                (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
            result = blend(original, adjusted, manifest["frozen_selection"], units)
            fold["selected"] = metrics(truth, result, units)
            fold["delta_auc"] = fold["selected"]["micro_auc"] - fold["baseline"]["micro_auc"]
            manifest["folds"][str(year)] = fold
            np.savez_compressed(out / f"validation_{year}.npz", predictions=result.reshape(-1, len(units)), labels=truth.reshape(-1, len(units)), case_ids=base["case_ids"], units=units, target_fractions=targets, logit_shifts=shifts)
            print("FOLD", year, "baseline", fold["baseline"]["micro_auc"], "selected", fold["selected"]["micro_auc"], "delta", fold["delta_auc"], flush=True)
            (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        manifest["passes_independent_holdout"] = manifest["folds"]["2022"]["delta_auc"] > 0
        if manifest["passes_independent_holdout"] and manifest["frozen_selection"]["weight"] > 0:
            final_targets = np.empty((len(mapping), len(units)), dtype=np.float32)
            for j, unit in enumerate(units):
                model = lgb.LGBMRegressor(**MODEL_PARAMETERS)
                model.fit(x.iloc[:len(train)], case_y[:, j])
                final_targets[:, j] = model.predict(x.iloc[len(train):])
                model.booster_.save_model(str(out / f"final_{unit}.txt"))
            np.savez_compressed(out / "test_fraction_targets.npz", targets=final_targets, case_ids=mapping["Run No"].to_numpy(), units=units)
            manifest["final_fit"] = "All supplied training cases IDs1..2922; target predictions for test IDs2923..3647. Apply frozen adjustment only to validated base-model family."
        manifest["status"] = "complete"
    except BaseException:
        manifest["status"] = "failed"
        manifest["error"] = traceback.format_exc()
        raise
    finally:
        manifest["runtime_seconds"] = time.time() - started
        manifest["artifacts"] = {str(path.relative_to(out)): checksum(path) for path in sorted(out.rglob("*")) if path.is_file() and path.name != "manifest.json"}
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        report = ["# Case-fraction calibration experiment", "", f"Run `{args.run_id}`, {manifest['timestamp_utc']}. Status: {manifest['status']}.", "", manifest["hypothesis"] + ".", "", f"Command: `{manifest['command']}`", "", f"Runtime: {manifest['runtime_seconds']:.2f}s. Exact source/data/artifact hashes, environment, all model parameters, input feature definitions, row IDs and base prediction provenance: `artifacts/{args.run_id}/manifest.json`.", "", "The model regresses each generator's weekly on-fraction from231 initial-row base features, the full168-hour price vector, and36 inflow maximum/standard-deviation features. It has fixed200 rounds,15 leaves, minimum20 cases/leaf, learning rate0.04, seed4173 and2 threads. No early stopping or validation fitting. Chronological case starts use a six-day purge.", "", "Predeclared selection: choose one global blend or a blend limited to Lio/Haukeli, weights0,0.1,0.25,0.5, by2021 micro AUC. Freeze before loading2022 predictions. The adjustment shifts each case-unit's hourly logits to match the independent regressor's fraction, then blends probabilities.", ""]
        for year, fold in manifest["folds"].items():
            if "selected" not in fold:
                continue
            report.append(f"- {year}: baseline {fold['baseline']['micro_auc']:.9f}; frozen-selection result {fold['selected']['micro_auc']:.9f}; delta {fold['delta_auc']:+.9f}. Train {fold['train_dates']}; validation {fold['validation_dates']}.")
            if "candidate_results" in fold:
                report.extend(["", "|2021 scope|Weight|Micro AUC|", "|---|---:|---:|"] + [f"|{result['scope']}|{result['weight']}|{result['micro_auc']:.9f}|" for result in fold["candidate_results"]] + [""])
        report += ["", f"Frozen choice: `{manifest.get('frozen_selection')}`. Positive independent2022 improvement: `{manifest.get('passes_independent_holdout')}`.", "", "Per-unit AUC/fraction errors and frozen adjusted validation predictions are saved in the artifact directory. Models trained on all provided labels and test fraction targets are saved only if the selected adjustment improves2022. This adjustment has only been tested against the specified base model; applying it to a stronger hourly model requires separate validation.", "", "Kaggle submission ID, leaderboard score, rank and observation time: none; this is local validation only."]
        if "error" in manifest:
            report += ["", "Failure:", "```", manifest["error"], "```"]
        (ROOT / "docs/case_calibration.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
