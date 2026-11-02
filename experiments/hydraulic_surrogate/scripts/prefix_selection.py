"""Select frozen per-generator LightGBM prefixes on 2021, evaluate 2022 once."""
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
from sklearn.metrics import log_loss, roc_auc_score

from sintef_project.features import build_features
from sintef_project.physics import add_physics_features
from sintef_project.context_features import add_context_features
from sintef_project.normalized import add_normalized_features

PREFIXES = [100, 150, 200, 250, 350]
FEATURE_SOURCES = [f"sintef_project/{name}.py" for name in ["features", "physics", "context_features", "normalized"]]
STRATEGIES = ["baseline_350", "unit_auc", "unit_log_loss"]


def checksum(path: Path) -> str:
    """Return the SHA256 digest of a file without loading it all at once."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def now() -> str:
    """Return a precise UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def save_manifest(out: Path, manifest: dict) -> None:
    """Write a complete manifest snapshot."""
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def scores(labels: np.ndarray, predictions: np.ndarray, units: list[str]) -> dict:
    """Measure the exact pooled competition objective and unit diagnostics."""
    return {
        "micro_auc": float(roc_auc_score(labels.ravel(), predictions.ravel())),
        "log_loss": float(log_loss(labels.ravel(), predictions.ravel(), labels=[0, 1])),
        "unit_auc": {unit: float(roc_auc_score(labels[:, j], predictions[:, j])) for j, unit in enumerate(units)},
        "unit_log_loss": {unit: float(log_loss(labels[:, j], predictions[:, j], labels=[0, 1])) for j, unit in enumerate(units)},
    }


def load_origin(base: Path, out: Path, stage: str) -> dict:
    """Archive the exact mutable upstream manifest version used by a stage."""
    raw = (base / "manifest.json").read_bytes()
    (out / f"base_manifest_{stage}.json").write_bytes(raw)
    return json.loads(raw)


def verify_inputs(manifest: dict, origin: dict) -> None:
    """Reject a feature or raw-data mismatch with the fitted model pipeline."""
    for relative in FEATURE_SOURCES:
        if checksum(ROOT / relative) != origin["source"][relative]:
            raise ValueError(f"Current feature source differs from fitted model: {relative}")
    for relative, expected in origin["data"].items():
        actual = checksum(ROOT / relative)
        if actual != expected:
            raise ValueError(f"Input data changed: {relative}")
        manifest["data"][relative] = actual


def fold_arrays(base: Path, out: Path, year: int, x: pd.DataFrame, train: pd.DataFrame,
                units: list[str], labels: np.ndarray, origin: dict, manifest: dict) -> tuple:
    """Align the held-out frame exactly to its original case and generator order."""
    prediction_path = base / f"validation_{year}.npz"
    manifest["inputs"][str(prediction_path.relative_to(ROOT))] = checksum(prediction_path)
    with np.load(prediction_path) as data:
        case_ids = data["case_ids"]
        baseline, truth = data["predictions"], data["labels"]
        if data["units"].tolist() != units:
            raise ValueError("Original generator order differs")
    dates = pd.to_datetime(train.starttime, utc=True)
    begin, end = pd.Timestamp(f"{year}-01-01", tz="UTC"), pd.Timestamp(f"{year + 1}-01-01", tz="UTC")
    cases = np.flatnonzero((dates >= begin) & (dates < end))
    rows = (cases[:, None] * 168 + np.arange(168)).ravel()
    np.testing.assert_array_equal(case_ids, train.iloc[cases]["Run No"].to_numpy())
    np.testing.assert_array_equal(truth, labels[rows])
    fold_origin = origin["folds"][str(year)]
    retained = np.flatnonzero(dates < begin - pd.Timedelta(days=6))
    assert dates.iloc[retained].max() + pd.Timedelta(hours=167) < begin
    assert fold_origin["train_ids"] == train.iloc[retained]["Run No"].tolist()
    assert fold_origin["validation_ids"] == case_ids.tolist()
    sample_path = base / f"train_sample_{year}.npy"
    manifest["inputs"][str(sample_path.relative_to(ROOT))] = checksum(sample_path)
    manifest["folds"][str(year)] = {
        "train_ids": fold_origin["train_ids"], "validation_ids": case_ids.tolist(),
        "train_dates": fold_origin["train_dates"], "validation_dates": fold_origin["validation_dates"],
        "purge_days": 6, "sampled_training_rows": fold_origin["sampled_training_rows"],
        "baseline": scores(truth, baseline, units),
    }
    save_manifest(out, manifest)
    return x.iloc[rows], truth, baseline, case_ids


def prefix_predictions(base: Path, out: Path, year: int, frame: pd.DataFrame,
                       units: list[str], baseline: np.ndarray, manifest: dict) -> np.ndarray:
    """Predict disjoint raw-score blocks, accumulating to the declared prefixes."""
    predictions = np.empty((len(PREFIXES), len(frame), len(units)), dtype=np.float32)
    for j, unit in enumerate(units):
        checkpoint = base / f"fold{year}_{unit}.txt"
        digest = checksum(checkpoint)
        manifest["inputs"][str(checkpoint.relative_to(ROOT))] = digest
        model = lgb.Booster(model_file=str(checkpoint))
        if model.current_iteration() != 350 or model.feature_name() != [name.replace(" ", "_") for name in frame.columns]:
            raise ValueError(f"Unexpected model iteration count or feature order: {checkpoint}")
        raw = np.zeros(len(frame), dtype=np.float64)
        previous = 0
        for k, count in enumerate(PREFIXES):
            raw += model.predict(frame, start_iteration=previous, num_iteration=count - previous,
                                 raw_score=True, num_threads=2)
            predictions[k, :, j] = 1 / (1 + np.exp(-raw))
            previous = count
        if checksum(checkpoint) != digest:
            raise ValueError(f"Checkpoint changed while predicting: {checkpoint}")
        error = float(np.max(np.abs(predictions[-1, :, j] - baseline[:, j])))
        if error > 1e-7:
            raise ValueError(f"350-tree reconstruction mismatch for {unit}: {error}")
        manifest["folds"][str(year)].setdefault("prefix_reconstruction_max_error", {})[unit] = error
        print(f"PREFIX {year} {unit} full_reconstruction_max_error={error:.3g}", flush=True)
    if not np.isfinite(predictions).all() or np.any((predictions < 0) | (predictions > 1)):
        raise ValueError("Predictions are not finite probabilities")
    return predictions


def choose_prefixes(predictions: np.ndarray, truth: np.ndarray, units: list[str]) -> tuple:
    """Choose two fixed rules, with exact ties preferring the larger prefix."""
    prefix_metrics = [scores(truth, p, units) for p in predictions]
    counts = {"baseline_350": {unit: 350 for unit in units}, "unit_auc": {}, "unit_log_loss": {}}
    for unit in units:
        counts["unit_auc"][unit] = PREFIXES[max(range(len(PREFIXES)), key=lambda k: (prefix_metrics[k]["unit_auc"][unit], PREFIXES[k]))]
        counts["unit_log_loss"][unit] = PREFIXES[min(range(len(PREFIXES)), key=lambda k: (prefix_metrics[k]["unit_log_loss"][unit], -PREFIXES[k]))]
    return counts, dict(zip(map(str, PREFIXES), prefix_metrics))


def assemble(predictions: np.ndarray, counts: dict, units: list[str]) -> np.ndarray:
    """Assemble one generator-specific prefix prediction table."""
    return np.column_stack([predictions[PREFIXES.index(counts[unit]), :, j] for j, unit in enumerate(units)])


def report(out: Path, manifest: dict) -> None:
    """Write a concise experiment report with links to complete provenance."""
    lines = ["# Generator-specific prefix selection", "", f"Run `{manifest['run_id']}`; status `{manifest['status']}`.", "", manifest["hypothesis"], "", f"Exact command: `{manifest['executions'][-1]['command']}`", "", manifest["selection_protocol"], "", manifest["reuse_bias"], "", f"Complete hashes, model parameters, feature definitions, raw input paths, split IDs, source snapshot, environment and runtime are in `{out.relative_to(ROOT)}/manifest.json`.", ""]
    if "selection" in manifest:
        selected = manifest["selection"]
        lines += [f"Frozen primary strategy: `{selected['strategy']}`, frozen at `{selected['frozen_utc']}` before opening 2022 predictions.", "", "|Generator|AUC rule|Log-loss rule|Primary|", "|---|---:|---:|---:|"]
        for unit in manifest["units"]:
            lines.append(f"|{unit}|{selected['candidate_counts']['unit_auc'][unit]}|{selected['candidate_counts']['unit_log_loss'][unit]}|{selected['counts'][unit]}|")
        lines += ["", "|2021 strategy|Micro AUC|Log loss|", "|---|---:|---:|"]
        for strategy, result in manifest["folds"]["2021"]["strategies"].items():
            lines.append(f"|{strategy}|{result['micro_auc']:.9f}|{result['log_loss']:.9f}|")
    for year, fold in manifest["folds"].items():
        if "selected" in fold:
            lines += ["", f"{year}: baseline micro AUC {fold['baseline']['micro_auc']:.12f}; selected {fold['selected']['micro_auc']:.12f}; delta {fold['delta_auc']:+.12f}."]
    if "meets_transfer_threshold" in manifest:
        lines += ["", f"Both folds ≥0.99: **{manifest['meets_transfer_threshold']}**. Independent 2022 improvement: **{manifest['improves_2022']}**."]
    lines += ["", "No refitting, early stopping, learned preprocessing, calibration or ensembling. The fixed pretrained model seed and sampling settings are inherited verbatim. Predictions retain `[0,1]` probabilities and original case-major/hour-major order. Prefix raw scores are accumulated over nonoverlapping tree ranges; 350-tree output must match the original saved predictions within 1e-7.", "", "Kaggle submission ID, leaderboard score, rank and observation time: none; no Kaggle action performed."]
    if "error" in manifest:
        lines += ["", "Failure:", "```", manifest["error"], "```"]
    (ROOT / "docs/prefix_selection.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    """Predeclare, select on 2021, persist the selection, and evaluate 2022."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="prefix-selection-001")
    parser.add_argument("--base-run", type=Path, default=ROOT / "artifacts/lgb-normalized-context-001")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    base = args.base_run.resolve()
    out = ROOT / "artifacts" / args.run_id
    started = time.monotonic()
    execution = {"started_utc": now(), "command": shlex.join([sys.executable, *sys.argv])}
    if args.resume:
        manifest = json.loads((out / "manifest.json").read_text())
        if manifest["status"] != "awaiting_2022" or "selection" not in manifest:
            raise ValueError("Resume is allowed only for a frozen selection awaiting 2022")
        if manifest["source"]["scripts/prefix_selection.py"] != checksum(Path(__file__)):
            raise ValueError("Selection script changed before resume")
    else:
        out.mkdir(parents=True, exist_ok=False)
        manifest = {
            "run_id": args.run_id, "timestamp_utc": now(), "status": "running", "executions": [],
            "hypothesis": "Early overfitting differs by generator; frozen per-generator prefixes may improve pooled micro AUC without changing fitted trees.",
            "selection_protocol": "Predeclared prefixes [100,150,200,250,350]. Compare baseline350, per-unit maximum AUC, and per-unit minimum log loss on2021. Per-unit exact ties prefer larger prefixes. Choose one strategy by highest2021 pooled microAUC; exact ties prefer baseline350 then unitAUC then unitLogLoss. Persist all counts and the primary choice before opening2022. Evaluate only the primary frozen strategy on2022. No2022 feedback into settings.",
            "reuse_bias": "2021 has already been reused for model-family and feature development; its selection score is optimistic. 2022 is an untouched holdout for this prefix choice, but it has also informed earlier project-level choices and is not a pristine final test. Local AUC is not a leaderboard score.",
            "prefixes": PREFIXES, "candidate_strategies": STRATEGIES, "prediction_threads": 2,
            "early_stopping": "None: truncate existing fixed350-round boosters only", "ensemble_weights": [1.0],
            "base_run": str(base), "source": {}, "data": {}, "inputs": {}, "folds": {},
            "kaggle": {"submission_id": None, "score": None, "rank": None, "observation_time": None},
            "environment": {"platform": platform.platform(), "python": sys.version, "cpu_count": os.cpu_count(),
                "hardware": subprocess.run(["system_profiler", "SPHardwareDataType"], capture_output=True, text=True).stdout.split("Serial Number")[0],
                "packages": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scikit-learn", "lightgbm", "pyyaml"]}},
        }
        for relative in ["scripts/prefix_selection.py", "scripts/train.py", *FEATURE_SOURCES, "sintef_project/kaggle_metric.py"]:
            path = ROOT / relative
            manifest["source"][relative] = checksum(path)
            destination = out / "source" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    manifest["executions"].append(execution)
    manifest["status"] = "running"
    save_manifest(out, manifest)
    try:
        origin = load_origin(base, out, "evaluation_start" if args.resume else "selection_start")
        if origin["arguments"]["feature_set"] != "normalized-context" or origin["arguments"]["trees"] != 350:
            raise ValueError("This predeclared probe requires normalized-context350 models")
        verify_inputs(manifest, origin)
        manifest["model_parameters"] = origin["model_parameters"]
        manifest["training_arguments"] = origin["arguments"]
        manifest["feature_spec"] = origin["feature_spec"]
        manifest["features"] = origin["features"]
        save_manifest(out, manifest)
        x, labels, train, _, metadata = build_features(ROOT)
        x, _ = add_physics_features(x, ROOT)
        x, _ = add_context_features(x, ROOT)
        x, _ = add_normalized_features(x, ROOT)
        x = x[origin["features"]]
        units = metadata["units"]
        manifest["units"] = units
        y = np.column_stack([labels[unit] for unit in units]).astype(np.uint8)
        print(f"FEATURES {x.shape}", flush=True)
        if not args.resume:
            xv, truth, baseline, case_ids = fold_arrays(base, out, 2021, x, train, units, y, origin, manifest)
            predictions = prefix_predictions(base, out, 2021, xv, units, baseline, manifest)
            counts, prefix_metrics = choose_prefixes(predictions, truth, units)
            strategy_predictions = {name: assemble(predictions, counts[name], units) for name in STRATEGIES}
            strategy_metrics = {name: scores(truth, strategy_predictions[name], units) for name in STRATEGIES}
            primary = max(STRATEGIES, key=lambda name: strategy_metrics[name]["micro_auc"])
            selection = {"strategy": primary, "counts": counts[primary], "candidate_counts": counts,
                         "frozen_utc": now(), "selection_fold": 2021, "evaluation_fold": 2022}
            # Freeze before opening any2022 prediction file or label slice.
            (out / "selection_2021.json").write_text(json.dumps(selection, indent=2) + "\n")
            manifest["selection"] = selection
            fold = manifest["folds"]["2021"]
            fold.update(prefix_metrics=prefix_metrics, strategies=strategy_metrics, selected=strategy_metrics[primary])
            fold["delta_auc"] = fold["selected"]["micro_auc"] - fold["baseline"]["micro_auc"]
            save_manifest(out, manifest)
            for name, prediction in strategy_predictions.items():
                np.savez_compressed(out / f"{name}_validation_2021.npz", predictions=prediction, labels=truth, case_ids=case_ids, units=units)
            np.savez_compressed(out / "prefixes_2021.npz", predictions=predictions, prefixes=PREFIXES, labels=truth, case_ids=case_ids, units=units)
            shutil.copy2(out / f"{primary}_validation_2021.npz", out / "validation_2021.npz")
            print("FROZEN", json.dumps(selection), flush=True)
            print("STRATEGIES2021", json.dumps({k: v["micro_auc"] for k, v in strategy_metrics.items()}), flush=True)
            del xv, predictions, strategy_predictions
        waiting_started = time.monotonic()
        while True:
            available = (base / "validation_2022.npz").is_file()
            current_origin = json.loads((base / "manifest.json").read_text())
            if available and "2022" in current_origin["folds"]:
                break
            if time.monotonic() - waiting_started >= 60:
                manifest["status"] = "awaiting_2022"
                print("AWAITING2022: rerun exact command with --resume once upstream fold completes", flush=True)
                return
            time.sleep(5)
        origin = load_origin(base, out, "evaluation_2022")
        xv, truth, baseline, case_ids = fold_arrays(base, out, 2022, x, train, units, y, origin, manifest)
        selected = np.empty_like(baseline)
        for j, unit in enumerate(units):
            path = base / f"fold2022_{unit}.txt"
            before = checksum(path)
            manifest["inputs"][str(path.relative_to(ROOT))] = before
            model = lgb.Booster(model_file=str(path))
            if model.current_iteration() != 350 or model.feature_name() != [name.replace(" ", "_") for name in xv.columns]:
                raise ValueError(f"Unexpected2022 model schema: {unit}")
            selected[:, j] = model.predict(xv, num_iteration=manifest["selection"]["counts"][unit], num_threads=2)
            if checksum(path) != before:
                raise ValueError(f"2022 checkpoint changed: {unit}")
            print(f"FROZEN_EVAL2022 {unit} trees={manifest['selection']['counts'][unit]}", flush=True)
        fold = manifest["folds"]["2022"]
        fold["selected"] = scores(truth, selected, units)
        fold["delta_auc"] = fold["selected"]["micro_auc"] - fold["baseline"]["micro_auc"]
        np.savez_compressed(out / "validation_2022.npz", predictions=selected, labels=truth, case_ids=case_ids, units=units)
        manifest["improves_2022"] = fold["delta_auc"] > 0
        manifest["meets_transfer_threshold"] = all(f["selected"]["micro_auc"] >= .99 for f in manifest["folds"].values())
        manifest["status"] = "complete"
        print("RESULT", json.dumps({year: {key: item[key] for key in ["baseline", "selected", "delta_auc"]} for year, item in manifest["folds"].items()}), flush=True)
    except BaseException:
        manifest["status"] = "failed"
        manifest["error"] = traceback.format_exc()
        raise
    finally:
        execution.update(finished_utc=now(), runtime_seconds=time.monotonic() - started, status=manifest["status"])
        manifest["runtime_seconds"] = sum(item["runtime_seconds"] for item in manifest["executions"])
        manifest["artifacts"] = {str(path.relative_to(out)): checksum(path) for path in sorted(out.rglob("*")) if path.is_file() and path.name != "manifest.json"}
        save_manifest(out, manifest)
        report(out, manifest)


if __name__ == "__main__":
    main()
