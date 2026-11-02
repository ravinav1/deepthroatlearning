"""Fixed Vinje count classifier with 2021-only blend selection and frozen 2022."""
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
from sintef_project.lp_features import add_lp_features
from sintef_project.normalized import add_normalized_features

VINJE = ["Vinje_G1", "Vinje_G2", "Vinje_G3"]
WEIGHTS = [0.0, 0.25, 0.5, 1.0]
FEATURE_SOURCES = [f"sintef_project/{name}.py" for name in ["features", "physics", "context_features", "lp_features", "normalized"]]


def checksum(path: Path) -> str:
    """Return a SHA256 digest with bounded memory use."""
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def metrics(y: np.ndarray, p: np.ndarray, units: list[str]) -> dict:
    """Measure the exact pooled objective and per-unit behavior."""
    return {"micro_auc": float(roc_auc_score(y.ravel(), p.ravel())),
            "log_loss": float(log_loss(y.ravel(), p.ravel(), labels=[0, 1])),
            "unit_auc": {unit: float(roc_auc_score(y[:, j], p[:, j])) for j, unit in enumerate(units)}}


def gate_status(base: Path) -> dict:
    """Require both earlier expensive jobs to finish before any new fit."""
    paths = {"baseline_lp": base / "manifest.json", "physics_large": ROOT / "artifacts/lgb-physics-large-001/manifest.json"}
    return {name: json.loads(path.read_text()).get("status", "running") if path.is_file() else "missing" for name, path in paths.items()}


def nested_fraction(labels: np.ndarray) -> float:
    """Count rare violations of the usual G1>=G2>=G3 commitment ordering."""
    return float(np.mean((labels[:, 0] < labels[:, 1]) | (labels[:, 1] < labels[:, 2])))


def blend(base: np.ndarray, count_probability: np.ndarray, indices: list[int], weight: float) -> np.ndarray:
    """Replace only Vinje probabilities by a fixed soft mixture."""
    result = base.copy()
    result[:, indices] = (1 - weight) * result[:, indices] + weight * count_probability
    return result


def report(out: Path, manifest: dict) -> None:
    """Write a reviewable experiment summary without changing shared logs."""
    lines = ["# Vinje count classifier", "", f"Run `{manifest['run_id']}`; status `{manifest['status']}`.", "", manifest["hypothesis"], "", f"Command: `{manifest['command']}`", "", manifest["selection_protocol"], "", manifest["nonnested_caveat"], "", "|Fold|Baseline micro AUC|Frozen blend micro AUC|Delta|", "|---|---:|---:|---:|"]
    for year, fold in manifest["folds"].items():
        if "selected" in fold:
            lines.append(f"|{year}|{fold['baseline']['micro_auc']:.12f}|{fold['selected']['micro_auc']:.12f}|{fold['delta_auc']:+.12f}|")
    lines += ["", f"Frozen count-model weight: `{manifest.get('frozen_weight')}`. Both folds≥0.998: `{manifest.get('meets_transfer_gate')}`. Independent2022 improvement: `{manifest.get('improves_2022')}`.", "", "The features, sampled row indices and purged splits are exactly those from normalized-LP v1. The model is multiclass LightGBM with count0..3,350 rounds,31 leaves, learning rate0.06, minimum100 rows/leaf,L2=1,column fraction0.9,maxbin255,seed4173 and2 threads. No early stopping. Other parameters are copied explicitly from the baseline and recorded in the manifest.", "", f"Runtime: {manifest['runtime_seconds']:.2f}s. Full source/data/model/sample/artifact hashes, complete row IDs, feature definitions, configuration and environment: `{out.relative_to(ROOT)}/manifest.json`.", "", "2021 and2022 have both been reused for broader project development. Only2021 chooses this experiment's blend; the frozen2022 result is a transfer check, not a pristine final test. No Kaggle action or main-repository transfer is performed by this script."]
    if "error" in manifest:
        lines += ["", "Failure:", "```", manifest["error"], "```"]
    (ROOT / "docs/vinje_count.md").write_text("\n".join(lines) + "\n")


def environment() -> dict:
    """Capture software and hardware for derived artifacts."""
    return {"platform": platform.platform(), "python": sys.version, "cpu_count": os.cpu_count(),
            "hardware": subprocess.run(["system_profiler", "SPHardwareDataType"], capture_output=True, text=True).stdout.split("Serial Number")[0],
            "packages": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "pyyaml"]}}


def export_csv(out: Path, predictions: np.ndarray, mapping: pd.DataFrame, units: list[str]) -> dict:
    """Export and reload exact competition IDs/columns with bounded rounding error."""
    template = pd.read_csv(ROOT / "sintef_project/sample_submission.csv")
    assert predictions.shape == (len(mapping) * 168, len(units))
    assert np.isfinite(predictions).all() and ((predictions >= 0) & (predictions <= 1)).all()
    cube = predictions.reshape(len(mapping), 168, len(units))
    values = {}
    for name in template.columns[1:]:
        unit, hour = name.removeprefix("result_committed_").rsplit("_t", 1)
        values[name] = cube[:, int(hour), units.index(unit)]
    frame = pd.DataFrame(values, index=mapping["Run No"]).loc[template["Run No"]].reset_index()
    assert frame.shape == template.shape and list(frame) == list(template)
    frame.to_csv(out / "submission.csv", index=False, float_format="%.9g")
    loaded = pd.read_csv(out / "submission.csv")
    assert list(loaded) == list(template) and loaded["Run No"].equals(template["Run No"])
    error = float(np.max(np.abs(loaded.iloc[:, 1:].to_numpy() - frame.iloc[:, 1:].to_numpy())))
    assert error <= 5.1e-10
    return {"shape": list(frame.shape), "sha256": checksum(out / "submission.csv"), "readback_max_abs_error": error, "probability_range": [float(predictions.min()), float(predictions.max())]}


def final_main() -> None:
    """Fit or replay the final count model using the exact baseline final sample."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--base-final-run", type=Path, default=ROOT / "artifacts/lgb-normalized-lp-final-001")
    parser.add_argument("--validation-run", type=Path, default=ROOT / "artifacts/vinje-count-001")
    parser.add_argument("--replay-from", type=Path)
    args = parser.parse_args()
    out = ROOT / "artifacts" / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = {"run_id": args.run_id, "timestamp_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
        "command": shlex.join([sys.executable, *sys.argv]), "hypothesis": "Reproduce the independently improved frozen50% Vinje count blend using all supplied training cases and the exact final baseline sample.",
        "environment": environment(), "source": {}, "data": {}, "inputs": {}, "seed": 4173, "early_stopping": "none; fixed350 rounds",
        "training": "none; checkpoint replay" if args.replay_from else "one fixed multiclass count model on baseline final sampled rows",
        "ensemble_weights": {"Vinje": [.5, .5], "other_units": [1., 0.]},
        "kaggle": {"submission_id": None, "score": None, "rank": None, "observation_time": None}}
    def record(path: Path) -> None:
        manifest["inputs"][str(path.resolve().relative_to(ROOT))] = checksum(path)
    def persist() -> None:
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        base, validation = args.base_final_run.resolve(), args.validation_run.resolve()
        for name, directory in [("baseline", base), ("selection", validation)]:
            record(directory / "manifest.json")
            shutil.copy2(directory / "manifest.json", out / f"{name}_manifest.json")
        origin = json.loads((base / "manifest.json").read_text())
        chosen = json.loads((validation / "manifest.json").read_text())
        if origin.get("status") != "complete" or chosen.get("status") != "complete" or chosen["frozen_weight"] != .5:
            raise ValueError("Require complete baseline and frozen50% count selection")
        if origin["features"] != chosen["features"] or origin["data"] != chosen["data"] or origin["feature_spec"] != chosen["feature_spec"]:
            raise ValueError("Final and validation feature/data recipes differ")
        manifest.update(data=origin["data"], model_parameters=chosen["model_parameters"], features=origin["features"],
                        feature_spec=origin["feature_spec"], frozen_weight=.5, validation_scores={year: f["selected"]["micro_auc"] for year, f in chosen["folds"].items()},
                        nonnested_caveat=chosen["nonnested_caveat"])
        for relative, expected in origin["data"].items():
            if checksum(ROOT / relative) != expected:
                raise ValueError(f"Changed data: {relative}")
        for relative in ["scripts/vinje_count.py", *FEATURE_SOURCES]:
            actual = checksum(ROOT / relative)
            if relative in FEATURE_SOURCES and (actual != origin["source"][relative] or actual != chosen["source"][relative]):
                raise ValueError(f"Feature source mismatch: {relative}")
            manifest["source"][relative] = actual
            dest = out / "source" / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, dest)
        sample_path = base / "train_sample_final.npy"
        raw_path = base / "test_predictions.npy"
        for path in [sample_path, raw_path]:
            record(path)
            if checksum(path) != origin["artifacts"][path.name]:
                raise ValueError(f"Baseline artifact changed: {path}")
        record(validation / "selection_2021.json")
        x, labels, train, mapping, metadata = build_features(ROOT)
        x, _ = add_physics_features(x, ROOT)
        x, _ = add_context_features(x, ROOT)
        x, lp_spec = add_lp_features(x, ROOT)
        x, _ = add_normalized_features(x, ROOT)
        if list(x.columns) != origin["features"]:
            raise ValueError("Final feature order mismatch")
        units = metadata["units"]
        indices = [units.index(unit) for unit in VINJE]
        sample = np.load(sample_path)
        nt = len(train) * 168
        assert len(np.unique(sample)) == len(sample) and sample.min() >= 0 and sample.max() < nt
        np.save(out / "train_sample_final.npy", sample)
        manifest.update(train_ids=train["Run No"].tolist(), train_dates=[str(train.starttime.min()), str(train.starttime.max())],
                        test_ids=mapping["Run No"].tolist(), test_dates=[str(mapping.starttime.min()), str(mapping.starttime.max())],
                        sampled_training_rows=len(sample), lp_input_provenance=lp_spec, units=units)
        xtest = x.iloc[nt:]
        persist()
        if args.replay_from:
            previous = args.replay_from.resolve()
            previous_manifest = json.loads((previous / "manifest.json").read_text())
            checkpoint = previous / "final_Vinje_count.txt"
            if previous_manifest["status"] != "complete" or checksum(checkpoint) != previous_manifest["artifacts"][checkpoint.name]:
                raise ValueError("Final count checkpoint missing or changed")
            record(previous / "manifest.json")
            record(checkpoint)
            model = lgb.Booster(model_file=str(checkpoint))
            class_probability = model.predict(xtest, num_threads=2)
            shutil.copy2(checkpoint, out / checkpoint.name)
            manifest["replay_origin"] = str(previous)
        else:
            vinje_y = np.column_stack([labels[unit] for unit in VINJE])
            target = vinje_y[sample].sum(axis=1).astype(np.uint8)
            manifest["sample_nonnested_fraction"] = nested_fraction(vinje_y[sample])
            manifest["training_count_distribution"] = np.bincount(target, minlength=4).tolist()
            model = lgb.LGBMClassifier(**chosen["model_parameters"])
            print("FINAL_FIT", x.shape, "sample_rows", len(sample), flush=True)
            model.fit(x.iloc[sample], target)
            assert model.classes_.tolist() == [0, 1, 2, 3]
            model.booster_.save_model(str(out / "final_Vinje_count.txt"))
            class_probability = model.predict_proba(xtest)
        replay = lgb.Booster(model_file=str(out / "final_Vinje_count.txt")).predict(xtest, num_threads=2)
        np.testing.assert_array_equal(replay, class_probability)
        manifest["checkpoint_replay_max_abs_error"] = 0.0
        cumulative = np.column_stack([class_probability[:, 1:].sum(axis=1), class_probability[:, 2:].sum(axis=1), class_probability[:, 3]]).astype(np.float32)
        original = np.load(raw_path)
        selected = blend(original, cumulative, indices, .5)
        np.save(out / "test_predictions.npy", selected)
        np.savez_compressed(out / "count_predictions_test.npz", count_probabilities=class_probability,
                            vinje_probabilities=cumulative, case_ids=mapping["Run No"].to_numpy(), units=VINJE)
        manifest["submission"] = export_csv(out, selected, mapping, units)
        if args.replay_from:
            np.testing.assert_array_equal(selected, np.load(previous / "test_predictions.npy"))
            assert checksum(out / "submission.csv") == checksum(previous / "submission.csv")
            manifest["full_artifact_replay_exact"] = True
        manifest["status"] = "complete"
        print("FINAL_RESULT", json.dumps(manifest["submission"]), flush=True)
    except BaseException:
        manifest["status"] = "failed"
        manifest["error"] = traceback.format_exc()
        raise
    finally:
        manifest["runtime_seconds"] = time.monotonic() - started
        manifest["artifacts"] = {str(path.relative_to(out)): checksum(path) for path in sorted(out.rglob("*")) if path.is_file() and path.name != "manifest.json"}
        persist()
        with (ROOT / "docs/vinje_count.md").open("a") as handle:
            handle.write(f"\n## {args.run_id}\n\nStatus: {manifest['status']}. Command: `{manifest['command']}`. Runtime: {manifest['runtime_seconds']:.2f}s. Complete provenance: `{out.relative_to(ROOT)}/manifest.json`. Checkpoint prediction replay error: {manifest.get('checkpoint_replay_max_abs_error')}. Submission: `{manifest.get('submission')}`. No Kaggle action or transfer.\n")
            if "error" in manifest:
                handle.write("\nFailure: " + manifest["error"].splitlines()[-1] + "\n")


def combine_main() -> None:
    """Evaluate the requested fixed component composition without additional tuning."""
    from sintef_project.postprocess import project_vinje
    parser = argparse.ArgumentParser()
    parser.add_argument("--combine", action="store_true")
    parser.add_argument("--run-id", default="vinje-count-combined-001")
    parser.add_argument("--count-run", type=Path, default=ROOT / "artifacts/vinje-count-001")
    parser.add_argument("--xgb-run", type=Path, default=ROOT / "artifacts/xgb-lp-specialists-001")
    parser.add_argument("--base-run", type=Path, default=ROOT / "artifacts/lgb-normalized-lp-001")
    args = parser.parse_args()
    out = ROOT / "artifacts" / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = {"run_id": args.run_id, "timestamp_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
        "command": shlex.join([sys.executable, *sys.argv]), "hypothesis": "Combine independently fixed50% specialist and50% Vinje-count mixtures on disjoint units, then apply the explicitly requested fixed Vinje L2 projection.",
        "environment": environment(), "source": {}, "inputs": {}, "folds": {}, "training": "none",
        "selection": "No new weight search. Both component weights were frozen using2021. Evaluate one explicitly requested composition on both folds.",
        "reuse_bias": "The combination was proposed after component results from both years were known. Both scores are reused development evidence; neither is claimed to be an untouched test of this newly composed pipeline.",
        "weights": {"Haukeli_Lio_Hogga_XGB": .5, "Vinje_count": .5}, "projection": "fixed L2 projection G1>=G2>=G3; rare nonnested ground truth preserved",
        "kaggle": {"submission_id": None, "score": None, "rank": None, "observation_time": None}}
    try:
        origins = {}
        for name, directory in [("count", args.count_run), ("xgb", args.xgb_run), ("baseline", args.base_run)]:
            path = directory / "manifest.json"
            manifest["inputs"][str(path)] = checksum(path)
            origins[name] = json.loads(path.read_text())
            shutil.copy2(path, out / f"{name}_manifest.json")
        assert origins["count"]["frozen_weight"] == .5 and origins["xgb"]["selected_weight"] == .5
        assert all(value["status"] == "complete" for value in origins.values())
        manifest["data"] = origins["baseline"]["data"]
        manifest["model_parameters"] = {name: value.get("model_parameters") for name, value in origins.items()}
        manifest["feature_spec"] = origins["baseline"]["feature_spec"]
        for relative in ["scripts/vinje_count.py", "sintef_project/postprocess.py"]:
            path = ROOT / relative
            manifest["source"][relative] = checksum(path)
            dest = out / "source" / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
        for year in (2021, 2022):
            arrays = {}
            for name, directory in [("count", args.count_run), ("xgb", args.xgb_run), ("baseline", args.base_run)]:
                path = directory / f"validation_{year}.npz"
                manifest["inputs"][str(path)] = checksum(path)
                with np.load(path) as saved:
                    arrays[name] = {key: saved[key] for key in ["predictions", "labels", "case_ids", "units"]}
            baseline, count, xgb = [arrays[name] for name in ["baseline", "count", "xgb"]]
            for component in [count, xgb]:
                for key in ["labels", "case_ids", "units"]:
                    np.testing.assert_array_equal(component[key], baseline[key])
            units = baseline["units"].tolist()
            indices = [units.index(unit) for unit in VINJE]
            specialist_indices = [units.index(unit) for unit in ["Haukeli_G1", "Lio_G1", "Hogga_G1"]]
            np.testing.assert_array_equal(count["predictions"][:, [j for j in range(14) if j not in indices]], baseline["predictions"][:, [j for j in range(14) if j not in indices]])
            np.testing.assert_array_equal(xgb["predictions"][:, [j for j in range(14) if j not in specialist_indices]], baseline["predictions"][:, [j for j in range(14) if j not in specialist_indices]])
            combined = xgb["predictions"].copy()
            combined[:, indices] = count["predictions"][:, indices]
            result = project_vinje(combined, units)
            truth = baseline["labels"]
            fold = {"split": origins["baseline"]["folds"][str(year)],
                    "baseline": metrics(truth, baseline["predictions"], units),
                    "xgb_standalone": metrics(truth, xgb["predictions"], units),
                    "count_standalone": metrics(truth, count["predictions"], units),
                    "combined_fixed_projection": metrics(truth, result, units)}
            fold["delta_vs_baseline"] = fold["combined_fixed_projection"]["micro_auc"] - fold["baseline"]["micro_auc"]
            fold["delta_vs_xgb"] = fold["combined_fixed_projection"]["micro_auc"] - fold["xgb_standalone"]["micro_auc"]
            manifest["folds"][str(year)] = fold
            np.savez_compressed(out / f"validation_{year}.npz", predictions=result, labels=truth, case_ids=baseline["case_ids"], units=units)
            print("COMBINED", year, "baseline", fold["baseline"]["micro_auc"], "xgb", fold["xgb_standalone"]["micro_auc"], "combined_projected", fold["combined_fixed_projection"]["micro_auc"], flush=True)
        manifest["meets_transfer_gate"] = all(fold["combined_fixed_projection"]["micro_auc"] >= .998 for fold in manifest["folds"].values())
        manifest["status"] = "complete"
    except BaseException:
        manifest["status"] = "failed"
        manifest["error"] = traceback.format_exc()
        raise
    finally:
        manifest["runtime_seconds"] = time.monotonic() - started
        manifest["artifacts"] = {str(path.relative_to(out)): checksum(path) for path in sorted(out.rglob("*")) if path.is_file() and path.name != "manifest.json"}
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        with (ROOT / "docs/vinje_count.md").open("a") as handle:
            handle.write(f"\n## Fixed specialist/count composition\n\nRun `{args.run_id}`, status `{manifest['status']}`. Command: `{manifest['command']}`. No extra tuning; both original component weights remain0.5, followed by the fixed Vinje projection. The composition was proposed after both years' component results were known and is development evidence.\n")
            for year, fold in manifest["folds"].items():
                handle.write(f"\n{year}: baseline {fold['baseline']['micro_auc']:.12f}; XGB standalone {fold['xgb_standalone']['micro_auc']:.12f}; fixed combined+projection {fold['combined_fixed_projection']['micro_auc']:.12f}.\n")
            handle.write(f"\nFull provenance and saved validation predictions: `{out.relative_to(ROOT)}/manifest.json`. Runtime {manifest['runtime_seconds']:.2f}s. No Kaggle action or transfer.\n")
            if "error" in manifest:
                handle.write("\nFailure: " + manifest["error"].splitlines()[-1] + "\n")


def combined_final_main() -> None:
    """Replay final count/XGB checkpoints from one rebuilt feature table and export."""
    import xgboost as xgb
    from sintef_project.postprocess import project_vinje
    parser = argparse.ArgumentParser()
    parser.add_argument("--combined-final", action="store_true")
    parser.add_argument("--run-id", default="vinje-count-combined-final-001")
    parser.add_argument("--base-final-run", type=Path, default=ROOT / "artifacts/lgb-normalized-lp-final-001")
    parser.add_argument("--count-final-run", type=Path, default=ROOT / "artifacts/vinje-count-final-001")
    parser.add_argument("--xgb-final-run", type=Path, default=ROOT / "artifacts/xgb-lp-specialists-final-001")
    parser.add_argument("--oof-run", type=Path, default=ROOT / "artifacts/vinje-count-combined-001")
    args = parser.parse_args()
    directories = {"baseline": args.base_final_run.resolve(), "count": args.count_final_run.resolve(),
                   "xgb": args.xgb_final_run.resolve(), "oof": args.oof_run.resolve()}
    origins = {name: json.loads((directory / "manifest.json").read_text()) for name, directory in directories.items()}
    if any(origin.get("status") != "complete" for origin in origins.values()):
        raise SystemExit("All final models and combined OOF evaluation must complete before export")
    out = ROOT / "artifacts" / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = {"run_id": args.run_id, "timestamp_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
        "command": shlex.join([sys.executable, *sys.argv]), "environment": environment(), "source": {}, "inputs": {},
        "training": "none; final count and all3 final XGB models reloaded and replayed from reconstructed features before merging",
        "recipe": "Start from baseline LP final probabilities. On Haukeli/Lio/Hogga use0.5baseline+0.5XGB; on Vinje use0.5baseline+0.5cumulative-count; leave other8 units baseline. Then fixed L2 Vinje projection. No further calibration, rounding to labels, smoothing or weight search.",
        "reuse_bias": origins["oof"]["reuse_bias"], "validation_scores": {year: fold["combined_fixed_projection"]["micro_auc"] for year, fold in origins["oof"]["folds"].items()},
        "data": origins["baseline"]["data"], "features": origins["baseline"]["features"], "feature_spec": origins["baseline"]["feature_spec"],
        "weights": origins["oof"]["weights"], "projection": origins["oof"]["projection"], "seed": None, "early_stopping": None,
        "kaggle": {"submission_id": None, "score": None, "rank": None, "observation_time": None}}
    manifest["environment"]["packages"]["xgboost"] = importlib.metadata.version("xgboost")
    def record(path: Path) -> None:
        manifest["inputs"][str(path.resolve().relative_to(ROOT))] = checksum(path)
    try:
        for name, directory in directories.items():
            record(directory / "manifest.json")
            shutil.copy2(directory / "manifest.json", out / f"{name}_manifest.json")
        for name in ["count", "xgb"]:
            if origins[name]["features"] != manifest["features"] or origins[name]["data"] != manifest["data"]:
                raise ValueError(f"Final {name} feature or raw-data recipe mismatch")
        for relative, expected in manifest["data"].items():
            if checksum(ROOT / relative) != expected:
                raise ValueError(f"Raw input changed: {relative}")
        for relative in ["scripts/vinje_count.py", "sintef_project/postprocess.py", *FEATURE_SOURCES]:
            path = ROOT / relative
            actual = checksum(path)
            if relative in FEATURE_SOURCES and any(actual != origins[name]["source"][relative] for name in ["baseline", "count", "xgb"]):
                raise ValueError(f"Feature source mismatch: {relative}")
            manifest["source"][relative] = actual
            destination = out / "source" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        x, _, train, mapping, metadata = build_features(ROOT)
        x, _ = add_physics_features(x, ROOT)
        x, _ = add_context_features(x, ROOT)
        x, lp_spec = add_lp_features(x, ROOT)
        x, _ = add_normalized_features(x, ROOT)
        if list(x.columns) != manifest["features"]:
            raise ValueError("Reconstructed feature columns/order mismatch")
        units = metadata["units"]
        xtest = x.iloc[len(train) * 168:]
        manifest.update(units=units, test_ids=mapping["Run No"].tolist(),
                        test_dates=[str(mapping.starttime.min()), str(mapping.starttime.max())], lp_input_provenance=lp_spec)
        count_checkpoint = directories["count"] / "final_Vinje_count.txt"
        record(count_checkpoint)
        if checksum(count_checkpoint) != origins["count"]["artifacts"][count_checkpoint.name]:
            raise ValueError("Final count checkpoint changed")
        count_model = lgb.Booster(model_file=str(count_checkpoint))
        count_probability = count_model.predict(xtest, num_threads=2)
        path = directories["count"] / "count_predictions_test.npz"
        record(path)
        with np.load(path) as saved:
            np.testing.assert_array_equal(saved["case_ids"], mapping["Run No"].to_numpy())
            assert saved["units"].tolist() == VINJE
            np.testing.assert_array_equal(count_probability, saved["count_probabilities"])
            cumulative = np.column_stack([count_probability[:, 1:].sum(axis=1), count_probability[:, 2:].sum(axis=1), count_probability[:, 3]]).astype(np.float32)
            np.testing.assert_array_equal(cumulative, saved["vinje_probabilities"])
        manifest["count_replay_max_abs_error"] = 0.0
        specialist_units = origins["xgb"]["units"]
        assert specialist_units == ["Haukeli_G1", "Lio_G1", "Hogga_G1"]
        path = directories["xgb"] / "test_predictions.npz"
        record(path)
        with np.load(path) as saved:
            np.testing.assert_array_equal(saved["case_ids"], mapping["Run No"].to_numpy())
            assert saved["units"].tolist() == specialist_units
            expected_specialists = saved["predictions"]
        specialist_probability = np.empty_like(expected_specialists)
        for j, unit in enumerate(specialist_units):
            checkpoint = directories["xgb"] / f"final_{unit}.ubj"
            record(checkpoint)
            if checksum(checkpoint) != origins["xgb"]["artifacts"][checkpoint.name]:
                raise ValueError(f"Final XGB checkpoint changed: {unit}")
            model = xgb.XGBClassifier()
            model.load_model(checkpoint)
            model.set_params(n_jobs=2)
            specialist_probability[:, j] = model.predict_proba(xtest)[:, 1]
            np.testing.assert_array_equal(specialist_probability[:, j], expected_specialists[:, j])
            print("REPLAY_EXACT", unit, flush=True)
        manifest["xgb_replay_max_abs_error"] = 0.0
        baseline_path = directories["baseline"] / "test_predictions.npy"
        record(baseline_path)
        if checksum(baseline_path) != origins["baseline"]["artifacts"][baseline_path.name]:
            raise ValueError("Baseline final probabilities changed")
        baseline = np.load(baseline_path)
        vinje_indices = [units.index(unit) for unit in VINJE]
        selected = blend(baseline, cumulative, vinje_indices, .5)
        count_matrix_path = directories["count"] / "test_predictions.npy"
        record(count_matrix_path)
        np.testing.assert_array_equal(selected, np.load(count_matrix_path))
        specialist_indices = [units.index(unit) for unit in specialist_units]
        selected[:, specialist_indices] = .5 * baseline[:, specialist_indices] + .5 * specialist_probability
        selected = project_vinje(selected, units)
        other = [j for j in range(len(units)) if j not in vinje_indices + specialist_indices]
        np.testing.assert_array_equal(selected[:, other], baseline[:, other])
        np.save(out / "test_predictions.npy", selected)
        np.savez_compressed(out / "replayed_components.npz", count_probabilities=count_probability, specialist_probabilities=specialist_probability,
                            case_ids=mapping["Run No"].to_numpy(), specialist_units=specialist_units, count_units=VINJE)
        manifest["submission"] = export_csv(out, selected, mapping, units)
        manifest["status"] = "complete"
        print("COMBINED_FINAL", json.dumps(manifest["submission"]), flush=True)
    except BaseException:
        manifest["status"] = "failed"
        manifest["error"] = traceback.format_exc()
        raise
    finally:
        manifest["runtime_seconds"] = time.monotonic() - started
        manifest["artifacts"] = {str(path.relative_to(out)): checksum(path) for path in sorted(out.rglob("*")) if path.is_file() and path.name != "manifest.json"}
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        with (ROOT / "docs/vinje_count.md").open("a") as handle:
            handle.write(f"\n## Combined final export and checkpoint replay\n\nRun `{args.run_id}`, status `{manifest['status']}`. Command: `{manifest['command']}`. No fitting. Reconstructed the shared feature table once, replayed final count and all three XGB checkpoints, verified equality with saved component probabilities, then merged the frozen weights and applied the fixed Vinje projection. Count replay maximum error {manifest.get('count_replay_max_abs_error')}; XGB replay maximum error {manifest.get('xgb_replay_max_abs_error')}.\n\nSubmission `{manifest.get('submission')}`. Runtime {manifest['runtime_seconds']:.2f}s. Complete recipe, checkpoints and hashes: `{out.relative_to(ROOT)}/manifest.json`. No Kaggle action or transfer.\n")
            if "error" in manifest:
                handle.write("\nFailure: " + manifest["error"].splitlines()[-1] + "\n")


def main() -> None:
    """Fit one predeclared count model per purged chronological fold."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="vinje-count-001")
    parser.add_argument("--base-run", type=Path, default=ROOT / "artifacts/lgb-normalized-lp-001")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    base = args.base_run.resolve()
    gate = gate_status(base)
    if args.check_only:
        print(json.dumps({"gate": gate, "ready": all(status == "complete" for status in gate.values())}), flush=True)
        return
    if any(status != "complete" for status in gate.values()):
        raise SystemExit(f"Training gate not ready: {gate}; no run directory created and no fit started")
    out = ROOT / "artifacts" / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = {
        "run_id": args.run_id, "timestamp_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
        "command": shlex.join([sys.executable, *sys.argv]), "gate_at_start": gate,
        "hypothesis": "Joint prediction of Vinje's active-unit count shares hydraulic demand information and regularizes the weakest third generator while preserving soft probabilities.",
        "selection_protocol": "Predeclared weights[0,.25,.5,1] apply jointly to only3 Vinje units. Choose largest2021 pooled microAUC; exact ties prefer smaller weight. Persist choice before accessing2022 validation predictions/label slice. Apply that one weight unchanged on2022.",
        "nonnested_caveat": "The target is the sum of3 true unit commitments. Cumulative count probabilities assume the usual G1>=G2>=G3 ordering, which rare observed labels violate. Report exact nonnested fractions per training sample and validation fold; do not relabel originals, force hard0/1 decisions, or assert exact nesting. Soft blending can preserve nonnested baseline probabilities.",
        "count_mapping": {"Vinje_G1": "P(count>=1)", "Vinje_G2": "P(count>=2)", "Vinje_G3": "P(count>=3)"},
        "candidate_weights": WEIGHTS, "seed": 4173, "early_stopping": "none; fixed350 rounds", "data": {}, "source": {}, "inputs": {}, "folds": {},
        "environment": {"platform": platform.platform(), "python": sys.version, "cpu_count": os.cpu_count(),
            "hardware": subprocess.run(["system_profiler", "SPHardwareDataType"], capture_output=True, text=True).stdout.split("Serial Number")[0],
            "packages": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "pyyaml"]}},
        "kaggle": {"submission_id": None, "score": None, "rank": None, "observation_time": None},
    }
    def persist() -> None:
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        origin_path = base / "manifest.json"
        origin = json.loads(origin_path.read_text())
        if origin["arguments"]["feature_set"] != "normalized-lp":
            raise ValueError("Require frozen normalized-LP v1 baseline")
        manifest["inputs"][str(origin_path.relative_to(ROOT))] = checksum(origin_path)
        shutil.copy2(origin_path, out / "base_manifest.json")
        for relative, expected in origin["data"].items():
            actual = checksum(ROOT / relative)
            if actual != expected:
                raise ValueError(f"Raw input changed: {relative}")
            manifest["data"][relative] = actual
        for relative in ["scripts/vinje_count.py", *FEATURE_SOURCES]:
            path = ROOT / relative
            actual = checksum(path)
            if relative in FEATURE_SOURCES and actual != origin["source"][relative]:
                raise ValueError(f"Feature code changed from baseline: {relative}")
            manifest["source"][relative] = actual
            dest = out / "source" / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
        parameters = origin["model_parameters"].copy()
        parameters.update(objective="multiclass", num_class=4, n_estimators=350, num_leaves=31,
                          learning_rate=.06, min_child_samples=100, reg_lambda=1.0,
                          colsample_bytree=.9, n_jobs=2, random_state=4173)
        manifest["model_parameters"] = lgb.LGBMClassifier(**parameters).get_params()
        manifest["features"] = origin["features"]
        manifest["feature_spec"] = origin["feature_spec"]
        manifest["baseline_training_arguments"] = origin["arguments"]
        persist()
        x, labels, train, _, metadata = build_features(ROOT)
        x, _ = add_physics_features(x, ROOT)
        x, _ = add_context_features(x, ROOT)
        x, lp_spec = add_lp_features(x, ROOT)
        x, _ = add_normalized_features(x, ROOT)
        if list(x.columns) != origin["features"]:
            raise ValueError("Feature schema or order differs from baseline")
        manifest["lp_input_provenance"] = lp_spec
        units = metadata["units"]
        vinje_indices = [units.index(unit) for unit in VINJE]
        y = np.column_stack([labels[unit] for unit in units]).astype(np.uint8)
        dates = pd.to_datetime(train.starttime, utc=True)
        print("FEATURES", x.shape, flush=True)
        for year in (2021, 2022):
            fold_origin = origin["folds"][str(year)]
            begin = pd.Timestamp(f"{year}-01-01", tz="UTC")
            end = pd.Timestamp(f"{year+1}-01-01", tz="UTC")
            tr_cases = np.flatnonzero(dates < begin - pd.Timedelta(days=6))
            va_cases = np.flatnonzero((dates >= begin) & (dates < end))
            assert dates.iloc[tr_cases].max() + pd.Timedelta(hours=167) < begin
            assert train.iloc[tr_cases]["Run No"].tolist() == fold_origin["train_ids"]
            assert train.iloc[va_cases]["Run No"].tolist() == fold_origin["validation_ids"]
            sample_path = base / f"train_sample_{year}.npy"
            sample = np.load(sample_path)
            manifest["inputs"][str(sample_path.relative_to(ROOT))] = checksum(sample_path)
            if np.unique(sample).size != len(sample) or not np.isin(sample // 168, tr_cases).all():
                raise ValueError("Training sample is duplicated or violates purge")
            np.save(out / f"train_sample_{year}.npy", sample)
            va = (va_cases[:, None] * 168 + np.arange(168)).ravel()
            counts = y[sample][:, vinje_indices].sum(axis=1).astype(np.uint8)
            model = lgb.LGBMClassifier(**parameters)
            model.fit(x.iloc[sample], counts)
            if model.classes_.tolist() != [0, 1, 2, 3]:
                raise ValueError("Missing count category in training sample")
            model.booster_.save_model(str(out / f"fold{year}_Vinje_count.txt"))
            class_probability = model.predict_proba(x.iloc[va])
            cumulative = np.column_stack([class_probability[:, 1:].sum(axis=1), class_probability[:, 2:].sum(axis=1), class_probability[:, 3]]).astype(np.float32)
            assert np.isfinite(cumulative).all() and ((cumulative >= 0) & (cumulative <= 1)).all()
            assert (cumulative[:, 0] >= cumulative[:, 1]).all() and (cumulative[:, 1] >= cumulative[:, 2]).all()
            prediction_path = base / f"validation_{year}.npz"
            manifest["inputs"][str(prediction_path.relative_to(ROOT))] = checksum(prediction_path)
            with np.load(prediction_path) as saved:
                np.testing.assert_array_equal(saved["labels"], y[va])
                np.testing.assert_array_equal(saved["case_ids"], train.iloc[va_cases]["Run No"].to_numpy())
                assert saved["units"].tolist() == units
                original = saved["predictions"]
            fold = {key: fold_origin[key] for key in ["train_ids", "validation_ids", "train_dates", "validation_dates", "purge_days", "sampled_training_rows"]}
            fold.update(baseline=metrics(y[va], original, units), training_sample_nonnested_fraction=nested_fraction(y[sample][:, vinje_indices]), validation_nonnested_fraction=nested_fraction(y[va][:, vinje_indices]), training_count_distribution=np.bincount(counts, minlength=4).tolist(), count_log_loss=float(log_loss(y[va][:, vinje_indices].sum(axis=1), class_probability, labels=[0, 1, 2, 3])))
            if year == 2021:
                scores = [metrics(y[va], blend(original, cumulative, vinje_indices, weight), units)["micro_auc"] for weight in WEIGHTS]
                winner = int(np.argmax(scores))
                manifest["frozen_weight"] = WEIGHTS[winner]
                manifest["selection_frozen_utc"] = datetime.now(timezone.utc).isoformat()
                fold["candidate_micro_auc"] = dict(zip(map(str, WEIGHTS), scores))
                manifest["folds"][str(year)] = fold
                persist()
                (out / "selection_2021.json").write_text(json.dumps({"weight": manifest["frozen_weight"], "frozen_utc": manifest["selection_frozen_utc"], "scores": fold["candidate_micro_auc"]}, indent=2) + "\n")
                print("FROZEN", manifest["frozen_weight"], "SCORES", scores, flush=True)
            selected = blend(original, cumulative, vinje_indices, manifest["frozen_weight"])
            fold["selected"] = metrics(y[va], selected, units)
            fold["delta_auc"] = fold["selected"]["micro_auc"] - fold["baseline"]["micro_auc"]
            manifest["folds"][str(year)] = fold
            np.savez_compressed(out / f"validation_{year}.npz", predictions=selected, labels=y[va], case_ids=train.iloc[va_cases]["Run No"].to_numpy(), units=units)
            np.savez_compressed(out / f"count_predictions_{year}.npz", count_probabilities=class_probability, vinje_probabilities=cumulative, case_ids=train.iloc[va_cases]["Run No"].to_numpy(), units=VINJE)
            persist()
            print("FOLD", year, "baseline", fold["baseline"]["micro_auc"], "selected", fold["selected"]["micro_auc"], "delta", fold["delta_auc"], "nonnested", fold["validation_nonnested_fraction"], flush=True)
        manifest["improves_2022"] = manifest["folds"]["2022"]["delta_auc"] > 0
        manifest["meets_transfer_gate"] = all(fold["selected"]["micro_auc"] >= .998 for fold in manifest["folds"].values())
        manifest["ensemble_weights"] = {"Vinje": [1-manifest["frozen_weight"], manifest["frozen_weight"]], "other_units": [1.0, 0.0]}
        manifest["status"] = "complete"
    except BaseException:
        manifest["status"] = "failed"
        manifest["error"] = traceback.format_exc()
        raise
    finally:
        manifest["runtime_seconds"] = time.monotonic() - started
        manifest["artifacts"] = {str(path.relative_to(out)): checksum(path) for path in sorted(out.rglob("*")) if path.is_file() and path.name != "manifest.json"}
        persist()
        report(out, manifest)


if __name__ == "__main__":
    if "--combined-final" in sys.argv:
        combined_final_main()
    elif "--final" in sys.argv:
        final_main()
    elif "--combine" in sys.argv:
        combine_main()
    else:
        main()
