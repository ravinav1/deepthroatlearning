"""Fixed CatBoost specialists on exact LP-v1 samples with held-out blending."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import catboost
from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scripts.vinje_count import FEATURE_SOURCES, checksum, environment, metrics
from sintef_project.features import build_features
from sintef_project.physics import add_physics_features
from sintef_project.context_features import add_context_features
from sintef_project.lp_features import add_lp_features
from sintef_project.normalized import add_normalized_features

SPECIALISTS = ["Haukeli_G1", "Lio_G1", "Hogga_G1"]
WEIGHTS = [0.0, 0.25, 0.5, 1.0]
PARAMETERS = {
    "iterations": 600, "depth": 6, "learning_rate": .08, "l2_leaf_reg": 5,
    "border_count": 128, "random_strength": 1, "bootstrap_type": "Bayesian",
    "bagging_temperature": 1, "boosting_type": "Plain", "loss_function": "Logloss",
    "random_seed": 4173, "thread_count": 2, "task_type": "CPU", "cat_features": [],
    "use_best_model": False, "allow_writing_files": False, "verbose": False,
}


def mix(base: np.ndarray, raw: np.ndarray, indices: list[int], weight: float) -> np.ndarray:
    """Apply a single fixed probability weight only to the three specialists."""
    result = base.copy()
    result[:, indices] = (1 - weight) * result[:, indices] + weight * raw
    return result


def write_report(out: Path, manifest: dict) -> None:
    """Save a concise report; the parent owns the shared experiment log."""
    lines = ["# CatBoost specialists", "", f"Run `{manifest['run_id']}`; status `{manifest['status']}`.", "", manifest["hypothesis"], "", manifest["selection_protocol"], "", "|Fold|LP-v1 baseline|Frozen CatBoost blend|Delta|", "|---|---:|---:|---:|"]
    for year, fold in manifest["folds"].items():
        if "selected" in fold:
            lines.append(f"|{year}|{fold['baseline']['micro_auc']:.12f}|{fold['selected']['micro_auc']:.12f}|{fold['delta_auc']:+.12f}|")
    lines += ["", f"Frozen joint weight: `{manifest.get('selected_weight')}`. Frozen2022 improvement: `{manifest.get('improves_2022')}`. Both folds≥0.998: `{manifest.get('meets_transfer_gate')}`.", "", "The three units use the same395 normalized-LP-v1 features, exact stored baseline sampled rows, and six-day purged chronological splits. CatBoost settings are fixed before fitting:600 iterations, depth6, learning rate0.08,L2 leaf regularization5,border_count128,random_strength1,Bayesian bootstrap with bagging_temperature1,Plain boosting,Logloss,seed4173 and2 CPU threads. No categorical features, evaluation set, early stopping, extra calibration, feature changes or parameter search. Every fitted model’s complete expanded defaults are saved through get_all_params().", "", f"Command: `{manifest['command']}`. Runtime: {manifest['runtime_seconds']:.2f}s. Full source snapshots, raw input/cache/model/sample hashes, environment, expanded parameters, exact split dates/IDs, selected probabilities and checkpoint replay checks: `{out.relative_to(ROOT)}/manifest.json`.", "", "Both years are reused project development folds. Only2021 selects this experiment’s joint blend;2022 is checked with that weight fixed. No GPU dependency, installation, network request, Kaggle action or main-repository transfer occurs in this script."]
    if "error" in manifest:
        lines += ["", "Failure:", "```", manifest["error"], "```"]
    (ROOT / "docs/catboost_specialists.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    """Fit one predeclared learner family and freeze its joint blend on2021."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="catboost-lp-specialists-001")
    parser.add_argument("--base-run", type=Path, default=ROOT / "artifacts/lgb-normalized-lp-001")
    args = parser.parse_args()
    base = args.base_run.resolve()
    out = ROOT / "artifacts" / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = {"run_id": args.run_id, "timestamp_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
        "command": shlex.join([sys.executable, *sys.argv]), "hypothesis": "A fixed CatBoost learner may provide complementary commitment errors to the LP-v1 LightGBM baseline on the three highest-priority weak generators.",
        "model_parameters_requested": PARAMETERS, "expanded_parameters": {}, "specialists": SPECIALISTS, "weight_grid": WEIGHTS,
        "selection_protocol": "Predeclare joint replacement weights[0,.25,.5,1] for Haukeli/Lio/Hogga only. Select highest2021 pooled microAUC; exact ties prefer smaller weight. Persist the choice before reading2022 validation predictions/label slice; apply only that frozen weight on2022. Other11 units remain exact baseline probabilities.",
        "seed": 4173, "early_stopping": "none; no eval_set; use_best_model=False; fixed600 iterations",
        "environment": environment(), "data": {}, "source": {}, "inputs": {}, "folds": {},
        "job_gate": "Parent confirmed one-thread joint LP cache is the only other heavy process; CatBoost is the second heavy job. No process-list access; latest artifact manifests were checked.",
        "kaggle": {"submission_id": None, "score": None, "rank": None, "observation_time": None}}
    manifest["environment"]["packages"]["catboost"] = catboost.__version__
    def persist() -> None:
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        path = base / "manifest.json"
        origin = json.loads(path.read_text())
        if origin.get("status") != "complete" or origin["arguments"]["feature_set"] != "normalized-lp":
            raise ValueError("Require completed LP-v1 baseline")
        manifest["inputs"][str(path.relative_to(ROOT))] = checksum(path)
        shutil.copy2(path, out / "base_manifest.json")
        for relative, expected in origin["data"].items():
            actual = checksum(ROOT / relative)
            if actual != expected:
                raise ValueError(f"Changed input: {relative}")
            manifest["data"][relative] = actual
        for relative in ["scripts/catboost_specialists.py", "scripts/vinje_count.py", *FEATURE_SOURCES]:
            actual = checksum(ROOT / relative)
            if relative in FEATURE_SOURCES and actual != origin["source"][relative]:
                raise ValueError(f"Changed feature builder: {relative}")
            manifest["source"][relative] = actual
            destination = out / "source" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, destination)
        columns = origin["features"]
        if len(columns) != 395:
            raise ValueError("Expected exact395-column baseline schema")
        manifest.update(features=columns, feature_spec=origin["feature_spec"], baseline_training_arguments=origin["arguments"])
        persist()
        (out / "protocol_predeclared.json").write_text(json.dumps({key: manifest[key] for key in ["timestamp_utc", "model_parameters_requested", "weight_grid", "specialists", "selection_protocol", "source", "data", "features"]}, indent=2) + "\n")
        x, labels, train, _, metadata = build_features(ROOT)
        x, _ = add_physics_features(x, ROOT)
        x, _ = add_context_features(x, ROOT)
        x, lp_spec = add_lp_features(x, ROOT)
        x, _ = add_normalized_features(x, ROOT)
        if list(x.columns) != columns:
            raise ValueError("Feature order differs from baseline")
        manifest["lp_input_provenance"] = lp_spec
        units = metadata["units"]
        indices = [units.index(unit) for unit in SPECIALISTS]
        y = np.column_stack([labels[unit] for unit in units]).astype(np.uint8)
        dates = pd.to_datetime(train.starttime, utc=True)
        print("FEATURES", x.shape, flush=True)
        for year in (2021, 2022):
            begin, end = pd.Timestamp(f"{year}-01-01", tz="UTC"), pd.Timestamp(f"{year+1}-01-01", tz="UTC")
            training_cases = np.flatnonzero(dates < begin - pd.Timedelta(days=6))
            validation_cases = np.flatnonzero((dates >= begin) & (dates < end))
            assert dates.iloc[training_cases].max() + pd.Timedelta(hours=167) < begin
            inherited = origin["folds"][str(year)]
            assert train.iloc[training_cases]["Run No"].tolist() == inherited["train_ids"]
            assert train.iloc[validation_cases]["Run No"].tolist() == inherited["validation_ids"]
            sample_path = base / f"train_sample_{year}.npy"
            sample = np.load(sample_path)
            manifest["inputs"][str(sample_path.relative_to(ROOT))] = checksum(sample_path)
            assert len(np.unique(sample)) == len(sample) and np.isin(sample // 168, training_cases).all()
            np.save(out / sample_path.name, sample)
            va = (validation_cases[:, None] * 168 + np.arange(168)).ravel()
            xt, xv = x.iloc[sample], x.iloc[va]
            raw = np.empty((len(va), len(SPECIALISTS)), dtype=np.float32)
            replay_errors = {}
            for j, unit in enumerate(SPECIALISTS):
                model = CatBoostClassifier(**PARAMETERS)
                model.fit(xt, y[sample, indices[j]])
                assert model.tree_count_ == 600 and model.classes_.tolist() == [0, 1]
                assert model.feature_names_ == columns
                expanded = model.get_all_params()
                manifest["expanded_parameters"].setdefault(str(year), {})[unit] = expanded
                (out / f"params_{year}_{unit}.json").write_text(json.dumps(expanded, indent=2) + "\n")
                checkpoint = out / f"fold{year}_{unit}.cbm"
                model.save_model(checkpoint)
                raw[:, j] = model.predict_proba(xv, thread_count=2)[:, 1]
                restored = CatBoostClassifier()
                restored.load_model(checkpoint)
                replay = restored.predict_proba(xv.iloc[:64], thread_count=2)[:, 1].astype(np.float32)
                np.testing.assert_array_equal(replay, raw[:64, j])
                replay_errors[unit] = 0.0
                persist()
                print("MODEL", year, unit, "auc", float(roc_auc_score(y[va, indices[j]], raw[:, j])), flush=True)
            path = base / f"validation_{year}.npz"
            manifest["inputs"][str(path.relative_to(ROOT))] = checksum(path)
            with np.load(path) as saved:
                np.testing.assert_array_equal(saved["labels"], y[va])
                np.testing.assert_array_equal(saved["case_ids"], train.iloc[validation_cases]["Run No"].to_numpy())
                assert saved["units"].tolist() == units
                original = saved["predictions"]
            fold = {key: inherited[key] for key in ["train_ids", "validation_ids", "train_dates", "validation_dates", "purge_days", "sampled_training_rows"]}
            fold.update(baseline=metrics(y[va], original, units), checkpoint_replay_max_errors=replay_errors,
                        raw_specialist_unit_auc={unit: float(roc_auc_score(y[va, indices[j]], raw[:, j])) for j, unit in enumerate(SPECIALISTS)})
            if year == 2021:
                scores = [metrics(y[va], mix(original, raw, indices, weight), units)["micro_auc"] for weight in WEIGHTS]
                manifest["selected_weight"] = WEIGHTS[int(np.argmax(scores))]
                manifest["selection_frozen_utc"] = datetime.now(timezone.utc).isoformat()
                fold["candidate_micro_auc"] = dict(zip(map(str, WEIGHTS), scores))
                manifest["folds"][str(year)] = fold
                persist()
                (out / "selection_2021.json").write_text(json.dumps({"weight": manifest["selected_weight"], "frozen_utc": manifest["selection_frozen_utc"], "scores": fold["candidate_micro_auc"]}, indent=2) + "\n")
                print("FROZEN", manifest["selected_weight"], scores, flush=True)
            selected = mix(original, raw, indices, manifest["selected_weight"])
            assert np.isfinite(selected).all() and ((selected >= 0) & (selected <= 1)).all()
            fold["selected"] = metrics(y[va], selected, units)
            fold["delta_auc"] = fold["selected"]["micro_auc"] - fold["baseline"]["micro_auc"]
            manifest["folds"][str(year)] = fold
            np.savez_compressed(out / f"validation_{year}.npz", predictions=selected, labels=y[va], case_ids=train.iloc[validation_cases]["Run No"].to_numpy(), units=units, specialist_predictions=raw, specialist_units=SPECIALISTS)
            persist()
            print("FOLD", year, fold["baseline"]["micro_auc"], fold["selected"]["micro_auc"], fold["delta_auc"], flush=True)
        manifest["improves_2022"] = manifest["folds"]["2022"]["delta_auc"] > 0
        manifest["meets_transfer_gate"] = all(fold["selected"]["micro_auc"] >= .998 for fold in manifest["folds"].values())
        manifest["ensemble_weights"] = {"specialists": [1 - manifest["selected_weight"], manifest["selected_weight"]], "other_units": [1., 0.]}
        manifest["status"] = "complete"
    except BaseException:
        manifest["status"] = "failed"
        manifest["error"] = traceback.format_exc()
        raise
    finally:
        manifest["runtime_seconds"] = time.monotonic() - started
        manifest["artifacts"] = {str(path.relative_to(out)): checksum(path) for path in sorted(out.rglob("*")) if path.is_file() and path.name != "manifest.json"}
        persist()
        write_report(out, manifest)


if __name__ == "__main__":
    main()
