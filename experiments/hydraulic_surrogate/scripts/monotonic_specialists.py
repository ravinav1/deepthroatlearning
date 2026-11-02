"""Fixed monotonic LP specialists, selected on2021 and frozen for2022."""
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
import lightgbm as lgb
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
WEIGHTS = [0.0, 0.5, 1.0]
DIRECTION_AUDIT = {
    "lp_qmin_ratio": "q/Qmin from input-only LP: greater modeled discharge relative to minimum operating flow supports greater commitment; positive shape prior retained.",
    "lp_power_fraction": "LP power/Pmax: greater modeled production supports greater commitment; positive shape prior retained.",
    "lp_economic_margin": "(price-(lambda_up-lambda_down)/Ebest)/(1+mean(abs(price))): a larger relative operating margin supports commitment; positive shape prior retained.",
    "limitations": "These are conditional modeling priors, not physical necessities. LPv1 omits directional storage, integer minimum flow and startup costs; output features are correlated and independently changing one while fixing all others may be infeasible. Forced generation can occur at negative margins. No negative direction follows from that fact, and no supplied definition supports reversing a sign. Retain all three requested +1 directions and constrain no other column.",
    "omitted_constraints": [],
    "documentation": "https://lightgbm.readthedocs.io/en/stable/Parameters.html#monotone_constraints",
    "method": "Explicit basic method (LightGBM default), monotone_penalty0; no method or penalty search.",
}


def positive_affine_aliases(frame: pd.DataFrame, source_names: list[str]) -> dict:
    """Find exact positive affine aliases using only legal feature values.

    A deterministic probe rejects candidates cheaply, but every accepted
    identity is verified over every row. Exact means float64 equality with
    no tolerance; approximate or rank-correlated columns are not aliases.
    """
    if frame.empty or frame.columns.duplicated().any():
        raise ValueError("Alias audit requires a nonempty unique-column table")
    results = {}
    for source in source_names:
        values = frame[source].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite alias source: {source}")
        low, high = int(np.argmin(values)), int(np.argmax(values))
        span = values[high] - values[low]
        if span <= 0:
            raise ValueError(f"Constant source cannot define an affine direction: {source}")
        probes = np.unique(np.concatenate([np.linspace(0, len(frame) - 1, min(32, len(frame)), dtype=np.int64), [low, high]]))
        matches = []
        for name in frame.columns:
            candidate = frame[name].to_numpy()
            slope = (float(candidate[high]) - float(candidate[low])) / span
            intercept = float(candidate[low]) - slope * values[low]
            if not np.isfinite(slope) or not np.isfinite(intercept) or slope <= 0:
                continue
            if not np.array_equal(candidate[probes].astype(np.float64), slope * values[probes] + intercept):
                continue
            if np.array_equal(candidate.astype(np.float64), slope * values + intercept):
                matches.append({"column": str(name), "slope": float(slope), "intercept": float(intercept), "verified_rows": len(frame)})
        if source not in [match["column"] for match in matches]:
            raise ValueError(f"Source failed identity audit: {source}")
        results[source] = matches
    return results


def mix(base: np.ndarray, raw: np.ndarray, indices: list[int], weight: float) -> np.ndarray:
    """Soft-replace the same three generators with one predeclared shared weight."""
    result = base.copy()
    result[:, indices] = (1 - weight) * result[:, indices] + weight * raw
    return result


def write_report(out: Path, manifest: dict) -> None:
    """Summarize all outcomes while keeping the shared experiment log root-owned."""
    lines = ["# Monotonic LP specialists", "", f"Run `{manifest['run_id']}`; status `{manifest['status']}`.", "", manifest["hypothesis"], "", "Direction audit before fitting:", ""]
    for key, value in DIRECTION_AUDIT.items():
        if key not in ["documentation", "omitted_constraints"]:
            lines.append(f"- `{key}`: {value}")
    lines += ["", "All other feature directions are0. LightGBM requires the full feature-order vector; every vector and named index is saved before fitting. [Official parameter documentation](https://lightgbm.readthedocs.io/en/stable/Parameters.html#monotone_constraints).", "", manifest["selection_protocol"], "", "|Fold|Baseline micro AUC|Frozen specialist blend|Delta|", "|---|---:|---:|---:|"]
    for year, fold in manifest["folds"].items():
        if "selected" in fold:
            lines.append(f"|{year}|{fold['baseline']['micro_auc']:.12f}|{fold['selected']['micro_auc']:.12f}|{fold['delta_auc']:+.12f}|")
    lines += ["", f"Frozen weight: `{manifest.get('selected_weight')}`. Independent2022 improvement: `{manifest.get('improves_2022')}`. Both folds≥0.998: `{manifest.get('meets_transfer_gate')}`.", "", f"Exact command: `{manifest['command']}`. Runtime {manifest['runtime_seconds']:.2f}s. Complete source snapshots, all data/input/model/sample/artifact hashes, environment, split dates/IDs, model settings, direction audit and saved predictions: `{out.relative_to(ROOT)}/manifest.json`.", "", "The baseline’s exact395 features, saved sampled rows and six-day purge are reused. Only the three specified units are refit, using350 trees,31 leaves, learning rate0.06, minimum100 rows/leaf,L2=1,column fraction0.9,maxbin255,seed4173 and2 threads. No early stopping, extra feature, projection, smoothing or calibration. Both years are reused project development folds;2022 is held out only from this experiment’s weight selection. No Kaggle action or transfer."]
    if "error" in manifest:
        lines += ["", "Failure:", "```", manifest["error"], "```"]
    destination = "monotonic_alias_specialists.md" if manifest.get("include_affine_aliases") else "monotonic_specialists.md"
    if manifest.get("include_affine_aliases"):
        lines += ["", "Corrected alias policy: all exact positive affine aliases of each original target coordinate are also constrained +1. Aliases are detected on every row of the complete legal395-feature table without labels or scores; no approximate aliases or score-based exceptions. Full slopes/intercepts and expanded vectors were frozen before fitting."]
    (ROOT / "docs" / destination).write_text("\n".join(lines) + "\n")


def main() -> None:
    """Run a fixed isolated monotonicity ablation on both chronological folds."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="lgb-monotonic-specialists-001")
    parser.add_argument("--base-run", type=Path, default=ROOT / "artifacts/lgb-normalized-lp-001")
    parser.add_argument("--reuse-failed-run", type=Path)
    parser.add_argument("--include-affine-aliases", action="store_true")
    parser.add_argument("--gate-run", type=Path, action="append", default=[])
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    gates = {str(path): json.loads((path / "manifest.json").read_text()).get("status", "running") if (path / "manifest.json").is_file() else "missing" for path in args.gate_run}
    if args.check_only:
        print(json.dumps({"gates": gates, "ready": all(status == "complete" for status in gates.values())}), flush=True)
        return
    if any(status != "complete" for status in gates.values()):
        raise SystemExit(f"Heavy-job gate not ready: {gates}; no feature build or fit started")
    base = args.base_run.resolve()
    out = ROOT / "artifacts" / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = {"run_id": args.run_id, "timestamp_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
        "command": shlex.join([sys.executable, *sys.argv]), "hypothesis": "Positive monotonic priors on each weak unit’s own LP dispatch/output/economic margin may reduce regime-sensitive corrections while preserving nonlinear unconstrained context.",
        "direction_audit": DIRECTION_AUDIT, "specialists": SPECIALISTS, "weight_grid": WEIGHTS,
        "include_affine_aliases": args.include_affine_aliases, "gate_at_start": gates,
        "alias_policy": "Before fitting, scan all395 columns against each of9 originally constrained coordinates. Infer slope/intercept from extrema, require positive slope, prescreen deterministic probes, verify exact float64 equality on every row. Include every accepted positive affine alias with+1; no labels/scores, tolerances, feature removal or later choice of aliases. Constant/negative/approximate relations are excluded.",
        "selection_protocol": "Predeclare joint specialist weights[0,.5,1] on Haukeli/Lio/Hogga only. Choose highest2021 pooled microAUC; exact ties prefer smaller weight. Freeze choice before reading2022 validation predictions/label slice, then evaluate only that weight on2022. All other units remain the original baseline probabilities.",
        "seed": 4173, "early_stopping": "none; fixed350 rounds", "environment": environment(),
        "data": {}, "source": {}, "inputs": {}, "folds": {},
        "kaggle": {"submission_id": None, "score": None, "rank": None, "observation_time": None}}
    def persist() -> None:
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        path = base / "manifest.json"
        origin = json.loads(path.read_text())
        if origin.get("status") != "complete" or origin["arguments"]["feature_set"] != "normalized-lp":
            raise ValueError("Require completed normalized-LP v1 baseline")
        manifest["inputs"][str(path.relative_to(ROOT))] = checksum(path)
        shutil.copy2(path, out / "base_manifest.json")
        for relative, expected in origin["data"].items():
            actual = checksum(ROOT / relative)
            if actual != expected:
                raise ValueError(f"Input changed: {relative}")
            manifest["data"][relative] = actual
        for relative in ["scripts/monotonic_specialists.py", "scripts/vinje_count.py", *FEATURE_SOURCES]:
            actual = checksum(ROOT / relative)
            if relative in FEATURE_SOURCES and actual != origin["source"][relative]:
                raise ValueError(f"Feature source changed: {relative}")
            manifest["source"][relative] = actual
            dest = out / "source" / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, dest)
        columns = origin["features"]
        if len(columns) != 395:
            raise ValueError("Require exact395-column baseline schema")
        model_parameters = {}
        named_constraints = {}
        for unit in SPECIALISTS:
            names = [f"{prefix}_{unit}" for prefix in ["lp_qmin_ratio", "lp_power_fraction", "lp_economic_margin"]]
            vector = [int(name in names) for name in columns]
            if sum(vector) != 3:
                raise ValueError(f"Missing requested constrained column: {unit}")
            named_constraints[unit] = {name: columns.index(name) for name in names}
            parameters = origin["model_parameters"].copy()
            parameters.update(n_jobs=2, monotone_constraints=vector, monotone_constraints_method="basic", monotone_penalty=0.0)
            model_parameters[unit] = lgb.LGBMClassifier(**parameters).get_params()
        manifest.update(features=columns, feature_spec=origin["feature_spec"], model_parameters=model_parameters,
                        named_constraints=named_constraints, baseline_training_arguments=origin["arguments"])
        persist()
        protocol_name = "protocol_initial.json" if args.include_affine_aliases else "protocol_predeclared.json"
        (out / protocol_name).write_text(json.dumps({key: manifest[key] for key in ["timestamp_utc", "direction_audit", "named_constraints", "features", "model_parameters", "weight_grid", "selection_protocol", "source", "data", "alias_policy", "include_affine_aliases"]}, indent=2) + "\n")
        x, labels, train, _, metadata = build_features(ROOT)
        x, _ = add_physics_features(x, ROOT)
        x, _ = add_context_features(x, ROOT)
        x, lp_spec = add_lp_features(x, ROOT)
        x, _ = add_normalized_features(x, ROOT)
        if list(x.columns) != columns:
            raise ValueError("Feature order differs from baseline")
        manifest["lp_input_provenance"] = lp_spec
        if args.include_affine_aliases:
            original_constraints = {unit: list(names) for unit, names in named_constraints.items()}
            aliases = positive_affine_aliases(x, [name for names in original_constraints.values() for name in names])
            manifest["affine_aliases"] = aliases
            manifest["original_named_constraints"] = named_constraints.copy()
            manifest["alias_audit_shape"] = list(x.shape)
            for unit in SPECIALISTS:
                expanded = {match["column"] for name in original_constraints[unit] for match in aliases[name]}
                named_constraints[unit] = {name: columns.index(name) for name in columns if name in expanded}
                model_parameters[unit]["monotone_constraints"] = [int(name in expanded) for name in columns]
            manifest["named_constraints"] = named_constraints
            manifest["model_parameters"] = model_parameters
            manifest["alias_protocol_frozen_utc"] = datetime.now(timezone.utc).isoformat()
            (out / "protocol_predeclared.json").write_text(json.dumps({key: manifest[key] for key in ["timestamp_utc", "direction_audit", "named_constraints", "features", "model_parameters", "weight_grid", "selection_protocol", "source", "data", "alias_policy", "include_affine_aliases", "affine_aliases", "alias_audit_shape", "alias_protocol_frozen_utc"]}, indent=2) + "\n")
            persist()
            print("ALIASES_FROZEN", json.dumps(named_constraints), flush=True)
        previous = None
        if args.reuse_failed_run:
            previous = json.loads((args.reuse_failed_run / "manifest.json").read_text())
            if previous["status"] != "failed" or previous["data"] != manifest["data"] or previous["model_parameters"] != model_parameters:
                raise ValueError("Failed-run checkpoint configuration mismatch")
            if any(previous["source"][path] != manifest["source"][path] for path in FEATURE_SOURCES):
                raise ValueError("Failed-run feature source changed")
            manifest["inputs"][str(args.reuse_failed_run / "manifest.json")] = checksum(args.reuse_failed_run / "manifest.json")
            manifest["checkpoint_reuse"] = {"path": str(args.reuse_failed_run), "rule": "Reuse only completed checkpoint whose recorded SHA256, exact model parameters, feature sources, raw data and sampled rows match; no predictions or selection were available in the failed run."}
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
            monotonic_checks = {}
            for j, unit in enumerate(SPECIALISTS):
                checkpoint_name = f"fold{year}_{unit}.txt"
                prior_path = args.reuse_failed_run / checkpoint_name if args.reuse_failed_run else None
                if previous is not None and prior_path.is_file():
                    if checksum(prior_path) != previous["artifacts"][checkpoint_name] or checksum(args.reuse_failed_run / sample_path.name) != checksum(sample_path):
                        raise ValueError("Failed-run checkpoint or row sample changed")
                    booster = lgb.Booster(model_file=str(prior_path))
                    manifest["inputs"][str(prior_path)] = checksum(prior_path)
                    shutil.copy2(prior_path, out / checkpoint_name)
                    print("REUSE_CHECKPOINT", year, unit, flush=True)
                else:
                    model = lgb.LGBMClassifier(**model_parameters[unit])
                    model.fit(xt, y[sample, indices[j]])
                    booster = model.booster_
                    booster.save_model(str(out / checkpoint_name))
                assert booster.current_iteration() == 350
                raw[:, j] = booster.predict(xv, num_threads=2)
                checks = {}
                for name in named_constraints[unit]:
                    lower, upper = xt.iloc[:32].copy(), xt.iloc[:32].copy()
                    lower[name], upper[name] = float(xt[name].min()), float(xt[name].max())
                    delta = booster.predict(upper, num_threads=2) - booster.predict(lower, num_threads=2)
                    if delta.min() < -1e-12:
                        raise ValueError(f"Positive monotonicity check failed: {unit}/{name}")
                    checks[name] = {"minimum_probability_delta": float(delta.min()), "maximum_probability_delta": float(delta.max()), "rows": 32, "low": float(xt[name].min()), "high": float(xt[name].max())}
                monotonic_checks[unit] = checks
                print("MODEL", year, unit, "auc", float(roc_auc_score(y[va, indices[j]], raw[:, j])), flush=True)
            path = base / f"validation_{year}.npz"
            manifest["inputs"][str(path.relative_to(ROOT))] = checksum(path)
            with np.load(path) as saved:
                np.testing.assert_array_equal(saved["labels"], y[va])
                np.testing.assert_array_equal(saved["case_ids"], train.iloc[validation_cases]["Run No"].to_numpy())
                assert saved["units"].tolist() == units
                original = saved["predictions"]
            fold = {key: inherited[key] for key in ["train_ids", "validation_ids", "train_dates", "validation_dates", "purge_days", "sampled_training_rows"]}
            fold.update(baseline=metrics(y[va], original, units), monotonic_checks=monotonic_checks,
                        raw_specialist_unit_auc={unit: float(roc_auc_score(y[va, indices[j]], raw[:, j])) for j, unit in enumerate(SPECIALISTS)})
            if year == 2021:
                scores = [metrics(y[va], mix(original, raw, indices, weight), units)["micro_auc"] for weight in WEIGHTS]
                winner = int(np.argmax(scores))
                manifest["selected_weight"] = WEIGHTS[winner]
                manifest["selection_frozen_utc"] = datetime.now(timezone.utc).isoformat()
                fold["candidate_micro_auc"] = dict(zip(map(str, WEIGHTS), scores))
                manifest["folds"][str(year)] = fold
                persist()
                (out / "selection_2021.json").write_text(json.dumps({"selected_weight": manifest["selected_weight"], "frozen_utc": manifest["selection_frozen_utc"], "scores": fold["candidate_micro_auc"]}, indent=2) + "\n")
                print("FROZEN", manifest["selected_weight"], scores, flush=True)
            selected = mix(original, raw, indices, manifest["selected_weight"])
            fold["selected"] = metrics(y[va], selected, units)
            fold["delta_auc"] = fold["selected"]["micro_auc"] - fold["baseline"]["micro_auc"]
            manifest["folds"][str(year)] = fold
            np.savez_compressed(out / f"validation_{year}.npz", predictions=selected, labels=y[va], case_ids=train.iloc[validation_cases]["Run No"].to_numpy(), units=units, specialist_predictions=raw, specialist_units=SPECIALISTS)
            persist()
            print("FOLD", year, fold["baseline"]["micro_auc"], fold["selected"]["micro_auc"], fold["delta_auc"], flush=True)
        manifest["improves_2022"] = manifest["folds"]["2022"]["delta_auc"] > 0
        manifest["meets_transfer_gate"] = all(fold["selected"]["micro_auc"] >= .998 for fold in manifest["folds"].values())
        manifest["ensemble_weights"] = {"specialists": [1 - manifest["selected_weight"], manifest["selected_weight"]], "other_units": [1.0, 0.0]}
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
