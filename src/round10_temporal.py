"""Fixed, one-CPU cached linear temporal-residual pilot; creates new artifacts only."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_auc_score

from round6_evaluate import load_cache
from round2_diagnostics import paired_block_bootstrap, transition_mask, PairDiagnostics
from round5_compare import official_metric

ROOT = Path(__file__).resolve().parents[1]
HOURS, GENERATORS = 168, 14
OFFSETS = (-24, -6, -2, -1, 1, 2, 6, 24)
PROTOCOL = ROOT / "reports/round10_temporal_protocol.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, values):
    with Path(path).open("x") as handle:
        json.dump(values, handle, indent=2, allow_nan=False)
        handle.write("\n")


def logit(values):
    assert np.all((values > 0) & (values < 1))
    return np.log(values) - np.log1p(-values)


def branch(arrays):
    return (.5 * arrays["old299"] + .5 * arrays["old673"]).reshape(-1, GENERATORS, HOURS)


def identity(arrays):
    # Retain the original expression and operation order exactly.
    return .5 * arrays["round3"] + .25 * arrays["old299"] + .25 * arrays["old673"]


def design(probabilities, ids, temporal, sampled):
    """Read full within-week context before selecting any training rows."""
    count = len(ids)
    assert probabilities.shape == (count, GENERATORS, HOURS)
    feature_logits = logit(np.clip(probabilities, 1e-6, 1 - 1e-6))
    choose = np.ones_like(probabilities, dtype=bool)
    if sampled:
        hours = np.arange(HOURS)[None, :]
        case_hours = hours % 4 == ((np.asarray(ids) + 299) % 4)[:, None]
        choose = np.broadcast_to(case_hours[:, None, :], probabilities.shape)
    names = ["own_logit", *[f"generator_{i}" for i in range(GENERATORS)]]
    if temporal:
        names += [f"neighbor_minus_own_{offset:+d}h" for offset in OFFSETS]
        names += [f"neighbor_missing_{offset:+d}h" for offset in OFFSETS]
    matrix = np.zeros((int(choose.sum()), len(names)), dtype=np.float64)
    matrix[:, 0] = feature_logits[choose]
    g = np.broadcast_to(np.arange(GENERATORS)[None, :, None], probabilities.shape)[choose]
    matrix[np.arange(len(matrix)), 1 + g] = 1
    if temporal:
        for j, offset in enumerate(OFFSETS):
            difference = np.zeros_like(feature_logits)
            missing = np.ones_like(feature_logits, dtype=bool)
            if offset > 0:
                difference[:, :, :-offset] = feature_logits[:, :, offset:] - feature_logits[:, :, :-offset]
                missing[:, :, :-offset] = False
            else:
                difference[:, :, -offset:] = feature_logits[:, :, :offset] - feature_logits[:, :, -offset:]
                missing[:, :, -offset:] = False
            matrix[:, 15 + j] = difference[choose]
            matrix[:, 23 + j] = missing[choose]
    assert np.isfinite(matrix).all()
    return matrix, names, choose


def objective(beta, matrix, labels, offset, penalty):
    logits = offset + matrix @ beta
    value = np.mean(np.logaddexp(0, logits) - labels * logits) + .5 * penalty * np.dot(beta, beta)
    gradient = matrix.T @ (expit(logits) - labels) / len(labels) + penalty * beta
    return float(value), gradient


def fit_residual(probabilities, labels, ids, temporal, protocol):
    start = time.monotonic()
    matrix, names, choose = design(probabilities, ids, temporal, sampled=True)
    continuous = [0] + (list(range(15, 23)) if temporal else [])
    center = np.zeros(matrix.shape[1])
    scale = np.ones(matrix.shape[1])
    center[continuous] = matrix[:, continuous].mean(axis=0)
    scale[continuous] = np.maximum(matrix[:, continuous].std(axis=0), 1e-12)
    matrix = (matrix - center) / scale
    target, offset = labels[choose].astype(np.float64), logit(probabilities[choose])
    result = minimize(objective, np.zeros(matrix.shape[1]), args=(matrix, target, offset, protocol["l2_penalty"]),
        method="L-BFGS-B", jac=True, options={"maxiter": 200, "gtol": 1e-7, "ftol": 1e-12, "maxls": 30})
    assert result.success, result.message
    value, gradient = objective(result.x, matrix, target, offset, protocol["l2_penalty"])
    return {"variant": "temporal" if temporal else "same_hour", "feature_names": names,
        "coefficients": result.x.tolist(), "center": center.tolist(), "scale": scale.tolist(),
        "shrinkage": protocol["shrinkage"], "l2_penalty": protocol["l2_penalty"],
        "training_rows": len(target), "optimization_success": bool(result.success),
        "optimizer_message": str(result.message), "optimizer_iterations": int(result.nit),
        "optimizer_function_calls": int(result.nfev), "penalized_training_objective": value,
        "gradient_infinity_norm": float(np.abs(gradient).max()), "fit_seconds": time.monotonic() - start,
        "training_objective_is_not_validation": True}


def predict_residual(probabilities, ids, model):
    temporal = model["variant"] == "temporal"
    matrix, names, _ = design(probabilities, ids, temporal, sampled=False)
    assert names == model["feature_names"]
    correction = ((matrix - np.asarray(model["center"])) / np.asarray(model["scale"])) @ np.asarray(model["coefficients"])
    delta = (model["shrinkage"] * correction).reshape(probabilities.shape)
    assert np.isfinite(delta).all() and np.max(np.abs(delta)) < 30
    # Shift the original odds; feature-logit clipping never changes the identity.
    corrected = probabilities / (probabilities + (1 - probabilities) * np.exp(-delta))
    assert np.isfinite(corrected).all() and np.all((corrected >= 0) & (corrected <= 1))
    return corrected, {"maximum_absolute_logit_correction": float(np.abs(delta).max()),
        "mean_absolute_logit_correction": float(np.abs(delta).mean()),
        "mean_absolute_probability_change": float(np.abs(corrected - probabilities).mean())}


def self_checks():
    # Different neighbouring cases/generators expose accidental cross-boundary shifts.
    logits = np.arange(2 * GENERATORS * HOURS).reshape(2, GENERATORS, HOURS) / 1000 - 2
    probabilities = expit(logits)
    ids = np.array([10, 20])
    matrix, _, choose = design(probabilities, ids, temporal=True, sampled=False)
    shaped = matrix.reshape(2, GENERATORS, HOURS, -1)
    for j, offset in enumerate(OFFSETS):
        if offset > 0:
            assert np.all(shaped[:, :, -offset:, 15 + j] == 0)
            assert np.all(shaped[:, :, -offset:, 23 + j] == 1)
            assert np.allclose(shaped[:, :, :-offset, 15 + j], offset / 1000)
        else:
            assert np.all(shaped[:, :, :-offset, 15 + j] == 0)
            assert np.all(shaped[:, :, :-offset, 23 + j] == 1)
            assert np.allclose(shaped[:, :, -offset:, 15 + j], offset / 1000)
    sampled, _, mask = design(probabilities, ids, temporal=True, sampled=True)
    assert np.array_equal(sampled, matrix[mask.ravel()]) and len(sampled) == 2 * GENERATORS * 42
    zero_model = {"variant": "temporal", "feature_names": design(probabilities, ids, True, False)[1],
        "coefficients": [0.] * 31, "center": [0.] * 31, "scale": [1.] * 31, "shrinkage": .1}
    corrected, _ = predict_residual(probabilities, ids, zero_model)
    assert np.array_equal(corrected, probabilities)
    small = sampled[:11, :3]
    labels = np.arange(11) % 2
    offset = np.linspace(-1, 1, 11)
    beta = np.array([.1, -.2, .3])
    _, analytic = objective(beta, small, labels, offset, .01)
    numerical = []
    for j in range(3):
        step = np.zeros(3)
        step[j] = 1e-6
        numerical.append((objective(beta + step, small, labels, offset, .01)[0] -
                          objective(beta - step, small, labels, offset, .01)[0]) / 2e-6)
    assert np.allclose(analytic, numerical, atol=1e-8, rtol=1e-5)


def evaluate(arrays, metadata, models, protocol):
    starts = pd.DatetimeIndex(pd.to_datetime(arrays["starttime"], utc=True))
    mask = starts >= pd.Timestamp("2022-01-07", tz="UTC")
    y = arrays["labels"][mask]
    p0 = branch(arrays)
    baseline = identity(arrays)
    variants = {"identity": baseline}
    change = {}
    for name, model in models.items():
        corrected, change[name] = predict_residual(p0, arrays["ids"], model)
        # Equivalent blend arithmetic, preserving exact identity when correction is zero.
        values = baseline + .5 * (corrected - p0).reshape(baseline.shape)
        direct = .5 * arrays["round3"] + .5 * corrected.reshape(baseline.shape)
        assert np.max(np.abs(values - direct)) <= 3e-16
        assert np.isfinite(values).all() and np.all((values >= 0) & (values <= 1))
        variants[name] = values
    columns = metadata["columns"]
    generators = [columns[g * HOURS].removeprefix("result_committed_").removesuffix("_t0") for g in range(GENERATORS)]
    score = official_metric()
    rows, quarters, transitions = [], [], []
    transition = transition_mask(y.reshape(-1, GENERATORS, HOURS), radius=2)
    for name, values in variants.items():
        p = values[mask]
        auc = float(roc_auc_score(y.ravel(), p.ravel()))
        solution, submission = pd.DataFrame(y, columns=columns), pd.DataFrame(p, columns=columns)
        for frame in (solution, submission):
            frame.insert(0, "Run No", arrays["ids"][mask])
        assert score(solution, submission) == auc
        clipped = np.clip(p, 1e-12, 1 - 1e-12)
        gen = {generator: float(roc_auc_score(y[:, g * HOURS:(g + 1) * HOURS].ravel(), p[:, g * HOURS:(g + 1) * HOURS].ravel()))
               for g, generator in enumerate(generators)}
        rows.append({"variant": name, "micro_roc_auc": auc,
            "mean_bce": float(-(y * np.log(clipped) + (1 - y) * np.log1p(-clipped)).mean()), "generator_auc": gen})
        pairs = PairDiagnostics(y, p)
        transitions.append({"variant": name, "near_transition_2h": pairs.subset(transition),
                            "outside_transition_2h": pairs.subset(~transition)})
        for quarter in range(1, 5):
            q = starts[mask].quarter == quarter
            quarters.append({"variant": name, "quarter": quarter, "cases": int(q.sum()),
                "micro_roc_auc": float(roc_auc_score(y[q].ravel(), p[q].ravel()))})
    for row in rows:
        row["delta_vs_identity"] = row["micro_roc_auc"] - rows[0]["micro_roc_auc"]
        row["delta_vs_same_hour"] = row["micro_roc_auc"] - rows[1]["micro_roc_auc"]
        row["generator_delta_vs_identity"] = {g: row["generator_auc"][g] - rows[0]["generator_auc"][g] for g in generators}
        row["generator_delta_vs_same_hour"] = {g: row["generator_auc"][g] - rows[1]["generator_auc"][g] for g in generators}
    bootstrap = []
    for reference in ("identity", "same_hour"):
        bootstrap.append({"comparison": f"temporal_minus_{reference}", "results": paired_block_bootstrap(
            y, variants[reference][mask], variants["temporal"][mask], block_lengths=(7, 14, 28),
            repetitions=protocol["bootstrap_repetitions"], seed=4173, starts=starts[mask])})
    output = ROOT / f"data/processed/round10_temporal_{metadata['mode']}_2022_predictions.npz"
    with output.open("xb") as handle:
        np.savez_compressed(handle, ids=arrays["ids"][mask], starttime=arrays["starttime"][mask], labels=y,
            columns=np.asarray(columns), identity=variants["identity"][mask], same_hour=variants["same_hour"][mask], temporal=variants["temporal"][mask],
            protocol_sha256=np.asarray(digest(PROTOCOL)))
    return {"year": 2022, "mode": metadata["mode"], "subset": "jan7_onward", "cases": int(mask.sum()),
        "results": rows, "quarters": quarters, "transition_diagnostics": transitions,
        "correction_sizes": change, "paired_block_bootstrap": bootstrap,
        "saved_predictions": str(output.relative_to(ROOT)), "prediction_sha256": digest(output), "official_metric_verified": True}


def main():
    start = time.monotonic()
    protocol = json.loads(PROTOCOL.read_text())
    assert protocol["l2_penalty"] == .01 and protocol["shrinkage"] == .1
    assert protocol["script_sha256"] == digest(__file__)
    assert protocol["bootstrap_repetitions"] == 200
    self_checks()
    arrays, metadata = load_cache(ROOT / "data/processed/round6_native_2021_baseline.npz", 2021, "native")
    starts = pd.DatetimeIndex(pd.to_datetime(arrays["starttime"], utc=True))
    keep = starts <= pd.Timestamp("2021-12-25", tz="UTC")
    assert keep.sum() == 359
    probabilities = branch(arrays)[keep]
    labels = arrays["labels"][keep].reshape(probabilities.shape)
    models = {}
    model_paths = {}
    for name, temporal in (("same_hour", False), ("temporal", True)):
        model = fit_residual(probabilities, labels, arrays["ids"][keep], temporal, protocol)
        assert model["training_rows"] == 211092
        model.update({"protocol_sha256": digest(PROTOCOL), "training_first_start": str(starts[keep][0]),
            "training_last_start": str(starts[keep][-1]), "training_cases": int(keep.sum()),
            "cache_sha256": digest(ROOT / "data/processed/round6_native_2021_baseline.npz")})
        path = ROOT / f"models/round10_temporal_{name}.json"
        write_json(path, model)
        models[name], model_paths[str(path.relative_to(ROOT))] = model, digest(path)
        print(json.dumps({"fitted": name, "iterations": model["optimizer_iterations"], "seconds": model["fit_seconds"]}), flush=True)
    evaluations = []
    for mode in ("native", "frozen"):
        arrays, metadata = load_cache(ROOT / f"data/processed/round6_{mode}_2022_baseline.npz", 2022, mode)
        evaluation = evaluate(arrays, metadata, models, protocol)
        evaluations.append(evaluation)
        print(json.dumps({"mode": mode, "results": [{k: row[k] for k in ("variant", "micro_roc_auc", "delta_vs_identity", "delta_vs_same_hour")} for row in evaluation["results"]]}), flush=True)
    gains = all(evaluation["results"][2]["delta_vs_identity"] > 0 and evaluation["results"][2]["delta_vs_same_hour"] > 0 for evaluation in evaluations)
    reliable = gains and all(result["percentile_interval_95"][0] > 0 for evaluation in evaluations
                            for comparison in evaluation["paired_block_bootstrap"] for result in comparison["results"])
    sources = {str(PROTOCOL.relative_to(ROOT)): digest(PROTOCOL), str(Path(__file__).relative_to(ROOT)): digest(__file__), **model_paths}
    for relative in ("src/round2_diagnostics.py", "src/round6_evaluate.py", "evaluation/kaggle_metric.py",
                     "data/processed/round6_native_2021_baseline.npz", "data/processed/round6_native_2022_baseline.npz", "data/processed/round6_frozen_2022_baseline.npz"):
        sources[relative] = digest(ROOT / relative)
    write_json(ROOT / "reports/round10_temporal_results.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.monotonic() - start,
        "protocol": protocol, "self_checks_passed": True, "new_base_fits": 0, "meta_fits": 2, "threads": 1,
        "model_paths": model_paths, "evaluations": evaluations,
        "positive_delta_against_both_controls_both_scenarios": gains,
        "all_predeclared_block_intervals_above_zero": reliable, "source_sha256": sources,
        "limitations": "Exploratory cached development pilot; 2021/2022 recipes were selected using historical labels. Native/frozen2022 share labels and are not independent replications. No honest-OOF or clean-holdout claim. Training2021 metrics are not validation. This is a shared linear temporal-logit correction, not a nonlinear input-conditioned stacker. Bootstrap is conditional on fixed reused predictions and does not remove selection bias. No competition predictions or submission created."})


if __name__ == "__main__":
    main()
