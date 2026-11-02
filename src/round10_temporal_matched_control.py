"""One prespecified boundary-matched control; preserves the original temporal run."""

import round10_temporal as pilot  # Sets BLAS/OMP to one thread before numeric imports.
import json
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

ROOT = pilot.ROOT
AMENDMENT = ROOT / "reports/round10_temporal_control_amendment.json"
ACTIVE = [*range(15), *range(23, 31)]


def fit_matched(protocol):
    arrays, _ = pilot.load_cache(ROOT / "data/processed/round6_native_2021_baseline.npz", 2021, "native")
    starts = pd.DatetimeIndex(pd.to_datetime(arrays["starttime"], utc=True))
    keep = starts <= pd.Timestamp("2021-12-25", tz="UTC")
    assert keep.sum() == 359
    probabilities = pilot.branch(arrays)[keep]
    labels = arrays["labels"][keep].reshape(probabilities.shape)
    # Reuse exactly the temporal builder and indicator encoding/order.
    full, names, choose = pilot.design(probabilities, arrays["ids"][keep], temporal=True, sampled=True)
    matrix = full[:, ACTIVE].copy()
    assert matrix.shape == (211092, 23)
    center, scale = np.zeros(23), np.ones(23)
    center[0], scale[0] = matrix[:, 0].mean(), matrix[:, 0].std()
    matrix = (matrix - center) / scale
    target, offset = labels[choose].astype(float), pilot.logit(probabilities[choose])
    start = time.monotonic()
    result = minimize(pilot.objective, np.zeros(23), args=(matrix, target, offset, protocol["l2_penalty"]),
        method="L-BFGS-B", jac=True, options={"maxiter": 200, "gtol": 1e-7, "ftol": 1e-12, "maxls": 30})
    assert result.success, result.message
    value, gradient = pilot.objective(result.x, matrix, target, offset, protocol["l2_penalty"])
    # Embed the23 active coefficients in the temporal layout with exactzero neighbour coefficients.
    coefficients, full_center, full_scale = np.zeros(31), np.zeros(31), np.ones(31)
    coefficients[ACTIVE], full_center[ACTIVE], full_scale[ACTIVE] = result.x, center, scale
    assert np.array_equal(coefficients[15:23], np.zeros(8))
    return {"variant": "temporal", "control_name": "same_hour_boundary_matched", "active_feature_indices": ACTIVE,
        "feature_names": names, "active_feature_names": [names[i] for i in ACTIVE],
        "coefficients": coefficients.tolist(), "center": full_center.tolist(), "scale": full_scale.tolist(),
        "l2_penalty": protocol["l2_penalty"], "shrinkage": protocol["shrinkage"],
        "training_cases": 359, "training_rows": 211092, "training_first_start": str(starts[keep][0]),
        "training_last_start": str(starts[keep][-1]), "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message), "optimizer_iterations": int(result.nit),
        "penalized_training_objective": value, "gradient_infinity_norm": float(np.abs(gradient).max()),
        "fit_seconds": time.monotonic() - start, "amendment_sha256": pilot.digest(AMENDMENT),
        "original_protocol_sha256": pilot.digest(pilot.PROTOCOL),
        "cache_sha256": pilot.digest(ROOT / "data/processed/round6_native_2021_baseline.npz")}


def main():
    start = time.monotonic()
    # The original run must be complete, preserving CPU1 and immutable original results.
    original = json.loads((ROOT / "reports/round10_temporal_results.json").read_text())
    amendment = json.loads(AMENDMENT.read_text())
    protocol = json.loads(pilot.PROTOCOL.read_text())
    assert amendment["original_protocol_sha256"] == pilot.digest(pilot.PROTOCOL)
    model = fit_matched(protocol)
    model_path = ROOT / "models/round10_temporal_same_hour_boundary_matched.json"
    pilot.write_json(model_path, model)
    print(json.dumps({"fitted": "boundary_matched_control", "seconds": model["fit_seconds"]}), flush=True)
    evaluations, sources = [], {}
    for mode in ("native", "frozen"):
        cache_path = ROOT / f"data/processed/round6_{mode}_2022_baseline.npz"
        arrays, metadata = pilot.load_cache(cache_path, 2022, mode)
        starts = pd.DatetimeIndex(pd.to_datetime(arrays["starttime"], utc=True))
        mask = starts >= pd.Timestamp("2022-01-07", tz="UTC")
        probabilities = pilot.branch(arrays)
        corrected, sizes = pilot.predict_residual(probabilities, arrays["ids"], model)
        baseline = pilot.identity(arrays)
        prediction = (baseline + .5 * (corrected - probabilities).reshape(baseline.shape))[mask]
        old_path = ROOT / f"data/processed/round10_temporal_{mode}_2022_predictions.npz"
        with np.load(old_path, allow_pickle=False) as saved:
            assert np.array_equal(saved["ids"], arrays["ids"][mask])
            assert np.array_equal(saved["identity"], baseline[mask])
            assert np.array_equal(saved["labels"], arrays["labels"][mask])
            temporal = saved["temporal"].copy()
            same_hour = saved["same_hour"].copy()
        y = arrays["labels"][mask]
        columns = metadata["columns"]
        generators = [columns[g * 168].removeprefix("result_committed_").removesuffix("_t0") for g in range(14)]
        auc = float(roc_auc_score(y.ravel(), prediction.ravel()))
        solution, submission = pd.DataFrame(y, columns=columns), pd.DataFrame(prediction, columns=columns)
        for frame in (solution, submission):
            frame.insert(0, "Run No", arrays["ids"][mask])
        assert pilot.official_metric()(solution, submission) == auc
        gen_rows, quarters = [], []
        for g, name in enumerate(generators):
            cols = slice(g * 168, (g + 1) * 168)
            yy, pp, tt = y[:, cols].ravel(), prediction[:, cols].ravel(), temporal[:, cols].ravel()
            match_auc = float(roc_auc_score(yy, pp))
            temporal_auc = float(roc_auc_score(yy, tt))
            gen_rows.append({"generator": name, "matched_control_auc": match_auc, "temporal_auc": temporal_auc,
                "temporal_minus_matched_control": temporal_auc - match_auc})
        for quarter in range(1, 5):
            choose = starts[mask].quarter == quarter
            before = float(roc_auc_score(y[choose].ravel(), prediction[choose].ravel()))
            after = float(roc_auc_score(y[choose].ravel(), temporal[choose].ravel()))
            quarters.append({"quarter": quarter, "cases": int(choose.sum()), "matched_control_auc": before,
                "temporal_auc": after, "temporal_minus_matched_control": after - before})
        bootstrap = pilot.paired_block_bootstrap(y, prediction, temporal, block_lengths=(7, 14, 28),
            repetitions=200, seed=4173, starts=starts[mask])
        near = pilot.transition_mask(y.reshape(-1, 14, 168), radius=2)
        pairs = pilot.PairDiagnostics(y, prediction)
        output = ROOT / f"data/processed/round10_temporal_{mode}_2022_matched_control_predictions.npz"
        with output.open("xb") as handle:
            np.savez_compressed(handle, ids=arrays["ids"][mask], starttime=arrays["starttime"][mask],
                columns=np.asarray(columns), matched_control=prediction,
                amendment_sha256=np.asarray(pilot.digest(AMENDMENT)))
        temporal_auc = float(roc_auc_score(y.ravel(), temporal.ravel()))
        row = {"mode": mode, "year": 2022, "subset": "jan7_onward", "cases": int(mask.sum()),
            "matched_control_micro_auc": auc, "temporal_micro_auc": temporal_auc,
            "temporal_minus_matched_control": temporal_auc - auc,
            "same_hour_micro_auc": float(roc_auc_score(y.ravel(), same_hour.ravel())),
            "identity_micro_auc": float(roc_auc_score(y.ravel(), baseline[mask].ravel())),
            "generators": gen_rows, "quarters": quarters, "paired_block_bootstrap": bootstrap,
            "near_transition_2h_matched_control": pairs.subset(near),
            "outside_transition_2h_matched_control": pairs.subset(~near), "correction_sizes": sizes,
            "prediction_path": str(output.relative_to(ROOT)), "prediction_sha256": pilot.digest(output),
            "official_metric_verified": True}
        evaluations.append(row)
        for path in (cache_path, old_path):
            sources[str(path.relative_to(ROOT))] = pilot.digest(path)
        print(json.dumps({k: row[k] for k in ("mode", "matched_control_micro_auc", "temporal_micro_auc", "temporal_minus_matched_control")}), flush=True)
    positive = original["positive_delta_against_both_controls_both_scenarios"] and all(e["temporal_minus_matched_control"] > 0 for e in evaluations)
    reliable = positive and original["all_predeclared_block_intervals_above_zero"] and all(b["percentile_interval_95"][0] > 0 for e in evaluations for b in e["paired_block_bootstrap"])
    for path in (AMENDMENT, pilot.PROTOCOL, model_path, ROOT / "src/round10_temporal.py",
                 ROOT / "src/round10_temporal_matched_control.py", ROOT / "reports/round10_temporal_results.json"):
        sources[str(path.relative_to(ROOT))] = pilot.digest(path)
    pilot.write_json(ROOT / "reports/round10_temporal_matched_control_results.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.monotonic() - start,
        "amendment": amendment, "evaluations": evaluations, "model_path": str(model_path.relative_to(ROOT)),
        "positive_against_all_three_controls_both_scenarios": positive,
        "all_original_and_matched_block_intervals_above_zero": reliable,
        "new_base_fits": 0, "additional_meta_fits": 1, "total_meta_fits_including_original": 3,
        "source_sha256": sources,
        "limitations": "The supplemental control repairs an independently identified boundary-feature confound. Original models and results are retained. Same reused-development and chronological-selection caveats apply; native/frozen2022 share labels. No hyperparameter search, full-data refit, competition prediction or submission."})


if __name__ == "__main__":
    main()
