"""Fit a frozen condition-dependent blend using chronological OOF experts only.

The gate estimates conditional ON probability and projects it onto the interval
between the two experts. This minimizes conditional BCE over that interval if the
probability estimate is accurate; AUC performance is measured separately.
"""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from features import HOURS, ROOT, feeding_reservoirs, load_data


FEATURE_NAMES = [
    "generator_id", "horizon_hour", "hour_of_day", "day_of_week", "month",
    "price_relative_to_week_scale", "terminal_local_water_value_over_scale",
    "normalized_price_minus_water_value", "week_mean_price", "week_price_scale",
    "initial_local_fill_fraction", "week_local_natural_inflow_capacity_fraction",
    "provided_river_min_flow_shortfall_m3_s", "xgboost_probability",
    "cnn_probability", "absolute_expert_disagreement",
]
PROFILES = ["price", "sampling", "regularized", "deep_regularized", "recency"]
FIXED_PARAMS = {
    "n_estimators": 300, "learning_rate": .03, "num_leaves": 8, "max_depth": 3,
    "min_child_samples": 5000, "reg_lambda": 10., "n_jobs": 4,
    "random_state": 42, "verbosity": -1,
}


def validate_config(config):
    """Protect the predeclared experiment from accidental per-year tuning."""
    assert config["fold_years"] == [2020, 2021, 2022]
    assert config["profiles"] == PROFILES
    assert config["xgboost_trees"] == 600 and config["cnn_epochs"] == 20
    assert config["cnn_seed"] == 42
    assert config["week_price_scale_floor"] == 1. and config["fixed_cnn_weight"] == .2
    assert config["tie_cnn_weight_for_metadata"] == .2
    assert config["model"] == FIXED_PARAMS
    assert len(FEATURE_NAMES) <= 16


def validate_order(cases, generators, columns):
    assert cases["Run No"].is_unique, "Duplicate case IDs."
    assert cases["starttime"].notna().all()
    assert len(generators) == len(set(generators))
    assert columns == [f"result_committed_{generator}_t{hour}"
                       for generator in generators for hour in range(HOURS)], "Generator/hour columns differ."


def purged_mask(cases, evaluation_year):
    """A training output horizon must finish before evaluation cases start."""
    if evaluation_year is None:
        return np.ones(len(cases), dtype=bool)
    boundary = pd.Timestamp(f"{evaluation_year}-01-01", tz="UTC")
    return (cases["starttime"] + pd.Timedelta(hours=HOURS) <= boundary).to_numpy()


def gate_target(labels, xgb, cnn):
    labels = np.asarray(labels)
    assert labels.shape == xgb.shape == cnn.shape
    assert np.isin(labels, [0, 1]).all()
    # Direct UC labels avoid treating an expert-selection probability as a blend.
    return labels.astype("uint8"), np.abs(cnn - xgb)


def convex_blend(xgb, cnn, weight):
    xgb, cnn, weight = np.broadcast_arrays(xgb, cnn, weight)
    for values in (xgb, cnn, weight):
        assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
    prediction = (1 - weight) * xgb + weight * cnn
    assert np.isfinite(prediction).all()
    assert ((prediction >= 0) & (prediction <= 1)).all()
    assert (prediction >= np.minimum(xgb, cnn) - 1e-12).all()
    assert (prediction <= np.maximum(xgb, cnn) + 1e-12).all()
    return prediction


def project_gate_probability(xgb, cnn, conditional_probability, tie_weight):
    """Project q onto the expert interval, then recover its equivalent CNN weight."""
    xgb, cnn, conditional_probability = np.broadcast_arrays(xgb, cnn, conditional_probability)
    for values in (xgb, cnn, conditional_probability):
        assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
    prediction = np.clip(conditional_probability, np.minimum(xgb, cnn), np.maximum(xgb, cnn))
    disagreement = cnn - xgb
    cnn_weight = np.full_like(prediction, tie_weight, dtype="float64")
    np.divide(prediction - xgb, disagreement, out=cnn_weight, where=disagreement != 0)
    assert np.isfinite(cnn_weight).all() and ((cnn_weight >= 0) & (cnn_weight <= 1)).all()
    reconstructed = convex_blend(xgb, cnn, cnn_weight)
    assert np.allclose(reconstructed, prediction, rtol=0, atol=1e-12)
    assert np.array_equal(prediction[disagreement == 0], xgb[disagreement == 0])
    return prediction, cnn_weight


def build_gate_features(cases, generators, inputs, topology, xgb, cnn, scale_floor):
    """Construct a compact array directly in case -> generator -> hour order.

    Storage is looked up only at case start; water value only at start + 168 h.
    The complete price/inflow/constraint horizon is supplied scheduling input.
    No historical future storage, commitment labels, or run IDs enter features.
    """
    starts = pd.DatetimeIndex(cases["starttime"])
    case_count, generator_count = len(cases), len(generators)
    horizon = np.arange(HOURS)
    timestamps = pd.DatetimeIndex(np.repeat(starts.array, HOURS))
    timestamps += pd.to_timedelta(np.tile(horizon, case_count), unit="h")
    price = inputs["price"].loc[timestamps].iloc[:, 0].to_numpy(dtype="float64").reshape(case_count, HOURS)
    week_mean = price.mean(axis=1)
    scale = np.maximum(price.std(axis=1), scale_floor)
    shortfall = np.maximum(
        inputs["min_flow"].loc[timestamps, "r_Vest_Vassdraget"].to_numpy()
        - inputs["inflow"].loc[timestamps, "r_Vest_Vassdraget"].to_numpy(), 0
    ).reshape(case_count, HOURS)
    shape = (case_count, generator_count, HOURS)
    assert xgb.shape == cnn.shape == (case_count * generator_count * HOURS,)
    features = np.empty((*shape, len(FEATURE_NAMES)), dtype="float32")
    features[..., 0] = np.arange(generator_count)[None, :, None]
    features[..., 1] = horizon[None, None, :]
    for index, values in [(2, timestamps.hour), (3, timestamps.dayofweek), (4, timestamps.month)]:
        features[..., index] = np.asarray(values).reshape(case_count, 1, HOURS)
    features[..., 5] = ((price - week_mean[:, None]) / scale[:, None])[:, None, :]
    features[..., 8] = week_mean[:, None, None]
    features[..., 9] = scale[:, None, None]
    features[..., 12] = shortfall[:, None, :]
    for index, generator in enumerate(generators):
        local = feeding_reservoirs(topology, generator.rsplit("_G", 1)[0])
        assert local, f"No local feeding reservoirs for {generator}."
        capacity = sum(topology["model"]["reservoir"][name]["max_vol"] for name in local)
        assert capacity > 0
        water_value = inputs["water_value"].loc[starts + pd.Timedelta(hours=HOURS), local].mean(
            axis=1, skipna=False).to_numpy()
        volume = inputs["volume"].loc[starts, local].sum(axis=1, skipna=False).to_numpy()
        natural_inflow = inputs["inflow"].loc[timestamps, local].sum(
            axis=1, skipna=False).to_numpy().reshape(case_count, HOURS)
        features[:, index, :, 6] = (water_value / scale)[:, None]
        features[:, index, :, 7] = (price - water_value[:, None]) / scale[:, None]
        features[:, index, :, 10] = (volume / capacity)[:, None]
        features[:, index, :, 11] = (natural_inflow.sum(axis=1) * 3600 / 1_000_000 / capacity)[:, None]
    features[..., 13] = xgb.reshape(shape)
    features[..., 14] = cnn.reshape(shape)
    features[..., 15] = np.abs(cnn - xgb).reshape(shape)
    assert np.isfinite(features).all(), "Missing or invalid supplied gate inputs."
    # Check every case/generator at first and last horizon hours before flattening.
    assert (features[:, :, 0, 0] == np.arange(generator_count)[None, :]).all()
    assert (features[:, :, -1, 1] == HOURS - 1).all()
    assert np.array_equal(features[..., 13].ravel(), xgb.astype("float32"))
    assert np.array_equal(features[..., 14].ravel(), cnn.astype("float32"))
    return features.reshape(-1, len(FEATURE_NAMES))


def load_experts(suffix, config, truth, template, columns, test_ids):
    """Validate source budgets, fit windows, case order and unchanged protocols."""
    members = config["profiles"] + ["cnn"]
    if suffix == "final":
        expected_ids = pd.Index(template["Run No"], name="Run No")
    else:
        expected_ids = truth.loc[truth["starttime"].dt.year == int(suffix)].sort_values("starttime").index
    assert expected_ids.is_unique and len(expected_ids)
    xgb_sum, cnn = None, None
    sources, protocols = {}, {}
    for member in members:
        prediction_path = ROOT / f"data/processed/stacking_{member}_{suffix}_predictions.csv"
        report_path = ROOT / f"reports/stacking_{member}_{suffix}_run.json"
        report_bytes = report_path.read_bytes()
        source = json.loads(report_bytes)
        assert source["mode"] == ("final" if suffix == "final" else "fold")
        assert source["fold_year"] == (None if suffix == "final" else int(suffix))
        assert source["prediction_run_ids"] == expected_ids.tolist()
        assert source["prediction_cases"] == len(expected_ids)
        fit_ids = pd.Index(source["training_run_ids"])
        assert fit_ids.is_unique and len(fit_ids) == source["training_cases"]
        assert not fit_ids.isin(test_ids).any(), "Competition test IDs in source fitting."
        assert not fit_ids.isin(expected_ids).any(), "Source expert fits its prediction cases."
        fit_starts = truth.loc[fit_ids, "starttime"]
        assert str(fit_starts.min()) == source["first_training_start"]
        assert str(fit_starts.max()) == source["last_training_start"]
        assert str(fit_starts.max() + pd.Timedelta(hours=HOURS)) == source["last_training_end"]
        if suffix != "final":
            boundary = pd.Timestamp(f"{suffix}-01-01", tz="UTC")
            assert (fit_starts + pd.Timedelta(hours=HOURS)).max() <= boundary
            expected_fit = truth.loc[truth["starttime"] + pd.Timedelta(hours=HOURS) <= boundary].sort_values("starttime")
            assert fit_ids.tolist() == expected_fit.index.tolist()
        else:
            assert fit_ids.tolist() == truth.sort_values("starttime").index.tolist()
        if member == "cnn":
            assert source["model_type"] == "cnn"
            assert source["epochs"] == source["config"]["epochs"] == config["cnn_epochs"]
            assert source["config"]["seed"] == config["cnn_seed"]
            assert not source["config"].get("stable_scaling", False)
            assert not source["config"].get("use_weekly_normalization", False)
            protocols[member] = {"config": source["config"], "feature_names": source["feature_names"]}
        else:
            assert source["model_type"] == "xgboost"
            assert source["trees"] == source["model_parameters"]["n_estimators"] == config["xgboost_trees"]
            protocols[member] = {"parameters": source["model_parameters"],
                                 "feature_options": source["feature_options"], "feature_names": source["feature_names"]}
        frame = pd.read_csv(prediction_path).set_index("Run No")
        assert frame.index.is_unique and frame.index.equals(expected_ids), f"Prediction cases/order differ: {member}/{suffix}"
        assert frame.columns.tolist() == columns, f"Prediction generator/hour order differs: {member}/{suffix}"
        values = frame.to_numpy(dtype="float64").ravel()
        assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
        if member == "cnn":
            cnn = values
        elif xgb_sum is None:
            xgb_sum = values
        else:
            xgb_sum += values
        sources[member] = {"predictions": str(prediction_path), "report": str(report_path),
                           "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
                           "model": source["model"], "training_cases": len(fit_ids),
                           "last_training_end": source["last_training_end"],
                           "budget": {"epochs" if member == "cnn" else "trees":
                                      source["epochs"] if member == "cnn" else source["trees"]}}
    return expected_ids, xgb_sum / len(config["profiles"]), cnn, sources, protocols


def safe_auc(labels, probabilities):
    return float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else None


def metrics(labels, probabilities, cases, generators):
    shape = (len(cases), len(generators), HOURS)
    labels, probabilities = labels.reshape(shape), probabilities.reshape(shape)
    months = cases["starttime"].dt.month.to_numpy()
    return {
        "micro_roc_auc": safe_auc(labels.ravel(), probabilities.ravel()),
        "generator_auc": {g: safe_auc(labels[:, i].ravel(), probabilities[:, i].ravel())
                          for i, g in enumerate(generators)},
        "month_auc": {str(month): safe_auc(labels[months == month].ravel(), probabilities[months == month].ravel())
                      for month in sorted(set(months))},
        "month_generator_auc": {
            str(month): {g: safe_auc(labels[months == month, i].ravel(), probabilities[months == month, i].ravel())
                         for i, g in enumerate(generators)} for month in sorted(set(months))},
        "month_grouping": "Case-start month; each case contributes all 168 supplied horizon hours.",
        "single_class_auc": "null when a subgroup contains only one label class",
    }


def distribution(values):
    points = [0, .05, .25, .5, .75, .95, 1]
    return {"mean": float(values.mean()), "std": float(values.std()),
            "quantiles": dict(zip([str(p) for p in points], np.quantile(values, points).tolist()))}


def weight_distribution(weight, cases, generators):
    values = weight.reshape(len(cases), len(generators), HOURS)
    months = cases["starttime"].dt.month.to_numpy()
    return {"overall": distribution(weight),
            "generator": {g: distribution(values[:, i].ravel()) for i, g in enumerate(generators)},
            "month": {str(m): distribution(values[months == m].ravel()) for m in sorted(set(months))}}


def current_reference(year):
    path = ROOT / "reports/cnn_five_seeds_comparison.json"
    report = json.loads(path.read_text())
    selected = next(row for row in report["results"] if row["year"] == year and row["variant"] == "five_seed_mean")
    return {"report": str(path), "year": year, "micro_roc_auc": selected["blend"]["micro_roc_auc"],
            "note": "Different base budgets: selected XGBoost checkpoints plus five separately selected CNN checkpoints; descriptive reference only. Gate uses five fixed 600-tree experts and one fixed 20-epoch seed-42 CNN."}


def run_checks():
    """Small no-fit checks for UC targets, projection, purge and array ordering."""
    labels = np.array([1, 0, 1, 0, 1], dtype="uint8")
    xgb = np.array([.1, .1, .9, .9, .5])
    cnn = np.array([.9, .9, .1, .1, .5])
    target, weight = gate_target(labels, xgb, cnn)
    assert np.array_equal(target, labels) and weight[-1] == 0
    for w in [np.zeros(5), np.ones(5), np.full(5, .37)]:
        p = convex_blend(xgb, cnn, w)
        assert (p >= np.minimum(xgb, cnn)).all() and (p <= np.maximum(xgb, cnn)).all()
    q = np.array([0., 1., .37, .83, .99])
    p, w = project_gate_probability(xgb, cnn, q, .2)
    assert np.allclose(p, [.1, .9, .37, .83, .5])
    assert w[-1] == .2 and np.allclose(convex_blend(xgb, cnn, w), p)
    cases = pd.DataFrame({"Run No": [100, 101, 102], "starttime": pd.to_datetime(
        ["2020-12-25", "2020-12-26", "2020-12-31"], utc=True)})
    assert purged_mask(cases, 2021).tolist() == [True, False, False]
    assert purged_mask(cases, None).all()
    dates = pd.date_range("2020-12-25", periods=7 * 24 + 24 + 1, freq="h", tz="UTC")
    starts = dates[[0, 24]]
    cases = pd.DataFrame({"Run No": [91, 17], "starttime": starts})
    generators = ["A_G1", "B_G1"]
    columns = [f"result_committed_{g}_t{h}" for g in generators for h in range(HOURS)]
    validate_order(cases, generators, columns)
    inputs = {
        "price": pd.DataFrame({"price": np.arange(len(dates)) + 1}, index=dates),
        "volume": pd.DataFrame({"A": 20., "B": 40.}, index=starts),
        "water_value": pd.DataFrame({"A": 5., "B": 9.}, index=starts + pd.Timedelta(hours=HOURS)),
        "inflow": pd.DataFrame({"A": 1., "B": 2., "r_Vest_Vassdraget": 3.}, index=dates),
        "min_flow": pd.DataFrame({"r_Vest_Vassdraget": 7.}, index=dates),
    }
    topology = {"model": {"reservoir": {"A": {"max_vol": 100.}, "B": {"max_vol": 200.}}},
                "connections": [{"from": name, "from_type": "reservoir", "to": name,
                                 "to_type": "plant"} for name in ["A", "B"]]}
    xgb = np.arange(2 * 2 * HOURS, dtype="float64") / (2 * 2 * HOURS)
    cnn = 1 - xgb
    matrix = build_gate_features(cases, generators, inputs, topology, xgb, cnn, 1.)
    assert matrix.shape == (2 * 2 * HOURS, len(FEATURE_NAMES))
    for case in range(2):
        for generator in range(2):
            for hour in [0, 1, HOURS - 1]:
                row = (case * 2 + generator) * HOURS + hour
                assert matrix[row, 0] == generator and matrix[row, 1] == hour
                assert matrix[row, 13] == np.float32(xgb[row])
                assert np.isclose(matrix[row, 6], [5., 9.][generator] / matrix[row, 9])
                assert matrix[row, 12] == 4.
    print("Gate no-fit checks passed: UC targets, interval projection/ties, convex bounds, purge boundary and supplied-input row ordering.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/conditional_gate.json"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--year", type=int, choices=[2021, 2022])
    mode.add_argument("--final", action="store_true")
    mode.add_argument("--check", action="store_true", help="Run small checks without fitting or creating artifacts.")
    args = parser.parse_args()
    config_bytes = Path(args.config).read_bytes()
    config = json.loads(config_bytes)
    validate_config(config)
    if args.check:
        run_checks()
        return
    suffix = "final" if args.final else str(args.year)
    model_path = ROOT / f"models/conditional_gate_{suffix}.joblib"
    report_path = ROOT / f"reports/conditional_gate_{suffix}_metrics.json"
    output = ROOT / ("submissions/submission_conditional_gate.csv" if args.final else
                     f"data/processed/conditional_gate_{suffix}_validation_predictions.csv")
    for path in (model_path, report_path, output):
        assert not path.exists(), f"Artifact already exists: {path}; retain existing runs."
    cases, columns, generators, inputs, topology = load_data(use_extended=True)
    validate_order(cases, generators, columns)
    truth = cases.set_index("Run No")
    template = pd.read_csv(ROOT / "data/raw/sample_submission.csv")
    mapping = pd.read_csv(ROOT / "data/raw/prediction_mapping.csv")
    mapping["starttime"] = pd.to_datetime(mapping["starttime"], utc=True)
    assert mapping["Run No"].is_unique and template["Run No"].is_unique
    assert set(mapping["Run No"]) == set(template["Run No"])
    test_ids = mapping["Run No"]
    assert not truth.index.isin(test_ids).any(), "Competition test IDs must stay out of gate fitting."
    fit_years = [year for year in config["fold_years"] if args.final or year < args.year]
    fit_ids, purged_ids, matrices, targets, weights = [], [], [], [], []
    sources, protocol = {}, None
    for year in fit_years:
        ids, xgb, cnn, year_sources, year_protocol = load_experts(
            str(year), config, truth, template, columns, test_ids)
        assert protocol is None or protocol == year_protocol, "Expert protocols changed between OOF years."
        protocol = year_protocol
        sources[str(year)] = year_sources
        context = truth.loc[ids, ["starttime"]].reset_index()
        validate_order(context, generators, columns)
        keep = purged_mask(context, None if args.final else args.year)
        assert keep.any()
        kept_context = context.loc[keep].copy()
        fit_ids.extend(ids[keep].tolist())
        purged_ids.extend(ids[~keep].tolist())
        outputs = len(generators) * HOURS
        xgb = xgb.reshape(len(ids), outputs)[keep].ravel()
        cnn = cnn.reshape(len(ids), outputs)[keep].ravel()
        # Construct supplied context before reading this fold's training labels.
        matrix = build_gate_features(kept_context, generators, inputs, topology,
                                     xgb, cnn, config["week_price_scale_floor"])
        labels = truth.loc[ids[keep], columns].to_numpy().ravel()
        target, weight = gate_target(labels, xgb, cnn)
        matrices.append(matrix)
        targets.append(target)
        weights.append(weight)
    assert len(set(fit_ids)) == len(fit_ids) and not set(fit_ids).intersection(test_ids)
    assert len(purged_ids) == (0 if args.final else 6), "Expected the six boundary-crossing daily cases to be purged."
    assert not set(fit_ids).intersection(purged_ids)
    matrix = np.concatenate(matrices)
    target, weight = np.concatenate(targets), np.concatenate(weights)
    del matrices, targets, weights
    effective = weight > 0
    assert set(target[effective]) == {0, 1}, "Need both effective UC target classes."
    fit_summary = {"outputs": len(target), "zero_weight_outputs": int((~effective).sum()),
                   "effective_sample_weight_sum": float(weight.sum()),
                   "weighted_uc_on_rate": float(np.average(target, weights=weight))}
    print(f"Gate {suffix}: {len(fit_ids)} OOF cases, {len(target):,} rows, {len(purged_ids)} purged; 16 features, 300 fixed trees.", flush=True)
    import lightgbm as lgb

    model = lgb.LGBMClassifier(objective="binary", **config["model"])
    with threadpool_limits(limits=4):
        model.fit(matrix, target, sample_weight=weight,
                  categorical_feature=[0], feature_name=FEATURE_NAMES)
    assert model.classes_.tolist() == [0, 1]
    assert model.n_estimators_ == config["model"]["n_estimators"]
    assert model.feature_name_ == FEATURE_NAMES
    del matrix, target, weight, effective
    # Seal the gate before loading evaluation expert predictions or scoring labels.
    checkpoint = {"model": model, "config": config, "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                  "feature_names": FEATURE_NAMES, "generators": generators,
                  "fit_run_ids": fit_ids, "purged_run_ids": purged_ids,
                  "source_protocols": protocol, "fit_sources": sources}
    with model_path.open("xb") as stream:
        joblib.dump(checkpoint, stream, compress=3)
    ids, xgb, cnn, prediction_sources, prediction_protocol = load_experts(
        suffix, config, truth, template, columns, test_ids)
    assert protocol == prediction_protocol, "Prediction experts do not match OOF fitting protocols."
    assert not set(fit_ids).intersection(ids), "Evaluation IDs entered gate fitting."
    prediction_cases = (mapping.set_index("Run No").loc[ids, ["starttime"]] if args.final else
                        truth.loc[ids, ["starttime"]]).reset_index()
    validate_order(prediction_cases, generators, columns)
    if not args.final:
        assert (truth.loc[fit_ids, "starttime"] + pd.Timedelta(hours=HOURS)).max() <= prediction_cases["starttime"].min()
    matrix = build_gate_features(prediction_cases, generators, inputs, topology,
                                 xgb, cnn, config["week_price_scale_floor"])
    with threadpool_limits(limits=4):
        conditional_probability = model.predict_proba(matrix)[:, 1]
    del matrix
    predictions, cnn_weight = project_gate_probability(xgb, cnn, conditional_probability,
                                                       config["tie_cnn_weight_for_metadata"])
    frame = pd.DataFrame(predictions.reshape(len(ids), len(columns)), index=ids, columns=columns).reset_index()
    assert frame.columns.tolist() == template.columns.tolist()
    if args.final:
        assert frame["Run No"].equals(template["Run No"])
    frame.to_csv(output, index=False, mode="x")
    report = {
        "config": config, "config_sha256": checkpoint["config_sha256"],
        "selected_parameters": config["model"], "objective_note": config["objective"],
        "fit_years": fit_years, "fit_cases": len(fit_ids), "fit_outputs": fit_summary["outputs"],
        "fit_run_ids": fit_ids, "purged_run_ids": purged_ids, "fit_target_summary": fit_summary,
        "features": FEATURE_NAMES, "feature_order": "case -> generator -> horizon hour",
        "feature_source_note": "Only case-start storage, supplied complete-week prices/natural inflows/minimum flows, terminal local water value at start+168h, calendar, generator and OOF expert probabilities. No run IDs, future historical storage or UC decisions as features.",
        "fit_sources": sources, "prediction_sources": prediction_sources,
        "source_protocols": protocol, "prediction_cases": len(ids),
        "cnn_weight_distribution": weight_distribution(cnn_weight, prediction_cases, generators),
        "estimated_conditional_on_probability_distribution": distribution(conditional_probability),
        "feature_importance_gain": dict(zip(FEATURE_NAMES, model.booster_.feature_importance(importance_type="gain").tolist())),
        "model_saved_before_evaluation": True, "fitted_trees": model.n_estimators_,
        "model": str(model_path), "predictions": str(output),
        "evaluation_note": config["evaluation_note"],
    }
    if not args.final:
        labels = truth.loc[ids, columns].to_numpy().ravel()
        baseline = convex_blend(xgb, cnn, config["fixed_cnn_weight"])
        report["evaluation"] = {
            "conditional_gate": metrics(labels, predictions, prediction_cases, generators),
            "unprojected_probability_diagnostic": metrics(labels, conditional_probability, prediction_cases, generators),
            "fixed_80_20_same_bases": metrics(labels, baseline, prediction_cases, generators),
            "equal_weight_xgboost_same_bases": metrics(labels, xgb, prediction_cases, generators),
            "cnn_same_bases": metrics(labels, cnn, prediction_cases, generators),
        }
        reference = current_reference(args.year)
        report["current_five_seed_reference_different_budgets"] = reference
        auc = report["evaluation"]["conditional_gate"]["micro_roc_auc"]
        baseline_auc = report["evaluation"]["fixed_80_20_same_bases"]["micro_roc_auc"]
        report["delta_vs_matched_80_20"] = auc - baseline_auc
        report["delta_vs_current_five_seed_reference_different_budgets"] = auc - reference["micro_roc_auc"]
        if args.year == 2022:
            lio = generators.index("Lio_G1")
            march = prediction_cases["starttime"].dt.month.to_numpy() == 3
            report["lio_march_2022_diagnosis"] = {
                "case_count": int(march.sum()), "grouping": "Case-start month == March 2022",
                "auc": {name: row["month_generator_auc"]["3"]["Lio_G1"]
                        for name, row in report["evaluation"].items()},
                "cnn_weight": distribution(cnn_weight.reshape(len(ids), len(generators), HOURS)[march, lio].ravel()),
            }
        print(f"Gate AUC {auc:.6f}; matched 80/20 {baseline_auc:.6f}; delta {auc-baseline_auc:+.6f}; current five-seed reference {reference['micro_roc_auc']:.6f} (different budgets).", flush=True)
    with report_path.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(f"Saved {len(ids)} cases to {output}", flush=True)


if __name__ == "__main__":
    main()
