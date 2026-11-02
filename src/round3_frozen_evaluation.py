"""Replay pre-2021 checkpoints and compare frozen ensembles on 2022. No fitting.

Run without arguments for the three sequential one-thread inference workers and
comparison. Workers use separate processes so Torch and XGBoost never coexist.
Existing artifacts are not overwritten; --compare reuses completed predictions.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from features import HOURS, ROOT, build_features, load_data, prediction_frame
from plant_blend import metrics, read_predictions

SOURCES = {
    "xgb_depth8": "round2_xgb_depth8_2021",
    "haukeli_specialist": "round2_xgb_haukeli_2021",
    "transformer_weighted": "round2_transformer_task_weights_2021",
}
OLD_2021 = {
    "xgb_rescaled": "goal_xgboost_rescaled_2021",
    "lightgbm_rescaled": "goal_lightgbm_rescaled_2021",
    "transformer": "transformer_rescaled_2021",
}
OLD_2022 = {
    "xgb_rescaled": "round2_frozen_xgb_2022",
    "lightgbm_rescaled": "round2_frozen_lgb_2022",
    "transformer": "round2_frozen_transformer_2022",
}
FEATURE_OPTIONS = (
    "use_extended", "use_system_context", "use_topology", "use_price_context",
    "use_water_context", "use_economic_context", "use_river_context",
    "use_inflow_profiles", "use_hydraulic_context", "use_water_budget",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def prediction_path(prefix):
    return ROOT / f"data/processed/{prefix}_validation_predictions.csv"


def worker(name):
    source = SOURCES[name]
    prefix = f"round3_frozen_{name}_2022"
    output = prediction_path(prefix)
    report_path = ROOT / f"reports/{prefix}_metrics.json"
    assert not output.exists() and not report_path.exists()
    source_report_path = ROOT / f"reports/{source}_metrics.json"
    source_report = read_json(source_report_path)
    config = source_report["config"]
    suffix = "pt" if name == "transformer_weighted" else "ubj"
    model_path = ROOT / f"models/{source}.{suffix}"
    source_prediction_path = prediction_path(source)
    protected = {str(path.relative_to(ROOT)): digest(path) for path in
                 (model_path, source_report_path, source_prediction_path)}
    cases, columns, generators, inputs, topology = load_data(use_extended=config.get("use_extended", False))
    boundary = pd.Timestamp(config["validation_start"], tz="UTC")
    assert boundary == pd.Timestamp("2021-01-01", tz="UTC")
    assert config["validation_end"] == "2022-01-01"
    train = cases.loc[cases.starttime + pd.Timedelta(hours=HOURS) <= boundary]
    selection = cases.loc[cases.starttime.dt.year == 2021].reset_index(drop=True)
    evaluation = cases.loc[cases.starttime.dt.year == 2022].reset_index(drop=True)
    assert len(train) == source_report["training_cases"] == 2186
    assert str(train.starttime.max()) == source_report["last_training_start"]
    assert len(selection) == len(evaluation) == 365
    assert not train["Run No"].isin(evaluation["Run No"]).any()
    assert (train.starttime + pd.Timedelta(hours=HOURS)).max() <= evaluation.starttime.min()
    competition_ids = pd.read_csv(ROOT / "data/raw/prediction_mapping.csv", usecols=["Run No"])["Run No"]
    assert not cases["Run No"].isin(competition_ids).any()
    options = {key: config.get(key, False) for key in FEATURE_OPTIONS}
    report = {"source": source, "source_hashes": protected, "source_config": config,
              "training_cases": len(train), "last_training_start": str(train.starttime.max()),
              "last_training_horizon_end": str((train.starttime + pd.Timedelta(hours=HOURS)).max()),
              "training_cohort_sha256_reconstructed": hashlib.sha256(train[["Run No", "starttime"]].to_csv(index=False).encode()).hexdigest(),
              "evaluation_cases": len(evaluation), "new_fitting": False, "scaler_refitted": False,
              "cpu_threads": 1, "prediction_path": str(output.relative_to(ROOT))}

    if name == "transformer_weighted":
        assert "xgboost" not in sys.modules
        import torch
        from temporal_cnn import predict, sequence_features
        from weekly_transformer import WeeklyTransformer
        torch.set_num_threads(1)
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        assert checkpoint["config"] == config
        assert checkpoint["generators"] == generators
        names = checkpoint["feature_names"]
        assert names == source_report["feature_names"]
        assert source_report["best_epoch"] == 28
        options.update({key: config.get(key, False) for key in ("use_weekly_normalization", "use_economic_rescaling")})
        mean, std = checkpoint["mean"].numpy(), checkpoint["std"].numpy()
        assert np.isfinite(mean).all() and np.isfinite(std).all() and (std > 0).all()
        model = WeeklyTransformer(len(names), len(generators), **config["model"])
        model.load_state_dict(checkpoint["state_dict"])
        report.update({"selected_epoch": 28, "feature_names": names,
                       "scaler_mean_sha256": hashlib.sha256(mean.tobytes()).hexdigest(),
                       "scaler_std_sha256": hashlib.sha256(std.tobytes()).hexdigest()})

        def predict_cases(subset):
            values, observed_names = sequence_features(subset, generators, inputs, topology, **options)
            assert observed_names == names
            tensor = torch.from_numpy((values - mean) / std)
            probabilities = predict(model, tensor, torch.device("cpu"), config["batch_size"])
            return prediction_frame(probabilities.transpose(1, 0, 2).ravel(), subset, columns, generators)
    else:
        assert "torch" not in sys.modules
        import xgboost as xgb
        from tree_preprocessing import rescale_tree_features
        from round2_tree_specialist import specialist_features
        model = xgb.XGBClassifier(n_jobs=1, enable_categorical=True)
        model.load_model(model_path)
        model.set_params(n_jobs=1)
        rounds = model.get_booster().num_boosted_rounds()
        assert rounds == source_report["best_iteration"] == (748 if name == "xgb_depth8" else 121)
        assert model.get_booster().feature_names == source_report["feature_names"]
        report.update({"trees": rounds, "feature_names": source_report["feature_names"]})

        def predict_cases(subset):
            if name == "haukeli_specialist":
                target = config["target_generator"]
                values = specialist_features(subset, target, inputs, topology, config)
                target_columns = source_report["output_columns"]
                assert len(target_columns) == HOURS and target == "Haukeli_G1"
                target_generators = [target]
            else:
                values = build_features(subset, generators, inputs, topology, **options)
                if config.get("use_economic_rescaling", False):
                    values = rescale_tree_features(values)
                values["generator_id"] = pd.Categorical(values["generator_id"], categories=range(len(generators)))
                target_columns, target_generators = columns, generators
            assert values.columns.tolist() == source_report["feature_names"]
            probabilities = model.predict_proba(values)[:, 1]
            return prediction_frame(probabilities, subset, target_columns, target_generators)

    replay_cases = selection.iloc[[0, 60, 120, 180, 240, 300, 364]].reset_index(drop=True)
    replay = predict_cases(replay_cases)
    replay_columns = replay.columns[1:].tolist()
    reference = pd.read_csv(source_prediction_path, float_precision="round_trip")
    assert reference.columns.tolist() == ["Run No", *columns]
    assert reference["Run No"].equals(selection["Run No"])
    cached = reference.set_index("Run No").loc[replay_cases["Run No"], replay_columns].to_numpy()
    difference = float(np.abs(replay[replay_columns].to_numpy() - cached).max())
    assert difference < (2e-6 if name == "transformer_weighted" else 1e-7), difference
    report.update({"source_replay_cases": replay_cases["Run No"].tolist(),
                   "source_replay_max_abs_difference": difference})
    print(f"{name}: replay passed ({difference:.3g}); evaluating 2022", flush=True)
    frame = predict_cases(evaluation)
    if name == "haukeli_specialist":
        old = pd.read_csv(prediction_path(OLD_2022["xgb_rescaled"]), float_precision="round_trip")
        assert old.columns.tolist() == ["Run No", *columns]
        assert old["Run No"].equals(evaluation["Run No"])
        old.loc[:, replay_columns] = frame[replay_columns].to_numpy()
        frame = old
        report["non_target_source"] = str(prediction_path(OLD_2022["xgb_rescaled"]).relative_to(ROOT))
        report["note"] = "Only Haukeli specialist predictions contribute to the ensemble; other cached columns are original frozen XGBoost."
    assert frame.columns.tolist() == ["Run No", *columns]
    assert frame["Run No"].equals(evaluation["Run No"])
    values = frame[columns].to_numpy()
    assert np.isfinite(values).all() and ((0 <= values) & (values <= 1)).all()
    assert all(digest(ROOT / path) == value for path, value in protected.items())
    frame.to_csv(output, index=False, mode="x")
    report["prediction_sha256"] = digest(output)
    write_json(report_path, report)
    print(f"Saved {output.relative_to(ROOT)}", flush=True)


def compare():
    output = ROOT / "reports/round3_frozen_ensemble_comparison.json"
    assert not output.exists()
    recipe_path = ROOT / "configs/round2_submission_depth8_haukeli_weighted.json"
    old_recipe_path = ROOT / "configs/goal_submission_rescaled_tree_mean.json"
    recipe, old_recipe = read_json(recipe_path), read_json(old_recipe_path)
    columns = pd.read_csv(ROOT / "data/raw/sample_submission.csv", nrows=0).columns[1:].tolist()
    assert len(columns) == 2352
    generators = list(dict.fromkeys(c.removeprefix("result_committed_").rsplit("_t", 1)[0] for c in columns))
    truth_path = ROOT / "data/raw/kernel/Unit_commitment_decisions.csv"
    truth = pd.read_csv(truth_path, usecols=["Run No", "starttime", *columns])
    truth["starttime"] = pd.to_datetime(truth.starttime, utc=True)
    protected = {str(p.relative_to(ROOT)): digest(p) for p in
                 (recipe_path, old_recipe_path, truth_path, ROOT / "evaluation/kaggle_metric.py")}
    rows = []
    for year in (2021, 2022):
        cases = truth.loc[truth.starttime.dt.year == year].sort_values("starttime").reset_index(drop=True)
        assert len(cases) == 365
        prefixes = dict(OLD_2021 if year == 2021 else OLD_2022)
        prefixes.update(SOURCES if year == 2021 else {name: f"round3_frozen_{name}_2022" for name in SOURCES})
        components = {}
        for name, prefix in prefixes.items():
            path = prediction_path(prefix)
            components[name] = read_predictions(path, columns, cases["Run No"])
            protected[str(path.relative_to(ROOT))] = digest(path)

        # Independent column-by-column reconstruction, separate from round2_compare.combine.
        predictions = {}
        for variant, weights in (("old_frozen", old_recipe["plant_weights"]), ("latest_frozen", recipe["plant_weights"])):
            values = np.empty((len(cases), len(columns)), dtype=np.float64)
            for i, column in enumerate(columns):
                generator = column.removeprefix("result_committed_").rsplit("_t", 1)[0]
                amounts = weights[generator.rsplit("_G", 1)[0]]
                assert abs(sum(amounts.values()) - 1) < 1e-12
                values[:, i] = sum(amount * components[name][:, i] for name, amount in amounts.items())
            assert np.isfinite(values).all() and ((0 <= values) & (values <= 1)).all()
            predictions[variant] = values
        subsets = {"all": np.ones(len(cases), dtype=bool)}
        if year == 2022:
            subsets["jan7_onward"] = (cases.starttime >= pd.Timestamp("2022-01-07", tz="UTC")).to_numpy()
            assert int(subsets["jan7_onward"].sum()) == 359
        for subset, mask in subsets.items():
            old_metrics = metrics(cases.loc[mask, columns].to_numpy(), predictions["old_frozen"][mask], generators)
            for variant, values in predictions.items():
                result = metrics(cases.loc[mask, columns].to_numpy(), values[mask], generators)
                rows.append({"year": year, "subset": subset, "variant": variant, "cases": int(mask.sum()),
                             **result, "delta_vs_old_frozen": result["micro_roc_auc"] - old_metrics["micro_roc_auc"],
                             "generator_delta_vs_old_frozen": {g: result["generator_auc"][g] - old_metrics["generator_auc"][g] for g in generators}})
        if year == 2021:
            known = read_json(ROOT / "reports/round2_depth8_combined_both_comparison.json")
            expected = {r["variant"]: r["micro_roc_auc"] for r in known["results"] if r["year"] == 2021}
            assert abs(rows[-2]["micro_roc_auc"] - expected["baseline"]) < 1e-12
            assert abs(rows[-1]["micro_roc_auc"] - expected["depth8_haukeli_weighted_mean"]) < 1e-12
        else:
            known = read_json(ROOT / "reports/round2_frozen_ensemble_2022_comparison.json")
            expected = next(r for r in known["results"] if r["variant"] == "candidate_replace")
            old_all = next(r for r in rows if r["year"] == 2022 and r["subset"] == "all" and r["variant"] == "old_frozen")
            assert abs(old_all["micro_roc_auc"] - expected["micro_roc_auc"]) < 1e-12
            for variant, values in predictions.items():
                frame = pd.DataFrame(values, columns=columns)
                frame.insert(0, "Run No", cases["Run No"])
                frame.to_csv(prediction_path(f"round3_frozen_{variant}_2022"), index=False, mode="x")
    assert all(digest(ROOT / path) == value for path, value in protected.items())
    report = {
        "recipe": str(recipe_path.relative_to(ROOT)), "old_recipe": str(old_recipe_path.relative_to(ROOT)),
        "new_fitting": False, "scalers_refitted": False, "new_model_selection": False,
        "independent_2021_recipe_reconstruction_passed": True,
        "original_frozen_2022_score_reproduced": True,
        "source_hashes": protected, "results": rows,
        "inference_reports": [f"reports/round3_frozen_{name}_2022_metrics.json" for name in SOURCES],
        "interpretation": "All models were fitted using case horizons ending by 2021-01-01 and selected on 2021. Their parameters and stored neural scalers remain frozen when predicting 2022. The additional Jan7+ subset removes six case starts overlapping 2021 selection horizons. 2021 and 2022 are heavily reused development years, not independent holdouts. The comparison isolates the recipe change at a common pre-2021 training cutoff; it does not establish performance on 2024. Cross-year drops also reflect different case difficulty and distributions."
    }
    write_json(output, report)
    print(json.dumps([{k:r[k] for k in ("year", "subset", "variant", "micro_roc_auc", "delta_vs_old_frozen")} for r in rows], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=list(SOURCES))
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.compare:
        compare()
    else:
        env = os.environ.copy()
        for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[name] = "1"
        for name in SOURCES:
            subprocess.run([sys.executable, __file__, "--worker", name], check=True, env=env)
        compare()


if __name__ == "__main__":
    main()
