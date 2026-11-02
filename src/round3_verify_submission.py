"""Independently verify the optional temporal-rank submission after export.

No fitting. Uses one CPU thread and replays 17 fixed test cases; all submission
probabilities are separately reconstructed from the saved source predictions.
"""

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import pandas as pd
import xgboost as xgb

from features import ROOT, build_features, load_data, prediction_frame
from tree_preprocessing import rescale_tree_features


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads((ROOT / path).read_text())


def frame(path, ids, columns, partial=False):
    result = pd.read_csv(ROOT / path, float_precision="round_trip")
    assert result.columns[0] == "Run No" and result.columns.is_unique
    assert result["Run No"].is_unique and result["Run No"].equals(ids)
    provided = result.columns[1:].tolist()
    assert (provided and set(provided).issubset(columns)) if partial else provided == columns
    values = result[provided].to_numpy()
    assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
    return result


def independent_blend(parts, columns, weights):
    values = np.zeros((len(next(iter(parts.values()))), len(columns)), dtype=np.float64)
    for index, column in enumerate(columns):
        plant = column.removeprefix("result_committed_").rsplit("_t", 1)[0].rsplit("_G", 1)[0]
        amounts = weights[plant]
        assert all(weight >= 0 for weight in amounts.values()) and abs(sum(amounts.values()) - 1) < 1e-12
        for name, weight in amounts.items():
            values[:, index] += weight * parts[name][column].to_numpy()
    return values


def main():
    output = ROOT / "reports/round3_final_verification.json"
    assert not output.exists()
    config_path = "configs/round3_xgb_ranks_full.json"
    model_path = "models/xgboost_round3_xgb_ranks_full.ubj"
    prediction_path = "data/processed/round3_xgb_ranks_full_predictions.csv"
    run_path = "reports/round3_xgb_ranks_full_run.json"
    recipe_path = "configs/round3_submission_temporal_ranks.json"
    submission_path = "submissions/submission_round3_temporal_ranks.csv"
    export_path = "reports/submission_round3_temporal_ranks_run.json"
    history_path = "reports/round3_xgb_ranks_both_comparison.json"
    config, run, recipe, exported = map(read, (config_path, run_path, recipe_path, export_path))
    old_recipe = read("configs/round2_submission_depth8_haukeli_weighted.json")
    source_config = read("configs/round3_xgb_ranks_2021.json")
    selection = read("reports/round3_xgb_ranks_selection_2021.json")
    assert selection["selected_variant"] == recipe["historical_variant"] == "candidate_replace"
    assert recipe["historical_report"] == history_path
    assert run["config"] == config and config["model_type"] == "xgboost"
    assert config["model"] == source_config["model"] and config["model"]["n_estimators"] == run["trees"] == 748
    assert config["model"]["n_jobs"] == 4 and config.get("early_stopping_rounds") is None
    options = {name: config.get(name, False) for name in (
        "use_extended", "use_system_context", "use_topology", "use_price_context", "use_water_context",
        "use_economic_context", "use_river_context", "use_inflow_profiles", "use_hydraulic_context",
        "use_water_budget", "use_temporal_ranks", "use_favourable_periods")}
    assert options["use_temporal_ranks"] and not options["use_favourable_periods"]
    assert all(value == source_config.get(name, False) for name, value in options.items())
    assert config.get("use_economic_rescaling") == source_config.get("use_economic_rescaling") == True
    protected = {path: sha(ROOT / path) for path in (config_path, model_path, prediction_path, run_path,
                 recipe_path, submission_path, export_path, history_path, "evaluation/kaggle_metric.py")}
    for key in ("selection_source", "combined_selection_source"):
        if config.get(key):
            path = config[key]
            protected[path] = sha(ROOT / path)
            if config.get(key + "_sha256"):
                assert protected[path] == config[key + "_sha256"]
    template = pd.read_csv(ROOT / "data/raw/sample_submission.csv", float_precision="round_trip")
    columns, ids = template.columns[1:].tolist(), template["Run No"]
    assert template.shape == (725, 2353) and len(columns) == 2352 and ids.is_unique
    submission = frame(submission_path, ids, columns)
    cached = frame(prediction_path, ids, columns)
    train, label_columns, generators, inputs, topology = load_data(use_extended=options["use_extended"])
    mapping = pd.read_csv(ROOT / "data/raw/prediction_mapping.csv")
    assert mapping["Run No"].is_unique and set(mapping["Run No"]) == set(ids)
    test = mapping.set_index("Run No").loc[ids].reset_index()
    test["starttime"] = pd.to_datetime(test.starttime, utc=True)
    assert label_columns == columns and len(generators) == 14
    assert len(train) == run["training_cases"] == 2922 and len(test) == run["test_cases"] == 725
    assert not train["Run No"].isin(ids).any()
    for prefix, cases in (("training", train), ("test", test)):
        assert str(cases.starttime.min()) == run[f"first_{prefix}_start"]
        assert str(cases.starttime.max()) == run[f"last_{prefix}_start"]
    assert str(train.starttime.min()) == "2015-01-01 00:00:00+00:00"
    assert str(train.starttime.max()) == "2022-12-31 00:00:00+00:00"
    assert str(test.starttime.min()) == "2023-01-01 00:00:00+00:00"
    assert str(test.starttime.max()) == "2024-12-25 00:00:00+00:00"
    cohort_hash = hashlib.sha256(train[["Run No", "starttime"]].to_csv(index=False).encode()).hexdigest()
    assert cohort_hash == run["training_case_ids_sha256"]
    assert (ROOT / run["model_path"]).resolve() == (ROOT / model_path).resolve()
    assert (ROOT / run["submission"]).resolve() == (ROOT / prediction_path).resolve()
    model = xgb.XGBClassifier(n_jobs=1, enable_categorical=True)
    model.load_model(ROOT / model_path)
    model.set_params(n_jobs=1)
    assert model.get_booster().num_boosted_rounds() == 748
    names = model.get_booster().feature_names
    assert names == run["features"] and len(names) == 115
    assert "temporal_rank_week" in names and "temporal_rank_remaining" in names
    sample = test.iloc[np.linspace(0, len(test) - 1, 17, dtype=int)].reset_index(drop=True)
    features = rescale_tree_features(build_features(sample, generators, inputs, topology, **options))
    assert features.columns.tolist() == names
    features["generator_id"] = pd.Categorical(features["generator_id"], categories=range(len(generators)))
    replay = prediction_frame(model.predict_proba(features)[:, 1], sample, columns, generators)[columns].to_numpy()
    saved = cached.set_index("Run No").loc[sample["Run No"], columns].to_numpy()
    difference = float(np.abs(replay - saved).max())
    assert difference < 1e-7 and np.array_equal(replay.astype(np.float32), saved.astype(np.float32))

    new_source = [name for name, path in recipe["sources"].items() if path == prediction_path]
    assert len(new_source) == 1
    name = new_source[0]
    expected_weights = {plant: {(name if key == "xgb_depth8" else key): weight for key, weight in amounts.items()}
                        for plant, amounts in old_recipe["plant_weights"].items()}
    expected_sources = {(name if key == "xgb_depth8" else key): (prediction_path if key == "xgb_depth8" else path)
                        for key, path in old_recipe["sources"].items()}
    assert recipe["plant_weights"] == expected_weights and recipe["sources"] == expected_sources
    parts = {name: frame(path, ids, columns, partial=True) for name, path in recipe["sources"].items()}
    recomputed = independent_blend(parts, columns, recipe["plant_weights"])
    assert np.array_equal(recomputed, submission[columns].to_numpy())
    assert exported["config"] == recipe and exported["config_sha256"] == sha(ROOT / recipe_path)
    assert exported["submission_sha256"] == sha(ROOT / submission_path)
    assert exported["historical_sha256"] == sha(ROOT / history_path)
    for name, path in recipe["sources"].items():
        assert exported["sources"][name]["path"] == path
        assert exported["sources"][name]["sha256"] == sha(ROOT / path)
        protected[path] = sha(ROOT / path)

    spec = importlib.util.spec_from_file_location("provided_metric", ROOT / "evaluation/kaggle_metric.py")
    metric = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metric)
    history = read(history_path)
    historical = []
    prefixes = {"xgb_rescaled": "goal_xgboost_rescaled", "lightgbm_rescaled": "goal_lightgbm_rescaled",
                "transformer": "transformer_rescaled", "xgb_depth8": "round2_xgb_depth8",
                "haukeli_specialist": "round2_xgb_haukeli", "transformer_weighted": "round2_transformer_task_weights"}
    for year in (2021, 2022):
        cases = train.loc[train.starttime.dt.year == year].reset_index(drop=True)
        components = {name: frame(f"data/processed/{prefix}_{year}_validation_predictions.csv", cases["Run No"], columns)
                      for name, prefix in prefixes.items()}
        for variant in ("baseline", "candidate_replace"):
            if variant == "candidate_replace":
                components["xgb_depth8"] = frame(f"data/processed/round3_xgb_ranks_{year}_validation_predictions.csv", cases["Run No"], columns)
            values = independent_blend(components, columns, old_recipe["plant_weights"])
            predictions = pd.DataFrame(values, columns=columns)
            predictions.insert(0, "Run No", cases["Run No"])
            actual = metric.score(cases[["Run No", *columns]], predictions)
            expected = next(row for row in history["results"] if row["year"] == year and row["variant"] == variant)
            assert actual == expected["micro_roc_auc"]
            historical.append({"year": year, "variant": variant, "official_micro_roc_auc": actual})
        assert historical[-1]["official_micro_roc_auc"] > historical[-2]["official_micro_roc_auc"]
    frozen = read("reports/round3_frozen_xgb_ranks_2022_comparison.json")
    frozen_verification = read("reports/round3_frozen_candidates_verification.json")
    assert frozen_verification["passed"]
    frozen_rows = [row for row in frozen["results"] if row["variant"] == "candidate_replace"]
    assert {row["subset"] for row in frozen_rows} == {"all", "jan7_onward"}
    assert all(row["delta_vs_baseline"] > 0 for row in frozen_rows)
    for path, expected in frozen_verification["artifact_hashes"].items():
        assert sha(ROOT / path) == expected
    snapshot_path = Path("/private/tmp/tdt4173-round3-protection.json")
    snapshot = json.loads(snapshot_path.read_text())
    assert len(snapshot) == 402
    changed = [path for path, expected in snapshot.items() if not (ROOT / path).exists() or sha(ROOT / path) != expected]
    assert not changed, changed
    assert "torch" not in sys.modules
    assert all(sha(ROOT / path) == expected for path, expected in protected.items())
    report = {"passed": True, "submission": submission_path, "submission_sha256": sha(ROOT / submission_path),
              "cases": 725, "outputs_per_case": 2352, "verified_probabilities": int(recomputed.size),
              "full_blend_max_absolute_difference": 0.0, "checkpoint_replay_cases": sample["Run No"].tolist(),
              "checkpoint_replay_max_absolute_difference": difference, "checkpoint_replay_float32_equal": True,
              "trees": 748, "feature_count": 115, "training_cases": 2922,
              "training_cohort_sha256": cohort_hash, "competition_training_id_overlap": 0,
              "first_training_start": run["first_training_start"], "last_training_start": run["last_training_start"],
              "historical_metric_checks": historical, "frozen_promotion_checks": frozen_rows,
              "protected_originals": {"snapshot": str(snapshot_path), "files_checked": 402, "changed": changed},
              "artifact_hashes": protected, "verification_script_sha256": sha(__file__), "cpu_threads": 1,
              "scope": "All CSV values independently blended; sampled17-case saved-model replay. Historical native scores reproduced with supplied metric. Reused development years, frozen promotion checks positive; no Kaggle upload or score claim."}
    with output.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({key: report[key] for key in ("passed", "submission", "verified_probabilities", "checkpoint_replay_max_absolute_difference")}, indent=2))


if __name__ == "__main__":
    main()
