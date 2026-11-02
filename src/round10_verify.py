"""Replay a confirmed two-seed Round10 rank submission; never fit a model."""

import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from features import ROOT, HOURS, load_data
from round5_lightgbm import OPTIONS, generator_features, training_rows


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report299", required=True)
    parser.add_argument("--report673", required=True)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--submission", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = ROOT / args.output
    assert output.stem.startswith("round10_") and not output.exists()
    started = time.monotonic()
    protected = {}

    def remember(relative):
        path = (ROOT / relative).resolve()
        key = str(path.relative_to(ROOT))
        current = digest(path)
        if key in protected:
            assert protected[key] == current, key
        else:
            protected[key] = current
        return path

    def read_json(relative):
        return json.loads(remember(relative).read_text())

    recipe = read_json(args.recipe)
    assert set(recipe["sources"]) == {"incumbent", "separate_lgb"}
    assert recipe["sources"]["incumbent"] == "submissions/submission_round3_temporal_ranks.csv"
    assert recipe["plant_weights"] and all(
        weights == {"incumbent": .5, "separate_lgb": .5}
        for weights in recipe["plant_weights"].values()
    )
    for relative in (
        "src/round10_verify.py", "src/round5_lightgbm.py", "src/features.py",
        "src/tree_preprocessing.py", "src/temporal_features.py", "evaluation/kaggle_metric.py",
        "configs/Tokke_Vinje_topology.yaml", "data/raw/kernel/Unit_commitment_decisions.csv",
        "data/raw/kernel/Historical_day_ahead_price_2015_2025.csv",
        "data/raw/kernel/Historical_volume_2015_2024.csv",
        "data/raw/kernel/Synthetic_water_value_2015_2024.csv",
        "data/raw/kernel/Historical_inflow_1958_2025.csv",
        "data/raw/extended/Constraint_min_volume.csv", "data/raw/extended/Constraint_min_flow.csv",
    ):
        remember(relative)
    train, columns, generators, inputs, topology = load_data(use_extended=True)
    assert len(train) == 2922 and len(generators) == 14 and len(columns) == 2352
    assert str(train.starttime.min()) == "2015-01-01 00:00:00+00:00"
    assert str(train.starttime.max()) == "2022-12-31 00:00:00+00:00"
    template = pd.read_csv(remember("data/raw/sample_submission.csv"), float_precision="round_trip")
    assert template.columns.tolist() == ["Run No", *columns]
    assert template.shape == (725, 2353) and template["Run No"].is_unique
    mapping = pd.read_csv(remember("data/raw/prediction_mapping.csv"))
    assert mapping["Run No"].is_unique and set(mapping["Run No"]) == set(template["Run No"])
    test = mapping.set_index("Run No").loc[template["Run No"]].reset_index()
    test["starttime"] = pd.to_datetime(test.starttime, utc=True)
    assert not train["Run No"].isin(test["Run No"]).any()
    cohort = hashlib.sha256(train[["Run No", "starttime"]].to_csv(index=False).encode()).hexdigest()
    plants = {generator.rsplit("_G", 1)[0] for generator in generators}
    assert set(recipe["plant_weights"]) == plants

    def read_predictions(relative):
        frame = pd.read_csv(remember(relative), float_precision="round_trip")
        assert frame.columns.tolist() == ["Run No", *columns]
        assert frame["Run No"].equals(template["Run No"])
        values = frame[columns].to_numpy()
        assert np.isfinite(values).all() and np.all((values >= 0) & (values <= 1))
        return values

    replay_rows, components = [], {}
    for seed, relative in ((299, args.report299), (673, args.report673)):
        run = read_json(relative)
        config = run["config"]
        name = Path(relative).stem.removesuffix("_run")
        assert name.startswith("round10_") and config == read_json(f"configs/{name}.json")
        assert config.get("full_fit") and run["fitting_performed"]
        assert not config.get("early_stopping_rounds")
        assert config.get("use_temporal_ranks")
        assert not config.get("use_shared_river_constraint_timing")
        assert not config.get("use_distinct_inflow_timing")
        reference = read_json("configs/round5_lgb_peer_relative_seed299_2021.json")
        for key in (*OPTIONS, "use_economic_rescaling", "hour_stride"):
            if key != "use_temporal_ranks":
                assert config.get(key, False) == reference.get(key, False), key
        assert config["trees_by_generator"] == reference["trees_by_generator"]
        expected_params = {key: value for key, value in reference["model"].items() if key != "n_jobs"}
        expected_params["random_state"] = seed
        assert {key: value for key, value in config["model"].items() if key != "n_jobs"} == expected_params
        assert run["training_cases"] == 2922 and run["evaluation_cases"] == 725
        assert run["training_case_ids_sha256"] == cohort
        assert run["competition_training_id_overlap"] == 0 and run["generators"] == generators
        assert set(run["generator_models"]) == set(generators)
        cached = read_predictions(run["prediction_path"])
        assert digest(ROOT / run["prediction_path"]) == run["prediction_sha256"]
        sampled_rows = training_rows(train, config["hour_stride"], seed)
        assert len(sampled_rows) == 2922 * 42
        for index, generator in enumerate(generators):
            saved = run["generator_models"][generator]
            target_columns = columns[index * HOURS:(index + 1) * HOURS]
            assert saved["generator_id"] == index and saved["output_columns"] == target_columns
            assert saved["training_rows"] == len(sampled_rows)
            assert saved["trees"] == config["trees_by_generator"][generator]
            model_path = remember(saved["model_path"])
            assert digest(model_path) == saved["model_sha256"]
            booster = lgb.Booster(model_file=str(model_path))
            features = generator_features(test, generator, index, inputs, topology, config)
            names = features.columns.tolist()
            assert len(names) == 115 and names == saved["feature_names"] == booster.feature_name()
            assert "temporal_rank_week" in names and "temporal_rank_remaining" in names
            assert booster.current_iteration() == saved["trees"]
            prediction = booster.predict(features, num_threads=1).reshape(len(test), HOURS)
            expected = cached[:, index * HOURS:(index + 1) * HOURS]
            difference = float(np.max(np.abs(prediction - expected)))
            assert np.array_equal(prediction, expected), (seed, generator, difference)
            replay_rows.append({"seed": seed, "generator": generator, "trees": saved["trees"],
                                "cases": 725, "features": 115, "max_absolute_difference": difference})
            del booster, features, prediction
            gc.collect()
        components[seed] = cached
        print(f"Seed {seed}: all 14 models replayed on all 725 competition cases.", flush=True)

    mean = read_predictions(recipe["sources"]["separate_lgb"])
    assert np.array_equal(mean, .5 * components[299] + .5 * components[673])
    round3 = read_predictions(recipe["sources"]["incumbent"])
    submission = read_predictions(args.submission)
    expected_submission = .5 * round3 + .5 * mean
    assert np.array_equal(submission, expected_submission)
    assert all(digest(ROOT / relative) == expected for relative, expected in protected.items())
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "status": "passed",
        "training_cases": 2922, "training_case_ids_sha256": cohort,
        "test_cases": 725, "outputs_per_case": 2352, "competition_training_id_overlap": 0,
        "replayed_models": 28, "replayed_component_probabilities": 2 * 725 * 2352,
        "all_replayed_predictions_exact": True, "seed_mean_exact": True, "final_blend_exact": True,
        "submission": args.submission, "submission_sha256": digest(ROOT / args.submission),
        "recipe": args.recipe, "per_generator_replay": replay_rows,
        "source_sha256": protected, "tracked_source_hashes_unchanged": True,
        "cpu_threads": 1, "fitting_performed": False, "elapsed_seconds": time.monotonic() - started,
        "limitations": "Full saved-model/export reproducibility; does not prove generalization or a new Kaggle best. "
                       "The unchanged Round3 sources were independently replayed in the prior audit.",
    }
    with output.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({key: report[key] for key in ("status", "replayed_models", "test_cases", "submission_sha256")}), flush=True)


if __name__ == "__main__":
    main()
