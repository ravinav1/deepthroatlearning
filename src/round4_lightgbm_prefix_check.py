"""Compare old/new saved LightGBMs at matched tree counts using only 2021 scores.

Each variant's gate requires positive ensemble deltas both at the saved
1499-tree screen count and at the deployment count (default 800). Never fits.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit

from features import HOURS, ROOT, build_features, load_data, prediction_frame
from plant_blend import metrics, read_predictions
from round2_compare import combine
from round3_frozen_evaluation import FEATURE_OPTIONS, digest, prediction_path, write_json
from round4_compare import EXPECTED, NATIVE, RECIPE, official_metric
from tree_preprocessing import rescale_tree_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="New saved model/config/report stem ending _2021.")
    parser.add_argument("--num-iteration", type=int, default=800)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--variant", choices=["candidate_replace", "candidate_mean"], help="Previously chosen 2021 variant whose gate is required.")
    args = parser.parse_args()
    assert Path(args.source).name == args.source and args.source.startswith("round4_") and args.source.endswith("_2021")
    assert Path(args.output_prefix).name == args.output_prefix and args.output_prefix.startswith("round4_")
    assert 0 < args.num_iteration <= 1499
    report_path = ROOT / f"reports/{args.output_prefix}_comparison.json"
    output_paths = {name: prediction_path(f"{args.output_prefix}_{name}_{args.num_iteration}_2021") for name in ("incumbent", "candidate")}
    assert not report_path.exists() and not any(path.exists() for path in output_paths.values())
    config_path = ROOT / f"configs/{args.source}.json"
    config = json.loads(config_path.read_text())
    assert config["validation_start"] == "2021-01-01" and config["validation_end"] == "2022-01-01"
    cases, columns, generators, inputs, topology = load_data(use_extended=config.get("use_extended", False))
    validation = cases.loc[cases.starttime.dt.year == 2021].reset_index(drop=True)
    boundary = pd.Timestamp("2021-01-01", tz="UTC")
    train = cases.loc[cases.starttime + pd.Timedelta(hours=HOURS) <= boundary]
    assert len(train) == 2186 and len(validation) == 365
    assert (train.starttime + pd.Timedelta(hours=HOURS)).max() <= boundary
    test_ids = pd.read_csv(ROOT / "data/raw/prediction_mapping.csv", usecols=["Run No"])["Run No"]
    assert not cases["Run No"].isin(test_ids).any()
    recipe = json.loads((ROOT / RECIPE).read_text())
    protected = {RECIPE: digest(ROOT / RECIPE), "evaluation/kaggle_metric.py": digest(ROOT / "evaluation/kaggle_metric.py")}
    components = {}
    for name, prefix in NATIVE.items():
        path = prediction_path(f"{prefix}_2021")
        components[name] = read_predictions(path, columns, validation["Run No"])
        protected[str(path.relative_to(ROOT))] = digest(path)
    baseline = combine(components, columns, recipe["plant_weights"])
    assert metrics(validation[columns].to_numpy(), baseline, generators)["micro_roc_auc"] == EXPECTED["native", 2021]
    full_values, prefix_values, replays = {}, {}, {}
    for name, source in (("incumbent", "goal_lightgbm_rescaled_2021"), ("candidate", args.source)):
        source_path, config_path = ROOT / f"reports/{source}_metrics.json", ROOT / f"configs/{source}.json"
        model_path, cache_path = ROOT / f"models/lightgbm_{source}.txt", prediction_path(source)
        for path in (source_path, config_path, model_path, cache_path):
            protected[str(path.relative_to(ROOT))] = digest(path)
        report = json.loads(source_path.read_text())
        source_config = report["config"]
        assert json.loads(config_path.read_text()) == source_config
        assert source_config["validation_start"] == "2021-01-01" and source_config["validation_end"] == "2022-01-01"
        assert source_config.get("use_extended", False) == config.get("use_extended", False)
        assert report["training_cases"] == len(train) and report["validation_cases"] == len(validation)
        assert report["last_training_start"] == str(train.starttime.max())
        model = lgb.Booster(model_file=str(model_path))
        assert model.current_iteration() == report["best_iteration"] == 1499
        names = report["feature_names"]
        assert model.feature_name() == names
        options = {key: source_config.get(key, False) for key in (*FEATURE_OPTIONS, "use_temporal_ranks", "use_favourable_periods")}

        def predict_cases(subset, tree_count):
            features = build_features(subset, generators, inputs, topology, **options)
            if source_config.get("use_economic_rescaling", False):
                features = rescale_tree_features(features)
            assert features.columns.tolist() == names
            return prediction_frame(expit(model.predict(features, raw_score=True, num_threads=1, num_iteration=tree_count)),
                                    subset, columns, generators)

        full_values[name] = read_predictions(cache_path, columns, validation["Run No"])
        indices = [0, 60, 120, 180, 240, 300, 364]
        replay = predict_cases(validation.iloc[indices].reset_index(drop=True), 1499)[columns].to_numpy()
        difference = float(np.abs(replay - full_values[name][indices]).max())
        assert difference < 1e-10
        replays[name] = {"source": source, "selected_trees": 1499, "prefix_trees": args.num_iteration,
                         "feature_names": names, "source_replay_max_absolute_difference": difference}
        print(f"{source}: source replay passed; predicting 2021 at {args.num_iteration} trees", flush=True)
        frame = predict_cases(validation, args.num_iteration)
        assert frame.columns.tolist() == ["Run No", *columns] and frame["Run No"].equals(validation["Run No"])
        prefix_values[name] = frame[columns].to_numpy()
        assert np.isfinite(prefix_values[name]).all() and ((prefix_values[name] >= 0) & (prefix_values[name] <= 1)).all()
        frame.to_csv(output_paths[name], index=False, mode="x")
    rows, score = [], official_metric()
    for tree_count, values in ((1499, full_values), (args.num_iteration, prefix_values)):
        variants = {}
        for variant, lightgbm in (("baseline", values["incumbent"]), ("candidate_replace", values["candidate"]),
                                  ("candidate_mean", .5 * (values["incumbent"] + values["candidate"]))):
            variants[variant] = combine({**components, "lightgbm_rescaled": lightgbm}, columns, recipe["plant_weights"])
        reference = metrics(validation[columns].to_numpy(), variants["baseline"], generators)
        for variant, predictions in variants.items():
            result = metrics(validation[columns].to_numpy(), predictions, generators)
            frame = pd.DataFrame(predictions, columns=columns)
            frame.insert(0, "Run No", validation["Run No"])
            official = score(validation[["Run No", *columns]], frame)
            assert official == result["micro_roc_auc"]
            rows.append({"year": 2021, "trees": tree_count, "variant": variant, "cases": len(validation), **result,
                         "official_micro_roc_auc": official, "delta_vs_matched_baseline": result["micro_roc_auc"] - reference["micro_roc_auc"],
                         "generator_delta_vs_matched_baseline": {g: result["generator_auc"][g] - reference["generator_auc"][g] for g in generators}})
    assert all(digest(ROOT / path) == expected for path, expected in protected.items())
    assert "torch" not in sys.modules and "xgboost" not in sys.modules
    gates = {variant: all(r["delta_vs_matched_baseline"] > 0 for r in rows if r["variant"] == variant)
             for variant in ("candidate_replace", "candidate_mean")}
    passed = gates[args.variant] if args.variant else None
    write_json(report_path, {"source": args.source, "baseline_config": RECIPE,
                            "source_hashes": protected, "model_checks": replays,
                            "saved_prediction_hashes": {str(path.relative_to(ROOT)): digest(path) for path in output_paths.values()},
                            "training_cases": len(train), "last_training_horizon_end": str((train.starttime + pd.Timedelta(hours=HOURS)).max()),
                            "training_cohort_sha256_reconstructed": hashlib.sha256(train[["Run No", "starttime"]].to_csv(index=False).encode()).hexdigest(),
                            "competition_ids_overlap": 0, "new_fitting": False, "cpu_threads": 1,
                            "gate": "Positive 2021 ensemble delta for the same chosen variant at both 1499 and the deployment-prefix count, relative to matching incumbent tree counts.",
                            "gate_by_variant": gates, "chosen_variant": args.variant, "gate_passed": passed,
                            "candidate_2022_evaluated": False, "results": rows})
    print(json.dumps({"gate_by_variant": gates, "chosen_variant": args.variant, "gate_passed": passed,
                      "results": [{k: r[k] for k in ("trees", "variant", "micro_roc_auc", "delta_vs_matched_baseline")} for r in rows]}, indent=2))


if __name__ == "__main__":
    main()
