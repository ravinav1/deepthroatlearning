"""Evaluate a saved pre-2021 XGBoost in the frozen 2022 ensemble; never fit.

Example: python src/round3_frozen_candidate.py --source round3_xgb_ranks_2021
Use --identity-check to replay the original round2 depth8 as verification only.
"""

import argparse
import hashlib
import inspect
import json
import os
import sys
from pathlib import Path

# Keep this standalone inference process on one CPU thread, without Torch.
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import pandas as pd
import xgboost as xgb

from features import HOURS, ROOT, build_features, load_data, prediction_frame
from plant_blend import metrics, read_predictions
from round2_compare import combine
from round3_frozen_evaluation import FEATURE_OPTIONS, OLD_2022, SOURCES, digest, prediction_path, write_json
from tree_preprocessing import rescale_tree_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", help="Saved model/report stem, including _2021; not a path")
    parser.add_argument("--output-prefix", help="Unique output stem; default derived from source")
    parser.add_argument("--identity-check", action="store_true")
    args = parser.parse_args()
    source = args.source or ("round2_xgb_depth8_2021" if args.identity_check else None)
    assert source and Path(source).name == source and source.endswith("_2021")
    assert source == "round2_xgb_depth8_2021" if args.identity_check else source.startswith("round3_xgb_")
    prefix = args.output_prefix or f"round3_frozen_{source.removeprefix('round3_').removesuffix('_2021')}_2022"
    assert Path(prefix).name == prefix
    output = prediction_path(prefix)
    report_path = ROOT / ("reports/round3_frozen_candidate_identity.json" if args.identity_check else f"reports/{prefix}_comparison.json")
    assert not report_path.exists() and (args.identity_check or not output.exists())
    source_path = ROOT / f"reports/{source}_metrics.json"
    config_path = ROOT / f"configs/{source}.json"
    model_path = ROOT / f"models/{source}.ubj"
    replay_path = prediction_path(source)
    protected = {str(path.relative_to(ROOT)): digest(path) for path in (source_path, config_path, model_path, replay_path)}
    source_report = json.loads(source_path.read_text())
    config = source_report["config"]
    assert json.loads(config_path.read_text()) == config
    assert config["validation_start"] == "2021-01-01" and config["validation_end"] == "2022-01-01"
    options = {name: config.get(name, False) for name in (*FEATURE_OPTIONS, "use_temporal_ranks", "use_favourable_periods")}
    parameters = inspect.signature(build_features).parameters
    assert all(name in parameters or not enabled for name, enabled in options.items())
    options = {name: enabled for name, enabled in options.items() if name in parameters}
    cases, columns, generators, inputs, topology = load_data(use_extended=options["use_extended"])
    train = cases.loc[cases.starttime + pd.Timedelta(hours=HOURS) <= pd.Timestamp("2021-01-01", tz="UTC")]
    evaluation = cases.loc[cases.starttime.dt.year == 2022].reset_index(drop=True)
    selection = cases.loc[cases.starttime.dt.year == 2021].reset_index(drop=True)
    assert len(train) == source_report["training_cases"] == 2186
    assert str(train.starttime.max()) == source_report["last_training_start"]
    assert len(selection) == source_report["validation_cases"] == len(evaluation) == 365
    assert (train.starttime + pd.Timedelta(hours=HOURS)).max() <= pd.Timestamp("2021-01-01", tz="UTC")
    competition_ids = pd.read_csv(ROOT / "data/raw/prediction_mapping.csv", usecols=["Run No"])["Run No"]
    assert not cases["Run No"].isin(competition_ids).any()
    assert not train["Run No"].isin(evaluation["Run No"]).any()
    model = xgb.XGBClassifier(n_jobs=1, enable_categorical=True)
    model.load_model(model_path)
    model.set_params(n_jobs=1)
    rounds = model.get_booster().num_boosted_rounds()
    assert rounds == source_report["best_iteration"]
    assert model.get_booster().feature_names == source_report["feature_names"]

    def predict(subset):
        frame = build_features(subset, generators, inputs, topology, **options)
        if config.get("use_economic_rescaling", False):
            frame = rescale_tree_features(frame)
        assert frame.columns.tolist() == source_report["feature_names"]
        frame["generator_id"] = pd.Categorical(frame["generator_id"], categories=range(len(generators)))
        probabilities = model.predict_proba(frame)[:, 1]
        return prediction_frame(probabilities, subset, columns, generators)

    replay_cases = selection.iloc[[0, 60, 120, 180, 240, 300, 364]].reset_index(drop=True)
    replay = predict(replay_cases)[columns].to_numpy()
    cached = pd.read_csv(replay_path, float_precision="round_trip")
    assert cached.columns.tolist() == ["Run No", *columns] and cached["Run No"].equals(selection["Run No"])
    reference = cached.set_index("Run No").loc[replay_cases["Run No"], columns].to_numpy()
    replay_difference = float(np.max(np.abs(replay - reference)))
    assert replay_difference < 1e-7
    print(f"{source}: source-year replay passed; predicting frozen 2022", flush=True)
    predictions = predict(evaluation)
    assert predictions.columns.tolist() == ["Run No", *columns] and predictions["Run No"].equals(evaluation["Run No"])
    candidate = predictions[columns].to_numpy()
    assert np.isfinite(candidate).all() and ((candidate >= 0) & (candidate <= 1)).all()

    recipe_path = ROOT / "configs/round2_submission_depth8_haukeli_weighted.json"
    previous_report_path = ROOT / "reports/round3_frozen_ensemble_comparison.json"
    recipe = json.loads(recipe_path.read_text())
    previous = json.loads(previous_report_path.read_text())
    prefixes = {**OLD_2022, **{name: f"round3_frozen_{name}_2022" for name in SOURCES}}
    components = {}
    for name, source_prefix in prefixes.items():
        path = prediction_path(source_prefix)
        relative = str(path.relative_to(ROOT))
        assert digest(path) == previous["source_hashes"][relative]
        components[name] = read_predictions(path, columns, evaluation["Run No"])
        protected[relative] = digest(path)
    protected.update({str(p.relative_to(ROOT)): digest(p) for p in (recipe_path, previous_report_path)})
    baseline = combine(components, columns, recipe["plant_weights"])
    cached_baseline = read_predictions(prediction_path("round3_frozen_latest_frozen_2022"), columns, evaluation["Run No"])
    assert np.array_equal(baseline, cached_baseline)
    truth = evaluation[columns].to_numpy()
    expected = next(r["micro_roc_auc"] for r in previous["results"] if r["year"] == 2022 and r["subset"] == "all" and r["variant"] == "latest_frozen")
    assert metrics(truth, baseline, generators)["micro_roc_auc"] == expected
    evidence = {"source": source, "source_hashes": protected, "feature_options": options,
                "feature_names": source_report["feature_names"], "trees": rounds,
                "training_cases": len(train), "last_training_start": str(train.starttime.max()),
                "last_training_horizon_end": str((train.starttime + pd.Timedelta(hours=HOURS)).max()),
                "training_cohort_sha256_reconstructed": hashlib.sha256(train[["Run No", "starttime"]].to_csv(index=False).encode()).hexdigest(),
                "competition_ids_overlap": 0, "new_fitting": False, "cpu_threads": 1,
                "source_replay_max_absolute_difference": replay_difference,
                "baseline_reconstruction_exact": True}
    if args.identity_check:
        difference = float(np.abs(candidate - components["xgb_depth8"]).max())
        assert difference < 1e-7
        evidence.update({"identity_check_passed": True, "candidate_max_difference": difference,
                         "note": "Original pre-2021 depth8 replay only; no candidate comparison or prediction artifact generated."})
    else:
        changed = {**components, "xgb_depth8": candidate}
        averaged = {**components, "xgb_depth8": .5 * (candidate + components["xgb_depth8"])}
        variants = {"baseline": baseline, "candidate_replace": combine(changed, columns, recipe["plant_weights"]),
                    "candidate_mean": combine(averaged, columns, recipe["plant_weights"])}
        results = []
        for subset, mask in (("all", np.ones(len(evaluation), dtype=bool)),
                             ("jan7_onward", (evaluation.starttime >= pd.Timestamp("2022-01-07", tz="UTC")).to_numpy())):
            reference = metrics(truth[mask], baseline[mask], generators)
            for variant, probabilities in variants.items():
                result = metrics(truth[mask], probabilities[mask], generators)
                results.append({"subset": subset, "variant": variant, "cases": int(mask.sum()), **result,
                                "delta_vs_baseline": result["micro_roc_auc"] - reference["micro_roc_auc"],
                                "generator_delta_vs_baseline": {g: result["generator_auc"][g] - reference["generator_auc"][g] for g in generators}})
        predictions.to_csv(output, index=False, mode="x")
        evidence.update({"prediction_path": str(output.relative_to(ROOT)), "prediction_sha256": digest(output),
                         "results": results, "validation_note": "Pre-2021 models and scalers remain frozen. Both years are reused development data. Jan7+ removes six starts overlapping source-selection horizons. Candidate replaces only the depth8 contribution; Haukeli is unchanged."})
    assert "torch" not in sys.modules
    assert all(digest(ROOT / path) == expected for path, expected in protected.items())
    write_json(report_path, evidence)
    print(json.dumps({"report": str(report_path.relative_to(ROOT)), "identity_check": args.identity_check,
                      "results": [{k: row[k] for k in ("subset", "variant", "micro_roc_auc", "delta_vs_baseline")} for row in evidence.get("results", [])]}, indent=2))


if __name__ == "__main__":
    main()
