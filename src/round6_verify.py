"""Replay 14 saved LightGBM generators and optionally verify a final blend; no fitting."""

import argparse
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import sys

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"


class RejectOtherModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "xgboost", "catboost"}:
            raise AssertionError(f"Verification must not import {fullname}.")


sys.meta_path.insert(0, RejectOtherModels())
import lightgbm as lgb
import numpy as np
import pandas as pd

from features import ROOT, HOURS, load_data
from round6_lightgbm import generator_features


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def read(path):
    return json.loads((ROOT / path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", required=True)
    parser.add_argument("--raw-predictions", required=True)
    parser.add_argument("--recipe")
    parser.add_argument("--submission")
    parser.add_argument("--average-report", help="Optional uniform component mean to independently reconstruct.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = ROOT / args.output
    assert output.stem.startswith("round6_") and not output.exists()
    assert bool(args.recipe) == bool(args.submission), "Supply recipe and submission together."
    source = read(args.source_report)
    config = source["config"]
    config_path = f"configs/{Path(args.source_report).stem.removesuffix('_run')}.json"
    assert read(config_path) == config and config["full_fit"] and source["fitting_performed"]
    assert (ROOT / source["prediction_path"]).resolve() == (ROOT / args.raw_predictions).resolve()
    assert digest(args.raw_predictions) == source["prediction_sha256"]
    hashes = {path: digest(path) for path in (args.source_report, args.raw_predictions, config_path,
              "data/raw/sample_submission.csv", "data/raw/prediction_mapping.csv",
              "data/raw/kernel/Unit_commitment_decisions.csv", "src/round6_lightgbm.py", "src/round5_lightgbm.py", "src/features.py", "src/tree_preprocessing.py")}
    template = pd.read_csv(ROOT / "data/raw/sample_submission.csv", float_precision="round_trip")
    ids, columns = template["Run No"], template.columns[1:].tolist()
    assert template.shape == (725, 2353) and ids.is_unique and len(set(columns)) == 2352

    def frame(path, partial=False):
        result = pd.read_csv(ROOT / path, float_precision="round_trip")
        assert result.columns.is_unique and result.columns[0] == "Run No"
        assert result["Run No"].is_unique and result["Run No"].equals(ids)
        provided = result.columns[1:].tolist()
        assert (provided and set(provided).issubset(columns)) if partial else provided == columns
        values = result[provided].to_numpy()
        assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
        hashes[path] = digest(path)
        return result

    raw = frame(args.raw_predictions)
    train, label_columns, generators, inputs, topology = load_data(use_extended=config.get("use_extended", False))
    mapping = pd.read_csv(ROOT / "data/raw/prediction_mapping.csv")
    assert train["Run No"].is_unique and mapping["Run No"].is_unique
    assert set(mapping["Run No"]) == set(ids) and not train["Run No"].isin(ids).any()
    assert label_columns == columns and source["generators"] == generators and len(generators) == 14
    assert list(source["generator_models"]) == generators
    assert len(train) == source["training_cases"] == 2922 and source["evaluation_cases"] == 725
    cohort_hash = hashlib.sha256(train[["Run No", "starttime"]].to_csv(index=False).encode()).hexdigest()
    assert cohort_hash == source["training_case_ids_sha256"] and source["competition_training_id_overlap"] == 0
    test = mapping.set_index("Run No").loc[ids].reset_index()
    test["starttime"] = pd.to_datetime(test.starttime, utc=True)
    for prefix, cases in (("training", train), ("evaluation", test)):
        assert str(cases.starttime.min()) == source[f"first_{prefix}_start"]
        assert str(cases.starttime.max()) == source[f"last_{prefix}_start"]
    indices = np.linspace(0, len(test) - 1, 5, dtype=int)
    sample = test.iloc[indices].reset_index(drop=True)
    checks, model_paths = {}, []
    for index, generator in enumerate(generators):
        saved = source["generator_models"][generator]
        target = [f"result_committed_{generator}_t{hour}" for hour in range(HOURS)]
        assert saved["generator_id"] == index and saved["output_columns"] == target == columns[index * HOURS:(index + 1) * HOURS]
        path = saved["model_path"]
        model_paths.append(path)
        assert digest(path) == saved["model_sha256"]
        hashes[path] = saved["model_sha256"]
        features = generator_features(sample, generator, index, inputs, topology, config)
        assert (features["generator_id"].to_numpy() == index).all()
        assert np.array_equal(features["horizon_hour"].to_numpy(), np.tile(np.arange(HOURS), 5))
        categorical_index = features.columns.get_loc("generator_id")
        assert f"[categorical_feature: {categorical_index}]" in (ROOT / path).read_text()
        booster = lgb.Booster(model_file=str(ROOT / path), params={"num_threads": 1})
        assert booster.feature_name() == saved["feature_names"] == features.columns.tolist()
        expected_trees = config.get("trees_by_generator", {}).get(generator, config["model"]["n_estimators"])
        assert booster.current_iteration() == booster.num_trees() == saved["trees"] == expected_trees
        replay = booster.predict(features, num_threads=1).reshape(5, HOURS)
        difference = float(np.abs(replay - raw.loc[indices, target].to_numpy()).max())
        assert difference <= 1e-10, (generator, difference)
        checks[generator] = {"generator_id": index, "trees": saved["trees"], "replay_max_absolute_difference": difference}
    assert len(set(model_paths)) == 14
    blend_check = None
    average_check = None
    if args.average_report:
        average = read(args.average_report)
        mean_frame = frame(average["output"])
        assert digest(average["output"]) == average["output_sha256"]
        members = []
        for member in average["sources"]:
            assert digest(member["path"]) == member["sha256"]
            assert member["weight"] == 1 / len(average["sources"])
            members.append(frame(member["path"])[columns].to_numpy())
        assert np.array_equal(np.mean(members, axis=0), mean_frame[columns].to_numpy())
        hashes[args.average_report] = digest(args.average_report)
        average_check = {"output": average["output"], "sources": len(members), "max_absolute_difference": 0.0}
    if args.recipe:
        recipe = read(args.recipe)
        submission = frame(args.submission)
        parts = {name: frame(path, partial=True) for name, path in recipe["sources"].items()}
        if average_check:
            assert average_check["output"] in recipe["sources"].values()
        expected = np.zeros((len(ids), len(columns)), dtype=np.float64)
        plants = {column.removeprefix("result_committed_").rsplit("_G", 1)[0] for column in columns}
        assert set(recipe["plant_weights"]) == plants
        for index, column in enumerate(columns):
            weights = recipe["plant_weights"][column.removeprefix("result_committed_").rsplit("_G", 1)[0]]
            amounts = np.asarray(list(weights.values()), dtype=float)
            assert np.isfinite(amounts).all() and (amounts >= 0).all() and abs(amounts.sum() - 1) < 1e-12
            for name, weight in weights.items():
                expected[:, index] += weight * parts[name][column].to_numpy()
        assert np.array_equal(expected, submission[columns].to_numpy())
        export_path = f"reports/{Path(args.submission).stem}_run.json"
        exported = read(export_path)
        assert exported["config"] == recipe and exported["config_sha256"] == digest(args.recipe)
        assert exported["submission_sha256"] == digest(args.submission)
        assert (ROOT / exported["submission"]).resolve() == (ROOT / args.submission).resolve()
        assert exported["template_sha256"] == digest("data/raw/sample_submission.csv")
        assert exported["historical_source"] == recipe["historical_report"]
        assert exported["historical_sha256"] == digest(recipe["historical_report"])
        assert exported["test_cases"] == 725 and exported["outputs_per_case"] == 2352
        assert set(exported["sources"]) == set(recipe["sources"])
        for name, path in recipe["sources"].items():
            assert exported["sources"][name]["path"] == path and exported["sources"][name]["sha256"] == digest(path)
        hashes.update({path: digest(path) for path in (args.recipe, export_path, recipe["historical_report"])})
        blend_check = {"submission": args.submission, "verified_probabilities": expected.size, "max_absolute_difference": 0.0}
    snapshot_path = "/private/tmp/tdt4173-round6-protection.json"
    snapshot = read(snapshot_path)
    assert len(snapshot) == 1190
    assert all((ROOT / path).exists() and digest(path) == expected for path, expected in snapshot.items())
    assert not any(name in sys.modules for name in ("torch", "xgboost", "catboost"))
    assert all(digest(path) == expected for path, expected in hashes.items())
    report = {"passed": True, "source_report": args.source_report, "raw_predictions": args.raw_predictions,
              "cases": 725, "outputs_per_case": 2352, "training_cohort_sha256": cohort_hash,
              "competition_training_id_overlap": 0, "replay_case_indices": indices.tolist(),
              "replay_case_ids": sample["Run No"].tolist(), "generator_checks": checks, "blend_check": blend_check,
              "average_check": average_check,
              "protected_originals": {"snapshot": snapshot_path, "sha256": digest(snapshot_path), "unchanged_files": len(snapshot)},
              "artifact_hashes": hashes, "verification_script_sha256": digest(__file__), "cpu_threads": 1, "fitting_performed": False}
    with output.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"passed": True, "output": str(output), "replayed_generators": len(checks)}))


if __name__ == "__main__":
    main()
