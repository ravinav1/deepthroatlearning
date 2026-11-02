"""Verify optional temporal features and unchanged saved-tree inference; no fitting."""

import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml

import features
from tree_preprocessing import rescale_tree_features


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = Path("/private/tmp/tdt4173-round3-source-before")
RANK_COLUMNS = ["temporal_rank_week", "temporal_rank_remaining"]
BLOCK_COLUMNS = [
    "temporal_favourable_duration", "temporal_favourable_elapsed",
    "temporal_favourable_remaining", "temporal_favourable_previous_distance",
    "temporal_favourable_next_distance", "temporal_favourable_left_censored",
    "temporal_favourable_right_censored",
]
BASE_OPTIONS = ["use_extended", "use_system_context", "use_topology", "use_price_context",
                "use_water_context", "use_economic_context", "use_river_context",
                "use_inflow_profiles", "use_hydraulic_context", "use_water_budget"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def equal_frames(left, right):
    assert left.columns.tolist() == right.columns.tolist()
    assert left.dtypes.tolist() == right.dtypes.tolist()
    assert left.shape == right.shape
    assert np.array_equal(left.to_numpy(), right.to_numpy())


def read_context():
    raw = ROOT / "data/raw"
    historical = pd.read_csv(raw / "kernel/Unit_commitment_decisions.csv", usecols=["Run No", "starttime"])
    competition = pd.read_csv(raw / "prediction_mapping.csv", usecols=["Run No", "starttime"])
    assert set(historical["Run No"]).isdisjoint(competition["Run No"])
    cases = pd.concat([historical, competition], ignore_index=True)
    cases["starttime"] = pd.to_datetime(cases["starttime"], utc=True)
    starts = pd.to_datetime(["2015-01-01", "2015-06-15", "2021-01-01", "2021-06-15",
                             "2022-06-15", "2023-06-15", "2024-12-25"], utc=True)
    cases = cases.set_index("starttime").loc[starts].reset_index().rename(columns={"index": "starttime"})
    columns = pd.read_csv(raw / "sample_submission.csv", nrows=0).columns[1:].tolist()
    generators = list(dict.fromkeys(c.removeprefix("result_committed_").rsplit("_t", 1)[0] for c in columns))
    names = {
        "price": "kernel/Historical_day_ahead_price_2015_2025.csv",
        "volume": "kernel/Historical_volume_2015_2024.csv",
        "water_value": "kernel/Synthetic_water_value_2015_2024.csv",
        "inflow": "kernel/Historical_inflow_1958_2025.csv",
        "min_volume": "extended/Constraint_min_volume.csv",
        "min_flow": "extended/Constraint_min_flow.csv",
    }
    inputs = {}
    for name, filename in names.items():
        frame = pd.read_csv(raw / filename)
        frame["date"] = pd.to_datetime(frame["date"], utc=True)
        inputs[name] = frame.set_index("date")
    topology = yaml.safe_load((ROOT / "configs/Tokke_Vinje_topology.yaml").read_text())
    return cases, columns, generators, inputs, topology


def independent_temporal_values(cases, generators, inputs, topology, frame):
    """Check flattening and per-generator water-value selection using separate algorithms."""
    values = frame.to_numpy().reshape(len(generators), len(cases), features.HOURS, -1)
    for case_index, case in cases.iterrows():
        stamps = pd.date_range(case.starttime, periods=features.HOURS, freq="h")
        price = inputs["price"].loc[stamps].iloc[:, 0]
        week_rank = ((price.rank(method="average") - .5) / features.HOURS).to_numpy(dtype="float32")
        remaining_rank = np.array([
            (price.iloc[hour:].rank(method="average").iloc[0] - .5) / (features.HOURS - hour)
            for hour in range(features.HOURS)], dtype="float32")
        for generator_index, generator in enumerate(generators):
            rows = values[generator_index, case_index]
            assert np.array_equal(rows[:, frame.columns.get_loc("generator_id")],
                                  np.full(features.HOURS, generator_index))
            assert np.array_equal(rows[:, frame.columns.get_loc("horizon_hour")], np.arange(features.HOURS))
            assert np.array_equal(rows[:, frame.columns.get_loc("price")], price.to_numpy(dtype="float32"))
            assert np.array_equal(rows[:, frame.columns.get_loc(RANK_COLUMNS[0])], week_rank)
            assert np.array_equal(rows[:, frame.columns.get_loc(RANK_COLUMNS[1])], remaining_rank)
            plant = generator.rsplit("_G", 1)[0]
            reservoirs = features.feeding_reservoirs(topology, plant)
            water = inputs["water_value"].loc[case.starttime + pd.Timedelta(hours=features.HOURS), reservoirs].mean()
            positive = price.to_numpy() > water
            expected = {name: np.zeros(features.HOURS, dtype="float32") for name in BLOCK_COLUMNS}
            for active, group in itertools.groupby(enumerate(positive), key=lambda item: item[1]):
                positions = [index for index, _ in group]
                if active:
                    expected[BLOCK_COLUMNS[0]][positions] = len(positions)
                    expected[BLOCK_COLUMNS[1]][positions] = np.arange(1, len(positions) + 1)
                    expected[BLOCK_COLUMNS[2]][positions] = np.arange(len(positions), 0, -1)
                    expected[BLOCK_COLUMNS[5]][positions] = positions[0] == 0
                    expected[BLOCK_COLUMNS[6]][positions] = positions[-1] == features.HOURS - 1
            positive_hours = np.flatnonzero(positive)
            for hour in range(features.HOURS):
                before = positive_hours[positive_hours <= hour]
                after = positive_hours[positive_hours >= hour]
                expected[BLOCK_COLUMNS[3]][hour] = hour - before[-1] if len(before) else features.HOURS
                expected[BLOCK_COLUMNS[4]][hour] = after[0] - hour if len(after) else features.HOURS
            for name, expected_values in expected.items():
                assert np.array_equal(rows[:, frame.columns.get_loc(name)], expected_values), (generator, case.starttime, name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="reports/round3_feature_integration_checks.json")
    args = parser.parse_args()
    output = ROOT / args.output
    assert not output.exists(), output
    assert "torch" not in sys.modules
    # The snapshot builder can safely share these unchanged pure helper modules.
    dependencies = ["topology_features.py", "price_features.py", "operating_features.py",
                    "hydraulic_features.py", "water_budget_features.py", "tree_preprocessing.py"]
    for name in dependencies:
        assert digest(SNAPSHOT / name) == digest(ROOT / "src" / name), name
    spec = importlib.util.spec_from_file_location("round3_original_features", SNAPSHOT / "features.py")
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    cases, columns, generators, inputs, topology = read_context()
    tree_config = json.loads((ROOT / "configs/round2_xgb_depth8_2021.json").read_text())
    neural_config = json.loads((ROOT / "configs/round2_transformer_task_weights_2021.json").read_text())
    configurations = {"minimal_defaults": {}, "selected_tree_defaults": tree_config,
                      "selected_neural_base_defaults": neural_config}
    checks = []
    for name, config in configurations.items():
        options = {key: config.get(key, False) for key in BASE_OPTIONS}
        before = original.build_features(cases, generators, inputs, topology, **options)
        after = features.build_features(cases, generators, inputs, topology, **options)
        equal_frames(before, after)
        for ranks, blocks, added in [(True, False, RANK_COLUMNS), (False, True, BLOCK_COLUMNS),
                                     (True, True, RANK_COLUMNS + BLOCK_COLUMNS)]:
            extended = features.build_features(cases, generators, inputs, topology, **options,
                                               use_temporal_ranks=ranks, use_favourable_periods=blocks)
            assert len(extended.columns) == len(before.columns) + len(added)
            assert [c for c in extended if c not in before] == added
            equal_frames(before, extended[before.columns])
            scaled = rescale_tree_features(extended)
            equal_frames(extended[added], scaled[added])
            old_scaled = rescale_tree_features(before)
            equal_frames(old_scaled, scaled[old_scaled.columns])
        checks.append({"configuration": name, "base_features": len(before.columns),
                       "default_bitwise_equal": True, "rank_extra": 2, "block_extra": 7,
                       "combined_extra": 9, "base_columns_unchanged": True,
                       "new_columns_unchanged_by_economic_rescaling": True})

    options = {key: tree_config.get(key, False) for key in BASE_OPTIONS}
    combined = features.build_features(cases, generators, inputs, topology, **options,
                                       use_temporal_ranks=True, use_favourable_periods=True)
    independent_temporal_values(cases, generators, inputs, topology, combined)
    cube = combined.to_numpy().reshape(len(generators), len(cases), features.HOURS, -1)
    for index in range(len(cases)):
        isolated = features.build_features(cases.iloc[[index]].reset_index(drop=True), generators, inputs, topology,
                                           **options, use_temporal_ranks=True, use_favourable_periods=True)
        assert np.array_equal(isolated.to_numpy().reshape(len(generators), features.HOURS, -1), cube[:, index])
    permutation = np.arange(len(cases))[::-1]
    permuted = features.build_features(cases.iloc[permutation].reset_index(drop=True), generators, inputs, topology,
                                       **options, use_temporal_ranks=True, use_favourable_periods=True)
    assert np.array_equal(permuted.to_numpy().reshape(cube.shape), cube[:, permutation])

    historical = cases.loc[cases.starttime.dt.year <= 2022].reset_index(drop=True)
    model_path = ROOT / "models/round2_xgb_depth8_2021.ubj"
    model_hash = digest(model_path)
    model = xgb.XGBClassifier(n_jobs=1, enable_categorical=True)
    model.load_model(model_path)
    model.set_params(n_jobs=1)
    predictions = []
    for builder in [original.build_features, features.build_features]:
        matrix = rescale_tree_features(builder(historical, generators, inputs, topology, **options))
        assert matrix.columns.tolist() == model.get_booster().feature_names
        matrix["generator_id"] = pd.Categorical(matrix["generator_id"], categories=range(len(generators)))
        predictions.append(model.predict_proba(matrix)[:, 1])
    assert np.array_equal(*predictions)
    restored = features.prediction_frame(predictions[1], historical, columns, generators)
    source_year_ids = historical.loc[historical.starttime.dt.year == 2021, "Run No"]
    saved = pd.read_csv(ROOT / "data/processed/round2_xgb_depth8_2021_validation_predictions.csv",
                        float_precision="round_trip").set_index("Run No").loc[source_year_ids, columns].to_numpy(dtype="float32")
    replay = restored.set_index("Run No").loc[source_year_ids, columns].to_numpy(dtype="float32")
    assert np.array_equal(saved, replay)
    assert digest(model_path) == model_hash
    assert "torch" not in sys.modules
    report = {
        "status": "passed", "training": "none", "inference_threads": 1, "labels_read": False,
        "cases": [{"run_id": int(row["Run No"]), "starttime": str(row.starttime)} for _, row in cases.iterrows()],
        "configuration_checks": checks,
        "row_order_and_independent_rank_block_calculation": "passed for all seven cases and all14generators",
        "single_case_and_reversed_case_order": "bitwise identical after restoring case positions",
        "saved_model": str(model_path.relative_to(ROOT)), "saved_model_sha256": model_hash,
        "saved_model_rounds": model.get_booster().num_boosted_rounds(),
        "old_new_default_prediction_max_difference": float(np.max(np.abs(predictions[0] - predictions[1]))),
        "saved_2021_csv_reproduced_at_float32_precision": True,
        "snapshot_features_sha256": digest(SNAPSHOT / "features.py"),
        "current_features_sha256": digest(ROOT / "src/features.py"),
        "temporal_features_sha256": digest(ROOT / "src/temporal_features.py"),
        "unchanged_helper_modules": dependencies,
    }
    with output.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
