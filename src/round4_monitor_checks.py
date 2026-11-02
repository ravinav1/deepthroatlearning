"""Check scoped checkpoint monitoring using cached probabilities only; no fitting."""

import argparse
import ast
import copy
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import sys
import tempfile

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"


class RejectTreeLibraries(importlib.abc.MetaPathFinder):
    """Make accidental Torch/tree-library mixing a failing integration check."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"xgboost", "lightgbm", "catboost"}:
            raise AssertionError(f"Monitor checks must not import {fullname}.")


sys.meta_path.insert(0, RejectTreeLibraries())
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from component_monitor import frozen_component_monitor

assert "torch" not in sys.modules, "The component monitor must not require Torch."
from train_cnn import ROOT, frozen_ensemble_monitor


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expect_rejected(callback):
    try:
        callback()
    except (AssertionError, KeyError):
        return
    raise AssertionError("An invalid monitor input was accepted.")


def synthetic_checks():
    columns = [f"result_committed_{plant}_G1_t0" for plant in ("Hogga", "Tokke", "Songa")]
    validation = pd.DataFrame({"Run No": [20, 10]})
    options = {
        "mode": "replace_component", "replace_component": "transformer_weighted",
        "prediction_paths": {"tree": "tree.csv", "transformer": "transformer.csv",
                             "transformer_weighted": "not_read.csv"},
        "plant_weights": {
            "Hogga": {"tree": .25, "transformer": .25, "transformer_weighted": .5},
            "Tokke": {"transformer": .75, "transformer_weighted": .25},
            "Songa": {"tree": 1.0},
        },
    }
    tree = pd.DataFrame([[10, .25, .5, .75], [20, .5, .75, 1.0]], columns=["Run No", *columns])
    transformer = pd.DataFrame([[10, .75, .5, .25], [20, .25, .5, .75]], columns=["Run No", *columns])
    rejected = []
    with tempfile.TemporaryDirectory(prefix="round4-monitor-") as directory:
        root = Path(directory)
        tree.to_csv(root / "tree.csv", index=False)
        transformer.to_csv(root / "transformer.csv", index=False)
        constant, weights = frozen_component_monitor(options, validation, columns, root)
        expected = np.array([[.1875, .375, 1.0], [.25, .375, .75]])
        np.testing.assert_array_equal(constant, expected)
        np.testing.assert_array_equal(weights, [.5, .25, 0.0])
        new = np.array([[.25, .75, .125], [.5, .25, .875]])
        changed = new + .0625
        np.testing.assert_array_equal((constant + weights * changed) - (constant + weights * new),
                                      np.broadcast_to(weights * .0625, constant.shape))
        np.testing.assert_array_equal((constant + weights * changed)[:, 2], constant[:, 2])

        bad_frames = {
            "duplicate_cached_ids": tree.assign(**{"Run No": [10, 10]}),
            "missing_cached_ids": tree.iloc[:1],
            "extra_cached_ids": pd.concat([tree, tree.iloc[:1].assign(**{"Run No": 30})]),
            "wrong_column_order": tree[["Run No", columns[1], columns[0], columns[2]]],
            "nonfinite_cached_probability": tree.assign(**{columns[0]: np.nan}),
            "negative_cached_probability": tree.assign(**{columns[0]: -.1}),
            "over_one_cached_probability": tree.assign(**{columns[0]: 1.1}),
        }
        for name, frame in bad_frames.items():
            frame.to_csv(root / "tree.csv", index=False)
            expect_rejected(lambda: frozen_component_monitor(options, validation, columns, root))
            rejected.append(name)
        tree.to_csv(root / "tree.csv", index=False)
        for name, update in [
            ("weights_not_normalized", lambda o: o["plant_weights"]["Hogga"].update(tree=.5)),
            ("negative_component_weight", lambda o: o["plant_weights"]["Hogga"].update(tree=-.25, transformer=.75)),
            ("nonfinite_component_weight", lambda o: o["plant_weights"]["Hogga"].update(tree=np.nan)),
            ("missing_constant_source", lambda o: o["prediction_paths"].pop("tree")),
            ("unknown_replacement", lambda o: o.update(replace_component="unknown")),
            ("missing_plant", lambda o: o["plant_weights"].pop("Songa")),
        ]:
            bad = copy.deepcopy(options)
            update(bad)
            expect_rejected(lambda: frozen_component_monitor(bad, validation, columns, root))
            rejected.append(name)
        expect_rejected(lambda: frozen_component_monitor(options, validation.assign(**{"Run No": [10, 10]}), columns, root))
        rejected.append("duplicate_validation_ids")
        expect_rejected(lambda: frozen_ensemble_monitor({"mode": "unknown"}, validation, columns))
        rejected.append("unknown_dispatch_mode")
    return {"row_alignment_exact": True, "constant_exact": True, "replacement_weight_exact": True,
            "replacement_cache_not_read": True, "replacement_perturbation_exact": True,
            "zero_weight_columns_unchanged": True, "invalid_inputs_rejected": rejected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/round4_monitor_checks.json")
    args = parser.parse_args()
    assert not args.output.exists(), f"Report already exists: {args.output}"
    synthetic = synthetic_checks()
    recipe_path = ROOT / "configs/round3_submission_temporal_ranks.json"
    recipe = json.loads(recipe_path.read_text())
    paths = [ROOT / f"configs/round4_transformer_{name}_2021.json" for name in ("reference", "lr", "decay")]
    configs = [json.loads(path.read_text()) for path in paths]
    old_config_path = ROOT / "configs/round2_transformer_task_weights_2021.json"
    old = json.loads(old_config_path.read_text())
    for config, lr, decay in zip(configs, [.001, .0005, .001], [.0001, .0001, .01]):
        unchanged = set(old) - {"ensemble_monitor", "learning_rate", "weight_decay"}
        assert all(config[key] == old[key] for key in unchanged)
        assert set(config) == set(old) | {"weight_decay"}
        assert config["learning_rate"] == lr and config["weight_decay"] == decay
        assert config["ensemble_monitor"] == configs[0]["ensemble_monitor"]
        assert config["ensemble_monitor"]["plant_weights"] == recipe["plant_weights"]
        assert config["ensemble_monitor"]["replace_component"] == "transformer_weighted"
    monitor = configs[0]["ensemble_monitor"]
    assert monitor["prediction_paths"]["xgb_depth8"] == "data/processed/round3_xgb_ranks_2021_validation_predictions.csv"
    assert set(monitor["prediction_paths"]) == set(recipe["sources"])
    columns = pd.read_csv(ROOT / "data/raw/sample_submission.csv", nrows=0).columns[1:].tolist()
    truth = pd.read_csv(ROOT / "data/raw/kernel/Unit_commitment_decisions.csv",
                        usecols=["Run No", "starttime", *columns], dtype={column: "uint8" for column in columns})
    truth["starttime"] = pd.to_datetime(truth.starttime, utc=True)
    cases = truth.loc[truth.starttime.dt.year == 2021].sort_values("starttime").reset_index(drop=True)
    test_ids = pd.read_csv(ROOT / "data/raw/prediction_mapping.csv", usecols=["Run No"])["Run No"]
    assert not cases["Run No"].isin(test_ids).any()
    components = {}
    for name, path in monitor["prediction_paths"].items():
        frame = pd.read_csv(ROOT / path, float_precision="round_trip")
        assert frame.columns.tolist() == ["Run No", *columns]
        assert frame["Run No"].is_unique and set(frame["Run No"]) == set(cases["Run No"])
        components[name] = frame.set_index("Run No").loc[cases["Run No"], columns].to_numpy()

    # Independent reconstruction loops over individual output columns and the
    # unmodified submission recipe, rather than the helper's plant-index groups.
    independent = np.zeros((len(cases), len(columns)), dtype=np.float64)
    independent_weights = np.zeros(len(columns), dtype=np.float64)
    for i, column in enumerate(columns):
        plant = column.removeprefix("result_committed_").rsplit("_G", 1)[0]
        for name, coefficient in recipe["plant_weights"][plant].items():
            independent[:, i] += coefficient * components[name][:, i]
        independent_weights[i] = recipe["plant_weights"][plant].get("transformer_weighted", 0.0)
    constant, weights = frozen_ensemble_monitor(monitor, cases, columns)
    reconstructed = constant + weights * components["transformer_weighted"]
    np.testing.assert_array_equal(weights, independent_weights)
    np.testing.assert_array_equal(reconstructed, independent)
    expected_auc = .9889840634255942
    actual_auc = float(roc_auc_score(cases[columns].to_numpy().ravel(), reconstructed.ravel()))
    assert actual_auc == expected_auc, (actual_auc, expected_auc)
    np.testing.assert_array_equal(reconstructed[:, weights == 0], constant[:, weights == 0])
    reversed_cases = cases.iloc[::-1]
    reversed_constant, reversed_weights = frozen_ensemble_monitor(monitor, reversed_cases, columns)
    np.testing.assert_array_equal(reversed_constant, constant[::-1])
    np.testing.assert_array_equal(reversed_weights, weights)

    # Compile only the original monitor function; do not import any snapshot
    # training code or tree estimators into the Torch process.
    legacy_tree = ast.parse(args.legacy_source.read_text())
    legacy_node = next(node for node in legacy_tree.body if isinstance(node, ast.FunctionDef)
                       and node.name == "frozen_ensemble_monitor")
    legacy_module = ast.Module(body=[legacy_node], type_ignores=[])
    legacy_globals = {"np": np, "pd": pd, "ROOT": ROOT}
    exec(compile(legacy_module, str(args.legacy_source), "exec"), legacy_globals)
    legacy_expected = legacy_globals["frozen_ensemble_monitor"](old["ensemble_monitor"], cases, columns)
    legacy_actual = frozen_ensemble_monitor(old["ensemble_monitor"], cases, columns)
    for actual, expected in zip(legacy_actual, legacy_expected):
        np.testing.assert_array_equal(actual, expected)
    assert not any(name in sys.modules for name in ("xgboost", "lightgbm", "catboost"))
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in [
        ROOT / "src/train_cnn.py", ROOT / "src/component_monitor.py", Path(__file__).resolve(),
        recipe_path, old_config_path, *paths, *[ROOT / path for path in monitor["prediction_paths"].values()],
    ]}
    report = {
        "passed": True, "no_fitting": True, "year": 2021, "cases": len(cases), "outputs": len(columns),
        "design": {
            "mode": "replace_component", "replacement": "transformer_weighted",
            "expression": "sum(fixed_weight * historical_component) excluding replacement + replacement_weight * current_epoch_prediction",
            "checkpoint_selection_only": True, "old_transformer_contribution_frozen": True,
            "legacy_mode_default_preserved": True, "new_cache_float_precision": "round_trip",
            "training_threads": 4,
        },
        "synthetic": synthetic,
        "real_cached_identity": {
            "array_equal_to_independent_recipe": True, "max_absolute_difference": float(np.abs(reconstructed - independent).max()),
            "micro_roc_auc": actual_auc, "expected_micro_roc_auc": expected_auc,
            "reversed_validation_alignment_exact": True, "zero_weight_columns_unchanged": True,
            "replacement_weights_by_plant": {plant: amounts.get("transformer_weighted", 0.0)
                                              for plant, amounts in recipe["plant_weights"].items()},
        },
        "legacy_identity": {"array_equal_to_prechange_function": True,
                            "source": str(args.legacy_source), "source_sha256": digest(args.legacy_source)},
        "config_invariants": {"cloned_from": str(old_config_path.relative_to(ROOT)), "epochs": 39, "patience": 40,
                              "seed": 42, "device": "cpu", "architecture_and_loss_weights_unchanged": True,
                              "only_trial_variables": ["learning_rate", "weight_decay"]},
        "imports": {"helper_does_not_require_torch": True, "tree_libraries_blocked_and_absent": True},
        "sha256": hashes,
    }
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"passed": True, "micro_roc_auc": actual_auc, "report": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
