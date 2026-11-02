"""Unit, source-scope and prefix-preservation checks for compact cache features."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from sintef_project.lp_features import Network, UNIT_ORDER
from sintef_project.lp_shop_features import COLUMNS, add_shop_features, compute_features, digest, feature_specification
from scripts.build_lp_shop_features import verify_saved_replay


def fixture() -> tuple[dict, dict, dict, Network]:
    """Independent known energy, flow, power and reservoir-dual differences."""
    network = Network(["up", "down"], UNIT_ORDER, np.zeros(14, dtype=int), np.ones(14, dtype=int),
                      np.array([], dtype=int), np.array([], dtype=int), np.array([-1, -1]),
                      np.zeros(14, dtype=bool), np.array([], dtype=bool), np.zeros(2, dtype=bool))
    case = {"price": np.full(168, 50.), "qmin": np.full(14, .25), "pmax": np.full(14, 2.),
            "energy": np.full((14, 4), 2.)}
    baseline = {"flow": np.full((14, 168), .5), "power": np.full((14, 168), 1.),
                "water_shadow": np.stack([np.full(168, 100.), np.full(168, 40.)])}
    result = {"state": np.full((14, 168), .5), "flow": np.full((14, 168), .75),
              "power": np.full((14, 168), 1.5),
              "water_shadow": np.stack([np.full(168, 90.), np.full(168, 40.)])}
    return case, result, baseline, network


class ShopFeatureTests(unittest.TestCase):
    def test_dimensionless_formulas_downstream_subtraction_and_column_order(self):
        case, result, baseline, network = fixture()
        features = compute_features(case, result, baseline, network)
        expected = np.tile(np.asarray([.5, 1., .25, 10/102], dtype=np.float32), 14)
        np.testing.assert_array_equal(features[0], expected)
        self.assertEqual(features.shape, (168, 56))
        self.assertEqual(COLUMNS[:4], [f"shop_{kind}_Hogga_G1" for kind in
                                      ["state", "delta_qmin_ratio", "delta_power_fraction", "delta_economic_margin"]])
        # Equal perturbations to upstream/downstream duals cancel exactly.
        result["water_shadow"] -= 25.
        np.testing.assert_array_equal(compute_features(case, result, baseline, network), features)

    def test_raw_arrays_remain_unchanged_and_unrelated_targets_are_unused(self):
        case, result, baseline, network = fixture()
        result["state"][0, 0] = -1e-9
        saved = {key: value.copy() for key, value in result.items()}
        before = compute_features(case, result, baseline, network)
        case["labels"] = np.full((14, 168), 999)
        case["total_calculation_time"] = 1e12
        np.testing.assert_array_equal(compute_features(case, result, baseline, network), before)
        self.assertEqual(before[0, 0], 0.)
        for key in saved:
            np.testing.assert_array_equal(saved[key], result[key])
        verify_saved_replay(result, saved)
        saved["flow"][0, 0] += 1e-8
        with self.assertRaises(AssertionError):
            verify_saved_replay(result, saved)

    def test_append_preserves_every_prefix_value_dtype_and_index_and_rejects_changes(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            project = root / "sintef_project"
            (project / "data/kernel").mkdir(parents=True)
            pd.DataFrame({"Run No": [1], "starttime": ["2015-01-01"]}).to_csv(project / "data/kernel/Unit_commitment_decisions.csv", index=False)
            pd.DataFrame(columns=["Run No", "starttime"]).to_csv(project / "prediction_mapping.csv", index=False)
            baseline = root / "baseline.json"
            baseline.write_text("{}")
            prefix = ["horizon_hour", *[f"original_{i}" for i in range(607)]]
            x = pd.DataFrame(np.arange(168*608, dtype=np.float32).reshape(168, 608), columns=prefix,
                             index=pd.Index(np.arange(168)+300))
            x["horizon_hour"] = np.arange(168, dtype=np.float32)
            original = x.copy(deep=True)
            cache = root / "cache"
            cache.mkdir()
            np.savez_compressed(cache / "features.npz", features=np.zeros((1, 168, 56), dtype=np.float32),
                                case_ids=np.array([1]), columns=np.asarray(COLUMNS), units=np.asarray(UNIT_ORDER))
            manifest = {"status": "complete", "feature_specification": feature_specification(),
                        "prefix_feature_columns": prefix, "source": {}, "data": {},
                        "baseline_manifest_path": "baseline.json", "baseline_manifest_sha256": digest(baseline),
                        "artifacts": {"features.npz": digest(cache / "features.npz")},
                        "case_dates": ["2015-01-01 00:00:00+00:00"]}
            (cache / "manifest.json").write_text(json.dumps(manifest))
            output, _ = add_shop_features(x, root, cache)
            pd.testing.assert_frame_equal(output.iloc[:, :608], original)
            pd.testing.assert_frame_equal(x, original)
            self.assertEqual(output.shape, (168, 664))
            with self.assertRaisesRegex(ValueError, "prefix differs"):
                add_shop_features(x[prefix[::-1]], root, cache)
            with (cache / "features.npz").open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "feature bytes changed"):
                add_shop_features(x, root, cache)


if __name__ == "__main__":
    unittest.main()
