"""Verify optimization arithmetic and input boundaries of physical features."""

import itertools
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import yaml

from sintef_project.physics import add_physics_features, switching_advantage


ROOT = Path(__file__).resolve().parents[1]


class SwitchingAdvantageTests(unittest.TestCase):
    def test_matches_exhaustive_binary_path_search(self) -> None:
        """Compare against independent exhaustive solutions with switch costs."""
        rng = np.random.default_rng(4173)
        profits = rng.normal(0, 10, (8, 7))
        paths = np.array(list(itertools.product((0, 1), repeat=7)))
        for start_cost, stop_cost in ((3.5, 1.5), (0, 0), (100, 2)):
            actual = switching_advantage(profits, start_cost, stop_cost)
            for case, profit in enumerate(profits):
                scores = paths @ profit
                scores -= start_cost * (np.diff(paths, axis=1) == 1).sum(axis=1)
                scores -= stop_cost * (np.diff(paths, axis=1) == -1).sum(axis=1)
                expected = [scores[paths[:, h] == 1].max() - scores[paths[:, h] == 0].max()
                            for h in range(7)]
                np.testing.assert_allclose(actual[case], expected, atol=1e-5)

    def test_no_switch_cost_reduces_to_hourly_profit(self) -> None:
        profit = np.array([[2.5, -3.5, 0], [-9, 5, 20]])
        np.testing.assert_allclose(switching_advantage(profit, 0, 0), profit)

    def test_rejects_nonfinite_or_wrong_shape(self) -> None:
        for profit in (np.array([1, 2]), np.array([[1, np.nan]])):
            with self.assertRaises(ValueError):
                switching_advantage(profit, 1, 1)


class PhysicsFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        topology = yaml.safe_load((ROOT / "sintef_project/data/extended/Tokke_Vinje_topology.yaml").read_text())
        cls.reservoirs = topology["model"]["reservoir"]

    def sample_features(self) -> pd.DataFrame:
        """Build two synthetic cases using schema only, without training labels."""
        values = {
            "horizon_hour": np.tile(np.arange(168), 2),
            "price": np.tile(np.linspace(10, 90, 168), 2),
        }
        for index, (reservoir, data) in enumerate(self.reservoirs.items()):
            values[f"volume_{reservoir}"] = np.full(336, data["max_vol"] * 0.5)
            values[f"water_value_{reservoir}"] = np.full(336, 40 + index)
            values[f"inflow_{reservoir}"] = np.full(336, index + 1)
            values[f"inflow_remaining_mm3_{reservoir}"] = np.tile(168 - np.arange(168), 2) * (index + 1) * 0.0036
        return pd.DataFrame(values, dtype=np.float32)

    def test_preserves_input_and_emits_finite_serializable_features(self) -> None:
        x = self.sample_features()
        before = x.copy(deep=True)
        result, spec = add_physics_features(x, ROOT)
        pd.testing.assert_frame_equal(x, before)
        pd.testing.assert_frame_equal(result[x.columns], before)
        self.assertEqual(len(spec["columns"]), 72)
        self.assertTrue(np.isfinite(result[spec["columns"]].to_numpy()).all())
        self.assertTrue(result[spec["columns"]].dtypes.eq("float32").all())
        json.dumps(spec)
        # Half-full Bandak is 71.23m from its supplied linear storage/head curve;
        # the fixed Hogga outlet is 60m. This is independent of generation labels.
        np.testing.assert_allclose(result["physics_head_Hogga"], 11.23, atol=1e-4)

    def test_cases_do_not_leak_into_each_other(self) -> None:
        x = self.sample_features()
        first, spec = add_physics_features(x, ROOT)
        x.loc[168:, "price"] += 1000
        for reservoir in self.reservoirs:
            x.loc[168:, f"volume_{reservoir}"] *= 0.1
        second, _ = add_physics_features(x, ROOT)
        pd.testing.assert_frame_equal(first.loc[:167, spec["columns"]], second.loc[:167, spec["columns"]])
        self.assertFalse(np.array_equal(first.loc[168:, spec["columns"]], second.loc[168:, spec["columns"]]))

    def test_unused_metadata_has_no_effect(self) -> None:
        x = self.sample_features()
        first, spec = add_physics_features(x, ROOT)
        x["total_calculation_time"] = np.arange(len(x)) * 1000
        x["result_committed_Hogga_G1_t0"] = np.arange(len(x)) % 2
        second, _ = add_physics_features(x, ROOT)
        pd.testing.assert_frame_equal(first[spec["columns"]], second[spec["columns"]])

    def test_rejects_incomplete_or_reordered_case(self) -> None:
        x = self.sample_features()
        with self.assertRaises(ValueError):
            add_physics_features(x.iloc[:-1], ROOT)
        with self.assertRaises(ValueError):
            add_physics_features(x.iloc[::-1], ROOT)


if __name__ == "__main__":
    unittest.main()
