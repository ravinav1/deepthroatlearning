"""Check monetary normalization and case boundaries without training data."""

from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from sintef_project.normalized import add_normalized_features


class NormalizedFeatureTests(unittest.TestCase):
    def fixture(self) -> pd.DataFrame:
        price = np.concatenate([np.tile([-2.0, 2.0], 84), np.tile([-9.0, 9.0], 84)])
        return pd.DataFrame({
            "horizon_hour": np.tile(np.arange(168), 2),
            "price": price,
            "price_offset_+1": price,
            "price_week_mean": np.zeros(336),
            "price_week_std": np.repeat([2.0, 9.0], 168),
            "price_minus_water_Bandak": price - 6,
            "water_value_Bandak": np.full(336, 6.0),
            "price_over_water_Bandak": price / 7,
            "price_week_rank": np.tile([0.25, 0.75], 168),
            "price_week_zscore": price / np.repeat([3.0, 10.0], 168),
            "physics_switch_margin_minimum_Hogga": price - 4,
            "physics_head_Hogga": np.full(336, 12.0),
            "volume_Bandak": np.full(336, 100.0),
            "inflow_Bandak": np.ones(336),
        }, dtype=np.float32)

    def test_correct_case_scale_and_no_raw_monetary_duplicates(self) -> None:
        x = self.fixture()
        before = x.copy(deep=True)
        actual, spec = add_normalized_features(x, Path("/nonexistent"))
        scale = np.repeat([3.0, 10.0], 168)
        np.testing.assert_allclose(actual["monetary_scale_inverse"], 1 / scale)
        np.testing.assert_allclose(actual["norm_price"], x["price"] / scale)
        np.testing.assert_allclose(actual["norm_water_value_Bandak"], 6 / scale)
        np.testing.assert_allclose(actual["norm_physics_switch_margin_minimum_Hogga"], (x["price"] - 4) / scale)
        self.assertEqual(len(actual.columns), len(x.columns) + 1)
        self.assertFalse(any(column in actual for column in spec["replaced_columns"]))
        pd.testing.assert_frame_equal(x, before)

    def test_relative_and_nonmonetary_features_are_preserved(self) -> None:
        x = self.fixture()
        actual, spec = add_normalized_features(x, Path("."))
        pd.testing.assert_frame_equal(actual[spec["retained_columns"]], x[spec["retained_columns"]])
        for column in ("price_over_water_Bandak", "price_week_rank", "price_week_zscore"):
            self.assertIn(column, spec["preserved_relative_features"])

    def test_label_and_solver_metadata_do_not_change_derived_features(self) -> None:
        x = self.fixture()
        baseline, spec = add_normalized_features(x, Path("."))
        x["result_committed_Hogga_G1_t0"] = np.arange(len(x)) % 2
        x["total_calculation_time"] = np.arange(len(x)) * 100
        changed, _ = add_normalized_features(x, Path("."))
        derived = spec["normalized_columns"] + ["monetary_scale_inverse"]
        pd.testing.assert_frame_equal(baseline[derived], changed[derived])

    def test_other_cases_cannot_change_scale(self) -> None:
        x = self.fixture()
        baseline, spec = add_normalized_features(x, Path("."))
        x.loc[168:, "price"] *= 100
        changed, _ = add_normalized_features(x, Path("."))
        pd.testing.assert_frame_equal(baseline.loc[:167], changed.loc[:167])
        self.assertNotEqual(baseline.loc[168, "monetary_scale_inverse"], changed.loc[168, "monetary_scale_inverse"])

    def test_zero_price_is_finite_and_invalid_inputs_fail(self) -> None:
        x = self.fixture()
        x["price"] = 0
        actual, _ = add_normalized_features(x, Path("."))
        np.testing.assert_array_equal(actual["monetary_scale_inverse"], np.ones(len(x)))
        self.assertTrue(np.isfinite(actual.to_numpy()).all())
        for invalid in (x.iloc[:0], x.iloc[:-1], x.iloc[::-1]):
            with self.assertRaises(ValueError):
                add_normalized_features(invalid, Path("."))
        x.loc[0, "price"] = np.nan
        with self.assertRaises(ValueError):
            add_normalized_features(x, Path("."))


if __name__ == "__main__":
    unittest.main()
