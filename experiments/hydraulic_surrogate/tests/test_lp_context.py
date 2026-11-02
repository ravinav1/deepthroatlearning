"""Check LP temporal features without solving or changing the frozen LP."""
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from sintef_project.lp_context import UNITS, add_lp_context_features
from sintef_project.normalized import add_normalized_features


def feature_frame(q: np.ndarray) -> pd.DataFrame:
    """Build complete synthetic cases with the same ratio for all units."""
    q = np.asarray(q).ravel()
    values = {"horizon_hour": np.tile(np.arange(168), len(q) // 168),
              "price": np.full(len(q), 10.)}
    values.update({f"lp_qmin_ratio_{unit}": q.copy() for unit in UNITS})
    return pd.DataFrame(values)


class LPContextTests(unittest.TestCase):
    def test_lags_leads_and_six_hour_means(self) -> None:
        result, _ = add_lp_context_features(feature_frame(np.arange(168)), Path("/unused"))
        unit = UNITS[0]
        self.assertEqual(result[f"lp_context_qmin_previous_3h_{unit}"].iloc[1], 0)
        self.assertEqual(result[f"lp_context_qmin_next_3h_{unit}"].iloc[-2], 167)
        self.assertEqual(result[f"lp_context_qmin_previous_1h_{unit}"].iloc[10], 9)
        self.assertEqual(result[f"lp_context_qmin_next_1h_{unit}"].iloc[10], 11)
        self.assertEqual(result[f"lp_context_qmin_previous_6h_mean_{unit}"].iloc[10], 6.5)
        self.assertEqual(result[f"lp_context_qmin_next_6h_mean_{unit}"].iloc[10], 13.5)
        self.assertEqual(result[f"lp_context_qmin_previous_6h_mean_{unit}"].iloc[0], 0)
        self.assertEqual(result[f"lp_context_qmin_next_6h_mean_{unit}"].iloc[-1], 167)

    def test_case_isolation_float32_and_no_mutation(self) -> None:
        original = feature_frame(np.r_[np.full(168, .25), np.full(168, 2.)])
        saved = original.copy(deep=True)
        result, spec = add_lp_context_features(original, Path("/unused"))
        pd.testing.assert_frame_equal(original, saved)
        self.assertEqual(len(spec["added_columns"]), 126)
        self.assertTrue(all(result[c].dtype == np.float32 for c in spec["added_columns"]))
        for unit in UNITS:
            for suffix in ["previous_1h", "next_3h", "previous_6h_mean", "next_6h_mean", "week_mean"]:
                np.testing.assert_array_equal(result[f"lp_context_qmin_{suffix}_{unit}"], original[f"lp_qmin_ratio_{unit}"])

    def test_week_fraction_includes_all_hours_and_exact_ties(self) -> None:
        q = np.zeros(168); q[:4] = [1., 2., .5, 1e-8]
        result, _ = add_lp_context_features(feature_frame(q), Path("/unused"))
        unit = UNITS[0]
        np.testing.assert_allclose(result[f"lp_context_qmin_week_mean_{unit}"], q.mean())
        np.testing.assert_allclose(result[f"lp_context_qmin_week_at_least_one_fraction_{unit}"], 2 / 168)
        np.testing.assert_allclose(result[f"lp_context_qmin_week_positive_fraction_{unit}"], 4 / 168)

    def test_complete_order_finite_inputs_and_duplicate_guard(self) -> None:
        original = feature_frame(np.zeros(168))
        for invalid in [original.iloc[::4], original.iloc[::-1], original.drop(columns=[f"lp_qmin_ratio_{UNITS[0]}"])]:
            with self.assertRaises(ValueError): add_lp_context_features(invalid, Path("/unused"))
        invalid = original.copy(); invalid.loc[0, f"lp_qmin_ratio_{UNITS[0]}"] = np.inf
        with self.assertRaises(ValueError): add_lp_context_features(invalid, Path("/unused"))
        result, _ = add_lp_context_features(original, Path("/unused"))
        with self.assertRaises(ValueError): add_lp_context_features(result, Path("/unused"))

    def test_dimensionless_context_survives_normalization(self) -> None:
        result, spec = add_lp_context_features(feature_frame(np.arange(168) / 50), Path("/unused"))
        normalized, _ = add_normalized_features(result, Path("/unused"))
        pd.testing.assert_frame_equal(result[spec["added_columns"]], normalized[spec["added_columns"]])


if __name__ == "__main__":
    unittest.main()
