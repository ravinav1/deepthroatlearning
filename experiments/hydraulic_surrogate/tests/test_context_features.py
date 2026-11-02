"""Semantic checks for known-week context, boundary handling, and normalization."""
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from sintef_project.context_features import add_context_features
from sintef_project.normalized import add_normalized_features


def frame(price: np.ndarray) -> pd.DataFrame:
    """Construct complete case rows for isolated feature tests."""
    return pd.DataFrame({"price": np.asarray(price).ravel(),
                         "horizon_hour": np.tile(np.arange(168), np.asarray(price).size // 168)})


class ContextFeatureTests(unittest.TestCase):
    def test_midpoint_ties_and_suffix(self) -> None:
        price = np.zeros(168); price[-3:] = [2, 2, 1]
        x, _ = add_context_features(frame(price), Path("/unused"))
        self.assertAlmostEqual(x.context_rank_remaining.iloc[-3], 2 / 3, places=6)
        self.assertAlmostEqual(x.context_rank_remaining.iloc[-2], .75)
        self.assertEqual(x.context_rank_remaining.iloc[-1], .5)

    def test_means_exclude_current_and_normalize_edges(self) -> None:
        x, _ = add_context_features(frame(np.arange(168)), Path("/unused"))
        self.assertEqual(x.price_previous_mean_6h.iloc[10], 6.5)
        self.assertEqual(x.price_next_mean_6h.iloc[10], 13.5)
        self.assertEqual(x.price_previous_mean_6h.iloc[0], 0)
        self.assertEqual(x.price_next_mean_6h.iloc[0], 3.5)
        self.assertEqual(x.price_previous_mean_24h.iloc[1], 0)
        self.assertEqual(x.price_next_mean_6h.iloc[-1], 167)
        self.assertEqual(x.price_minus_next_mean_6h.iloc[10], -3.5)

    def test_cases_isolated_and_input_unchanged(self) -> None:
        original = frame(np.r_[np.full(168, 2), np.full(168, 100)])
        saved = original.copy(deep=True)
        result, spec = add_context_features(original, Path("/unused"))
        pd.testing.assert_frame_equal(original, saved)
        np.testing.assert_array_equal(result.context_rank_remaining, .5)
        np.testing.assert_array_equal(result.price_previous_mean_24h, original.price)
        np.testing.assert_array_equal(result.price_next_mean_24h, original.price)
        self.assertEqual(len(spec["added_columns"]), 13)

    def test_full_horizon_before_sampling(self) -> None:
        x = frame(np.arange(168))
        with self.assertRaises(ValueError): add_context_features(x.iloc[::4], Path("/unused"))
        with self.assertRaises(ValueError): add_context_features(x.iloc[::-1], Path("/unused"))
        x.loc[10, "price"] = np.nan
        with self.assertRaises(ValueError): add_context_features(x, Path("/unused"))

    def test_dimensionless_rank_survives_normalization(self) -> None:
        context, _ = add_context_features(frame(np.arange(168)), Path("/unused"))
        normalized, _ = add_normalized_features(context, Path("/unused"))
        np.testing.assert_array_equal(normalized.context_rank_remaining, context.context_rank_remaining)
        np.testing.assert_allclose(normalized.norm_price_next_mean_6h,
                                   context.price_next_mean_6h / 84.5, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
