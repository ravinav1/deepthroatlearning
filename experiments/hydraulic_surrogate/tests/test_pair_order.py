"""Check scope, convex arithmetic and validation of the soft pair correction."""

import unittest

import numpy as np

from sintef_project.pair_order import soften_pair_order_violations


class PairOrderTests(unittest.TestCase):
    def setUp(self):
        self.units = ["Vinje_G1", "Tokke_G2", "Tokke_G1", "Vinje_G2"]
        self.values = np.array([[.9, .8, .2, .6], [.7, .1, .9, .4], [.4, .5, .5, .2]], dtype=np.float32)

    def test_only_violating_named_pair_changes(self):
        original = self.values.copy()
        result = soften_pair_order_violations(self.values, self.units, .5)
        np.testing.assert_allclose(result[0, [2, 1]], [.35, .65])
        np.testing.assert_array_equal(result[:, [0, 3]], original[:, [0, 3]])
        np.testing.assert_array_equal(result[1:], original[1:])
        np.testing.assert_array_equal(self.values, original)
        self.assertEqual(result.dtype, self.values.dtype)

    def test_endpoints_and_pair_total(self):
        np.testing.assert_array_equal(soften_pair_order_violations(self.values, self.units, 0), self.values)
        full = soften_pair_order_violations(self.values, self.units, 1)
        self.assertEqual(full[0, 1], full[0, 2])
        for alpha in [0., .25, .5, 1.]:
            result = soften_pair_order_violations(self.values, self.units, alpha)
            np.testing.assert_allclose(result[:, [1, 2]].sum(axis=1), self.values[:, [1, 2]].sum(axis=1), atol=1e-7)
            self.assertAlmostEqual(float(result[0, 1] - result[0, 2]), .6 * (1 - alpha), places=6)
            self.assertTrue(((result >= 0) & (result <= 1)).all())

    def test_invalid_probability_schema_and_alpha_fail(self):
        for alpha in [-.1, 1.1, float("nan")]:
            with self.assertRaises(ValueError):
                soften_pair_order_violations(self.values, self.units, alpha)
        invalid = self.values.copy(); invalid[0, 0] = np.nan
        with self.assertRaises(ValueError):
            soften_pair_order_violations(invalid, self.units, .5)
        with self.assertRaises(ValueError):
            soften_pair_order_violations(self.values, ["a"] * 4, .5)
        with self.assertRaises(ValueError):
            soften_pair_order_violations(self.values, self.units, .5, first="missing")


if __name__ == "__main__":
    unittest.main()
