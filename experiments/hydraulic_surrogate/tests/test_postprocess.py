"""Check case isolation, edge normalization, and named-unit projection."""
import unittest
import numpy as np

from sintef_project.postprocess import smooth_predictions, project_vinje


class PostprocessTests(unittest.TestCase):
    def test_smoothing_respects_case_boundary(self) -> None:
        p = np.r_[np.zeros((168, 1)), np.ones((168, 1))]
        np.testing.assert_array_equal(smooth_predictions(p, 25, 1), p)

    def test_edge_mean_and_blending(self) -> None:
        p = np.zeros((168, 1)); p[0] = 1
        result = smooth_predictions(p, 3, .5)
        self.assertAlmostEqual(result[0, 0], .75)
        self.assertAlmostEqual(result[1, 0], 1 / 6)
        self.assertEqual(result[2, 0], 0)
        self.assertEqual(p[0, 0], 1)

    def test_identity_and_invalid_input(self) -> None:
        p = np.ones((168, 2)) / 2
        np.testing.assert_array_equal(smooth_predictions(p, 7, 0), p)
        for window in [0, 2, 168, 3.5, True]:
            with self.assertRaises(ValueError): smooth_predictions(p, window, .5)
        for invalid in [np.zeros((169, 2)), np.full((168, 1), np.nan), np.full((168, 1), 2)]:
            with self.assertRaises(ValueError): smooth_predictions(invalid, 3, 1)
        with self.assertRaises(ValueError): smooth_predictions(p, 3, 1.1)

    def test_projection_named_order_and_sum(self) -> None:
        units = ["Vinje_G3", "unrelated", "Vinje_G1", "Vinje_G2"]
        p = np.tile([.7, .9, .1, .4], (168, 1))
        result = project_vinje(p, units)
        np.testing.assert_allclose(result[:, [2, 3, 0]], .4)
        np.testing.assert_array_equal(result[:, 1], p[:, 1])
        np.testing.assert_allclose(result.sum(1), p.sum(1))
        np.testing.assert_array_equal(project_vinje(result, units), result)
        np.testing.assert_array_equal(p[0], [.7, .9, .1, .4])

    def test_single_violating_pair(self) -> None:
        p = np.tile([.8, .1, .5], (168, 1))
        result = project_vinje(p, ["Vinje_G1", "Vinje_G2", "Vinje_G3"])
        np.testing.assert_allclose(result, np.tile([.8, .3, .3], (168, 1)))


if __name__ == "__main__":
    unittest.main()
