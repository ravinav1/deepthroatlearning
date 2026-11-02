"""Verify weighted pooled ranking and complete-case block resampling."""
import unittest
import numpy as np
from sklearn.metrics import roc_auc_score
from sintef_project.auc_uncertainty import WeightedAUC, circular_block_weights


class AUCUncertaintyTests(unittest.TestCase):
    def test_weighted_ties_match_reference(self):
        y = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]])
        p = np.array([[.1, .5, .5, .7], [.1, .7, .8, .2], [.4, .8, .2, .2]])
        prepared = WeightedAUC.prepare(y, p, 3)
        for weights in [np.ones(3), np.array([2, 0, 1]), np.array([.5, 2, 3])]:
            expected = roc_auc_score(y.ravel(), p.ravel(), sample_weight=np.repeat(weights, 4))
            self.assertAlmostEqual(prepared.score(weights), expected, places=14)

    def test_complete_block_circularity(self):
        weights = circular_block_weights(np.random.default_rng(4173), 365, 28)
        self.assertEqual(weights.shape, (365,))
        self.assertEqual(weights.sum(), 365)
        np.testing.assert_array_equal(
            circular_block_weights(np.random.default_rng(1), 7, 7), np.ones(7))

    def test_invalid_inputs_rejected(self):
        with self.assertRaises(ValueError):
            WeightedAUC.prepare(np.array([0, 1]), np.array([0., np.nan]), 1)
        prepared = WeightedAUC.prepare(np.array([0, 0, 1, 1]), np.arange(4), 2)
        with self.assertRaises(ValueError):
            prepared.score(np.array([1, 0]))
        with self.assertRaises(ValueError):
            prepared.score(np.array([1, -1]))


if __name__ == "__main__":
    unittest.main()
