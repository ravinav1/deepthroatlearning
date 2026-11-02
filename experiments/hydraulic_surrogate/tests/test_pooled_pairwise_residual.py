"""Numerical and integration checks for the pooled residual ranking objective."""
import tempfile
import unittest
from pathlib import Path

import lightgbm as lgb
import numpy as np
from scipy.special import expit

from sintef_project.pooled_pairwise_residual import (
    PooledPairwiseObjective, PointwiseBCEObjective, base_logits,
    apply_residual, bce_derivatives, pair_derivatives,
)


class FakeDataset:
    def __init__(self, labels):
        self.labels = labels

    def get_label(self):
        return self.labels


class PooledResidualTests(unittest.TestCase):
    def test_pair_gradient_curvature_and_majorizer(self):
        scores = np.array([-.7, .3, 1.2, -.1, .8])
        positive = np.array([0, 0, 2, 4, 2, 4])
        negative = np.array([1, 3, 1, 3, 3, 1])
        scale = len(scores) / (2 * len(positive))
        loss, grad, diagonal = pair_derivatives(scores, positive, negative, scale)
        exact_loss, exact_grad, hessian = 0., np.zeros(5), np.zeros((5, 5))
        for a, b in zip(positive, negative):
            q = expit(scores[b] - scores[a])
            vector = np.zeros(5); vector[a] = 1; vector[b] = -1
            exact_loss += np.logaddexp(0, scores[b] - scores[a]) * scale
            exact_grad -= q * vector * scale
            hessian += q * (1 - q) * np.outer(vector, vector) * scale
        self.assertAlmostEqual(loss, exact_loss, places=14)
        np.testing.assert_allclose(grad, exact_grad, atol=1e-15)
        np.testing.assert_allclose(diagonal, np.diag(hessian), atol=1e-15)
        eps = 1e-5
        finite_gradient = np.zeros(5); finite_hessian = np.zeros((5, 5))
        for i in range(5):
            shift = np.eye(5)[i] * eps
            plus = pair_derivatives(scores + shift, positive, negative, scale)
            minus = pair_derivatives(scores - shift, positive, negative, scale)
            finite_gradient[i] = (plus[0] - minus[0]) / (2 * eps)
            finite_hessian[:, i] = (plus[1] - minus[1]) / (2 * eps)
        np.testing.assert_allclose(grad, finite_gradient, atol=1e-10)
        np.testing.assert_allclose(hessian, finite_hessian, atol=1e-10)
        self.assertGreaterEqual(np.linalg.eigvalsh(2 * np.diag(diagonal) - hessian).min(), -1e-12)
        shifted = pair_derivatives(scores + 17, positive, negative, scale)
        np.testing.assert_allclose(shifted[1], grad, atol=1e-14)
        self.assertAlmostEqual(shifted[0], loss, places=13)
        self.assertAlmostEqual(grad.sum(), 0, places=14)

    def test_bce_reference_and_balanced_normalization(self):
        y = np.array([1, 1, 0, 0])
        scores = np.zeros(4)
        positive, negative = np.array([0, 0, 1, 1]), np.array([2, 3, 2, 3])
        _, pair_grad, pair_diag = pair_derivatives(scores, positive, negative, 4 / 8)
        _, bce_grad, bce_hess = bce_derivatives(scores, y)
        np.testing.assert_array_equal(pair_grad, bce_grad)
        np.testing.assert_array_equal(pair_diag, bce_hess)
        scores = np.array([-.3, .7, 1.2, -.8]); eps = 1e-5
        _, grad, hess = bce_derivatives(scores, y)
        for i in range(4):
            shift = np.eye(4)[i] * eps
            plus, minus = bce_derivatives(scores + shift, y), bce_derivatives(scores - shift, y)
            self.assertAlmostEqual(grad[i], (plus[0] - minus[0]) / (2 * eps), places=9)
            self.assertAlmostEqual(hess[i], (plus[1][i] - minus[1][i]) / (2 * eps), places=9)

    def test_uniform_population_draw_replay_and_cross_groups(self):
        labels = np.tile([1, 0], 24)
        initial = np.zeros(48)
        first = PooledPairwiseObjective(labels, initial, pairs_per_round=100_000)
        second = PooledPairwiseObjective(labels, initial, pairs_per_round=100_000)
        positive, negative = first.sample_pairs(0)
        p2, n2 = second.sample_pairs(0)
        np.testing.assert_array_equal(positive, p2); np.testing.assert_array_equal(negative, n2)
        self.assertTrue(np.all(labels[positive] == 1) and np.all(labels[negative] == 0))
        self.assertTrue(np.any(positive // 12 != negative // 12))
        self.assertTrue(np.any(positive % 4 != negative % 4))
        counts = np.bincount(positive, minlength=48)[labels == 1]
        self.assertLess(np.max(np.abs(counts - counts.mean())) / counts.mean(), .06)
        next_positive, _ = first.sample_pairs(1)
        self.assertFalse(np.array_equal(positive, next_positive))
        g1, h1 = first(initial, FakeDataset(labels)); g2, h2 = second(initial, FakeDataset(labels))
        np.testing.assert_array_equal(g1, g2); np.testing.assert_array_equal(h1, h2)
        self.assertEqual(first.trace, second.trace)

    def test_unbalanced_population_matches_balanced_bce_at_zero(self):
        labels = np.array([1, 0, 0, 0, 1, 0, 0])
        scores = np.zeros(7)
        positive, negative = np.meshgrid(np.flatnonzero(labels == 1), np.flatnonzero(labels == 0), indexing='ij')
        positive, negative = positive.ravel(), negative.ravel()
        _, grad, diagonal = pair_derivatives(scores, positive, negative, len(scores) / (2 * len(positive)))
        control = PointwiseBCEObjective(labels, scores, class_balanced=True)
        _, bce_grad, bce_hessian = bce_derivatives(scores, labels, control.weights)
        np.testing.assert_allclose(grad, bce_grad, atol=1e-15)
        np.testing.assert_allclose(diagonal, bce_hessian, atol=1e-15)
        self.assertAlmostEqual(control.weights.mean(), 1, places=14)

    def test_zero_identity_and_invalid_inputs(self):
        probability = np.array([0., .1, .7, 1.], dtype=np.float32)
        result = apply_residual(probability, np.zeros(4))
        self.assertEqual(result.dtype, probability.dtype)
        self.assertEqual(result.tobytes(), probability.tobytes())
        with self.assertRaises(ValueError):
            PointwiseBCEObjective(np.array([.5, 1]), np.zeros(2))
        with self.assertRaises(ValueError):
            PointwiseBCEObjective(np.array([0, 1]), np.array([0, np.inf]))
        with self.assertRaises(ValueError):
            base_logits(probability, epsilon=0)

    def test_interior_near_endpoint_order_is_preserved(self):
        probability = np.array([1e-20, 1e-7, 2e-7, 4e-7, 1 - 4e-7, 1 - 2e-7, 1 - 1e-7], dtype=np.float32)
        logits = base_logits(probability)
        self.assertTrue(np.all(np.diff(logits) > 0))
        exact = probability.astype(np.float64)
        np.testing.assert_array_equal(logits, np.log(exact) - np.log1p(-exact))
        np.testing.assert_allclose(expit(logits), exact, atol=1e-15, rtol=1e-14)

    def test_offset_and_saved_model_replay(self):
        rng = np.random.default_rng(22)
        x = rng.normal(size=(160, 4)).astype(np.float32)
        labels = (x[:, 0] + .4 * x[:, 1] > 0).astype(np.uint8)
        initial = base_logits(expit(.3 * x[:, 0]))
        dataset = lgb.Dataset(x, label=labels, init_score=initial,
                              params={'num_threads': 1, 'min_data_in_leaf': 5})
        for kind in ['pairwise', 'bce', 'balanced_bce']:
            state = (PooledPairwiseObjective(labels, initial, pairs_per_round=512) if kind == 'pairwise'
                     else PointwiseBCEObjective(labels, initial, class_balanced=kind == 'balanced_bce'))
            def objective(scores, dataset):
                return state(scores, dataset)
            model = lgb.train({'objective': objective, 'verbosity': -1, 'num_threads': 1,
                               'max_depth': 2, 'num_leaves': 4, 'learning_rate': .03,
                               'min_data_in_leaf': 5, 'lambda_l2': 1, 'seed': 4173,
                               'deterministic': True, 'force_col_wise': True}, dataset, num_boost_round=3)
            self.assertEqual(len(state.trace), 3)
            residual = model.predict(x, raw_score=True)
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / 'model.txt'; model.save_model(str(path))
                restored = lgb.Booster(model_file=str(path)).predict(x, raw_score=True, num_threads=1)
            np.testing.assert_array_equal(residual, restored)
            np.testing.assert_array_equal(expit(initial + residual), expit(initial + restored))
        np.testing.assert_allclose(expit(base_logits(np.array([.1, .7]))), [.1, .7], atol=1e-15)
        self.assertTrue(np.isfinite(base_logits(np.array([0., 1.]))).all())


if __name__ == '__main__':
    unittest.main()
