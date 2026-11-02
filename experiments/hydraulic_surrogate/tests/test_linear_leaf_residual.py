"""Check feature selection, train-only scaling, offsets and linear-leaf replay."""
from pathlib import Path
import tempfile
import unittest

import lightgbm as lgb
import numpy as np
from scipy.special import expit

from sintef_project.linear_leaf_residual import (
    select_gain_features, residual_sample, fit_normalizer, normalize, fit_unit, leaf_geometry,
)
from sintef_project.pooled_pairwise_residual import apply_residual, base_logits
from scripts.screen_linear_leaf_residual import ResourceBudget, ResourceBudgetExceeded


class LinearLeafTests(unittest.TestCase):
    def test_resource_budget_rejects_before_next_stage(self):
        clock = [0.]
        memory = [100]
        budget = ResourceBudget(10., 200, clock=lambda: clock[0], memory=lambda: memory[0])
        budget.check('fit')
        clock[0] = 11.
        with self.assertRaises(ResourceBudgetExceeded): budget.check('before labels')
        clock[0] = 1.; memory[0] = 201
        with self.assertRaises(ResourceBudgetExceeded): budget.check('before labels')

    def test_gain_ties_and_shared_sample(self):
        np.testing.assert_array_equal(select_gain_features(np.array([1., 3., 3., 0.]), 3), [1, 2, 0])
        sample = residual_sample(121632)
        self.assertEqual(len(sample), 48652)
        np.testing.assert_array_equal(sample, residual_sample(121632))
        self.assertTrue(np.all(np.diff(sample) > 0))
        with self.assertRaises(ValueError): select_gain_features(np.array([1., np.nan]))

    def test_training_normalizer_retains_missing_and_zero(self):
        train = np.array([[0., 7., np.nan], [2., 7., 4.], [4., 7., 8.]])
        mean, scale = fit_normalizer(train)
        np.testing.assert_allclose(mean, [2., 7., 6.])
        np.testing.assert_allclose(scale, [np.sqrt(8/3), 1., 2.])
        transformed = normalize(train, mean, scale)
        self.assertTrue(np.isnan(transformed[0, 2]))
        self.assertFalse(np.isnan(transformed[0, 0]))
        np.testing.assert_array_equal(transformed[:, 1], np.zeros(3))
        screen = normalize(np.array([[1000., 7., 10.]]), mean, scale)
        self.assertGreater(screen[0, 0], 100.)
        np.testing.assert_allclose(mean, [2., 7., 6.])
        with self.assertRaises(ValueError): fit_normalizer(np.array([[np.nan]]))

    def test_linear_geometry_offset_and_saved_replay(self):
        rng = np.random.default_rng(4173)
        x = rng.normal(size=(4096, 4)).astype(np.float32)
        p = expit(.3*x[:, 0]).astype(np.float32)
        y = (rng.random(len(x)) < expit(1.4*x[:, 0]-.9*x[:, 1]+.5*x[:, 2])).astype(np.uint8)
        params = {'num_threads': 2, 'tree_learner': 'serial', 'device_type': 'cpu',
                  'verbosity': -1, 'seed': 4173, 'data_random_seed': 4173,
                  'deterministic': True, 'force_col_wise': True,
                  'max_depth': 3, 'num_leaves': 8, 'learning_rate': .05,
                  'min_data_in_leaf': 512, 'lambda_l2': 10., 'linear_lambda': 10., 'max_bin': 255}
        for linear in [False, True]:
            model, trace = fit_unit(x, y, p, params, 4, linear)
            self.assertEqual(len(trace), 4)
            residual = model.predict(x, raw_score=True, num_threads=2)
            geometry = leaf_geometry(model)
            self.assertEqual(geometry['nonzero_linear_coefficients_by_tree'][0], 0)
            self.assertEqual(geometry['total_nonzero_linear_coefficients'] > 0, linear)
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder)/'model.txt'; model.save_model(str(path))
                replay = lgb.Booster(model_file=str(path)).predict(x, raw_score=True, num_threads=2)
            np.testing.assert_array_equal(residual, replay)
            expected = expit(base_logits(p)+residual)
            np.testing.assert_array_equal(apply_residual(p, replay), expected)
            self.assertGreater(np.max(np.abs(expected-expit(residual))), .01)
        result = apply_residual(p, np.zeros_like(p))
        self.assertEqual(result.dtype, p.dtype)
        self.assertEqual(result.tobytes(), p.tobytes())


if __name__ == '__main__': unittest.main()
