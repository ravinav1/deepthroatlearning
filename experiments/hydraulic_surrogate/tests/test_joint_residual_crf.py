"""Exact small-chain enumeration, gradients and independent-case identity."""
from itertools import product
import unittest
import numpy as np
from scipy.special import logsumexp
from sintef_project.joint_residual_crf import (STATES, forward_backward, logits,
    objective_gradient, observed_statistics, predict, unary_objective)


class JointResidualCRFTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(4173)
        self.p = rng.uniform(.1, .9, (2, 3, 3))
        self.z = rng.normal(size=(2, 3, 4)); self.z[:, :, 0] = 1
        self.theta = rng.normal(scale=.1, size=36)
        self.y = rng.integers(0, 2, size=(2, 3, 3))

    def test_brute_force_partition_and_marginals(self):
        partition, marginal, _ = forward_backward(self.p, self.z, self.theta)
        for c in range(2):
            paths = np.asarray([STATES[list(path)] for path in product(range(8), repeat=3)])
            scores = np.asarray([np.sum(logits(self.p[c:c + 1]) * path) +
                                 self.theta @ observed_statistics(path[None], self.z[c:c + 1]) for path in paths])
            log_z = logsumexp(scores)
            expected = np.einsum('n,ntg->tg', np.exp(scores - log_z), paths)
            np.testing.assert_allclose(partition[c], log_z, atol=1e-12, rtol=0)
            np.testing.assert_allclose(marginal[c] @ STATES, expected, atol=1e-12, rtol=0)

    def test_analytic_gradients(self):
        for function, theta in [(objective_gradient, self.theta), (unary_objective, self.theta[:12])]:
            _, gradient = function(theta, self.p, self.y, self.z)
            difference = np.empty_like(theta)
            for j in range(len(theta)):
                delta = np.zeros_like(theta); delta[j] = 1e-5
                difference[j] = (function(theta + delta, self.p, self.y, self.z)[0] -
                                 function(theta - delta, self.p, self.y, self.z)[0]) / 2e-5
            np.testing.assert_allclose(gradient, difference, rtol=1e-5, atol=1e-9)

    def test_zero_identity_and_case_isolation(self):
        for kind, size in [('joint', 36), ('unary', 12)]:
            np.testing.assert_array_equal(self.p, predict(self.p, self.z, {'kind': kind, 'theta': [0] * size}))
        _, marginal, _ = forward_backward(self.p, self.z, np.zeros(36))
        np.testing.assert_allclose(marginal @ STATES, self.p, atol=1e-12, rtol=0)
        together = predict(self.p, self.z, {'kind': 'joint', 'theta': self.theta})
        separate = np.concatenate([predict(self.p[i:i + 1], self.z[i:i + 1],
                                            {'kind': 'joint', 'theta': self.theta}) for i in range(2)])
        np.testing.assert_allclose(together, separate, atol=1e-14, rtol=0)


if __name__ == '__main__':
    unittest.main()
