"""Small exhaustive checks for binary commitment with an actual water budget."""
import itertools
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import linprog

from sintef_project.integer_water_budget import IntegerWaterBudgetLP
from sintef_project.lp_shop_relaxation import ShopRelaxationLP
from test_lp_shop_relaxation import fixture


class IntegerBudgetTests(unittest.TestCase):
    def test_integer_result_matches_exhaustive_fixed_paths(self):
        network, case = fixture(3)
        case['price'][:] = [0, 100, 0]
        solver = IntegerWaterBudgetLP(network, 3, 1)
        result, proof = solver.solve_integer(case, [0])
        self.assertTrue(proof['proven_optimal'])
        np.testing.assert_array_equal(result['state'], [[0, 1, 0]])
        self.assertAlmostEqual(result['objective'], 60.)
        self.assertAlmostEqual(result['transition_cost'], 40.)
        relaxed = ShopRelaxationLP(network, 3, 1).solve(case)
        self.assertGreater(relaxed['objective'], result['objective'])
        objectives = []
        for path in itertools.product([0, 1], repeat=3):
            fixed_solver = ShopRelaxationLP(network, 3, 1)
            def adapter(c, **kwargs):
                bounds = kwargs['bounds'].copy()
                bounds[fixed_solver.state.ravel(), 0] = path
                bounds[fixed_solver.state.ravel(), 1] = path
                return linprog(c, **{**kwargs, 'bounds': bounds})
            try:
                with patch('sintef_project.lp_shop_relaxation.linprog', adapter):
                    candidate = fixed_solver.solve(case)
                objectives.append(candidate['objective'])
            except RuntimeError:
                pass
        self.assertAlmostEqual(result['objective'], max(objectives))
        self.assertTrue(proof['fixed_lp_replay_bitwise'])

    def test_insufficient_water_cannot_create_fractional_on_state(self):
        network, case = fixture(1)
        case['initial'][:] = .25
        result, proof = IntegerWaterBudgetLP(network, 1, 1).solve_integer(case, [0])
        self.assertTrue(proof['proven_optimal'])
        np.testing.assert_array_equal(result['state'], [[0.]])
        self.assertAlmostEqual(result['flow'].sum(), 0.)
        self.assertAlmostEqual(result['storage'][0, -1], .25)


if __name__ == '__main__':
    unittest.main()
