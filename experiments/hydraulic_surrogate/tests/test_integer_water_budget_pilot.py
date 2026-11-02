"""Verify that the recorder preserves both independent fixed-state LP outputs."""
import unittest
import numpy as np
from scripts.pilot_integer_water_budget import capture_solve
from test_lp_shop_relaxation import fixture


class IntegerPilotTests(unittest.TestCase):
    def test_saves_both_independent_conditional_solutions_and_duals(self):
        network,case=fixture(3)
        case['price'][:]=[0,100,0]
        result,proof,saved=capture_solve(network,case,[0],2.)
        self.assertTrue(proof['proven_optimal'])
        self.assertEqual(len(saved),2)
        self.assertAlmostEqual(result['objective'],60.)
        for name in ['solution','eqlin_marginals','ineqlin_marginals','lower_marginals','upper_marginals']:
            np.testing.assert_array_equal(saved[0][name],saved[1][name])
            self.assertFalse(np.shares_memory(saved[0][name],saved[1][name]))


if __name__=='__main__':
    unittest.main()
