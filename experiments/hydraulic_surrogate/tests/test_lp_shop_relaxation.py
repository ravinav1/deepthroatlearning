"""Independent economic and feasibility checks for linked fractional states."""

import unittest

import numpy as np

from sintef_project.lp_features import HydraulicLP, Network
from sintef_project.lp_shop_relaxation import ShopRelaxationLP, state_interval


def fixture(horizon: int = 3) -> tuple[Network, dict]:
    """One reservoir, one unit, one segment, no bypass or forced flow."""
    network = Network(["r"], ["g"], np.array([0]), np.array([-1]),
                      np.array([], dtype=int), np.array([], dtype=int), np.array([-1]),
                      np.array([False]), np.array([], dtype=bool), np.array([False]))
    case = {"price": np.full(horizon, 100.), "initial": np.array([.5]),
            "capacity": np.array([10.]), "minimum": np.zeros((1, horizon)),
            "natural": np.zeros((1, horizon)), "terminal_value": np.array([100.]),
            "widths": np.array([[1.]]), "energy": np.array([[2.]]),
            "qmin": np.array([.5]), "qmax": np.array([1.]),
            "pmin": np.array([1.]), "pmax": np.array([2.]),
            "start_cost": np.array([20.]), "stop_cost": np.array([20.]),
            "bypass_minimum": np.zeros((0, horizon)),
            "vest_natural": np.zeros(horizon), "vest_minimum": np.zeros(horizon),
            "slack_penalty": 10000., "bypass_cost": 10., "spill_cost": 1000.}
    return network, case


class ShopRelaxationTests(unittest.TestCase):
    def test_fractional_state_cost_is_not_binary_or_q_over_qmin(self):
        network, case = fixture()
        case["price"][:] = [0, 100, 0]
        result = ShopRelaxationLP(network, 3, 1).solve(case)
        np.testing.assert_allclose(result["flow"], [[0, .5, 0]], atol=1e-10)
        np.testing.assert_allclose(result["state"], [[0, .5, 0]], atol=1e-10)
        self.assertAlmostEqual(result["transition_cost"], 20.)
        self.assertAlmostEqual(result["objective"], 80.)
        self.assertAlmostEqual(result["gross_objective"], 100.)
        self.assertAlmostEqual(result["state"][0, 1], result["flow"][0, 1] / case["qmax"][0])
        self.assertNotEqual(result["state"][0, 1], result["flow"][0, 1] / case["qmin"][0])

    def test_zero_cost_preserves_feasible_baseline_objective(self):
        network, case = fixture()
        case["price"][:] = [0, 100, 0]
        original = HydraulicLP(network, 3, 1).solve(case)
        linked = ShopRelaxationLP(network, 3, 1).solve(case, 0.)
        lower, upper = state_interval(case, original["flow"], original["power"])
        self.assertTrue((lower <= upper + 1e-10).all())
        self.assertAlmostEqual(original["objective"], linked["objective"])
        self.assertEqual(linked["transition_cost"], 0.)
        np.testing.assert_allclose(original["flow"], linked["flow"], atol=1e-10)

    def test_common_float64_rhs_separates_legacy_rounding_from_state_effect(self):
        network, case = fixture(1)
        case["initial"][:] = .50000003
        case["natural"] = np.zeros((1, 1), dtype=np.float32)
        legacy = HydraulicLP(network, 1, 1).solve(case)
        case["natural"] = case["natural"].astype(np.float64)
        promoted = HydraulicLP(network, 1, 1).solve(case)
        linked = ShopRelaxationLP(network, 1, 1).solve(case, 0.)
        self.assertGreater(abs(legacy["flow"][0, 0] - promoted["flow"][0, 0]), 1e-10)
        self.assertAlmostEqual(linked["objective"], promoted["objective"], places=9)
        self.assertAlmostEqual(linked["total_balance_error"], 0., places=10)

    def test_zero_production_forces_zero_state_and_boundaries_are_free(self):
        network, case = fixture(1)
        case["start_cost"][:] = 1e6
        case["stop_cost"][:] = 2e6
        on = ShopRelaxationLP(network, 1, 1).solve(case)
        self.assertAlmostEqual(on["power"][0, 0], 1.)
        self.assertEqual(on["transition_cost"], 0.)
        self.assertEqual(on["raw_start"].shape, (1, 0))
        case["initial"][:] = 0
        off = ShopRelaxationLP(network, 1, 1).solve(case)
        np.testing.assert_array_equal(off["state"], [[0.]])
        self.assertAlmostEqual(off["total_balance_error"], 0.)

    def test_incompatible_bounds_restrict_baseline_instead_of_false_equivalence(self):
        network, case = fixture(1)
        case["pmin"][:] = 3.  # Above the fixed feasible Pmax: only off is possible.
        original = HydraulicLP(network, 1, 1).solve(case)
        lower, upper = state_interval(case, original["flow"], original["power"])
        self.assertTrue((lower > upper).any())
        linked = ShopRelaxationLP(network, 1, 1).solve(case, 0.)
        self.assertLess(linked["objective"], original["objective"])
        self.assertAlmostEqual(linked["flow"].sum(), 0.)

    def test_negative_price_uses_frozen_average_and_conserves_water(self):
        network, case = fixture(1)
        case.update(price=np.array([-1.]), capacity=np.array([0.]),
                    widths=np.array([[.5, .5]]), energy=np.array([[4., 2.]]),
                    pmax=np.array([3.]), initial=np.array([.25]))
        original = HydraulicLP(network, 1, 2).solve(case)
        linked = ShopRelaxationLP(network, 1, 2).solve(case, 0.)
        np.testing.assert_allclose(linked["flow"], [[.25]], atol=1e-10)
        np.testing.assert_allclose(linked["power"], [[.75]], atol=1e-10)
        self.assertAlmostEqual(original["objective"], linked["objective"])
        self.assertAlmostEqual(linked["total_balance_error"], 0.)

    def test_asymmetric_cost_reconstruction_replay_and_input_immutability(self):
        network, case = fixture()
        case["price"][:] = [0, 100, 0]
        case["stop_cost"][:] = 10
        saved = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in case.items()}
        first = ShopRelaxationLP(network, 3, 1).solve(case)
        second = ShopRelaxationLP(network, 3, 1).solve(case)
        self.assertAlmostEqual(first["transition_cost"], 15.)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key], err_msg=key)
        for key in saved:
            np.testing.assert_array_equal(case[key], saved[key], err_msg=key)
        with self.assertRaises(ValueError):
            ShopRelaxationLP(network, 3, 1).solve(case, .5)


if __name__ == "__main__":
    unittest.main()
