"""Check fractional transitions, faithful physical state and subset preservation."""

from pathlib import Path
import unittest

import numpy as np

from sintef_project.lp_features import HydroCaseData
from sintef_project.lp_joint_plant import JointCurve, JointPlantLP, build_joint_curve
from sintef_project.lp_joint_switching import SwitchingJointLP, build_subset_joint_curve
from test_lp_features import network, simple_case


def one_mode_curve() -> JointCurve:
    """One physical active mode: one volume and one MWh per hour."""
    return JointCurve("plant", np.array([0]), np.array([1.]), np.array([1.]),
                      np.ones((1, 1)), np.ones((1, 1)), np.ones((1, 1)), np.array([True]), {})


def multihour_case(prices: list[float], initial: float) -> dict:
    """Create an unconstrained one-reservoir case with finite initial water."""
    case = simple_case()
    n = len(prices)
    case["price"] = np.asarray(prices, dtype=float)
    case["initial"][:] = initial
    case["capacity"][:] = max(1., initial)
    case["terminal_value"][:] = 0
    for key in ["minimum", "natural", "bypass_minimum"]:
        case[key] = np.repeat(case[key], n, axis=1)
    for key in ["vest_natural", "vest_minimum"]:
        case[key] = np.repeat(case[key], n)
    return case


class JointSwitchingTests(unittest.TestCase):
    def test_free_boundary_does_not_charge_initial_start_or_final_stop(self):
        result = SwitchingJointLP(network(), horizon=2, segments=1).solve(
            multihour_case([10., 10.], 2.), [one_mode_curve()], np.array([100.]), np.array([100.]))
        np.testing.assert_allclose(result["joint_on_fraction"], 1.)
        self.assertAlmostEqual(result["switching_cost"], 0.)
        self.assertAlmostEqual(result["objective"], 20.)

    def test_switch_cost_smooths_state_only_by_physical_operation(self):
        case = multihour_case([10., -1., 10.], 3.)
        solver = SwitchingJointLP(network(), horizon=3, segments=1)
        free = solver.solve(case, [one_mode_curve()], np.zeros(1), np.zeros(1))
        costly = solver.solve(case, [one_mode_curve()], np.array([20.]), np.array([20.]))
        np.testing.assert_allclose(free["joint_on_fraction"], [[1., 0., 1.]])
        np.testing.assert_allclose(costly["joint_on_fraction"], 1.)
        np.testing.assert_allclose(costly["power"], costly["joint_on_fraction"])
        np.testing.assert_allclose(costly["flow"], costly["joint_on_fraction"])
        self.assertAlmostEqual(costly["objective"], 19.)
        self.assertLessEqual(costly["objective"], free["objective"])
        self.assertAlmostEqual(costly["total_balance_error"], 0.)

    def test_fractional_startup_is_paid_exactly_and_cannot_hide_zero_output(self):
        case = multihour_case([-10., 10.], .5)
        result = SwitchingJointLP(network(), horizon=2, segments=1).solve(
            case, [one_mode_curve()], np.array([2.]), np.array([2.]))
        np.testing.assert_allclose(result["joint_on_fraction"], [[0., .5]])
        np.testing.assert_allclose(result["startup_fraction"], [[.5]])
        self.assertAlmostEqual(result["switching_cost"], 1.)
        self.assertAlmostEqual(result["objective"], 4.)
        np.testing.assert_allclose(result["power"], result["joint_on_fraction"])

    def test_negative_price_forced_release_uses_lower_physical_vertex(self):
        curve = JointCurve("plant", np.array([0]), np.array([.5, .5, 1.]), np.array([3., 1., 4.]),
                           np.array([[.5], [.5], [1.]]), np.array([[3.], [1.], [4.]]),
                           np.ones((3, 1)), np.array([True, False, True]), {})
        case = simple_case()
        case["initial"][:] = .5
        case["capacity"][:] = 0
        case["price"][:] = -1
        result = SwitchingJointLP(network(), horizon=1, segments=1).solve(case, [curve], np.array([5.]), np.array([5.]))
        self.assertAlmostEqual(result["power"][0, 0], 1.)
        self.assertAlmostEqual(result["flow"][0, 0], .5)
        self.assertEqual(result["old_joint_q_max"], 0.)

    def test_real_subset_hulls_keep_every_feasible_active_mask(self):
        data = HydroCaseData(Path(__file__).resolve().parents[1])
        for plant in ["Tokke", "Vinje"]:
            curve = build_subset_joint_curve(data, 2192, plant)
            expected = {int(mask) for mask, n in curve.specification["feasible_by_subset"].items() if n}
            self.assertEqual(set(curve.specification["vertex_subsets"]), expected)
            self.assertTrue((curve.total_flow > 0).all())
            for k, unit in enumerate(curve.specification["units"]):
                p_min = data.model["generator"][unit]["p_min"]
                self.assertTrue((curve.unit_power[:, k] >= p_min * curve.unit_on[:, k] - 1e-8).all())

    def test_zero_cost_matches_global_hull_objective_and_conservation(self):
        data = HydroCaseData(Path(__file__).resolve().parents[1])
        case = data.case(2192)
        subset = [build_subset_joint_curve(data, 2192, plant) for plant in ["Tokke", "Vinje"]]
        global_hull = [build_joint_curve(data, 2192, plant) for plant in ["Tokke", "Vinje"]]
        before = JointPlantLP(data.network).solve(case, global_hull)
        after = SwitchingJointLP(data.network).solve(case, subset, np.zeros(14), np.zeros(14))
        self.assertAlmostEqual(before["objective"], after["objective"], delta=1e-3)
        self.assertLess(abs(after["total_balance_error"]), 1e-8)
        self.assertEqual(after["old_joint_q_max"], 0.)
        self.assertLessEqual(after["maximum_convex_mass"], 1 + 1e-8)


if __name__ == "__main__":
    unittest.main()
