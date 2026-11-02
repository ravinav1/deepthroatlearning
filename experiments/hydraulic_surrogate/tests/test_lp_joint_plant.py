"""Physical vertex, convex allocation, negative-price and routing checks."""

from pathlib import Path
import unittest

import numpy as np

from sintef_project.lp_features import HydroCaseData
from sintef_project.lp_joint_plant import (
    COMMON_LOAD_GRID, JointCurve, JointPlantLP, build_joint_curve,
    envelope_indices, simultaneous_points, result_joint_features,
)
from test_lp_features import network, simple_case


class JointPlantTests(unittest.TestCase):
    def test_simultaneous_losses_use_total_and_shared_penstock_flow(self):
        generator = {"penstock": 1, "p_min": .01, "p_max": 1000,
                     "turb_eff_curves": [{"ref": 100, "x": [.1, 100], "y": [100, 100]}],
                     "gen_eff_curve": {"x": [0, 1000], "y": [100, 100]}}
        plant = {"main_loss": [.0012], "penstock_loss": [.001]}
        power, head, valid = simultaneous_points({"a": generator, "b": generator}, plant, 100., ["a", "b"], np.array([[10., 20.]]))
        expected_head = 100 - (.0012 + .001) * 30 ** 2
        np.testing.assert_allclose(head, expected_head)
        np.testing.assert_allclose(power, .00981 * np.array([[10., 20.]]) * expected_head)
        self.assertTrue(valid[0])

    def test_upper_and_lower_envelopes_bound_all_points(self):
        q = np.array([0., 1., 1., 2., 2., 3.])
        p = np.array([0., 3., .8, 5., 2., 6.])
        upper = envelope_indices(q, p, True)
        lower = envelope_indices(q, p, False)
        self.assertTrue((np.interp(q, q[upper], p[upper]) >= p - 1e-10).all())
        self.assertTrue((np.interp(q, q[lower], p[lower]) <= p + 1e-10).all())
        self.assertIn(2, lower)

    def test_negative_price_can_select_lower_physical_vertex(self):
        curve = JointCurve("plant", np.array([0]), np.array([.5, .5, 1.]), np.array([3., 1., 4.]),
                           np.array([[.5], [.5], [1.]]), np.array([[3.], [1.], [4.]]),
                           np.ones((3, 1)), np.array([True, False, True]), {})
        case = simple_case()
        case["initial"][:] = .5
        case["capacity"][:] = 0
        case["price"][:] = -1
        result = JointPlantLP(network(), horizon=1, segments=1).solve(case, [curve])
        self.assertAlmostEqual(result["flow"][0, 0], .5)
        self.assertAlmostEqual(result["power"][0, 0], 1.)
        self.assertAlmostEqual(result["mode_weights_plant"][1, 0], 1.)
        self.assertAlmostEqual(result["old_joint_q_max"], 0.)
        self.assertAlmostEqual(result["total_balance_error"], 0.)

    def test_single_mode_partial_time_sharing_has_exact_allocations(self):
        curve = JointCurve("plant", np.array([0]), np.array([.5]), np.array([2.]),
                           np.array([[.5]]), np.array([[2.]]), np.ones((1, 1)), np.array([True]), {})
        case = simple_case()
        case["initial"][:] = .25
        case["capacity"][:] = 0
        result = JointPlantLP(network(), horizon=1, segments=1).solve(case, [curve])
        self.assertAlmostEqual(result["maximum_convex_mass"], .5)
        self.assertAlmostEqual(result["flow"][0, 0], .25)
        self.assertAlmostEqual(result["power"][0, 0], 1.)
        self.assertAlmostEqual(result["joint_on_fraction"][0, 0], .5)
        self.assertAlmostEqual(result["total_balance_error"], 0.)

    def test_real_vertices_respect_unit_limits_and_allocation_sums(self):
        data = HydroCaseData(Path(__file__).resolve().parents[1])
        self.assertEqual(len(COMMON_LOAD_GRID), 128)
        self.assertEqual(COMMON_LOAD_GRID[0], 1 / 128)
        self.assertEqual(COMMON_LOAD_GRID[-1], 1.)
        for plant in ["Tokke", "Vinje"]:
            curve = build_joint_curve(data, 2192, plant)
            np.testing.assert_allclose(curve.unit_flow.sum(axis=1), curve.total_flow)
            np.testing.assert_allclose(curve.unit_power.sum(axis=1), curve.total_power)
            for j, unit in enumerate(curve.specification["units"]):
                active = curve.unit_on[:, j] > 0
                params = data.model["generator"][unit]
                self.assertTrue((curve.unit_power[active, j] >= params["p_min"] - 1e-8).all())
                self.assertTrue((curve.unit_power[active, j] <= params["p_max"] + 1e-8).all())
                self.assertTrue((curve.unit_flow[~active, j] == 0).all())
            self.assertEqual(curve.specification["candidate_count"], (2 ** len(curve.unit_indices) - 1) * 128)

    def test_full_network_feature_schema_and_joint_counts(self):
        data = HydroCaseData(Path(__file__).resolve().parents[1])
        case = data.case(2192)
        curves = [build_joint_curve(data, 2192, plant) for plant in ["Tokke", "Vinje"]]
        result = JointPlantLP(data.network).solve(case, curves)
        features = result_joint_features(case, result, data.network)
        self.assertEqual(features.shape, (168, 87))
        self.assertTrue(all(name.startswith("lpjoint_") for name in features))
        self.assertEqual(result["old_joint_q_max"], 0)
        self.assertLessEqual(result["maximum_convex_mass"], 1 + 1e-8)
        for plant in ["Tokke", "Vinje"]:
            columns = [name for name in features if name.startswith(f"lpjoint_on_fraction_{plant}_")]
            np.testing.assert_allclose(features[columns].sum(axis=1), features[f"lpjoint_active_unit_count_{plant}"], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
