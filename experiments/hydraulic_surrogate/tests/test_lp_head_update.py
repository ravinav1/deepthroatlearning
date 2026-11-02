"""Independent hourly-segment, conservation and reservoir-head checks."""

from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from sintef_project.lp_features import HydraulicLP
from sintef_project.lp_head_update import HourlyHeadLP, single_reservoir_heads
from test_lp_features import network, simple_case


class HourlyHeadTests(unittest.TestCase):
    def test_constant_3d_coefficients_match_frozen_solver(self):
        case = simple_case(horizon=2)
        expected = HydraulicLP(network(), horizon=2, segments=1).solve(case)
        case["widths"] = np.repeat(case["widths"][:, :, None], 2, axis=2)
        case["energy"] = np.repeat(case["energy"][:, :, None], 2, axis=2)
        actual = HourlyHeadLP(network(), horizon=2, segments=1).solve(case)
        np.testing.assert_allclose(actual["flow"], expected["flow"])
        np.testing.assert_allclose(actual["storage"], expected["storage"])
        self.assertAlmostEqual(actual["objective"], expected["objective"])

    def test_hourly_energy_changes_dispatch_without_creating_water(self):
        case = simple_case(horizon=2)
        case["price"][:] = 50
        case["initial"][:] = 1
        case["energy"] = np.array([[[2., 4.]]])
        case["widths"] = np.array([[[1., 1.]]])
        result = HourlyHeadLP(network(), horizon=2, segments=1).solve(case)
        np.testing.assert_allclose(result["flow"], [[0., 1.]])
        np.testing.assert_allclose(result["power"], [[0., 4.]])
        self.assertAlmostEqual(result["total_balance_error"], 0)

    def test_negative_prices_use_each_hours_width_weighted_conversion(self):
        case = simple_case(horizon=2)
        case["initial"][:] = 0
        case["natural"][:] = .25
        case["capacity"][:] = 0
        case["price"][:] = -1
        case["energy"] = np.array([[[4., 8.], [2., 4.]]])
        case["widths"] = np.array([[[.5, .25], [.5, .75]]])
        result = HourlyHeadLP(network(), horizon=2, segments=2).solve(case)
        np.testing.assert_allclose(result["flow"], [[.25, .25]])
        np.testing.assert_allclose(result["power"], [[.75, 1.25]])
        np.testing.assert_allclose(result["effective_segment_energy"], [[[3., 5.], [3., 5.]]])
        self.assertAlmostEqual(result["total_balance_error"], 0)

    def test_nonconcave_or_malformed_hourly_segments_are_rejected(self):
        case = simple_case(horizon=2)
        case["energy"] = np.array([[[2., 2.], [4., 4.]]])
        case["widths"] = np.full((1, 2, 2), .5)
        solver = HourlyHeadLP(network(), horizon=2, segments=2)
        with self.assertRaisesRegex(ValueError, "nonincreasing"):
            solver.solve(case)
        case["energy"] = np.ones((1, 2, 3))
        with self.assertRaisesRegex(ValueError, "Expected"):
            solver.solve(case)

    def test_head_uses_initial_and_predicted_boundary_levels_only(self):
        start = pd.Timestamp("2021-01-01", tz="UTC")
        data = SimpleNamespace(cases=pd.DataFrame({"starttime": [start]}),
                               volume=pd.DataFrame({"Bandak": [2., 999.], "Vinjevatn": [4., 999.]}, index=[start, start + pd.Timedelta(days=1)]),
                               model={"reservoir": {"Bandak": {"vol_head": {"x": [0, 10], "y": [60, 70]}},
                                                    "Vinjevatn": {"vol_head": {"x": [0, 10], "y": [462, 472]}}},
                                      "plant": {"Hogga": {"outlet_line": 60}, "Tokke": {"outlet_line": 72}}})
        storage = np.zeros((9, 2))
        storage[0] = [6, 10]
        storage[1] = [6, 8]
        heads = single_reservoir_heads(data, 0, {"storage": storage})
        np.testing.assert_allclose(heads["Hogga"], [4, 8])
        np.testing.assert_allclose(heads["Tokke"], [395, 397])
        self.assertEqual(heads["initial_Hogga"], 2)


if __name__ == "__main__":
    unittest.main()
