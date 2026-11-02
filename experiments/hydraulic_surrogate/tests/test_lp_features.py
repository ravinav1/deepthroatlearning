"""Small independent water-balance, economics, sign and corner-case LP checks."""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from sintef_project.lp_features import HydroCaseData, HydraulicLP, Network, unit_curve


def network(n_pools=1, bypass=False):
    """Create a one-unit network with optional second storage and bypass."""
    return Network([f"r{i}" for i in range(n_pools)], ["g"], np.array([0]), np.array([1 if n_pools == 2 else -1]),
                   np.array([0] if bypass else [], dtype=int), np.array([-1] if bypass else [], dtype=int),
                   np.full(n_pools, -1), np.array([False]), np.array([False] if bypass else [], dtype=bool), np.zeros(n_pools, dtype=bool))


def simple_case(n_pools=1, horizon=1, bypass=False):
    """Return simple independent numeric fixtures in Mm3 and MWh."""
    return {"price": np.full(horizon, 49.0), "initial": np.array([2.0] + [0.0] * (n_pools - 1)),
            "capacity": np.full(n_pools, 10.0), "minimum": np.zeros((n_pools, horizon)), "natural": np.zeros((n_pools, horizon)),
            "terminal_value": np.full(n_pools, 100.0), "widths": np.array([[1.0]]), "energy": np.array([[2.0]]),
            "qmin": np.array([0.5]), "qmax": np.array([1.0]), "pmax": np.array([2.0]),
            "bypass_minimum": np.zeros((1 if bypass else 0, horizon)), "vest_natural": np.zeros(horizon), "vest_minimum": np.zeros(horizon),
            "slack_penalty": 10000.0, "bypass_cost": 10.0, "spill_cost": 1000.0}


class HydraulicLPTests(unittest.TestCase):
    def test_local_water_value_threshold_and_dual_sign(self):
        solver = HydraulicLP(network(), horizon=1, segments=1)
        case = simple_case()
        below = solver.solve(case)
        self.assertAlmostEqual(below["flow"][0, 0], 0)
        self.assertAlmostEqual(below["water_shadow"][0, 0], 100)
        case["initial"][0] += 0.001
        perturbed = solver.solve(case)
        self.assertAlmostEqual((perturbed["objective"] - below["objective"]) / 0.001, below["water_shadow"][0, 0], places=5)
        case["price"][:] = 51
        above = solver.solve(case)
        self.assertAlmostEqual(above["flow"][0, 0], 1)

    def test_global_terminal_values_recover_local_threshold(self):
        solver = HydraulicLP(network(2), horizon=1, segments=1)
        case = simple_case(2)
        case["terminal_value"] = np.array([140.0, 40.0])
        below = solver.solve(case)
        self.assertAlmostEqual(below["flow"][0, 0], 0)
        case["price"][:] = 51
        above = solver.solve(case)
        self.assertAlmostEqual(above["flow"][0, 0], 1)
        self.assertAlmostEqual(above["storage"][1, 0], 1)
        self.assertAlmostEqual(above["total_balance_error"], 0, places=8)

    def test_soft_bounds_never_create_water(self):
        solver = HydraulicLP(network(), horizon=1, segments=1)
        case = simple_case()
        case["initial"][:] = 0
        case["minimum"][:] = 3
        result = solver.solve(case)
        self.assertAlmostEqual(result["minimum_slack"][0, 0], 3)
        self.assertAlmostEqual(result["storage"][0, 0], 0)
        self.assertAlmostEqual(result["flow"].sum(), 0)
        self.assertAlmostEqual(result["total_balance_error"], 0)


class InputProvenanceTests(unittest.TestCase):
    def fixture(self):
        """Create legal synthetic time series without accessing commitment CSVs."""
        root = Path(__file__).resolve().parents[1]
        data = HydroCaseData.__new__(HydroCaseData)
        data.topology = yaml.safe_load((root / "sintef_project/data/extended/Tokke_Vinje_topology.yaml").read_text())
        data.model = data.topology["model"]
        data.plant_downstream_reservoir = {edge["from"]: edge["to"] for edge in data.topology["connections"] if edge["from_type"] == "plant" and edge["to_type"] == "reservoir"}
        days = pd.date_range("2020-01-01", periods=9, freq="D", tz="UTC")
        hours = pd.date_range(days[0], periods=192, freq="h")
        reservoirs = list(data.model["reservoir"])
        data.cases = pd.DataFrame({"Run No": [1, 2], "starttime": days[:2]})
        data.volume = pd.DataFrame({name: np.full(9, data.model["reservoir"][name]["max_vol"] * .5) for name in reservoirs}, index=days)
        data.water = pd.DataFrame(50.0, index=days, columns=reservoirs)
        data.price = pd.DataFrame({"price": np.tile(np.arange(24) + 20.0, 8)}, index=hours)
        data.inflow = pd.DataFrame(1.0, index=hours, columns=[*reservoirs, "r_Vest_Vassdraget"])
        data.minimum = pd.DataFrame(0.0, index=hours, columns=["Totak", "Byrtevatn", "Staavatn"])
        data.flow_minimum = pd.DataFrame(0.0, index=hours, columns=["b_Bandak", "b_Vinjevatn", "b_Byrtevatn", "b_Hyljelihyl", "b_Kjelavatn", "r_Vest_Vassdraget"])
        return data

    def test_label_metadata_and_post_start_volumes_cannot_change_case(self):
        data = self.fixture()
        baseline = data.case(0)
        data.cases["result_committed_Hogga_G1_t0"] = [0, 1]
        data.cases["total_calculation_time"] = [1e9, -1e9]
        data.volume.iloc[1:] *= 0.01
        changed = data.case(0)
        for key in baseline:
            np.testing.assert_array_equal(baseline[key], changed[key], err_msg=key)
        # A later case legitimately has a different initial volume after that
        # perturbation; this ensures the mutation was not an ineffective no-op.
        self.assertFalse(np.array_equal(baseline["initial"], data.case(1)["initial"]))

    def test_terminal_water_value_uses_exactly_start_plus_seven_days(self):
        data = self.fixture()
        baseline = data.case(0)
        data.water.iloc[:7] *= 10
        unchanged = data.case(0)
        np.testing.assert_array_equal(baseline["terminal_value"], unchanged["terminal_value"])
        data.water.iloc[7] *= 2
        changed = data.case(0)
        np.testing.assert_allclose(changed["terminal_value"], 2 * baseline["terminal_value"])

    def test_generation_curve_respects_turbine_minimum_and_power_cap(self):
        generator = {"penstock": 1, "p_min": .1, "p_max": .9,
                     "turb_eff_curves": [{"ref": 100, "x": [.4, 1.0], "y": [100, 100]}],
                     "gen_eff_curve": {"x": [0, 2], "y": [100, 100]}}
        plant = {"penstock_loss": [0], "main_loss": ["0e-6"]}
        curve = unit_curve(generator, plant, 100)
        self.assertAlmostEqual(curve["qmin"], .4 * .0036)
        self.assertLessEqual(curve["pmax"], .9 + 1e-8)
        self.assertTrue((curve["widths"] > 0).all())
        self.assertTrue((np.diff(curve["energy"]) <= 1e-6).all())

    def test_mandatory_bypass_uses_its_own_destination(self):
        solver = HydraulicLP(network(2, bypass=True), horizon=1, segments=1)
        case = simple_case(2, bypass=True)
        case["terminal_value"] = np.array([140.0, 40.0])
        case["bypass_minimum"][:] = 0.5
        result = solver.solve(case)
        self.assertAlmostEqual(result["bypass"][0, 0], 0.5)
        self.assertAlmostEqual(result["storage"][1, 0], 0)
        self.assertAlmostEqual(result["total_balance_error"], 0)

    def test_negative_price_uses_linear_conversion_not_segment_skipping(self):
        solver = HydraulicLP(network(), horizon=1, segments=2)
        case = simple_case()
        case["price"][:] = -1
        case["capacity"][:] = 0
        case["widths"] = np.array([[0.5, 0.5]])
        case["energy"] = np.array([[4.0, 2.0]])
        case["initial"][:] = 0.25
        result = solver.solve(case)
        self.assertAlmostEqual(result["flow"][0, 0], 0.25)
        self.assertAlmostEqual(result["power"][0, 0], 0.75)

    def test_terminal_hour_inflow_and_horizon_conservation(self):
        solver = HydraulicLP(network(), horizon=3, segments=1)
        case = simple_case(horizon=3)
        case["initial"][:] = 0
        case["natural"][0, -1] = 0.5
        result = solver.solve(case)
        np.testing.assert_allclose(result["storage"], [[0, 0, 0.5]])
        self.assertAlmostEqual(result["objective"], 50)
        self.assertAlmostEqual(result["total_balance_error"], 0)


if __name__ == "__main__":
    unittest.main()
