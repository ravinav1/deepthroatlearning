"""Checks for isolated shared-loss correction without double counting."""

from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from sintef_project.lp_features import HydroCaseData, UNIT_ORDER
from sintef_project.lp_head_update import shared_loss_correction
from sintef_project.lp_shared_loss_update import update_case_shared_losses


class SharedLossTests(unittest.TestCase):
    def test_extra_plus_self_equals_total_shared_loss_at_previous_dispatch(self):
        data = SimpleNamespace(model={"plant": {"Tokke": {"main_loss": [.0012], "penstock_loss": [.001, .002]}},
                                      "generator": {f"Tokke_G{i}": {"penstock": 1 if i <= 2 else 2} for i in range(1, 5)}})
        flow = np.zeros((14, 2))
        flow[1:5] = np.array([[10, 20], [20, 0], [30, 40], [40, 10]]) * .0036
        correction = shared_loss_correction(data, {"flow": flow}, "Tokke")
        total = flow[1:5].sum(axis=0) / .0036
        for i in range(1, 5):
            unit = f"Tokke_G{i}"
            own = flow[i] / .0036
            group = flow[1:3].sum(axis=0) / .0036 if i <= 2 else flow[3:5].sum(axis=0) / .0036
            coefficient = .001 if i <= 2 else .002
            self_loss = (.0012 + coefficient) * own ** 2
            exact = .0012 * total ** 2 + coefficient * group ** 2
            np.testing.assert_allclose(correction[unit] + self_loss, exact, atol=1e-12)

    def test_zero_previous_flow_recovers_initial_curves_and_preserves_terminal_value(self):
        data = HydroCaseData(Path(__file__).resolve().parents[1])
        case = data.case(2192)
        before = {key: np.array(value, copy=True) for key, value in case.items()}
        changed, details = update_case_shared_losses(data, 2192, case, {"flow": np.zeros((14, 168))})
        for key in ["energy", "widths"]:
            np.testing.assert_allclose(changed[key], np.repeat(case[key][:, :, None], 168, axis=2))
        np.testing.assert_array_equal(changed["terminal_value"], case["terminal_value"])
        self.assertEqual(set(details), {unit for unit in UNIT_ORDER if unit.startswith(("Tokke_", "Vinje_"))})
        for key in before:
            np.testing.assert_array_equal(before[key], case[key])


if __name__ == "__main__":
    unittest.main()
