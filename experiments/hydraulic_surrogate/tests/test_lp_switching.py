"""Analytical checks of startup smoothing using LP dual-price margins."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import yaml

from sintef_project.lp_features import UNIT_ORDER
from sintef_project.lp_switching import add_lp_switching_features


class LPSwitchingTests(unittest.TestCase):
    def inputs(self, root, start=.75, stop=.75):
        folder = root / "data/extended"
        folder.mkdir(parents=True)
        generator = {unit: {"p_min": 1.0, "p_max": 2.0, "startcost_const": start, "stopcost_const": stop}
                     for unit in UNIT_ORDER}
        (folder / "Tokke_Vinje_topology.yaml").write_text(yaml.safe_dump({"model": {"generator": generator}}))
        margin = np.full((2, 168), -1.0)
        margin[:, 84] = .1
        x = pd.DataFrame({"horizon_hour": np.tile(np.arange(168), 2),
                          "price": np.repeat([9., 19.], 168),
                          **{f"lp2_economic_margin_{unit}": margin.ravel() for unit in UNIT_ORDER}})
        return x, margin

    def test_fixed_costs_change_isolated_profit_decision_at_correct_case_scale(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            x, _ = self.inputs(root)
            result, spec = add_lp_switching_features(x, root)
            # Same normalized margin, scales10/20: isolated nominal1MW profit
            # is1/2 EUR. Two switches cost1.5, giving advantages-.5/.5 EUR.
            self.assertAlmostEqual(result.iloc[84]["lpdp_switch_margin_min_Haukeli_G1"], -.05, places=6)
            self.assertAlmostEqual(result.iloc[168 + 84]["lpdp_switch_margin_min_Haukeli_G1"], .025, places=6)
            self.assertAlmostEqual(result.iloc[84]["lpdp_switch_margin_max_Haukeli_G1"], .025, places=6)
            self.assertEqual(len(spec["columns"]), 28)
            self.assertEqual(len(result.columns) - len(x.columns), 28)

    def test_zero_costs_recover_each_hour_margin_without_cross_case_effect(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            x, margin = self.inputs(root, start=0, stop=0)
            result, _ = add_lp_switching_features(x, root)
            for unit in UNIT_ORDER:
                for point in ["min", "max"]:
                    np.testing.assert_allclose(result[f"lpdp_switch_margin_{point}_{unit}"], margin.ravel(), atol=1e-6)

    def test_labels_do_not_change_features_and_normalized_input_is_equivalent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            x, _ = self.inputs(root)
            original = x.copy()
            result, spec = add_lp_switching_features(x, root)
            pd.testing.assert_frame_equal(x, original)
            x["result_committed_Haukeli_G1_t0"] = np.arange(len(x)) % 2
            x["total_calculation_time"] = -1e9
            x["monetary_scale_inverse"] = np.repeat([.1, .05], 168)
            changed, _ = add_lp_switching_features(x.drop(columns="price"), root)
            np.testing.assert_allclose(result[spec["columns"]], changed[spec["columns"]], atol=1e-6)

    def test_multiple_lp_versions_require_explicit_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            x, _ = self.inputs(root)
            for unit in UNIT_ORDER:
                x[f"lp_economic_margin_{unit}"] = x[f"lp2_economic_margin_{unit}"]
            with self.assertRaisesRegex(ValueError, "multiple"):
                add_lp_switching_features(x, root)
            _, spec = add_lp_switching_features(x, root, lp_prefix="lp")
            self.assertEqual(spec["lp_prefix"], "lp")


if __name__ == "__main__":
    unittest.main()
