"""Independent conservation and directionality tests for Haukeli split storage."""

from pathlib import Path
import hashlib
import json
import tempfile
import unittest

import numpy as np
import pandas as pd

from sintef_project.lp_features import HydraulicLP, Network, UNIT_ORDER
from sintef_project.lp_features_v2 import (
    HAUKELI, DirectedHaukeliData, add_lp_features_v2, directed_haukeli_network, split_haukeli_case,
)


def fixture():
    """Nine-pool fixture with one active Haukeli turbine and no bypass routes."""
    network = Network([str(i) for i in range(9)], ["Haukeli_G1"], np.array([HAUKELI]),
                      np.array([-1]), np.array([], dtype=int), np.array([], dtype=int),
                      np.full(9, -1), np.array([False]), np.array([], dtype=bool),
                      np.zeros(9, dtype=bool))
    case = {"price": np.array([1.0]), "initial": np.zeros(9), "capacity": np.full(9, 10.0),
            "minimum": np.zeros((9, 1)), "natural": np.zeros((9, 1)),
            "terminal_value": np.full(9, 100.0), "widths": np.array([[1.0]]),
            "energy": np.array([[2.0]]), "qmin": np.array([0.1]), "qmax": np.array([1.0]),
            "pmax": np.array([2.0]), "bypass_minimum": np.zeros((0, 1)),
            "vest_natural": np.zeros(1), "vest_minimum": np.zeros(1)}
    case["initial"][HAUKELI] = .2
    case["natural"][HAUKELI] = .4
    return network, case


class DirectedStorageTests(unittest.TestCase):
    def test_downstream_inflow_cannot_be_stored_uphill(self):
        network, case = fixture()
        pooled = HydraulicLP(network, horizon=1, segments=1).solve(case)
        self.assertAlmostEqual(pooled["flow"].sum(), 0)
        split = split_haukeli_case(case, np.array([.2, 0.0]), np.array([[.4], [0.0]]),
                                   np.array([.3, 10.0]), np.zeros((2, 1)))
        directed = HydraulicLP(directed_haukeli_network(network), horizon=1, segments=1).solve(split)
        self.assertAlmostEqual(directed["storage"][-1, 0], 0)
        self.assertAlmostEqual(directed["storage"][HAUKELI, 0], .3)
        self.assertAlmostEqual(directed["flow"][0, 0], .3)
        self.assertAlmostEqual(directed["bypass"][-1, 0], 0)
        self.assertAlmostEqual(directed["total_balance_error"], 0, places=10)

    def test_upstream_water_can_flow_downstream_without_creation(self):
        network, case = fixture()
        case["price"][:] = 100
        split = split_haukeli_case(case, np.array([.2, 1.0]), np.zeros((2, 1)),
                                   np.array([.3, 10.0]), np.zeros((2, 1)))
        result = HydraulicLP(directed_haukeli_network(network), horizon=1, segments=1).solve(split)
        self.assertAlmostEqual(result["flow"][0, 0], 1)
        self.assertGreaterEqual(result["bypass"][-1, 0], .8 - 1e-10)
        self.assertAlmostEqual(result["storage"][:, -1].sum(), .2)
        self.assertAlmostEqual(result["total_balance_error"], 0, places=10)
        self.assertTrue((result["storage"] >= -1e-10).all())

    def test_split_does_not_mutate_base_case(self):
        _, case = fixture()
        before = {key: value.copy() for key, value in case.items()}
        split_haukeli_case(case, np.array([.2, 0]), np.array([[.4], [0]]),
                           np.array([.3, 10]), np.zeros((2, 1)))
        for key in before:
            np.testing.assert_array_equal(before[key], case[key])

    def test_legal_inputs_ignore_labels_and_post_start_volume(self):
        data = DirectedHaukeliData(Path(__file__).resolve().parents[1])
        case = data.case(0)
        data.cases["result_committed_Haukeli_G1_t0"] = 1
        data.cases["total_calculation_time"] = -1e9
        data.volume.iloc[1:] *= .1
        changed = data.case(0)
        for key in case:
            np.testing.assert_array_equal(case[key], changed[key], err_msg=key)
        self.assertAlmostEqual(case["capacity"][HAUKELI], .3)
        self.assertAlmostEqual(case["capacity"][-1], 32.1)


class CacheIntegrityTests(unittest.TestCase):
    def create_cache(self, root):
        """Create a tiny independent cache with two complete known-input cases."""
        kernel = root / "sintef_project/data/kernel"
        kernel.mkdir(parents=True)
        pd.DataFrame({"Run No": [1], "starttime": ["2020-01-01"]}).to_csv(kernel / "Unit_commitment_decisions.csv", index=False)
        pd.DataFrame({"Run No": [2], "starttime": ["2020-01-02"]}).to_csv(root / "sintef_project/prediction_mapping.csv", index=False)
        (root / "snapshot.py").write_text("original source\n")
        cache = root / "cache"
        cache.mkdir()
        names = [f"lp2_test_{i}" for i in range(81)]
        values = np.arange(2 * 168 * 81, dtype=np.float32).reshape(2, 168, 81)
        np.savez_compressed(cache / "features.npz", features=values, columns=np.array(names),
                            units=np.array(UNIT_ORDER), case_ids=np.array([1, 2]))
        digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        manifest = {"status": "complete", "math": {"version": "haukeli_directed_lp_v2"},
                    "source": {"snapshot.py": digest(root / "snapshot.py")}, "data": {},
                    "selected_case_dates": ["2020-01-01 00:00:00+00:00", "2020-01-02 00:00:00+00:00"],
                    "feature_names": names, "artifacts": {"features.npz": digest(cache / "features.npz")}}
        (cache / "manifest.json").write_text(json.dumps(manifest))
        return cache, pd.DataFrame({"horizon_hour": np.tile(np.arange(168), 2)})

    def test_cache_load_rejects_changed_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache, x = self.create_cache(root)
            loaded, spec = add_lp_features_v2(x, root, cache)
            self.assertEqual(loaded.shape, (336, 82))
            self.assertEqual(len(spec["columns"]), 81)
            (root / "snapshot.py").write_text("changed source\n")
            with self.assertRaisesRegex(ValueError, "source checksum mismatch"):
                add_lp_features_v2(x, root, cache)

    def test_cache_load_rejects_incomplete_or_reordered_hours(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache, x = self.create_cache(root)
            with self.assertRaisesRegex(ValueError, "complete case-hour order"):
                add_lp_features_v2(x.iloc[:-1], root, cache)
            with self.assertRaisesRegex(ValueError, "complete case-hour order"):
                add_lp_features_v2(x.iloc[::-1], root, cache)


if __name__ == "__main__":
    unittest.main()
