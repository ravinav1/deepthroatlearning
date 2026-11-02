"""Reject incomplete evidence and check independent feature reconstruction."""
import copy
import unittest
from types import SimpleNamespace

import numpy as np

from scripts.audit_integer_water_budget_cache import (
    check_complete_manifest, independent_features, same,
)


class IntegerCachePhysicalAuditTests(unittest.TestCase):
    def test_incomplete_cache_rejected_before_schema_or_feature_access(self):
        with self.assertRaisesRegex(ValueError, 'not complete'):
            check_complete_manifest({'status': 'building'}, {})

    def test_bitwise_check_rejects_signed_zero_and_dtype(self):
        for changed in [np.array([-0.], dtype=np.float64), np.array([0.], dtype=np.float32)]:
            with self.assertRaisesRegex(ValueError, 'bytes differ'):
                same({'x': np.array([0.], dtype=np.float64)}, {'x': changed})

    def test_independent_feature_units_order_and_sea_boundary(self):
        network = SimpleNamespace(unit_up=np.zeros(14, dtype=int),
                                  unit_down=np.array([1] * 13 + [-1]))
        case = {'qmin': np.full(14, 2.), 'pmax': np.full(14, 10.),
                'energy': np.full((14, 4), 4.), 'price': np.full(168, -2.)}
        result = {'state': np.full((14, 168), .5), 'flow': np.full((14, 168), 3.),
                  'power': np.full((14, 168), 8.), 'raw_start': np.full((14, 167), .25),
                  'raw_stop': np.full((14, 167), .125),
                  'water_shadow': np.stack([np.full(168, 7.), np.full(168, 3.)])}
        control = {'flow': np.ones((14, 168)), 'power': np.full((14, 168), 3.),
                   'water_shadow': np.stack([np.full(168, 13.), np.full(168, 5.)])}
        values = independent_features(case, result, control, network)
        self.assertEqual(values.shape, (168, 84))
        self.assertEqual(values.dtype, np.float32)
        np.testing.assert_array_equal(values[0, :6], np.array([.5, 1., .5, 0, 0, 1/3], dtype=np.float32))
        np.testing.assert_array_equal(values[1, :6], np.array([.5, 1., .5, .25, .125, 1/3], dtype=np.float32))
        self.assertEqual(values[1, -1], np.float32(.5))

    def test_nonfinite_reconstruction_rejected(self):
        network = SimpleNamespace(unit_up=np.zeros(14, dtype=int), unit_down=np.full(14, -1))
        case = {'qmin': np.ones(14), 'pmax': np.ones(14), 'energy': np.ones((14, 4)), 'price': np.zeros(168)}
        result = {'state': np.ones((14, 168)), 'flow': np.ones((14, 168)),
                  'power': np.ones((14, 168)), 'raw_start': np.zeros((14, 167)),
                  'raw_stop': np.zeros((14, 167)), 'water_shadow': np.zeros((1, 168))}
        control = copy.deepcopy(result)
        control['water_shadow'][0, 3] = np.nan
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            independent_features(case, result, control, network)


if __name__ == '__main__':
    unittest.main()
