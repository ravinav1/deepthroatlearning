"""Guard float64-before-weighting and byte-exact comparison in export checks."""
import unittest
import hashlib
import json

import numpy as np

from scripts.verify_lp_joint_export import EMPTY_KAGGLE, WEIGHTS, bitwise_equal, reconstruct_float64, verify_observation_only_update
from sintef_project.postprocess import DEFAULT_UNITS, project_vinje


class JointExportVerificationTests(unittest.TestCase):
    def test_promote_before_weighting_and_preserve_source_order(self) -> None:
        rng = np.random.default_rng(4173)
        arrays = [rng.random((168, 14), dtype=np.float32) for _ in range(3)]
        expected = np.zeros((168, 14), dtype=np.float64)
        wrong = np.zeros_like(expected)
        for source, weight in zip(arrays, WEIGHTS):
            expected += weight * source.astype(np.float64)
            wrong += weight * source
        expected = project_vinje(expected, DEFAULT_UNITS)
        actual = reconstruct_float64(arrays, WEIGHTS, DEFAULT_UNITS)
        self.assertTrue(bitwise_equal(actual, expected))
        self.assertFalse(bitwise_equal(actual, project_vinje(wrong, DEFAULT_UNITS)))

    def test_byte_comparison_detects_signed_zero_and_dtype(self) -> None:
        self.assertFalse(bitwise_equal(np.array([0.]), np.array([-0.])))
        self.assertFalse(bitwise_equal(np.array([1.], dtype=np.float32), np.array([1.], dtype=np.float64)))

    def test_invalid_weights_fail_closed(self) -> None:
        source = np.full((168, 14), .5, dtype=np.float32)
        with self.assertRaises(ValueError):
            reconstruct_float64([source], [-1.], DEFAULT_UNITS)

    def test_only_observation_metadata_update_is_permitted(self) -> None:
        historical = {'status': 'complete', 'kaggle': EMPTY_KAGGLE.copy(),
                      'weights': [.25, .75], 'artifacts': {'submission.csv': 'immutable_hash'}}
        expected = hashlib.sha256(json.dumps(historical, indent=2).encode()).hexdigest()
        current = historical.copy()
        current['kaggle'] = {'submission_id': 'recorded', 'score': .99604, 'rank': 28, 'observation_time': 'observed'}
        self.assertEqual(verify_observation_only_update(current, expected)['historical_sha256'], expected)
        current['weights'] = [.5, .5]
        with self.assertRaises(ValueError):
            verify_observation_only_update(current, expected)


if __name__ == '__main__':
    unittest.main()
