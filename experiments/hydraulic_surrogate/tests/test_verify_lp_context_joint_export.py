"""Guard four-source arithmetic and strict provenance in final export checks."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.verify_lp_context_joint_export import (
    EMPTY_KAGGLE, WEIGHTS, bitwise_equal, reconstruct_float64,
    verify_cache_reference, verify_observation_only_update, verify_context_manifest_reference,
)
from sintef_project.postprocess import DEFAULT_UNITS, project_vinje


class CombinedExportVerificationTests(unittest.TestCase):
    def test_four_source_promotion_and_projection_once(self) -> None:
        rng = np.random.default_rng(4173)
        arrays = [rng.random((168, 14), dtype=np.float32) for _ in range(4)]
        expected = np.zeros((168, 14), dtype=np.float64)
        premature_projection = np.zeros_like(expected)
        premature_weighting = np.zeros_like(expected)
        for source, weight in zip(arrays, WEIGHTS):
            expected += weight * source.astype(np.float64)
            premature_weighting += weight * source
            premature_projection += weight * project_vinje(source.astype(np.float64), DEFAULT_UNITS)
        actual = reconstruct_float64(arrays, WEIGHTS, DEFAULT_UNITS)
        self.assertTrue(bitwise_equal(actual, project_vinje(expected, DEFAULT_UNITS)))
        self.assertFalse(bitwise_equal(actual, project_vinje(premature_weighting, DEFAULT_UNITS)))
        self.assertFalse(bitwise_equal(actual, premature_projection))

    def test_cached_shared_reference_is_checked_each_time(self) -> None:
        cache = {'source': {'formula.py': 'source_hash'}, 'data': {'input.csv': 'data_hash'}}
        checked = {'manifest_sha256': 'manifest_hash', 'features_sha256': 'feature_hash'}
        reference = {**checked, 'source_verified': cache['source'], 'data_verified': cache['data']}
        verify_cache_reference(reference, cache, checked)
        for key in reference:
            modified = copy.deepcopy(reference)
            modified[key] = 'wrong' if isinstance(modified[key], str) else {'changed': 'wrong'}
            with self.subTest(key=key), self.assertRaises(ValueError):
                verify_cache_reference(modified, cache, checked)

    def test_only_known_observation_field_can_change(self) -> None:
        original = {'status': 'complete', 'kaggle': EMPTY_KAGGLE.copy(),
                    'weights': [.25, .75], 'artifacts': {'submission.csv': 'unchanged'}}
        original_sha = hashlib.sha256(json.dumps(original, indent=2).encode()).hexdigest()
        observed = copy.deepcopy(original)
        observed['kaggle'] = {'score': .99604, 'rank': 28}
        verify_observation_only_update(observed, original_sha)
        observed['artifacts']['submission.csv'] = 'changed'
        with self.assertRaises(ValueError):
            verify_observation_only_update(observed, original_sha)

    def test_exact_observed_reference_and_only_pinned_historical_exception(self) -> None:
        original = {'status': 'complete', 'kaggle': EMPTY_KAGGLE.copy(), 'weights': [.25, .75]}
        original_sha = hashlib.sha256(json.dumps(original, indent=2).encode()).hexdigest()
        observed = {**original, 'kaggle': {'score': .99604}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.json'
            path.write_text(json.dumps(observed, indent=2))
            current_sha = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertIsNone(verify_context_manifest_reference(path, current_sha))
            with patch('scripts.verify_lp_context_joint_export.ORIGINAL_CONTEXT_MANIFEST_SHA256', original_sha):
                update = verify_context_manifest_reference(path, original_sha)
                self.assertEqual(update['historical_sha256'], original_sha)
                with self.assertRaises(ValueError):
                    verify_context_manifest_reference(path, 'f' * 64)
                path.write_text(json.dumps({**observed, 'weights': [.5, .5]}, indent=2))
                with self.assertRaises(ValueError):
                    verify_context_manifest_reference(path, original_sha)

    def test_malformed_inputs_fail_closed(self) -> None:
        source = np.full((168, 14), .5, dtype=np.float32)
        for arrays, weights in [([], []), ([source], [-1.]), ([source] * 4, WEIGHTS[:3]),
                                ([source, source[:1]], [.5, .5]),
                                ([np.full_like(source, np.nan)], [1.])]:
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                reconstruct_float64(arrays, weights, DEFAULT_UNITS)
        self.assertFalse(bitwise_equal(np.array([0.]), np.array([-0.])))
        self.assertFalse(bitwise_equal(np.array([1.], dtype=np.float32), np.array([1.], dtype=np.float64)))


if __name__ == '__main__':
    unittest.main()
