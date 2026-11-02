"""Tests for local year gating and frozen specialist splicing."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.train_nonlinear_pq_specialists import WEIGHTS, compose, validate_lock


class NonlinearPQProtocolTests(unittest.TestCase):
    def test_lock_requires_fixed_bytes_protocol_and_both_improvements(self):
        lock = {'version': 'nonlinear_pq_selection_v1', 'year': 2021, 'protocol_sha256': 'a' * 64,
                'candidate_weights': WEIGHTS, 'selected_weight': .1, 'eligible_for_2022': True,
                'selected_micro_auc': .995, 'anchor_micro_auc': .994, 'control_micro_auc': .9945,
                'scores_2021': {'0.0': .994, '0.1': .995, '0.25': .9949, '0.5': .9947},
                'anchor_sha256': 'b' * 64, 'control_sha256': 'c' * 64, 'raw_prediction_sha256': 'd' * 64}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'lock.json'
            path.write_text(json.dumps(lock))
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            validate_lock(path, sha, 'a' * 64)
            with self.assertRaises(ValueError):
                validate_lock(path, sha, 'b' * 64)
            for change in [{'eligible_for_2022': False}, {'selected_weight': 0.},
                           {'control_micro_auc': .996}, {'anchor_micro_auc': .996},
                           {'scores_2021': {'0.0': .994}},
                           {'scores_2021': {'0.0': .994, '0.1': .995, '0.25': .996, '0.5': .9947}},
                           {'selected_micro_auc': .996}, {'anchor_sha256': 'not-a-digest'}]:
                path.write_text(json.dumps({**lock, **change}))
                new_sha = hashlib.sha256(path.read_bytes()).hexdigest()
                with self.assertRaises(ValueError):
                    validate_lock(path, new_sha, 'a' * 64)
                with self.assertRaises(ValueError):
                    validate_lock(path, sha, 'a' * 64)

    def test_splice_changes_only_declared_units_and_zero_is_exact(self):
        units = ['Haukeli_G1', 'Lio_G1', 'Hogga_G1', 'Vinje_G1', 'Vinje_G2', 'Vinje_G3']
        anchor = np.tile(np.asarray([[.1, .2, .3, .9, .6, .1], [.9, .8, .7, .8, .7, .2]]), (84, 1))
        raw = np.tile(np.asarray([[.7, .5, .9], [.2, .3, .4]]), (84, 1))
        result = compose(anchor, raw, [0, 1, 2], .25, units)
        np.testing.assert_array_equal(result[:, 3:], anchor[:, 3:])
        np.testing.assert_allclose(result[:, :3], .75 * anchor[:, :3] + .25 * raw)
        np.testing.assert_array_equal(compose(anchor, raw, [0, 1, 2], 0., units), anchor)


if __name__ == '__main__':
    unittest.main()
