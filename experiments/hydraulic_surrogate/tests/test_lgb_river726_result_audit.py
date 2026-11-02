"""Synthetic lock/prefix/control rejection checks; no cache or fit is opened."""
import copy
import unittest

import numpy as np
import pandas as pd

from scripts.audit_lgb_river726_result import check_lock, prefix_fingerprint, verify_control


class ResultAuditTests(unittest.TestCase):
    def fixture(self):
        protocol = {'candidate_weights': [0., .1, .25, .5],
                    'source': {'scripts/train_lgb_river726.py': 'trainer'},
                    'anchor': {'2021': {'prediction_sha256': 'anchor'}},
                    'control': {'2021': {'prediction_sha256': 'control'}}}
        lock = {'version': 'lgb_river726_selection_lock_v1', 'selection_year': 2021,
                'protocol_sha256': 'protocol', 'trainer_sha256': 'trainer',
                'candidate_weights': [0., .1, .25, .5],
                'scores_2021': {'0.0': .8, '0.1': .9, '0.25': .9, '0.5': .85},
                'selected_weight': .1, 'selected_2021_micro_auc': .9,
                'anchor_2021_micro_auc': .8, 'control_2021_micro_auc': .88,
                'eligible_for_2022': True, 'anchor_prediction_sha256': 'anchor',
                'control_prediction_sha256': 'control'}
        return protocol, lock

    def test_lower_weight_tie_and_strict_control_gate(self):
        protocol, lock = self.fixture()
        check_lock(lock, protocol, 'protocol')
        bad = copy.deepcopy(lock)
        bad['selected_weight'] = .25
        with self.assertRaises(AssertionError):
            check_lock(bad, protocol, 'protocol')
        bad = copy.deepcopy(lock)
        bad['control_2021_micro_auc'] = .9
        with self.assertRaises(AssertionError):
            check_lock(bad, protocol, 'protocol')

    def test_protocol_and_reference_substitution_are_rejected(self):
        protocol, lock = self.fixture()
        for field in ['protocol_sha256', 'trainer_sha256', 'anchor_prediction_sha256', 'control_prediction_sha256']:
            bad = copy.deepcopy(lock)
            bad[field] = 'changed'
            with self.assertRaises(AssertionError):
                check_lock(bad, protocol, 'protocol')

    def test_prefix_fingerprint_catches_dtype_value_and_order_changes(self):
        prefix = pd.DataFrame({'one': np.asarray([1., 2.], dtype=np.float32),
                               'two': np.asarray([0, 1], dtype=np.int16)})
        names = ['one', 'two']
        expected = prefix_fingerprint(prefix, names)
        extended = prefix.assign(extra=np.asarray([9., 10.], dtype=np.float32))
        self.assertEqual(prefix_fingerprint(extended, names), expected)
        changed = extended.copy()
        changed['one'] = changed['one'].astype(np.float64)
        self.assertNotEqual(prefix_fingerprint(changed, names), expected)
        changed = extended.copy()
        changed.loc[0, 'two'] = 1
        self.assertNotEqual(prefix_fingerprint(changed, names), expected)
        with self.assertRaises(AssertionError):
            prefix_fingerprint(prefix[['two', 'one']], names)

    def test_selected_blend_cannot_replace_raw664_control(self):
        with self.assertRaises(AssertionError):
            verify_control({'prediction': 'artifacts/lgb-shop664-2021-001/selected_validation_2021.npz'}, 2021, {})


if __name__ == '__main__':
    unittest.main()
