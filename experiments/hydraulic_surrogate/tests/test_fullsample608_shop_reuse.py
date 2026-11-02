"""Frozen tie, strict-gate, and conditional-year controls."""
import unittest
from scripts.select_fullsample608_shop_reuse import decision, check_lock, WEIGHTS


class ReuseSelectionTests(unittest.TestCase):
    def test_lower_weight_ties_and_identity_rejection(self):
        self.assertEqual(decision({str(w): .99 for w in WEIGHTS}, .98), (0., False))
        self.assertEqual(decision({'0.0': .98, '0.1': .99, '0.25': .99, '0.5': .98}, .985), (.1, True))

    def test_matched_control_requires_strict_improvement(self):
        scores = {'0.0': .98, '0.1': .99, '0.25': .995, '0.5': .99}
        self.assertEqual(decision(scores, .995), (.25, False))
        self.assertEqual(decision(scores, .996), (.25, False))
        self.assertEqual(decision(scores, .994), (.25, True))

    def test_lock_requires_actual_winner_and_exact_source(self):
        lock = {'version': 'fullsample608_shop_reuse_lock_v1', 'selection_year': 2021,
                'protocol_sha256': 'a', 'selector_sha256': 'b', 'candidate_weights': WEIGHTS,
                'scores_2021': {'0.0': .98, '0.1': .99, '0.25': .995, '0.5': .99},
                'selected_weight': .25, 'eligible_for_2022': True,
                'selected_2021_micro_auc': .995, 'anchor_2021_micro_auc': .98, 'control_2021_micro_auc': .994}
        check_lock(lock, 'a', 'b')
        for change in [{'selected_weight': .1}, {'eligible_for_2022': False}, {'selector_sha256': 'changed'},
                       {'control_2021_micro_auc': .995}]:
            with self.assertRaises(ValueError):
                check_lock(dict(lock, **change), 'a', 'b')


if __name__ == '__main__':
    unittest.main()
