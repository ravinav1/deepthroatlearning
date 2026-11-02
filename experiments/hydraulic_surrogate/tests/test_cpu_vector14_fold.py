"""Prospective selection and native-prefix contracts without competition access."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from scripts.cpu_vector14_fold_common import choose, validate_lock, sha, ROUNDS, WEIGHTS
from scripts.train_cpu_vector14_fold import tree_prefix


class VectorFoldTest(unittest.TestCase):
    def test_joint_ties_prefer_round_then_weight(self):
        scores = {f'{r}:{w}': .9 for r in ROUNDS for w in WEIGHTS}
        self.assertEqual(choose(scores), (600, 0.))
        scores['600:0.5'] = scores['1800:0.1'] = .91
        self.assertEqual(choose(scores), (600, .5))
        scores['600:0.1'] = .91
        self.assertEqual(choose(scores), (600, .1))

    def test_missing_capacity_and_nonfinite_scores_reject(self):
        scores = {f'{r}:{w}': .9 for r in ROUNDS for w in WEIGHTS}
        del scores['1800:0.5']
        with self.assertRaises(AssertionError): choose(scores)
        scores['1800:0.5'] = float('nan')
        with self.assertRaises(AssertionError): choose(scores)

    def test_prefix_checks_coefficients_base_and_round_boundaries(self):
        def saved(count):
            return {'learner': {'learner_model_param': {'base_score': '[.5]'},
                'objective': {'name': 'binary:logistic'}, 'gradient_booster': {'model': {
                    'trees': [{'id': i, 'leaf_weights': [i] * 14} for i in range(count)],
                    'tree_info': [0] * count, 'iteration_indptr': list(range(count + 1))}}}}
        long, short = saved(18), saved(6); tree_prefix(long, short, 6)
        for mutation in ['tree', 'base', 'boundary']:
            bad = copy.deepcopy(short)
            if mutation == 'tree': bad['learner']['gradient_booster']['model']['trees'][2]['leaf_weights'][0] += 1
            elif mutation == 'base': bad['learner']['learner_model_param']['base_score'] = '[.4]'
            else: bad['learner']['gradient_booster']['model']['iteration_indptr'][3] = 4
            with self.assertRaises(AssertionError): tree_prefix(long, bad, 6)

    def test_lock_rejects_other_protocol_audit_or_false_identity(self):
        scores = {f'{r}:{w}': .9 for r in ROUNDS for w in WEIGHTS}; scores['600:0.1'] = .91
        lock = {'version': 'cpu_vector14_2021_lock_v1', 'protocol_sha256': 'protocol',
                'candidate_rounds': ROUNDS, 'candidate_weights': WEIGHTS, 'scores_2021': scores,
                'selected_rounds': 600, 'selected_weight': .1, 'eligible_for_2022': True,
                'selected_2021_micro_auc': .91, 'anchor_2021_micro_auc': .9, 'control_2021_micro_auc': .905,
                'anchor_prediction_sha256': 'anchor', 'control_prediction_sha256': 'control'}
        p = {kind: {'2021': {'prediction_sha256': kind}} for kind in ['anchor', 'control']}
        with tempfile.TemporaryDirectory() as tmp:
            lp, ap = Path(tmp) / 'lock.json', Path(tmp) / 'audit.json'
            def check(locked, protocol):
                lp.write_text(json.dumps(locked)); h = sha(lp)
                ap.write_text(json.dumps({'status': 'passed', 'protocol_sha256': protocol,
                    'selection_lock_sha256': h, 'eligible_for_2022': True}))
                return validate_lock(lp, h, p, 'protocol', ap, sha(ap))
            check(lock, 'protocol')
            with self.assertRaises(AssertionError): check(lock, 'different')
            bad = copy.deepcopy(lock); bad['anchor_2021_micro_auc'] = .89
            with self.assertRaises(AssertionError): check(bad, 'protocol')


if __name__ == '__main__': unittest.main()
