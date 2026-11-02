"""Prefix comparison must reject tree or intercept drift independently of scores."""
import copy
import unittest
from scripts.pilot_cpu_vector14_timing import compare_prefix


def payload(count):
    return {'learner': {'learner_model_param': {'base_score': '[.5]', 'num_target': '14'},
                       'objective': {'name': 'binary:logistic'}, 'gradient_booster': {'model': {
                           'trees': [{'id': n, 'leaf_weights': [float(n)] * 14} for n in range(count)],
                           'tree_info': [0] * count, 'iteration_indptr': list(range(count + 1))}}}}


class TestVectorPrefix(unittest.TestCase):
    def test_exact(self):
        compare_prefix(payload(64), payload(16))

    def test_tree_change(self):
        short = payload(16)
        short['learner']['gradient_booster']['model']['trees'][3]['leaf_weights'][7] += 1
        with self.assertRaisesRegex(ValueError, 'tree payloads'):
            compare_prefix(payload(64), short)

    def test_base_score_change(self):
        short = payload(16)
        short['learner']['learner_model_param']['base_score'] = '[.4]'
        with self.assertRaisesRegex(ValueError, 'Base score'):
            compare_prefix(payload(64), short)

    def test_iteration_boundary_change(self):
        short = payload(16)
        short['learner']['gradient_booster']['model']['iteration_indptr'][5] += 1
        with self.assertRaisesRegex(ValueError, 'Round boundaries'):
            compare_prefix(payload(64), short)


if __name__ == '__main__': unittest.main()
