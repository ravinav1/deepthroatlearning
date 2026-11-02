"""Actual-auditor contracts use tiny synthetic fixtures and no model fits."""
import copy
import unittest
import numpy as np
from scripts.audit_cpu_vector14_fold import forest_prediction, native_config, exit_evidence


def forest():
    tree = {'id': 0, 'tree_param': {'num_nodes': '3', 'size_leaf_vector': '14'},
            'left_children': [1, -1, -1], 'right_children': [2, 0, 1],
            'split_indices': [2, 0, 0], 'split_conditions': [1., 0., 0.],
            'leaf_weights': [-1.] * 14 + [2.] * 14}
    return {'learner': {'learner_model_param': {'num_target': '14', 'num_feature': '664', 'base_score': '[0.5]'},
        'objective': {'name': 'binary:logistic'}, 'gradient_booster': {'model': {
            'gbtree_model_param': {'num_trees': '1'}, 'trees': [tree], 'tree_info': [0], 'iteration_indptr': [0, 1]}}}}


class VectorAuditTest(unittest.TestCase):
    def test_vector_native_leaf_indices_and_boundary(self):
        x = np.zeros((3, 664), dtype=np.float32); x[:, 2] = [0., 1., 2.]
        actual = forest_prediction(forest(), x, 1)
        expected = 1 / (1 + np.exp(-np.array([-1., 2., 2.], dtype=np.float32)))
        np.testing.assert_array_equal(actual, np.repeat(expected[:, None], 14, axis=1))

    def test_scalar_or_bad_count_or_cycle_rejected(self):
        x = np.zeros((3, 664), dtype=np.float32)
        for mutation in ['scalar', 'count', 'cycle']:
            bad = forest()
            if mutation == 'scalar': bad['learner']['gradient_booster']['model']['trees'][0]['tree_param']['size_leaf_vector'] = '1'
            elif mutation == 'count': bad['learner']['gradient_booster']['model']['gbtree_model_param']['num_trees'] = '2'
            else: bad['learner']['gradient_booster']['model']['trees'][0]['left_children'][0] = 0
            with self.assertRaises(AssertionError): forest_prediction(bad, x, 1)

    def test_exit_evidence_requires_full_pid_scope_and_finalization(self):
        m = {'protocol_sha256': 'protocol', 'command': ['python', 'runner'], 'timestamp_utc': '2026-11-02T20:00:00+00:00', 'runtime_seconds': 60}
        o = {'status': 'passed', 'version': 'cpu_vector14_process_exit_v1', 'mode': 'fit', 'candidate_manifest_sha256': 'candidate',
             'protocol_sha256': 'protocol', 'observed_launch_command': m['command'], 'launch_observed': True,
             'coordinator_pid': 1, 'checked_pids': [1, 2, 3], 'command': ['ps', '-p', '1,2,3', '-o', 'pid=,stat=,command='],
             'returncode': 1, 'stdout': '', 'stderr': '', 'timestamp_utc': '2026-11-02T20:01:01+00:00'}
        exit_evidence(o, m, 'candidate', [2, 3], 'fit')
        for key, value in [('checked_pids', [1, 2]), ('returncode', 0), ('stdout', '3 S python'),
                           ('candidate_manifest_sha256', 'wrong'), ('timestamp_utc', '2026-11-02T20:00:59+00:00')]:
            with self.assertRaises(AssertionError): exit_evidence({**o, key: value}, m, 'candidate', [2, 3], 'fit')


if __name__ == '__main__': unittest.main()
