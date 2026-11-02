"""Check exactly declared capacity overrides, native device and locked comparisons."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from scripts.train_cuda_xgb608 import PARAMETERS as BASE
from scripts import train_cuda_xgb608_capacity as runner
from scripts.verify_cuda_xgb608_capacity_return import verify_native_config


class CapacityTests(unittest.TestCase):
    def test_only_declared_overrides(self):
        changes = {key: value for key, value in runner.PARAMETERS.items() if BASE[key] != value}
        self.assertEqual(changes, {'n_estimators': 1200, 'max_depth': 8, 'learning_rate': .03, 'device': 'cuda:0'})
        self.assertEqual(set(BASE), set(runner.PARAMETERS))
        self.assertEqual(runner.WEIGHTS, [0., .1, .25, .5])

    def test_native_parameters_and_device(self):
        path = Path('artifacts/verify-cuda-xgb608-2021-return-001/download/cuda-xgb608-2021-001/config_2021_Hogga_G1.json')
        config = json.loads(path.read_text())
        config['learner']['generic_param']['device'] = 'cuda:0'
        config['learner']['gradient_booster']['gbtree_model_param']['num_trees'] = '1200'
        tree = config['learner']['gradient_booster']['tree_train_param']
        tree.update(eta='0.03', learning_rate='0.03', max_depth='8')
        verify_native_config(config, runner.PARAMETERS)
        for section, name, value in [('generic_param', 'device', 'cuda:1'), ('generic_param', 'seed', '2')]:
            changed = deepcopy(config); changed['learner'][section][name] = value
            with self.assertRaises(ValueError): verify_native_config(changed, runner.PARAMETERS)
        changed = deepcopy(config)
        changed['learner']['gradient_booster']['gbtree_model_param']['num_trees'] = '600'
        with self.assertRaises(ValueError): verify_native_config(changed, runner.PARAMETERS)

    def test_lock_refuses_failed_matched_control(self):
        protocol = {'local_selector_sha256': 'a'*64, 'anchor': {'2021': {'prediction_sha256': 'b'*64}},
                    'control': {'2021': {'prediction_sha256': 'c'*64}}}
        lock = {'version': 'cuda_xgb608_capacity_2021_blend_lock_v1', 'selection_year': 2021,
                'training_protocol_sha256': 'd'*64, 'candidate_weights': runner.WEIGHTS,
                'selected_weight': .25, 'selector_sha256': 'a'*64,
                'scores_2021': {'0.0': .90, '0.1': .91, '0.25': .93, '0.5': .92},
                'selected_2021_micro_auc': .93, 'anchor_2021_micro_auc': .90,
                'control_2021_micro_auc': .925, 'training_2021_complete': True, 'eligible_for_2022': True,
                'anchor_prediction_sha256': 'b'*64, 'control_prediction_sha256': 'c'*64,
                'gpu_run_manifest_sha256': 'e'*64, 'gpu_validation_2021_sha256': 'f'*64}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'lock.json';path.write_text(json.dumps(lock))
            runner.validate_lock(path, runner.digest(path), protocol, 'd'*64)
            lock['control_2021_micro_auc'] = .94;path.write_text(json.dumps(lock))
            with self.assertRaises(ValueError): runner.validate_lock(path, runner.digest(path), protocol, 'd'*64)


if __name__ == '__main__':
    unittest.main()
