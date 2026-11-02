"""Targeted664 checks for chronological locks, matched control and input discovery."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import train_cuda_xgb664 as runner
from scripts import launch_cuda_xgb664_dataset as launcher
from scripts import select_cuda_xgb664_blend as selector
from scripts.verify_cuda_xgb664_return import verify_native_config
from scripts.train_cuda_xgb608 import PARAMETERS as PRIOR_PARAMETERS


class CUDA664Tests(unittest.TestCase):
    def test_matched_parameters_and_grid(self):
        self.assertEqual(runner.PARAMETERS, PRIOR_PARAMETERS)
        self.assertEqual(runner.WEIGHTS, [0., .1, .25, .5])

    def test_native_config_rejects_tree_feature_or_hyperparameter_changes(self):
        root = Path(__file__).resolve().parents[1]
        path = root / 'artifacts/verify-cuda-xgb608-2021-return-001/download/cuda-xgb608-2021-001/config_2021_Hogga_G1.json'
        config = json.loads(path.read_text())
        config['learner']['learner_model_param']['num_feature'] = '664'
        verify_native_config(config, runner.PARAMETERS)
        for section, name, value in [('learner_model_param', 'num_feature', '608'),
                                      ('gbtree_model_param', 'num_trees', '599'),
                                      ('tree_train_param', 'min_child_weight', '21')]:
            changed = json.loads(json.dumps(config))
            container = changed['learner'] if section == 'learner_model_param' else changed['learner']['gradient_booster']
            container[section][name] = value
            with self.assertRaises(ValueError):
                verify_native_config(changed, runner.PARAMETERS)

    def test_lock_requires_argmax_and_both_comparisons(self):
        protocol = {'local_selector_sha256': 'a'*64,
                    'anchor': {'2021': {'prediction_sha256': 'b'*64}},
                    'control': {'2021': {'prediction_sha256': 'c'*64}}}
        lock = {'version': 'cuda_xgb664_2021_blend_lock_v1', 'selection_year': 2021,
                'training_protocol_sha256': 'd'*64, 'candidate_weights': runner.WEIGHTS,
                'selected_weight': .25, 'selector_sha256': 'a'*64,
                'scores_2021': {'0.0': .90, '0.1': .91, '0.25': .93, '0.5': .92},
                'selected_2021_micro_auc': .93, 'anchor_2021_micro_auc': .90,
                'control_2021_micro_auc': .925, 'training_2021_complete': True, 'eligible_for_2022': True,
                'anchor_prediction_sha256': 'b'*64, 'control_prediction_sha256': 'c'*64,
                'gpu_run_manifest_sha256': 'e'*64, 'gpu_validation_2021_sha256': 'f'*64}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'lock.json'
            def check(value):
                path.write_text(json.dumps(value))
                return runner.validate_lock(path, runner.digest(path), protocol, 'd'*64)
            check(lock)
            for delta in [{'control_2021_micro_auc': .94}, {'control_2021_micro_auc': .93},
                          {'selected_weight': .1}, {'eligible_for_2022': False},
                          {'training_2021_complete': False}, {'control_prediction_sha256': '0'*64},
                          {'gpu_validation_2021_sha256': 'missing'}, {'anchor_2021_micro_auc': .91}]:
                with self.assertRaises(ValueError):
                    check({**lock, **delta})
            tied = {**lock, 'scores_2021': {'0.0': .90, '0.1': .93, '0.25': .93, '0.5': .92}}
            with self.assertRaises(ValueError):
                check(tied)

    def test_reference_requires_byte_and_label_alignment(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            labels = np.array([[0, 1], [1, 0]], dtype=np.uint8)
            ids = np.array([9])
            values = np.array([[.1, .9], [.8, .2]])
            path = root / 'prediction.npz'
            np.savez(path, labels=labels, predictions=values, case_ids=ids, units=np.array(['A', 'B']))
            prediction_sha = runner.digest(path)
            manifest_path = root / 'manifest.json'
            manifest_path.write_text(json.dumps({'status': 'complete', 'artifacts': {path.name: prediction_sha}}))
            record = {'manifest': manifest_path.name, 'manifest_sha256': runner.digest(manifest_path),
                      'prediction': path.name, 'prediction_sha256': prediction_sha}
            with patch.object(selector, 'ROOT', root):
                np.testing.assert_array_equal(selector.load_pinned_prediction(record, labels, ids, ['A', 'B']), values)
                with self.assertRaisesRegex(ValueError, 'alignment'):
                    selector.load_pinned_prediction(record, labels[::-1], ids, ['A', 'B'])
                path.write_bytes(b'tampered')
                with self.assertRaisesRegex(ValueError, 'predictions changed'):
                    selector.load_pinned_prediction(record, labels, ids, ['A', 'B'])

    def test_smoke_discovery_never_requires_input_bundles(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            protocol = {}
            for name, key in [('train_cuda_xgb664.py', 'trainer_sha256'),
                              ('cuda_xgb664_loader.py', 'shared_loader_sha256'),
                              ('cuda_xgb608_loader.py', 'prior_loader_sha256'),
                              ('train_cuda_tcn.py', 'core_loader_sha256'),
                              ('launch_cuda_xgb.py', 'launcher_sha256')]:
                path = root / name
                path.write_text(name)
                protocol[key] = runner.digest(path)
            protocol_path = root / 'cuda_xgb664_protocol.json'
            protocol_path.write_text(json.dumps(protocol))
            found = launcher.discover(root, runner.digest(protocol_path))
            self.assertNotIn('bundle', found)
            self.assertNotIn('shop_bundle', found)
            (root / 'duplicate').mkdir()
            (root / 'duplicate' / protocol_path.name).write_bytes(protocol_path.read_bytes())
            with self.assertRaisesRegex(ValueError, 'found 2'):
                launcher.discover(root, runner.digest(protocol_path))


if __name__ == '__main__':
    unittest.main()
