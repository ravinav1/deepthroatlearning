"""Synthetic archive and native-config checks; no XGBoost import or model fit."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts.verify_cuda_xgb_return import digest, import_archive, verify_native_config


class GPUXGBReturnTests(unittest.TestCase):
    def test_safe_exact_hash_archive_import_and_exclusive_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / 'transfer.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('gpu/manifest.json', '{}')
                z.writestr('smoke/manifest.json', '{}')
                z.writestr('gpu.log', 'Training log')
            target = root / 'out'
            import_archive(archive, digest(archive), target, 'gpu', 'smoke')
            self.assertEqual((target / 'gpu/manifest.json').read_text(), '{}')
            self.assertEqual((target / 'gpu.log').read_text(), 'Training log')
            with self.assertRaisesRegex(ValueError, 'already exists'):
                import_archive(archive, digest(archive), target, 'gpu', 'smoke')
            with self.assertRaisesRegex(ValueError, 'hosted SHA256'):
                import_archive(archive, '0'*64, root / 'wronghash', 'gpu', 'smoke')

    def test_escape_unlisted_namespace_and_symlink_fail_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for i, name in enumerate(['gpu/../../outside', '/gpu/x', 'unexpected/x', 'gpu\\x']):
                archive = root / f'{i}.zip'
                with zipfile.ZipFile(archive, 'w') as z:
                    z.writestr(name, 'x')
                target = root / f'out{i}'
                with self.assertRaises(ValueError):
                    import_archive(archive, digest(archive), target, 'gpu', 'smoke')
                self.assertFalse(target.exists())
            archive = root / 'symlink.zip'
            item = zipfile.ZipInfo('gpu/link')
            item.external_attr = 0o120777 << 16
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr(item, '../../outside')
            with self.assertRaisesRegex(ValueError, 'symlinks'):
                import_archive(archive, digest(archive), root / 'symlinkout', 'gpu', 'smoke')

    def test_native_gpu_full_recipe_and_tree_count_are_enforced(self):
        params = {'learning_rate': .05, 'max_depth': 6, 'min_child_weight': 20,
                  'reg_lambda': 4., 'reg_alpha': 0., 'max_bin': 128, 'subsample': .9,
                  'colsample_bytree': .9, 'colsample_bylevel': 1., 'colsample_bynode': 1.,
                  'max_delta_step': 0., 'grow_policy': 'depthwise'}
        tree = {'eta': '0.0500000007', 'max_depth': '6', 'min_child_weight': '20',
                'lambda': '4', 'alpha': '0', 'max_bin': '128', 'subsample': '0.899999976',
                'colsample_bytree': '0.899999976', 'colsample_bylevel': '1', 'colsample_bynode': '1',
                'max_delta_step': '0', 'grow_policy': 'depthwise'}
        config = {'learner': {'generic_param': {'device': 'cuda:1', 'nthread': '2', 'seed': '4173'},
                              'objective': {'name': 'binary:logistic'}, 'learner_model_param': {'num_feature': '395'},
                              'gradient_booster': {'updater': 'grow_gpu_hist',
                                  'gbtree_model_param': {'num_trees': '600', 'num_parallel_tree': '1'},
                                  'tree_train_param': tree}}}
        verify_native_config(config, params)
        for branch, key, value in [('generic_param', 'device', 'cpu'), ('generic_param', 'seed', '0')]:
            bad = copy.deepcopy(config)
            bad['learner'][branch][key] = value
            with self.assertRaises(ValueError):
                verify_native_config(bad, params)
        for key, value in [('eta', '.1'), ('max_depth', '8'), ('subsample', '1')]:
            bad = copy.deepcopy(config)
            bad['learner']['gradient_booster']['tree_train_param'][key] = value
            with self.assertRaises(ValueError):
                verify_native_config(bad, params)
        bad = copy.deepcopy(config)
        bad['learner']['gradient_booster']['gbtree_model_param']['num_trees'] = '599'
        with self.assertRaises(ValueError):
            verify_native_config(bad, params)


if __name__ == '__main__':
    unittest.main()
