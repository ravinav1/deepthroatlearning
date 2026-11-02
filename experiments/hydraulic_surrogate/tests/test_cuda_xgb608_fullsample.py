"""Verify isolated full-row inclusion, original GPU recipe and fixed selection gate."""
import hashlib
import io
import json
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from scripts.train_cuda_xgb608 import PARAMETERS as BASE
from scripts import train_cuda_xgb608_fullsample as runner
from scripts.verify_cuda_xgb608_fullsample_return import verify_native_config


class FullSampleTests(unittest.TestCase):
    def test_parameters_identical_and_grid_fixed(self):
        self.assertEqual(runner.PARAMETERS, BASE)
        self.assertEqual(runner.WEIGHTS, [0., .1, .25, .5])

    def test_actual_full_purged_rows_and_original_subset(self):
        metadata = pd.read_csv('sintef_project/data/kernel/Unit_commitment_decisions.csv', usecols=['Run No', 'starttime'])
        dates = pd.to_datetime(metadata.starttime, utc=True)
        old = json.loads(Path('docs/cuda_xgb608_protocol.json').read_text())
        protocol = {'sample_fraction': 1., 'sample_sha256': {}}
        for year in [2021, 2022]:
            ids = old['folds'][str(year)]['train_ids']
            cases = np.flatnonzero(metadata['Run No'].isin(ids).to_numpy())
            rows = (cases[:, None] * 168 + np.arange(168)).ravel()
            memory = io.BytesIO(); np.save(memory, rows)
            protocol['sample_sha256'][str(year)] = hashlib.sha256(memory.getvalue()).hexdigest()
            actual = runner.reconstruct_sample(dates, year, protocol, np, pd)
            np.testing.assert_array_equal(actual, rows)
            self.assertEqual(len(actual), len(ids) * 168)
            self.assertTrue(np.isin(np.load(f'artifacts/lgb-normalized-lp-context-joint-001/train_sample_{year}.npy'), actual).all())
            bad = deepcopy(protocol); bad['sample_sha256'][str(year)] = '0' * 64
            with self.assertRaises(ValueError): runner.reconstruct_sample(dates, year, bad, np, pd)
        protocol['sample_fraction'] = .4
        with self.assertRaises(ValueError): runner.reconstruct_sample(dates, 2021, protocol, np, pd)

    def test_native_original600_gpu1_and_capacity_rejected(self):
        path = Path('artifacts/verify-cuda-xgb608-2021-return-001/download/cuda-xgb608-2021-001/config_2021_Hogga_G1.json')
        native = json.loads(path.read_text())
        verify_native_config(native, runner.PARAMETERS)
        changed = deepcopy(native); changed['learner']['generic_param']['device'] = 'cuda:0'
        with self.assertRaises(ValueError): verify_native_config(changed, runner.PARAMETERS)
        changed = deepcopy(native); changed['learner']['gradient_booster']['gbtree_model_param']['num_trees'] = '1200'
        with self.assertRaises(ValueError): verify_native_config(changed, runner.PARAMETERS)

    def test_failed_same_weight_control_lock_rejected(self):
        protocol = {'local_selector_sha256': 'a' * 64, 'anchor': {'2021': {'prediction_sha256': 'b' * 64}},
                    'control': {'2021': {'prediction_sha256': 'c' * 64}}}
        lock = {'version': 'cuda_xgb608_fullsample_2021_blend_lock_v1', 'selection_year': 2021,
                'training_protocol_sha256': 'd' * 64, 'candidate_weights': runner.WEIGHTS,
                'selected_weight': .25, 'selector_sha256': 'a' * 64,
                'scores_2021': {'0.0': .90, '0.1': .91, '0.25': .93, '0.5': .92},
                'selected_2021_micro_auc': .93, 'anchor_2021_micro_auc': .90,
                'control_2021_micro_auc': .925, 'training_2021_complete': True, 'eligible_for_2022': True,
                'anchor_prediction_sha256': 'b' * 64, 'control_prediction_sha256': 'c' * 64,
                'gpu_run_manifest_sha256': 'e' * 64, 'gpu_validation_2021_sha256': 'f' * 64}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'lock.json'; path.write_text(json.dumps(lock))
            runner.validate_lock(path, runner.digest(path), protocol, 'd' * 64)
            lock['control_2021_micro_auc'] = .94; path.write_text(json.dumps(lock))
            with self.assertRaises(ValueError): runner.validate_lock(path, runner.digest(path), protocol, 'd' * 64)


if __name__ == '__main__':
    unittest.main()
