"""Small CPU-only checks for GPU rejection, exact samples and fold gating."""
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from scripts.train_cuda_xgb import require_gpu_config, reconstruct_sample, validate_lock
from scripts.select_cuda_xgb_blend import require_anchor_array


class CUDAxgbProtocolTests(unittest.TestCase):
    def test_anchor_bytes_are_verified_against_frozen_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'selected_validation_2021.npz'
            path.write_bytes(b'original trusted bytes')
            manifest = {'artifacts': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}}
            require_anchor_array(path, manifest)
            path.write_bytes(b'tampered replacement')
            with self.assertRaises(ValueError):
                require_anchor_array(path, manifest)

    def test_gpu_device_and_updater_are_both_required(self) -> None:
        config = {'learner': {'generic_param': {'device': 'cuda:1'},
                              'gradient_booster': {'updater': 'grow_gpu_hist'}}}
        require_gpu_config(config)
        for device in ['cpu', 'cuda:0']:
            config['learner']['generic_param']['device'] = device
            with self.assertRaises(ValueError):
                require_gpu_config(config)
        config['learner']['generic_param']['device'] = 'cuda:1'
        config['learner']['gradient_booster']['updater'] = 'grow_quantile_histmaker'
        with self.assertRaises(ValueError):
            require_gpu_config(config)

    def test_second_year_consumes_first_year_rng_call_and_exact_hash(self) -> None:
        dates = pd.to_datetime(['2020-01-01', '2020-12-25', '2020-12-26',
                                '2021-01-01', '2021-12-25', '2021-12-26', '2022-01-01'], utc=True)
        rng = np.random.default_rng(4173)
        expected2021 = np.sort(rng.choice(np.arange(2 * 168), size=int(2 * 168 * .4), replace=False))
        expected2022 = np.sort(rng.choice(np.arange(5 * 168), size=int(5 * 168 * .4), replace=False))
        hashes = {}
        for year, sample in [(2021, expected2021), (2022, expected2022)]:
            buffer = io.BytesIO()
            np.save(buffer, sample)
            hashes[f'train_sample_{year}.npy'] = hashlib.sha256(buffer.getvalue()).hexdigest()
        baseline = {'artifacts': hashes}
        np.testing.assert_array_equal(reconstruct_sample(dates, 2022, baseline, np, pd), expected2022)
        hashes['train_sample_2022.npy'] = '0' * 64
        with self.assertRaises(ValueError):
            reconstruct_sample(dates, 2022, baseline, np, pd)

    def test_2022_lock_is_bound_to_protocol_anchor_and_exact_bytes(self) -> None:
        protocol = {'anchor': {'run_id': 'anchor', 'analysis_sha256': 'a' * 64}}
        lock = {'version': 'cuda_xgb_2021_blend_lock_v1', 'selection_year': 2021,
                'training_protocol_sha256': 'b' * 64, 'anchor_run': 'anchor',
                'anchor_sha256': 'a' * 64, 'candidate_weights': [0., .1, .25, .5],
                'selected_weight': .1, 'training_2021_complete': True,
                'gpu_run_manifest_sha256': 'c' * 64, 'gpu_validation_2021_sha256': 'd' * 64}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'selection.json'
            path.write_text(json.dumps(lock))
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            validate_lock(path, expected, protocol, 'b' * 64)
            with self.assertRaises(ValueError):
                validate_lock(path, expected, protocol, 'x' * 64)
            path.write_text(json.dumps({**lock, 'selected_weight': .5}))
            with self.assertRaises(ValueError):
                validate_lock(path, expected, protocol, 'b' * 64)


if __name__ == '__main__':
    unittest.main()
