"""Small staging, fixed-protocol and exact-sample tests; no model/feature construction."""
from __future__ import annotations

import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from scripts import cuda_xgb608_loader as loader
from scripts import train_cuda_xgb608 as runner
from scripts.train_cuda_xgb import PARAMETERS as ORIGINAL_PARAMETERS


class CUDA608ProtocolTests(unittest.TestCase):
    def test_model_hyperparameters_equal_original395(self):
        self.assertEqual(runner.PARAMETERS, ORIGINAL_PARAMETERS)
        self.assertEqual(runner.WEIGHTS, [0., .1, .25, .5])

    def test_portable_spec_ignores_only_two_cache_paths(self):
        original = {'lp': {'cache_directory': '/a', 'hash': 'one'},
                    'lp_joint': {'cache_directory': '/b', 'hash': 'two'}, 'other': {'k': 1}}
        portable = {'lp': {'cache_directory': '/remote/a', 'hash': 'one'},
                    'lp_joint': {'cache_directory': '/remote/b', 'hash': 'two'}, 'other': {'k': 1}}
        self.assertEqual(loader.portable_spec(original), loader.portable_spec(portable))
        portable['lp_joint']['hash'] = 'changed'
        self.assertNotEqual(loader.portable_spec(original), loader.portable_spec(portable))
        self.assertEqual(original['lp']['cache_directory'], '/a')

    def test_staging_union_reuse_tamper_and_collision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, extra, merged = root/'base', root/'extra', root/'merged'
            base.mkdir(); extra.mkdir()
            (base/'raw.csv').write_text('original\n'); (extra/'joint.npz').write_bytes(b'fixture')
            def manifest(directory, name):
                path = directory/name
                return {'payload_files': [{'path': name, 'sha256': runner.digest(path), 'size_bytes': path.stat().st_size}]}
            bm, em = manifest(base, 'raw.csv'), manifest(extra, 'joint.npz')
            loader.merge_stages(base, extra, merged, bm, em)
            self.assertEqual((merged/'raw.csv').read_text(), 'original\n')
            loader.merge_stages(base, extra, merged, bm, em)
            (merged/'joint.npz').write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError, 'hash/size mismatch'):
                loader.merge_stages(base, extra, merged, bm, em)
            with self.assertRaisesRegex(ValueError, 'overlaps original'):
                loader.merge_stages(base, base, root/'collision', bm, bm)
            (merged/'unlisted').write_text('unexpected')
            with self.assertRaisesRegex(ValueError, 'missing/unlisted'):
                loader.merge_stages(base, extra, merged, bm, em)

    def test2022_requires_improving_exact_argmax_and_provenance(self):
        protocol = {'anchor': {'run_id': 'anchor608', 'analysis_sha256': 'a'*64}, 'local_selector_sha256': 'e'*64}
        lock = {'version': 'cuda_xgb608_2021_blend_lock_v1', 'selection_year': 2021,
                'training_protocol_sha256': 'b'*64, 'anchor_run': 'anchor608', 'anchor_sha256': 'a'*64,
                'candidate_weights': runner.WEIGHTS, 'selected_weight': .25, 'training_2021_complete': True,
                'gpu_run_manifest_sha256': 'c'*64, 'gpu_validation_2021_sha256': 'd'*64,
                'selector_sha256': 'e'*64, 'scores_2021': {'0.0': .9, '0.1': .91, '0.25': .92, '0.5': .915},
                'selected_2021_micro_auc': .92, 'anchor_2021_micro_auc': .9}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'selection.json'
            def verify(data):
                path.write_text(json.dumps(data))
                return runner.validate_lock(path, runner.digest(path), protocol, 'b'*64)
            verify(lock)
            for changed in [dict(selected_weight=.1), dict(selector_sha256='f'*64),
                            dict(scores_2021={'0.0': .9}), dict(selected_2021_micro_auc=.93),
                            dict(gpu_validation_2021_sha256='missing'),
                            dict(scores_2021={'0.0': .9, '0.1': float('nan'), '0.25': .92, '0.5': .915})]:
                with self.assertRaises(ValueError): verify({**lock, **changed})
            with self.assertRaisesRegex(ValueError, 'nonzero candidate'):
                verify({**lock, 'scores_2021': {str(w): .9 for w in runner.WEIGHTS},
                        'selected_weight': 0., 'selected_2021_micro_auc': .9})

    def test_exact_frozen_samples_both_years_without_real_features(self):
        root = Path(__file__).resolve().parents[1]
        baseline = json.loads((root/'artifacts/lgb-normalized-lp-context-joint-001/manifest.json').read_text())
        metadata = pd.read_csv(root/'sintef_project/data/kernel/Unit_commitment_decisions.csv', usecols=['Run No', 'starttime'])
        dates = pd.to_datetime(metadata.starttime, utc=True)
        for year in [2021, 2022]:
            sample = runner.reconstruct_sample(dates, year, baseline, np, pd)
            payload = io.BytesIO(); np.save(payload, sample)
            self.assertEqual(loader.core.hashlib.sha256(payload.getvalue()).hexdigest(), baseline['artifacts'][f'train_sample_{year}.npy'])


if __name__ == '__main__':
    unittest.main()
