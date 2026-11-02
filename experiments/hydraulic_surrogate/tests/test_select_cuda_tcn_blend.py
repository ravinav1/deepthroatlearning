"""Small synthetic download fixtures; no Torch imports or real feature table."""
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.metrics import roc_auc_score

from scripts import select_cuda_tcn_blend as selector


class DownloadVerifierTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.run = self.root / 'download'
        self.run.mkdir()
        baseline_path = self.root / 'artifacts/lgb-normalized-lp-001/manifest.json'
        baseline_path.parent.mkdir(parents=True)
        self.baseline = {'feature_spec': {'lp': {'cache_directory': '/old/path', 'version': 1}}}
        baseline_path.write_text(json.dumps(self.baseline))
        self.protocol = {'feature_columns': ['a', 'b', 'c'],
                         'folds': {'2021': {'train_ids': [1, 2], 'validation_ids': [3, 4]}}}
        (self.run / 'train_cuda_tcn.py').write_text('# Synthetic fixture, never executed\n')
        self.protocol['trainer_sha256'] = selector.digest(self.run / 'train_cuda_tcn.py')
        (self.run / 'protocol.json').write_text(json.dumps(self.protocol))
        self.protocol_sha = selector.digest(self.run / 'protocol.json')
        np.savez_compressed(self.run / 'normalization.npz', mean=np.zeros(3), scale=np.ones(3),
                            columns=self.protocol['feature_columns'], training_case_ids=[1, 2])
        np.save(self.run / 'epoch_training_case_orders.npy', np.tile([1, 2], (24, 1)))
        self.units = list(selector.DEFAULT_UNITS)
        rng = np.random.default_rng(817)
        self.values = rng.uniform(.01, .99, (336, 14)).astype(np.float32)
        self.labels = rng.integers(0, 2, (336, 14), dtype=np.int8)
        np.savez_compressed(self.run / 'validation_2021.npz', predictions=self.values,
                            labels=self.labels, case_ids=[3, 4], units=self.units)
        np.save(self.run / 'replay_predictions_2021.npy', self.values)
        (self.run / 'tcn_2021.pt').write_bytes(b'Only a checksum fixture; never deserialize')
        self.bundle = {'fixture': True}
        self.manifest = {'status': 'complete', 'mode': 'fit', 'year': 2021, 'seed': 4173,
                         'completed_epochs': 24, 'protocol_sha256': self.protocol_sha,
                         'trainer_sha256': self.protocol['trainer_sha256'], 'bundle': self.bundle,
                         'baseline_reference_sha256': selector.digest(baseline_path),
                         'train_ids': [1, 2], 'validation_ids': [3, 4], 'purge_days': 6,
                         'all_training_case_hours': True, 'training_rows': 336,
                         'early_stopping': None, 'checkpoint_selection': 'fixed final epoch24 only',
                         'feature_spec': {'lp': {'cache_directory': '/portable/path', 'version': 1}},
                         'cuda': {'selected_device': 'cuda:0', 'cpu_fallback': False,
                                  'deterministic_algorithms': True, 'float32': True, 'amp': False,
                                  'compile': False, 'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'devices': ['fixture']},
                         'environment': {'venv': {'system_site_packages': True, 'package_actions': 'none'}},
                         'epochs': [{'epoch': i+1, 'training_bce': .5, 'seconds': 1,
                                     'learning_rate': .0001+.0009*(1+math.cos(math.pi*i/24))/2} for i in range(24)],
                         'units': self.units, 'raw_micro_auc': float(roc_auc_score(self.labels.ravel(), self.values.ravel())),
                         'unit_auc': {u: float(roc_auc_score(self.labels[:, j], self.values[:, j])) for j, u in enumerate(self.units)},
                         'checkpoint_replay': {'fresh_model': True, 'saved_float64_normalizer': True,
                                               'eval_inference_mode': True, 'bitwise_equal': True,
                                               'max_abs_probability_difference': 0.}}
        self.refresh()

    def tearDown(self):
        self.tmp.cleanup()

    def refresh(self):
        self.manifest['artifacts'] = {p.name: selector.digest(p) for p in self.run.iterdir() if p.name != 'manifest.json'}
        self.manifest['checkpoint_sha256'] = selector.digest(self.run / 'tcn_2021.pt')
        self.manifest['validation_sha256'] = selector.digest(self.run / 'validation_2021.npz')
        self.manifest['checkpoint_replay']['replay_predictions_sha256'] = selector.digest(self.run / 'replay_predictions_2021.npy')
        (self.run / 'manifest.json').write_text(json.dumps(self.manifest))

    def verify(self):
        with patch.object(selector, 'ROOT', self.root):
            return selector.verify_download(self.run, self.protocol, self.protocol_sha, 2021, self.baseline, self.bundle)

    def test_valid_download_and_portable_cache_path(self):
        proof, raw, labels, ids, units = self.verify()
        self.assertEqual(proof['training_arrays']['epochs'], 24)
        self.assertFalse(proof['local_checkpoint_inference'])
        self.assertEqual(raw.tobytes(), self.values.tobytes())

    def test_tampered_checkpoint_fails_before_array_decode(self):
        (self.run / 'tcn_2021.pt').write_bytes(b'Changed')
        with patch.object(selector.np, 'load', side_effect=AssertionError('Arrays decoded too early')):
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                self.verify()

    def test_extra_unlisted_file_is_rejected_before_array_decode(self):
        (self.run / 'unlisted.txt').write_text('unexpected')
        with patch.object(selector.np, 'load', side_effect=AssertionError('Arrays decoded too early')):
            with self.assertRaisesRegex(ValueError, 'directory allowlist differs'):
                self.verify()

    def test_symlink_is_rejected_even_when_content_hash_matches(self):
        checkpoint = self.run / 'tcn_2021.pt'
        outside = self.root / 'outside.pt'
        checkpoint.rename(outside)
        checkpoint.symlink_to(outside)
        with patch.object(selector.np, 'load', side_effect=AssertionError('Arrays decoded too early')):
            with self.assertRaisesRegex(ValueError, 'contains symlinks'):
                self.verify()

    def test_changed_replay_even_with_refreshed_file_hash_is_rejected(self):
        altered = self.values.copy()
        altered[0, 0] = .999
        np.save(self.run / 'replay_predictions_2021.npy', altered)
        self.refresh()
        with self.assertRaisesRegex(ValueError, 'replay predictions differ'):
            self.verify()

    def test_heldout_case_in_statistics_is_rejected(self):
        np.savez_compressed(self.run / 'normalization.npz', mean=np.zeros(3), scale=np.ones(3),
                            columns=self.protocol['feature_columns'], training_case_ids=[1, 3])
        self.refresh()
        with self.assertRaisesRegex(ValueError, 'different training cases'):
            self.verify()

    def test_missing_epoch_case_is_rejected(self):
        orders = np.tile([1, 2], (24, 1))
        orders[20] = [1, 1]
        np.save(self.run / 'epoch_training_case_orders.npy', orders)
        self.refresh()
        with self.assertRaisesRegex(ValueError, 'omits/repeats/adds'):
            self.verify()

    def test_feature_formula_change_is_rejected(self):
        self.manifest['feature_spec']['lp']['version'] = 2
        self.refresh()
        with self.assertRaisesRegex(ValueError, 'Feature formulas differ'):
            self.verify()

    def test_only_requested_blends_are_scored_and_ties_choose_zero(self):
        prior = selector.project_vinje(self.values, self.units)
        scores = selector.score_blends(prior, prior, self.labels, self.units, [0., .25])
        self.assertEqual(set(scores), {'0.0', '0.25'})
        self.assertEqual(scores['0.0'], scores['0.25'])
        self.assertEqual(max([0., .25], key=lambda w: scores[str(w)]), 0.)


if __name__ == '__main__':
    unittest.main()
