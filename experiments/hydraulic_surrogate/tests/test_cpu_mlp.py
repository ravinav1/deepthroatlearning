"""Verify target withholding, frozen normalization and portable MLP replay."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits

from sintef_project.mlp_helpers import build_without_later_targets, fit_standardization, standardize, portable_predict
from sintef_project.postprocess import DEFAULT_UNITS, project_vinje
from scripts.cpu_mlp import mix, save_checkpoint


class CPUMlpTests(unittest.TestCase):
    def test_later_targets_are_not_decoded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / 'sintef_project/data/kernel/Unit_commitment_decisions.csv'
            target.parent.mkdir(parents=True)
            target.write_text('Run No,starttime,total_calculation_time,result_committed_A_t0\n'
                              '1,2020-01-01 00:00:00+00:00,1,1\n'
                              '2,2021-01-01 00:00:00+00:00,1,0\n'
                              '3,2022-01-01 00:00:00+00:00,1,WITHHELD_NOT_A_LABEL\n')
            original = pd.read_csv
            with patch.object(pd, 'read_csv', wraps=original) as reader:
                result = build_without_later_targets(root, lambda _: pd.read_csv(target))
            self.assertEqual(result['result_committed_A_t0'].tolist(), [1, 0, 0])
            self.assertEqual([call.kwargs for call in reader.call_args_list],
                             [{'usecols': ['Run No', 'starttime']}, {'nrows': 2}])

    def test_frozen_population_statistics_and_clipping(self) -> None:
        training = np.asarray([[1., 3.], [3., 3.], [5., 3.]], dtype=np.float32)
        mean, deviation, scale = fit_standardization(training)
        np.testing.assert_array_equal(mean, [3., 3.])
        np.testing.assert_allclose(deviation, [np.sqrt(8. / 3.), 0.], rtol=0, atol=1e-15)
        self.assertEqual(mean.dtype, np.float64)
        self.assertEqual(scale[1], 1.)
        held = np.asarray([[1.e9, 3.], [-1.e9, 3.]], dtype=np.float32)
        normalized = standardize(held, mean, scale, chunk_rows=1)
        np.testing.assert_array_equal(normalized, [[10., 0.], [-10., 0.]])
        self.assertEqual(normalized.dtype, np.float32)
        np.testing.assert_array_equal(mean, [3., 3.])

    def test_nonfinite_features_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            fit_standardization(np.asarray([[np.nan]], dtype=np.float32))
        with self.assertRaises(ValueError):
            standardize(np.asarray([[np.inf]], dtype=np.float32), np.zeros(1), np.ones(1))

    def test_zero_weight_preserves_projected_anchor(self) -> None:
        rng = np.random.default_rng(31)
        anchor = project_vinje(rng.uniform(size=(168, 14)), DEFAULT_UNITS)
        neural = rng.uniform(size=(168, 14)).astype(np.float32)
        np.testing.assert_array_equal(mix(anchor, neural, list(DEFAULT_UNITS), 0.), anchor)

    def test_pickle_and_portable_replay_are_exact(self) -> None:
        rng = np.random.default_rng(4173)
        x = rng.standard_normal((128, 4), dtype=np.float32)
        y = rng.integers(0, 2, size=(128, 14), dtype=np.uint8)
        model = MLPClassifier(hidden_layer_sizes=(5, 3), random_state=4173, max_iter=3,
                              batch_size=32, early_stopping=False, tol=0., n_iter_no_change=1000)
        with threadpool_limits(2), warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)
            model.fit(x, y)
            expected = model.predict_proba(x)
            np.testing.assert_array_equal(portable_predict(x, model.coefs_, model.intercepts_), expected)
            with tempfile.TemporaryDirectory() as directory:
                proof = save_checkpoint(Path(directory), 2021, model, x, expected,
                                        ['a', 'b', 'c', 'd'], list(DEFAULT_UNITS))
                self.assertEqual(proof['replayed_rows'], 128)
                self.assertEqual(proof['pickle_replay_max_error'], 0.)
                self.assertEqual(proof['portable_replay_max_error'], 0.)


if __name__ == '__main__':
    unittest.main()
