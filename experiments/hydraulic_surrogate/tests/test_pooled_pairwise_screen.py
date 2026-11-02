"""Check unit-expanded feature alignment and the screen target decoder boundary."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from scripts.screen_pooled_pairwise_residual import expand_features, read_screen_labels
from sintef_project.pooled_pairwise_residual import base_logits


class PairwiseScreenTests(unittest.TestCase):
    def test_expanded_features_follow_case_hour_unit_order(self):
        features = np.arange(3 * 608, dtype=np.float32).reshape(3, 608)
        probability = np.arange(1, 13, dtype=np.float32).reshape(3, 4) / 13
        expanded = expand_features(features, probability)
        self.assertEqual(expanded.shape, (12, 613))
        for hour in range(3):
            for unit in range(4):
                row = expanded[hour * 4 + unit]
                np.testing.assert_array_equal(row[:608], features[hour])
                np.testing.assert_array_equal(row[608:612], np.eye(4)[unit])
        np.testing.assert_array_equal(expanded[:, -1], base_logits(probability).ravel().astype(np.float32))

    def test_later_target_values_are_not_decoded(self):
        units = ['test_unit']
        columns = [f'result_committed_test_unit_t{h}' for h in range(168)]
        table = pd.DataFrame({'Run No': [10, 11, 12], 'starttime': ['2019-12-25', '2020-01-01', '2021-01-01'],
                              **{c: ['poison', '1', 'poison'] for c in columns}})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'labels.csv'; table.to_csv(path, index=False)
            result = read_screen_labels(path, np.array([11]), units)
            np.testing.assert_array_equal(result, np.ones((1, 168, 1)))
            with self.assertRaises(ValueError):
                read_screen_labels(path, np.array([12]), units)
            with self.assertRaises(AssertionError):
                read_screen_labels(path, np.array([99]), units)


if __name__ == '__main__':
    unittest.main()
