"""Check early OOF horizon separation and exact reused generator alignment."""
import unittest
import numpy as np
import pandas as pd
from scripts.build_full14_early_oof import split_indices, place_reused, VINJE


class EarlyOOFTests(unittest.TestCase):
    def test_actual_split_boundaries(self):
        dates = pd.Series(pd.to_datetime(['2017-12-25', '2017-12-26', '2018-01-01', '2018-12-25',
                                         '2018-12-31', '2019-01-01', '2019-12-25', '2019-12-26',
                                         '2020-01-01', '2020-12-25', '2020-12-26'], utc=True))
        tr, va = split_indices(dates, 2018)
        np.testing.assert_array_equal(tr, [0])
        np.testing.assert_array_equal(va, [2, 3, 4])
        tr, va = split_indices(dates, 2019)
        np.testing.assert_array_equal(tr, [0, 1, 2, 3])
        np.testing.assert_array_equal(va, [5, 6])
        tr, va = split_indices(dates, 2020)
        np.testing.assert_array_equal(tr, [0, 1, 2, 3, 4, 5, 6])
        np.testing.assert_array_equal(va, [8, 9])

    def test_exact_reuse_with_permuted_full_schema(self):
        units = ['other0', VINJE[2], *[f'other{i}' for i in range(1, 11)], VINJE[0], VINJE[1]]
        reused = np.arange(18, dtype=np.float32).reshape(2, 3, 3) / 20
        result = np.full((2, 3, 14), -1, dtype=np.float32)
        place_reused(result, reused, units)
        for j, name in enumerate(VINJE):
            np.testing.assert_array_equal(result[..., units.index(name)], reused[..., j])
        self.assertTrue(np.all(result[..., [units.index(f'other{i}') for i in range(11)]] == -1))

    def test_changed_scope_and_invalid_reuse_rejected(self):
        units = [*VINJE, *[f'other{i}' for i in range(11)]]
        result = np.empty((2, 3, 14), dtype=np.float32)
        for reused in [np.zeros((2, 3, 3), dtype=np.float64), np.full((2, 3, 3), 2, dtype=np.float32)]:
            with self.assertRaises(ValueError):
                place_reused(result, reused, units)
        with self.assertRaises(ValueError):
            split_indices(pd.Series(pd.to_datetime(['2020-01-01', '2021-01-01'], utc=True)), 2021)


if __name__ == '__main__':
    unittest.main()
