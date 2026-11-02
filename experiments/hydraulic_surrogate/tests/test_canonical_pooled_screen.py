"""Chronology, row-unit alignment and fixed selection gates; no model fitting."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.screen_canonical_pooled import early_splits, read_screen_labels, select_candidate, ROUNDS, WEIGHTS


class CanonicalPooledScreenTest(unittest.TestCase):
    def setUp(self):
        self.metadata = pd.DataFrame({'Run No': np.arange(1, 3648),
            'starttime': pd.date_range('2015-01-01', periods=3647, tz='UTC').astype(str)})

    def test_six_day_purge_and_third_rng_sample(self):
        train, screen, actual = early_splits(self.metadata)
        self.assertEqual(self.metadata.iloc[train[-1]].starttime[:10], '2019-12-25')
        self.assertEqual(self.metadata.iloc[screen[0]].starttime[:10], '2020-01-01')
        self.assertEqual(self.metadata.iloc[screen[-1]].starttime[:10], '2020-12-25')
        rng = np.random.default_rng(4173)
        for year in [2018, 2019, 2020]:
            cutoff = pd.Timestamp(f'{year}-01-01', tz='UTC') - pd.Timedelta(days=6)
            count = sum(pd.Timestamp(date) < cutoff for date in self.metadata.starttime)
            expected = np.sort(rng.choice(count * 168, size=int(count * 168 * .4), replace=False))
        self.assertEqual(actual.tobytes(), expected.tobytes())
        self.assertLess(actual[-1], (train[-1] + 1) * 168)
        labels = np.arange(len(self.metadata) * 168 * 14).reshape(-1, 14)
        expanded = labels[actual].reshape(-1)
        self.assertEqual(expanded[14 * 17 + 9], labels[actual[17], 9])

    def test_labels_reader_omits_every_unselected_line_and_unit_hour_order(self):
        _, screen, _ = early_splits(self.metadata)
        ids = self.metadata.iloc[screen]['Run No'].to_numpy()
        units = [f'Unit{j}' for j in range(14)]
        calls = []
        def synthetic_read(path, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return self.metadata
            self.assertEqual(kwargs['skiprows'], [i + 1 for i in range(3647) if i not in set(screen)])
            expected_columns = ['Run No'] + [f'result_committed_{unit}_t{hour}' for unit in units for hour in range(168)]
            self.assertEqual(kwargs['usecols'], expected_columns)
            values = {'Run No': ids}
            values.update({name: np.full(360, (j + hour) % 2, dtype=np.uint8)
                           for j, unit in enumerate(units) for hour in range(168)
                           for name in [f'result_committed_{unit}_t{hour}']})
            return pd.DataFrame(values)
        with patch('scripts.screen_canonical_pooled.pd.read_csv', side_effect=synthetic_read):
            labels = read_screen_labels(None, ids, units)
        self.assertEqual(labels.shape, (360, 168, 14))
        self.assertEqual(len(calls), 2)
        for hour in [0, 1, 167]:
            for unit in range(14):
                self.assertEqual(labels[99, hour, unit], (hour + unit) % 2)

    def test_lower_round_then_weight_ties_and_no_half_year_rescue(self):
        anchor = {name: .8 for name in ['full', 'first_half', 'second_half']}
        controls = {str(w): dict(anchor) for w in WEIGHTS}
        grid = {f'{r}:{w}': dict(anchor) for r in ROUNDS for w in WEIGHTS}
        for r in ROUNDS:
            for w in [.1, .25]:
                grid[f'{r}:{w}'] = {name: .9 for name in anchor}
        result = select_candidate(grid, controls, anchor)
        self.assertEqual((result['selected_rounds'], result['selected_weight']), (700, .1))
        self.assertTrue(result['eligible_for_later_validation'])
        grid['1400:0.5'] = {'full': .95, 'first_half': .95, 'second_half': .79}
        result = select_candidate(grid, controls, anchor)
        self.assertEqual((result['selected_rounds'], result['selected_weight']), (1400, .5))
        self.assertFalse(result['eligible_for_later_validation'])
        grid = {f'{r}:{w}': dict(anchor) for r in ROUNDS for w in WEIGHTS}
        result = select_candidate(grid, controls, anchor)
        self.assertEqual((result['selected_rounds'], result['selected_weight']), (700, 0.))
        self.assertFalse(result['eligible_for_later_validation'])


if __name__ == '__main__':
    unittest.main()
