"""Boundary, missing-history, and timestamp tests for pure historical features."""
import unittest

import numpy as np
import pandas as pd

from sintef_project.past_state_features import build_past_state_features


class PastStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.volume = pd.DataFrame({'A': np.arange(10, dtype=float)},
                                   index=pd.date_range('2020-01-01', periods=10, freq='D', tz='UTC'))
        self.inflow = pd.DataFrame({'A': np.ones(24 * 12)},
                                   index=pd.date_range('2019-12-30', periods=24 * 12, freq='h', tz='UTC'))
        self.starts = pd.DatetimeIndex(['2020-01-08', '2020-01-02'], tz='UTC')

    def build(self, **kwargs: object) -> pd.DataFrame:
        return build_past_state_features(self.starts, self.volume, self.inflow, {'A': 10.}, **kwargs)[0]

    def test_exact_units_and_case_order(self) -> None:
        result = self.build()
        self.assertTrue(result.index.equals(self.starts))
        self.assertEqual(result.shape, (2, 15))
        self.assertTrue(all(dtype == np.float32 for dtype in result.dtypes))
        for days in (1, 3, 7):
            self.assertAlmostEqual(result.iloc[0][f'past_state_delta_fill_{days}d_A'], days / 10.)
            self.assertAlmostEqual(result.iloc[0][f'past_state_inflow_capacity_{days}d_A'], days * 24 * .0036 / 10.)
            self.assertAlmostEqual(result.iloc[0][f'past_state_withdrawal_proxy_{days}d_A'], days * (24 * .0036 - 1) / 10.)

    def test_no_future_or_current_hour_inflow(self) -> None:
        self.starts = self.starts[:1]
        expected = self.build()
        self.volume.loc[self.volume.index > self.starts[0], 'A'] = 1e15
        self.inflow.loc[self.inflow.index >= self.starts[0], 'A'] = 1e15
        pd.testing.assert_frame_equal(expected, self.build())
        # Removing future rows entirely must also leave every value unchanged.
        self.volume = self.volume.loc[self.volume.index <= self.starts[0]]
        self.inflow = self.inflow.loc[self.inflow.index < self.starts[0]]
        pd.testing.assert_frame_equal(expected, self.build())

    def test_early_missing_volume_uses_zero_and_flag(self) -> None:
        result = self.build()
        self.assertEqual(result.iloc[1]['past_state_volume_missing_3d_A'], 1)
        self.assertEqual(result.iloc[1]['past_state_delta_fill_3d_A'], 0)
        self.assertEqual(result.iloc[1]['past_state_withdrawal_proxy_3d_A'], 0)
        self.assertEqual(result.iloc[1]['past_state_inflow_missing_3d_A'], 0)
        self.assertGreater(result.iloc[1]['past_state_inflow_capacity_3d_A'], 0)

    def test_missing_hour_rejects_partial_window(self) -> None:
        self.inflow = self.inflow.drop(pd.Timestamp('2020-01-07T23:00Z'))
        result = self.build()
        self.assertEqual(result.iloc[0]['past_state_inflow_missing_1d_A'], 1)
        self.assertEqual(result.iloc[0]['past_state_inflow_capacity_1d_A'], 0)
        self.assertEqual(result.iloc[0]['past_state_withdrawal_proxy_1d_A'], 0)
        self.assertEqual(result.iloc[1]['past_state_inflow_missing_1d_A'], 0)

    def test_missing_current_volume_does_not_use_later_value(self) -> None:
        self.volume = self.volume.drop(self.starts[0])
        result = self.build()
        self.assertEqual(result.iloc[0]['past_state_volume_missing_1d_A'], 1)
        self.assertEqual(result.iloc[0]['past_state_delta_fill_1d_A'], 0)

    def test_nonfinite_history_flag_and_no_mutation(self) -> None:
        self.inflow.loc[pd.Timestamp('2020-01-07T01:00Z'), 'A'] = np.inf
        before = self.inflow.copy()
        result = self.build()
        self.assertEqual(result.iloc[0]['past_state_inflow_missing_1d_A'], 1)
        self.assertTrue(np.isfinite(result.to_numpy()).all())
        pd.testing.assert_frame_equal(self.inflow, before)

    def test_invalid_timestamps_and_capacity(self) -> None:
        with self.assertRaises(ValueError):
            build_past_state_features(self.starts.tz_localize(None), self.volume, self.inflow, {'A': 10.})
        with self.assertRaises(ValueError):
            build_past_state_features(self.starts, self.volume, self.inflow, {'A': 0.})
        with self.assertRaises(ValueError):
            build_past_state_features(self.starts, self.volume, pd.concat([self.inflow, self.inflow.iloc[:1]]), {'A': 10.})
        with self.assertRaises(ValueError):
            self.build(lookback_days=(1, 0))


if __name__ == '__main__':
    unittest.main()
