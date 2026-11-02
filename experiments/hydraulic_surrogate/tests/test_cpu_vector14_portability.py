"""Synthetic only: preserve model inputs and reject altered or invalid schemas."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import xgboost as xgb
from scripts.cpu_vector14_portability import write_dataframe_rows, verify_dmatrix_blocks


class PortabilityTest(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({'float32': np.array([0, -0., 1.25, 1e-30], dtype=np.float32),
                             'float64': np.array([1 / 3, 1e20, -1e-20, np.pi], dtype=np.float64),
                             'integer': np.array([16777217, 1, -7, 0], dtype=np.int64)})

    def test_native_dataframe_and_npy_bytes(self):
        frame = self.frame(); rows = np.array([3, 1, 2], dtype=np.int64)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'synthetic.npy'
            record = write_dataframe_rows(frame, rows, path, frame.columns.tolist(), xgb, 2)
            self.assertTrue(record['dataframe_and_npy_native_inputs_bitwise'])
            self.assertEqual(np.load(path).tobytes(), frame.iloc[rows].to_numpy(dtype=np.float32).tobytes())
            with self.assertRaisesRegex(ValueError, 'exists'):
                write_dataframe_rows(frame, rows, path, frame.columns.tolist(), xgb)

    def test_reject_nonfinite_wrong_dtype_and_duplicate_rows(self):
        with self.assertRaisesRegex(ValueError, 'float32'):
            verify_dmatrix_blocks(np.zeros((2, 1), dtype=np.float64), ['a'], xgb)
        with self.assertRaisesRegex(ValueError, 'finite'):
            verify_dmatrix_blocks(np.array([[np.nan]], dtype=np.float32), ['a'], xgb)
        frame = self.frame()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                write_dataframe_rows(frame, np.array([1, 1]), Path(tmp) / 'bad.npy', frame.columns.tolist(), xgb)


if __name__ == '__main__': unittest.main()
