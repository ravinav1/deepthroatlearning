"""Prevent later-year target decoding when extending the frozen latent map."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.latent_shadow_splice_2022 import SELECTION_SHA, TARGET_UNITS, read_new_training_labels


class Shadow2022DecoderTests(unittest.TestCase):
    def test_selected_training_rows_skip_poison_later_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dates = pd.date_range('2020-12-26', periods=365, freq='D', tz='UTC')
            ids = list(range(2187, 2552))
            columns = [f'result_committed_{u}_t{h}' for u in TARGET_UNITS for h in range(168)]
            data = {'Run No': [*ids, 3000], 'starttime': [*dates.astype(str), '2022-01-01 00:00:00+00:00']}
            data.update({name: [0] * 365 + ['FORBIDDEN'] for name in columns})
            path = root / 'labels.csv'
            pd.DataFrame(data).to_csv(path, index=False)
            lock = root / 'lock.json'
            lock.write_text(json.dumps({'eligible_for_2022': True, 'year': 2021, 'selected_weight': .05}))
            with patch('scripts.latent_shadow_splice_2022.digest', return_value=SELECTION_SHA):
                result = read_new_training_labels(path, ids, lock, SELECTION_SHA)
                np.testing.assert_array_equal(result, np.zeros((365, 3, 168), dtype=np.uint8))
                for invalid in [ids[:-1], ids[::-1], [*ids[:-1], 3000]]:
                    with self.assertRaises(ValueError):
                        read_new_training_labels(path, invalid, lock, SELECTION_SHA)

    def test_invalid_lock_stops_before_label_file_read(self):
        with self.assertRaises(ValueError):
            read_new_training_labels(Path('/absent'), [], Path('/absent-lock'), 'bad')


if __name__ == '__main__':
    unittest.main()
