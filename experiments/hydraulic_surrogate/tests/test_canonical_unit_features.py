"""Canonical generator mapping checks with synthetic inputs; no model training."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from sintef_project.canonical_unit_features import (
    canonical_block, feature_specification, write_canonical_rows,
)

ROOT = Path(__file__).resolve().parents[1]


class CanonicalUnitFeaturesTest(unittest.TestCase):
    def setUp(self):
        protocol = json.loads((ROOT / 'docs/lgb_shop664_protocol.json').read_text())
        self.columns, self.units = protocol['feature_columns'], protocol['units']
        values = np.arange(4 * 664, dtype=np.float32).reshape(4, 664)
        self.frame = pd.DataFrame(values, columns=self.columns)
        self.spec = feature_specification(self.columns, self.units)

    def test_semantic_ownership_and_case_hour_then_unit_order(self):
        actual = canonical_block(self.frame, np.array([3, 0, 2]), self.spec)
        self.assertEqual(actual.shape, (42, 115))
        names = self.spec['columns']
        for hour_index, source_row in enumerate([3, 0, 2]):
            for unit_index, unit in enumerate(self.units):
                row = actual[hour_index * 14 + unit_index]
                plant = unit.rsplit('_G', 1)[0]
                self.assertEqual(row[names.index('own_shop_state')], self.frame.loc[source_row, 'shop_state_' + unit])
                self.assertEqual(row[names.index('own_physics_head')], self.frame.loc[source_row, 'physics_head_' + plant])
                self.assertEqual(row[names.index('norm_price')], self.frame.loc[source_row, 'norm_price'])
                self.assertEqual(row[names.index('global_shop_state_Haukeli_G1')], self.frame.loc[source_row, 'shop_state_Haukeli_G1'])
                self.assertEqual(row[names.index('unit_id')], unit_index)
        tokke_g4 = actual[self.units.index('Tokke_G4')]
        self.assertEqual(tokke_g4[names.index('unit_position')], 4)
        self.assertEqual(tokke_g4[names.index('plant_unit_count')], 4)
        self.assertEqual(actual[self.units.index('Hogga_G1'), names.index('plant_unit_count')], 1)

    def test_streamed_blocks_preserve_all_output_bytes_including_signed_zero(self):
        self.frame.loc[0, 'shop_state_Hogga_G1'] = np.float32(-0.0)
        rows = np.array([2, 0, 3])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'canonical.npy'
            record = write_canonical_rows(self.frame, rows, path, self.spec, block_rows=1)
            actual = np.load(path, allow_pickle=False)
            expected = canonical_block(self.frame, rows, self.spec)
            self.assertEqual(actual.tobytes(), expected.tobytes())
            self.assertEqual(len(record['blocks']), 3)
            self.assertTrue(np.signbit(actual[14, self.spec['columns'].index('own_shop_state')]))
            with self.assertRaises(ValueError):
                write_canonical_rows(self.frame, rows, path, self.spec)

    def test_invalid_schema_rows_and_mapping_reject(self):
        for rows in [np.array([0, 0]), np.array([-1]), np.array([4]), np.array([.5])]:
            with self.assertRaises(ValueError):
                canonical_block(self.frame, rows, self.spec)
        with self.assertRaises(ValueError):
            canonical_block(self.frame[self.columns[::-1]], np.array([0]), self.spec)
        changed = copy.deepcopy(self.spec)
        changed['identity_values'][0][0] = 1
        with self.assertRaises(ValueError):
            canonical_block(self.frame, np.array([0]), changed)
        self.frame.loc[0, 'norm_price'] = np.nan
        with self.assertRaises(ValueError):
            canonical_block(self.frame, np.array([0]), self.spec)


if __name__ == '__main__':
    unittest.main()
