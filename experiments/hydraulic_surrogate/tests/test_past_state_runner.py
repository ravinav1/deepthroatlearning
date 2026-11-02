"""Verify the ablation feature loader does not decode 2022 target rows."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from scripts import lp_past_state_specialists as runner


class PastStateLabelAccessTests(unittest.TestCase):
    def test_only_metadata_and_pre2022_targets_are_read(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / 'sintef_project/data/kernel/Unit_commitment_decisions.csv'
            target.parent.mkdir(parents=True)
            target.write_text('Run No,starttime,total_calculation_time,result_committed_A_t0\n'
                              '1,2020-01-01 00:00:00+00:00,1,1\n'
                              '2,2021-01-01 00:00:00+00:00,1,0\n'
                              '3,2022-01-01 00:00:00+00:00,1,WITHHELD_NOT_A_LABEL\n')
            original_read = pd.read_csv
            with patch.object(runner, 'ROOT', root), \
                    patch.object(runner, 'build_features', lambda _: pd.read_csv(target)), \
                    patch.object(pd, 'read_csv', wraps=original_read) as reader:
                result = runner.build_without_2022_targets()
            self.assertEqual(result['Run No'].tolist(), [1, 2, 3])
            self.assertEqual(result['result_committed_A_t0'].tolist(), [1, 0, 0])
            self.assertEqual(len(reader.call_args_list), 2)
            self.assertEqual(reader.call_args_list[0].kwargs, {'usecols': ['Run No', 'starttime']})
            self.assertEqual(reader.call_args_list[1].kwargs, {'nrows': 2})


if __name__ == '__main__':
    unittest.main()
