"""Guard the selection lock and export-compatible blend arithmetic."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from lp_joint_compare import blend, choose_weight, write_new_json
from sintef_project.postprocess import DEFAULT_UNITS, project_vinje


class JointComparisonTests(unittest.TestCase):
    def test_exact_tie_retains_smaller_joint_weight(self) -> None:
        scores = {'0.0': .99, '0.25': .991, '0.5': .991, '0.75': .989, '1.0': .98}
        self.assertEqual(choose_weight(scores), .25)

    def test_lock_cannot_be_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'selection.json'
            write_new_json(path, {'weight': .25})
            with self.assertRaises(FileExistsError):
                write_new_json(path, {'weight': .75})
            self.assertEqual(json.loads(path.read_text()), {'weight': .25})

    def test_flattened_mix_matches_generic_export_order(self) -> None:
        rng = np.random.default_rng(84)
        inputs = {name: rng.uniform(size=(168, 14)).astype(np.float64)
                  for name in ['base', 'context', 'joint']}
        before = {name: values.copy() for name, values in inputs.items()}
        for weight in [0., .25, .5, .75, 1.]:
            exported = np.zeros((168, 14), dtype=np.float64)
            for source, part in zip(['base', 'context', 'joint'],
                                    [.25 * (1 - weight), .75 * (1 - weight), weight]):
                exported += part * inputs[source]
            np.testing.assert_array_equal(blend(inputs, list(DEFAULT_UNITS), weight),
                                          project_vinje(exported, DEFAULT_UNITS))
        for name in inputs:
            np.testing.assert_array_equal(inputs[name], before[name])


if __name__ == '__main__':
    unittest.main()
