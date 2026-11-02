"""Guard four-source exporter arithmetic and exact zero-weight anchoring."""
import unittest

import numpy as np

from scripts.lp_context_joint_compare import blend
from sintef_project.postprocess import DEFAULT_UNITS, project_vinje


class ContextJointComparisonTests(unittest.TestCase):
    def test_four_source_export_arithmetic_and_anchor(self) -> None:
        rng = np.random.default_rng(4173)
        sources = ['base', 'context', 'joint', 'combined']
        inputs = {name: rng.uniform(size=(168, 14)) for name in sources}
        anchor = np.zeros((168, 14), dtype=np.float64)
        for name, weight in zip(sources[:3], [.1875, .5625, .25]):
            anchor += weight * inputs[name]
        np.testing.assert_array_equal(blend(inputs, list(DEFAULT_UNITS), 0.),
                                      project_vinje(anchor, DEFAULT_UNITS))
        for new_weight in [.25, .5, .75, 1.]:
            expected = np.zeros_like(anchor)
            for name, weight in zip(sources, [.1875 * (1-new_weight), .5625 * (1-new_weight),
                                              .25 * (1-new_weight), new_weight]):
                expected += weight * inputs[name]
            np.testing.assert_array_equal(blend(inputs, list(DEFAULT_UNITS), new_weight),
                                          project_vinje(expected, DEFAULT_UNITS))


if __name__ == '__main__':
    unittest.main()
