"""Physical-domain, unit-conversion, and whole-case switching checks."""
import itertools
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from sintef_project.nonlinear_pq import NonlinearPQInputs, original_power, physical_on_curve, on_profit
from sintef_project.physics import switching_advantage


def simple_unit():
    """Return a constant-efficiency unit with independently calculable limits."""
    return ({'turb_eff_curves': [{'ref': 100., 'x': [1., 10.], 'y': [100., 100.]}],
             'gen_eff_curve': {'x': [0., 100.], 'y': [100., 100.]}, 'penstock': 1,
             'p_min': 2., 'p_max': 8.}, {'main_loss': [0.], 'penstock_loss': [0.]})


class NonlinearPQTests(unittest.TestCase):
    def test_exact_power_limits_and_original_not_origin_hull(self):
        gen, plant = simple_unit()
        curve = physical_on_curve(gen, plant, 100.)
        self.assertAlmostEqual(curve['qmin_m3s'], 2. / .981, places=10)
        self.assertAlmostEqual(curve['qmax_m3s'], 8. / .981, places=10)
        self.assertAlmostEqual(curve['power_mw'][0], 2., places=10)
        self.assertAlmostEqual(curve['power_mw'][-1], 8., places=10)
        self.assertTrue((curve['q_m3s'] > 0).all())
        gen['turb_eff_curves'][0] = {'ref': 100., 'x': [1., 4., 10.], 'y': [20., 70., 100.]}
        gen['p_min'] = .1
        curve = physical_on_curve(gen, plant, 100.)
        q = np.asarray([1., 4.])
        power = original_power(q, gen, plant, 100.)[0]
        np.testing.assert_allclose(power, [.1962, 2.7468], atol=1e-10)
        self.assertTrue(np.any(curve['q_m3s'] == 4.))

    def test_water_conversion_once_and_negative_profit_preserved(self):
        curve = {'q_m3s': np.asarray([1., 2.]), 'power_mw': np.asarray([1., 2.])}
        np.testing.assert_allclose(on_profit(np.asarray([100.]), np.asarray([1000.]), curve), [192.8])
        np.testing.assert_allclose(on_profit(np.asarray([-100.]), np.asarray([1000.]), curve), [-103.6])
        self.assertLess(on_profit(np.asarray([0.]), np.asarray([1000.]), curve)[0], 0)

    def test_effective_head_bounds_and_impossible_domain(self):
        gen, plant = simple_unit()
        plant['main_loss'] = [.4]
        curve = physical_on_curve(gen, plant, 100.)
        power, head, lower, upper = original_power(curve['q_m3s'], gen, plant, 100.)
        self.assertTrue((head > 0).all())
        self.assertTrue((curve['q_m3s'] >= lower - 1e-8).all())
        self.assertTrue((curve['q_m3s'] <= upper + 1e-8).all())
        self.assertTrue((power >= gen['p_min'] - 1e-8).all())
        self.assertTrue((power <= gen['p_max'] + 1e-8).all())
        gen['p_min'] = 1e9
        with self.assertRaises(ValueError):
            physical_on_curve(gen, plant, 100.)

    def test_switching_matches_bruteforce_with_free_boundaries(self):
        profits = np.asarray([[-4., 9., -2., 3., -7., 2., 6.]])
        expected = np.empty(7)
        objectives = []
        for state in itertools.product([0, 1], repeat=7):
            value = float(np.dot(profits[0], state))
            value -= sum(5. if b > a else 2. if b < a else 0. for a, b in zip(state[:-1], state[1:]))
            objectives.append((state, value))
        for hour in range(7):
            expected[hour] = max(v for s, v in objectives if s[hour]) - max(v for s, v in objectives if not s[hour])
        np.testing.assert_allclose(switching_advantage(profits, 5., 2.)[0], expected, atol=1e-6)
        np.testing.assert_allclose(switching_advantage(profits, 0., 0.), profits, atol=1e-6)

    def test_real_case_reads_only_label_metadata_and_ignores_future_storage(self):
        root = Path(__file__).resolve().parents[1]
        read = pd.read_csv
        target = (root / 'sintef_project/data/kernel/Unit_commitment_decisions.csv').resolve()

        def guarded(path, *args, **kwargs):
            if isinstance(path, (str, Path)) and Path(path).resolve() == target:
                self.assertEqual(kwargs.get('usecols'), ['Run No', 'starttime'])
            return read(path, *args, **kwargs)

        with patch.object(pd, 'read_csv', guarded):
            inputs = NonlinearPQInputs(root, root / 'artifacts/lp-full-001')
            first, raw = inputs.one_case(0)
        self.assertEqual(first.shape, (168, 6))
        self.assertTrue(np.isfinite(first).all())
        future = inputs.volume.index > inputs.cases.iloc[0].starttime
        inputs.volume.loc[future] = -1e6
        second, repeated = inputs.one_case(0)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(raw['raw_on_profit_eur'], repeated['raw_on_profit_eur'])


if __name__ == '__main__':
    unittest.main()
