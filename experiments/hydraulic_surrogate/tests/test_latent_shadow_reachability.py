"""Synthetic checks for the bounded inverse-shadow diagnostic."""
import itertools
import unittest

import numpy as np
import pandas as pd

from scripts.diagnose_latent_shadow_reachability import GRID, choose_best, optimal_schedule, schedule_value, select_indices


class LatentShadowTest(unittest.TestCase):
    def test_exhaustive_paths_and_free_boundaries(self):
        for rewards in [np.asarray([-4.,2.,-1.,6.,-3.]),np.asarray([-3.,-1.,-2.]),np.asarray([2.,2.,2.])]:
            for start,stop in [(0.,0.),(4.,1.),(1.,5.)]:
                path,value=optimal_schedule(rewards,start,stop)
                exhaustive=max(schedule_value(np.asarray(bits),rewards,start,stop)
                               for bits in itertools.product([0,1],repeat=len(rewards)))
                self.assertAlmostEqual(value,exhaustive)
                self.assertAlmostEqual(schedule_value(path,rewards,start,stop),exhaustive)
        np.testing.assert_array_equal(optimal_schedule(np.zeros(4),3.,2.)[0],np.zeros(4))
        np.testing.assert_array_equal(optimal_schedule(np.ones(4),100.,100.)[0],np.ones(4))

    def test_economically_inactive_shift_and_zero_discharge(self):
        rewards=np.asarray([-4.,2.,-1.,6.,-3.])
        path,value=optimal_schedule(rewards,4.,1.)
        common=np.asarray([2.,-1.,3.,.5,-.25])
        shifted,shifted_value=optimal_schedule(rewards,4.,1.,common)
        np.testing.assert_array_equal(path,shifted)
        self.assertAlmostEqual(shifted_value,value+common.sum())
        for delta in GRID:
            self.assertEqual(.0036*0.*delta*700.*180.,0.)

    def test_displacement_ties_and_metadata_only_dates(self):
        self.assertEqual(GRID[int(choose_best(np.ones(17)))],0.)
        cost=np.ones(17);cost[6]=cost[10]=0
        self.assertEqual(GRID[int(choose_best(cost))],-.25)
        dates=pd.date_range('2015-01-01','2022-12-31',tz='UTC')
        cases=pd.DataFrame({'Run No':np.arange(len(dates)),'starttime':dates})
        index,blocks=select_indices(cases)
        self.assertEqual(len(index),36)
        self.assertEqual(len(set(index)),36)
        self.assertEqual(blocks.count('2019'),12)
        self.assertLessEqual(cases.iloc[index].starttime.max(),pd.Timestamp('2020-12-25',tz='UTC'))


if __name__=='__main__':
    unittest.main()
