"""Deterministic metadata population and distinct replay ordering without solves."""
import unittest
import numpy as np
from scripts.pilot_hyljelihyl_river_persistent import population,orders


class PersistentPilotTests(unittest.TestCase):
    def test_metadata_union_preserves_canonicals_and_endpoints(self):
        canonical=[0,133,265,398,531,664,796,929,1062,1195,1327,1460,1461,1494,1527,1560,
                   1593,1626,1660,1693,1726,1759,1792,1825,1826,1859,1891,1924,1957,1989,
                   2022,2054,2087,2120,2152,2185]
        selected=population(3647,canonical)
        self.assertEqual(selected,sorted(set(canonical+np.rint(np.linspace(0,3646,24)).astype(int).tolist())))
        self.assertTrue(set(canonical)<=set(selected));self.assertEqual(selected[0],0);self.assertEqual(selected[-1],3646)
        self.assertEqual(len(selected),len(set(selected)))

    def test_orders_are_exact_replays_with_fixed_distinct_permutation(self):
        ids=[i+1 for i in population(3647,list(range(36)))];a,b=orders(ids)
        self.assertEqual(a,ids);self.assertEqual(sorted(b),ids);self.assertNotEqual(a,b)
        self.assertEqual(orders(ids),[a,b]);self.assertEqual(b,np.random.default_rng(4173).permutation(ids).tolist())


if __name__=='__main__':unittest.main()
