"""Temporal and fixed-population gates for the descriptive all-unit diagnostic."""
import unittest
from scripts.diagnose_all14_integer_water_budget import legal_dates,physical_gates
class All14DiagnosticTests(unittest.TestCase):
    def test_no_target_interval_reaches_2021(self):
        self.assertTrue(legal_dates(['2015-01-01','2019-12-31','2020-12-25']))
        self.assertFalse(legal_dates(['2020-12-26']))
        self.assertFalse(legal_dates(['2021-01-01']))
        self.assertFalse(legal_dates([]))
    def test_partial_population_and_unproven_cases_cannot_score(self):
        ids=list(range(1,13));records=[{'case_id':i,'status':'complete','proof':{'mip_status':0,'proven_optimal':True},
            'fixed_lp_all_arrays_bitwise':True,'all_slacks_zero':True,'objective_subset_checks':True,'peak_rss_bytes':1000}for i in ids]
        self.assertTrue(all(physical_gates(records,ids).values()))
        self.assertFalse(all(physical_gates(records[:-1],ids).values()))
        self.assertFalse(all(physical_gates(list(reversed(records)),ids).values()))
        records[-1]['proof']={'mip_status':1,'proven_optimal':False}
        self.assertFalse(all(physical_gates(records,ids).values()))
if __name__=='__main__':unittest.main()
