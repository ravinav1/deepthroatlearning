"""Verify exact bounded scope and incumbent retention for proof-only completion."""
from pathlib import Path
import tempfile,unittest
from unittest.mock import patch
import numpy as np
from scripts.complete_all14_integer_water_budget import worker,physical_gates,legal_dates
class CompletionTests(unittest.TestCase):
    def test_exact600_cap_and_failure_incumbent_preserved(self):
        error=RuntimeError('synthetic unproven completion')
        error.saved_integer_arrays={'integer_solution':np.array([1.])}
        error.saved_integer_proof={'mip_status':1,'proven_optimal':False,'mip_dual_bound_scaled':-2.}
        error.saved_lp_calls=[]
        with tempfile.TemporaryDirectory()as tmp,patch('scripts.complete_all14_integer_water_budget.capture_solve',side_effect=error)as solve:
            r=worker(tmp,2186,None,{'price':np.ones(168)},{'objective':3.},{'objective':2.},{})
            self.assertEqual(solve.call_args.args[2:],(list(range(14)),600.))
            self.assertEqual(r['status'],'failed');self.assertEqual(r['proof'],error.saved_integer_proof)
            with np.load(Path(tmp)/'case_2186/failed_incumbent.npz')as z:np.testing.assert_array_equal(z['integer_solution'],[1.])
    def test_original_population_and_cutoff_gate_retained(self):
        self.assertTrue(legal_dates(['2020-12-25']));self.assertFalse(legal_dates(['2020-12-26']))
        self.assertFalse(all(physical_gates([],list(range(12))).values()))
if __name__=='__main__':unittest.main()
