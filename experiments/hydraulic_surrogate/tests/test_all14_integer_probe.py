"""Probe contract tests use synthetic arrays and never invoke a competition solve."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scripts.probe_all14_integer_water_budget import gate, identical_arrays, worker


class All14IntegerProbeTests(unittest.TestCase):
    def test_byte_identity_includes_dtype_and_signed_zero(self):
        a={'x':np.asarray([0.],dtype=np.float64)}
        self.assertTrue(identical_arrays(a,{'x':a['x'].copy()}))
        self.assertFalse(identical_arrays(a,{'x':np.asarray([-0.],dtype=np.float64)}))
        self.assertFalse(identical_arrays(a,{'x':np.asarray([0.],dtype=np.float32)}))
        self.assertFalse(identical_arrays(a,{'y':a['x']}))

    def test_unproven_incumbent_or_resource_failure_never_advances(self):
        rows=[{'case_id':i,'status':'complete','proof':{'proven_optimal':True,'mip_status':0},
               'fixed_lp_all_arrays_bitwise':True,'all_slacks_zero':True,
               'objective_subset_checks':True,'peak_rss_bytes':1000}for i in [1,1826]]
        self.assertTrue(all(gate(rows).values()))
        rows[1]['proof']={'proven_optimal':False,'mip_status':1}
        self.assertFalse(all(gate(rows).values()))
        rows[1]['proof']={'proven_optimal':True,'mip_status':0}
        rows[1]['peak_rss_bytes']=2*1024**3
        self.assertFalse(all(gate(rows).values()))
        self.assertFalse(all(gate(rows[:1]).values()))

    def test_native_failure_preserves_incumbent_bounds_and_lp_arrays(self):
        error=RuntimeError('synthetic unproven incumbent fixture')
        error.saved_integer_arrays={'integer_solution':np.asarray([1.,2.])}
        error.saved_integer_proof={'mip_status':1,'proven_optimal':False,'mip_dual_bound_scaled':-2.}
        error.saved_lp_calls=[{'solution':np.asarray([3.,4.]),'status':np.asarray(1)}]
        with tempfile.TemporaryDirectory()as tmp,patch('scripts.probe_all14_integer_water_budget.capture_solve',side_effect=error)as solve:
            row=worker(tmp,1,None,{'price':np.ones(168)},{'objective':3.},{'objective':2.},{'synthetic':True})
            self.assertEqual(solve.call_args.args[2:],(list(range(14)),60.))
            self.assertEqual(row['status'],'failed');self.assertEqual(row['proof'],error.saved_integer_proof)
            folder=Path(tmp)/'case_0001'
            with np.load(folder/'failed_incumbent.npz')as z:np.testing.assert_array_equal(z['integer_solution'],error.saved_integer_arrays['integer_solution'])
            with np.load(folder/'fixed_lp_1.npz')as z:np.testing.assert_array_equal(z['solution'],error.saved_lp_calls[0]['solution'])
            self.assertEqual(json.loads((folder/'proof.json').read_text())['status'],'failed')


if __name__=='__main__':unittest.main()
