"""Test strict retry/eligibility and immutable replay helpers."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
from scripts.build_integer_water_budget_cache import retry_allowed,eligible_attempt,same_arrays,checked_copy,digest


class FullIntegerCacheTests(unittest.TestCase):
    def test_only_first_native_status_one_allows_retry(self):
        for status in [0,1,2,3,4,None]:
            self.assertEqual(retry_allowed({'proof':{'mip_status':status}},60.),status==1)
            self.assertFalse(retry_allowed({'proof':{'mip_status':status}},600.))

    def test_eligibility_requires_every_physical_and_replay_gate(self):
        proof={'mip_status':0,'proven_optimal':True,'fixed_lp_replay_bitwise':True}
        good={'status':'complete','proof':proof,'fixed_lp_all_arrays_bitwise':True,'all_slacks_zero':True}
        self.assertTrue(eligible_attempt(good))
        for key in ['fixed_lp_all_arrays_bitwise','all_slacks_zero']:
            self.assertFalse(eligible_attempt(dict(good,**{key:False})))
        self.assertFalse(eligible_attempt(dict(good,proof=dict(proof,mip_status=1))))

    def test_replay_and_copy_are_byte_sensitive(self):
        same_arrays({'a':np.array([1.])},{'a':np.array([1.])})
        with self.assertRaises(AssertionError):
            same_arrays({'a':np.array([1.])},{'a':np.array([np.nextafter(1.,2.)])})
        with self.assertRaises(AssertionError):
            same_arrays({'a':np.array([0.])},{'a':np.array([-0.])})
        with self.assertRaises(AssertionError):
            same_arrays({'a':np.array([1.],dtype=np.float32)},{'a':np.array([1.],dtype=np.float64)})
        with tempfile.TemporaryDirectory() as temp:
            a,b=Path(temp)/'a',Path(temp)/'b'
            a.write_bytes(b'original')
            checked_copy(a,b,digest(a))
            self.assertEqual(a.read_bytes(),b.read_bytes())
            with self.assertRaises(ValueError):
                checked_copy(a,b,'0'*64)


if __name__=='__main__':
    unittest.main()
