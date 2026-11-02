"""Check independent final CSV alignment and generic isotonic projection."""
import unittest
import numpy as np
from scripts.audit_cuda_xgb664_final_export import expected_csv_values,independent_projection


class ExportAuditTests(unittest.TestCase):
    def test_case_reordering_and_unit_hour_axes(self):
        units=['A','B']
        p=np.arange(2*168*2,dtype=np.float64).reshape(336,2)
        actual=expected_csv_values(p,np.array([7,2]),np.array([2,7]),units)
        np.testing.assert_array_equal(actual[0,:168],p[168:,0])
        np.testing.assert_array_equal(actual[0,168:],p[168:,1])
        np.testing.assert_array_equal(actual[1,:168],p[:168,0])
        with self.assertRaises(ValueError):expected_csv_values(p,np.array([7,7]),np.array([2,7]),units)
        with self.assertRaises(ValueError):expected_csv_values(p,np.array([7,2]),np.array([2,8]),units)

    def test_generic_projection_preserves_other_units_and_order(self):
        units=['Z','Vinje_G1','Vinje_G2','Vinje_G3']
        p=np.array([[.7,.1,.8,.9],[.4,.9,.1,.8],[.3,.6,.4,.2]])
        q=independent_projection(p,units)
        np.testing.assert_array_equal(q[:,0],p[:,0])
        self.assertTrue((np.diff(q[:,1:],axis=1)<=0).all())
        np.testing.assert_allclose(q[:,1:].sum(1),p[:,1:].sum(1),rtol=0,atol=1e-15)
        np.testing.assert_array_equal(q[0,1:],.6)
        np.testing.assert_array_equal(q[1,1:],[.9,.45,.45])


if __name__=='__main__':
    unittest.main()
