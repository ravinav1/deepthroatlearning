"""Check physical feature meaning and immutable prefix preservation."""
from types import SimpleNamespace
import unittest
import numpy as np
import pandas as pd
from sintef_project.integer_water_budget_features import compute_features, append_preserving_prefix, COLUMNS
from sintef_project.lp_features import UNIT_ORDER


def fixture():
    network=SimpleNamespace(units=UNIT_ORDER,pools=['up','down'],unit_up=[0]*14,unit_down=[1]*14)
    case={'price':np.ones(168)*3,'qmin':np.ones(14)*2,'pmax':np.ones(14)*10,'energy':np.ones((14,4))*5}
    result={'state':np.zeros((14,168)),'flow':np.ones((14,168))*6,'power':np.ones((14,168))*7,
            'raw_start':np.zeros((14,167)),'raw_stop':np.zeros((14,167)),
            'water_shadow':np.array([np.ones(168)*30,np.ones(168)*5])}
    for key in ['minimum_slack','maximum_slack','bypass_slack','vest_slack']:
        result[key]=np.zeros(168)
    reference={'flow':np.ones((14,168))*2,'power':np.ones((14,168))*2,
               'water_shadow':np.array([np.ones(168)*40,np.ones(168)*5])}
    proof={'proven_optimal':True,'mip_status':0,'fixed_lp_replay_bitwise':True}
    return network,case,result,reference,proof


class IntegerFeatureTests(unittest.TestCase):
    def test_rejects_broadcastable_duals_and_wrong_price_horizon(self):
        network,case,result,reference,proof=fixture()
        result['water_shadow']=np.zeros((2,1))
        with self.assertRaises(ValueError):
            compute_features(case,result,reference,network,proof)
        network,case,result,reference,proof=fixture()
        case['price']=np.ones(1)
        with self.assertRaises(ValueError):
            compute_features(case,result,reference,network,proof)

    def test_transition_alignment_units_and_conditional_margin(self):
        network,case,result,reference,proof=fixture()
        result['state'][0,1:3]=1
        result['raw_start'][0,0]=1
        result['raw_stop'][0,2]=1
        values=compute_features(case,result,reference,network,proof)
        self.assertEqual(values.dtype,np.float32)
        self.assertEqual(values.shape,(168,84))
        np.testing.assert_array_equal(values[:,1],2)
        np.testing.assert_array_equal(values[:,2],.5)
        np.testing.assert_array_equal(values[:,5],.5)
        self.assertEqual(values[0,3],0)
        self.assertEqual(values[1,3],1)
        self.assertEqual(values[3,4],1)
        result['water_shadow']+=123
        reference['water_shadow']+=456
        np.testing.assert_array_equal(values,compute_features(case,result,reference,network,proof))

    def test_rejects_unproven_slack_or_nonbinary_case(self):
        network,case,result,reference,proof=fixture()
        for bad in [dict(proof,mip_status=1),dict(proof,proven_optimal=False),dict(proof,fixed_lp_replay_bitwise=False)]:
            with self.assertRaises(ValueError):
                compute_features(case,result,reference,network,bad)
        result['state'][0,0]=.5
        with self.assertRaises(ValueError):
            compute_features(case,result,reference,network,proof)
        result['state'][0,0]=0
        result['minimum_slack'][0]=1e-12
        with self.assertRaises(ValueError):
            compute_features(case,result,reference,network,proof)

    def test_prefix_values_dtypes_and_positions_unchanged(self):
        columns=[f'x{i}' for i in range(664)]
        prefix=pd.DataFrame(np.arange(1328,dtype=np.float32).reshape(2,664),columns=columns,index=[31,17])
        prefix['x0']=np.asarray([11,12],dtype=np.int16)
        result=append_preserving_prefix(prefix,np.zeros((2,84),dtype=np.float32),columns)
        pd.testing.assert_frame_equal(prefix,result.iloc[:,:664])
        self.assertEqual(result.columns[664:].tolist(),COLUMNS)
        with self.assertRaises(ValueError):
            append_preserving_prefix(prefix,np.zeros((2,84),dtype=np.float32),columns[::-1])


if __name__=='__main__':
    unittest.main()
