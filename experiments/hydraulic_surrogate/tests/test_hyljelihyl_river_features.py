"""Synthetic physical-unit, downstream-index and strict feature eligibility checks."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from test_lp_hyljelihyl_v2 import fixture
from sintef_project.lp_hyljelihyl_v2 import split_network
from sintef_project.hyljelihyl_river_features import compute_features,append_preserving_prefix,add_river_cost_features


def physical_fixture():
    net,_=fixture();other=split_network(net)
    case={'price':np.zeros(168),'qmin':np.ones(14),'pmax':np.full(14,2.),'energy':np.full((14,4),2.)}
    split={'capacity':np.r_[np.ones(4),307.,np.ones(4),.26]}
    base={'state':np.full((14,168),.3),'flow':np.full((14,168),.3),'power':np.full((14,168),.6),
          'water_shadow':np.repeat(np.arange(9,dtype=float)[:,None],168,axis=1),'status':np.array(0)}
    new={'state':np.full((14,168),.7),'flow':np.full((14,168),.7),'power':np.full((14,168),1.4),
        'water_shadow':np.repeat((10*np.arange(10,dtype=float))[:,None],168,axis=1),'status':np.array(0),
        'storage':np.zeros((10,168)),'bypass':np.zeros((12,168)),
        'minimum_slack':np.zeros((10,168)),'maximum_slack':np.zeros((10,168)),
        'bypass_slack':np.zeros((12,168)),'vest_slack':np.zeros(168)}
    new['storage'][9]=.13;new['storage'][4]=153.5
    new['bypass'][11]=.144;new['bypass'][10]=.288;new['bypass'][4]=.432
    new['river_volume']=new['bypass'][11].copy();new['water_shadow'][9]=15
    return case,split,new,base,net,other


class RiverFeatureTests(unittest.TestCase):
    def test_each_arms_own_downstream_and_exact_flow_units(self):
        args=physical_fixture();values=compute_features(*args)
        self.assertEqual(values.shape,(168,62));self.assertEqual(values.dtype,np.float32)
        np.testing.assert_allclose(values[:,12*4+3],-26.)
        np.testing.assert_allclose(values[:,56:],np.tile([.5,.5,1.,2.,3.,12.5],(168,1)))
        args[2]['river_volume'][:]=.864;args[2]['bypass'][11]=.864
        np.testing.assert_allclose(compute_features(*args)[:,58],6.)

    def test_broadcast_nonfinite_capacity_and_slack_fail_closed(self):
        for name in ['shadow_shape','capacity','nonfinite','slack','river_mismatch','state']:
            with self.subTest(name=name):
                args=physical_fixture()
                if name=='shadow_shape':args[2]['water_shadow']=np.zeros((10,1))
                elif name=='capacity':args[1]['capacity'][4]=307.26
                elif name=='nonfinite':args[0]['price'][0]=np.nan
                elif name=='slack':args[2]['minimum_slack'][0,0]=-1e-14
                elif name=='river_mismatch':args[2]['river_volume'][0]=0
                else:args[2]['state'][0,0]=1.1
                with self.assertRaises(ValueError):compute_features(*args)

    def test_prefix_dtype_and_signed_zero_bytes_preserved(self):
        prefix=pd.DataFrame({f'x{i}':np.array([-0.,1.],dtype=np.float32) for i in range(664)})
        prefix['x1']=np.array([1,2],dtype=np.int64)
        output=append_preserving_prefix(prefix,np.zeros((2,62),dtype=np.float32),prefix.columns.tolist())
        self.assertEqual(output.shape,(2,726))
        for c in prefix:self.assertEqual(output[c].dtype,prefix[c].dtype);self.assertEqual(output[c].to_numpy().tobytes(),prefix[c].to_numpy().tobytes())

    def test_incomplete_cache_never_loads_feature_or_target_files(self):
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp);cache=root/'cache';cache.mkdir()
            for name,data in [('manifest.json',{'status':'building'}),('protocol.json',{}),('audit.json',{})]:
                (cache/name).write_text(json.dumps(data))
            sha=lambda name:hashlib.sha256((cache/name).read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError,'Incomplete'):
                add_river_cost_features(pd.DataFrame(),root,cache,sha('manifest.json'),sha('protocol.json'),cache/'audit.json',sha('audit.json'))


if __name__=='__main__':unittest.main()
