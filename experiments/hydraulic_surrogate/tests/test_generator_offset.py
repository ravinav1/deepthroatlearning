"""Independent finite differences and rank invariants for bounded offsets."""
import unittest
import numpy as np
from sintef_project.generator_offset import apply_offsets, group_indices, pair_loss_gradient, balanced_loss_gradient, verify_ranks, ridge_loss_gradient_hessian, fit_offsets, pair_hessian, balanced_hessian

UNITS = ['Hogga_G1','Tokke_G1','Tokke_G2','Tokke_G3','Tokke_G4','Lio_G1','Byrte_G1','Vinje_G1','Vinje_G2','Vinje_G3','Songa_G1','Kjela_G1','Haukeli_G1','Vafoss_G1']

class OffsetTests(unittest.TestCase):
    def test_ridge_derivatives_and_fresh_fit_replay(self):
        b=np.linspace(-.4,.4,12); loss,g,h=ridge_loss_gradient_hessian(b); step=1e-6
        numeric=np.column_stack([(ridge_loss_gradient_hessian(b+np.eye(12)[j]*step)[1]-ridge_loss_gradient_hessian(b-np.eye(12)[j]*step)[1])/(2*step)for j in range(12)])
        np.testing.assert_allclose(h,numeric,atol=1e-12)
        self.assertAlmostEqual(loss,.005*np.mean(b*b))
        rng=np.random.default_rng(4173); p=rng.uniform(.01,.99,(100,14)); y=(rng.random(p.shape)<p).astype(np.uint8)
        for kind in ['pairwise','balanced_bce']:
            a,trace,evidence=fit_offsets(p,y,UNITS,kind,pairs=5000)
            c,replay,again=fit_offsets(p,y,UNITS,kind,pairs=5000)
            self.assertEqual(a.tobytes(),c.tobytes());self.assertEqual(trace,replay)
            for name in evidence:self.assertEqual(evidence[name].tobytes(),again[name].tobytes())
    def test_pair_gradient_and_common_shift(self):
        rng=np.random.default_rng(4173); b=rng.normal(size=12)*.1
        margin=rng.normal(size=100); pg=rng.integers(12,size=100); ng=rng.integers(12,size=100)
        fun=lambda x:pair_loss_gradient(x,margin,pg,ng)
        loss,g=fun(b); step=1e-6
        numerical=np.array([(fun(b+np.eye(12)[j]*step)[0]-fun(b-np.eye(12)[j]*step)[0])/(2*step)for j in range(12)])
        np.testing.assert_allclose(g,numerical,atol=1e-9)
        numeric_h=np.column_stack([(fun(b+np.eye(12)[j]*step)[1]-fun(b-np.eye(12)[j]*step)[1])/(2*step)for j in range(12)])
        np.testing.assert_allclose(pair_hessian(b,margin,pg,ng),numeric_h,atol=1e-10)
        self.assertAlmostEqual(loss,fun(b+.23)[0],places=14)
        self.assertAlmostEqual(g.sum(),0,places=14)
    def test_balanced_gradient(self):
        rng=np.random.default_rng(17); b=rng.normal(size=12)*.1; z=rng.normal(size=150)
        y=rng.integers(2,size=150); groups=rng.integers(12,size=150)
        w=np.where(y==1,len(y)/(2*y.sum()),len(y)/(2*(len(y)-y.sum())))
        fun=lambda x:balanced_loss_gradient(x,z,y,groups,w)
        step=1e-6; numeric=np.array([(fun(b+np.eye(12)[j]*step)[0]-fun(b-np.eye(12)[j]*step)[0])/(2*step)for j in range(12)])
        np.testing.assert_allclose(fun(b)[1],numeric,atol=1e-9)
        numeric_h=np.column_stack([(fun(b+np.eye(12)[j]*step)[1]-fun(b-np.eye(12)[j]*step)[1])/(2*step)for j in range(12)])
        np.testing.assert_allclose(balanced_hessian(b,z,groups,w),numeric_h,atol=1e-10)
    def test_seven_day_training_boundary(self):
        from scripts.screen_generator_offset import eligible_dates
        dates=np.array(['2018-12-25','2018-12-26','2018-12-31'])
        np.testing.assert_array_equal(eligible_dates(dates,'2019-01-01'),[True,False,False])
    def test_identity_endpoints_ranks_and_shared_vinje(self):
        groups=group_indices(UNITS); self.assertEqual(len(np.unique(groups)),12)
        p=np.tile(np.array([0,1e-15,1e-8,.1,.1,.8,1-1e-14,1])[:,None],(1,14))
        identity=apply_offsets(p,np.zeros(12),groups); self.assertEqual(p.tobytes(),identity.tobytes())
        q=apply_offsets(p,np.linspace(-.5,.5,12),groups)
        np.testing.assert_array_equal(q[[0,-1]],p[[0,-1]])
        self.assertTrue(verify_ranks(p,q,UNITS)['all_unit_ranks_and_ties_exact'])
        bad=q.copy();bad[3,0]=bad[2,0]
        with self.assertRaises(ValueError):verify_ranks(p,bad,UNITS)
if __name__=='__main__':unittest.main()
