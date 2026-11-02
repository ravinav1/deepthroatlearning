"""Exact identity, scope, boundedness and stable selection for direct splicing."""
import unittest

import numpy as np

from scripts.latent_shadow_splice_2021 import TARGET_UNITS,WEIGHTS,select_weight,splice
from sintef_project.postprocess import project_vinje

UNITS=['Hogga_G1','Tokke_G1','Tokke_G2','Tokke_G3','Tokke_G4','Lio_G1','Byrte_G1',
       'Vinje_G1','Vinje_G2','Vinje_G3','Haukeli_G1','Songa_G1','Kjela_G1','Vesle Kjela_G1']


class ShadowSpliceTests(unittest.TestCase):
    def test_identity_and_exact_three_unit_scope(self):
        rng=np.random.default_rng(4173)
        anchor=project_vinje(rng.uniform(size=(336,14)),UNITS)
        raw=rng.uniform(size=(336,3))
        index=[UNITS.index(u) for u in TARGET_UNITS]
        other=[j for j,u in enumerate(UNITS) if u not in TARGET_UNITS]
        for weight in WEIGHTS:
            result=splice(anchor,raw,UNITS,weight)
            np.testing.assert_array_equal(result[:,other],anchor[:,other])
            if weight:
                np.testing.assert_array_equal(result[:,index],(1-weight)*anchor[:,index]+weight*raw)
            else:
                np.testing.assert_array_equal(result,anchor)

    def test_full_grid_and_stable_lower_weight_ties(self):
        scores={str(w):.99 for w in WEIGHTS}
        self.assertEqual(select_weight(scores),0.)
        scores['0.025']=scores['0.05']=.995
        self.assertEqual(select_weight(scores),.025)
        del scores['0.1']
        with self.assertRaises(ValueError):
            select_weight(scores)
        scores['0.1']=float('nan')
        with self.assertRaises(ValueError):
            select_weight(scores)

    def test_invalid_probabilities_and_unregistered_weights(self):
        anchor=np.full((168,14),.5)
        raw=np.full((168,3),.5)
        for value in [-.1,1.1,float('nan')]:
            corrupted=raw.copy();corrupted[0,0]=value
            with self.assertRaises(ValueError):
                splice(anchor,corrupted,UNITS,.025)
        with self.assertRaises(ValueError):
            splice(anchor,raw,UNITS,.2)


if __name__=='__main__':
    unittest.main()
