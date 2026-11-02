"""Small exact-sample, nested-projection and submission-mapping regressions."""
from __future__ import annotations

import hashlib
import io
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from scripts.export_cuda_xgb_final import compose, submission_frame
from scripts.train_cuda_xgb_final import final_sample
from sintef_project.postprocess import project_vinje


class FinalSampleTests(unittest.TestCase):
    def test_first_full_rng_draw_matches_baseline_artifact(self):
        path=Path(__file__).resolve().parents[1]/'artifacts/lgb-normalized-lp-final-001/train_sample_final.npy'
        expected='4e38d45e87950d4d84849224ddb64fa9b5bf7b21d20a7d9b998223ce5e0fa83d'
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),expected)
        actual=final_sample(2922*168,expected,np)
        np.testing.assert_array_equal(actual,np.load(path,allow_pickle=False))
        self.assertEqual(len(actual),196358)

    def test_wrong_hash_and_wrong_row_scope_fail(self):
        expected=np.sort(np.random.default_rng(4173).choice(100,size=40,replace=False))
        buffer=io.BytesIO();np.save(buffer,expected)
        sha=hashlib.sha256(buffer.getvalue()).hexdigest()
        np.testing.assert_array_equal(final_sample(100,sha,np),expected)
        with self.assertRaisesRegex(ValueError,'sample differs'):
            final_sample(101,sha,np)
        with self.assertRaisesRegex(ValueError,'sample differs'):
            final_sample(100,'0'*64,np)


class FinalExportTests(unittest.TestCase):
    def test_projection_must_remain_nested(self):
        units=['Vinje_G1','Vinje_G2','Vinje_G3']
        prior=np.tile([.3,.7,.4],(168,1))
        raw=np.tile([.8,.2,.4],(168,1))
        anchor=project_vinje(prior,units)
        nested=compose(anchor,raw,units)
        np.testing.assert_allclose(nested[0],[.575,.425,.4],rtol=0,atol=1e-16)
        flat=project_vinje(.75*prior+.25*raw,units)
        self.assertGreater(float(np.max(np.abs(nested-flat))),.07)
        with self.assertRaisesRegex(ValueError,'already projected'):
            compose(prior,raw,units)
        with self.assertRaisesRegex(ValueError,'valid raw'):
            compose(anchor,np.full_like(raw,-.1),units)

    def test_template_controls_ids_and_unit_major_columns(self):
        units=['Vinje_G1','Vinje_G2','Vinje_G3']
        ids=np.array([20,10])
        pred=np.arange(2*168*3,dtype=np.float64).reshape(336,3)/(2*168*3)
        columns=['Run No']+[f'result_committed_{u}_t{h}' for u in units for h in range(168)]
        template=pd.DataFrame(np.zeros((2,len(columns))),columns=columns)
        template['Run No']=[10,20]
        result=submission_frame(pred,ids,units,template)
        self.assertEqual(result['Run No'].tolist(),[10,20])
        self.assertEqual(result.loc[0,'result_committed_Vinje_G2_t7'],pred[168+7,1])
        self.assertEqual(result.loc[1,'result_committed_Vinje_G3_t167'],pred[167,2])
        with self.assertRaisesRegex(ValueError,'schema'):
            submission_frame(pred,np.array([10,10]),units,template)
        with self.assertRaisesRegex(ValueError,'schema'):
            submission_frame(pred,ids,units,template[columns[::-1]])


if __name__=='__main__':
    unittest.main()
