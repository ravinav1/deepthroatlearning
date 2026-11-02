"""Synthetic final eligibility, nested composition and submission alignment checks."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from scripts.train_lgb_shop664 import sha
from scripts.train_lgb_shop664_final import verify_confirmation
from scripts.export_lgb_shop664_final import compose,submission_frame
from sintef_project.postprocess import DEFAULT_UNITS,project_vinje


class LgbShop664FinalTests(unittest.TestCase):
    def test_nested_projected_anchor_and_raw_inputs_preserved(self):
        rng=np.random.default_rng(17);raw=rng.random((168,14)).astype(np.float32);anchor=project_vinje(rng.random((168,14)),DEFAULT_UNITS)
        a,b=anchor.copy(),raw.copy();expected=project_vinje(.9*anchor+.1*raw.astype(np.float64),DEFAULT_UNITS)
        actual=compose(anchor,raw,list(DEFAULT_UNITS),.1)
        self.assertEqual(actual.tobytes(),expected.tobytes());self.assertEqual(anchor.tobytes(),a.tobytes());self.assertEqual(raw.tobytes(),b.tobytes())

    def test_case_id_and_hour_unit_alignment(self):
        units=list(DEFAULT_UNITS);columns=['Run No']+[f'result_committed_{u}_t{h}'for u in units for h in range(168)]
        template=pd.DataFrame(np.zeros((2,len(columns))),columns=columns);template['Run No']=[8,4]
        cube=np.arange(2*168*14,dtype=float).reshape(2,168,14)/(2*168*14)
        frame=submission_frame(cube.reshape(-1,14),np.array([4,8]),units,template)
        self.assertEqual(frame['Run No'].tolist(),[8,4]);self.assertEqual(frame.loc[0,'result_committed_Tokke_G2_t117'],cube[1,117,2])
        with self.assertRaises(AssertionError):submission_frame(cube.reshape(-1,14),np.array([4,4]),units,template)

    def test_rejected_confirmation_cannot_seal_final(self):
        with tempfile.TemporaryDirectory()as tmp,patch('scripts.train_lgb_shop664_final.ROOT',Path(tmp)):
            root=Path(tmp);lock={'eligible_for_2022':True,'selected_weight':.1,'protocol_sha256':'v'};lp=root/'lock.json';lp.write_text(json.dumps(lock))
            p={'validation_protocol_sha256':'v','selected_weight':.1,'selection_lock':'lock.json','selection_lock_sha256':sha(lp),'accepted_validation':{}}
            for year in ['2021','2022']:
                m={'status':'complete','year':int(year),'beats_anchor_and_control':True,'protocol_sha256':'v','selected_weight':.1,'selected_micro_auc':.995,'anchor_micro_auc':.994,'control_micro_auc':.993,'completed_units':14,'completed_rounds_per_unit':350,'checkpoint_replay_bitwise':True,'accepted':True,'artifacts':{}}
                mp=root/f'm{year}.json';mp.write_text(json.dumps(m));ap=root/f'a{year}.json';a={'status':'passed','candidate_manifest_sha256':sha(mp),'protocol_sha256':'v','selected_weight':.1,'selection_lock_sha256':sha(lp),'eligible_for_2022':True,'accepted':True};ap.write_text(json.dumps(a))
                p['accepted_validation'][year]={'manifest':mp.name,'manifest_sha256':sha(mp),'audit':ap.name,'audit_sha256':sha(ap)}
            self.assertEqual(verify_confirmation(p)['selected_weight'],.1)
            m=json.loads((root/'m2022.json').read_text());m['accepted']=False;(root/'m2022.json').write_text(json.dumps(m));p['accepted_validation']['2022']['manifest_sha256']=sha(root/'m2022.json')
            a=json.loads((root/'a2022.json').read_text());a['candidate_manifest_sha256']=sha(root/'m2022.json');(root/'a2022.json').write_text(json.dumps(a));p['accepted_validation']['2022']['audit_sha256']=sha(root/'a2022.json')
            with self.assertRaises(AssertionError):verify_confirmation(p)


if __name__=='__main__':unittest.main()
