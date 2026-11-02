"""Conditional726 final acceptance, locked recipe and submission alignment checks."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from scripts.train_lgb_shop664 import sha
from scripts.train_lgb_river726_final import verify_confirmation
from scripts import train_lgb_river726 as fold
from sintef_project.postprocess import DEFAULT_UNITS,project_vinje


class LgbRiver726FinalTests(unittest.TestCase):
    def accepted(self,root):
        v={'version':fold.VERSION,'source':{'scripts/train_lgb_river726.py':'trainer'},
           'cache':{'directory':'cache','audit_sha256':'physical'},
           'anchor':{'2021':{'prediction_sha256':'anchor'}},'control':{'2021':{'prediction_sha256':'control'}}}
        vp=root/'validation.json';vp.write_text(json.dumps(v));vh=sha(vp)
        lock={'version':fold.LOCK_VERSION,'selection_year':2021,'protocol_sha256':vh,'trainer_sha256':'trainer',
              'candidate_weights':[0.,.1,.25,.5],'scores_2021':{'0.0':.994,'0.1':.995,'0.25':.996,'0.5':.9955},
              'selected_weight':.25,'selected_2021_micro_auc':.996,'anchor_2021_micro_auc':.994,'control_2021_micro_auc':.993,
              'eligible_for_2022':True,'anchor_prediction_sha256':'anchor','control_prediction_sha256':'control'}
        lp=root/'lock.json';lp.write_text(json.dumps(lock))
        p={'validation_protocol':vp.name,'validation_protocol_sha256':vh,'selected_weight':.25,
           'selection_lock':lp.name,'selection_lock_sha256':sha(lp),'accepted_validation':{},'cache':v['cache']}
        for year in ['2021','2022']:
            m={'status':'complete','year':int(year),'beats_anchor_and_control':True,'protocol_sha256':vh,
               'selected_weight':.25,'selected_micro_auc':.996,'anchor_micro_auc':.994,'control_micro_auc':.993,
               'completed_units':14,'completed_rounds_per_unit':350,'checkpoint_replay_bitwise':True,'accepted':True,'artifacts':{}}
            mp=root/f'm{year}.json';mp.write_text(json.dumps(m))
            a={'status':'passed','candidate_manifest_sha256':sha(mp),'protocol_sha256':vh,'selected_weight':.25,
               'selection_lock_sha256':sha(lp),'eligible_for_2022':True,'accepted':True}
            ap=root/f'a{year}.json';ap.write_text(json.dumps(a))
            p['accepted_validation'][year]={'manifest':mp.name,'manifest_sha256':sha(mp),'audit':ap.name,'audit_sha256':sha(ap)}
        return p

    def test_rejected_confirmation_cannot_finalize(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);p=self.accepted(root)
            with patch('scripts.train_lgb_river726_final.ROOT',root),patch('scripts.train_lgb_shop664_final.ROOT',root),patch.object(fold,'check_cache_contract'):
                self.assertEqual(verify_confirmation(p)['selected_weight'],.25)
                m=json.loads((root/'m2022.json').read_text());m['accepted']=False;(root/'m2022.json').write_text(json.dumps(m))
                record=p['accepted_validation']['2022'];record['manifest_sha256']=sha(root/'m2022.json')
                a=json.loads((root/'a2022.json').read_text());a['candidate_manifest_sha256']=record['manifest_sha256'];(root/'a2022.json').write_text(json.dumps(a));record['audit_sha256']=sha(root/'a2022.json')
                with self.assertRaises(AssertionError):verify_confirmation(p)

    def test_changed_cache_or_weight_cannot_reuse_acceptance(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);p=self.accepted(root)
            with patch('scripts.train_lgb_river726_final.ROOT',root),patch('scripts.train_lgb_shop664_final.ROOT',root),patch.object(fold,'check_cache_contract'):
                for changed in [{**p,'cache':{'directory':'different'}},{**p,'selected_weight':.1}]:
                    with self.assertRaises(AssertionError):verify_confirmation(changed)

    def test_nested_formula_and_exact_case_hour_unit_alignment(self):
        from scripts.export_lgb_river726_final import compose,submission_frame
        rng=np.random.default_rng(4173);units=list(DEFAULT_UNITS)
        raw=rng.random((336,14)).astype(np.float32);anchor=project_vinje(rng.random((336,14)),units)
        a,b=anchor.copy(),raw.copy();expected=project_vinje(.75*anchor+.25*raw.astype(np.float64),units)
        actual=compose(anchor,raw,units,.25)
        self.assertEqual(actual.tobytes(),expected.tobytes());self.assertEqual(anchor.tobytes(),a.tobytes());self.assertEqual(raw.tobytes(),b.tobytes())
        columns=['Run No']+[f'result_committed_{u}_t{h}'for u in units for h in range(168)]
        template=pd.DataFrame(np.zeros((2,len(columns))),columns=columns);template['Run No']=[8,4]
        frame=submission_frame(actual,np.array([4,8]),units,template)
        self.assertEqual(frame['Run No'].tolist(),[8,4]);self.assertEqual(frame.loc[0,'result_committed_Tokke_G2_t117'],actual[168+117,2])
        with self.assertRaises(AssertionError):submission_frame(actual,np.array([4,4]),units,template)

if __name__=='__main__':unittest.main()
