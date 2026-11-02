"""Chronology, lock, target-cutoff and fixed training recipe checks."""
from pathlib import Path
import json
import tempfile
import unittest
import numpy as np
import pandas as pd
from scripts.train_lgb_shop664 import WEIGHTS,choose,folds_and_samples,sha,validate_lock
from scripts.train_cuda_tcn import read_targets_through_year


class LgbShop664Tests(unittest.TestCase):
    def test_purge_and_repeated_rng_identity(self):
        dates=pd.date_range('2019-01-01','2022-12-31',freq='D',tz='UTC')
        meta=pd.DataFrame({'Run No':np.arange(1,len(dates)+1),'starttime':dates})
        split=folds_and_samples(meta);rng=np.random.default_rng(4173)
        for year in [2021,2022]:
            s=split[str(year)];begin=pd.Timestamp(f'{year}-01-01',tz='UTC')
            self.assertEqual(dates[s['train_cases'][-1]],begin-pd.Timedelta(days=7))
            self.assertLess(dates[s['train_cases'][-1]]+pd.Timedelta(hours=167),begin)
            rows=(s['train_cases'][:,None]*168+np.arange(168)).ravel()
            expected=np.sort(rng.choice(rows,size=int(len(rows)*.4),replace=False))
            self.assertEqual(s['sample'].tobytes(),expected.tobytes())
            self.assertEqual(len(s['validation_cases']),365)

    def test_ties_identity_and_strict_control_gate(self):
        self.assertEqual(choose({str(w):.99 for w in WEIGHTS},.98),(0.,False))
        scores={'0.0':.98,'0.1':.99,'0.25':.99,'0.5':.98}
        self.assertEqual(choose(scores,.985),(.1,True))
        self.assertEqual(choose(scores,.99),(.1,False))
        self.assertEqual(choose({'0.0':.98,'0.25':.99},.985,.25),(.25,True))

    def test_lock_requires_exact_protocol_and_independent_winner(self):
        protocol={'source':{'scripts/train_lgb_shop664.py':'trainer'},'anchor':{'2021':{'prediction_sha256':'anchor'}},'control':{'2021':{'prediction_sha256':'control'}}}
        lock={'version':'lgb_shop664_selection_lock_v1','selection_year':2021,'protocol_sha256':'protocol',
              'trainer_sha256':'trainer','candidate_weights':WEIGHTS,'scores_2021':{'0.0':.98,'0.1':.99,'0.25':.995,'0.5':.99},
              'selected_weight':.25,'selected_2021_micro_auc':.995,'anchor_2021_micro_auc':.98,
              'control_2021_micro_auc':.994,'eligible_for_2022':True,'anchor_prediction_sha256':'anchor','control_prediction_sha256':'control'}
        with tempfile.TemporaryDirectory()as tmp:
            lp=Path(tmp)/'lock.json';ap=Path(tmp)/'audit.json'
            for change in [{},{'selected_weight':.1},{'eligible_for_2022':False},{'protocol_sha256':'wrong'}]:
                lp.write_text(json.dumps(dict(lock,**change)));ap.write_text(json.dumps({'status':'passed','selection_lock_sha256':sha(lp),'eligible_for_2022':True}))
                if change:
                    with self.assertRaises(AssertionError):validate_lock(lp,sha(lp),protocol,'protocol',ap,sha(ap))
                else:self.assertEqual(validate_lock(lp,sha(lp),protocol,'protocol',ap,sha(ap))['selected_weight'],.25)

    def test_original_loader_never_decodes_future_target_values(self):
        with tempfile.TemporaryDirectory()as tmp:
            p=Path(tmp)/'targets.csv'
            p.write_text('Run No,starttime,result_committed_X_t0\n1,2021-01-01,1\n2,2022-01-01,POISON_FUTURE_VALUE\n')
            loaded=read_targets_through_year(p,2021,pd.read_csv)
            self.assertEqual(loaded.loc[0,'result_committed_X_t0'],1)
            self.assertTrue(pd.isna(loaded.loc[1,'result_committed_X_t0']))


if __name__=='__main__':unittest.main()
