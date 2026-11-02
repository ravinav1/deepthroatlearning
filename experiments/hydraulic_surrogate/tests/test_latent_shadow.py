"""Leakage, inverse uncertainty, numerical and control tests for the small map."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.special import softmax

from sintef_project.latent_shadow import GRID, PRIOR, TARGET_UNITS, ShadowInputs, fit_mapping, objective_gradient, partition_indices, predict_distribution, read_selected_labels, schedule_mixture, soft_targets, uncertainty_summary, validate_partition_metadata

ROOT=Path(__file__).resolve().parents[1]


class LatentShadowTests(unittest.TestCase):
    def test_selected_label_rows_skip_poisoned_later_values(self):
        columns=[f'result_committed_{u}_t{h}' for u in TARGET_UNITS for h in range(168)]
        frame=pd.DataFrame([[1,'2017-01-01',*([1]*504)],[2,'2021-01-01',*(['POISON']*504)]],columns=['Run No','starttime',*columns])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'labels.csv';frame.to_csv(path,index=False)
            values=read_selected_labels(path,[1])
            np.testing.assert_array_equal(values,np.ones((1,3,168)))
            with self.assertRaises(ValueError):
                read_selected_labels(path,[2])

    def test_feature_builder_uses_metadata_and_initial_storage_only(self):
        original=pd.read_csv
        def guarded(path,*args,**kwargs):
            if str(path).endswith('Unit_commitment_decisions.csv'):
                self.assertEqual(kwargs.get('usecols'),['Run No','starttime'])
            return original(path,*args,**kwargs)
        with patch('pandas.read_csv',side_effect=guarded):
            inputs=ShadowInputs(ROOT,ROOT/'artifacts/lp-full-001')
            first=inputs.grid_case(0)
            start=inputs.cases.iloc[0].starttime
            inputs.volume.loc[start+pd.Timedelta(hours=1)] = -9e12
            second=inputs.grid_case(0)
        for key in first:
            np.testing.assert_array_equal(first[key],second[key])

    def test_uncertain_disconnected_inverse_targets(self):
        np.testing.assert_allclose(soft_targets(np.ones(17)),softmax(PRIOR),atol=1e-15)
        errors=np.full(17,100.);errors[2]=errors[14]=0.
        target=soft_targets(errors)
        self.assertAlmostEqual(target[2],target[14])
        self.assertGreater(target[2],.49)
        self.assertLess(target[8],1e-12)
        low,high=uncertainty_summary(target)
        self.assertEqual(low,-.75);self.assertEqual(high,.75)

    def test_analytic_gradient_matches_finite_differences(self):
        rng=np.random.default_rng(4173)
        design=np.column_stack([np.ones(4),rng.normal(size=(4,2))])
        target=soft_targets(rng.integers(0,10,size=(4,17)))
        theta=rng.normal(scale=.1,size=51)
        _,gradient=objective_gradient(theta,design,target)
        numerical=[]
        for j in range(len(theta)):
            plus,minus=theta.copy(),theta.copy();plus[j]+=1e-6;minus[j]-=1e-6
            numerical.append((objective_gradient(plus,design,target)[0]-objective_gradient(minus,design,target)[0])/2e-6)
        np.testing.assert_allclose(gradient,numerical,rtol=2e-5,atol=1e-8)

    def test_synthetic_real_features_beat_permutation_and_constant(self):
        rng=np.random.default_rng(4173)
        x=rng.normal(size=(240,8))
        target=softmax(PRIOR+8*x[:,0,None]*GRID,axis=1)
        model=fit_mapping(x[:160],target[:160])
        permuted=fit_mapping(x[:160],target[rng.permutation(160)])
        constant=fit_mapping(x[:160],target[:160],constant=True)
        def loss(m):
            return -np.mean(np.sum(target[160:]*np.log(predict_distribution(x[160:],m)),axis=1))
        self.assertLess(loss(model),loss(permuted))
        self.assertLess(loss(model),loss(constant))
        self.assertEqual(np.asarray(constant['weights']).shape,(17,1))
        np.testing.assert_array_equal(predict_distribution(x[160:],constant),predict_distribution(x[160:]*1000,constant))

    def test_case_isolated_grid_marginalization(self):
        rng=np.random.default_rng(4173)
        pi=softmax(rng.normal(size=(4,17)),axis=1)
        paths=rng.integers(0,2,size=(4,17,168),dtype=np.uint8)
        prediction=schedule_mixture(pi,paths)
        for i in range(4):
            np.testing.assert_array_equal(prediction[i:i+1],schedule_mixture(pi[i:i+1],paths[i:i+1]))
        np.testing.assert_array_equal(prediction[::-1],schedule_mixture(pi[::-1],paths[::-1]))
        self.assertTrue(((prediction>=0)&(prediction<=1)).all())

    def test_training_screen_purge(self):
        dates=pd.date_range('2015-01-01','2022-12-31',tz='UTC')
        cases=pd.DataFrame({'starttime':dates})
        train=cases.iloc[partition_indices(cases,'train')].starttime
        screen=cases.iloc[partition_indices(cases,'screen')].starttime
        self.assertEqual(train.max(),pd.Timestamp('2018-12-25',tz='UTC'))
        self.assertEqual(screen.min(),pd.Timestamp('2019-01-01',tz='UTC'))
        self.assertEqual(train.max()+pd.Timedelta(days=7),screen.min())
        self.assertEqual(screen.max(),pd.Timestamp('2020-12-25',tz='UTC'))

    def test_exact_partition_scope_rejects_missing_or_reordered_cases(self):
        dates=pd.date_range('2015-01-01','2022-12-31',tz='UTC')
        metadata=pd.DataFrame({'Run No':np.arange(1,len(dates)+1),'starttime':dates})
        selected=metadata.iloc[partition_indices(metadata,'train')]
        ids,starts=selected['Run No'].tolist(),selected.starttime.astype(str).tolist()
        validate_partition_metadata(ids,starts,metadata,'train')
        with self.assertRaises(ValueError):
            validate_partition_metadata(ids[:100]+ids[101:],starts[:100]+starts[101:],metadata,'train')
        swapped=ids.copy();swapped[100],swapped[101]=swapped[101],swapped[100]
        with self.assertRaises(ValueError):
            validate_partition_metadata(swapped,starts,metadata,'train')


if __name__=='__main__':
    unittest.main()
