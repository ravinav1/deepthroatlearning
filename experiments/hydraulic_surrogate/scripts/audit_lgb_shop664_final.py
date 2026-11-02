"""Independently replay accepted LGB664 final checkpoints or audit their exact CSV."""
from __future__ import annotations
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[key]='2'
import argparse
from datetime import datetime, timezone
import importlib.util
import io
import json
from pathlib import Path
import re
import resource
import shutil
import sys
import time
import traceback
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.train_lgb_shop664_final import check_final_protocol
from scripts.train_lgb_shop664 import sha, environment
from scripts.audit_pooled_pairwise_residual_result import project


def main() -> None:
    """Require actual acceptance and externally pinned complete artifacts, without fitting."""
    ap=argparse.ArgumentParser()
    ap.add_argument('--protocol',type=Path,required=True)
    ap.add_argument('--protocol-sha256',required=True)
    ap.add_argument('--final-run',type=Path,required=True)
    ap.add_argument('--final-manifest-sha256',required=True)
    ap.add_argument('--run-id',required=True)
    ap.add_argument('--export-run',type=Path)
    ap.add_argument('--export-manifest-sha256')
    ap.add_argument('--raw-audit',type=Path)
    ap.add_argument('--raw-audit-sha256')
    a=ap.parse_args()
    assert Path(a.run_id).name==a.run_id and a.run_id not in ['', '.', '..']
    out=ROOT/'artifacts'/a.run_id;out.mkdir(exist_ok=False);start=time.monotonic()
    report={'status':'initializing','timestamp_utc':datetime.now(timezone.utc).isoformat(),
            'command':[sys.executable,*sys.argv],'training':False,
            'mode':'actual_csv' if a.export_run else 'full_checkpoint_replay','environment':environment()}
    try:
        p,v,loader_protocol=check_final_protocol(a.protocol,a.protocol_sha256)
        assert v['version']=='lgb_shop664_feature_ablation_v1'
        assert sha(a.final_run/'manifest.json')==a.final_manifest_sha256
        m=json.loads((a.final_run/'manifest.json').read_text())
        assert m['status']=='complete' and m['mode']=='final' and m['training'] is True
        assert m['protocol_sha256']==a.protocol_sha256 and m['checkpoint_replay_bitwise']
        assert m['completed_units']==14 and m['completed_rounds_per_unit']==350
        assert set(m['models'])==set(v['units']) and m['model_parameters']==p['model_parameters']
        assert m['features']==v['feature_columns'] and m['units']==v['units']
        assert m['selected_weight']==p['selected_weight']==.1
        assert m['selection_lock_sha256']==p['selection_lock_sha256']
        assert m['accepted_validation']==p['accepted_validation']
        assert m['maximum_decoded_target_year']==2022
        for name,expected in m['artifacts'].items():
            assert sha(a.final_run/name)==expected,name
        metadata=pd.read_csv(ROOT/'sintef_project/data/kernel/Unit_commitment_decisions.csv',usecols=['Run No','starttime'])
        mapping=pd.read_csv(ROOT/'sintef_project/prediction_mapping.csv')
        assert len(metadata)==2922 and len(mapping)==725
        assert metadata['Run No'].tolist()==m['train_ids']==p['train_ids']
        ids=mapping['Run No'].tolist()
        assert ids==m['test_ids']==p['test_ids'] and len(set(ids))==725
        expected_sample=np.sort(np.random.default_rng(4173).choice(2922*168,int(2922*168*.4),replace=False))
        sample=np.load(a.final_run/'train_sample_final.npy',allow_pickle=False)
        assert sample.dtype==expected_sample.dtype and sample.tobytes()==expected_sample.tobytes()
        assert len(sample)==m['sampled_training_rows']==196358
        assert sha(a.final_run/'train_sample_final.npy')==m['sample_sha256']==p['sample']['sha256']
        raw=np.load(a.final_run/'test_predictions.npy',allow_pickle=False)
        saved_replay=np.load(a.final_run/'replay_test_predictions.npy',allow_pickle=False)
        assert raw.dtype==saved_replay.dtype==np.float32 and raw.shape==saved_replay.shape==(121800,14)
        assert raw.tobytes()==saved_replay.tobytes() and np.isfinite(raw).all() and 0<=raw.min()<=raw.max()<=1
        native_expected={'boosting':'gbdt','objective':'binary','device_type':'cpu','num_iterations':'350',
            'learning_rate':'0.06','num_leaves':'31','num_threads':'2','seed':'4173','deterministic':'1',
            'force_col_wise':'1','max_depth':'-1','min_data_in_leaf':'100','min_sum_hessian_in_leaf':'0.001',
            'bagging_fraction':'1','bagging_freq':'0','feature_fraction':'0.9','early_stopping_round':'0',
            'lambda_l1':'0','lambda_l2':'1','min_gain_to_split':'0','max_bin':'255','linear_tree':'0'}
        for unit in v['units']:
            checkpoint=a.final_run/f'final_{unit}.txt';record=m['models'][unit]
            assert sha(checkpoint)==record['checkpoint_sha256']
            native=dict(re.findall(r'^\[([^:\]]+): ([^\]]*)\]$',checkpoint.read_text(),re.MULTILINE))
            assert all(native[k]==value for k,value in native_expected.items()),unit
            assert record['trees']==350 and record['features']==664 and record['replay_rows']==121800
            assert record['float64_replay_bitwise'] and record['checkpoint_replay_max_error']==0
        if not a.export_run:
            # First feature-module import must use the original loader's isolated staged source.
            spec=importlib.util.spec_from_file_location('independent_lgb664_final_loader',ROOT/'scripts/cuda_xgb664_loader.py')
            loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
            x,_,train,test,meta,_=loader.build_verified_features(ROOT/v['stage'],2022,loader_protocol)
            assert x.shape==(3647*168,664) and x.columns.tolist()==v['feature_columns']
            assert train['Run No'].tolist()==m['train_ids'] and test['Run No'].tolist()==ids
            assert json.loads(json.dumps(meta['feature_spec']))==m['feature_spec']
            xt=x.iloc[2922*168:].copy();del x
            for index,unit in enumerate(v['units']):
                checkpoint=a.final_run/f'final_{unit}.txt'
                model=lgb.Booster(model_file=str(checkpoint))
                assert model.num_trees()==350 and model.num_feature()==664
                assert model.feature_name()==[c.replace(' ','_') for c in v['feature_columns']]
                prediction=model.predict(xt,num_threads=2)
                assert prediction.astype(np.float32).tobytes()==raw[:,index].tobytes(),unit
                second=lgb.Booster(model_file=str(checkpoint)).predict(xt,num_threads=2)
                assert prediction.dtype==second.dtype==np.float64 and prediction.tobytes()==second.tobytes()
                del model
            report['all14_full_test_checkpoints_replayed']=True
            report['all14_independent_float64_replays_bitwise']=True
        else:
            assert a.raw_audit and a.raw_audit_sha256 and sha(a.raw_audit)==a.raw_audit_sha256
            proof=json.loads(a.raw_audit.read_text())
            assert proof['status']=='passed' and proof['mode']=='full_checkpoint_replay'
            assert proof['protocol_sha256']==a.protocol_sha256 and proof['final_manifest_sha256']==a.final_manifest_sha256
            assert proof['all14_full_test_checkpoints_replayed'] and proof['all14_independent_float64_replays_bitwise']
            assert proof['source_sha256']==sha(Path(__file__))
            assert sha(a.export_run/'manifest.json')==a.export_manifest_sha256
            exported=json.loads((a.export_run/'manifest.json').read_text())
            assert exported['status']=='complete' and exported['protocol_sha256']==a.protocol_sha256
            assert exported['final_manifest_sha256']==a.final_manifest_sha256 and exported['final_independent_audit_sha256']==a.raw_audit_sha256
            for name,expected in exported['artifacts'].items():
                assert sha(a.export_run/name)==expected,name
            anchor_dir=ROOT/p['anchor']['directory']
            assert sha(anchor_dir/'manifest.json')==p['anchor']['manifest_sha256']
            am=json.loads((anchor_dir/'manifest.json').read_text())
            for name,expected in am['artifacts'].items():assert sha(anchor_dir/name)==expected,name
            assert sha(anchor_dir/'test_predictions.npy')==p['anchor']['prediction_sha256']
            anchor=np.load(anchor_dir/'test_predictions.npy',allow_pickle=False)
            assert anchor.dtype==np.float64 and anchor.shape==raw.shape
            assert np.array_equal(project(anchor,v['units']),anchor)
            expected=project(.9*anchor+.1*raw.astype(np.float64),v['units'])
            saved=np.load(a.export_run/'test_predictions.npy',allow_pickle=False)
            assert saved.dtype==np.float64 and saved.shape==expected.shape
            error=float(np.max(np.abs(saved-expected)));assert error<=5e-16
            template=pd.read_csv(ROOT/'sintef_project/sample_submission.csv')
            columns=['Run No']+[f'result_committed_{unit}_t{hour}' for unit in v['units'] for hour in range(168)]
            assert template.columns.tolist()==columns and set(template['Run No'])==set(ids)
            cube=expected.reshape(725,168,14)
            matrix=np.concatenate([cube[:,:,j] for j in range(14)],axis=1)
            frame=pd.DataFrame(matrix,index=pd.Index(ids,name='Run No'),columns=columns[1:]).loc[template['Run No']].reset_index()
            buffer=io.StringIO();frame.to_csv(buffer,index=False,float_format='%.9g')
            assert buffer.getvalue().encode()==(a.export_run/'submission.csv').read_bytes()
            readback=pd.read_csv(a.export_run/'submission.csv')
            assert readback.shape==(725,2353) and readback.columns.tolist()==columns
            assert readback['Run No'].tolist()==template['Run No'].tolist()
            values=readback.iloc[:,1:].to_numpy();assert np.isfinite(values).all() and 0<=values.min()<=values.max()<=1
            readback_error=float(np.max(np.abs(values-frame.iloc[:,1:].to_numpy())));assert readback_error<=5.000001e-10
            assert sha(a.export_run/'submission.csv')==exported['submission_sha256']
            report.update(export_manifest_sha256=a.export_manifest_sha256,raw_audit_sha256=a.raw_audit_sha256,
                          submission_sha256=exported['submission_sha256'],independent_projection_max_error=error,
                          exact_csv_bytes_reconstructed=True,csv_readback_max_error=readback_error)
        report.update(status='passed',protocol_sha256=a.protocol_sha256,final_manifest_sha256=a.final_manifest_sha256,
                      selection_lock_sha256=p['selection_lock_sha256'],selected_weight=.1,
                      sample_sha256=p['sample']['sha256'],sample_rows=196358,test_rows=121800)
    except BaseException:
        report.update(status='failed',error=traceback.format_exc());raise
    finally:
        shutil.copy2(__file__,out/Path(__file__).name)
        report.update(runtime_seconds=time.monotonic()-start,
                      peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024),
                      source_sha256=sha(Path(__file__)),independent_math_helper_sha256=sha(ROOT/'scripts/audit_pooled_pairwise_residual_result.py'))
        (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'status':report['status'],'manifest_sha256':sha(out/'manifest.json')}),flush=True)


if __name__=='__main__':
    main()
