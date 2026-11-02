"""Independently replay all LGB664 models and reconstruct the frozen fold gate."""
from __future__ import annotations
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[key] = '2'
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import re
import resource
import shutil
import sys
import time
import traceback
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_pooled_pairwise_residual_result import auc, project


def sha(path: Path) -> str:
    """Hash exact artifacts with bounded memory."""
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main() -> None:
    """Review a completed fold only, after its training process releases memory."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--year', type=int, choices=[2021,2022], required=True)
    ap.add_argument('--candidate-run', type=Path, required=True)
    ap.add_argument('--candidate-manifest-sha256', required=True)
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--selection-lock', type=Path)
    ap.add_argument('--selection-lock-sha256')
    args = ap.parse_args()
    assert Path(args.run_id).name == args.run_id and args.run_id not in ['', '.', '..']
    out = ROOT/'artifacts'/args.run_id
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    report = {'status':'initializing','timestamp_utc':datetime.now(timezone.utc).isoformat(),
              'year':args.year,'training':False,'command':[sys.executable,*sys.argv],
              'environment':{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,
                             'pandas':pd.__version__,'lightgbm':lgb.__version__}}
    try:
        directory = args.candidate_run.resolve()
        assert sha(directory/'manifest.json') == args.candidate_manifest_sha256
        m = json.loads((directory/'manifest.json').read_text())
        assert m['status']=='complete' and m['year']==args.year and m['completed_units']==14
        pp = ROOT/'docs/lgb_shop664_protocol.json'
        assert sha(pp)==m['protocol_sha256']=='882e8c3124ba834322c8f062a7544e9ca3e9512b385e8830db387c38f7c97d08'
        p = json.loads(pp.read_text())
        for group in ['source','inputs','data']:
            for name, expected in p[group].items(): assert sha(ROOT/name)==expected,name
        for name, expected in m['artifacts'].items(): assert sha(directory/name)==expected,name
        assert m['features']==p['feature_columns'] and m['units']==p['units'] and len(m['features'])==664
        assert m['model_parameters']==p['model_parameters'] and m['completed_rounds_per_unit']==350
        assert m['maximum_decoded_target_year']==args.year and m['checkpoint_replay_bitwise']
        seal_path=directory/'fit_and_prediction_seal.json'
        assert sha(seal_path)==m['fit_and_prediction_seal_sha256']
        seal=json.loads(seal_path.read_text())
        assert seal['completed_units']==14 and seal['prediction_sha256']==sha(directory/'raw_predictions_before_selection.npy')
        metadata=pd.read_csv(ROOT/'sintef_project/data/kernel/Unit_commitment_decisions.csv',usecols=['Run No','starttime'])
        dates=pd.to_datetime(metadata.starttime,utc=True)
        rng=np.random.default_rng(4173)
        for year in [2021,2022]:
            train=np.flatnonzero(dates<pd.Timestamp(f'{year}-01-01',tz='UTC')-pd.Timedelta(days=6))
            rows=(train[:,None]*168+np.arange(168)).ravel()
            sampled=np.sort(rng.choice(rows,int(len(rows)*.4),replace=False))
            if year==args.year: break
        sample=np.load(directory/f'train_sample_{args.year}.npy',allow_pickle=False)
        assert sample.dtype==sampled.dtype and sample.tobytes()==sampled.tobytes()
        assert sha(directory/f'train_sample_{args.year}.npy')==p['samples'][str(args.year)]['sha256']==m['sample_sha256']
        assert len(sample)==m['sampled_training_rows'] and metadata.iloc[train]['Run No'].tolist()==m['train_ids']
        selected=(dates>=pd.Timestamp(f'{args.year}-01-01',tz='UTC'))&(dates<pd.Timestamp(f'{args.year+1}-01-01',tz='UTC'))
        case_rows=np.flatnonzero(selected);validation=(case_rows[:,None]*168+np.arange(168)).ravel()
        ids=metadata.loc[selected,'Run No'].to_numpy();units=p['units']
        assert len(ids)==365 and ids.tolist()==m['validation_ids']==p['folds'][str(args.year)]['validation_ids']
        with np.load(directory/f'validation_{args.year}.npz',allow_pickle=False) as saved:
            raw=saved['predictions'].copy();y=saved['labels'].copy()
            assert saved['case_ids'].tolist()==ids.tolist() and saved['units'].tolist()==units
        assert raw.dtype==np.float32 and raw.shape==y.shape==(365*168,14)
        assert np.isfinite(raw).all() and 0<=raw.min()<=raw.max()<=1
        assert raw.tobytes()==np.load(directory/'raw_predictions_before_selection.npy',allow_pickle=False).tobytes()
        columns=[f'result_committed_{unit}_t{hour}'for unit in units for hour in range(168)]
        target=pd.read_csv(ROOT/'sintef_project/data/kernel/Unit_commitment_decisions.csv',
                           skiprows=(np.flatnonzero(~selected.to_numpy())+1).tolist(),
                           usecols=['Run No',*columns],dtype={c:'uint8'for c in columns})
        assert target['Run No'].tolist()==ids.tolist()
        original_y=target[columns].to_numpy().reshape(365,14,168).transpose(0,2,1).reshape(-1,14)
        assert np.array_equal(original_y,y)
        # The original loader rejects foreign feature modules, so import it in
        # this fresh process before any sintef_project feature module is loaded.
        loader_path=ROOT/'scripts/cuda_xgb664_loader.py'
        spec=importlib.util.spec_from_file_location('independent_lgb664_replay_loader',loader_path)
        loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
        loader_protocol=json.loads((ROOT/p['loader_protocol']).read_text())
        x,_,_,_,meta,_=loader.build_verified_features(ROOT/p['stage'],args.year,loader_protocol)
        xv=x.iloc[validation].copy();del x
        assert xv.shape==(61320,664) and xv.columns.tolist()==p['feature_columns']
        assert json.loads(json.dumps(meta['feature_spec']))==m['feature_spec']
        native_expected={'boosting':'gbdt','objective':'binary','device_type':'cpu','num_iterations':'350',
            'learning_rate':'0.06','num_leaves':'31','num_threads':'2','seed':'4173','deterministic':'1',
            'force_col_wise':'1','max_depth':'-1','min_data_in_leaf':'100','min_sum_hessian_in_leaf':'0.001',
            'bagging_fraction':'1','bagging_freq':'0','feature_fraction':'0.9','early_stopping_round':'0',
            'lambda_l1':'0','lambda_l2':'1','min_gain_to_split':'0','max_bin':'255','linear_tree':'0'}
        for index,unit in enumerate(units):
            path=directory/f'fold{args.year}_{unit}.txt'
            assert sha(path)==m['models'][unit]['checkpoint_sha256']==seal['models'][unit]
            native=dict(re.findall(r'^\[([^:\]]+): ([^\]]*)\]$',path.read_text(),re.MULTILINE))
            assert all(native[k]==v for k,v in native_expected.items()),unit
            model=lgb.Booster(model_file=str(path))
            assert model.num_trees()==350 and model.num_feature()==664
            assert model.feature_name()==[c.replace(' ','_')for c in p['feature_columns']]
            prediction=model.predict(xv,num_threads=2)
            assert prediction.astype(np.float32).tobytes()==raw[:,index].tobytes(),unit
            assert m['models'][unit]['float64_checkpoint_replay_bitwise'] and m['models'][unit]['replay_rows']==61320
            del model
        del xv
        values={}
        for kind in ['anchor','control']:
            record=p[kind][str(args.year)]
            with np.load(ROOT/record['prediction'],allow_pickle=False)as saved:
                assert np.array_equal(saved['labels'],y) and saved['case_ids'].tolist()==ids.tolist() and saved['units'].tolist()==units
                values[kind]=saved['predictions'].astype(np.float64)
        if args.year==2021:
            lock_path=directory/'selection_2021.json'
            assert sha(lock_path)==m['selection_lock_sha256']
            lock=json.loads(lock_path.read_text());weights=p['candidate_weights']
            assert seal['timestamp_utc']<=lock['locked_utc']
        else:
            assert args.selection_lock and sha(args.selection_lock)==args.selection_lock_sha256==m['selection_lock_sha256']
            lock_path=args.selection_lock;lock=json.loads(lock_path.read_text());weights=[0.,lock['selected_weight']]
        scores={str(w):auc(y,project((1-w)*values['anchor']+w*raw.astype(np.float64),units))for w in weights}
        assert set(scores)==set(m['scores'])
        for w,value in scores.items():assert abs(value-m['scores'][w])<1e-13
        chosen=max(weights,key=lambda w:(scores[str(w)],-w))if args.year==2021 else lock['selected_weight']
        assert chosen==lock['selected_weight']==m['selected_weight']
        control=auc(y,project((1-chosen)*values['anchor']+chosen*values['control'],units))
        assert abs(control-m['control_micro_auc'])<1e-13
        eligible=chosen>0 and scores[str(chosen)]>max(scores['0.0'],control)
        assert eligible==m['beats_anchor_and_control']
        assert eligible==(lock['eligible_for_2022'] if args.year==2021 else m['accepted'])
        expected=project((1-chosen)*values['anchor']+chosen*raw.astype(np.float64),units)
        with np.load(directory/f'selected_validation_{args.year}.npz',allow_pickle=False)as saved:
            assert np.array_equal(saved['labels'],y) and saved['case_ids'].tolist()==ids.tolist()
            error=float(np.max(np.abs(saved['predictions']-expected)))
            assert error<=5e-16 and saved['predictions'].dtype==np.float64
        report.update(status='passed',protocol_sha256=sha(pp),candidate_manifest_sha256=args.candidate_manifest_sha256,
                      selection_lock_sha256=sha(lock_path),eligible_for_2022=bool(eligible)if args.year==2021 else None,
                      accepted=bool(eligible)if args.year==2022 else None,selected_weight=chosen,scores=scores,
                      control_micro_auc=control,all14_native_checkpoints_replayed=True,
                      original_target_values_verified=True,independent_projection_max_error=error,
                      source_review='Exact saved artifacts, fresh664inputs, all14native350tree replay, rank-based pooledAUC and min-max isotonic projection; no fit or new weight.')
    except BaseException:
        report.update(status='failed',error=traceback.format_exc());raise
    finally:
        shutil.copy2(__file__,out/Path(__file__).name)
        report.update(runtime_seconds=time.monotonic()-started,
                      peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024),
                      source_sha256=sha(Path(__file__)),independent_math_helper_sha256=sha(ROOT/'scripts/audit_pooled_pairwise_residual_result.py'))
        (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'status':report['status'],'manifest_sha256':sha(out/'manifest.json')}),flush=True)


if __name__=='__main__':main()
