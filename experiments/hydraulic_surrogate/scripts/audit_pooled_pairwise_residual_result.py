"""Independently audit frozen residual-screen arrays, scope, locks and exact checkpoint replay."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib,json,platform,shlex,shutil,sys,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata
import lightgbm as lgb
ROOT=Path(__file__).resolve().parents[1]

def sha(p:Path)->str:
    """Hash a file in bounded memory."""
    h=hashlib.sha256()
    with p.open('rb')as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def auc(y:np.ndarray,p:np.ndarray)->float:
    """Compute pooled AUC independently through average ranks and positive rank sum."""
    y=y.ravel();p=p.ravel();n=int(y.sum());m=len(y)-n
    assert n>0 and m>0 and np.isfinite(p).all()
    ranks=rankdata(p,method='average')
    return float((ranks[y==1].sum()-n*(n+1)/2)/(n*m))

def project(value:np.ndarray,units:list[str])->np.ndarray:
    """Use the generic min-max characterization of decreasing isotonic projection."""
    shape=value.shape;flat=value.reshape(-1,14).astype(np.float64);result=flat.copy();ix=[units.index(f'Vinje_G{i}')for i in [1,2,3]];x=flat[:,ix]
    for i in range(3):
        result[:,ix[i]]=np.min(np.stack([np.max(np.stack([x[:,a:b+1].mean(axis=1)for b in range(i,3)]),axis=0)for a in range(i+1)]),axis=0)
    return result.reshape(shape)

def logits(p:np.ndarray)->np.ndarray:
    """Match the fixed endpoint-only safeguard with otherwise exact float64 logits."""
    p=p.astype(np.float64);p=np.where(p==0,1e-12,np.where(p==1,1-1e-12,p));return np.log(p)-np.log1p(-p)

def main()->None:
    """Check a completed result only, without fitting or selecting new parameters."""
    ap=argparse.ArgumentParser();ap.add_argument('--manifest-sha256',required=True);ap.add_argument('--run-id',required=True);args=ap.parse_args();assert Path(args.run_id).name==args.run_id and args.run_id not in ['', '.', '..']
    out=ROOT/'artifacts'/args.run_id;out.mkdir(exist_ok=False);start=time.monotonic();r={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'status':'initializing','training':False,'command':shlex.join([sys.executable,*sys.argv]),'environment':{'python':sys.version,'platform':platform.platform()}}
    try:
        d=ROOT/'artifacts/pooled-pairwise-residual-screen-001';assert sha(d/'manifest.json')==args.manifest_sha256;m=json.loads((d/'manifest.json').read_text());assert m['status']=='complete'
        pp=ROOT/'docs/pooled_pairwise_residual_protocol_v2.json';assert sha(pp)=='dc01041ec5c239d683c9f7f2d9d5efe36ba19b6fa47e9311bf695442c85405a1'==m['protocol_sha256'];p=json.loads(pp.read_text())
        for group in ['source','inputs','data']:
            for n,h in p[group].items():assert sha(ROOT/n)==h,n
        for n,h in m['artifacts'].items():assert sha(d/n)==h,n
        assert m['model_parameters']==p['model_parameters'] and m['objective_parameters']==p['objective_parameters']
        oof=ROOT/p['oof_directory'];assert sha(oof/'manifest.json')==m['oof_manifest_sha256']=='1016522707ec517f9e86dbabfe75686cf9b8d963aade721483d2f49bf0586146'
        fit=json.loads((d/'fit_lock.json').read_text());lock=json.loads((d/'prediction_lock.json').read_text());assert sha(d/'fit_lock.json')==m['fit_lock_sha256']==lock['fit_lock_sha256'] and sha(d/'prediction_lock.json')==m['prediction_lock_sha256']
        assert fit['maximum_target_year']==2019 and fit['protocol_sha256']==sha(pp) and fit['oof_manifest_sha256']==m['oof_manifest_sha256']
        assert fit['timestamp_utc']<=lock['timestamp_utc']<=m['screen_labels_opened_utc']
        assert lock['screen_predictions_sha256']==sha(d/'screen_input_predictions.npz') and lock['screen_residuals_sha256']==sha(d/'screen_residuals.npz')
        assert lock['all_three_checkpoint_replays_exact'] and lock['identity_bytes_exact']
        train_labels=[];train_ids=[]
        for year in [2018,2019]:
            with np.load(oof/f'oof_{year}.npz',allow_pickle=False)as z:train_labels.append(z['labels'].ravel());train_ids+=z['case_ids'].tolist()
        y_train=np.concatenate(train_labels);assert fit['training_ids']==train_ids==m['training_ids'] and len(train_ids)==724 and len(y_train)==1702848
        positive=np.flatnonzero(y_train==1);negative=np.flatnonzero(y_train==0)
        assert m['class_counts']=={'positive':len(positive),'negative':len(negative)}
        for kind in ['pairwise','bce','balanced_bce']:
            assert sha(d/f'{kind}.txt')==fit['model_sha256'][kind] and sha(d/f'{kind}_trace.json')==fit['trace_sha256'][kind]
            tr=json.loads((d/f'{kind}_trace.json').read_text());assert len(tr)==80 and [x['round']for x in tr]==list(range(1,81))
            if kind=='pairwise':
                for j,t in enumerate(tr):
                    rng=np.random.default_rng(np.random.SeedSequence([4173,j]));pos=positive[rng.integers(0,len(positive),size=1000000)];neg=negative[rng.integers(0,len(negative),size=1000000)]
                    h=hashlib.sha256();h.update(pos.tobytes());h.update(neg.tobytes());assert h.hexdigest()==t['pair_sha256']
                    assert t['pairs']==1000000 and t['scale']==len(y_train)/2000000
        with np.load(oof/'oof_2020.npz',allow_pickle=False)as z:base=z['probability'];ids=z['case_ids'];units=z['units'].tolist();dates=pd.to_datetime(z['dates'],utc=True)
        assert ids.tolist()==m['screen_ids']==p['case_ids']['2020'] and base.shape==(360,168,14) and units==p['units']
        metadata=pd.read_csv(ROOT/'sintef_project/data/kernel/Unit_commitment_decisions.csv',usecols=['Run No','starttime']);selected=metadata['Run No'].isin(ids).to_numpy();np.testing.assert_array_equal(metadata.loc[selected,'Run No'],ids)
        assert dates.min()==pd.Timestamp('2020-01-01',tz='UTC') and dates.max()==pd.Timestamp('2020-12-25',tz='UTC')
        columns=[f'result_committed_{u}_t{h}'for u in units for h in range(168)]
        target=pd.read_csv(ROOT/'sintef_project/data/kernel/Unit_commitment_decisions.csv',skiprows=(np.flatnonzero(~selected)+1).tolist(),usecols=['Run No',*columns],dtype={c:'uint8'for c in columns});np.testing.assert_array_equal(target['Run No'],ids);y=target[columns].to_numpy().reshape(360,14,168).transpose(0,2,1)
        with np.load(d/'screen_input_predictions.npz',allow_pickle=False)as z:raw={k:z[k]for k in ['identity','pairwise','bce','balanced_bce']};np.testing.assert_array_equal(z['case_ids'],ids);assert z['units'].tolist()==units
        with np.load(d/'screen_residuals.npz',allow_pickle=False)as z:residuals={k:z[k]for k in ['pairwise','bce','balanced_bce']};np.testing.assert_array_equal(z['case_ids'],ids)
        assert raw['identity'].dtype==base.dtype and raw['identity'].tobytes()==base.tobytes()
        results={};half=dates<pd.Timestamp('2020-07-01',tz='UTC')
        for kind,value in raw.items():
            if kind!='identity':
                delta=residuals[kind];expected=expit(logits(base)+delta);expected[delta==0]=base[delta==0];np.testing.assert_array_equal(expected,value)
            expected=project(value,units)
            with np.load(d/f'screen_{kind}.npz',allow_pickle=False)as z:
                np.testing.assert_array_equal(z['labels'],y);np.testing.assert_array_equal(z['case_ids'],ids);np.testing.assert_array_equal(z['raw_predictions'],value);np.testing.assert_allclose(z['predictions'],expected,atol=5e-16,rtol=0)
                # AUC uses the saved fixed projection bytes, avoiding alternate tie rounding in generic proof arithmetic.
                pred=z['predictions']
            scores={'projected_auc':auc(y,pred),'raw_auc':auc(y,value),'halves':{'first_half':auc(y[half],pred[half]),'second_half':auc(y[~half],pred[~half])}}
            for key in ['projected_auc','raw_auc']:assert abs(scores[key]-m['results'][kind][key])<1e-12
            for key in scores['halves']:assert abs(scores['halves'][key]-m['results'][kind]['halves'][key]['projected_auc'])<1e-12
            results[kind]=scores
        features=np.load(oof/'features_2020.npy',mmap_mode='r',allow_pickle=False)
        for kind in residuals:
            model=lgb.Booster(model_file=str(d/f'{kind}.txt'));assert model.current_iteration()==80 and model.num_feature()==623
            for a in range(0,360,16):
                b=min(a+16,360);f=features[a*168:b*168];p0=base[a:b];x=np.empty((p0.size,623),np.float32);x[:,608:622]=0
                for j in range(14):x[j::14,:608]=f;x[j::14,608+j]=1
                x[:,-1]=logits(p0).ravel();actual=model.predict(x,raw_score=True,num_threads=2).reshape(b-a,168,14);np.testing.assert_array_equal(actual,residuals[kind][a:b])
        checks={'whole_year_beats_all_controls':all(results['pairwise']['projected_auc']>results[k]['projected_auc']for k in ['identity','bce','balanced_bce'])}
        for key in ['first_half','second_half']:checks[key+'_beats_identity']=results['pairwise']['halves'][key]>results['identity']['halves'][key]
        assert checks==m['gate_checks'] and bool(all(checks.values()))==m['gate_passed']
        gain=(results['pairwise']['projected_auc']-results['identity']['projected_auc'])/(1-results['identity']['projected_auc']);assert abs(gain-m['relative_error_reduction'])<1e-10 and (gain>=.05)==m['substantial_gain']
        r.update(status='passed',screen_manifest_sha256=args.manifest_sha256,protocol_sha256=sha(pp),gate_passed=m['gate_passed'],gate_checks=checks,results=results,relative_error_reduction=gain,substantial_gain=m['substantial_gain'],all_three_checkpoint_replays_bitwise=True,all_80_pair_draws_reconstructed=True,selected_2020_labels_verified=True,source_and_artifact_closure=True)
    except BaseException:r.update(status='failed',error=traceback.format_exc());raise
    finally:
        shutil.copy2(__file__,out/'audit.py');r['runtime_seconds']=time.monotonic()-start;r['artifacts']={p.name:sha(p)for p in out.iterdir()if p.is_file()};(out/'manifest.json').write_text(json.dumps(r,indent=2));print(json.dumps({'status':r['status'],'gate_passed':r.get('gate_passed'),'manifest_sha256':sha(out/'manifest.json'),'runtime_seconds':r['runtime_seconds']}),flush=True)
if __name__=='__main__':main()
