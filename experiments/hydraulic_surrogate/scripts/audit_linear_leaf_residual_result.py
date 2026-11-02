"""Independently replay and score the fixed linear-leaf versus constant-leaf screen."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json,sys,time,platform,shutil,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_pooled_pairwise_residual_result import sha,auc,project,logits

def main()->None:
    """Verify saved artifacts only, without another competition fit or parameter search."""
    ap=argparse.ArgumentParser();ap.add_argument('--manifest-sha256',required=True);ap.add_argument('--run-id',required=True);a=ap.parse_args();assert Path(a.run_id).name==a.run_id and a.run_id not in ['', '.', '..'];out=ROOT/'artifacts'/a.run_id;out.mkdir(exist_ok=False);start=time.monotonic();r={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'status':'initializing','training':False,'command':[sys.executable,*sys.argv],'environment':{'python':sys.version,'platform':platform.platform()}}
    try:
        d=ROOT/'artifacts/linear-leaf-residual-screen-001';assert sha(d/'manifest.json')==a.manifest_sha256;m=json.loads((d/'manifest.json').read_text());assert m['status']=='complete'
        pp=ROOT/'docs/linear_leaf_residual_protocol.json';assert sha(pp)=='ad6197db7076990ff111d6bba68bbb596687eed93d028713f782aaec8da5e4db'==m['protocol_sha256'];p=json.loads(pp.read_text())
        for group in ['source','inputs','data']:
            for n,h in p[group].items():assert sha(ROOT/n)==h,n
        for n,h in m['artifacts'].items():assert sha(d/n)==h,n
        assert m['completed_models']==28 and m['model_parameters']==p['model_parameters'] and m['resource_budget']==p['resource_budget']
        assert m['runtime_seconds']<600 and m['peak_rss_bytes']<4*1024**3
        fit=json.loads((d/'fit_lock.json').read_text());lock=json.loads((d/'prediction_lock.json').read_text());assert sha(d/'fit_lock.json')==m['fit_lock_sha256']==lock['fit_lock_sha256'] and sha(d/'prediction_lock.json')==m['prediction_lock_sha256']
        assert fit['maximum_target_year']==2019 and fit['protocol_sha256']==sha(pp);assert fit['timestamp_utc']<=lock['timestamp_utc']<=m['screen_labels_opened_utc']
        for n,h in fit['artifacts'].items():assert sha(d/n)==h,n
        assert sha(d/'screen_input_predictions.npz')==lock['screen_predictions_sha256'] and sha(d/'screen_residuals.npz')==lock['screen_residuals_sha256']
        oof=ROOT/p['oof_directory'];train_ids=[]
        for year in [2018,2019]:
            with np.load(oof/f'oof_{year}.npz',allow_pickle=False)as z:train_ids+=z['case_ids'].tolist()
        assert train_ids==fit['training_case_ids']==m['training_case_ids'] and len(train_ids)==724
        sample=np.sort(np.random.default_rng(4173).choice(121632,size=int(121632*.4),replace=False));np.testing.assert_array_equal(sample,np.load(d/'residual_training_rows.npy'));assert len(sample)==48652 and sha(d/'residual_training_rows.npy')==p['residual_sample_sha256']==fit['training_rows_sha256']
        with np.load(oof/'oof_2020.npz',allow_pickle=False)as z:base=z['probability'];ids=z['case_ids'];units=z['units'].tolist();dates=pd.to_datetime(z['dates'],utc=True)
        assert ids.tolist()==p['case_ids']['2020'] and base.shape==(360,168,14)
        meta=pd.read_csv(ROOT/'sintef_project/data/kernel/Unit_commitment_decisions.csv',usecols=['Run No','starttime']);selected=meta['Run No'].isin(ids).to_numpy();np.testing.assert_array_equal(meta.loc[selected,'Run No'],ids);cols=[f'result_committed_{u}_t{h}'for u in units for h in range(168)];target=pd.read_csv(ROOT/'sintef_project/data/kernel/Unit_commitment_decisions.csv',skiprows=(np.flatnonzero(~selected)+1).tolist(),usecols=['Run No',*cols],dtype={c:'uint8'for c in cols});np.testing.assert_array_equal(target['Run No'],ids);y=target[cols].to_numpy().reshape(360,14,168).transpose(0,2,1)
        with np.load(d/'screen_input_predictions.npz',allow_pickle=False)as z:raw={k:z[k]for k in ['identity','constant','linear']};np.testing.assert_array_equal(z['case_ids'],ids);assert z['units'].tolist()==units
        with np.load(d/'screen_residuals.npz',allow_pickle=False)as z:residuals={k:z[k]for k in ['constant','linear']};np.testing.assert_array_equal(z['case_ids'],ids)
        assert raw['identity'].dtype==base.dtype and raw['identity'].tobytes()==base.tobytes()
        features={year:np.load(oof/f'features_{year}.npy',mmap_mode='r',allow_pickle=False)for year in [2018,2019,2020]};slopes={}
        for j,unit in enumerate(units):
            choice=p['feature_selection'][unit];ix=np.asarray(choice['indices']);selected=np.concatenate([features[t][:,ix]for t in [2018,2019]])[sample].astype(np.float64);mean=selected.mean(axis=0);scale=selected.std(axis=0);scale=np.where(scale>1e-12,scale,1.)
            with np.load(d/f'normalizer_{unit}.npz')as z:np.testing.assert_array_equal(ix,z['indices']);np.testing.assert_allclose(mean,z['mean'],atol=1e-12,rtol=1e-14);np.testing.assert_allclose(scale,z['scale'],atol=1e-12,rtol=1e-14);mean=z['mean'];scale=z['scale']
            values=((features[2020][:,ix].astype(np.float64)-mean)/scale).astype(np.float32)
            for kind in ['constant','linear']:
                cp=d/f'{kind}_{unit}.txt';model=lgb.Booster(model_file=str(cp));assert model.current_iteration()==40 and model.num_feature()==32 and sha(cp)==m['models'][unit][kind]['checkpoint_sha256'];actual=model.predict(values,raw_score=True,num_threads=2).reshape(360,168);np.testing.assert_array_equal(actual,residuals[kind][:,:,j])
                counts=[]
                for tree in model.dump_model()['tree_info']:
                    stack=[tree['tree_structure']];count=0
                    while stack:
                        node=stack.pop()
                        if 'split_index'in node:stack.extend([node['left_child'],node['right_child']])
                        else:coef=np.asarray(node.get('leaf_coeff',[]));assert np.isfinite(coef).all();count+=int(np.count_nonzero(coef))
                    counts.append(count)
                assert counts==m['models'][unit][kind]['geometry']['nonzero_linear_coefficients_by_tree'] and counts[0]==0
                if kind=='constant':assert sum(counts)==0
                else:slopes[unit]=sum(counts)
                trace=json.loads((d/f'{kind}_{unit}_trace.json').read_text());assert len(trace)==40 and [t['round']for t in trace]==list(range(1,41))
        assert any(slopes.values());results={};first=dates<pd.Timestamp('2020-07-01',tz='UTC')
        for kind,prob in raw.items():
            if kind!='identity':delta=residuals[kind];expected=expit(logits(base)+delta);expected[delta==0]=base[delta==0];np.testing.assert_array_equal(expected,prob)
            expected=project(prob,units)
            with np.load(d/f'screen_{kind}.npz',allow_pickle=False)as z:np.testing.assert_array_equal(z['labels'],y);np.testing.assert_array_equal(z['case_ids'],ids);np.testing.assert_array_equal(z['raw_predictions'],prob);np.testing.assert_allclose(z['predictions'],expected,atol=5e-16,rtol=0);pred=z['predictions']
            result={'projected_auc':auc(y,pred),'raw_auc':auc(y,prob),'halves':{'first_half':auc(y[first],pred[first]),'second_half':auc(y[~first],pred[~first])}}
            for k in ['projected_auc','raw_auc']:assert abs(result[k]-m['results'][kind][k])<1e-12
            for k in result['halves']:assert abs(result['halves'][k]-m['results'][kind]['halves'][k]['projected_auc'])<1e-12
            results[kind]=result
        checks={'whole_screen_beats_both_controls':results['linear']['projected_auc']>max(results[k]['projected_auc']for k in ['identity','constant'])}
        for h in ['first_half','second_half']:checks[h+'_beats_both_controls']=results['linear']['halves'][h]>max(results[k]['halves'][h]for k in ['identity','constant'])
        gain=(results['linear']['projected_auc']-results['identity']['projected_auc'])/(1-results['identity']['projected_auc']);assert checks==m['gate_checks'] and bool(all(checks.values()))==m['gate_passed'] and abs(gain-m['relative_error_reduction'])<1e-10
        r.update(status='passed',protocol_sha256=sha(pp),screen_manifest_sha256=a.manifest_sha256,gate_passed=m['gate_passed'],gate_checks=checks,results=results,relative_error_reduction=gain,slopes=slopes,all_28_checkpoint_replays_bitwise=True,normalizers_reconstructed=True,original_2020_targets_verified=True,helper_source_sha256=sha(ROOT/'scripts/audit_pooled_pairwise_residual_result.py'))
    except BaseException:r.update(status='failed',error=traceback.format_exc());raise
    finally:
        shutil.copy2(__file__,out/'audit.py');shutil.copy2(ROOT/'scripts/audit_pooled_pairwise_residual_result.py',out/'audit_helpers.py');r['runtime_seconds']=time.monotonic()-start;r['artifacts']={p.name:sha(p)for p in out.iterdir()if p.is_file()};(out/'manifest.json').write_text(json.dumps(r,indent=2));print(json.dumps({'status':r['status'],'gate_passed':r.get('gate_passed'),'manifest_sha256':sha(out/'manifest.json'),'runtime_seconds':r['runtime_seconds']}),flush=True)
if __name__=='__main__':main()
