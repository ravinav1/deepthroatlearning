"""Conditional final/replay recipe; refuses unaccepted chronological evidence."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import importlib.util,json,os,shlex,shutil,sys,time,traceback
from pathlib import Path
for variable in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[variable]='2'
import numpy as np
import lightgbm as lgb
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.train_lgb_shop664 import sha,environment,peak_rss,memory_free_percent
from scripts import train_lgb_river726 as fold
from scripts.train_lgb_shop664_final import verify_confirmation as original_verify_confirmation


def verify_confirmation(protocol: dict) -> dict:
    """Reuse reviewed acceptance checks, then validate the actual726 lock and cache."""
    lock=original_verify_confirmation(protocol)
    vp=ROOT/protocol['validation_protocol'];assert sha(vp)==protocol['validation_protocol_sha256']
    v=json.loads(vp.read_text());assert v['version']==fold.VERSION
    first=protocol['accepted_validation']['2021']
    checked=fold.validate_lock(ROOT/protocol['selection_lock'],protocol['selection_lock_sha256'],v,
        protocol['validation_protocol_sha256'],ROOT/first['audit'],first['audit_sha256'])
    assert checked==lock and protocol['cache']==v['cache']
    fold.check_cache_contract(protocol['cache'])
    return lock


def check_final_protocol(path: Path, expected: str) -> tuple[dict,dict,dict]:
    """Verify full local source/input closure and original full-fit sample identity."""
    assert sha(path)==expected;p=json.loads(path.read_text())
    assert p['version']=='lgb_river726_accepted_final_v1'and p['environment']==environment()
    for section in ['source','inputs']:
        for name,h in p[section].items():assert sha(ROOT/name)==h,name
    verify_confirmation(p)
    vp=ROOT/p['validation_protocol'];assert sha(vp)==p['validation_protocol_sha256']
    validation=json.loads(vp.read_text());original_fold,loader_protocol,metadata,_=fold.check_closure(validation)
    assert p['cache']==validation['cache']
    assert p['model_parameters']==validation['model_parameters']
    original=json.loads((ROOT/p['original_final_manifest']).read_text());expected_params=dict(original['model_parameters']);expected_params['n_jobs']=2
    assert p['model_parameters']==expected_params and original['arguments']['years']==[]
    sample=np.load(ROOT/p['sample']['path'],allow_pickle=False)
    expected_sample=np.sort(np.random.default_rng(4173).choice(len(metadata)*168,size=int(len(metadata)*168*.4),replace=False))
    assert len(metadata)==2922 and sample.dtype==expected_sample.dtype and sample.shape==expected_sample.shape
    assert sample.tobytes()==expected_sample.tobytes()and len(sample)==196358
    return p,validation,loader_protocol


def main()->None:
    """Fit only an independently approved final protocol, or replay its pinned checkpoint set."""
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--protocol-sha256',required=True)
    ap.add_argument('--run-id',required=True);ap.add_argument('--preflight',action='store_true');ap.add_argument('--independent-audit',type=Path)
    ap.add_argument('--independent-audit-sha256');ap.add_argument('--inference-from',type=Path);ap.add_argument('--inference-manifest-sha256')
    a=ap.parse_args();assert Path(a.run_id).name==a.run_id and a.run_id not in ['','.','..']
    out=ROOT/'artifacts'/a.run_id;out.mkdir(exist_ok=False);start=time.monotonic()
    report={'status':'initializing','timestamp_utc':datetime.now(timezone.utc).isoformat(),'command':shlex.join([sys.executable,*sys.argv]),
            'protocol_sha256':a.protocol_sha256,'mode':'replay'if a.inference_from else'final','training':not a.preflight and not bool(a.inference_from),
            'environment':environment(),'completed_units':0,'models':{},'kaggle_submission':None}
    try:
        p,v,loader_protocol=check_final_protocol(a.protocol,a.protocol_sha256)
        shutil.copy2(a.protocol,out/'protocol.json')
        for name in p['source']:
            dest=out/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
        if a.preflight:report.update(status='passed_preflight_only',labels_decoded=False);return
        assert a.independent_audit and sha(a.independent_audit)==a.independent_audit_sha256
        audit=json.loads(a.independent_audit.read_text());assert audit['status']=='passed'and audit['protocol_sha256']==a.protocol_sha256
        report['independent_audit_sha256']=a.independent_audit_sha256
        origin=None
        if a.inference_from:
            assert sha(a.inference_from/'manifest.json')==a.inference_manifest_sha256
            origin=json.loads((a.inference_from/'manifest.json').read_text())
            assert origin['status']=='complete'and origin['protocol_sha256']==a.protocol_sha256
            assert origin['model_parameters']==p['model_parameters']and origin['checkpoint_replay_bitwise']
            report['inference_manifest_sha256']=a.inference_manifest_sha256
        free=memory_free_percent();report['memory_free_percent_before_build']=free
        if free<v['resources']['minimum_free_percent_before_build']:
            report.update(status='operationally_deferred_resources',training=False);return
        original_fold=json.loads((ROOT/fold.BASE_PROTOCOL).read_text())
        x,labels,train,mapping,meta,baseline=fold.build_features(v,original_fold,loader_protocol,2022)
        assert x.shape==(3647*168,726)and x.columns.tolist()==v['feature_columns']
        assert train['Run No'].tolist()==p['train_ids']and mapping['Run No'].tolist()==p['test_ids']
        sample=np.load(ROOT/p['sample']['path'],allow_pickle=False);units=v['units'];y=np.column_stack([labels[u]for u in units]).astype(np.uint8)
        assert np.isin(y,[0,1]).all()and y.shape==(2922*168,14)
        xt=x.iloc[sample].copy()if origin is None else None;test=x.iloc[2922*168:].copy();del x
        assert test.shape==(725*168,726)
        raw=np.zeros((len(test),14),dtype=np.float32);replay_values=np.zeros_like(raw)
        report.update(features=v['feature_columns'],feature_spec=meta['feature_spec'],units=units,model_parameters=p['model_parameters'],
                      train_ids=p['train_ids'],test_ids=p['test_ids'],sampled_training_rows=len(sample),sample_sha256=p['sample']['sha256'],
                      accepted_validation=p['accepted_validation'],selected_weight=p['selected_weight'],selection_lock_sha256=p['selection_lock_sha256'],maximum_decoded_target_year=2022,cache=p['cache'],feature_count=726)
        shutil.copy2(ROOT/p['sample']['path'],out/'train_sample_final.npy')
        for j,unit in enumerate(units):
            free=memory_free_percent()
            if free<v['resources']['minimum_free_percent_before_unit']:
                report.update(status='operationally_deferred_resources',memory_free_percent_stopped=free);return
            tick=time.monotonic();checkpoint=out/f'final_{unit}.txt'
            if origin is None:
                model=lgb.LGBMClassifier(**p['model_parameters']);model.fit(xt,y[sample,j]);pred=model.predict_proba(test)[:,1]
                model.booster_.save_model(str(checkpoint));del model
            else:
                source=a.inference_from/checkpoint.name;assert sha(source)==origin['artifacts'][checkpoint.name]
                shutil.copy2(source,checkpoint);first=lgb.Booster(model_file=str(checkpoint));pred=first.predict(test,num_threads=2);del first
            second=lgb.Booster(model_file=str(checkpoint));assert second.num_trees()==350 and second.num_feature()==726
            again=second.predict(test,num_threads=2);assert pred.dtype==again.dtype and pred.tobytes()==again.tobytes()
            raw[:,j]=pred;replay_values[:,j]=again;del second
            report['models'][unit]={'checkpoint_sha256':sha(checkpoint),'trees':350,'features':726,'checkpoint_replay_max_error':0.,'float64_replay_bitwise':True,'replay_rows':len(test),'runtime_seconds':time.monotonic()-tick}
            report['completed_units']=j+1;(out/'progress.json').write_text(json.dumps(report,indent=2))
            print(json.dumps({'unit':unit,'completed_units':j+1,'seconds':report['models'][unit]['runtime_seconds'],'peak_rss_bytes':peak_rss()}),flush=True)
        assert raw.tobytes()==replay_values.tobytes()and np.isfinite(raw).all()and 0<=raw.min()<=raw.max()<=1
        np.save(out/'test_predictions.npy',raw);np.save(out/'replay_test_predictions.npy',replay_values)
        report.update(status='complete',completed_rounds_per_unit=350,checkpoint_replay_bitwise=True,raw_probability_range=[float(raw.min()),float(raw.max())])
    except BaseException:report.update(status='failed',error=traceback.format_exc());raise
    finally:
        report.update(runtime_seconds=time.monotonic()-start,peak_rss_bytes=peak_rss())
        report['artifacts']={f.relative_to(out).as_posix():sha(f)for f in out.rglob('*')if f.is_file()and f.name!='manifest.json'}
        (out/'manifest.json').write_text(json.dumps(report,indent=2))
        (out/'experiment_record.md').write_text(f'\n## {a.run_id} — {report["timestamp_utc"]}\n\nStatus {report["status"]}; conditional acceptedLGB726 final/replay, original196358-row sample,14×350trees/2CPUthreads. Command `{report["command"]}`. Runtime {report["runtime_seconds"]:.3f}s, peakRSS {report["peak_rss_bytes"]}. Exact source/input/checkpoint/acceptance/replay evidence in `artifacts/{a.run_id}/manifest.json`. No submission performed by trainer.\n')


if __name__=='__main__':main()
