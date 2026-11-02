"""Chronological 2018→2019 then gated 2018/19→2020 bounded offset screens."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import shlex
import shutil
import sys
import time
import traceback

for variable in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[variable]='1'
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sintef_project.generator_offset import apply_offsets,fit_offsets,group_indices,verify_ranks
from sintef_project.postprocess import project_vinje


def digest(path: Path) -> str:
    """Stream immutable input hashes."""
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def eligible_dates(dates: np.ndarray, boundary: str) -> np.ndarray:
    """Require every training schedule to end before the next screen begins."""
    return np.asarray(pd.to_datetime(dates,utc=True)+pd.Timedelta(days=7)<=pd.Timestamp(boundary,tz='UTC'))


def read_labels(ids: np.ndarray, units: list[str], maximum_year: int) -> np.ndarray:
    """Decode only selected case targets after checking the metadata year scope."""
    path=ROOT/'sintef_project/data/kernel/Unit_commitment_decisions.csv'
    meta=pd.read_csv(path,usecols=['Run No','starttime'])
    mask=meta['Run No'].isin(ids).to_numpy()
    np.testing.assert_array_equal(meta.loc[mask,'Run No'].to_numpy(),ids)
    dates=pd.to_datetime(meta.loc[mask,'starttime'],utc=True)
    if not len(dates) or dates.dt.year.max()>maximum_year or maximum_year>2020:
        raise ValueError('Illegal target scope')
    cols=[f'result_committed_{u}_t{h}'for u in units for h in range(168)]
    frame=pd.read_csv(path,skiprows=(np.flatnonzero(~mask)+1).tolist(),usecols=['Run No',*cols],dtype={c:'uint8'for c in cols})
    np.testing.assert_array_equal(frame['Run No'],ids)
    y=frame[cols].to_numpy().reshape(len(ids),14,168).transpose(0,2,1)
    if not np.isin(y,[0,1]).all():raise ValueError('Invalid binary targets')
    return y


def evaluate(p: np.ndarray,y: np.ndarray,dates: np.ndarray,units: list[str],year: int) -> dict:
    """Report exact pooled, unit and predetermined half-year AUCs."""
    first=pd.to_datetime(dates,utc=True)<pd.Timestamp(f'{year}-07-01',tz='UTC')
    return {'auc':float(roc_auc_score(y.ravel(),p.ravel())),
            'halves':{name:float(roc_auc_score(y[mask].ravel(),p[mask].ravel()))for name,mask in [('first',first),('second',~first)]},
            'unit_auc':{u:float(roc_auc_score(y[:,:,j].ravel(),p[:,:,j].ravel()))for j,u in enumerate(units)}}


def main() -> None:
    """Freeze both models and screen predictions before reading screen targets."""
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--year',type=int,choices=[2019,2020],required=True)
    parser.add_argument('--protocol-sha256',required=True)
    parser.add_argument('--preflight-audit',type=Path,required=True)
    parser.add_argument('--preflight-audit-sha256',required=True)
    parser.add_argument('--prior-manifest',type=Path)
    parser.add_argument('--prior-manifest-sha256')
    args=parser.parse_args()
    if Path(args.run_id).name!=args.run_id or args.run_id in ['', '.', '..']:raise ValueError('Invalid run ID')
    out=ROOT/'artifacts'/args.run_id;out.mkdir(exist_ok=False);started=time.monotonic()
    report={'run_id':args.run_id,'status':'initializing','timestamp_utc':datetime.now(timezone.utc).isoformat(),
            'command':shlex.join([sys.executable,*sys.argv]),'screen_year':args.year,'screen_labels_opened_utc':None,
            'kaggle':None,'scope':'Reused early single608LGB OOF mechanism screen, not accepted GPU ensemble',
            'environment':{'python':sys.version,'platform':platform.platform(),'machine':platform.machine(),
                           'packages':{n:importlib.metadata.version(n)for n in ['numpy','scipy','pandas','scikit-learn']}}}
    try:
        pp=ROOT/'docs/generator_offset_protocol.json'
        if digest(pp)!=args.protocol_sha256:raise ValueError('Protocol hash differs')
        p=json.loads(pp.read_text())
        if digest(args.preflight_audit)!=args.preflight_audit_sha256:raise ValueError('Audit hash differs')
        audit=json.loads(args.preflight_audit.read_text())
        if audit['status']!='passed' or audit['protocol_sha256']!=args.protocol_sha256:raise ValueError('Exact independent source approval required')
        for category in ['source','data','inputs']:
            for path,h in p[category].items():
                if digest(ROOT/path)!=h:raise ValueError('Changed input '+path)
        if args.year==2020:
            if args.prior_manifest is None or digest(args.prior_manifest)!=args.prior_manifest_sha256:raise ValueError('Externally pinned prior result required')
            prior=json.loads(args.prior_manifest.read_text())
            if prior['status']!='complete' or prior['screen_year']!=2019 or not prior['accepted'] or prior['protocol_sha256']!=args.protocol_sha256:
                raise ValueError('2019 gate did not pass')
            for path,h in prior['artifacts'].items():
                if digest(args.prior_manifest.parent/path)!=h:raise ValueError('Prior result artifact changed')
            report['prior_manifest_sha256']=args.prior_manifest_sha256
        elif args.prior_manifest is not None:raise ValueError('Initial stage takes no prior result')
        report.update(protocol_sha256=args.protocol_sha256,preflight_audit_sha256=args.preflight_audit_sha256,
                      source=p['source'],data=p['data'],inputs=p['inputs'],parameters=p['parameters'])
        for path in p['source']:
            target=out/'source'/path;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/path,target)
        shutil.copy2(pp,out/'protocol.json')
        units=p['units']; group=group_indices(units); arrays={}
        for year in range(2018,args.year+1):
            with np.load(ROOT/p['oof_directory']/f'oof_{year}.npz',allow_pickle=False)as saved:
                arrays[year]={k:saved[k]for k in ['probability','dates','case_ids','units']}
            if arrays[year]['units'].tolist()!=units:raise ValueError('Unit order changed')
            np.testing.assert_array_equal(arrays[year]['case_ids'],p['oof_case_ids'][str(year)])
        train_p=[];train_ids=[]
        for year in range(2018,args.year):
            a=arrays[year];mask=eligible_dates(a['dates'],f'{args.year}-01-01')
            train_ids.append(a['case_ids'][mask]);train_p.append(a['probability'][mask])
        ids=np.concatenate(train_ids)
        np.testing.assert_array_equal(ids,p['training_ids'][str(args.year)])
        base=np.concatenate(train_p);base=project_vinje(base.reshape(-1,14),units).reshape(base.shape)
        target=read_labels(ids,units,args.year-1)
        report['training_ids']=ids.tolist();report['training_target_maximum_year']=args.year-1
        screen=arrays[args.year]; identity=project_vinje(screen['probability'].reshape(-1,14),units).reshape(screen['probability'].shape)
        predictions={'identity':identity};fits={};invariants={}
        for kind in ['pairwise','balanced_bce']:
            b,trace,evidence=fit_offsets(base,target,units,kind)
            replay,replay_trace,replay_evidence=fit_offsets(base,target,units,kind)
            if b.dtype!=replay.dtype or b.tobytes()!=replay.tobytes() or trace!=replay_trace:raise ValueError('Fit replay differs')
            for key in evidence:
                if evidence[key].tobytes()!=replay_evidence[key].tobytes():raise ValueError('Pair replay differs')
            np.save(out/f'{kind}_offsets.npy',b)
            if evidence:np.savez_compressed(out/'training_pairs.npz',**evidence)
            predictions[kind]=apply_offsets(identity,b,group)
            invariants[kind]=verify_ranks(identity,predictions[kind],units)
            fits[kind]=trace
            print('OFFSET_FIT_COMPLETE',kind,trace,flush=True)
        np.savez_compressed(out/'screen_predictions.npz',**predictions,case_ids=screen['case_ids'],units=np.asarray(units),dates=screen['dates'])
        lock={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':args.protocol_sha256,'training_ids':ids.tolist(),
              'fits':fits,'invariants':invariants,'fit_replay_bitwise':True,
              'frozen_artifacts':{x.name:digest(x)for x in out.glob('*')if x.is_file()}}
        (out/'fit_lock.json').write_text(json.dumps(lock,indent=2)+'\n')
        report['fit_lock_sha256']=digest(out/'fit_lock.json');report['screen_labels_opened_utc']=datetime.now(timezone.utc).isoformat()
        labels=read_labels(screen['case_ids'],units,args.year)
        scores={k:evaluate(v,labels,screen['dates'],units,args.year)for k,v in predictions.items()}
        ranks_equal=all(scores[k]['unit_auc']==scores['identity']['unit_auc']for k in fits)
        gates={'convergence_and_replay':True,'ranks_and_vinje':ranks_equal,
               'full_beats_identity':scores['pairwise']['auc']>scores['identity']['auc'],
               'full_beats_balanced_bce':scores['pairwise']['auc']>scores['balanced_bce']['auc'],
               **{f'{h}_beats_identity':scores['pairwise']['halves'][h]>scores['identity']['halves'][h]for h in ['first','second']}}
        report.update(status='complete',scores=scores,fits=fits,invariants=invariants,gates=gates,accepted=all(gates.values()))
        print(json.dumps({'status':report['status'],'scores':{k:v['auc']for k,v in scores.items()},'gates':gates,'accepted':report['accepted']}),flush=True)
    except BaseException:
        report.update(status='failed',error=traceback.format_exc());raise
    finally:
        report['runtime_seconds']=time.monotonic()-started;report['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report['artifacts']={x.relative_to(out).as_posix():digest(x)for x in out.rglob('*')if x.is_file()and x.name!='manifest.json'}
        (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
        (out/'experiment_record.md').write_text(f"\n## {args.run_id} — {report['timestamp_utc']}\n\nStatus: {report['status']}. Fixed bounded12-group pairwise offsets with shared Vinje bias and lambda0.01 shrinkage versus identical balanced-BCE parameter family. Command: `{report['command']}`. Exact source, inputs, rows, environment, parameters, scores, convergence, replay and artifact checksums: `artifacts/{args.run_id}/manifest.json`. Runtime {report['runtime_seconds']:.3f} seconds. Acceptance: {report.get('accepted')}. No public submission or GPU ensemble claim.\n")
if __name__=='__main__':main()
