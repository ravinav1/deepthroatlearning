"""Package a private GPU1 full-row ablation without changing the600-tree recipe."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import platform
import shlex
import shutil
import sys
import time
import traceback
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.train_cuda_xgb664_fullsample import PARAMETERS,WEIGHTS,digest,require
from scripts.train_cuda_xgb664 import PARAMETERS as BASE_PARAMETERS
from scripts.build_cuda_xgb608_package import zip_verified
CODE=['train_cuda_xgb664_fullsample.py','cuda_xgb664_loader.py','cuda_xgb608_loader.py','train_cuda_tcn.py','launch_cuda_xgb.py','launch_cuda_xgb664_fullsample_dataset.py']
SOURCE=[*[f'scripts/{n}' for n in CODE],'scripts/select_cuda_xgb664_fullsample_blend.py','scripts/verify_cuda_xgb664_fullsample_return.py','scripts/build_cuda_xgb664_fullsample_package.py','tests/test_cuda_xgb664_fullsample.py','docs/cuda_xgb664_fullsample.md','sintef_project/postprocess.py']

def main()->None:
    """Pin the exact unchanged input closure and predeclare comparisons before upload."""
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',default='cuda-xgb664-fullsample-package-001');parser.add_argument('--fullsample608-acceptance-audit',type=Path,required=True);parser.add_argument('--fullsample608-acceptance-audit-sha256',required=True);args=parser.parse_args()
    out=ROOT/'artifacts'/args.run_id;out.mkdir(exist_ok=False);started=time.monotonic()
    report={'run_id':args.run_id,'timestamp_utc':datetime.now(timezone.utc).isoformat(),'command':shlex.join([sys.executable,*sys.argv]),'status':'initializing','training':False,'upload':False,'kaggle':None,'environment':{'python':sys.version,'platform':platform.platform()}}
    try:
        require(digest(args.fullsample608_acceptance_audit)==args.fullsample608_acceptance_audit_sha256,'External prerequisite acceptance audit changed')
        prerequisite=json.loads(args.fullsample608_acceptance_audit.read_text())
        require(prerequisite['status']=='passed' and prerequisite['accepted'] and prerequisite['protocol_sha256']=='d5add5941279d980f156015b4b5ca70da2af4b6d8b99c4841050569b02b7255e','Fullsample608 locked2022 must independently confirm first')
        selected608=ROOT/'artifacts/cuda-xgb608-fullsample-select-2022-001/manifest.json'
        require(digest(selected608)==prerequisite['selection_manifest_sha256'] and json.loads(selected608.read_text())['accepted'],'Prerequisite selector closure changed')
        old_path=ROOT/'docs/cuda_xgb664_protocol.json';require(digest(old_path)=='4e09f0fe59973d7a9c2c5e5792feb30c29b8af04a67c27bafbe1386ae9da8182','Original accepted SHOP protocol changed');old=json.loads(old_path.read_text())
        require(old['model_parameters']==BASE_PARAMETERS,'Baseline parameters changed')
        require(PARAMETERS == BASE_PARAMETERS, 'All original model parameters must remain identical')
        require(digest(ROOT/'scripts/cuda_xgb664_loader.py')==old['shared_loader_sha256'] and digest(ROOT/'scripts/cuda_xgb608_loader.py')==old['prior_loader_sha256'] and digest(ROOT/'scripts/train_cuda_tcn.py')==old['core_loader_sha256'],'Original loaders changed')
        protocol=json.loads(json.dumps(old))
        protocol.update(version='cuda_xgb664_fullsample_v1', created_utc=report['timestamp_utc'],
            prerequisite_fullsample608_acceptance_audit_sha256=args.fullsample608_acceptance_audit_sha256,
            trainer_sha256=digest(ROOT/'scripts/train_cuda_xgb664_fullsample.py'),
            local_selector_sha256=digest(ROOT/'scripts/select_cuda_xgb664_fullsample_blend.py'),
            return_verifier_sha256=digest(ROOT/'scripts/verify_cuda_xgb664_fullsample_return.py'),
            dataset_launcher_sha256=digest(ROOT/'scripts/launch_cuda_xgb664_fullsample_dataset.py'),
            model_parameters=PARAMETERS, candidate_weights=WEIGHTS,
            fullsample_baseline_protocol_sha256=digest(old_path), sample_fraction=1.0,
            baseline_40pct_sample_sha256=old['sample_sha256'], sample_sha256={}, sample_rows={},
            sampling='Every case-hour of the unchanged purged training partition exactly once, in original order; no external row subsampling. Original per-tree subsample=.9 remains fixed.',
            selection='2021 fixed global0/.1/.25/.5, lower-weight ties; beat accepted25%SHOP664 anchor and same-weight rawSHOP664600/depth6/40% control; immutable lock before2022.',
            caveats='Row inclusion fraction is the only modeling change:1.0 versus.4. All664 features,600 trees,depth6,lr.05 and other parameters remain identical. Reused chronological development folds; no leaderboard weight selection.',
            anchor={}, control={})
        metadata=pd.read_csv(ROOT/'sintef_project/data/kernel/Unit_commitment_decisions.csv',usecols=['Run No','starttime'])
        dates=pd.to_datetime(metadata.starttime,utc=True)
        for year in [2021,2022]:
            begin=pd.Timestamp(f'{year}-01-01',tz='UTC')
            cases=np.flatnonzero(dates < begin-pd.Timedelta(days=6))
            require(metadata.iloc[cases]['Run No'].tolist()==old['folds'][str(year)]['train_ids'], 'Training partition changed')
            require(dates.iloc[cases].max()+pd.Timedelta(hours=167)<begin,'Training horizon overlaps validation')
            sample=(cases[:,None]*168+np.arange(168)).ravel()
            sample_path=out/f'train_sample_{year}.npy';np.save(sample_path,sample)
            protocol['sample_sha256'][str(year)]=digest(sample_path)
            protocol['sample_rows'][str(year)]=len(sample)
            protocol['folds'][str(year)]['sampled_training_rows']=len(sample)
            old_sample_path=ROOT/f'artifacts/lgb-normalized-lp-context-joint-001/train_sample_{year}.npy'
            require(digest(old_sample_path)==old['sample_sha256'][str(year)], 'Original40% sample changed')
            require(np.isin(np.load(old_sample_path,allow_pickle=False),sample).all(), 'Original sample is not nested in full rows')
        for year in [2021,2022]:
            for kind,directory in [('anchor',ROOT/f'artifacts/cuda-xgb664-select-{year}-001'),('control',ROOT/f'artifacts/verify-cuda-xgb664-{year}-return-001/download/cuda-xgb664-{year}-001')]:
                manifest=directory/'manifest.json';m=json.loads(manifest.read_text());name=f'selected_validation_{year}.npz' if kind=='anchor' else f'validation_{year}.npz';prediction=directory/name
                require(m['status']=='complete' and digest(prediction)==m['artifacts'][name],'Reference incomplete/changed')
                if kind=='anchor':require(m['selected_weight']==.25 and (year==2021 or m['accepted']),'Wrong anchor')
                protocol[kind][str(year)]={'manifest':manifest.relative_to(ROOT).as_posix(),'manifest_sha256':digest(manifest),'prediction':prediction.relative_to(ROOT).as_posix(),'prediction_sha256':digest(prediction)}
        path=ROOT/'docs/cuda_xgb664_fullsample_protocol.json'
        with path.open('x') as stream:json.dump(protocol,stream,indent=2)
        payload=out/'dataset';code=payload/'code';code.mkdir(parents=True)
        for name in CODE:shutil.copy2(ROOT/'scripts'/name,code/name)
        shutil.copy2(path,code/path.name);shutil.copy2(ROOT/'docs/cuda_xgb664_fullsample.md',code/'README.md')
        records=[{'path':p.relative_to(payload).as_posix(),'sha256':digest(p),'size_bytes':p.stat().st_size} for p in sorted(payload.rglob('*')) if p.is_file()]
        (payload/'dataset_manifest.json').write_text(json.dumps({'visibility':'PRIVATE','files':records,'protocol_sha256':digest(path),'contains_data_or_labels_or_predictions':False},indent=2))
        archive=out/'cuda-xgb664-fullsample-private-dataset.zip';zip_verified(payload,archive)
        report.update(status='complete',protocol_sha256=digest(path),dataset_archive_sha256=digest(archive),dataset_archive_bytes=archive.stat().st_size,source={name:digest(ROOT/name) for name in SOURCE},independent_audit_required=True)
        print(json.dumps({k:report[k] for k in ['status','protocol_sha256','dataset_archive_sha256','dataset_archive_bytes']}),flush=True)
    except BaseException:report.update(status='failed',error=traceback.format_exc());raise
    finally:
        for name in SOURCE:
            if (ROOT/name).is_file():
                target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
        report['runtime_seconds']=time.monotonic()-started;report['artifacts']={p.relative_to(out).as_posix():digest(p) for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'}
        (out/'manifest.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
