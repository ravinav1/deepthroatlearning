"""Package a distinct private GPU0 capacity recipe without modifying608 inputs."""
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
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.train_cuda_xgb608_capacity import PARAMETERS,WEIGHTS,digest,require
from scripts.train_cuda_xgb608 import PARAMETERS as BASE_PARAMETERS
from scripts.build_cuda_xgb608_package import zip_verified
CODE=['train_cuda_xgb608_capacity.py','cuda_xgb608_loader.py','train_cuda_tcn.py','launch_cuda_xgb.py','launch_cuda_xgb608_capacity_dataset.py']
SOURCE=[*[f'scripts/{n}' for n in CODE],'scripts/select_cuda_xgb608_capacity_blend.py','scripts/verify_cuda_xgb608_capacity_return.py','scripts/build_cuda_xgb608_capacity_package.py','tests/test_cuda_xgb608_capacity.py','docs/cuda_xgb608_capacity.md','sintef_project/postprocess.py']

def main()->None:
    """Pin the exact unchanged input closure and predeclare comparisons before upload."""
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',default='cuda-xgb608-capacity-package-001');args=parser.parse_args()
    out=ROOT/'artifacts'/args.run_id;out.mkdir(exist_ok=False);started=time.monotonic()
    report={'run_id':args.run_id,'timestamp_utc':datetime.now(timezone.utc).isoformat(),'command':shlex.join([sys.executable,*sys.argv]),'status':'initializing','training':False,'upload':False,'kaggle':None,'environment':{'python':sys.version,'platform':platform.platform()}}
    try:
        old_path=ROOT/'docs/cuda_xgb608_protocol.json';old=json.loads(old_path.read_text())
        require(old['model_parameters']==BASE_PARAMETERS,'Baseline parameters changed')
        require({k:v for k,v in PARAMETERS.items() if BASE_PARAMETERS[k]!=v}=={'n_estimators':1200,'max_depth':8,'learning_rate':.03,'device':'cuda:0'},'Unexpected override')
        require(digest(ROOT/'scripts/cuda_xgb608_loader.py')==old['shared_loader_sha256'] and digest(ROOT/'scripts/train_cuda_tcn.py')==old['core_loader_sha256'],'Original loaders changed')
        protocol=json.loads(json.dumps(old));protocol.update(version='cuda_xgb608_capacity_v1',created_utc=report['timestamp_utc'],trainer_sha256=digest(ROOT/'scripts/train_cuda_xgb608_capacity.py'),local_selector_sha256=digest(ROOT/'scripts/select_cuda_xgb608_capacity_blend.py'),return_verifier_sha256=digest(ROOT/'scripts/verify_cuda_xgb608_capacity_return.py'),dataset_launcher_sha256=digest(ROOT/'scripts/launch_cuda_xgb608_capacity_dataset.py'),model_parameters=PARAMETERS,candidate_weights=WEIGHTS,capacity_baseline_protocol_sha256=digest(old_path),selection='2021fixedglobal0/.1/.25/.5 lowerweightties;beataccepted50%GPU608anchorandsameweightraw600d6control;lockbefore2022',caveats='Capacity+learningraterecipe on identical40%rows/features; reusedchronologicalfolds; noLBweightselection',anchor={},control={})
        for year in [2021,2022]:
            for kind,directory in [('anchor',ROOT/f'artifacts/cuda-xgb608-select-{year}-001'),('control',ROOT/f'artifacts/verify-cuda-xgb608-{year}-return-001/download/cuda-xgb608-{year}-001')]:
                manifest=directory/'manifest.json';m=json.loads(manifest.read_text());name=f'selected_validation_{year}.npz' if kind=='anchor' else f'validation_{year}.npz';prediction=directory/name
                require(m['status']=='complete' and digest(prediction)==m['artifacts'][name],'Reference incomplete/changed')
                if kind=='anchor':require(m['selected_weight']==.5 and (year==2021 or m['accepted']),'Wrong anchor')
                protocol[kind][str(year)]={'manifest':manifest.relative_to(ROOT).as_posix(),'manifest_sha256':digest(manifest),'prediction':prediction.relative_to(ROOT).as_posix(),'prediction_sha256':digest(prediction)}
        path=ROOT/'docs/cuda_xgb608_capacity_protocol.json'
        with path.open('x') as stream:json.dump(protocol,stream,indent=2)
        payload=out/'dataset';code=payload/'code';code.mkdir(parents=True)
        for name in CODE:shutil.copy2(ROOT/'scripts'/name,code/name)
        shutil.copy2(path,code/path.name);shutil.copy2(ROOT/'docs/cuda_xgb608_capacity.md',code/'README.md')
        records=[{'path':p.relative_to(payload).as_posix(),'sha256':digest(p),'size_bytes':p.stat().st_size} for p in sorted(payload.rglob('*')) if p.is_file()]
        (payload/'dataset_manifest.json').write_text(json.dumps({'visibility':'PRIVATE','files':records,'protocol_sha256':digest(path),'contains_data_or_labels_or_predictions':False},indent=2))
        archive=out/'cuda-xgb608-capacity-private-dataset.zip';zip_verified(payload,archive)
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
