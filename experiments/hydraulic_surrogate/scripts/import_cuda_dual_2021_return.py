"""Import one externally hashed2021 GPU result and run its frozen local selector."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
RECIPES={
    'shop664':{'protocol':'cuda_xgb664_protocol.json','protocol_sha256':'4e09f0fe59973d7a9c2c5e5792feb30c29b8af04a67c27bafbe1386ae9da8182','verifier':'verify_cuda_xgb664_return.py','selector':'select_cuda_xgb664_blend.py','trainer':'train_cuda_xgb664.py','gpu_run':'cuda-xgb664-2021-001','smoke_run':'cuda-xgb664-smoke-001'},
    'capacity608':{'protocol':'cuda_xgb608_capacity_protocol.json','protocol_sha256':'e9172028088440d8a46a5966946c63cb723606ea916833d3c81e5f67ccfae026','verifier':'verify_cuda_xgb608_capacity_return.py','selector':'select_cuda_xgb608_capacity_blend.py','trainer':'train_cuda_xgb608_capacity.py','gpu_run':'cuda-xgb608-capacity-2021-001','smoke_run':'cuda-xgb608-capacity-smoke-001'},
}

def digest(path:Path)->str:
    """Compute streaming SHA256 without decoding file contents."""
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def main()->None:
    """Verify immutable2021 outputs only; never launch or select a2022 run."""
    parser=argparse.ArgumentParser()
    parser.add_argument('--experiment',choices=RECIPES,required=True)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--archive-sha256',required=True)
    parser.add_argument('--manifest-sha256',required=True)
    parser.add_argument('--smoke-manifest-sha256',required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--selection-run-id',required=True)
    args=parser.parse_args()
    for name in [args.run_id,args.selection_run_id]:
        if Path(name).name!=name or name in ['', '.', '..']:raise ValueError('Invalid output run ID')
    out=ROOT/'artifacts'/args.run_id;out.mkdir(exist_ok=False);started=time.monotonic()
    report={'run_id':args.run_id,'experiment':args.experiment,'year':2021,'timestamp_utc':datetime.now(timezone.utc).isoformat(),'command':shlex.join([sys.executable,*sys.argv]),'status':'initializing','training':False,'kaggle':None,'environment':{'python':sys.version,'platform':platform.platform()},'source_sha256':digest(Path(__file__))}
    try:
        recipe=RECIPES[args.experiment];path=ROOT/'docs'/recipe['protocol'];assert digest(path)==recipe['protocol_sha256'],'Frozen protocol changed';protocol=json.loads(path.read_text())
        for name,key in [(recipe['verifier'],'return_verifier_sha256'),(recipe['selector'],'local_selector_sha256'),(recipe['trainer'],'trainer_sha256')]:
            source=ROOT/'scripts'/name;assert digest(source)==protocol[key],name
            target=out/'frozen_source'/name;target.parent.mkdir(exist_ok=True);shutil.copy2(source,target)
        assert digest(ROOT/'sintef_project/postprocess.py')==protocol['postprocess_sha256'],'Projection source changed'
        spec=importlib.util.spec_from_file_location('reviewed_gpu2021_return',ROOT/'scripts'/recipe['verifier']);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        imported=out/'download'
        helper.import_archive(args.archive,args.archive_sha256,imported,recipe['gpu_run'],recipe['smoke_run'])
        directory=imported/recipe['gpu_run'];smoke=imported/recipe['smoke_run']/'manifest.json'
        report['verification']=helper.verify_return(directory,smoke,args.manifest_sha256,args.smoke_manifest_sha256,protocol,2021,ROOT,recipe['protocol_sha256'])
        selector_command=[sys.executable,str(ROOT/'scripts'/recipe['selector']),'--year','2021','--gpu-run',str(directory),'--gpu-manifest-sha256',args.manifest_sha256,'--smoke-manifest',str(smoke),'--smoke-manifest-sha256',args.smoke_manifest_sha256,'--protocol',str(path),'--run-id',args.selection_run_id]
        with (out/'selection.log').open('x') as log:subprocess.run(selector_command,stdout=log,stderr=subprocess.STDOUT,check=True)
        selection_path=ROOT/'artifacts'/args.selection_run_id/'manifest.json';selection=json.loads(selection_path.read_text());assert selection['status']=='complete'
        lock_path=selection_path.parent/'selection_2021.json';lock=json.loads(lock_path.read_text())
        report.update(status='complete',protocol_sha256=recipe['protocol_sha256'],archive_sha256=args.archive_sha256,hosted_manifest_sha256=args.manifest_sha256,hosted_smoke_manifest_sha256=args.smoke_manifest_sha256,selection_command=selector_command,selection_run=args.selection_run_id,selection_manifest_sha256=digest(selection_path),selection_lock_sha256=digest(lock_path),selected_weight=lock['selected_weight'],selected_micro_auc=lock['selected_2021_micro_auc'],anchor_micro_auc=lock['anchor_2021_micro_auc'],control_micro_auc=lock['control_2021_micro_auc'],eligible_for_2022=lock['eligible_for_2022'],note='2021 result only. Independent lock verification required before any2022 training; no2022 action performed here.')
        print(json.dumps({k:report[k] for k in ['status','selected_weight','selected_micro_auc','anchor_micro_auc','control_micro_auc','eligible_for_2022','selection_lock_sha256']},indent=2),flush=True)
    except BaseException:report.update(status='failed',error=traceback.format_exc());raise
    finally:
        shutil.copy2(__file__,out/'import_cuda_dual_2021_return.py');report['runtime_seconds']=time.monotonic()-started
        report['artifacts']={p.relative_to(out).as_posix():digest(p) for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'}
        (out/'manifest.json').write_text(json.dumps(report,indent=2))
        (out/'experiment_record.md').write_text(f'\n## {args.run_id} — {report["timestamp_utc"]}\n\nStatus {report["status"]}; strict2021 GPU return import and frozen selector. Command `{report["command"]}`. Selected weight {report.get("selected_weight")}, AUC {report.get("selected_micro_auc")}, anchor {report.get("anchor_micro_auc")}, matchedcontrol {report.get("control_micro_auc")}, eligible2022 {report.get("eligible_for_2022")}. Runtime {report["runtime_seconds"]:.3f}s. All hashes/environment/sources in manifest; no training or Kaggle submission.\n')

if __name__=='__main__':main()
