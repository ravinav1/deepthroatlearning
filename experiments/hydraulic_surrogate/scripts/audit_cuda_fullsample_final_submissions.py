"""Audit actual fullsample608 primary and predeclared standalone CSV exports."""
from datetime import datetime,timezone
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
HELPER_SHA='861d66d33ba2cb3fcc7fd177beeb0f7595ce66c42a27e37fc835da5dc6fdda2e'
FINAL_SHA='4bf91a4ceb200a884802f40af82e3f1f186f50d647c4d44f7bd4d6f601b6714e'
DIAGNOSTIC_SHA='ffae7413c80ae91f5bec45de0c28100fba712c982ea1e1c59b1900c429e51b37'


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb')as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True)
    parser.add_argument('--primary-manifest-sha256',required=True);parser.add_argument('--diagnostic-manifest-sha256',required=True)
    args=parser.parse_args()
    if Path(args.run_id).name!=args.run_id or args.run_id in ['', '.', '..']:raise ValueError('Invalid run ID')
    out=ROOT/'artifacts'/args.run_id;out.mkdir(exist_ok=False);start=time.monotonic()
    report={'status':'running','timestamp_utc':datetime.now(timezone.utc).isoformat(),'command':[sys.executable,*sys.argv],
            'training':False,'environment':{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__}}
    try:
        helper_path=ROOT/'scripts/audit_cuda_dual_final_submissions.py'
        assert sha(helper_path)==HELPER_SHA
        spec=importlib.util.spec_from_file_location('frozen_dual_audit',helper_path);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        primary=ROOT/'artifacts/submission-cuda-xgb608-fullsample-anchor-blend-001'
        diagnostic=ROOT/'artifacts/submission-cuda-xgb608-fullsample-standalone-001'
        pp=ROOT/'docs/cuda_xgb608_fullsample_final_protocol.json';dp=ROOT/'docs/cuda_xgb608_fullsample_standalone_protocol.json'
        assert sha(pp)==FINAL_SHA and sha(dp)==DIAGNOSTIC_SHA
        d=json.loads(dp.read_text());p=json.loads(pp.read_text())
        for section in ['source','inputs']:
            for name,h in d[section].items():assert sha(ROOT/name)==h
        assert sha(diagnostic/'manifest.json')==args.diagnostic_manifest_sha256
        dm=json.loads((diagnostic/'manifest.json').read_text())
        assert dm['protocol_sha256']==DIAGNOSTIC_SHA and dm['primary_verified_manifest_sha256']==args.primary_manifest_sha256
        assert not dm['diagnostic_export']['automatic_promotion']
        results=helper.audit_one('fullsample608',primary,args.primary_manifest_sha256,diagnostic,args.diagnostic_manifest_sha256,pp,FINAL_SHA)
        pm=json.loads((primary/'manifest.json').read_text())
        hosted=[x for x in (primary/'download').rglob('manifest.json')if sha(x)==pm['hosted_manifest_sha256']]
        sample=hosted[0].parent/'train_sample_final.npy'
        assert sha(sample)==p['train_sample_sha256']=='faba3c7f9b70486171bd0f3704fc6c558d05f359d4f8aa0160a0dfbc3d89ad01'
        actual=np.load(sample,allow_pickle=False);expected=np.arange(490896,dtype=np.int64)
        assert actual.dtype==expected.dtype and actual.tobytes()==expected.tobytes()
        report.update(status='passed',results=results,protocol_sha256=FINAL_SHA,diagnostic_protocol_sha256=DIAGNOSTIC_SHA,
                      helper_source_sha256=HELPER_SHA,all490896_training_rows_verified=True)
    except BaseException:
        report.update(status='failed',error=traceback.format_exc());raise
    finally:
        shutil.copy2(__file__,out/Path(__file__).name)
        report['runtime_seconds']=time.monotonic()-start;report['audit_source_sha256']=sha(Path(__file__))
        (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'status':report['status'],'manifest_sha256':sha(out/'manifest.json'),'runtime_seconds':report['runtime_seconds']}),flush=True)
if __name__=='__main__':main()
