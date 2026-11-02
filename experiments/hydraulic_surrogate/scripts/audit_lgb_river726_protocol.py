"""Audit the actual frozen LGB726 protocol after independent full-cache acceptance."""
from datetime import datetime,timezone
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
import traceback
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import train_lgb_river726 as runner
from scripts.train_lgb_shop664 import sha,np,pd


def main() -> None:
    """Verify exact source/sample/control/cache closure without building features."""
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--protocol-sha256',required=True)
    ap.add_argument('--preflight',type=Path,required=True);ap.add_argument('--preflight-sha256',required=True);ap.add_argument('--run-id',required=True)
    a=ap.parse_args();assert Path(a.run_id).name==a.run_id and a.run_id not in ['.','..']
    out=ROOT/'artifacts'/a.run_id;out.mkdir(exist_ok=False);start=time.monotonic()
    r={'status':'initializing','timestamp_utc':datetime.now(timezone.utc).isoformat(),'command':[sys.executable,*sys.argv],
        'training':False,'labels_decoded':False,'features_built':False,'physical_solutions_recomputed':False}
    try:
        assert sha(a.protocol)==a.protocol_sha256;p=json.loads(a.protocol.read_text())
        template=ROOT/'artifacts/lgb-river726-template-independent-audit-001/manifest.json'
        assert sha(template)=='eb34cb9cfbdde95aee2e7fec1276fff606b74467f2867fdfd07731c16cf26850'
        t=json.loads(template.read_text());assert t['status']=='passed_template_only'
        for name,h in t['source'].items():assert sha(ROOT/name)==p['source'][name]==h
        for name,h in p['source'].items():assert sha(ROOT/name)==h
        for name,h in p['inputs'].items():assert sha(ROOT/name)==h
        assert sha(ROOT/runner.BASE_PROTOCOL)==runner.BASE_SHA
        original=json.loads((ROOT/runner.BASE_PROTOCOL).read_text())
        for key in ['model_parameters','anchor','samples','folds','units']:assert p[key]==original[key]
        assert p['candidate_weights']==[0.,.1,.25,.5]and p['early_stopping']is False and p['final_training']is False
        assert p['resources']=={'model_cpu_threads':2,'minimum_free_percent_before_build':40,
            'minimum_free_percent_before_unit':20,'maximum_simultaneous_large_local_jobs':1}
        assert p['formula']=='project_vinje((1-w)*saved_projected_public_bestSHOP664+w*rawLGB726),float64'
        assert p['control']==t['controls']
        cache=p['cache'];assert cache['directory']=='artifacts/hyljelihyl-river-full-cache-002'
        assert cache['manifest_sha256']=='3a195f72b9c837ab4aa14e06662be078f72699e44122c81697efe32628a3c738'
        assert cache['protocol_sha256']=='e4b7bdf6954838739174f767a0a69ab110317fef069e646ac8725ab8cc9e187e'
        assert sha(ROOT/cache['audit'])==cache['audit_sha256']
        accepted=json.loads((ROOT/cache['audit']).read_text())
        assert accepted['status']=='passed'and accepted['cache_manifest_sha256']==cache['manifest_sha256']
        assert accepted['cache_protocol_sha256']==cache['protocol_sha256']and accepted['case_count']==3647
        assert accepted['all_cases_physical_and_replay_verified']and accepted['canonical_reuse_verified']and accepted['workers_exited']
        assert accepted['independent_features_byte_exact']and accepted['immutable_receipts_verified']and accepted['phase_lineage_and_orders_verified']
        assert accepted['features_sha256']=='0d9b6c20e0054ae3f2380cdd96579c69c22f1eb71a6b4ec411babc717ec39191'
        physical=json.loads((ROOT/cache['directory']/'protocol.json').read_text())
        assert p['feature_columns']==original['feature_columns']+physical['feature_specification']['columns']
        assert len(p['feature_columns'])==len(set(p['feature_columns']))==726
        assert p['source']['sintef_project/hyljelihyl_river_features.py']==physical['source']['sintef_project/hyljelihyl_river_features.py']
        target=(ROOT/'sintef_project/data/kernel/Unit_commitment_decisions.csv').resolve();read=pd.read_csv
        def guarded(path,*args,**kwargs):
            if isinstance(path,(str,Path))and Path(path).resolve()==target:assert not args and kwargs=={'usecols':['Run No','starttime']}
            return read(path,*args,**kwargs)
        with patch.object(pd,'read_csv',guarded):_,_,meta,folds=runner.check_closure(p)
        for year,split in folds.items():
            assert len(split['validation_cases'])==365
            assert len(split['sample'])=={'2021':146899,'2022':171427}[year]
            assert (pd.to_datetime(meta.iloc[split['train_cases']].starttime,utc=True)+pd.Timedelta(hours=167)<pd.Timestamp(f'{year}-01-01',tz='UTC')).all()
        assert sha(a.preflight)==a.preflight_sha256;pre=json.loads(a.preflight.read_text())
        assert pre['status']=='passed_preflight_only'and not pre['training']and not pre['labels_decoded']and pre['year']==2021
        assert pre['protocol_sha256']==a.protocol_sha256 and pre['completed_units']==0 and pre['models']=={}
        for name,h in pre['artifacts'].items():assert sha(a.preflight.parent/name)==h
        r.update(status='passed',protocol_sha256=a.protocol_sha256,preflight_manifest_sha256=a.preflight_sha256,
            cache_manifest_sha256=cache['manifest_sha256'],cache_audit_sha256=cache['audit_sha256'],case_count=3647,
            exact_reviewed_source_unchanged=True,exact_original_samples_and_parameters=True,feature_count=726,
            control_audits_bound=True,template_audit_sha256=sha(template),
            scope='Enable frozen2021 fit under existing2thread/resource authorization. Only independently eligible2021 selection lock enables the fixed2022 continuation; no finalfit or submission authorized by this protocol review.')
    except BaseException:r.update(status='failed',error=traceback.format_exc());raise
    finally:
        shutil.copy2(__file__,out/Path(__file__).name);r['runtime_seconds']=time.monotonic()-start
        r['artifacts']={q.name:sha(q)for q in out.iterdir()if q.is_file()and q.name!='manifest.json'}
        (out/'manifest.json').write_text(json.dumps(r,indent=2));print(json.dumps({'status':r['status'],'manifest_sha256':sha(out/'manifest.json'),'seconds':r['runtime_seconds']}),flush=True)

if __name__=='__main__':main()
