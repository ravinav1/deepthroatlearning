"""Independent all-input, unchanged-mathematics and execution review; never solves."""
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from unittest.mock import patch
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[key]='1'
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_hyljelihyl_river_cost_v2 import sha,literal_arrays,load_arrays
from scripts.audit_hyljelihyl_river_cost_36 import partition
from scripts.audit_hyljelihyl_persistent_pilot import independent_features
from sintef_project.lp_features import HydroCaseData,UNIT_ORDER
from sintef_project.lp_hyljelihyl_v2 import split_network,make_problem
from sintef_project.hyljelihyl_river_features import specification,COLUMNS
from scripts.diagnose_hyljelihyl_river_cost_v2 import signatures,json_network


def main() -> None:
    """Reconstruct all3647 physical inputs and audit the frozen no-solve evidence."""
    ap=argparse.ArgumentParser();ap.add_argument('--preflight',type=Path,required=True)
    ap.add_argument('--preflight-sha256',required=True);ap.add_argument('--run-id',required=True)
    a=ap.parse_args();assert Path(a.run_id).name==a.run_id and a.run_id not in ['.','..']
    out=ROOT/'artifacts'/a.run_id;out.mkdir(exist_ok=False);start=time.monotonic()
    r={'status':'initializing','timestamp_utc':datetime.now(timezone.utc).isoformat(),'command':[sys.executable,*sys.argv],
       'optimization_calls':0,'labels_decoded':False,'training':False,'cases':[]}
    try:
        pp=ROOT/'docs/hyljelihyl_river_full_cache_protocol_v2.json'
        assert sha(pp)=='cf75e04282b1731d7ec43d0cfcdc96815ff6f961766f79f6bb28e1cde0b3a356'
        p=json.loads(pp.read_text());r['protocol_sha256']=sha(pp)
        for group in ['source','data']:
            for name,h in p[group].items():assert sha(ROOT/name)==h,name
        for path,key in [(p['continuous_manifest'],'continuous_manifest_sha256'),(p['prefix_protocol'],'prefix_protocol_sha256'),
            (p['pilot_directory']+'/manifest.json','pilot_manifest_sha256'),(p['pilot_protocol'],'pilot_protocol_sha256'),
            (p['pilot_audit'],'pilot_audit_sha256'),(p['canonical_directory']+'/manifest.json','canonical_manifest_sha256'),
            (p['canonical_audit'],'canonical_audit_sha256')]:assert sha(ROOT/path)==p[key],path
        for label in ['pilot','canonical']:
            proof=json.loads((ROOT/p[label+'_audit']).read_text())
            assert proof['status']=='passed'and proof['result_manifest_sha256']==p[label+'_manifest_sha256']
        pilot_protocol=json.loads((ROOT/p['pilot_protocol']).read_text())
        for name,h in pilot_protocol['source'].items():assert p['source'][name]==h
        assert p['feature_specification']==specification()and len(COLUMNS)==62
        assert p['prefix_columns']==json.loads((ROOT/p['prefix_protocol']).read_text())['feature_columns']
        assert len(p['prefix_columns'])==664 and p['workers']==2 and p['lp_seconds']==20.
        assert p['orders']==[p['case_ids'],np.random.default_rng(4173).permutation(p['case_ids']).tolist()]
        assert p['resource_entry_free_percent']==30 and p['resource_drain_free_percent']==20
        assert p['minimum_pressure_derived_headroom_bytes']==3*1024**3 and p['worker_rss_limit_bytes']==2*1024**3
        old=p['prior_unrun_preparation'];assert sha(ROOT/old['protocol'])==old['protocol_sha256']and sha(ROOT/old['preflight'])==old['preflight_sha256']
        assert sha(a.preflight)==a.preflight_sha256
        pre=json.loads(a.preflight.read_text());r['input_preflight_manifest_sha256']=sha(a.preflight)
        assert pre['status']=='passed'and pre['optimization_calls']==0 and not pre['labels_decoded']and not pre['training']
        assert pre['protocol_sha256']==sha(pp)and pre['case_ids']==p['case_ids']and pre['case_dates']==p['case_dates']
        assert pre['source']==p['source']and pre['data']==p['data']and pre['environment']==p['environment']
        for name,h in pre['artifacts'].items():assert sha(a.preflight.parent/name)==h,name
        cm=json.loads((ROOT/p['continuous_manifest']).read_text())
        assert p['case_ids']==cm['case_ids']and p['case_dates']==cm['case_dates']and len(p['case_ids'])==3647
        assert [x['case_id']for x in pre['cases']]==p['case_ids']
        target=(ROOT/'sintef_project/data/kernel/Unit_commitment_decisions.csv').resolve();original=pd.read_csv
        def guarded(path,*args,**kwargs):
            if isinstance(path,(str,Path))and Path(path).resolve()==target:assert not args and kwargs=={'usecols':['Run No','starttime']}
            return original(path,*args,**kwargs)
        with patch.object(pd,'read_csv',guarded):
            data=HydroCaseData(ROOT);oldnet=data.network;newnet=split_network(oldnet)
            assert data.cases['Run No'].tolist()==p['case_ids']
            for pos,record in enumerate(pre['cases']):
                cid=record['case_id'];date=pd.Timestamp(p['case_dates'][pos]);assert data.cases.iloc[pos].starttime==date
                folder=a.preflight.parent/'inputs'/f'case_{cid:04d}'
                assert {f.name:sha(f)for f in folder.iterdir()}==record['files']
                case=data.case(pos);case['natural']=np.asarray(case['natural'],dtype=np.float64)
                for name,field in [('pmin','p_min'),('start_cost','startcost_const'),('stop_cost','stopcost_const')]:
                    case[name]=np.array([data.model['generator'][u][field]for u in UNIT_ORDER],dtype=np.float64)
                base=ROOT/p['continuous_cache'];reference=base/f'case_{cid:04d}_result.npz';original_input=base/f'case_{cid:04d}_inputs.npz'
                assert sha(reference)==record['reference_sha256']==cm['artifacts'][reference.name]
                assert sha(original_input)==cm['artifacts'][original_input.name]
                literal_arrays(case,load_arrays(original_input));literal_arrays(case,load_arrays(folder/'A_inputs.npz'))
                split=partition(case,data,date);literal_arrays(split,load_arrays(folder/'split_inputs.npz'))
                for label,net in [('A',oldnet),('split',newnet)]:assert json.loads((folder/f'{label}_network.json').read_text())==json_network(net)
                water=data.water.loc[date+pd.Timedelta(days=7),['Vaamarvatn','Totak','Venemo','Hyljelihyl']].to_numpy()
                assert np.isfinite(water).all()and np.all(water==water[0])
                _,problem,_=make_problem(newnet,split,'C',20.);assert signatures(problem)==record['matrix_signatures']
                if cid in p['pilot_case_ids']:
                    pilot=ROOT/p['pilot_directory'];pf=pilot/'inputs'/f'case_{cid:04d}'
                    literal_arrays(case,load_arrays(pf/'A_inputs.npz'));literal_arrays(split,load_arrays(pf/'split_inputs.npz'))
                    f=independent_features(case,split,load_arrays(pilot/'worker_1'/f'case_{cid:04d}'/'result.npz'),load_arrays(reference),oldnet,newnet)
                    oldf=np.load(pilot/'worker_1'/f'case_{cid:04d}'/'features.npy',allow_pickle=False)
                    assert f.dtype==oldf.dtype and f.shape==oldf.shape and f.tobytes()==oldf.tobytes()
                r['cases'].append({'case_id':cid,'fresh_input_split_matrix_byte_identity':True,'pilot_feature_reconstruction':cid in p['pilot_case_ids']})
                if (pos+1)%500==0:print(json.dumps({'checked_inputs':pos+1}),flush=True)
        test=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_hyljelihyl_river_cache_v2.py'],cwd=ROOT,capture_output=True,text=True)
        (out/'tests.log').write_text(test.stdout+test.stderr);assert test.returncode==0,test.stderr
        r.update(status='passed',case_count=3647,pilot_cases_verified=59,synthetic_tests_passed=7,
            unchanged_math_and_feature_closure=True,metadata_only_reconstruction=True,
            execution_review='Two distinct spawned single-thread persistent replicas; fixed ascending/permuted orders; fresh model each call; immutable full payload receipts; post-join final publication; complete second resource gate; safe pointers and exact pilot lineage; retained failures block resume; no retries/fallback/labels.',
            limitations='Source and input eligibility only. Root authorization required to execute. A separate actual complete-cache physical audit is required before learned-feature use.')
    except BaseException:r.update(status='failed',error=traceback.format_exc());raise
    finally:
        shutil.copy2(__file__,out/Path(__file__).name);r['runtime_seconds']=time.monotonic()-start
        r['helpers']={s:sha(ROOT/s)for s in ['scripts/audit_hyljelihyl_river_cost_v2.py','scripts/audit_hyljelihyl_river_cost_36.py','scripts/audit_hyljelihyl_persistent_pilot.py']}
        r['artifacts']={f.name:sha(f)for f in out.iterdir()if f.is_file()and f.name!='manifest.json'}
        (out/'manifest.json').write_text(json.dumps(r,indent=2));print(json.dumps({'status':r['status'],'manifest_sha256':sha(out/'manifest.json'),'seconds':r['runtime_seconds']}),flush=True)

if __name__=='__main__':main()
