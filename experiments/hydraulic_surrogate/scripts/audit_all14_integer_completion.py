"""Independent no-solve review of the fixed twelve-case all14 diagnostic."""
from datetime import datetime,timezone
import ast,hashlib,json,os
from pathlib import Path
import platform,shutil,subprocess,sys,time,traceback
from unittest.mock import patch
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[name]='1'
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'artifacts/all14-integer-water-budget-completion-independent-preflight-001';OUT.mkdir(exist_ok=False)
start=time.monotonic()
def sha(p):
    with p.open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
report={'status':'running','timestamp_utc':datetime.now(timezone.utc).isoformat(),'command':f'{sys.executable} scripts/audit_all14_integer_completion.py','environment':{'python':sys.version,'platform':platform.platform()},'solves':0,'labels_decoded':False,'training':False,'kaggle_submission':None}
try:
    pp=ROOT/'docs/all14_integer_water_budget_completion_protocol.json'
    assert sha(pp)=='63fa814bf8db436194272bf0596ad848e841625deb7358bd081825fbbc2b7b7d'
    p=json.loads(pp.read_text());report['protocol_sha256']=sha(pp)
    for section in ['source','data','inputs']:
        for name,h in p[section].items():assert sha(ROOT/name)==h,name
    pre=ROOT/'artifacts/all14-integer-water-budget-completion-preflight-001/manifest.json'
    assert sha(pre)=='5b919aa35a537c0deeaa97b2d6707faca320e3c41d5d6f7f2844b154260dbf8e'
    pm=json.loads(pre.read_text());assert pm['status']=='passed_preflight_only'and pm['solves']==0 and not pm['labels_decoded']
    assert pm['protocol_sha256']==sha(pp)
    for name,h in pm['artifacts'].items():assert sha(pre.parent/name)==h,name
    original=json.loads((ROOT/'docs/integer_water_budget_protocol.json').read_text())
    assert [r['case_id']for r in p['cases']]==original['case_ids']and p['blocks']==original['blocks']
    assert p['unit_indices']==list(range(14)) and p['workers']==p['attempts_per_case']==1
    assert p['milp_seconds']==600. and p['reused_milp_seconds']==60. and p['fixed_lp_seconds_each']==1. and p['peak_rss_limit_bytes']==2*1024**3
    assert p['resource_minimum_free_percent']==30 and not p['automatic_promotion']
    two=json.loads((ROOT/'docs/all14_integer_water_budget_probe_protocol_v2.json').read_text())
    for name in two['source']:assert p['source'][name]==two['source'][name]
    reused=ROOT/p['reuse']['directory'];rm=json.loads((reused/'manifest.json').read_text())
    assert sha(reused/'manifest.json')==p['reuse']['manifest_sha256']and not rm['physical_feasibility_passed'] and not rm['labels_decoded']
    proof=ROOT/p['reuse']['audit'];assert sha(proof)==p['reuse']['audit_sha256']and json.loads(proof.read_text())['status']=='passed'
    assert p['completion_case_ids']==[2186] and len(p['reuse']['case_ids'])==11
    assert p['reuse']['case_ids']==[r['case_id']for r in rm['cases']if r['proof']['proven_optimal']]
    assert pm['new_case_ids']==[2186] and pm['reused_case_ids']==p['reuse']['case_ids']
    for i in p['reuse']['case_ids']:
        a=reused/f'case_{i:04d}';b=pre.parent/f'case_{i:04d}'
        af={f.relative_to(a).as_posix():f for f in a.rglob('*')if f.is_file()};bf={f.relative_to(b).as_posix():f for f in b.rglob('*')if f.is_file()}
        assert set(af)==set(bf)
        for name in af:assert af[name].read_bytes()==bf[name].read_bytes()
    from sintef_project.lp_features import HydroCaseData,UNIT_ORDER
    target=(ROOT/'sintef_project/data/kernel/Unit_commitment_decisions.csv').resolve();old=pd.read_csv
    def guarded(path,*args,**kwargs):
        if isinstance(path,(str,Path))and Path(path).resolve()==target:assert not args and kwargs=={'usecols':['Run No','starttime']}
        return old(path,*args,**kwargs)
    with patch.object(pd,'read_csv',guarded):
        data=HydroCaseData(ROOT)
        for selected in p['cases']:
            i,index=selected['case_id'],selected['metadata_position']
            assert int(data.cases.iloc[index]['Run No'])==i and str(data.cases.iloc[index].starttime)==selected['date']
            assert pd.Timestamp(selected['date'])+pd.Timedelta(hours=167)<pd.Timestamp('2021-01-01',tz='UTC')
            fresh=data.case(index);fresh['natural']=np.asarray(fresh['natural'],dtype=np.float64)
            for name,field in [('pmin','p_min'),('start_cost','startcost_const'),('stop_cost','stopcost_const')]:fresh[name]=np.asarray([data.model['generator'][u][field]for u in UNIT_ORDER],dtype=np.float64)
            with np.load(ROOT/p['cache']/f'case_{i:04d}_inputs.npz',allow_pickle=False)as saved:
                assert set(saved.files)==set(fresh)
                for key in saved.files:assert saved[key].dtype==fresh[key].dtype and saved[key].shape==fresh[key].shape and saved[key].tobytes()==fresh[key].tobytes(),(i,key)
            assert (pre.parent/f'case_{i:04d}_inputs.npz').read_bytes()==(ROOT/p['cache']/f'case_{i:04d}_inputs.npz').read_bytes()
            control=json.loads((ROOT/p['comparison']/f'case_{i:04d}_proof.json').read_text())
            assert control['status']=='complete'and control['proof']['proven_optimal']and control['input_sha256']==sha(ROOT/p['cache']/f'case_{i:04d}_inputs.npz')
    source=(ROOT/'scripts/complete_all14_integer_water_budget.py').read_text()
    functions={n.name:n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
    old_worker=next(n for n in ast.parse((ROOT/'scripts/probe_all14_integer_water_budget_v2.py').read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='worker')
    assert ast.dump(functions['worker'])==ast.dump(old_worker).replace('value=60.0','value=600.0')
    old={n.name:n for n in ast.parse((ROOT/'scripts/diagnose_all14_integer_water_budget.py').read_text()).body if isinstance(n,ast.FunctionDef)}
    for name in ['legal_dates','physical_gates','read_labels']:assert ast.dump(functions[name])==ast.dump(old[name]),name
    assert "assert list(jobs)==p['completion_case_ids']==[2186]and len(records)==11"in source
    assert source.index("(out/'physical_lock.json').write_text")<source.index("if not all(gates.values()):return")<source.index("labels=read_labels(ids)")
    assert "ProcessPoolExecutor(max_workers=1"in source and "max_tasks_per_child=1"in source
    assert "records[i]=pool.submit(worker,*jobs[i]).result()"in source and source.count('pool.submit(')==1
    test=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_all14_integer_completion.py'],cwd=ROOT,text=True,capture_output=True)
    (OUT/'tests.txt').write_text(test.stdout+test.stderr);assert test.returncode==0,test.stderr
    from scripts.complete_all14_integer_water_budget import physical_gates
    ids=original['case_ids'];rows=[{'case_id':i,'status':'complete','proof':{'mip_status':0,'proven_optimal':True},'fixed_lp_all_arrays_bitwise':True,'all_slacks_zero':True,'objective_subset_checks':True,'peak_rss_bytes':1000}for i in ids]
    assert all(physical_gates(rows,ids).values())
    for key,value in [('fixed_lp_all_arrays_bitwise',False),('all_slacks_zero',False),('objective_subset_checks',False),('peak_rss_bytes',2*1024**3)]:
        changed=json.loads(json.dumps(rows));changed[0][key]=value;assert not all(physical_gates(changed,ids).values())
    report.update(status='passed',preflight_manifest_sha256=sha(pre),all12_fresh_input_reconstruction_byte_exact=True,exact_original_ids_blocks=True,all_intervals_before_2021=True,all_eleven_proven_case_bytes_preserved=True,worker_AST_identical_except_60to600=True,sole_new_case_id=2186,tests_passed=2,extra_negative_gate_cases=4,review='Exactly one fresh2186-only600second proof completion; previous11proven directories exactbytes; original failedrun/source remain immutable. WorkerAST identical except60to600; same2x1s fixedLPs and allmath. Resourceguard30percent/no swapgrowth; all12 physicalgates+seal before selectedpre2021labels. No furtherretry/fallback/model/publicpromotion.')
except BaseException:
    report.update(status='failed',error=traceback.format_exc());raise
finally:
    shutil.copy2(__file__,OUT/Path(__file__).name)
    report['runtime_seconds']=time.monotonic()-start
    report['artifacts']={p.name:sha(p)for p in OUT.iterdir()if p.is_file()and p.name!='manifest.json'}
    (OUT/'manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps({'status':report['status'],'manifest_sha256':sha(OUT/'manifest.json'),'runtime_seconds':report['runtime_seconds']}),flush=True)
