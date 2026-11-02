"""Immutable/resumable C-only cache, two persistent independent replica workers."""
from __future__ import annotations
import argparse
import ast
from datetime import datetime,timezone
import json
import multiprocessing as mp
import os
from pathlib import Path
import resource
import shlex
import shutil
import sys
import time
import traceback
from unittest.mock import patch
for variable in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[variable]='1'
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sintef_project.lp_features import HydroCaseData,UNIT_ORDER
from sintef_project.lp_hyljelihyl_v2 import split_network,split_case,make_problem,solve
import sintef_project.lp_hyljelihyl_v2 as river_module
from sintef_project.hyljelihyl_river_features import compute_features,specification,COLUMNS
from scripts.diagnose_hyljelihyl_river_cost_36 import json_network,load_network,signatures
from scripts.pilot_hyljelihyl_river_persistent import resources,orders
from scripts.diagnose_integer_water_budget import arrays,digest
from scripts.probe_all14_integer_water_budget_v2 import identical_arrays,all_slacks_exact_zero
from scripts.pilot_integer_water_budget import environment

PROTOCOL=ROOT/'docs/hyljelihyl_river_full_cache_protocol_v3.json'
NEW=['scripts/build_hyljelihyl_river_cache.py','tests/test_hyljelihyl_river_cache.py','docs/hyljelihyl_river_full_cache.md',
    'scripts/build_hyljelihyl_river_cache_v2.py','tests/test_hyljelihyl_river_cache_v2.py','docs/hyljelihyl_river_full_cache_v2.md',
    'scripts/build_hyljelihyl_river_cache_v3.py','tests/test_hyljelihyl_river_cache_v3.py','docs/hyljelihyl_river_full_cache_v3.md']


def write_json(path: Path, value: dict, immutable=False) -> None:
    """Publish through atomic rename; finalized receipts may never be overwritten."""
    if immutable and path.exists():raise ValueError('Immutable file already exists: '+str(path))
    temporary=path.with_name(path.name+'.temporary')
    temporary.write_text(json.dumps(value,indent=2));os.replace(temporary,path)


def freeze() -> None:
    """Freeze mathematics, exact prefix, import lineage and full-case execution order."""
    pilot_protocol=ROOT/'docs/hyljelihyl_river_persistent_pilot_protocol.json';old=json.loads(pilot_protocol.read_text())
    cache=json.loads((ROOT/old['cache']/'manifest.json').read_text())
    prefix=ROOT/'docs/cuda_xgb664_protocol.json';prefix_p=json.loads(prefix.read_text())
    p={'version':'hyljelihyl_river_full_cache_v3','created_utc':datetime.now(timezone.utc).isoformat(),
       'source':{**old['source'],**{n:digest(ROOT/n)for n in NEW}},'data':old['data'],
       'case_ids':cache['case_ids'],'case_dates':cache['case_dates'],'orders':orders(cache['case_ids']),
       'continuous_cache':old['cache'],'continuous_manifest':old['cache']+'/manifest.json',
       'continuous_manifest_sha256':old['cache_manifest_sha256'],
       'prefix_protocol':str(prefix.relative_to(ROOT)),'prefix_protocol_sha256':digest(prefix),
       'prefix_columns':prefix_p['feature_columns'],'feature_specification':specification(),
       'environment':environment(),'lp_seconds':20.,'workers':2,'seed':4173,
       'resource_entry_free_percent':30,'resource_drain_free_percent':20,
       'minimum_pressure_derived_headroom_bytes':3*1024**3,'worker_rss_limit_bytes':2*1024**3,
       'replay_execution_type':'Two distinct persistent single-thread processes; fresh C model/solver each call; ascending versus seed4173 permutation. Explicit59-case pilot import with36 canonical fresh-process lineage.',
       'pilot_directory':'artifacts/hyljelihyl-river-persistent-pilot-001',
       'pilot_manifest_sha256':'134ad85d2533e33ac218f4be70c4ef5d246cef37fb77a4656f6c4e855538dc49',
       'pilot_protocol':str(pilot_protocol.relative_to(ROOT)),'pilot_protocol_sha256':digest(pilot_protocol),
       'pilot_audit':'artifacts/hyljelihyl-river-persistent-result-independent-audit-002/manifest.json',
       'pilot_audit_sha256':'7fdf874bb958062424ffbb4278cacb4f5dc84de820a8a968386c9d8d8bd135f8',
       'pilot_case_ids':[c['case_id']for c in old['cases']],
       'canonical_case_ids':old['canonical_case_ids'],'canonical_directory':old['canonical_directory'],
       'canonical_manifest_sha256':old['canonical_manifest_sha256'],'canonical_audit':old['canonical_audit'],
       'canonical_audit_sha256':old['canonical_audit_sha256'],
       'resume':'Only externally hashed same-protocol resource-drain manifests with verified child exit and no case/pair failures; skip verified complete receipts. Operationally interrupted runs require a separately reviewed continuation. Native or replay failures are never automatically retried.',
       'eligibility':'Every3647 case optimal/zero absolute slack/full KKT and byte-exact result,primal-dual,feature replay. No partial cache, fallback or silent substitution.',
       'labels_decoded':False,'training':False,
       'prior_unrun_preparation':{'protocol':'docs/hyljelihyl_river_full_cache_protocol.json',
            'protocol_sha256':'0dd15f72e932834afd78c8647b3abfb873cb3b9416d0c1472d826731cc265f48',
            'preflight':'artifacts/hyljelihyl-river-full-cache-preflight-001/manifest.json',
            'preflight_sha256':digest(ROOT/'artifacts/hyljelihyl-river-full-cache-preflight-001/manifest.json'),
            'reason':'Preserve unrun v1; v2 added post-join publication, complete entry checks and strict receipt pointers. Mathematics/features unchanged.'},
       'compatible_input_preflight':{
            'protocol':'docs/hyljelihyl_river_full_cache_protocol_v2.json',
            'protocol_sha256':'cf75e04282b1731d7ec43d0cfcdc96815ff6f961766f79f6bb28e1cde0b3a356',
            'manifest':'artifacts/hyljelihyl-river-full-cache-v2-preflight-001/manifest.json',
            'manifest_sha256':'569aee683a4a79ef2ed909f4c8ec5b747dd46105d4a49866a2df61880919b0b3',
            'input_audit':'artifacts/hyljelihyl-river-full-cache-v2-independent-preflight-001/manifest.json',
            'input_audit_sha256':'68bc2d9fbfa19f390b09e55f2e41e7085270a1ecd3012e7d3f6337738367dc32',
            'blocked_launch_review':'artifacts/hyljelihyl-river-full-cache-v2-launch-review-001/manifest.json',
            'blocked_launch_review_sha256':'05bcd5134dcc51a8e4155f3713c72c5044f435d4e921a2e3dcacc501d1c5dd7f',
            'reason':'Exact prepared inputs reused; v3 changes only durable replay-failure recording and resource-drain-only resume. Prior v2 launch remains blocked.'}}
    assert len(p['case_ids'])==3647 and len(p['prefix_columns'])==664
    pilot=ROOT/p['pilot_directory'];pm=json.loads((pilot/'manifest.json').read_text())
    assert digest(pilot/'manifest.json')==p['pilot_manifest_sha256']and pm['status']=='passed'
    assert digest(ROOT/p['pilot_audit'])==p['pilot_audit_sha256']
    audit=json.loads((ROOT/p['pilot_audit']).read_text());assert audit['status']=='passed'and audit['result_manifest_sha256']==p['pilot_manifest_sha256']
    p['pilot_worker_manifests_sha256']=[digest(pilot/f'worker_{i}/manifest.json')for i in [1,2]]
    with PROTOCOL.open('x')as f:json.dump(p,f,indent=2)
    print(digest(PROTOCOL),flush=True)


def verify_protocol(p: dict, expected: str) -> None:
    """Require exact immutable mathematical/source/data/import identities."""
    assert digest(PROTOCOL)==expected and p==json.loads(PROTOCOL.read_text())
    assert environment()==p['environment']
    for group in ['source','data']:
        for name,h in p[group].items():assert digest(ROOT/name)==h,name
    for name,h in [(p['continuous_manifest'],p['continuous_manifest_sha256']),
        (p['prefix_protocol'],p['prefix_protocol_sha256']),
        (p['pilot_directory']+'/manifest.json',p['pilot_manifest_sha256']),
        (p['pilot_protocol'],p['pilot_protocol_sha256']),(p['pilot_audit'],p['pilot_audit_sha256']),
        (p['canonical_directory']+'/manifest.json',p['canonical_manifest_sha256']),
        (p['canonical_audit'],p['canonical_audit_sha256'])]:assert digest(ROOT/name)==h,name
    pilot_audit=json.loads((ROOT/p['pilot_audit']).read_text())
    assert (pilot_audit['status']=='passed' and pilot_audit['result_manifest_sha256']==p['pilot_manifest_sha256']
        and pilot_audit['protocol_sha256']==p['pilot_protocol_sha256'] and pilot_audit['case_count']==59
        and pilot_audit['all_cases_physical_and_replay_verified'] and pilot_audit['canonical_reuse_verified']
        and pilot_audit['workers_exited'])
    canonical_audit=json.loads((ROOT/p['canonical_audit']).read_text())
    assert canonical_audit['status']=='passed' and canonical_audit['result_manifest_sha256']==p['canonical_manifest_sha256']
    assert len(p['case_ids'])==3647 and len(set(p['case_ids']))==3647 and p['orders']==orders(p['case_ids'])
    assert p['lp_seconds']==20. and p['workers']==2 and p['feature_specification']==specification()
    prior=p['prior_unrun_preparation']
    assert digest(ROOT/prior['protocol'])==prior['protocol_sha256'] and digest(ROOT/prior['preflight'])==prior['preflight_sha256']
    prior_manifest=json.loads((ROOT/prior['preflight']).read_text())
    assert prior_manifest['status']=='passed' and prior_manifest['optimization_calls']==0
    compatible=p['compatible_input_preflight']
    for name in ['protocol','manifest','input_audit','blocked_launch_review']:
        assert digest(ROOT/compatible[name])==compatible[name+'_sha256']
    input_audit=json.loads((ROOT/compatible['input_audit']).read_text())
    assert (input_audit['status']=='passed' and input_audit['case_count']==3647
        and input_audit['protocol_sha256']==compatible['protocol_sha256']
        and input_audit['input_preflight_manifest_sha256']==compatible['manifest_sha256'])
    assert json.loads((ROOT/compatible['blocked_launch_review']).read_text())['status']=='blocked_source_revision'
    old=json.loads((ROOT/compatible['protocol']).read_text())
    for field in ['data','case_ids','case_dates','feature_specification','environment','orders','lp_seconds','workers']:
        assert old[field]==p[field]
    for name,h in old['source'].items():assert p['source'][name]==h
    def input_ast(path):
        return ast.dump(next(node for node in ast.parse(path.read_text()).body if isinstance(node,ast.FunctionDef)and node.name=='prepare_inputs'),include_attributes=False)
    assert input_ast(ROOT/'scripts/build_hyljelihyl_river_cache_v2.py')==input_ast(Path(__file__))


def prepare_inputs(out: Path,p: dict) -> list[dict]:
    """Fresh all-case metadata-only reconstruction, without any optimization."""
    raw=pd.read_csv;target=(ROOT/'sintef_project/data/kernel/Unit_commitment_decisions.csv').resolve()
    def guarded(path,*args,**kwargs):
        if isinstance(path,(str,Path))and Path(path).resolve()==target:assert not args and kwargs=={'usecols':['Run No','starttime']}
        return raw(path,*args,**kwargs)
    records=[];cm=json.loads((ROOT/p['continuous_manifest']).read_text())
    with patch.object(pd,'read_csv',guarded):
        data=HydroCaseData(ROOT);oldnet=data.network;newnet=split_network(oldnet)
        assert data.cases['Run No'].tolist()==p['case_ids']
        for index,(cid,date)in enumerate(zip(p['case_ids'],p['case_dates'])):
            assert str(data.cases.iloc[index].starttime)==date;date=pd.Timestamp(date)
            case=data.case(index);case['natural']=np.asarray(case['natural'],dtype=np.float64)
            for key,field in [('pmin','p_min'),('start_cost','startcost_const'),('stop_cost','stopcost_const')]:
                case[key]=np.array([data.model['generator'][u][field]for u in UNIT_ORDER],dtype=np.float64)
            reference_path=ROOT/p['continuous_cache']/f'case_{cid:04d}_result.npz'
            input_path=ROOT/p['continuous_cache']/f'case_{cid:04d}_inputs.npz'
            assert digest(input_path)==cm['artifacts'][input_path.name]and digest(reference_path)==cm['artifacts'][reference_path.name]
            assert identical_arrays(case,arrays(input_path))
            times=pd.date_range(date,periods=168,freq='h');water=data.water.loc[date+pd.Timedelta(days=7),['Vaamarvatn','Totak','Venemo','Hyljelihyl']].to_numpy()
            assert np.isfinite(water).all()and np.all(water==water[0])
            minimum=data.minimum.loc[times,'Hyljelihyl'].to_numpy()if'Hyljelihyl'in data.minimum else np.zeros(168)
            split=split_case(case,float(data.volume.loc[date,'Hyljelihyl']),data.inflow.loc[times,'Hyljelihyl'].to_numpy(dtype=np.float64)*.0036,minimum)
            assert abs(split['capacity'][9]-.26)<1e-12 and abs(split['capacity'][4]-307.)<1e-12
            folder=out/'inputs'/f'case_{cid:04d}';folder.mkdir(parents=True)
            np.savez_compressed(folder/'A_inputs.npz',**case);np.savez_compressed(folder/'split_inputs.npz',**split)
            for name,net in [('A',oldnet),('split',newnet)]:write_json(folder/f'{name}_network.json',json_network(net),True)
            _,problem,_=make_problem(newnet,split,'C',20.)
            record={'case_id':cid,'fresh_input_byte_exact':True,'local_terminal_values_equal':True,
                'matrix_signatures':signatures(problem),'reference_sha256':digest(reference_path)}
            if cid in p['pilot_case_ids']:
                old=ROOT/p['pilot_directory']/'inputs'/f'case_{cid:04d}'
                assert identical_arrays(case,arrays(old/'A_inputs.npz'))and identical_arrays(split,arrays(old/'split_inputs.npz'))
                for name,net in [('A',oldnet),('split',newnet)]:assert json.loads((old/f'{name}_network.json').read_text())==json_network(net)
                prior=arrays(ROOT/p['pilot_directory']/'worker_1'/f'case_{cid:04d}'/'result.npz')
                values=compute_features(case,split,prior,arrays(reference_path),oldnet,newnet)
                previous=np.load(ROOT/p['pilot_directory']/'worker_1'/f'case_{cid:04d}'/'features.npy',allow_pickle=False)
                assert values.dtype==previous.dtype and values.shape==previous.shape and values.tobytes()==previous.tobytes()
                record['pilot_inputs_and_features_byte_exact']=True
            record['files']={f.name:digest(f)for f in folder.iterdir()if f.is_file()};records.append(record)
            if (index+1)%100==0:print(json.dumps({'event':'input_preflight','cases':index+1}),flush=True)
    return records


def checked_paths(cache: Path,receipt: dict) -> None:
    """Reject path escapes, changed files and incomplete result receipts."""
    for name,h in receipt['artifacts'].items():
        path=cache/name
        if Path(name).is_absolute()or not path.resolve().is_relative_to(cache.resolve()):raise ValueError('Receipt path escape')
        if digest(path)!=h:raise ValueError('Receipt payload changed: '+name)


def verify_pair(first: Path,second: Path,cid: int) -> dict:
    """Check actual optimal physical outputs and complete literal replica arrays."""
    proofs=[json.loads((f/'proof.json').read_text())for f in [first,second]]
    for folder,proof in zip([first,second],proofs):
        assert proof['status']=='complete'and proof['case_id']==cid and proof['native']['status']==0
        assert proof['all_slacks_zero']and np.isfinite(list(proof['kkt'].values())).all() and max(proof['kkt'].values())<=1e-5
        assert set(proof['artifacts'])=={'result.npz','primal_dual.npz','features.npy','native_output.npz'}
        assert {f.name for f in folder.iterdir()}==set(proof['artifacts'])|{'proof.json'}
        for name,h in proof['artifacts'].items():assert digest(folder/name)==h
    assert proofs[0]['pid']!=proofs[1]['pid']
    for name in ['result.npz','primal_dual.npz','native_output.npz']:
        assert identical_arrays(arrays(first/name),arrays(second/name))
    a,b=np.load(first/'features.npy',allow_pickle=False),np.load(second/'features.npy',allow_pickle=False)
    assert a.shape==b.shape==(168,62)and a.dtype==b.dtype==np.float32 and np.isfinite(a).all()and a.tobytes()==b.tobytes()
    return {'pids':[p['pid']for p in proofs],'replays_byte_exact':True,'features_sha256':digest(first/'features.npy')}


def publish_receipt(cache: Path,cid: int,paths: list[Path],p_sha: str,input_record: dict,origin: dict) -> dict:
    """Write one immutable case receipt only after both complete replica checks."""
    pair=verify_pair(*paths,cid)
    if origin['type']=='phase':
        for path in paths:
            proof=json.loads((path/'proof.json').read_text())
            assert proof['protocol_sha256']==p_sha and proof['input_preflight_sha256']==origin['input_preflight_sha256']
            assert proof['input_record']==input_record
    receipt={'status':'complete','case_id':cid,'protocol_sha256':p_sha,'input_record':input_record,'origin':origin,
        'replica_paths':[str(x.relative_to(cache))for x in paths],**pair,
        'features_path':str((paths[0]/'features.npy').relative_to(cache)),
        'artifacts':{str(f.relative_to(cache)):digest(f)for path in paths for f in path.iterdir()if f.is_file()}}
    folder=cache/'cases'/f'case_{cid:04d}';folder.mkdir(parents=True,exist_ok=True)
    write_json(folder/'complete.json',receipt,True);return receipt


def verify_receipt(cache: Path,cid: int,p_sha: str,protocol: dict,input_record: dict|None=None) -> dict:
    """Resume only a fully finalized, byte-intact, matching-protocol case."""
    path=cache/'cases'/f'case_{cid:04d}'/'complete.json';r=json.loads(path.read_text())
    assert r['status']=='complete'and r['case_id']==cid and r['protocol_sha256']==p_sha
    if input_record is not None:assert r['input_record']==input_record
    assert len(r['replica_paths'])==2 and len(set(r['replica_paths']))==2
    for name in [*r['replica_paths'],r['features_path']]:
        assert not Path(name).is_absolute() and (cache/name).resolve().is_relative_to(cache.resolve())
    assert r['features_path']==str(Path(r['replica_paths'][0])/'features.npy')
    assert digest(cache/r['features_path'])==r['features_sha256']
    expected_files={str(f.relative_to(cache))for name in r['replica_paths']for f in (cache/name).iterdir()if f.is_file()}
    assert set(r['artifacts'])==expected_files
    checked_paths(cache,r);pair=verify_pair(*[cache/x for x in r['replica_paths']],cid)
    if r['origin']['type']=='phase':
        launch=json.loads((cache/'phases'/f"phase_{r['origin']['phase']:03d}"/'launch.json').read_text())
        assert launch['protocol_sha256']==p_sha and launch['input_preflight_sha256']==r['origin']['input_preflight_sha256']
        for index,name in enumerate(r['replica_paths']):
            proof=json.loads((cache/name/'proof.json').read_text())
            assert proof['protocol_sha256']==p_sha and proof['input_preflight_sha256']==launch['input_preflight_sha256']
            assert proof['input_record']==r['input_record'] and proof['pid']==launch['worker_pids'][index]
            assert cid in launch['orders'][index]
    elif r['origin']['type']=='pilot_import':
        assert cid in protocol['pilot_case_ids']
        assert r['origin']['pilot_manifest_sha256']==protocol['pilot_manifest_sha256']
        assert r['origin']['worker_manifests_sha256']==protocol['pilot_worker_manifests_sha256']
        assert r['origin']['canonical_fresh_process_lineage']==(cid in protocol['canonical_case_ids'])
        pilot=ROOT/protocol['pilot_directory'];assert digest(pilot/'manifest.json')==protocol['pilot_manifest_sha256']
        pilot_manifest=json.loads((pilot/'manifest.json').read_text())
        for index,name in enumerate(r['replica_paths'],1):
            expected=f'imports/worker_{index}/case_{cid:04d}';assert name==expected
            assert digest(pilot/f'worker_{index}/manifest.json')==protocol['pilot_worker_manifests_sha256'][index-1]
            for file in (cache/name).iterdir():assert digest(file)==pilot_manifest['artifacts'][f'worker_{index}/case_{cid:04d}/'+file.name]
    else:raise ValueError('Unknown receipt origin')
    assert pair['features_sha256']==r['features_sha256'];return r


def publish_ready_pairs(cache: Path,phase: Path,ids: list[int],complete: dict,
                        p_sha: str,inputs: dict,origin: dict) -> None:
    """Publish available complete pairs; repeat after joins to cover final writes."""
    for cid in ids:
        if cid in complete:continue
        paths=[phase/f'worker_{i}'/f'case_{cid:04d}'for i in [1,2]]
        if all((path/'proof.json').exists()for path in paths):
            proofs=[json.loads((path/'proof.json').read_text())for path in paths]
            if all(proof['status']=='complete'for proof in proofs):
                try:complete[cid]=publish_receipt(cache,cid,paths,p_sha,inputs[cid],origin)
                except Exception:
                    failure=phase/'pair_failures'/f'case_{cid:04d}.json';failure.parent.mkdir(exist_ok=True)
                    write_json(failure,{'status':'failed_pair_verification','case_id':cid,'protocol_sha256':p_sha,
                        'origin':origin,'replica_paths':[str(path.relative_to(cache))for path in paths],
                        'artifacts':{str(f.relative_to(cache)):digest(f)for path in paths for f in path.iterdir()if f.is_file()},
                        'error':traceback.format_exc()},True)
                    raise


def verify_entry_resources(current: dict,p: dict,previous: dict|None=None) -> None:
    """Both entry measurements independently satisfy memory/headroom thresholds."""
    assert current['free_percent']>=p['resource_entry_free_percent']
    assert current['pressure_derived_free_bytes']>=p['minimum_pressure_derived_headroom_bytes']
    if previous is not None:assert current['swapouts']<=previous['swapouts']


def retained_failures(cache: Path) -> list[dict]:
    """Any failed native/case proof prevents automatic retry, even after interruption."""
    failures=[]
    for path in sorted((cache/'phases').glob('phase_*/worker_*/case_*/proof.json')):
        proof=json.loads(path.read_text())
        if proof['status']!='complete':failures.append({'path':str(path.relative_to(cache)),'sha256':digest(path),'native':proof.get('native'),'error':proof.get('error')})
    for path in sorted((cache/'phases').glob('phase_*/pair_failures/case_*.json')):
        failure=json.loads(path.read_text());checked_paths(cache,failure)
        failures.append({'path':str(path.relative_to(cache)),'sha256':digest(path),'case_id':failure['case_id'],'error':failure['error']})
    return failures


def drain_workers(cache: Path,workers: list,timeout: float=60.) -> dict:
    """Request stop between calls and join children; never signal/terminate a solver."""
    (cache/'STOP_AFTER_ACTIVE').touch(exist_ok=True)
    deadline=time.monotonic()+timeout
    for worker in workers:
        if worker.pid is not None:worker.join(timeout=max(0,deadline-time.monotonic()))
    exited=all(worker.pid is None or not worker.is_alive()for worker in workers)
    return {'all_workers_exited':exited,'workers':[{'pid':w.pid,'exitcode':w.exitcode}for w in workers]}


def verify_resume(cache: Path,manifest_sha256: str,protocol_sha256: str,protocol: dict) -> dict:
    """Only a sealed, exited, non-native-failure run can resume in a new phase."""
    assert digest(cache/'manifest.json')==manifest_sha256
    prior=json.loads((cache/'manifest.json').read_text())
    assert prior['status']=='incomplete_resource_drain'
    assert prior.get('all_workers_exited')is True and not prior.get('failures') and not retained_failures(cache)
    assert prior['protocol_sha256']==protocol_sha256
    checked_paths(cache,prior)
    receipts={path.parent.name for path in (cache/'cases').glob('case_*/complete.json')}
    assert receipts=={f'case_{int(cid):04d}'for cid in prior['completed_cases']}
    for cid,h in prior['completed_cases'].items():
        assert digest(cache/'cases'/f'case_{int(cid):04d}'/'complete.json')==h
        verify_receipt(cache,int(cid),protocol_sha256,protocol)
    return prior


def full_worker(cache_name: str,phase_name: str,replica: int,p: dict,p_sha: str,
                preflight_name: str,preflight_sha: str,remaining: list[int],initial_swapouts: int) -> None:
    """A persistent replica writes only new phase outputs and drains between calls."""
    cache,phase,preflight=Path(cache_name),Path(phase_name),Path(preflight_name)
    dest=phase/f'worker_{replica}';dest.mkdir();tick=time.monotonic()
    record={'status':'initializing','pid':os.getpid(),'replica':replica,'order':remaining,'cases':[],
        'protocol_sha256':p_sha,'input_preflight_sha256':preflight_sha,'environment':environment()}
    try:
        verify_protocol(p,p_sha);assert digest(preflight/'manifest.json')==preflight_sha
        inputs={r['case_id']:r for r in json.loads((preflight/'manifest.json').read_text())['cases']}
        original_call=river_module.linprog
        for cid in remaining:
            current=resources()
            if ((cache/'STOP_AFTER_ACTIVE').exists()or current['free_percent']<p['resource_drain_free_percent']
                or current['pressure_derived_free_bytes']<p['minimum_pressure_derived_headroom_bytes'] or current['swapouts']>initial_swapouts):
                record.update(status='drained',drain_resources=current);break
            source=preflight/'inputs'/f'case_{cid:04d}'
            for name,h in inputs[cid]['files'].items():assert digest(source/name)==h
            case,split=arrays(source/'A_inputs.npz'),arrays(source/'split_inputs.npz')
            oldnet=load_network(json.loads((source/'A_network.json').read_text()));newnet=load_network(json.loads((source/'split_network.json').read_text()))
            reference_path=ROOT/p['continuous_cache']/f'case_{cid:04d}_result.npz';assert digest(reference_path)==inputs[cid]['reference_sha256']
            folder=dest/f'case_{cid:04d}';folder.mkdir();started=time.monotonic()
            cr={'case_id':cid,'pid':os.getpid(),'status':'initializing','protocol_sha256':p_sha,
                'input_preflight_sha256':preflight_sha,'input_record':inputs[cid],'resources':current}
            def retain(*args,**kwargs):
                native=original_call(*args,**kwargs);saved={}
                if native.x is not None:saved['x']=native.x
                if native.fun is not None:saved['objective_scaled']=np.asarray(native.fun)
                for name in ['eqlin','ineqlin','lower','upper']:
                    for field in ['marginals','residual']:
                        value=getattr(getattr(native,name,None),field,None)
                        if value is not None:saved[name+'_'+field]=np.asarray(value)
                np.savez_compressed(folder/'native_output.npz',**saved);cr['native']={'status':int(native.status),'message':str(native.message)}
                return native
            try:
                with patch.object(river_module,'linprog',retain):result,saved,proof=solve(newnet,split,'C',20.)
                np.savez_compressed(folder/'result.npz',**result);np.savez_compressed(folder/'primal_dual.npz',**saved)
                assert all_slacks_exact_zero(result)
                values=compute_features(case,split,result,arrays(reference_path),oldnet,newnet);np.save(folder/'features.npy',values)
                cr.update(status='complete',kkt=proof,all_slacks_zero=True)
                rss=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin'else 1024)
                cr['peak_rss_bytes']=rss;assert rss<p['worker_rss_limit_bytes']
            except Exception:cr.update(status='failed',error=traceback.format_exc());raise
            finally:
                cr['runtime_seconds']=time.monotonic()-started;cr['artifacts']={f.name:digest(f)for f in folder.iterdir()if f.is_file()}
                write_json(folder/'proof.json',cr,True);record['cases'].append(cr)
            with (dest/'progress.jsonl').open('a')as f:f.write(json.dumps({'case_id':cid,'completed':len(record['cases']),'seconds':time.monotonic()-tick})+'\n')
        else:record['status']='complete'
    except Exception:record.update(status='failed',error=traceback.format_exc())
    finally:
        record['runtime_seconds']=time.monotonic()-tick;write_json(dest/'manifest.json',record,True)
    if record['status']=='failed':raise RuntimeError(record['error'])


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--freeze-protocol',action='store_true');parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--protocol-sha256');parser.add_argument('--run-id');parser.add_argument('--input-preflight',type=Path)
    parser.add_argument('--input-preflight-sha256');parser.add_argument('--independent-audit',type=Path);parser.add_argument('--independent-audit-sha256')
    parser.add_argument('--resume-manifest-sha256');args=parser.parse_args()
    if args.freeze_protocol:freeze();return
    assert args.run_id and Path(args.run_id).name==args.run_id and args.run_id not in ['.','..']
    cache=ROOT/'artifacts'/args.run_id;start=time.monotonic();workers=[];complete={};phase=None
    if args.resume_manifest_sha256:
        assert not args.preflight
        prior=verify_resume(cache,args.resume_manifest_sha256,args.protocol_sha256,json.loads(PROTOCOL.read_text()))
        prior_path=cache/f'prior_manifest_{args.resume_manifest_sha256[:12]}.json'
        assert not prior_path.exists();shutil.copy2(cache/'manifest.json',prior_path)
    else:cache.mkdir(exist_ok=False)
    report={'status':'initializing','timestamp_utc':datetime.now(timezone.utc).isoformat(),'command':shlex.join([sys.executable,*sys.argv]),
        'protocol_sha256':args.protocol_sha256,'labels_decoded':False,'training':False,'completed_cases':{},'failures':[],
        'all_workers_exited':True,'resumed_manifest_sha256':args.resume_manifest_sha256}
    try:
        p=json.loads(PROTOCOL.read_text());verify_protocol(p,args.protocol_sha256)
        report.update({k:p[k]for k in ['source','data','case_ids','case_dates','prefix_columns','feature_specification','replay_execution_type','environment']})
        if not args.resume_manifest_sha256:
            shutil.copy2(PROTOCOL,cache/'protocol.json')
            for name in p['source']:
                target=cache/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
        if args.preflight:
            report['cases']=prepare_inputs(cache,p);report.update(status='passed',optimization_calls=0)
        else:
            assert digest(args.input_preflight/'manifest.json')==args.input_preflight_sha256
            pre=json.loads((args.input_preflight/'manifest.json').read_text());assert pre['status']=='passed'
            compatibility=p['compatible_input_preflight']
            if pre['protocol_sha256']!=args.protocol_sha256:
                assert args.input_preflight_sha256==compatibility['manifest_sha256'] and pre['protocol_sha256']==compatibility['protocol_sha256']
                assert args.input_preflight.resolve()==(ROOT/compatibility['manifest']).parent.resolve()
            for field in ['data','case_ids','case_dates','feature_specification','environment']:assert pre[field]==p[field]
            for name,h in pre['source'].items():assert p['source'][name]==h
            assert [r['case_id']for r in pre['cases']]==p['case_ids']
            assert digest(args.independent_audit)==args.independent_audit_sha256
            audit=json.loads(args.independent_audit.read_text());assert audit['status']=='passed'and audit['protocol_sha256']==args.protocol_sha256
            assert audit['input_preflight_manifest_sha256']==args.input_preflight_sha256
            report.update(input_preflight=str(args.input_preflight.relative_to(ROOT)),input_preflight_manifest_sha256=args.input_preflight_sha256,
                independent_source_audit_sha256=args.independent_audit_sha256)
            inputs={r['case_id']:r for r in pre['cases']}
            for cid in p['case_ids']:
                if (cache/'cases'/f'case_{cid:04d}'/'complete.json').exists():
                    complete[cid]=verify_receipt(cache,cid,args.protocol_sha256,p,inputs[cid])
            pilot=ROOT/p['pilot_directory'];pm=json.loads((pilot/'manifest.json').read_text())
            for cid in p['pilot_case_ids']:
                if cid in complete:continue
                paths=[]
                for replica in [1,2]:
                    assert digest(pilot/f'worker_{replica}/manifest.json')==p['pilot_worker_manifests_sha256'][replica-1]
                    source=pilot/f'worker_{replica}'/f'case_{cid:04d}';dest=cache/'imports'/f'worker_{replica}'/f'case_{cid:04d}'
                    for f in source.iterdir():assert digest(f)==pm['artifacts'][str(f.relative_to(pilot))]
                    if dest.exists():
                        assert {f.name:digest(f)for f in dest.iterdir()}=={f.name:digest(f)for f in source.iterdir()}
                    else:shutil.copytree(source,dest)
                    paths.append(dest)
                complete[cid]=publish_receipt(cache,cid,paths,args.protocol_sha256,inputs[cid],
                    {'type':'pilot_import','pilot_manifest_sha256':p['pilot_manifest_sha256'],
                     'worker_manifests_sha256':p['pilot_worker_manifests_sha256'],'canonical_fresh_process_lineage':cid in p['canonical_case_ids']})
            report['imported_case_count']=len(p['pilot_case_ids']);report['initial_complete_case_count']=len(complete)
            current=resources();report['initial_resources']=current
            verify_entry_resources(current,p)
            current_again=resources();verify_entry_resources(current_again,p,current)
            report['entry_resource_recheck']=current_again
            if (cache/'STOP_AFTER_ACTIVE').exists():
                assert args.resume_manifest_sha256
                os.replace(cache/'STOP_AFTER_ACTIVE',cache/f'prior_stop_{args.resume_manifest_sha256[:12]}')
            phase_index=1
            while (cache/'phases'/f'phase_{phase_index:03d}').exists():phase_index+=1
            phase=cache/'phases'/f'phase_{phase_index:03d}';phase.mkdir(parents=True)
            remaining=[[cid for cid in order if cid not in complete]for order in p['orders']]
            ctx=mp.get_context('spawn');workers=[ctx.Process(target=full_worker,args=(str(cache),str(phase),i,p,args.protocol_sha256,
                str(args.input_preflight),args.input_preflight_sha256,remaining[i-1],current['swapouts']))for i in [1,2]]
            report['all_workers_exited']=False
            for worker in workers:worker.start()
            assert workers[0].pid!=workers[1].pid;report['phase_worker_pids']=[w.pid for w in workers]
            write_json(phase/'launch.json',{'protocol_sha256':args.protocol_sha256,'worker_pids':report['phase_worker_pids'],
                'input_preflight_sha256':args.input_preflight_sha256,'orders':remaining,'initial_complete_cases':sorted(complete)},True)
            origin={'type':'phase','phase':phase_index,'input_preflight_sha256':args.input_preflight_sha256}
            while True:
                publish_ready_pairs(cache,phase,remaining[0],complete,args.protocol_sha256,inputs,origin)
                alive=[w.is_alive()for w in workers]
                current_resource=resources()
                if (current_resource['free_percent']<p['resource_drain_free_percent']
                    or current_resource['pressure_derived_free_bytes']<p['minimum_pressure_derived_headroom_bytes']
                    or current_resource['swapouts']>current['swapouts'] or retained_failures(cache)):
                    (cache/'STOP_AFTER_ACTIVE').touch(exist_ok=True)
                event={'event':'progress','completed_cases':len(complete),'initial_complete_cases':report['initial_complete_case_count'],
                    'fresh_completed':len(complete)-report['initial_complete_case_count'],'seconds':time.monotonic()-start,'alive':alive,'resources':current_resource}
                with (cache/'progress.jsonl').open('a')as f:f.write(json.dumps(event)+'\n')
                print(json.dumps(event),flush=True)
                if not any(alive):break
                for worker in workers:worker.join(timeout=10)
            for worker in workers:worker.join()
            report['all_workers_exited']=True;report['worker_exitcodes']=[w.exitcode for w in workers]
            publish_ready_pairs(cache,phase,remaining[0],complete,args.protocol_sha256,inputs,origin)
            wm=[json.loads((phase/f'worker_{i}'/'manifest.json').read_text())for i in [1,2]]
            report['failures']=retained_failures(cache)+[{'replica':i+1,'error':m.get('error')}for i,m in enumerate(wm)if m['status']=='failed']
            assert all(m['pid']==workers[i].pid and m['order']==remaining[i] and m['protocol_sha256']==args.protocol_sha256
                and m['environment']==p['environment']for i,m in enumerate(wm))
            for cid in complete:report['completed_cases'][str(cid)]=digest(cache/'cases'/f'case_{cid:04d}'/'complete.json')
            if len(complete)==3647 and not report['failures']:
                features=np.stack([np.load(cache/complete[cid]['features_path'],allow_pickle=False)for cid in p['case_ids']])
                assert features.shape==(3647,168,62)and features.dtype==np.float32 and np.isfinite(features).all()
                np.savez_compressed(cache/'features.npz',features=features,case_ids=np.array(p['case_ids']),columns=np.array(COLUMNS),units=np.array(UNIT_ORDER))
                report.update(status='complete',features_sha256=digest(cache/'features.npz'),all_cases_physical_and_replay_verified=True)
            else:report['status']='failed'if report['failures']else 'incomplete_resource_drain'
    except BaseException:
        report.update(status='incomplete_operational',error=traceback.format_exc())
    finally:
        if workers:
            try:
                report['shutdown']=drain_workers(cache,workers);report['all_workers_exited']=report['shutdown']['all_workers_exited']
            except BaseException:
                report.update(all_workers_exited=False,shutdown_error=traceback.format_exc())
        if not report['all_workers_exited']:
            report.update(status='blocked_active_children',artifacts={})
        else:
            report['failures']=report.get('failures',[])+[f for f in retained_failures(cache)if f not in report.get('failures',[])]
            if report['failures']:report['status']='failed'
            report['completed_cases']={str(int(path.parent.name.split('_')[1])):digest(path)
                for path in sorted((cache/'cases').glob('case_*/complete.json'))}
            report['artifacts']={str(f.relative_to(cache)):digest(f)for f in cache.rglob('*')
                if f.is_file()and f not in [cache/'manifest.json',cache/'experiment_record.md']}
        report['runtime_seconds']=time.monotonic()-start
        write_json(cache/'manifest.json',report)
        (cache/'experiment_record.md').write_text('\n## '+args.run_id+' — '+report['timestamp_utc']+'\n\nStatus '+report['status']+
            '; C-only river62 all-input cache, no labels/training. Source/data/execution/receipt/feature proof in `'+str((cache/'manifest.json').relative_to(ROOT))+'`, SHA `'+digest(cache/'manifest.json')+'`.\n')
    print(json.dumps({'status':report['status'],'manifest_sha256':digest(cache/'manifest.json')}),flush=True)
    if report['status']not in ['passed','complete']:raise SystemExit(1)


if __name__=='__main__':main()
