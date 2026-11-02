"""Input-only order/history replay pilot with two persistent single-thread processes."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import resource
import shlex
import shutil
import subprocess
import sys
import time
import traceback
from unittest.mock import patch
for variable in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[variable]='1'
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sintef_project.lp_features import HydroCaseData,UNIT_ORDER
from sintef_project.lp_hyljelihyl_v2 import split_network,split_case,make_problem,solve
import sintef_project.lp_hyljelihyl_v2 as river_module
from sintef_project.hyljelihyl_river_features import compute_features,specification
from scripts.diagnose_hyljelihyl_river_cost_36 import json_network,load_network,signatures
from scripts.diagnose_integer_water_budget import arrays,digest
from scripts.probe_all14_integer_water_budget_v2 import identical_arrays,all_slacks_exact_zero
from scripts.pilot_integer_water_budget import environment
from scripts.continue_integer_water_budget_cache_v2 import resource_state

PROTOCOL=ROOT/'docs/hyljelihyl_river_persistent_pilot_protocol.json'
NEW=['scripts/pilot_hyljelihyl_river_persistent.py','sintef_project/hyljelihyl_river_features.py',
     'tests/test_hyljelihyl_river_features.py','tests/test_hyljelihyl_persistent_pilot.py',
     'docs/hyljelihyl_river_persistent_pilot.md']


def population(total: int, canonical_positions: list[int]) -> list[int]:
    """Union exact canonical positions with24 deterministic evenly spaced positions."""
    if total<24:raise ValueError('Too few cases')
    return sorted(set(canonical_positions)|set(np.rint(np.linspace(0,total-1,24)).astype(int).tolist()))


def orders(ids: list[int]) -> list[list[int]]:
    """Distinct persistent workers receive ascending and fixed-seed permutation orders."""
    return [list(ids),np.random.default_rng(4173).permutation(ids).tolist()]


def resources() -> dict:
    """Measured pressure-derived free memory, without a large probe allocation."""
    result=resource_state()
    text=subprocess.run(['memory_pressure','-Q'],check=True,text=True,capture_output=True).stdout
    match=re.search(r'The system has (\d+)',text)
    if match is None:raise ValueError('Physical memory telemetry missing')
    result['physical_bytes']=int(match.group(1))
    result['pressure_derived_free_bytes']=result['physical_bytes']*result['free_percent']//100
    return result


def freeze() -> None:
    """Pin the full result audit before preparing execution/feature evidence."""
    old_path=ROOT/'docs/hyljelihyl_river_cost_36_protocol.json';old=json.loads(old_path.read_text())
    cache=json.loads((ROOT/old['cache']/'manifest.json').read_text())
    positions=population(len(cache['case_ids']),[c['metadata_position']for c in old['cases']])
    cases=[{'case_id':cache['case_ids'][i],'metadata_position':i,'date':cache['case_dates'][i]}for i in positions]
    p={'version':'hyljelihyl_persistent_worker_order_replay_v1','created_utc':datetime.now(timezone.utc).isoformat(),
       'cases':cases,'canonical_case_ids':[c['case_id']for c in old['cases']],
       'orders':orders([c['case_id']for c in cases]),'selection':'Exact canonical36 plus24 rounded linspace metadata positions over3647; sorted union; no labels.',
       'source':{**old['source'],**{n:digest(ROOT/n)for n in NEW}},'data':old['data'],'inputs':{},
       'cache':old['cache'],'cache_manifest_sha256':old['cache_manifest_sha256'],
       'canonical_directory':'artifacts/hyljelihyl-river-cost-36-diagnostic-001',
       'canonical_manifest_sha256':'f403ed698047e6e47e94160fac245b019e6e286f8d9256014187bf6d1d29b4e7',
       'canonical_audit':'artifacts/hyljelihyl-river-cost-36-result-independent-audit-001/manifest.json',
       'canonical_audit_sha256':'0be8e29f96f08fe25c20a5b69182f1efff2a289e6a60ebafe715c9f13b14b85f',
       'feature_specification':specification(),'environment':environment(),'lp_seconds':20.,'workers':2,
       'resource_minimum_free_percent':30,'minimum_pressure_derived_headroom_bytes':3*1024**3,
       'worker_rss_limit_bytes':2*1024**3,'labels_decoded':False,'training':False,
       'replay_execution_type':'Two separately spawned persistent processes, single solver thread each, fresh model/solver per call; ascending versus seed4173 permutation. Not fresh process per case.',
       'gate':'Every selected case optimal/zero absolute slack/full KKT<=1e-5; entire result and primal/dual arrays literal-byte equal across workers; all36 agree with canonical fresh-process results and primal/dual arrays; features finite and literal-byte equal; no target decoding.',
       'scope':'Timing/replay/schema pilot only, not a full feature cache. No retry, fallback, fitting or automatic promotion.'}
    for path in [old_path,ROOT/p['canonical_directory']/'manifest.json',ROOT/p['canonical_audit']]:
        p['inputs'][str(path.relative_to(ROOT))]=digest(path)
    cm=json.loads((ROOT/p['canonical_directory']/'manifest.json').read_text())
    for c in cases:
        cid=c['case_id']
        for kind in ['inputs','result']:
            name=p['cache']+f'/case_{cid:04d}_{kind}.npz';assert digest(ROOT/name)==cache['artifacts'][Path(name).name]
            p['inputs'][name]=digest(ROOT/name)
        if cid in p['canonical_case_ids']:
            for name in ['A_inputs.npz','split_inputs.npz','A_network.json','split_network.json','C_1/result.npz','C_1/primal_dual.npz','C_1/proof.json']:
                relative=f'case_{cid:04d}/'+name;path=ROOT/p['canonical_directory']/relative
                assert digest(path)==cm['artifacts'][relative];p['inputs'][str(path.relative_to(ROOT))]=digest(path)
    with PROTOCOL.open('x')as f:json.dump(p,f,indent=2)
    print(digest(PROTOCOL),flush=True)


def persistent_worker(root_name: str, replica: int, protocol: dict, protocol_sha256: str,
                      initial_swapouts: int, input_hashes: dict, input_seal_sha256: str) -> None:
    """One process executes its entire prescribed order; save raw output before checks."""
    out=Path(root_name);dest=out/f'worker_{replica}';dest.mkdir(exist_ok=False)
    started=time.monotonic();record={'status':'initializing','pid':os.getpid(),'replica':replica,
        'order':protocol['orders'][replica-1],'environment':environment(),'cases':[]}
    original_call=river_module.linprog
    try:
        assert digest(PROTOCOL)==protocol_sha256 and json.loads(PROTOCOL.read_text())==protocol
        assert digest(out/'prepared_inputs.json')==input_seal_sha256
        assert json.loads((out/'prepared_inputs.json').read_text())['inputs']==input_hashes
        assert environment()==protocol['environment']
        for name,h in protocol['source'].items():assert digest(ROOT/name)==h,name
        for cid in record['order']:
            current=resources()
            assert current['free_percent']>=30 and current['swapouts']<=initial_swapouts
            assert current['pressure_derived_free_bytes']>=protocol['minimum_pressure_derived_headroom_bytes']
            source=out/'inputs'/f'case_{cid:04d}';folder=dest/f'case_{cid:04d}';folder.mkdir()
            for name,h in input_hashes[str(cid)].items():assert digest(source/name)==h,name
            case=arrays(source/'A_inputs.npz');split=arrays(source/'split_inputs.npz')
            oldnet=load_network(json.loads((source/'A_network.json').read_text()))
            newnet=load_network(json.loads((source/'split_network.json').read_text()))
            cr={'case_id':cid,'pid':os.getpid(),'status':'initializing','resources':current};tick=time.monotonic()
            def retain(*args,**kwargs):
                native=original_call(*args,**kwargs);saved={}
                if native.x is not None:saved['x']=native.x
                if native.fun is not None:saved['objective_scaled']=np.asarray(native.fun)
                for name in ['eqlin','ineqlin','lower','upper']:
                    for field in ['marginals','residual']:
                        value=getattr(getattr(native,name,None),field,None)
                        if value is not None:saved[name+'_'+field]=np.asarray(value)
                np.savez_compressed(folder/'native_output.npz',**saved)
                cr['native']={'status':int(native.status),'message':str(native.message)}
                return native
            try:
                with patch.object(river_module,'linprog',retain):result,saved,proof=solve(newnet,split,'C',20.)
                np.savez_compressed(folder/'result.npz',**result);np.savez_compressed(folder/'primal_dual.npz',**saved)
                assert all_slacks_exact_zero(result)
                reference_path=ROOT/protocol['cache']/f'case_{cid:04d}_result.npz'
                assert digest(reference_path)==protocol['inputs'][str(reference_path.relative_to(ROOT))]
                reference=arrays(reference_path)
                features=compute_features(case,split,result,reference,oldnet,newnet)
                np.save(folder/'features.npy',features)
                cr.update(status='complete',kkt=proof,all_slacks_zero=True)
            except Exception:
                cr.update(status='failed',error=traceback.format_exc());raise
            finally:
                cr['runtime_seconds']=time.monotonic()-tick
                cr['artifacts']={x.name:digest(x)for x in folder.iterdir()if x.is_file()}
                (folder/'proof.json').write_text(json.dumps(cr,indent=2));record['cases'].append(cr)
            with (dest/'progress.jsonl').open('a')as f:f.write(json.dumps({'case_id':cid,'completed':len(record['cases']),'seconds':time.monotonic()-started})+'\n')
        record['status']='complete'
    except Exception:record.update(status='failed',error=traceback.format_exc())
    finally:
        record.update(runtime_seconds=time.monotonic()-started,
            peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024))
        (dest/'manifest.json').write_text(json.dumps(record,indent=2))
    if record['status']!='complete':raise RuntimeError(record['error'])


def main() -> None:
    """Metadata-only preflight or externally approved label-free execution."""
    parser=argparse.ArgumentParser();parser.add_argument('--freeze-protocol',action='store_true');parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--protocol-sha256');parser.add_argument('--independent-audit',type=Path);parser.add_argument('--independent-audit-sha256');parser.add_argument('--run-id')
    args=parser.parse_args()
    if args.freeze_protocol:freeze();return
    assert args.run_id and Path(args.run_id).name==args.run_id and args.run_id not in ['.','..']
    out=ROOT/'artifacts'/args.run_id;out.mkdir(exist_ok=False);start=time.monotonic()
    report={'status':'initializing','timestamp_utc':datetime.now(timezone.utc).isoformat(),'command':shlex.join([sys.executable,*sys.argv]),
        'protocol_sha256':args.protocol_sha256,'labels_decoded':False,'training':False,'model_cache_eligible':False,'cases':[]}
    try:
        assert digest(PROTOCOL)==args.protocol_sha256;p=json.loads(PROTOCOL.read_text());assert p['environment']==environment()
        for section in ['source','data','inputs']:
            for name,h in p[section].items():assert digest(ROOT/name)==h,name
        assert digest(ROOT/p['cache']/'manifest.json')==p['cache_manifest_sha256']
        assert digest(ROOT/p['canonical_directory']/'manifest.json')==p['canonical_manifest_sha256']
        canonical_manifest=json.loads((ROOT/p['canonical_directory']/'manifest.json').read_text())
        audit=json.loads((ROOT/p['canonical_audit']).read_text());assert digest(ROOT/p['canonical_audit'])==p['canonical_audit_sha256']
        assert audit['status']=='passed' and audit['result_manifest_sha256']==p['canonical_manifest_sha256']
        if not args.preflight:
            assert digest(args.independent_audit)==args.independent_audit_sha256
            a=json.loads(args.independent_audit.read_text());assert a['status']=='passed'and a['protocol_sha256']==args.protocol_sha256
            report['independent_audit_sha256']=args.independent_audit_sha256
        shutil.copy2(PROTOCOL,out/'protocol.json')
        for name in p['source']:
            target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
        raw_read=pd.read_csv;label_path=(ROOT/'sintef_project/data/kernel/Unit_commitment_decisions.csv').resolve()
        def guarded(path,*positional,**kwargs):
            if isinstance(path,(str,Path))and Path(path).resolve()==label_path:
                assert not positional and kwargs=={'usecols':['Run No','starttime']},'Pilot may never decode targets'
            return raw_read(path,*positional,**kwargs)
        with patch.object(pd,'read_csv',guarded):
            data=HydroCaseData(ROOT);oldnet=data.network;newnet=split_network(oldnet)
            actual_positions=population(len(data.cases),[int(data.cases.index[data.cases['Run No']==i][0])for i in p['canonical_case_ids']])
            assert actual_positions==[c['metadata_position']for c in p['cases']]
            assert p['orders']==orders([c['case_id']for c in p['cases']])
            for selected in p['cases']:
                cid,index=selected['case_id'],selected['metadata_position'];date=pd.Timestamp(selected['date'])
                assert int(data.cases.iloc[index]['Run No'])==cid and data.cases.iloc[index].starttime==date
                case=data.case(index);case['natural']=np.asarray(case['natural'],dtype=np.float64)
                for key,field in [('pmin','p_min'),('start_cost','startcost_const'),('stop_cost','stopcost_const')]:
                    case[key]=np.array([data.model['generator'][u][field]for u in UNIT_ORDER],dtype=np.float64)
                assert identical_arrays(case,arrays(ROOT/p['cache']/f'case_{cid:04d}_inputs.npz'))
                times=pd.date_range(date,periods=168,freq='h');water=data.water.loc[date+pd.Timedelta(days=7),['Vaamarvatn','Totak','Venemo','Hyljelihyl']].to_numpy()
                assert np.isfinite(water).all()and np.all(water==water[0])
                minimum=data.minimum.loc[times,'Hyljelihyl'].to_numpy()if'Hyljelihyl'in data.minimum else np.zeros(168)
                split=split_case(case,float(data.volume.loc[date,'Hyljelihyl']),data.inflow.loc[times,'Hyljelihyl'].to_numpy(dtype=np.float64)*.0036,minimum)
                folder=out/'inputs'/f'case_{cid:04d}';folder.mkdir(parents=True)
                np.savez_compressed(folder/'A_inputs.npz',**case);np.savez_compressed(folder/'split_inputs.npz',**split)
                for name,net in [('A',oldnet),('split',newnet)]: (folder/f'{name}_network.json').write_text(json.dumps(json_network(net),indent=2))
                _,problem,_=make_problem(newnet,split,'C',20.)
                record={'case_id':cid,'date':str(date),'fresh_input_byte_exact':True,'local_terminal_values_equal':True,'matrix_signatures':signatures(problem)}
                if cid in p['canonical_case_ids']:
                    canonical=ROOT/p['canonical_directory']/f'case_{cid:04d}'
                    assert identical_arrays(case,arrays(canonical/'A_inputs.npz'))and identical_arrays(split,arrays(canonical/'split_inputs.npz'))
                    old_record=next(r for r in canonical_manifest['cases']if r['case_id']==cid)
                    assert record['matrix_signatures']==old_record['matrix_signatures']['C']
                    values=compute_features(case,split,arrays(canonical/'C_1/result.npz'),arrays(ROOT/p['cache']/f'case_{cid:04d}_result.npz'),oldnet,newnet)
                    np.save(folder/'canonical_features.npy',values);record['canonical_feature_sha256']=digest(folder/'canonical_features.npy')
                report['cases'].append(record)
        input_hashes={str(c['case_id']):{f.name:digest(f)for f in (out/'inputs'/f'case_{c["case_id"]:04d}').iterdir()if f.is_file()}for c in p['cases']}
        report['prepared_input_hashes']=input_hashes
        (out/'prepared_inputs.json').write_text(json.dumps({'protocol_sha256':args.protocol_sha256,'inputs':input_hashes},indent=2))
        report['prepared_input_seal_sha256']=digest(out/'prepared_inputs.json')
        if args.preflight:report.update(status='passed',optimization_calls=0)
        else:
            current=resources();report['initial_resources']=current
            assert current['free_percent']>=30 and current['pressure_derived_free_bytes']>=p['minimum_pressure_derived_headroom_bytes']
            ctx=mp.get_context('spawn');workers=[ctx.Process(target=persistent_worker,args=(str(out),i,p,args.protocol_sha256,current['swapouts'],input_hashes,report['prepared_input_seal_sha256']))for i in [1,2]]
            for worker in workers:worker.start()
            report['worker_pids']=[w.pid for w in workers];assert len(set(report['worker_pids']))==2
            while any(w.is_alive()for w in workers):
                for worker in workers:worker.join(timeout=10)
                print(json.dumps({'event':'heartbeat','alive':[w.is_alive()for w in workers],'seconds':time.monotonic()-start}),flush=True)
            assert all(w.exitcode==0 for w in workers)
            manifests=[json.loads((out/f'worker_{i}'/'manifest.json').read_text())for i in [1,2]]
            for i,m in enumerate(manifests):
                assert m['status']=='complete'and m['pid']==report['worker_pids'][i]and m['order']==p['orders'][i]
                assert m['environment']==p['environment']
                assert [c['case_id']for c in m['cases']]==m['order']and 0<m['peak_rss_bytes']<p['worker_rss_limit_bytes']
            for c in p['cases']:
                cid=c['case_id'];one=out/'worker_1'/f'case_{cid:04d}';two=out/'worker_2'/f'case_{cid:04d}'
                for name in ['result.npz','primal_dual.npz']:
                    assert identical_arrays(arrays(one/name),arrays(two/name))
                    if cid in p['canonical_case_ids']:assert identical_arrays(arrays(one/name),arrays(ROOT/p['canonical_directory']/f'case_{cid:04d}/C_1'/name))
                a,b=np.load(one/'features.npy',allow_pickle=False),np.load(two/'features.npy',allow_pickle=False)
                assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
                if cid in p['canonical_case_ids']:assert digest(one/'features.npy')==digest(out/'inputs'/f'case_{cid:04d}'/'canonical_features.npy')
            report.update(status='passed',all_worker_replays_byte_exact=True,canonical_fresh_process_agreement=True,
                feature_replays_byte_exact=True,worker_manifests_sha256=[digest(out/f'worker_{i}'/'manifest.json')for i in [1,2]])
    except Exception:report.update(status='failed',error=traceback.format_exc());raise
    finally:
        report['runtime_seconds']=time.monotonic()-start
        report['artifacts']={str(f.relative_to(out)):digest(f)for f in out.rglob('*')if f.is_file()and f.name!='manifest.json'}
        (out/'manifest.json').write_text(json.dumps(report,indent=2))
        (out/'experiment_record.md').write_text('\n## '+args.run_id+' — '+report['timestamp_utc']+'\n\nStatus '+report['status']+
            '; label-free persistent-worker/62-feature pilot, not model-eligible cache. Full command/environment/source/data/replay evidence: `'+str((out/'manifest.json').relative_to(ROOT))+'`, SHA `'+digest(out/'manifest.json')+'`.\n')
    print(json.dumps({'status':report['status'],'manifest_sha256':digest(out/'manifest.json')}),flush=True)


if __name__=='__main__':main()
