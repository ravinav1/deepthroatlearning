"""Immutable, resumable, label-free full integer water-budget cache builder."""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
import resource
import shlex
import shutil
import sys
import time
import traceback
import uuid
from unittest.mock import patch

for variable in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[variable]='1'

import numpy as np
import pandas as pd

SOURCE_ROOT=Path(__file__).resolve().parents[1]
ROOT=Path(os.environ.get('TDT4173_WORKSPACE_ROOT',SOURCE_ROOT)).resolve()
sys.path.insert(0,str(SOURCE_ROOT))
from scripts.pilot_integer_water_budget import capture_solve,environment
from scripts.diagnose_integer_water_budget import arrays,digest,TARGETS
from sintef_project.integer_water_budget_features import compute_features,feature_specification,COLUMNS
from sintef_project.lp_features import HydroCaseData,UNIT_ORDER

EXTRA_SOURCE=['scripts/build_integer_water_budget_cache.py','docs/integer_water_budget_full_cache.md',
              'sintef_project/integer_water_budget_features.py','tests/test_integer_water_budget_features.py',
              'tests/test_integer_water_budget_full_cache.py']


def atomic_json(path:Path,value:dict)->None:
    """Publish a complete aggregate checkpoint without a partially written JSON."""
    temporary=path.with_name(path.name+'.temporary')
    with temporary.open('w') as stream:
        json.dump(value,stream,indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary,path)


def checked_copy(source:Path,destination:Path,expected:str)->None:
    """Preserve immutable evidence exactly, refusing a checksum mismatch."""
    if digest(source)!=expected:
        raise ValueError(f'Input changed: {source}')
    destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,destination)
    if digest(destination)!=expected:
        raise ValueError('Evidence copy changed bytes')


def same_arrays(first:dict,second:dict)->None:
    """Reject alternate optima or any raw numerical replay difference."""
    if set(first)!=set(second):
        raise ValueError('Replay field set differs')
    for key in first:
        a,b=np.asarray(first[key]),np.asarray(second[key])
        if a.shape!=b.shape or a.dtype!=b.dtype or a.tobytes()!=b.tobytes():
            raise AssertionError(f'Bitwise replay field differs: {key}')


def retry_allowed(record:dict,seconds:float)->bool:
    """Only an original time/iteration-limit status permits a 600-second retry."""
    return seconds==60. and record.get('proof',{}).get('mip_status')==1


def eligible_attempt(record:dict)->bool:
    """Require proven physical feasibility and fully saved conditional replay."""
    p=record.get('proof',{})
    return bool(record.get('status')=='complete' and p.get('mip_status')==0 and p.get('proven_optimal')
                and p.get('fixed_lp_replay_bitwise') and record.get('fixed_lp_all_arrays_bitwise')
                and record.get('all_slacks_zero'))


def freeze(path:Path)->None:
    """Freeze formulas, source, data, all metadata and exactly reusable evidence."""
    pilot=ROOT/'artifacts/integer-water-budget-timing-pilot-001'
    pm=json.loads((pilot/'manifest.json').read_text())
    pp=json.loads((pilot/'protocol.json').read_text())
    completion=ROOT/'artifacts/integer-water-budget-completion-001'
    dm=json.loads((completion/'manifest.json').read_text())
    cm=json.loads((ROOT/pp['cache']/'manifest.json').read_text())
    sources=dict(pp['source'])
    sources.update({name:digest(SOURCE_ROOT/name) for name in EXTRA_SOURCE})
    audits=['artifacts/integer-water-budget-completion-independent-audit-001/manifest.json',
            'artifacts/integer-water-budget-pilot-independent-audit-001/manifest.json',
            'artifacts/integer-water-budget-feature-preflight-002/manifest.json']
    assert pm['all_attempts_proven'] and pm['all_case_replays_bitwise'] and pm['all_slacks_zero'] and dm['screen_passed']
    canonical={str(r['case_id']):{kind:{'path':str((completion/f'case_{r["case_id"]:04d}_{suffix}').relative_to(ROOT)),
                                           'sha256':digest(completion/f'case_{r["case_id"]:04d}_{suffix}')}
                                 for kind,suffix in [('result','integer.npz'),('proof','proof.json')]}
               for r in dm['cases']}
    pilot_files={name:sha for name,sha in pm['artifacts'].items() if name.startswith('case_')}
    p={'version':'integer_water_budget_full_cache_v1','created_utc':datetime.now(timezone.utc).isoformat(),
       'source':sources,'data':pp['data'],'environment':environment(),'targets':TARGETS,
       'feature_specification':feature_specification(),'case_ids':cm['case_ids'],'case_dates':cm['case_dates'],
       'continuous_cache':pp['cache'],'continuous_cache_manifest_sha256':digest(ROOT/pp['cache']/'manifest.json'),
       'prefix_protocol':'docs/cuda_xgb664_protocol.json','prefix_protocol_sha256':digest(ROOT/'docs/cuda_xgb664_protocol.json'),
       'prefix_columns':json.loads((ROOT/'docs/cuda_xgb664_protocol.json').read_text())['feature_columns'],
       'audits':{name:digest(ROOT/name) for name in audits},'canonical_diagnostic':canonical,
       'completion_manifest_sha256':digest(completion/'manifest.json'),
       'pilot_directory':str(pilot.relative_to(ROOT)),'pilot_manifest_sha256':digest(pilot/'manifest.json'),
       'pilot_protocol_sha256':digest(pilot/'protocol.json'),'pilot_case_ids':pm['case_ids'],'pilot_files':pilot_files,
       'workers':2,'solver_threads':1,'first_milp_seconds':60.,'retry_milp_seconds':600.,'conditional_lp_seconds_each':1.,
       'retry_policy':'Only native MILP status 1 after the 60-second attempt. One 600-second fresh retry per replica. Never substitute relaxed results.',
       'replay_policy':'Two fresh-process MILP replicas, both saved conditional LP outputs, all raw result arrays/scalars exactly equal.',
       'resume_policy':'Explicitly hashed same-protocol prior manifest; only complete, fully verified immutable case directories are imported.',
       'labels_decoded':False,'training':False}
    with path.open('x') as stream:
        json.dump(p,stream,indent=2)
    print(digest(path))


def validate_attempt(directory:Path)->dict:
    """Independently re-read both persisted conditional LP outputs and proof."""
    record=json.loads((directory/'proof.json').read_text())
    for name,sha in record['artifacts'].items():
        if digest(directory/name)!=sha:
            raise ValueError('Attempt payload checksum changed')
    if eligible_attempt(record):
        same_arrays(arrays(directory/'fixed_lp_1.npz'),arrays(directory/'fixed_lp_2.npz'))
    return record


def run_attempt(output:str,case_id:int,replica:int,seconds:float,network,cache_name:str,
                input_sha:str,control_sha:str,protocol_sha:str)->dict:
    """Execute one isolated attempt and atomically publish all success/failure data."""
    parent=Path(output)/'cases'/f'case_{case_id:04d}'
    parent.mkdir(parents=True,exist_ok=True)
    destination=parent/f'replica_{replica}_{int(seconds)}s'
    temporary=parent/f'.attempt_{uuid.uuid4().hex}'
    temporary.mkdir(exist_ok=False)
    started=time.monotonic()
    record={'case_id':case_id,'replica':replica,'milp_limit_seconds':seconds,'protocol_sha256':protocol_sha,
            'timestamp_utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),'status':'initializing',
            'input_sha256':input_sha,'control_sha256':control_sha,'proof':{},'environment':environment()}
    saved=[]
    try:
        cache=Path(cache_name)
        ip,cp=cache/f'case_{case_id:04d}_inputs.npz',cache/f'case_{case_id:04d}_result.npz'
        if digest(ip)!=input_sha or digest(cp)!=control_sha:
            raise ValueError('Worker physical input/control changed')
        case,control=arrays(ip),arrays(cp)
        result,proof,saved=capture_solve(network,case,[UNIT_ORDER.index(u) for u in TARGETS],seconds)
        record['proof']=proof
        np.savez_compressed(temporary/'result.npz',**result)
        if result['objective']>control['objective']+1e-4:
            raise ValueError('Integer result exceeds relaxed objective')
        if len(saved)!=2:
            raise ValueError('Both fixed-state LP outputs are required')
        same_arrays(saved[0],saved[1])
        scale=1+np.abs(case['price']).mean()
        record.update(status='complete',objective=float(result['objective']),
                      original_relaxed_objective=float(control['objective']),
                      native_absolute_gap_eur=float((proof['mip_objective_scaled']-proof['mip_dual_bound_scaled'])*scale),
                      all_slacks_zero=all(np.max(np.abs(result[k]))==0 for k in ['minimum_slack','maximum_slack','bypass_slack','vest_slack']),
                      fixed_lp_all_arrays_bitwise=True)
    except Exception as error:
        record.update(status='failed',error=traceback.format_exc())
        if hasattr(error,'saved_integer_proof'):
            record['proof']=error.saved_integer_proof
        saved=getattr(error,'saved_lp_calls',saved)
        np.savez_compressed(temporary/'failed_incumbent.npz',**getattr(error,'saved_integer_arrays',{}))
    finally:
        for i,fields in enumerate(saved,1):
            np.savez_compressed(temporary/f'fixed_lp_{i}.npz',**fields)
        record.update(runtime_seconds=time.monotonic()-started,
                      peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024))
        record['artifacts']={f.name:digest(f) for f in temporary.iterdir() if f.is_file()}
        (temporary/'proof.json').write_text(json.dumps(record,indent=2))
        for f in temporary.iterdir():
            with f.open('rb') as stream:
                os.fsync(stream.fileno())
        if destination.exists():
            raise ValueError('Immutable attempt destination already exists')
        os.rename(temporary,destination)
    return {'case_id':case_id,'replica':replica,'seconds':seconds,'directory':str(destination),'record':record}


def verify_complete_case(directory:Path,protocol_sha:str)->dict:
    """Reject incomplete or changed case directories during resume and assembly."""
    path=directory/'complete.json'
    m=json.loads(path.read_text())
    if m['status']!='complete' or m['protocol_sha256']!=protocol_sha:
        raise ValueError('Case is incomplete or uses a different protocol')
    for name,sha in m['artifacts'].items():
        if digest(directory/name)!=sha:
            raise ValueError('Complete case artifact changed')
    for name in m['accepted_attempts']:
        if not eligible_attempt(validate_attempt(directory/name)):
            raise ValueError('Completed case references an ineligible attempt')
    same_arrays(arrays(directory/m['accepted_attempts'][0]/'result.npz'),arrays(directory/m['accepted_attempts'][1]/'result.npz'))
    same_arrays(arrays(directory/'canonical_result.npz'),arrays(directory/m['accepted_attempts'][0]/'result.npz'))
    return m


def finish_case(out:Path,case_id:int,accepted:list[Path],p:dict,protocol_sha:str,network,cache:Path)->dict:
    """Publish an immutable accepted case only after two physical proofs agree."""
    directory=out/'cases'/f'case_{case_id:04d}'
    reports=[validate_attempt(path) for path in accepted]
    if not all(eligible_attempt(r) for r in reports):
        raise ValueError('Cannot publish an unproven case')
    first,second=[arrays(path/'result.npz') for path in accepted]
    same_arrays(first,second)
    canonical=p['canonical_diagnostic'].get(str(case_id))
    if canonical:
        checked_copy(ROOT/canonical['result']['path'],directory/'canonical_result.npz',canonical['result']['sha256'])
        checked_copy(ROOT/canonical['proof']['path'],directory/'canonical_proof.json',canonical['proof']['sha256'])
        same_arrays(arrays(directory/'canonical_result.npz'),first)
        origin={'kind':'preserved_diagnostic','files':canonical}
    else:
        checked_copy(accepted[0]/'result.npz',directory/'canonical_result.npz',digest(accepted[0]/'result.npz'))
        checked_copy(accepted[0]/'proof.json',directory/'canonical_proof.json',digest(accepted[0]/'proof.json'))
        origin={'kind':'first_accepted_replica','directory':accepted[0].name}
    ip,cp=cache/f'case_{case_id:04d}_inputs.npz',cache/f'case_{case_id:04d}_result.npz'
    checked_copy(ip,directory/'inputs.npz',reports[0]['input_sha256'])
    for r in reports:
        if r['case_id']!=case_id or r['input_sha256']!=digest(ip) or r['control_sha256']!=digest(cp):
            raise ValueError('Replica input/control identity mismatch')
    features=compute_features(arrays(ip),first,arrays(cp),network,reports[0]['proof'])
    np.save(directory/'features.npy',features,allow_pickle=False)
    m={'status':'complete','case_id':case_id,'protocol_sha256':protocol_sha,'canonical_origin':origin,
       'input_sha256':digest(ip),'control_sha256':digest(cp),'accepted_attempts':[x.name for x in accepted],
       'fresh_milp_result_replay_bitwise':True,'both_fixed_lp_outputs_verified':True,'all_slacks_zero':True,
       'artifacts':{f.relative_to(directory).as_posix():digest(f) for f in directory.rglob('*') if f.is_file()}}
    atomic_json(directory/'complete.json',m)
    return verify_complete_case(directory,protocol_sha)


def main()->None:
    """Build physics only; keep failures and require explicit complete-cache eligibility."""
    parser=argparse.ArgumentParser()
    parser.add_argument('--freeze-protocol',action='store_true')
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--protocol',type=Path,default=ROOT/'docs/integer_water_budget_full_cache_protocol.json')
    parser.add_argument('--protocol-sha256')
    parser.add_argument('--run-id')
    parser.add_argument('--reuse-cache',type=Path)
    parser.add_argument('--reuse-cache-sha256')
    args=parser.parse_args()
    if args.freeze_protocol:
        freeze(args.protocol)
        return
    if args.run_id is None or Path(args.run_id).name!=args.run_id or args.run_id in ['', '.', '..']:
        raise ValueError('Unique single-component run ID required')
    if args.protocol_sha256 is None or digest(args.protocol)!=args.protocol_sha256:
        raise ValueError('Explicit reviewed protocol hash required')
    p=json.loads(args.protocol.read_text())
    if p['environment']!=environment() or p['feature_specification']!=feature_specification():
        raise ValueError('Frozen solver environment or feature formulas changed')
    for group,root in [('source',SOURCE_ROOT),('data',ROOT),('audits',ROOT)]:
        for name,sha in p[group].items():
            if digest(root/name)!=sha:
                raise ValueError(f'Frozen {group} changed: {name}')
    for key,sha in [('prefix_protocol','prefix_protocol_sha256')]:
        if digest(ROOT/p[key])!=p[sha]:
            raise ValueError('Frozen 664 prefix changed')
    cache=ROOT/p['continuous_cache']
    if digest(cache/'manifest.json')!=p['continuous_cache_manifest_sha256']:
        raise ValueError('Continuous reference cache changed')
    cm=json.loads((cache/'manifest.json').read_text())
    pilot=ROOT/p['pilot_directory']
    if digest(pilot/'manifest.json')!=p['pilot_manifest_sha256'] or digest(pilot/'protocol.json')!=p['pilot_protocol_sha256']:
        raise ValueError('Audited pilot changed')
    out=ROOT/'artifacts'/args.run_id
    out.mkdir(exist_ok=False)
    started=time.monotonic()
    report={'run_id':args.run_id,'status':'initializing','timestamp_utc':datetime.now(timezone.utc).isoformat(),
            'command':shlex.join([sys.executable,*sys.argv]),'protocol_sha256':args.protocol_sha256,
            'environment':environment(),'source':p['source'],'data':p['data'],'labels_decoded':False,'training':False,
            'kaggle_submission':None,'case_ids':p['case_ids'],'case_dates':p['case_dates'],'feature_specification':feature_specification(),
            'prefix_columns':p['prefix_columns'],'completed_cases':{},'attempts':[],'failures':[],'resumed_case_ids':[]}
    def checkpoint(event:dict)->None:
        event.update(elapsed_seconds=time.monotonic()-started,completed_cases=len(report['completed_cases']))
        n=len(report['completed_cases'])
        event['eta_seconds_linear']=event['elapsed_seconds']*(3647-n)/max(1,n)
        with (out/'progress.jsonl').open('a') as stream:
            stream.write(json.dumps(event)+'\n');stream.flush();os.fsync(stream.fileno())
        print(json.dumps(event),flush=True)
        report['runtime_seconds']=event['elapsed_seconds']
        atomic_json(out/'manifest.json',report)
    try:
        checked_copy(args.protocol,out/'protocol.json',args.protocol_sha256)
        for name,sha in p['source'].items():
            checked_copy(SOURCE_ROOT/name,out/'source'/name,sha)
        label_path=(ROOT/'sintef_project/data/kernel/Unit_commitment_decisions.csv').resolve()
        original_read=pd.read_csv
        def guarded(path,*positional,**kwargs):
            if isinstance(path,(str,Path)) and Path(path).resolve()==label_path:
                if positional or kwargs!={'usecols':['Run No','starttime']}:
                    raise ValueError('Full cache may read target metadata only')
            return original_read(path,*positional,**kwargs)
        with patch.object(pd,'read_csv',guarded):
            data=HydroCaseData(ROOT)
            if data.cases['Run No'].tolist()!=p['case_ids'] or data.cases.starttime.astype(str).tolist()!=p['case_dates'] or len(data.cases)!=3647:
                raise ValueError('Full metadata identity changed')
            for index,case_id in enumerate(p['case_ids']):
                reconstructed=data.case(index)
                reconstructed['natural']=np.asarray(reconstructed['natural'],dtype=np.float64)
                for name,field in [('pmin','p_min'),('start_cost','startcost_const'),('stop_cost','stopcost_const')]:
                    reconstructed[name]=np.asarray([data.model['generator'][u][field] for u in UNIT_ORDER],dtype=np.float64)
                ip,cp=cache/f'case_{case_id:04d}_inputs.npz',cache/f'case_{case_id:04d}_result.npz'
                for path in [ip,cp]:
                    if digest(path)!=cm['artifacts'][path.name]:
                        raise ValueError('Saved continuous physical data changed')
                same_arrays(reconstructed,arrays(ip))
        for record in p['canonical_diagnostic'].values():
            for artifact in record.values():
                if digest(ROOT/artifact['path'])!=artifact['sha256']:
                    raise ValueError('Canonical diagnostic bytes changed')
        if args.reuse_cache:
            old_path=args.reuse_cache/'manifest.json'
            if args.reuse_cache_sha256 is None or digest(old_path)!=args.reuse_cache_sha256:
                raise ValueError('Explicit prior manifest hash required')
            old=json.loads(old_path.read_text())
            if old['protocol_sha256']!=args.protocol_sha256 or old['status'] not in ['failed','incomplete','complete']:
                raise ValueError('Only finalized exact-protocol caches may supply completed cases')
            report['reuse_manifest_sha256']=args.reuse_cache_sha256
            for case_id,sha in old['completed_cases'].items():
                src=args.reuse_cache/'cases'/f'case_{int(case_id):04d}'
                if digest(src/'complete.json')!=sha:
                    raise ValueError('Prior case proof changed')
                verify_complete_case(src,args.protocol_sha256)
                dst=out/'cases'/src.name
                shutil.copytree(src,dst)
                verify_complete_case(dst,args.protocol_sha256)
                report['completed_cases'][case_id]=sha
                report['resumed_case_ids'].append(int(case_id))
        for case_id in p['pilot_case_ids']:
            if str(case_id) in report['completed_cases']:
                continue
            accepted=[]
            pilot_input=f'case_{case_id:04d}_inputs.npz'
            checked_copy(pilot/pilot_input,out/'cases'/f'case_{case_id:04d}'/'preserved_pilot_inputs.npz',p['pilot_files'][pilot_input])
            same_arrays(arrays(pilot/pilot_input),arrays(cache/pilot_input))
            for replica in [1,2]:
                source_prefix=f'case_{case_id:04d}/attempt_{replica}/'
                directory=out/'cases'/f'case_{case_id:04d}'/f'pilot_replica_{replica}'
                for name,sha in p['pilot_files'].items():
                    if name.startswith(source_prefix):
                        checked_copy(pilot/name,directory/name[len(source_prefix):],sha)
                if not eligible_attempt(validate_attempt(directory)):
                    raise ValueError('Preserved pilot attempt is not eligible')
                accepted.append(directory)
            finish_case(out,case_id,accepted,p,args.protocol_sha256,data.network,cache)
            report['completed_cases'][str(case_id)]=digest(out/'cases'/f'case_{case_id:04d}'/'complete.json')
        if args.preflight:
            report['status']='passed_preflight_only'
            report['checks']='All 3647 inputs reconstructed bitwise; source/data/audit/solver pins passed; 24 complete pilot cases and overlapping canonical case preserved; no solves or labels.'
            return
        report['status']='building_input_only'
        checkpoint({'event':'start','preserved_pilot_cases':len(p['pilot_case_ids'])})
        queue=deque((case_id,replica,60.) for case_id in p['case_ids'] if str(case_id) not in report['completed_cases'] for replica in [1,2])
        accepted={}
        with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn'),max_tasks_per_child=1) as pool:
            pending={}
            while queue or pending:
                while queue and len(pending)<2:
                    case_id,replica,seconds=queue.popleft()
                    ip,cp=f'case_{case_id:04d}_inputs.npz',f'case_{case_id:04d}_result.npz'
                    future=pool.submit(run_attempt,str(out),case_id,replica,seconds,data.network,str(cache),cm['artifacts'][ip],cm['artifacts'][cp],args.protocol_sha256)
                    pending[future]=(case_id,replica,seconds)
                done,_=wait(pending,timeout=20,return_when=FIRST_COMPLETED)
                if not done:
                    checkpoint({'event':'heartbeat','pending_attempts':list(pending.values()),'queued_attempts':len(queue)})
                for future in done:
                    task=pending.pop(future)
                    value=future.result()
                    case_id,replica,seconds=task
                    record=validate_attempt(Path(value['directory']))
                    report['attempts'].append({'case_id':case_id,'replica':replica,'limit_seconds':seconds,
                                               'directory':str(Path(value['directory']).relative_to(out)),
                                               'proof_sha256':digest(Path(value['directory'])/'proof.json'),
                                               'status':record['status'],'mip_status':record['proof'].get('mip_status'),
                                               'runtime_seconds':record['runtime_seconds']})
                    if eligible_attempt(record):
                        accepted.setdefault(case_id,{})[replica]=Path(value['directory'])
                        if len(accepted[case_id])==2:
                            try:
                                finish_case(out,case_id,[accepted[case_id][i] for i in [1,2]],p,args.protocol_sha256,data.network,cache)
                                report['completed_cases'][str(case_id)]=digest(out/'cases'/f'case_{case_id:04d}'/'complete.json')
                            except Exception:
                                report['failures'].append({'case_id':case_id,'stage':'case_replay_or_publication','error':traceback.format_exc()})
                    elif retry_allowed(record,seconds):
                        queue.append((case_id,replica,600.))
                    else:
                        report['failures'].append({'case_id':case_id,'replica':replica,'stage':'attempt_ineligible','directory':value['directory']})
                    checkpoint({'event':'attempt_complete','case_id':case_id,'replica':replica,'limit_seconds':seconds,
                                'mip_status':record['proof'].get('mip_status'),'seconds':record['runtime_seconds'],
                                'failures':len(report['failures']),'queued_attempts':len(queue)})
        if len(report['completed_cases'])!=3647 or report['failures']:
            report['status']='incomplete'
            return
        values=np.empty((3647,168,84),dtype=np.float32)
        for index,case_id in enumerate(p['case_ids']):
            directory=out/'cases'/f'case_{case_id:04d}'
            verify_complete_case(directory,args.protocol_sha256)
            values[index]=np.load(directory/'features.npy',allow_pickle=False)
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite final feature cache')
        np.savez_compressed(out/'features.npz',features=values,case_ids=np.asarray(p['case_ids']),units=np.asarray(UNIT_ORDER),columns=np.asarray(COLUMNS))
        report.update(status='complete',feature_shape=list(values.shape),all_cases_proven_and_replayed=True,
                      features_sha256=digest(out/'features.npz'))
    except BaseException:
        report.update(status='failed',error=traceback.format_exc())
        raise
    finally:
        report['runtime_seconds']=time.monotonic()-started
        report['peak_rss_bytes']=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024)
        report['artifacts']={f.relative_to(out).as_posix():digest(f) for f in out.rglob('*') if f.is_file() and f.name not in ['manifest.json','progress.jsonl']}
        atomic_json(out/'manifest.json',report)
        (out/'experiment_record.md').write_text(f'\n## {args.run_id} — {report["timestamp_utc"]}\n\nStatus {report["status"]}; {len(report["completed_cases"])} of 3647 complete cases. Label-free exact frozen integer water-budget cache, two workers and fresh MILP/conditional-LP replays, 60-second attempts with status-1-only 600-second retries. No relaxed fallback. Command `{report["command"]}`. Source/data/environment/protocol/case/output/failure proof: `artifacts/{args.run_id}/manifest.json`. Runtime {report["runtime_seconds"]:.3f} s. No model fit or submission.\n')


if __name__=='__main__':
    main()
