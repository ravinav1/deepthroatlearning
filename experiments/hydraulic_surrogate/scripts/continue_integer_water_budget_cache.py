"""Four-worker continuation with explicit compatibility receipts and graceful drain."""
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
import signal
import subprocess
import re
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

from scripts.build_integer_water_budget_cache import (
    atomic_json, checked_copy, same_arrays, retry_allowed, eligible_attempt,
    validate_attempt, run_attempt, verify_complete_case, finish_case,
)

BASE_PROTOCOL_SHA = 'f8aaa986c7afdaa802101f84c9cc33ce89309339efcfd321930431b035e8df9a'
EXECUTION_SOURCE = ['scripts/continue_integer_water_budget_cache.py',
                    'scripts/check_integer_cache_transition_resources.py',
                    'tests/test_integer_cache_continuation.py',
                    'docs/integer_water_budget_four_worker_protocol.md']


def check_compatibility(original: dict, continuation: dict) -> None:
    """Permit only explicit execution metadata changes; retain every physical pin."""
    mutable = {'version', 'created_utc', 'workers', 'source', 'resume_policy'}
    extra = {'compatible_base', 'execution'}
    if set(continuation) != set(original) | extra:
        raise ValueError('Continuation protocol field set differs')
    for key in set(original)-mutable:
        if continuation[key] != original[key]:
            raise ValueError('Mathematical compatibility changed: '+key)
    if original['workers'] != 2 or continuation['workers'] != 4:
        raise ValueError('Only the declared two-to-four worker change is allowed')
    if set(continuation['source']) != set(original['source']) | set(EXECUTION_SOURCE):
        raise ValueError('Execution source allowlist changed')
    for name, sha in original['source'].items():
        if continuation['source'][name] != sha:
            raise ValueError('Original frozen source changed: '+name)
    if continuation['compatible_base'] != {'path':'docs/integer_water_budget_full_cache_protocol.json','sha256':BASE_PROTOCOL_SHA}:
        raise ValueError('Wrong compatible base protocol')


def freeze(path: Path) -> None:
    """Freeze a separately named execution recipe without changing original bytes."""
    base_path=ROOT/'docs/integer_water_budget_full_cache_protocol.json'
    if digest(base_path)!=BASE_PROTOCOL_SHA:
        raise ValueError('Original protocol changed')
    original=json.loads(base_path.read_text())
    p=json.loads(json.dumps(original))
    p.update(version='integer_water_budget_four_worker_v1',created_utc=datetime.now(timezone.utc).isoformat(),workers=4,
             resume_policy='Import only externally hashed finalized same-protocol or exact compatible-base complete cases; original case proofs preserved byte-for-byte.',
             compatible_base={'path':str(base_path.relative_to(ROOT)),'sha256':BASE_PROTOCOL_SHA},
             execution={'workers':4,'solver_threads':1,'drain':'SIGINT/SIGTERM or STOP_AFTER_ACTIVE stops new submissions and lets active tasks publish.',
                        'headroom_probe_bytes':3*1024**3,'headroom_minimum_free_percent':30,
                        'runtime_minimum_free_percent':20,'runtime_swap_growth':'Drain on any swapout growth from startup baseline',
                        'resource_check_interval_seconds':20,'maximum_resource_proof_age_seconds':900})
    p['source'].update({name:digest(SOURCE_ROOT/name)for name in EXECUTION_SOURCE})
    check_compatibility(original,p)
    with path.open('x')as stream:json.dump(p,stream,indent=2)
    print(digest(path))


def resource_state() -> dict:
    """Read system memory pressure and swap counts without process enumeration."""
    pressure=subprocess.run(['memory_pressure','-Q'],check=True,text=True,capture_output=True).stdout
    vm=subprocess.run(['vm_stat'],check=True,text=True,capture_output=True).stdout
    free=re.search(r'System-wide memory free percentage:\s*(\d+)%',pressure)
    swap=re.search(r'Swapouts:\s*(\d+)',vm)
    if free is None or swap is None:raise ValueError('Cannot parse system resource guards')
    return {'free_percent':int(free.group(1)),'swapouts':int(swap.group(1))}


def import_complete_case(source: Path, destination: Path, source_protocol_sha: str,
                         target_protocol_sha: str, source_manifest_sha: str) -> dict:
    """Preserve old complete.json and publish a new explicit compatibility receipt."""
    original=verify_complete_case(source,source_protocol_sha)
    old_sha=digest(source/'complete.json')
    shutil.copytree(source,destination)
    if source_protocol_sha==target_protocol_sha:
        return verify_complete_case(destination,target_protocol_sha)
    if source_protocol_sha!=BASE_PROTOCOL_SHA:
        raise ValueError('Unapproved source protocol')
    preserved=destination/'compatible_original_complete.json'
    os.rename(destination/'complete.json',preserved)
    if digest(preserved)!=old_sha:raise ValueError('Original case receipt changed')
    receipt=json.loads(json.dumps(original))
    receipt['protocol_sha256']=target_protocol_sha
    receipt['compatibility']={'original_protocol_sha256':source_protocol_sha,'original_manifest_sha256':source_manifest_sha,
                              'original_complete_sha256':old_sha,'original_complete_path':preserved.name,
                              'all_original_payload_bytes_preserved':True}
    receipt['artifacts'][preserved.name]=old_sha
    atomic_json(destination/'complete.json',receipt)
    for name,sha in original['artifacts'].items():
        if digest(destination/name)!=sha:raise ValueError('Imported raw payload changed')
    return verify_complete_case(destination,target_protocol_sha)


def main()->None:
    """Build physics only; keep failures and require explicit complete-cache eligibility."""
    parser=argparse.ArgumentParser()
    parser.add_argument('--freeze-protocol',action='store_true')
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--protocol',type=Path,default=ROOT/'docs/integer_water_budget_four_worker_protocol.json')
    parser.add_argument('--protocol-sha256')
    parser.add_argument('--run-id')
    parser.add_argument('--reuse-cache',type=Path)
    parser.add_argument('--reuse-cache-sha256')
    parser.add_argument('--preflight-audit',type=Path)
    parser.add_argument('--preflight-audit-sha256')
    parser.add_argument('--resource-proof',type=Path)
    parser.add_argument('--resource-proof-sha256')
    args=parser.parse_args()
    if args.freeze_protocol:
        freeze(args.protocol)
        return
    if args.run_id is None or Path(args.run_id).name!=args.run_id or args.run_id in ['', '.', '..']:
        raise ValueError('Unique single-component run ID required')
    if args.protocol_sha256 is None or digest(args.protocol)!=args.protocol_sha256:
        raise ValueError('Explicit reviewed protocol hash required')
    p=json.loads(args.protocol.read_text())
    base_path=ROOT/p['compatible_base']['path']
    if digest(base_path)!=BASE_PROTOCOL_SHA:raise ValueError('Original mathematical protocol changed')
    original=json.loads(base_path.read_text());check_compatibility(original,p)
    if not args.preflight:
        if args.preflight_audit is None or digest(args.preflight_audit)!=args.preflight_audit_sha256:
            raise ValueError('Explicit independent preflight proof required')
        audit=json.loads(args.preflight_audit.read_text())
        if audit['status']!='passed' or audit['protocol_sha256']!=args.protocol_sha256:
            raise ValueError('Require approval of this exact continuation')
        if args.resource_proof is None or digest(args.resource_proof)!=args.resource_proof_sha256:
            raise ValueError('Explicit headroom and drain-harness proof required')
        resource_proof=json.loads(args.resource_proof.read_text())
        age=(datetime.now(timezone.utc)-datetime.fromisoformat(resource_proof['timestamp_utc'])).total_seconds()
        if (resource_proof['status']!='passed' or resource_proof['protocol_sha256']!=args.protocol_sha256
                or not resource_proof['drain_harness_passed'] or not resource_proof['headroom_probe_passed']
                or not 0<=age<=p['execution']['maximum_resource_proof_age_seconds']):
            raise ValueError('Transition resource proof absent or stale')
    drain={'requested':False,'reason':None}
    def request_drain(signum,frame):
        drain.update(requested=True,reason='signal '+str(signum))
    signal.signal(signal.SIGINT,request_drain);signal.signal(signal.SIGTERM,request_drain)
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
            'prefix_columns':p['prefix_columns'],'workers':4,'base_protocol_sha256':BASE_PROTOCOL_SHA,
            'preflight_audit_sha256':args.preflight_audit_sha256,'resource_proof_sha256':args.resource_proof_sha256,'completed_cases':{},'attempts':[],'failures':[],'resumed_case_ids':[]}
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
        checked_copy(base_path,out/'compatible_base_protocol.json',BASE_PROTOCOL_SHA)
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
            if old['protocol_sha256'] not in [args.protocol_sha256,BASE_PROTOCOL_SHA] or old['status'] not in ['failed','incomplete','complete']:
                raise ValueError('Only finalized exact compatible caches may supply completed cases')
            expected_original=original if old['protocol_sha256']==BASE_PROTOCOL_SHA else p
            if old['source']!=expected_original['source'] or old['data']!=p['data'] or old['feature_specification']!=p['feature_specification']:
                raise ValueError('Reuse lineage changed')
            checked_copy(old_path,out/'reuse_manifest.json',args.reuse_cache_sha256)
            report['reuse_manifest_sha256']=args.reuse_cache_sha256
            for case_id,sha in old['completed_cases'].items():
                src=args.reuse_cache/'cases'/f'case_{int(case_id):04d}'
                if digest(src/'complete.json')!=sha:
                    raise ValueError('Prior case proof changed')
                dst=out/'cases'/src.name
                import_complete_case(src,dst,old['protocol_sha256'],args.protocol_sha256,args.reuse_cache_sha256)
                report['completed_cases'][case_id]=digest(dst/'complete.json')
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
        baseline_resources=resource_state();last_resource_check=0.
        report['startup_resources']=baseline_resources
        if baseline_resources['free_percent']<p['execution']['headroom_minimum_free_percent']:
            raise ValueError('Insufficient current memory headroom')
        with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn'),max_tasks_per_child=1) as pool:
            pending={}
            while (queue and not drain['requested']) or pending:
                if (out/'STOP_AFTER_ACTIVE').exists():drain.update(requested=True,reason='stop file')
                if time.monotonic()-last_resource_check>=p['execution']['resource_check_interval_seconds']:
                    state=resource_state();last_resource_check=time.monotonic()
                    if state['free_percent']<p['execution']['runtime_minimum_free_percent'] or state['swapouts']>baseline_resources['swapouts']:
                        drain.update(requested=True,reason='resource guard '+json.dumps(state))
                    checkpoint({'event':'resource_check','resources':state,'drain':dict(drain)})
                if drain['requested'] and not pending:break
                while queue and not drain['requested'] and len(pending)<4:
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
        report['graceful_drain']=dict(drain);report['remaining_queued_attempts']=len(queue)
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
        (out/'experiment_record.md').write_text(f'\n## {args.run_id} — {report["timestamp_utc"]}\n\nStatus {report["status"]}; {len(report["completed_cases"])} of 3647 complete cases. Label-free exact frozen integer water-budget cache, four workers and fresh MILP/conditional-LP replays; exact compatible original case receipts preserved, 60-second attempts with status-1-only 600-second retries. No relaxed fallback. Command `{report["command"]}`. Source/data/environment/protocol/case/output/failure proof: `artifacts/{args.run_id}/manifest.json`. Runtime {report["runtime_seconds"]:.3f} s. No model fit or submission.\n')


if __name__=='__main__':
    main()
