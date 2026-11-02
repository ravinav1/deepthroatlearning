"""Assemble a new accepted aggregate; never rewrite original integer-cache evidence."""
from __future__ import annotations
import os
for variable in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[variable]='1'
import argparse
from datetime import datetime,timezone,timedelta
import json
from pathlib import Path
import shutil
import shlex
import sys
import time
import traceback
from unittest.mock import patch
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sintef_project.lp_features import HydroCaseData,UNIT_ORDER
from sintef_project.integer_water_budget_features import COLUMNS,feature_specification,compute_features
from sintef_project.integer_water_budget_aggregate_features import (
    COMPLETION_SHA,EXCEPTION_AUDIT_SHA,digest,validate_aggregate_manifest)
from scripts.build_integer_water_budget_cache import verify_complete_case,same_arrays
from scripts.audit_integer_water_budget_cache import arrays,verify_hashes,audit_case
from scripts.complete_integer_case2012 import failed_record,verify_physics
from scripts.pilot_integer_water_budget import environment
from sintef_project.lp_shop_relaxation import ShopRelaxationLP

PROTOCOL=ROOT/'docs/integer_water_budget_aggregate_protocol.json'
OLD_PROTOCOL='docs/integer_water_budget_four_worker_protocol_v2.json'
OLD_SHA='7571930990cadc542c7ad4f47e7f1195d7a239885c887ad45b051afd9b9377d9'
BASE_SHA='f8aaa986c7afdaa802101f84c9cc33ce89309339efcfd321930431b035e8df9a'
IMPORT_MANIFEST='artifacts/integer-water-budget-full-001/manifest.json'
IMPORT_SHA='d6da75c846cc0b90b2462839ec368c5abda3fd2ba8a8a8e4cee681272de29867'
EXIT_COMMAND="pgrep -fl 'multiprocessing.spawn|continue_integer_water_budget_cache_v2'"
NEW=['scripts/build_integer_water_budget_aggregate.py','sintef_project/integer_water_budget_aggregate_features.py',
     'tests/test_integer_water_budget_aggregate.py','docs/integer_water_budget_aggregate.md']


def write_json(path: Path,value: dict) -> None:
    """Create immutable assembly records; no existing file is overwritten."""
    with path.open('x')as stream:json.dump(value,stream,indent=2)


def inventory(path: Path) -> dict:
    """Exact recursive file inventory, with no symlink escapes."""
    result={}
    for file in sorted(path.rglob('*')):
        if file.is_symlink():raise ValueError('All symlinks are forbidden in preserved evidence')
        if file.is_file():
            if not file.resolve().is_relative_to(path.resolve()):raise ValueError('Inventory path escape')
            result[str(file.relative_to(path))]=digest(file)
    return result


def copy_directory_exact(source: Path,destination: Path,expected: dict|None=None) -> dict:
    """Copy every old byte without adding anything inside the preserved directory."""
    before=inventory(source)
    if expected is not None and before!=expected:raise ValueError('Source inventory changed')
    shutil.copytree(source,destination)
    if inventory(destination)!=before or inventory(source)!=before:raise ValueError('Copy did not preserve every source byte')
    return before


def freeze() -> None:
    """Freeze sole numerical exception and feature recipe before final assembly."""
    assert digest(ROOT/OLD_PROTOCOL)==OLD_SHA and digest(ROOT/IMPORT_MANIFEST)==IMPORT_SHA
    old=json.loads((ROOT/OLD_PROTOCOL).read_text());imports=json.loads((ROOT/IMPORT_MANIFEST).read_text())
    completion_path=ROOT/'artifacts/integer-case2012-completion-001/manifest.json'
    completion=json.loads(completion_path.read_text());assert digest(completion_path)==COMPLETION_SHA
    cp=ROOT/'artifacts/integer-case2012-completion-001/protocol.json';exception_protocol=json.loads(cp.read_text())
    source={**old['source'],**exception_protocol['source'],**{name:digest(ROOT/name)for name in NEW}}
    p={key:old[key]for key in ['data','environment','case_ids','case_dates','feature_specification','prefix_protocol',
        'prefix_protocol_sha256','prefix_columns','continuous_cache','continuous_cache_manifest_sha256']}
    p.update(version='integer_water_budget_aggregate_v1',created_utc=datetime.now(timezone.utc).isoformat(),source=source,
        original_cache='artifacts/integer-water-budget-full-four-worker-001',original_protocol=OLD_PROTOCOL,original_protocol_sha256=OLD_SHA,
        original_coordinator_pid=11014,original_exit_process_command=EXIT_COMMAND,
        original_base_protocol=old['compatible_base']['path'],original_base_protocol_sha256=BASE_SHA,
        imported_manifest=IMPORT_MANIFEST,imported_manifest_sha256=IMPORT_SHA,
        imported_case_ids=sorted(map(int,imports['completed_cases'])),exception_case_ids=[2012],
        expected_counts={'ordinary':3010,'imported':636,'exception':1},preflight_case_ids=[1,1000,2012],
        completion_directory=str(completion_path.parent.relative_to(ROOT)),completion_manifest_sha256=COMPLETION_SHA,
        completion_protocol_sha256=digest(cp),exception_actual_audit='artifacts/integer-case2012-completion-result-independent-audit-001/manifest.json',
        exception_actual_audit_sha256=EXCEPTION_AUDIT_SHA,
        acceptance_basis='3,646 unchanged ordinary/imported strict receipts plus sole case2012_dual_feasibility_certificate; failed proofs remain failed.',
        receipt_schema='cases/case_N/aggregate_receipt.json; original_case directory copied byte-exact; explicit kind/protocol/feature/result/exception fields.',
        original_exit_gate='Externally hashed finalized original manifest with exactly case2012 absent; queue empty; two declared failed attempts; separately pinned coordinator/all-worker exit observation.',
        model_gate='All3647 accepted receipts, exact84 feature assembly and separately pinned independent actual aggregate physical audit required.',
        optimization_calls=0,labels_decoded=False,training=False)
    assert len(p['imported_case_ids'])==636 and 2012 not in p['imported_case_ids']
    write_json(PROTOCOL,p);print(digest(PROTOCOL),flush=True)


def verify_protocol(p: dict,sha: str) -> tuple[dict,dict]:
    """Check immutable source/data and accepted one-case evidence."""
    assert digest(PROTOCOL)==sha and p==json.loads(PROTOCOL.read_text()) and p['environment']==environment()
    for group in ['source','data']:verify_hashes(ROOT,p[group])
    for pathkey,hashkey in [('original_protocol','original_protocol_sha256'),('original_base_protocol','original_base_protocol_sha256'),
        ('imported_manifest','imported_manifest_sha256'),('prefix_protocol','prefix_protocol_sha256'),('exception_actual_audit','exception_actual_audit_sha256')]:
        assert digest(ROOT/p[pathkey])==p[hashkey]
    old=json.loads((ROOT/p['original_protocol']).read_text())
    for field in ['data','case_ids','case_dates','feature_specification','prefix_columns','environment']:assert old[field]==p[field]
    assert p['feature_specification']==feature_specification() and p['exception_case_ids']==[2012]
    completion=ROOT/p['completion_directory'];assert digest(completion/'manifest.json')==p['completion_manifest_sha256']==COMPLETION_SHA
    cm=json.loads((completion/'manifest.json').read_text());verify_hashes(completion,cm['artifacts'])
    audit=json.loads((ROOT/p['exception_actual_audit']).read_text())
    assert (audit['status']=='passed' and audit['completion_manifest_sha256']==COMPLETION_SHA
        and audit['case2012_exception_independently_accepted'] and audit['all_workers_exited']
        and audit['original_failures_preserved'] and not audit['old_v2_eligible'])
    assert cm['status']=='complete_case2012_exception' and cm['original_status_preserved']=='failed'
    assert digest(completion/'protocol.json')==p['completion_protocol_sha256']==audit['protocol_sha256']
    return old,cm


def validate_original_final(m: dict,p: dict) -> None:
    """Never assemble from a running or otherwise incomplete original population."""
    assert m['status']=='incomplete' and m['protocol_sha256']==p['original_protocol_sha256']
    assert m['remaining_queued_attempts']==0 and not m['graceful_drain']['requested']
    assert m['case_ids']==p['case_ids'] and m['case_dates']==p['case_dates']
    assert set(m['completed_cases'])=={str(cid)for cid in p['case_ids']if cid!=2012}
    assert len(m['failures'])==2
    assert {(f['case_id'],f['replica'],f['stage'])for f in m['failures']}=={(2012,1,'attempt_ineligible'),(2012,2,'attempt_ineligible')}
    assert sorted(m['resumed_case_ids'])==p['imported_case_ids']
    assert m['reuse_manifest_sha256']==IMPORT_SHA and m['labels_decoded']is False and m['training']is False


def verify_original_exit(observation: dict,original: dict,p: dict,manifest_sha: str) -> None:
    """Bind real scoped process output to the finalized coordinator/worker lineage."""
    o=observation;cache=ROOT/p['original_cache']
    assert (o['status']=='passed' and o['original_cache_manifest_sha256']==manifest_sha
        and o['original_cache_protocol_sha256']==p['original_protocol_sha256'] and o['original_cache_path']==p['original_cache'])
    worker_pids=sorted({json.loads((cache/attempt['directory']/'proof.json').read_text())['pid']for attempt in original['attempts']})
    assert o['expected_worker_pids']==worker_pids and o['expected_coordinator_pid']==p['original_coordinator_pid']
    assert o['command']==p['original_exit_process_command']==EXIT_COMMAND and o['returncode']in [0,1]
    output=o['scoped_process_output'];live={int(line.split()[0])for line in output.splitlines()if line.strip()}
    assert (o['returncode']==0)==bool(live)
    assert not live.intersection(worker_pids) and p['original_coordinator_pid']not in live
    assert datetime.fromisoformat(o['timestamp_utc'])>=datetime.fromisoformat(original['timestamp_utc'])+timedelta(seconds=original['runtime_seconds'])
    assert o['coordinator_exited']is True and o['all_workers_exited']is True


def build_case(out: Path,cid: int,p: dict,p_sha: str,old: dict,completion: dict,
               source_manifest_sha: str|None,network) -> dict:
    """Preserve an old directory and publish a separate typed aggregate receipt."""
    source=ROOT/p['original_cache']/'cases'/f'case_{cid:04d}'
    folder=out/'cases'/f'case_{cid:04d}';folder.mkdir(parents=True)
    copied=folder/'original_case';old_inventory=copy_directory_exact(source,copied)
    continuous=ROOT/p['continuous_cache'];ip=continuous/f'case_{cid:04d}_inputs.npz';cp=continuous/f'case_{cid:04d}_result.npz'
    reference_manifest=json.loads((continuous/'manifest.json').read_text())
    assert digest(ip)==reference_manifest['artifacts'][ip.name] and digest(cp)==reference_manifest['artifacts'][cp.name]
    r={'status':'accepted','case_id':cid,'aggregate_protocol_sha256':p_sha,'original_cache_manifest_sha256':source_manifest_sha,
        'original_cache_protocol_sha256':OLD_SHA,'original_source_case_path':str(source.relative_to(ROOT)),
        'original_case_path':'original_case','original_case_inventory':old_inventory,'input_sha256':digest(ip),'control_sha256':digest(cp),
        'compatibility':None,'exception':None}
    if cid!=2012:
        receipt=verify_complete_case(copied,OLD_SHA);assert receipt['case_id']==cid
        assert receipt['input_sha256']==r['input_sha256'] and receipt['control_sha256']==r['control_sha256']
        same_arrays(arrays(copied/'inputs.npz'),arrays(ip))
        imported=cid in p['imported_case_ids'];kind='imported'if imported else'ordinary'
        if imported:
            imports=json.loads((ROOT/p['imported_manifest']).read_text());compat=receipt['compatibility']
            assert compat=={'original_protocol_sha256':BASE_SHA,'original_manifest_sha256':IMPORT_SHA,
                'original_complete_sha256':imports['completed_cases'][str(cid)],'original_complete_path':'compatible_original_complete.json',
                'all_original_payload_bytes_preserved':True}
            assert digest(copied/'compatible_original_complete.json')==compat['original_complete_sha256']
            original=json.loads((copied/'compatible_original_complete.json').read_text());expected=json.loads(json.dumps(original))
            expected['protocol_sha256']=OLD_SHA;expected['compatibility']=compat
            expected['artifacts']['compatible_original_complete.json']=compat['original_complete_sha256'];assert expected==receipt
            r['compatibility']=compat
        else:assert 'compatibility'not in receipt
        r.update(kind=kind,accepted_attempt_protocol_sha256=BASE_SHA if imported else OLD_SHA,
            original_complete_path='original_case/complete.json',original_complete_sha256=digest(copied/'complete.json'),
            input_path='original_case/inputs.npz',result_path='original_case/canonical_result.npz',features_path='original_case/features.npy')
    else:
        assert not (copied/'complete.json').exists()
        original_failures=[]
        for replica in [1,2]:
            attempt=copied/f'replica_{replica}_60s';record=json.loads((attempt/'proof.json').read_text());failed_record(record)
            verify_hashes(attempt,record['artifacts']);original_failures.append({'replica':replica,'path':f'original_case/replica_{replica}_60s/proof.json','sha256':digest(attempt/'proof.json')})
        shutil.copy2(ip,folder/'inputs.npz');shutil.copy2(ROOT/p['completion_directory']/'exception_case2012_result.npz',folder/'exception_result.npz')
        case,control=arrays(ip),arrays(cp);results=[];witnesses=[]
        for replica in [1,2]:
            fresh=ROOT/p['completion_directory']/'cases/case_2012'/f'replica_{replica}_60s'
            record=json.loads((fresh/'proof.json').read_text());failed_record(record)
            item=completion['replicas'][replica-1]
            result,checks=verify_physics(network,case,control,fresh,record,item['lp_call_metadata'][0]['iterations'],item['checks']['certificate'])
            same_arrays(result,arrays(folder/'exception_result.npz'));results.append(result)
            for name in ['failed_incumbent.npz','fixed_lp_1.npz','fixed_lp_2.npz']:
                assert digest(fresh/name)==digest(copied/f'replica_{replica}_60s'/name)
            witnesses.append({**record['proof'],'fixed_lp_replay_bitwise':True})
        same_arrays(results[0],results[1])
        # Supplemental replay fact is established above, never written into a failed proof.
        values=compute_features(case,results[0],control,network,witnesses[0]);np.save(folder/'features.npy',values,allow_pickle=False)
        r.update(kind='exception',accepted_attempt_protocol_sha256=p['completion_protocol_sha256'],
            original_complete_path=None,original_complete_sha256=None,input_path='inputs.npz',
            result_path='exception_result.npz',features_path='features.npy',
            exception={'acceptance_basis':'case2012_dual_feasibility_certificate','completion_manifest_sha256':COMPLETION_SHA,
                'actual_audit_sha256':EXCEPTION_AUDIT_SHA,'completion_evidence_path':'exception_evidence/completion',
                'actual_audit_evidence_path':'exception_evidence/actual_audit.json','original_failed_proofs':original_failures,
                'original_status_preserved':'failed','original_objective_guard_still_fails':True,
                'native_status':0,'conditional_replays_verified':True,'feature_witness_is_supplemental_not_a_rewritten_proof':True})
    values=np.load(folder/r['features_path'],allow_pickle=False)
    assert values.shape==(168,84) and values.dtype==np.float32 and np.isfinite(values).all()
    r.update(features_sha256=digest(folder/r['features_path']),result_sha256=digest(folder/r['result_path']),artifacts=inventory(folder))
    write_json(folder/'aggregate_receipt.json',r);return r


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--freeze-protocol',action='store_true');parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--protocol-sha256');parser.add_argument('--run-id');parser.add_argument('--original-manifest-sha256')
    parser.add_argument('--original-exit-observation',type=Path);parser.add_argument('--original-exit-observation-sha256')
    parser.add_argument('--independent-source-audit',type=Path);parser.add_argument('--independent-source-audit-sha256');args=parser.parse_args()
    if args.freeze_protocol:freeze();return
    assert args.run_id and Path(args.run_id).name==args.run_id and args.run_id not in ['.','..']
    out=ROOT/'artifacts'/args.run_id;out.mkdir(exist_ok=False);start=time.monotonic()
    m={'status':'initializing','timestamp_utc':datetime.now(timezone.utc).isoformat(),'command':shlex.join([sys.executable,*sys.argv]),
        'protocol_sha256':args.protocol_sha256,'optimization_calls':0,'labels_decoded':False,'training':False,'completed_cases':{},'failures':[]}
    try:
        p=json.loads(PROTOCOL.read_text());old,completion=verify_protocol(p,args.protocol_sha256)
        m.update({key:p[key]for key in ['source','data','environment','case_ids','case_dates','feature_specification','prefix_columns',
            'completion_manifest_sha256','exception_actual_audit_sha256']})
        shutil.copy2(PROTOCOL,out/'protocol.json');shutil.copy2(ROOT/p['original_protocol'],out/'original_execution_protocol.json')
        shutil.copy2(ROOT/p['original_base_protocol'],out/'compatible_base_protocol.json')
        for name in p['source']:
            destination=out/'source'/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,destination)
        evidence=out/'exception_evidence';evidence.mkdir()
        copy_directory_exact(ROOT/p['completion_directory'],evidence/'completion')
        shutil.copy2(ROOT/p['exception_actual_audit'],evidence/'actual_audit.json')
        assert digest(ROOT/p['continuous_cache']/'manifest.json')==p['continuous_cache_manifest_sha256']
        if not args.preflight:
            source=ROOT/p['original_cache'];assert digest(source/'manifest.json')==args.original_manifest_sha256
            original=json.loads((source/'manifest.json').read_text());validate_original_final(original,p)
            assert original['source']==old['source'] and original['data']==p['data'] and original['feature_specification']==p['feature_specification']
            verify_hashes(source,original['artifacts'])
            assert digest(args.original_exit_observation)==args.original_exit_observation_sha256
            exit_proof=json.loads(args.original_exit_observation.read_text())
            verify_original_exit(exit_proof,original,p,args.original_manifest_sha256)
            assert digest(args.independent_source_audit)==args.independent_source_audit_sha256
            review=json.loads(args.independent_source_audit.read_text());assert review['status']=='passed' and review['protocol_sha256']==args.protocol_sha256
            shutil.copy2(source/'manifest.json',out/'original_cache_manifest.json');shutil.copy2(args.original_exit_observation,out/'original_exit_observation.json')
            m.update(original_cache_manifest_sha256=args.original_manifest_sha256,original_exit_observation_sha256=args.original_exit_observation_sha256,
                independent_source_audit_sha256=args.independent_source_audit_sha256)
        raw=pd.read_csv;target=(ROOT/'sintef_project/data/kernel/Unit_commitment_decisions.csv').resolve()
        def guarded(path,*positional,**kwargs):
            if isinstance(path,(str,Path))and Path(path).resolve()==target:assert not positional and kwargs=={'usecols':['Run No','starttime']}
            return raw(path,*positional,**kwargs)
        with patch.object(pd,'read_csv',guarded):data=HydroCaseData(ROOT)
        assert data.cases['Run No'].tolist()==p['case_ids'] and data.cases.starttime.astype(str).tolist()==p['case_dates']
        selected=p['preflight_case_ids']if args.preflight else p['case_ids'];receipts={}
        for index,cid in enumerate(selected):
            if not args.preflight and cid!=2012:
                assert digest(ROOT/p['original_cache']/'cases'/f'case_{cid:04d}'/'complete.json')==original['completed_cases'][str(cid)]
            receipt=build_case(out,cid,p,args.protocol_sha256,old,completion,args.original_manifest_sha256,data.network)
            receipts[cid]=receipt;m['completed_cases'][str(cid)]=digest(out/'cases'/f'case_{cid:04d}'/'aggregate_receipt.json')
            if args.preflight and cid!=2012:
                folder=out/'cases'/f'case_{cid:04d}'/'original_case';old_receipt=json.loads((folder/'complete.json').read_text())
                ip=ROOT/p['continuous_cache']/f'case_{cid:04d}_inputs.npz';cp=ROOT/p['continuous_cache']/f'case_{cid:04d}_result.npz'
                audit_case(folder,old_receipt,arrays(ip),arrays(cp),ShopRelaxationLP(data.network),receipt['accepted_attempt_protocol_sha256'],old)
            if (index+1)%100==0:print(json.dumps({'assembled':index+1,'total':len(selected),'seconds':time.monotonic()-start}),flush=True)
        if args.preflight:m.update(status='passed_preflight_only',preflight_case_ids=selected,aggregate_eligible=False)
        else:
            assert digest(ROOT/p['original_cache']/'manifest.json')==args.original_manifest_sha256
            values=np.stack([np.load(out/'cases'/f'case_{cid:04d}'/receipts[cid]['features_path'],allow_pickle=False)for cid in p['case_ids']])
            np.savez_compressed(out/'features.npz',features=values,case_ids=np.asarray(p['case_ids']),columns=np.asarray(COLUMNS),units=np.asarray(UNIT_ORDER))
            m.update(status='complete',features_sha256=digest(out/'features.npz'),all_cases_accepted_and_replayed=True,
                all_cases_native_strict_gate_passed=False,original_cache_still_ineligible=True,
                ordinary_case_ids=[cid for cid in p['case_ids']if receipts[cid]['kind']=='ordinary'],imported_case_ids=p['imported_case_ids'],exception_case_ids=[2012],
                ordinary_case_count=3010,imported_case_count=636,exception_count=1,retained_original_failures=original['failures'])
            validate_aggregate_manifest(m,p)
    except BaseException:
        m.update(status='failed',error=traceback.format_exc());raise
    finally:
        m['runtime_seconds']=time.monotonic()-start;m['artifacts']=inventory(out)
        write_json(out/'manifest.json',m)
        (out/'experiment_record.md').write_text('\n## '+args.run_id+' — '+m['timestamp_utc']+'\n\nStatus '+m['status']+
            '; separate input-only integer aggregate, zero solves/labels/training, original failures preserved. Full recipe/command/data/source/receipt provenance: `'+
            str((out/'manifest.json').relative_to(ROOT))+'`, SHA `'+digest(out/'manifest.json')+'`.\n')
    print(json.dumps({'status':m['status'],'manifest_sha256':digest(out/'manifest.json')}),flush=True)


if __name__=='__main__':main()
