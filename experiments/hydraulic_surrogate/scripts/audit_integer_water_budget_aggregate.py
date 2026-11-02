"""Independent full integer aggregate audit; preserve the original failed cache."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone,timedelta
from fractions import Fraction
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from unittest.mock import patch
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[key]='1'
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_integer_water_budget_cache import audit_case,arrays,digest,same,independent_features,COLUMNS,TARGETS
from scripts.audit_integer_exact_certificate import independent_exact,decode
from scripts.audit_integer_case2012_result import decoded
from scripts.audit_hyljelihyl_river_results import check_lp
from sintef_project.lp_hyljelihyl_v2 import capture_problem
from sintef_project.lp_features import HydroCaseData,UNIT_ORDER
from sintef_project.lp_shop_relaxation import ShopRelaxationLP

AGGREGATE_PROTOCOL_SHA='aacf3ee1492fbb48f1d62827cfc7306bad20841c48088cdb1b68842b94fe0c48'
OLD_SHA='7571930990cadc542c7ad4f47e7f1195d7a239885c887ad45b051afd9b9377d9'
BASE_SHA='f8aaa986c7afdaa802101f84c9cc33ce89309339efcfd321930431b035e8df9a'
IMPORT_SHA='d6da75c846cc0b90b2462839ec368c5abda3fd2ba8a8a8e4cee681272de29867'
COMPLETION_SHA='7da83d48119d48a0826a6d3edf01a265ee932edb8ec7f899be6dc2bb9dce8984'
COMPLETION_PROTOCOL_SHA='6499a285d267e8ccab0cb2063e682880c8837423fdd8606d9c83d11063cff989'
EXCEPTION_AUDIT_SHA='2708cafb0b2ed2ae7f0b43ee1515d63dc80ebcd3908109277a2a7977fde66b02'
CERTIFICATE_SHA='a6892f16ca3677f40f21ac24fd896b7202f5f5212fbd5b83488961c15c26916c'
# Filled with exact source hashes during preparation, before any actual aggregate audit.
EXPECTED_HELPERS = {'scripts/audit_integer_water_budget_cache.py': '78ff7efe84b00129bafa112c9a2a6be8149100f99f456313af8178788f9660e9', 'scripts/audit_integer_exact_certificate.py': '002462f72456e3ebda9ef58c0870012b8f2310ed6d99c5df8366fa86994b69f8', 'scripts/audit_integer_case2012_result.py': '7fb904afcc5093c45b272076d597b66dd3e37bc1080d94892e3b03c54302d73f', 'scripts/audit_hyljelihyl_river_results.py': 'fa9e397689fe0cc32b1faabecaa33d93593b2c33e82270e3e947385d13b43c42', 'sintef_project/lp_hyljelihyl_v2.py': 'd4362e0b174536f8e912461f93d9945666017612605b2b67b060579580770193', 'sintef_project/lp_features.py': 'bd16e0b4e9e32976b7eb50d5334e05b94f2b074bc0d8eb83a787f26f95268720', 'sintef_project/lp_shop_relaxation.py': '1ac82e04d2df6571c71acfbc6be6c4ac482ea69de441471804c2b1bce1ce1aaf'}


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def safe(root: Path,name: str) -> Path:
    """Reject escaping paths, aliases and symlinks, including in-root links."""
    q=Path(name);assert name and not q.is_absolute()and '..'not in q.parts and str(q)==name
    assert not root.is_symlink()
    for i in range(1,len(q.parts)+1):assert not root.joinpath(*q.parts[:i]).is_symlink()
    result=root/q;assert result.resolve().is_relative_to(root.resolve());return result


def inventory(root: Path) -> dict:
    result={}
    for q in root.rglob('*'):
        assert not q.is_symlink()
        if q.is_file():result[str(q.relative_to(root))]=digest(q)
    return result


def hashes(root: Path,records: dict) -> None:
    for name,h in records.items():assert digest(safe(root,name))==h,name


def groups(manifest: dict,protocol: dict) -> dict[str,set[int]]:
    """Require the sole declared exception and exact disjoint population counts."""
    assert protocol['version']=='integer_water_budget_aggregate_v1'
    assert len(protocol['case_ids'])==len(set(protocol['case_ids']))==3647
    g={name:set(manifest[name+'_case_ids'])for name in ['ordinary','imported','exception']}
    for name,n in [('ordinary',3010),('imported',636),('exception',1)]:
        assert len(manifest[name+'_case_ids'])==len(g[name])==n
    assert g['exception']=={2012}and protocol['exception_case_ids']==[2012]
    assert manifest['imported_case_ids']==protocol['imported_case_ids']
    assert not(g['ordinary']&g['imported']or g['ordinary']&g['exception']or g['imported']&g['exception'])
    assert set.union(*g.values())==set(protocol['case_ids'])
    assert [manifest['ordinary_case_count'],manifest['imported_case_count'],manifest['exception_count']]==[3010,636,1]
    return g


def compatibility(folder: Path,receipt: dict,cid: int,old_imports: Path,imports: dict) -> None:
    """Use the original636 receipt transformation and payload checks unchanged."""
    original_folder=old_imports/'cases'/f'case_{cid:04d}';original_path=folder/'compatible_original_complete.json'
    assert digest(original_path)==imports['completed_cases'][str(cid)]==digest(original_folder/'complete.json')
    expected_compat={'original_protocol_sha256':BASE_SHA,'original_manifest_sha256':IMPORT_SHA,
        'original_complete_sha256':digest(original_path),'original_complete_path':original_path.name,
        'all_original_payload_bytes_preserved':True}
    assert receipt['compatibility']==expected_compat
    original_receipt=read(original_path);expected=json.loads(json.dumps(original_receipt))
    expected['protocol_sha256']=OLD_SHA;expected['compatibility']=expected_compat
    expected['artifacts'][original_path.name]=digest(original_path)
    assert expected==receipt
    hashes(original_folder,original_receipt['artifacts']);hashes(folder,original_receipt['artifacts'])


def receipt_contract(folder: Path,r: dict,cid: int,kind: str,p_sha: str,old_manifest_sha: str) -> None:
    """Require separate typed acceptance without relabelling preserved old proofs."""
    assert r['status']=='accepted'and r['case_id']==cid and r['kind']==kind
    assert r['aggregate_protocol_sha256']==p_sha and r['original_cache_manifest_sha256']==old_manifest_sha
    assert r['original_cache_protocol_sha256']==OLD_SHA and r['original_case_path']=='original_case'
    assert inventory(folder/'original_case')==r['original_case_inventory']
    actual=inventory(folder);actual.pop('aggregate_receipt.json');assert actual==r['artifacts']
    for key in ['input','result','features']:
        assert digest(safe(folder,r[key+'_path']))==r[key+'_sha256']
    if kind=='exception':
        assert cid==2012 and r['original_complete_path']is None and r['original_complete_sha256']is None
        assert not(folder/'original_case/complete.json').exists()and r['compatibility']is None
        assert r['accepted_attempt_protocol_sha256']==COMPLETION_PROTOCOL_SHA
        assert r['input_path']=='inputs.npz'and r['result_path']=='exception_result.npz'and r['features_path']=='features.npy'
        assert r['exception']['acceptance_basis']=='case2012_dual_feasibility_certificate'
    else:
        assert r['exception']is None
        assert r['original_complete_path']=='original_case/complete.json'
        assert digest(folder/r['original_complete_path'])==r['original_complete_sha256']
        assert r['input_path']=='original_case/inputs.npz'and r['result_path']=='original_case/canonical_result.npz'
        assert r['features_path']=='original_case/features.npy'
        assert r['accepted_attempt_protocol_sha256']==(BASE_SHA if kind=='imported'else OLD_SHA)
        if kind=='ordinary':assert r['compatibility']is None


def exception_certificate(case: dict,control: dict,network,result: dict,completion: Path,cm: dict,cp: dict,original: Path) -> dict:
    """Independently verify actual fresh/old failures, KKT, exact bound and decoded fields."""
    assert digest(completion/'manifest.json')==COMPLETION_SHA and digest(completion/'protocol.json')==COMPLETION_PROTOCOL_SHA
    assert cm['status']=='complete_case2012_exception'and cm['case_id']==2012
    assert cm['original_status_preserved']=='failed'and not cm['old_v2_eligible']and not cm['aggregate_eligible']
    assert cp['allowed_case_ids']==[2012]and cp['targets']==TARGETS and cp['absolute_eur_bound']=={'numerator':'1','denominator':'100000'}
    for group in ['source','data','certificate_inputs']:hashes(ROOT,cp[group])
    hashes(completion,cm['artifacts'])
    model,problem=capture_problem(network,case,1.);indices=model.state[[UNIT_ORDER.index(u)for u in TARGETS]].ravel()
    scale=float(1+np.abs(case['price']).mean());assert np.isfinite(scale)and scale>=1
    assert digest(ROOT/cp['certificate'])==CERTIFICATE_SHA;certificate=read(ROOT/cp['certificate'])
    assert len(cm['replicas'])==2 and len({x['pid']for x in cm['replicas']})==2
    previous=None;records=[]
    for replica,observed in enumerate(cm['replicas'],1):
        folder=completion/'cases/case_2012'/f'replica_{replica}_60s';record=read(folder/'proof.json')
        assert digest(folder/'proof.json')==observed['fresh_failed_proof_sha256']
        assert record['pid']==observed['pid']and record['protocol_sha256']==COMPLETION_PROTOCOL_SHA
        assert record['status']=='failed'and'Fixed-state LP worsened the MILP incumbent objective'in record['error']
        assert record['input_sha256']==cp['case_input_sha256']and record['control_sha256']==cp['case_control_sha256']
        assert record['environment']==cp['environment'];hashes(folder,record['artifacts'])
        proof=record['proof'];assert proof['mip_status']==0 and proof['proven_optimal']and proof['mip_gap']==0
        assert proof['selected_units']==TARGETS and proof['mip_rel_gap_requested']==0
        assert proof['mip_time_limit_seconds']==60 and proof['fixed_lp_time_limit_seconds_each']==1
        for old in cp['original_attempts']:
            source=ROOT/old['directory'];copied=original/source.name
            assert digest(source/'proof.json')==digest(copied/'proof.json')==old['proof_sha256']
            oldproof=read(copied/'proof.json');assert oldproof['status']=='failed'and oldproof['protocol_sha256']==OLD_SHA
            assert 'Fixed-state LP worsened the MILP incumbent objective'in oldproof['error']
            hashes(copied,oldproof['artifacts'])
            for name,h in old['payloads'].items():
                assert digest(folder/name)==digest(source/name)==digest(copied/name)==h
                same(arrays(folder/name),arrays(copied/name))
        native=arrays(folder/'failed_incumbent.npz');lp=arrays(folder/'fixed_lp_1.npz');same(lp,arrays(folder/'fixed_lp_2.npz'))
        assert bool(lp['success'])and int(lp['status'])==0
        xn=native['integer_solution'];states=native['integer_states'];assert states.shape==(3,168)and np.isin(states,[0,1]).all()
        assert np.isfinite(xn).all();np.testing.assert_array_equal(np.rint(xn[indices]),states.ravel())
        assert np.max(np.abs(xn[indices]-states.ravel()))<=1e-5
        assert max(np.max(np.abs(problem['A_eq']@xn-problem['b_eq'])),np.max(problem['A_ub']@xn-problem['b_ub']),
            np.max(problem['bounds'][:,0]-xn),np.max(xn-problem['bounds'][:,1]))<=1e-5
        fixed={**problem,'bounds':problem['bounds'].copy()};fixed['bounds'][indices,0]=fixed['bounds'][indices,1]=states.ravel()
        np.testing.assert_array_equal(lp['solution'][indices],states.ravel())
        kkt=check_lp(fixed,{('x'if name=='solution'else'objective_scaled'if name=='objective'else name):v for name,v in lp.items()})
        native_objective=proof['mip_objective_scaled'];bound=proof['mip_dual_bound_scaled']
        assert np.isfinite([native_objective,bound]).all()and bound<=native_objective+1e-7
        np.testing.assert_allclose(problem['c']@xn,native_objective,atol=1e-5,rtol=2e-11)
        exact=independent_exact(fixed,xn,lp,native_objective,scale);cert=observed['checks']['certificate']
        assert cert==certificate['replicas'][replica-1]['certificate']
        for key,value in exact['parts'].items():assert value==decode(cert['parts_scaled'][key])
        for key,savedkey in [('exact_difference','exact_dot_difference_scaled'),('reported_difference','reported_difference_scaled'),
            ('bound','bound_scaled'),('euro_bound','bound_eur'),('max_stationarity','maximum_exact_stationarity_residual')]:
            assert exact[key]==decode(cert[savedkey])
        assert exact['euro_bound']<=Fraction(1,100000)==decode(cert['absolute_eur_budget'])
        assert exact['reported_difference']>Fraction.from_float(1e-6)
        assert observed['lp_call_metadata']==[{'iterations':6577,'status':0}]*2
        reconstructed=decoded(model,problem,network,case,native,lp,6577)
        assert set(reconstructed)==set(result)
        for key,value in reconstructed.items():np.testing.assert_allclose(result[key],value,rtol=2e-11,atol=1e-5,err_msg=key)
        for key in ['minimum_slack','maximum_slack','bypass_slack','vest_slack']:assert np.max(np.abs(result[key]))==0
        assert result['objective']<=control['objective']+1e-4
        if previous is not None:same(previous[0],native);same(previous[1],lp)
        previous=native,lp;records.append({'replica':replica,'pid':record['pid'],'KKT':kkt,'exact_bound_eur':float(exact['euro_bound'])})
    return {'case_id':2012,'acceptance_basis':'case2012_dual_feasibility_certificate','original_strict_guard_still_fails':True,
            'exact_rational_certificate_and_all_decoded_fields_independently_verified':True,'replicas':records}


def exit_contract(observation: dict,original: dict,protocol: dict,manifest_sha: str,worker_pids: list[int]) -> None:
    """Require actual post-finalization scoped process evidence, not booleans alone."""
    assert observation['status']=='passed'and observation['original_cache_manifest_sha256']==manifest_sha
    assert observation['original_cache_protocol_sha256']==OLD_SHA
    assert observation['original_cache_path']==protocol['original_cache']
    assert protocol['original_coordinator_pid']==11014
    assert observation['expected_coordinator_pid']==11014 and observation['expected_worker_pids']==sorted(set(worker_pids))
    command="pgrep -fl 'multiprocessing.spawn|continue_integer_water_budget_cache_v2'"
    assert observation['command']==protocol['original_exit_process_command']==command
    assert observation['returncode']in [0,1]
    live={int(line.split()[0])for line in observation['scoped_process_output'].splitlines()if line.strip()}
    assert (observation['returncode']==0)==bool(live)
    assert not set(worker_pids)&live and 11014 not in live
    assert datetime.fromisoformat(observation['timestamp_utc'])>=datetime.fromisoformat(original['timestamp_utc'])+timedelta(seconds=original['runtime_seconds'])
    assert observation['coordinator_exited']is True and observation['all_workers_exited']is True


def main() -> None:
    ap=argparse.ArgumentParser();ap.add_argument('--cache',type=Path,required=True);ap.add_argument('--manifest-sha256',required=True)
    ap.add_argument('--protocol-sha256',required=True);ap.add_argument('--source-audit',type=Path,required=True)
    ap.add_argument('--source-audit-sha256',required=True);ap.add_argument('--run-id',required=True);a=ap.parse_args()
    assert Path(a.run_id).name==a.run_id and a.run_id not in ['.','..']
    out=ROOT/'artifacts'/a.run_id;out.mkdir(exist_ok=False);start=time.monotonic()
    report={'status':'initializing','timestamp_utc':datetime.now(timezone.utc).isoformat(),'command':[sys.executable,*sys.argv],
        'optimization_calls':0,'labels_decoded':False,'training':False,'original_cache_still_ineligible':True,'cases':[]}
    try:
        assert EXPECTED_HELPERS
        for name,h in EXPECTED_HELPERS.items():assert digest(ROOT/name)==h,name
        cache=a.cache.resolve();assert digest(cache/'manifest.json')==a.manifest_sha256
        assert digest(cache/'protocol.json')==a.protocol_sha256==AGGREGATE_PROTOCOL_SHA;p=read(cache/'protocol.json');m=read(cache/'manifest.json')
        assert m['status']=='complete'and not m['failures']and not m['labels_decoded']and not m['training']
        assert m['protocol_sha256']==a.protocol_sha256 and m['all_cases_accepted_and_replayed']
        assert m['all_cases_native_strict_gate_passed']is False and m['original_cache_still_ineligible']is True
        group=groups(m,p)
        assert p['expected_counts']=={'ordinary':3010,'imported':636,'exception':1}
        for key in ['source','data','environment','case_ids','case_dates','feature_specification','prefix_columns']:
            assert m[key]==p[key]
        assert set(m['completed_cases'])=={str(cid)for cid in p['case_ids']}
        for g in ['source','data']:hashes(ROOT,p[g])
        hashes(cache/'source',p['source']);hashes(cache,m['artifacts'])
        inv=inventory(cache);inv.pop('manifest.json');inv.pop('experiment_record.md',None);assert inv==m['artifacts']
        assert digest(a.source_audit)==a.source_audit_sha256==m['independent_source_audit_sha256']
        review=read(a.source_audit);assert review['status']=='passed'and review['protocol_sha256']==a.protocol_sha256
        assert digest(ROOT/p['original_protocol'])==digest(cache/'original_execution_protocol.json')==OLD_SHA
        old=read(cache/'original_execution_protocol.json')
        assert digest(ROOT/p['original_base_protocol'])==digest(cache/'compatible_base_protocol.json')==BASE_SHA
        base=read(cache/'compatible_base_protocol.json');mutable={'version','created_utc','workers','source','resume_policy'}
        assert set(old)==set(base)|{'compatible_base','execution'}
        for key in set(base)-mutable:assert old[key]==base[key]
        for name,h in base['source'].items():assert old['source'][name]==h
        for g in ['source','data','audits']:hashes(ROOT,old[g])
        for key in ['data','environment','case_ids','case_dates','feature_specification','prefix_columns']:
            assert p[key]==old[key]
        assert old['targets']==TARGETS and old['workers']==4 and old['solver_threads']==1
        assert digest(ROOT/p['prefix_protocol'])==p['prefix_protocol_sha256']
        assert p['prefix_columns']==read(ROOT/p['prefix_protocol'])['feature_columns']and len(p['prefix_columns'])==664
        source=ROOT/p['original_cache'];old_manifest_sha=m['original_cache_manifest_sha256']
        assert digest(source/'manifest.json')==digest(cache/'original_cache_manifest.json')==old_manifest_sha
        original=read(cache/'original_cache_manifest.json');assert original['status']=='incomplete'and original['protocol_sha256']==OLD_SHA
        for key in ['source','data','environment','feature_specification','prefix_columns']:assert original[key]==old[key]
        assert original['workers']==4 and original['base_protocol_sha256']==BASE_SHA
        assert digest(source/'protocol.json')==OLD_SHA and digest(source/'compatible_base_protocol.json')==BASE_SHA
        hashes(source/'source',old['source'])
        assert original['remaining_queued_attempts']==0 and not original['graceful_drain']['requested']
        assert original['case_ids']==p['case_ids']and original['case_dates']==p['case_dates']
        assert set(original['completed_cases'])=={str(cid)for cid in p['case_ids']if cid!=2012}
        assert len(original['failures'])==2 and {(f['case_id'],f['replica'],f['stage'])for f in original['failures']}=={(2012,1,'attempt_ineligible'),(2012,2,'attempt_ineligible')}
        assert m['retained_original_failures']==original['failures']and original['reuse_manifest_sha256']==IMPORT_SHA
        assert not original['labels_decoded']and not original['training'];hashes(source,original['artifacts'])
        exitpath=cache/'original_exit_observation.json';assert digest(exitpath)==m['original_exit_observation_sha256']
        exitproof=read(exitpath);assert exitproof['status']=='passed'and exitproof['original_cache_manifest_sha256']==old_manifest_sha
        assert exitproof['coordinator_exited']and exitproof['all_workers_exited']
        worker_pids=[read(safe(source,attempt['directory'])/'proof.json')['pid']for attempt in original['attempts']]
        exit_contract(exitproof,original,p,old_manifest_sha,worker_pids)
        assert digest(ROOT/p['imported_manifest'])==IMPORT_SHA
        imports=read(ROOT/p['imported_manifest']);old_imports=(ROOT/p['imported_manifest']).parent
        assert imports['protocol_sha256']==BASE_SHA and imports['status']=='failed'
        for key in ['source','data','feature_specification']:assert imports[key]==base[key]
        assert digest(source/'reuse_manifest.json')==IMPORT_SHA
        assert group['imported']==set(map(int,imports['completed_cases']))==set(original['resumed_case_ids'])
        pilot=ROOT/old['pilot_directory'];assert digest(pilot/'manifest.json')==old['pilot_manifest_sha256']
        assert digest(pilot/'protocol.json')==old['pilot_protocol_sha256'];hashes(pilot,old['pilot_files'])
        for canonical in old['canonical_diagnostic'].values():
            for item in canonical.values():assert digest(ROOT/item['path'])==item['sha256']
        continuous=ROOT/p['continuous_cache'];assert digest(continuous/'manifest.json')==p['continuous_cache_manifest_sha256']
        controls=read(continuous/'manifest.json');assert controls['case_ids']==p['case_ids']and controls['case_dates']==p['case_dates']
        completion=cache/'exception_evidence/completion';assert digest(completion/'manifest.json')==p['completion_manifest_sha256']==m['completion_manifest_sha256']==COMPLETION_SHA
        assert inventory(completion)==inventory(ROOT/p['completion_directory'])
        cm=read(completion/'manifest.json');cp=read(completion/'protocol.json')
        assert digest(cache/'exception_evidence/actual_audit.json')==digest(ROOT/p['exception_actual_audit'])==p['exception_actual_audit_sha256']==m['exception_actual_audit_sha256']==EXCEPTION_AUDIT_SHA
        accepted=read(cache/'exception_evidence/actual_audit.json')
        assert accepted['status']=='passed'and accepted['completion_manifest_sha256']==COMPLETION_SHA and accepted['case2012_exception_independently_accepted']
        assert accepted['all_workers_exited']and accepted['original_failures_preserved']and not accepted['old_v2_eligible']
        target=(ROOT/'sintef_project/data/kernel/Unit_commitment_decisions.csv').resolve();reader=pd.read_csv
        def guarded(path,*args,**kwargs):
            if isinstance(path,(str,Path))and Path(path).resolve()==target:assert not args and kwargs=={'usecols':['Run No','starttime']}
            return reader(path,*args,**kwargs)
        with patch.object(pd,'read_csv',guarded):
            data=HydroCaseData(ROOT);assert data.cases['Run No'].tolist()==p['case_ids']and data.cases.starttime.astype(str).tolist()==p['case_dates']
            solver=ShopRelaxationLP(data.network);rebuilt=np.empty((3647,168,84),dtype=np.float32)
            for pos,cid in enumerate(p['case_ids']):
                folder=cache/'cases'/f'case_{cid:04d}';rp=folder/'aggregate_receipt.json'
                assert digest(rp)==m['completed_cases'][str(cid)];r=read(rp)
                kind=next(k for k,ids in group.items()if cid in ids);receipt_contract(folder,r,cid,kind,a.protocol_sha256,old_manifest_sha)
                source_case=source/'cases'/f'case_{cid:04d}';assert r['original_source_case_path']==str(source_case.relative_to(ROOT))
                assert inventory(source_case)==r['original_case_inventory']
                copied=folder/'original_case'
                case=data.case(pos);case['natural']=np.asarray(case['natural'],dtype=np.float64)
                for key,field in [('pmin','p_min'),('start_cost','startcost_const'),('stop_cost','stopcost_const')]:
                    case[key]=np.asarray([data.model['generator'][unit][field]for unit in UNIT_ORDER],dtype=np.float64)
                ip=continuous/f'case_{cid:04d}_inputs.npz';cpfile=continuous/f'case_{cid:04d}_result.npz'
                assert digest(ip)==r['input_sha256']==controls['artifacts'][ip.name]
                assert digest(cpfile)==r['control_sha256']==controls['artifacts'][cpfile.name]
                same(case,arrays(ip));same(case,arrays(folder/r['input_path']));control=arrays(cpfile)
                if kind!='exception':
                    oldreceipt=read(copied/'complete.json');assert oldreceipt['case_id']==cid and oldreceipt['protocol_sha256']==OLD_SHA
                    assert r['original_complete_sha256']==original['completed_cases'][str(cid)]
                    if kind=='imported':compatibility(copied,oldreceipt,cid,old_imports,imports);assert r['compatibility']==oldreceipt['compatibility']
                    else:assert 'compatibility'not in oldreceipt
                    canonical=old['canonical_diagnostic'].get(str(cid))
                    if canonical:
                        assert oldreceipt['canonical_origin']=={'kind':'preserved_diagnostic','files':canonical}
                        for key,name in [('result','canonical_result.npz'),('proof','canonical_proof.json')]:assert digest(copied/name)==canonical[key]['sha256']
                    else:
                        first=oldreceipt['accepted_attempts'][0];assert oldreceipt['canonical_origin']=={'kind':'first_accepted_replica','directory':first}
                        assert digest(copied/'canonical_proof.json')==digest(copied/first/'proof.json')
                    rebuilt[pos],record=audit_case(copied,oldreceipt,case,control,solver,r['accepted_attempt_protocol_sha256'],old)
                else:
                    ex=r['exception'];assert ex['completion_manifest_sha256']==COMPLETION_SHA and ex['actual_audit_sha256']==EXCEPTION_AUDIT_SHA
                    assert ex['completion_evidence_path']=='exception_evidence/completion'and ex['actual_audit_evidence_path']=='exception_evidence/actual_audit.json'
                    assert ex['original_status_preserved']=='failed'and ex['original_objective_guard_still_fails']
                    assert ex['native_status']==0 and ex['conditional_replays_verified']and ex['feature_witness_is_supplemental_not_a_rewritten_proof']
                    expected_failures=[{'replica':i,'path':f'original_case/replica_{i}_60s/proof.json','sha256':digest(copied/f'replica_{i}_60s/proof.json')}for i in [1,2]]
                    assert ex['original_failed_proofs']==expected_failures
                    assert digest(folder/r['result_path'])==digest(completion/'exception_case2012_result.npz')==accepted['result_sha256']
                    result=arrays(folder/r['result_path']);same(result,arrays(completion/'exception_case2012_result.npz'))
                    record=exception_certificate(case,control,data.network,result,completion,cm,cp,copied)
                    rebuilt[pos]=independent_features(case,result,control,data.network)
                    same({'features':rebuilt[pos]},{'features':np.load(folder/r['features_path'],allow_pickle=False)})
                report['cases'].append({**record,'kind':kind,'original_case_all_bytes_preserved':True})
                if(pos+1)%100==0:print(json.dumps({'audited_cases':pos+1,'seconds':time.monotonic()-start}),flush=True)
        feature=cache/'features.npz';assert digest(feature)==m['features_sha256']==m['artifacts']['features.npz']
        saved=arrays(feature);assert set(saved)=={'features','case_ids','columns','units'}
        assert saved['case_ids'].tolist()==p['case_ids']and saved['columns'].tolist()==COLUMNS and saved['units'].tolist()==UNIT_ORDER
        same({'features':rebuilt},{'features':saved['features']})
        assert digest(source/'manifest.json')==old_manifest_sha and digest(cache/'manifest.json')==a.manifest_sha256
        report.update(status='passed',cache_manifest_sha256=a.manifest_sha256,cache_protocol_sha256=a.protocol_sha256,
            features_sha256=m['features_sha256'],case_count=3647,all_cases_physical_and_replay_verified=True,
            compatible_receipts_verified=True,compatible_case_count=636,ordinary_case_count=3010,imported_case_count=636,
            exception_count=1,exception_case_ids=[2012],completion_manifest_sha256=COMPLETION_SHA,exception_actual_audit_sha256=EXCEPTION_AUDIT_SHA,
            all_84_features_independently_reconstructed=True,original_workers_and_coordinator_exited=True,
            limitation='Ordinary/imported cases retain the original native and conditional-LP contract; only case2012 uses its separately declared exact one-sided objective certificate. Original failed cache remains ineligible.')
    except BaseException:report.update(status='failed',error=traceback.format_exc());raise
    finally:
        shutil.copy2(__file__,out/Path(__file__).name);report['runtime_seconds']=time.monotonic()-start
        report['helpers']={name:digest(ROOT/name)for name in EXPECTED_HELPERS}
        report['artifacts']={q.name:digest(q)for q in out.iterdir()if q.is_file()and q.name!='manifest.json'}
        (out/'manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps({'status':report['status'],'manifest_sha256':digest(out/'manifest.json')}),flush=True)

if __name__=='__main__':main()
