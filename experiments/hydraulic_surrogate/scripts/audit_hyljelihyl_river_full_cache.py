"""Independently audit a complete river cache; no optimization or target decoding."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone,timedelta
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_hyljelihyl_river_cost_v2 import sha,literal_arrays,load_arrays,projection_matrix
from scripts.audit_hyljelihyl_persistent_pilot import independent_features
from scripts.audit_hyljelihyl_persistent_result import decoded_result
from scripts.audit_hyljelihyl_river_results import check_lp
from scripts.diagnose_hyljelihyl_river_cost_v2 import load_network,signatures
from sintef_project.lp_hyljelihyl_v2 import make_problem
from sintef_project.hyljelihyl_river_features import COLUMNS,specification
from sintef_project.lp_features import UNIT_ORDER

PROTOCOL_SHA='e4b7bdf6954838739174f767a0a69ab110317fef069e646ac8725ab8cc9e187e'
HELPERS=['scripts/audit_hyljelihyl_river_cost_v2.py','scripts/audit_hyljelihyl_persistent_pilot.py',
         'scripts/audit_hyljelihyl_persistent_result.py','scripts/audit_hyljelihyl_river_results.py',
         'scripts/diagnose_hyljelihyl_river_cost_v2.py','sintef_project/lp_hyljelihyl_v2.py',
         'sintef_project/hyljelihyl_river_features.py','sintef_project/lp_features.py',
         'scripts/audit_integer_water_budget_cache.py']
EXPECTED_HELPERS={'scripts/audit_hyljelihyl_river_cost_v2.py': '86021f18454a0651c6ad8faf3bbb1633c106635c43fc70b37181ea8c3f68c488', 'scripts/audit_hyljelihyl_persistent_pilot.py': '17f4b8e3072e1df76816b3a73e458ae7eacff4ccac99d6d0870668f47066a52f', 'scripts/audit_hyljelihyl_persistent_result.py': 'ef67c0d49f70622d2ef5eeba5bdbc1dacb32a59c0434a7726be3c87ff6f27b49', 'scripts/audit_hyljelihyl_river_results.py': 'fa9e397689fe0cc32b1faabecaa33d93593b2c33e82270e3e947385d13b43c42', 'scripts/diagnose_hyljelihyl_river_cost_v2.py': '398f8ac785035dd6183b9dfa6a408b8fefe59797baf6807773ea90894eec16b1', 'sintef_project/lp_hyljelihyl_v2.py': 'd4362e0b174536f8e912461f93d9945666017612605b2b67b060579580770193', 'sintef_project/hyljelihyl_river_features.py': '6746028a301a15e007156a485c44c32366f36fcd225599a0a162987de4d4104b', 'sintef_project/lp_features.py': 'bd16e0b4e9e32976b7eb50d5334e05b94f2b074bc0d8eb83a787f26f95268720', 'scripts/audit_integer_water_budget_cache.py': '78ff7efe84b00129bafa112c9a2a6be8149100f99f456313af8178788f9660e9'}


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def safe(root: Path,name: str) -> Path:
    """Accept only canonical relative in-root paths without aliases or escapes."""
    p=Path(name)
    assert name and not p.is_absolute()and '..'not in p.parts and str(p)==name
    result=root/p;assert result.resolve().is_relative_to(root.resolve())
    assert not root.is_symlink()
    for i in range(1,len(p.parts)+1):assert not root.joinpath(*p.parts[:i]).is_symlink()
    return result


def hashes(root: Path,values: dict) -> None:
    for name,h in values.items():assert sha(safe(root,name))==h,name


def receipt_paths(cache: Path,r: dict,cid: int,protocol_sha: str,input_record: dict) -> list[Path]:
    """Bind exact two-replica payloads and the feature pointer independently."""
    assert r['status']=='complete'and r['case_id']==cid and r['protocol_sha256']==protocol_sha
    assert r['input_record']==input_record and len(r['replica_paths'])==len(set(r['replica_paths']))==2
    paths=[safe(cache,n)for n in r['replica_paths']]
    assert r['features_path']==str(Path(r['replica_paths'][0])/'features.npy')
    assert sha(safe(cache,r['features_path']))==r['features_sha256']
    exact={str(f.relative_to(cache)):sha(f)for path in paths for f in path.iterdir()}
    assert exact==r['artifacts']
    assert r['replays_byte_exact']and len(r['pids'])==len(set(r['pids']))==2
    return paths


def phase_lineage(r: dict,cid: int,launch: dict,phase_index: int,proofs: list[dict],pre_sha: str,p_sha: str) -> None:
    """Reject swapped phase replicas, PID/order changes and other input seals."""
    assert r['origin']=={'type':'phase','phase':phase_index,'input_preflight_sha256':pre_sha}
    assert launch['protocol_sha256']==p_sha and launch['input_preflight_sha256']==pre_sha
    assert len(set(launch['worker_pids']))==2 and r['pids']==launch['worker_pids']
    for i,proof in enumerate(proofs,1):
        assert r['replica_paths'][i-1]==f'phases/phase_{phase_index:03d}/worker_{i}/case_{cid:04d}'
        assert cid in launch['orders'][i-1]
        assert proof['pid']==launch['worker_pids'][i-1]and proof['protocol_sha256']==p_sha
        assert proof['input_preflight_sha256']==pre_sha and proof['input_record']==r['input_record']


def pilot_lineage(cache: Path,r: dict,cid: int,p: dict,pilot_manifest: dict) -> None:
    """Verify original pilot copies, including exact worker identity and proof bytes."""
    assert cid in p['pilot_case_ids']
    assert r['pids']==pilot_manifest['worker_pids']
    assert r['origin']=={'type':'pilot_import','pilot_manifest_sha256':p['pilot_manifest_sha256'],
        'worker_manifests_sha256':p['pilot_worker_manifests_sha256'],
        'canonical_fresh_process_lineage':cid in p['canonical_case_ids']}
    for i in [1,2]:
        assert r['replica_paths'][i-1]==f'imports/worker_{i}/case_{cid:04d}'
        path=cache/r['replica_paths'][i-1]
        for f in path.iterdir():assert sha(f)==pilot_manifest['artifacts'][f'worker_{i}/case_{cid:04d}/{f.name}']


def check_resource(value: dict,free: float,swap: int|None=None) -> None:
    assert value['free_percent']>=free and value['pressure_derived_free_bytes']>=3*1024**3
    if swap is not None:assert value['swapouts']<=swap


def main() -> None:
    """Require every receipt, both saved calls, all KKT/decoded/feature checks and exits."""
    ap=argparse.ArgumentParser();ap.add_argument('--cache',type=Path,required=True)
    ap.add_argument('--manifest-sha256',required=True);ap.add_argument('--source-audit',type=Path,required=True)
    ap.add_argument('--source-audit-sha256',required=True);ap.add_argument('--exit-observation',type=Path,required=True)
    ap.add_argument('--exit-observation-sha256',required=True);ap.add_argument('--run-id',required=True)
    a=ap.parse_args();assert Path(a.run_id).name==a.run_id and a.run_id not in ['.','..']
    out=ROOT/'artifacts'/a.run_id;out.mkdir(exist_ok=False);start=time.monotonic()
    report={'status':'initializing','timestamp_utc':datetime.now(timezone.utc).isoformat(),'command':[sys.executable,*sys.argv],
        'optimization_calls':0,'labels_decoded':False,'training':False,'cases':[]}
    try:
        for name,h in EXPECTED_HELPERS.items():assert sha(ROOT/name)==h,name
        assert not a.cache.is_symlink()
        cache=a.cache.resolve();assert sha(cache/'manifest.json')==a.manifest_sha256
        m=read(cache/'manifest.json');p=read(cache/'protocol.json')
        assert sha(cache/'protocol.json')==m['protocol_sha256']==PROTOCOL_SHA
        assert m['status']=='complete'and not m['failures']and not m['labels_decoded']and not m['training']
        assert m['all_workers_exited']and m['all_cases_physical_and_replay_verified']
        for key in ['source','data','case_ids','case_dates','prefix_columns','feature_specification','environment','replay_execution_type']:
            assert m[key]==p[key]
        assert len(p['case_ids'])==len(set(p['case_ids']))==3647
        assert p['orders']==[p['case_ids'],np.random.default_rng(4173).permutation(p['case_ids']).tolist()]
        assert p['feature_specification']==specification()and len(p['prefix_columns'])==664
        for group in ['source','data']:hashes(ROOT,p[group])
        hashes(cache,m['artifacts'])
        actual_files={str(f.relative_to(cache))for f in cache.rglob('*')if f.is_file()and f not in [cache/'manifest.json',cache/'experiment_record.md']}
        assert actual_files==set(m['artifacts'])
        for name in p['source']:assert sha(cache/'source'/name)==p['source'][name]
        assert sha(ROOT/p['prefix_protocol'])==p['prefix_protocol_sha256']
        assert p['prefix_columns']==read(ROOT/p['prefix_protocol'])['feature_columns']
        assert sha(ROOT/p['continuous_manifest'])==p['continuous_manifest_sha256']
        cm=read(ROOT/p['continuous_manifest']);assert cm['case_ids']==p['case_ids']and cm['case_dates']==p['case_dates']
        pre=ROOT/m['input_preflight'];assert sha(pre/'manifest.json')==m['input_preflight_manifest_sha256']
        pm=read(pre/'manifest.json');assert pm['status']=='passed'and pm['optimization_calls']==0
        compatibility=p['compatible_input_preflight']
        for key in ['protocol','manifest','input_audit','blocked_launch_review']:
            assert sha(ROOT/compatibility[key])==compatibility[key+'_sha256']
        assert pm['protocol_sha256']==compatibility['protocol_sha256'] and not pm['labels_decoded']
        assert sha(pre/'manifest.json')==compatibility['manifest_sha256']
        assert pre.resolve()==(ROOT/compatibility['manifest']).parent.resolve()
        hashes(pre,pm['artifacts']);inputs={r['case_id']:r for r in pm['cases']};assert list(inputs)==p['case_ids']
        assert sha(a.source_audit)==a.source_audit_sha256==m['independent_source_audit_sha256']
        source_audit=read(a.source_audit)
        assert source_audit['status']=='passed'and source_audit['protocol_sha256']==PROTOCOL_SHA
        assert source_audit['input_preflight_manifest_sha256']==m['input_preflight_manifest_sha256']and source_audit['case_count']==3647
        hashes(a.source_audit.parent,source_audit['artifacts'])
        pilot=ROOT/p['pilot_directory'];assert sha(pilot/'manifest.json')==p['pilot_manifest_sha256']
        pilot_manifest=read(pilot/'manifest.json');assert pilot_manifest['status']=='passed'
        assert sha(ROOT/p['pilot_audit'])==p['pilot_audit_sha256'];pilot_audit=read(ROOT/p['pilot_audit'])
        assert pilot_audit['status']=='passed'and pilot_audit['result_manifest_sha256']==p['pilot_manifest_sha256']
        assert pilot_audit['all_cases_physical_and_replay_verified']and pilot_audit['canonical_reuse_verified']and pilot_audit['workers_exited']
        assert sha(ROOT/p['pilot_protocol'])==p['pilot_protocol_sha256'];pilot_protocol=read(ROOT/p['pilot_protocol'])
        assert p['pilot_case_ids']==[x['case_id']for x in pilot_protocol['cases']]
        for i in [1,2]:
            assert sha(pilot/f'worker_{i}/manifest.json')==p['pilot_worker_manifests_sha256'][i-1]
            pw=read(pilot/f'worker_{i}/manifest.json')
            assert pw['pid']==pilot_manifest['worker_pids'][i-1]and pw['replica']==i and pw['status']=='complete'
            assert pw['order']==pilot_protocol['orders'][i-1]and 0<pw['peak_rss_bytes']<2*1024**3
        canonical=ROOT/p['canonical_directory'];assert sha(canonical/'manifest.json')==p['canonical_manifest_sha256']
        assert sha(ROOT/p['canonical_audit'])==p['canonical_audit_sha256'];ca=read(ROOT/p['canonical_audit'])
        assert ca['status']=='passed'and ca['result_manifest_sha256']==p['canonical_manifest_sha256']
        canonical_manifest=read(canonical/'manifest.json')
        assert set(m['completed_cases'])=={str(cid)for cid in p['case_ids']}
        assert {q.parent.name for q in (cache/'cases').glob('case_*/complete.json')}=={f'case_{cid:04d}'for cid in p['case_ids']}
        receipts={cid:read(cache/'cases'/f'case_{cid:04d}'/'complete.json')for cid in p['case_ids']}
        for cid,r in receipts.items():assert sha(cache/'cases'/f'case_{cid:04d}'/'complete.json')==m['completed_cases'][str(cid)]
        imported={cid for cid,r in receipts.items()if r['origin']['type']=='pilot_import'}
        assert imported==set(p['pilot_case_ids'])and m['imported_case_count']==59
        # Prior manifests are retained verbatim. Only STOP_AFTER_ACTIVE is archived on resume.
        priors={sha(q):read(q)for q in cache.glob('prior_manifest_*.json')}
        chain=[];previous=m.get('resumed_manifest_sha256')
        while previous is not None:
            assert previous in priors and previous not in chain;prior=priors[previous];chain.append(previous)
            assert prior['protocol_sha256']==PROTOCOL_SHA and prior['status']=='incomplete_resource_drain'
            assert prior['all_workers_exited']and not prior['failures']
            for name,h in prior['artifacts'].items():
                q=cache/f'prior_stop_{previous[:12]}'if name=='STOP_AFTER_ACTIVE'else safe(cache,name)
                assert sha(q)==h,name
            for cid,h in prior['completed_cases'].items():assert m['completed_cases'][cid]==h
            previous=prior.get('resumed_manifest_sha256')
        assert set(chain)==set(priors)
        assert not list((cache/'phases').glob('phase_*/pair_failures/*.json'))
        phases={};workers={};all_pids=[]
        phase_paths=sorted((cache/'phases').glob('phase_*'))
        assert phase_paths and [q.name for q in phase_paths]==[f'phase_{i:03d}'for i in range(1,len(phase_paths)+1)]
        for index,folder in enumerate(phase_paths,1):
            launch=read(folder/'launch.json');phases[index]=launch
            initial=sorted(cid for cid,r in receipts.items()if r['origin']['type']=='pilot_import'or r['origin']['phase']<index)
            assert launch['initial_complete_cases']==initial
            assert launch['orders']==[[cid for cid in order if cid not in initial]for order in p['orders']]
            assert launch['protocol_sha256']==PROTOCOL_SHA and launch['input_preflight_sha256']==m['input_preflight_manifest_sha256']
            assert len(set(launch['worker_pids']))==2;all_pids.extend(launch['worker_pids'])
            for i in [1,2]:
                worker=read(folder/f'worker_{i}/manifest.json');workers[index,i]=worker
                assert worker['status']in ['complete','drained']and worker['pid']==launch['worker_pids'][i-1]and worker['replica']==i
                assert worker['protocol_sha256']==PROTOCOL_SHA and worker['input_preflight_sha256']==launch['input_preflight_sha256']
                assert worker['environment']==p['environment']and worker['order']==launch['orders'][i-1]
                actual_ids=[r['case_id']for r in worker['cases']];assert actual_ids==worker['order'][:len(actual_ids)]
                if worker['status']=='complete':assert actual_ids==worker['order']
                paths=list((folder/f'worker_{i}').glob('case_*/proof.json'))
                assert {int(q.parent.name.split('_')[1])for q in paths}==set(actual_ids)
                for proof in worker['cases']:
                    assert proof['status']=='complete'
                    assert read(folder/f'worker_{i}'/f"case_{proof['case_id']:04d}"/'proof.json')==proof
        observation=read(a.exit_observation);assert sha(a.exit_observation)==a.exit_observation_sha256
        assert observation['status']=='passed'and observation['returncode']in [0,1]
        assert observation['cache_manifest_sha256']==a.manifest_sha256 and observation['cache_protocol_sha256']==PROTOCOL_SHA
        assert observation['cache_directory']==str(cache.relative_to(ROOT))
        assert datetime.fromisoformat(observation['timestamp_utc'])>=datetime.fromisoformat(m['timestamp_utc'])+timedelta(seconds=m['runtime_seconds'])
        assert sorted(observation['expected_worker_pids'])==sorted(set(all_pids))
        live={int(line.split()[0])for line in observation['scoped_process_output'].splitlines()if line.strip()}
        assert not set(all_pids)&live
        last=phases[len(phases)];assert m['phase_worker_pids']==last['worker_pids']and m['worker_exitcodes']==[0,0]
        assert m['shutdown']['all_workers_exited']and m['shutdown']['workers']==[{'pid':pid,'exitcode':0}for pid in last['worker_pids']]
        check_resource(m['initial_resources'],30);check_resource(m['entry_resource_recheck'],30,m['initial_resources']['swapouts'])
        assert sha(cache/'features.npz')==m['features_sha256']==m['artifacts']['features.npz']
        with np.load(cache/'features.npz',allow_pickle=False)as z:
            assert set(z.files)=={'features','case_ids','columns','units'}
            assert z['case_ids'].tolist()==p['case_ids']and z['columns'].tolist()==COLUMNS and z['units'].tolist()==UNIT_ORDER
            aggregate=z['features']
        assert aggregate.shape==(3647,168,62)and aggregate.dtype==np.float32 and np.isfinite(aggregate).all()
        max_rss=0;calls=0
        for position,cid in enumerate(p['case_ids']):
            r=receipts[cid];paths=receipt_paths(cache,r,cid,PROTOCOL_SHA,inputs[cid]);proofs=[read(q/'proof.json')for q in paths]
            if r['origin']['type']=='pilot_import':pilot_lineage(cache,r,cid,p,pilot_manifest)
            else:
                index=r['origin']['phase'];phase_lineage(r,cid,phases[index],index,proofs,m['input_preflight_manifest_sha256'],PROTOCOL_SHA)
                for i,proof in enumerate(proofs,1):assert proof in workers[index,i]['cases']
            directory=pre/'inputs'/f'case_{cid:04d}';hashes(directory,inputs[cid]['files'])
            case,split=load_arrays(directory/'A_inputs.npz'),load_arrays(directory/'split_inputs.npz')
            oldnet=load_network(read(directory/'A_network.json'));newnet=load_network(read(directory/'split_network.json'))
            oldmodel,pa,_=make_problem(oldnet,case,'A');model,problem,segments=make_problem(newnet,split,'C',20.)
            assert signatures(problem)==inputs[cid]['matrix_signatures']
            reference_path=ROOT/p['continuous_cache']/f'case_{cid:04d}_result.npz'
            assert sha(reference_path)==inputs[cid]['reference_sha256']==cm['artifacts'][reference_path.name]
            reference=load_arrays(reference_path);copies=[]
            for folder,proof in zip(paths,proofs,strict=True):
                assert proof['case_id']==cid and proof['status']=='complete'and proof['native']['status']==0 and proof['all_slacks_zero']
                assert proof['pid']in r['pids']
                assert set(proof['artifacts'])=={'result.npz','primal_dual.npz','features.npy','native_output.npz'}
                assert {q.name for q in folder.iterdir()}==set(proof['artifacts'])|{'proof.json'};hashes(folder,proof['artifacts'])
                if r['origin']['type']=='pilot_import':
                    check_resource(proof['resources'],30,pilot_manifest['initial_resources']['swapouts'])
                else:
                    phase_index=r['origin']['phase']
                    phase_manifest=m if phase_index==len(phases)else next(v for v in priors.values()if v.get('phase_worker_pids')==phases[phase_index]['worker_pids'])
                    check_resource(proof['resources'],20,phase_manifest['initial_resources']['swapouts'])
                    assert 0<proof['peak_rss_bytes']<2*1024**3;max_rss=max(max_rss,proof['peak_rss_bytes'])
                saved=load_arrays(folder/'primal_dual.npz');literal_arrays(saved,load_arrays(folder/'native_output.npz'))
                checks=check_lp(problem,saved)
                assert set(checks)==set(proof['kkt'])
                for key,value in checks.items():np.testing.assert_allclose(value,proof['kkt'][key],atol=1e-12,rtol=0)
                result=load_arrays(folder/'result.npz');expected=decoded_result(model,problem,segments,saved,split)
                assert set(result)==set(expected)
                for key,value in expected.items():np.testing.assert_allclose(result[key],value,rtol=2e-11,atol=1e-7,err_msg=key)
                for key in ['minimum_slack','maximum_slack','bypass_slack','vest_slack']:assert np.max(np.abs(result[key]))==0
                projected=projection_matrix(oldmodel,model)@saved['x'][:model.n_variables]
                assert max(np.max(np.abs(pa['A_eq']@projected-pa['b_eq'])),np.max(pa['A_ub']@projected-pa['b_ub']),
                    np.max(pa['bounds'][:,0]-projected),np.max(projected-pa['bounds'][:,1]))<=1e-5
                expected_features=independent_features(case,split,result,reference,oldnet,newnet)
                f=np.load(folder/'features.npy',allow_pickle=False)
                assert f.shape==(168,62)and f.dtype==np.float32 and np.isfinite(f).all()
                assert f.tobytes()==expected_features.tobytes()==aggregate[position].tobytes()
                copies.append((result,saved,f));calls+=1
            literal_arrays(copies[0][0],copies[1][0]);literal_arrays(copies[0][1],copies[1][1])
            assert copies[0][2].tobytes()==copies[1][2].tobytes()
            if cid in p['canonical_case_ids']:
                for k,name in [(0,'result.npz'),(1,'primal_dual.npz')]:
                    q=canonical/f'case_{cid:04d}/C_1'/name
                    assert sha(q)==canonical_manifest['artifacts'][str(q.relative_to(canonical))]
                    literal_arrays(copies[0][k],load_arrays(q))
            report['cases'].append({'case_id':cid,'origin':r['origin'],'full_LP_KKT_and_decoded_fields_verified':True,
                'both_replica_bytes_and_independent62_features_exact':True,'original_A_projection_feasible':True})
            if (position+1)%100==0:print(json.dumps({'audited_cases':position+1,'seconds':time.monotonic()-start}),flush=True)
        assert calls==7294
        report.update(status='passed',cache_manifest_sha256=a.manifest_sha256,cache_protocol_sha256=PROTOCOL_SHA,
            features_sha256=m['features_sha256'],case_count=3647,calls_checked=calls,all_cases_physical_and_replay_verified=True,
            canonical_reuse_verified=True,canonical_case_count=36,pilot_import_count=59,workers_exited=True,
            phase_count=len(phases),phase_lineage_and_orders_verified=True,immutable_receipts_verified=True,
            independent_features_byte_exact=True,maximum_new_worker_peak_rss_bytes=max_rss,
            source_audit_sha256=a.source_audit_sha256,worker_exit_observation_sha256=a.exit_observation_sha256,
            limitation='Numerical LP certificates use the frozen1e-5 KKT contract. This is an input-only feature cache audit; no label score or learned-model acceptance is inferred.')
    except BaseException:report.update(status='failed',error=traceback.format_exc());raise
    finally:
        shutil.copy2(__file__,out/Path(__file__).name);report['helpers']={name:sha(ROOT/name)for name in HELPERS}
        report['runtime_seconds']=time.monotonic()-start
        report['artifacts']={q.name:sha(q)for q in out.iterdir()if q.is_file()and q.name!='manifest.json'}
        (out/'manifest.json').write_text(json.dumps(report,indent=2))
        print(json.dumps({'status':report['status'],'manifest_sha256':sha(out/'manifest.json'),'seconds':report['runtime_seconds']}),flush=True)

if __name__=='__main__':main()
