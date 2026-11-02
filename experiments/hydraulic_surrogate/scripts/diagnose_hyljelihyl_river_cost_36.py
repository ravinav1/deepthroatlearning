"""Extend the fixed river diagnostic to its original36 metadata cases, with exact12 reuse."""

from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import shlex
import shutil
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

for variable in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[variable] = '1'

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sintef_project.lp_features import HydroCaseData, Network, UNIT_ORDER
from sintef_project.lp_hyljelihyl_v2 import (
    split_network, split_case, make_problem, capture_problem, solve,
    project_to_original, feasibility, BREAKPOINTS, MARGINAL)
import sintef_project.lp_hyljelihyl_v2 as river_module
from scripts.diagnose_integer_water_budget import arrays, digest
from scripts.diagnose_lp_shop_relaxation import metrics
from scripts.pilot_integer_water_budget import environment
from scripts.probe_all14_integer_water_budget_v2 import identical_arrays, all_slacks_exact_zero
from scripts.continue_integer_water_budget_cache_v2 import resource_state

PROTOCOL = ROOT / 'docs/hyljelihyl_river_cost_36_protocol.json'
NEW = ['scripts/diagnose_hyljelihyl_river_cost_36.py', 'tests/test_hyljelihyl_river_reuse.py',
       'docs/hyljelihyl_river_cost_36_diagnostic.md']


def json_network(network: Network) -> dict:
    """Serialize the small topology explicitly."""
    return {k: v.tolist() if isinstance(v,np.ndarray) else v for k,v in vars(network).items()}


def load_network(value: dict) -> Network:
    """Restore numeric and Boolean fields without pickle."""
    return Network(**{k:v if k in ['pools','units'] else np.asarray(v,dtype=bool if k.startswith('vest_') else int)
                      for k,v in value.items()})


def signatures(problem: dict) -> dict:
    """Pin full matrix structure and values without serializing solver objects."""
    fields = {k:np.asarray(problem[k]) for k in ['c','bounds','b_eq','b_ub']}
    for key in ['A_eq','A_ub']:
        matrix = problem[key].tocsc()
        fields.update({key+'_shape':np.asarray(matrix.shape),key+'_data':matrix.data,
                       key+'_indices':matrix.indices,key+'_indptr':matrix.indptr})
    return {k:hashlib.sha256(str((v.dtype.str,v.shape)).encode()+v.tobytes()).hexdigest() for k,v in fields.items()}


def freeze() -> None:
    """Use the original36 metadata IDs; preserve mathematics and twelve completed cases."""
    old_path=ROOT/'docs/hyljelihyl_river_cost_protocol_v2.json'
    old=json.loads(old_path.read_text())
    selection_path=ROOT/'docs/lp_shop_relaxation_protocol.json'
    selection=json.loads(selection_path.read_text())
    cache=json.loads((ROOT/old['cache']/'manifest.json').read_text())
    p=json.loads(json.dumps(old))
    p.update(version='hyljelihyl_directed_river_cost_36_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        cases=[{'case_id':i,'metadata_position':cache['case_ids'].index(i),
                'date':cache['case_dates'][cache['case_ids'].index(i)]} for i in selection['case_ids']],
        blocks=selection['date_blocks'],source={**old['source'],**{n:digest(ROOT/n) for n in NEW}},
        selection='Exact36 IDs/blocks from original relaxed-state protocol. Reused development data, no holdout claim or case selection from new errors.',
        revision='Population extension only; exact twelve approved directories reused,24 new cases. A/B/C math, caps, gates and metric formulas unchanged.',
        gate=old['gate'].replace('All12x3','All36x3'),
        target_rule=old['target_rule'].replace('twelve','thirty-six'))
    p['reuse']={'directory':'artifacts/hyljelihyl-river-cost-v2-diagnostic-001',
        'manifest_sha256':'fbb1ac2019fd0a8bf81695d7bc7959dc5d887c6149ecd55f74be60383e3ed401',
        'audit':'artifacts/hyljelihyl-river-cost-v2-result-independent-audit-001/manifest.json',
        'audit_sha256':'783a1d687a3e8e4a9833c0e7b23ca0a688738784630a0ed8aef7559594a25496',
        'case_ids':[c['case_id'] for c in old['cases']]}
    p['new_case_ids']=[c['case_id'] for c in p['cases'] if c['case_id'] not in p['reuse']['case_ids']]
    assert len(p['cases'])==36 and len(p['new_case_ids'])==24
    for q in [old_path,selection_path,ROOT/p['reuse']['directory']/'manifest.json',ROOT/p['reuse']['audit']]:
        p['inputs'][str(q.relative_to(ROOT))]=digest(q)
    for c in p['cases']:
        for suffix in ['inputs','result']:
            name=p['cache']+f'/case_{c["case_id"]:04d}_{suffix}.npz'
            assert digest(ROOT/name)==cache['artifacts'][Path(name).name]
            p['inputs'][name]=digest(ROOT/name)
    with PROTOCOL.open('x') as stream:json.dump(p,stream,indent=2)
    print(digest(PROTOCOL),flush=True)


def reuse_case(source: Path, destination: Path, case_id: int, manifest: dict) -> dict:
    """Copy only an exact fully manifested old case; reject missing, extra or changed files."""
    prefix=f'case_{case_id:04d}/'
    expected={name[len(prefix):]:h for name,h in manifest['artifacts'].items() if name.startswith(prefix)}
    folder=source/f'case_{case_id:04d}'
    actual={str(q.relative_to(folder)):digest(q) for q in folder.rglob('*') if q.is_file()}
    assert expected and actual==expected,'Original reused payload closure differs'
    shutil.copytree(folder,destination)
    copied={str(q.relative_to(destination)):digest(q) for q in destination.rglob('*') if q.is_file()}
    assert copied==expected,'Reused case bytes changed'
    return {'original_directory':str(folder.relative_to(ROOT)) if folder.is_relative_to(ROOT) else str(folder),
            'original_manifest_sha256':digest(source/'manifest.json'),
            'case_payload_hashes':copied,'all_original_bytes_preserved':True}


def worker(job_path: Path) -> None:
    """One fresh process, one arm, full saved solver output including failures."""
    job=json.loads(job_path.read_text());out=job_path.parent
    protocol=json.loads(Path(job['protocol']).read_text())
    assert digest(Path(job['protocol']))==job['protocol_sha256']
    for name,h in protocol['source'].items():assert digest(ROOT/name)==h,name
    for name in ['input','network']:
        assert digest(Path(job[name]))==job[name+'_sha256']
    case=arrays(Path(job['input']));net=load_network(json.loads(Path(job['network']).read_text()))
    start=time.monotonic();record={'status':'initializing','case_id':job['case_id'],'arm':job['arm'],
        'replica':job['replica'],'pid':os.getpid(),'environment':environment(),'job_sha256':digest(job_path)}
    native_call=river_module.linprog
    def retain_native(*positional,**kwargs):
        native=native_call(*positional,**kwargs)
        retained={}
        if native.x is not None:retained['x']=np.asarray(native.x)
        if native.fun is not None:retained['objective_scaled']=np.asarray(native.fun)
        for name in ['eqlin','ineqlin','lower','upper']:
            for field in ['marginals','residual']:
                value=getattr(getattr(native,name,None),field,None)
                if value is not None:retained[name+'_'+field]=np.asarray(value)
        np.savez_compressed(out/'native_output.npz',**retained)
        record['native']={'status':int(native.status),'success':bool(native.success),'message':str(native.message)}
        return native
    try:
        with patch.object(river_module,'linprog',retain_native):
            result,saved,proof=solve(net,case,job['arm'],protocol['lp_seconds'])
        np.savez_compressed(out/'result.npz',**result);np.savez_compressed(out/'primal_dual.npz',**saved)
        record.update(status='complete',kkt=proof,all_slacks_zero=all_slacks_exact_zero(result),
                      objective=float(result['objective']))
    except Exception:
        record.update(status='failed',error=traceback.format_exc())
    record.update(runtime_seconds=time.monotonic()-start,
        peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024))
    record['artifacts']={p.name:digest(p) for p in out.iterdir() if p.is_file()}
    (out/'proof.json').write_text(json.dumps(record,indent=2))
    if record['status']!='complete':raise RuntimeError(record['error'])


def read_labels(cases: list[dict]) -> np.ndarray:
    """Decode only the frozen legal rows, after physical seal creation."""
    ids=[c['case_id'] for c in cases]
    assert all(pd.Timestamp(c['date'])+pd.Timedelta(days=7)<=pd.Timestamp('2021-01-01',tz='UTC') for c in cases)
    path=ROOT/'sintef_project/data/kernel/Unit_commitment_decisions.csv'
    meta=pd.read_csv(path,usecols=['Run No','starttime']);mask=meta['Run No'].isin(ids).to_numpy()
    assert meta.loc[mask,'Run No'].tolist()==ids
    cols=[f'result_committed_{u}_t{h}' for u in UNIT_ORDER for h in range(168)]
    frame=pd.read_csv(path,skiprows=(np.flatnonzero(~mask)+1).tolist(),usecols=['Run No',*cols],dtype={c:'uint8' for c in cols})
    assert frame['Run No'].tolist()==ids
    values=frame[cols].to_numpy().reshape(len(ids),14,168);assert np.isin(values,[0,1]).all()
    return values


def main() -> None:
    """No-solve preflight or independently reviewed physical diagnostic."""
    parser=argparse.ArgumentParser();parser.add_argument('--freeze-protocol',action='store_true')
    parser.add_argument('--worker',type=Path);parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--protocol-sha256');parser.add_argument('--independent-audit',type=Path)
    parser.add_argument('--independent-audit-sha256');parser.add_argument('--run-id')
    args=parser.parse_args()
    if args.freeze_protocol:freeze();return
    if args.worker:worker(args.worker);return
    assert args.run_id and Path(args.run_id).name==args.run_id and args.run_id not in ['.','..']
    out=ROOT/'artifacts'/args.run_id;out.mkdir(exist_ok=False);start=time.monotonic()
    report={'status':'initializing','timestamp_utc':datetime.now(timezone.utc).isoformat(),
        'command':shlex.join([sys.executable,*sys.argv]),'protocol_sha256':args.protocol_sha256,
        'labels_decoded':False,'training':False,'automatic_promotion':False,'cases':[],'environment':environment()}
    try:
        assert digest(PROTOCOL)==args.protocol_sha256
        p=json.loads(PROTOCOL.read_text());assert p['environment']==environment()
        for section in ['source','data','inputs']:
            for name,h in p[section].items():assert digest(ROOT/name)==h,name
        assert digest(ROOT/p['cache']/'manifest.json')==p['cache_manifest_sha256']
        if not args.preflight:
            assert digest(args.independent_audit)==args.independent_audit_sha256
            audit=json.loads(args.independent_audit.read_text())
            assert audit['status']=='passed' and audit['protocol_sha256']==args.protocol_sha256
            report['independent_audit_sha256']=args.independent_audit_sha256
        shutil.copy2(PROTOCOL,out/'protocol.json')
        for name in p['source']:
            target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
        raw_read=pd.read_csv;label_path=(ROOT/'sintef_project/data/kernel/Unit_commitment_decisions.csv').resolve()
        def guarded(path,*positional,**kwargs):
            if isinstance(path,(str,Path)) and Path(path).resolve()==label_path:
                assert not positional and kwargs=={'usecols':['Run No','starttime']},'Target decoding forbidden before physical seal'
            return raw_read(path,*positional,**kwargs)
        reused=json.loads((ROOT/p['reuse']['directory']/'manifest.json').read_text())
        assert digest(ROOT/p['reuse']['directory']/'manifest.json')==p['reuse']['manifest_sha256']
        reuse_audit=json.loads((ROOT/p['reuse']['audit']).read_text())
        assert digest(ROOT/p['reuse']['audit'])==p['reuse']['audit_sha256']
        assert reuse_audit['status']=='passed' and reuse_audit['result_manifest_sha256']==p['reuse']['manifest_sha256']
        prepared=[]
        with patch.object(pd,'read_csv',guarded):
            data=HydroCaseData(ROOT);network=data.network;split_net=split_network(network)
            for selected in p['cases']:
                cid,index=selected['case_id'],selected['metadata_position'];date=pd.Timestamp(selected['date'])
                assert int(data.cases.iloc[index]['Run No'])==cid and data.cases.iloc[index].starttime==date
                assert date+pd.Timedelta(days=7)<=pd.Timestamp('2021-01-01',tz='UTC')
                case=data.case(index);case['natural']=np.asarray(case['natural'],dtype=np.float64)
                for key,field in [('pmin','p_min'),('start_cost','startcost_const'),('stop_cost','stopcost_const')]:
                    case[key]=np.array([data.model['generator'][u][field] for u in UNIT_ORDER],dtype=np.float64)
                cached=arrays(ROOT/p['cache']/f'case_{cid:04d}_inputs.npz');assert identical_arrays(case,cached)
                times=pd.date_range(date,periods=168,freq='h')
                local=data.water.loc[date+pd.Timedelta(days=7),['Vaamarvatn','Totak','Venemo','Hyljelihyl']].to_numpy()
                assert np.isfinite(local).all() and np.all(local==local[0]),'Split terminal local values disagree'
                assert data.model['reservoir']['Hyljelihyl']['max_vol']==.26
                river=data.model['river']['r_Hyljelihyl'];assert 'max_flow_const' not in river
                assert river['flow_cost_curve']['x']==p['breakpoints_m3s'] and river['flow_cost_curve']['y']==p['marginal_costs']
                assert data.model['river']['b_Hyljelihyl']['max_flow_const']==1000
                hy_min=data.minimum.loc[times,'Hyljelihyl'].to_numpy() if 'Hyljelihyl' in data.minimum else np.zeros(168)
                split=split_case(case,float(data.volume.loc[date,'Hyljelihyl']),
                    data.inflow.loc[times,'Hyljelihyl'].to_numpy(dtype=np.float64)*.0036,hy_min)
                directory=out/f'case_{cid:04d}';reuse_receipt=None
                if cid in p['reuse']['case_ids']:
                    reuse_receipt=reuse_case(ROOT/p['reuse']['directory'],directory,cid,reused)
                    assert identical_arrays(case,arrays(directory/'A_inputs.npz'))
                    assert identical_arrays(split,arrays(directory/'split_inputs.npz'))
                    for name,net in [('A',network),('split',split_net)]:
                        assert json.loads((directory/f'{name}_network.json').read_text())==json_network(net)
                else:
                    directory.mkdir()
                    np.savez_compressed(directory/'A_inputs.npz',**case);np.savez_compressed(directory/'split_inputs.npz',**split)
                    for name,net in [('A',network),('split',split_net)]:
                        (directory/f'{name}_network.json').write_text(json.dumps(json_network(net),indent=2))
                record={'case_id':cid,'date':str(date),'local_terminal_values_equal':True,'fresh_input_byte_exact':True,
                        'local_terminal_value':float(local[0]),'matrix_signatures':{}}
                for arm in p['arms']:
                    _,problem,_=make_problem(network if arm=='A' else split_net,case if arm=='A' else split,arm,p['lp_seconds'])
                    record['matrix_signatures'][arm]=signatures(problem)
                if reuse_receipt is not None:
                    previous=next(r for r in reused['cases'] if r['case_id']==cid)
                    assert previous['matrix_signatures']==record['matrix_signatures']
                    record['reuse']=reuse_receipt;record['arms']=previous['arms']
                report['cases'].append(record);prepared.append((cid,directory,case,split))
        if args.preflight:
            report.update(status='passed',optimization_calls=0)
        else:
            baseline_resource=resource_state();report['initial_resources']=baseline_resource
            for cid,directory,case,split in prepared:
                record=next(r for r in report['cases'] if r['case_id']==cid)
                if 'reuse' in record:
                    print(json.dumps({'case_id':cid,'status':'reused_physical_passed'}),flush=True)
                    continue
                record['arms']={} 
                for arm in p['arms']:
                    for replica in [1,2]:
                        current=resource_state()
                        assert current['free_percent']>=30 and current['swapouts']<=baseline_resource['swapouts'],current
                        dest=directory/f'{arm}_{replica}';dest.mkdir()
                        prefix='A' if arm=='A' else 'split'
                        job={'case_id':cid,'arm':arm,'replica':replica,'protocol':str(PROTOCOL),'protocol_sha256':args.protocol_sha256,
                            'input':str(directory/f'{prefix}_inputs.npz'),'network':str(directory/f'{prefix}_network.json'),'resources':current}
                        for name in ['input','network']:job[name+'_sha256']=digest(Path(job[name]))
                        (dest/'job.json').write_text(json.dumps(job,indent=2))
                        with (dest/'worker.log').open('w') as log:
                            subprocess.run([sys.executable,str(Path(__file__)),'--worker',str(dest/'job.json')],
                                stdout=log,stderr=subprocess.STDOUT,check=True,timeout=p['process_timeout_seconds'])
                    proofs=[json.loads((directory/f'{arm}_{i}'/'proof.json').read_text()) for i in [1,2]]
                    assert all(r['status']=='complete' and r['all_slacks_zero'] and 0<r['peak_rss_bytes']<p['peak_rss_limit_bytes'] for r in proofs)
                    for name in ['result.npz','primal_dual.npz']:
                        assert identical_arrays(arrays(directory/f'{arm}_1'/name),arrays(directory/f'{arm}_2'/name))
                    result=arrays(directory/f'{arm}_1'/'result.npz')
                    if arm=='A':assert identical_arrays(result,arrays(ROOT/p['cache']/f'case_{cid:04d}_result.npz'))
                    original,problem=capture_problem(network,case)
                    if arm!='A':
                        model,_,_=make_problem(split_net,split,arm)
                        projected=project_to_original(original,model,arrays(directory/f'{arm}_1'/'primal_dual.npz')['x'])
                        projection=feasibility(problem,projected);assert max(projection.values())<=1e-5
                    else:projection=None
                    record['arms'][arm]={'objective':float(result['objective']),'replays_byte_exact':True,
                        'projection':projection,'worker_proof_sha256':[digest(directory/f'{arm}_{i}'/'proof.json') for i in [1,2]]}
                assert record['arms']['A']['objective']+1e-4>=record['arms']['B']['objective']
                assert record['arms']['B']['objective']+1e-4>=record['arms']['C']['objective']
                print(json.dumps({'case_id':cid,'status':'physical_passed','objectives':{a:record['arms'][a]['objective'] for a in p['arms']}}),flush=True)
            seal={'status':'passed','timestamp_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':args.protocol_sha256,
                  'case_count':36,'cases':report['cases']}
            (out/'physical_lock.json').write_text(json.dumps(seal,indent=2));report['physical_lock_sha256']=digest(out/'physical_lock.json')
            truth=read_labels(p['cases']);report['labels_decoded']=True;report['labels_opened_utc']=datetime.now(timezone.utc).isoformat()
            report['metrics']={};diagnostic={'truth':truth}
            for arm in p['arms']:
                results=[arrays(directory/f'{arm}_1'/'result.npz') for _,directory,_,_ in prepared]
                q=np.stack([r['flow']/case['qmin'][:,None] for r,(_,_,case,_) in zip(results,prepared)])
                power=np.stack([r['power']/case['pmin'][:,None] for r,(_,_,case,_) in zip(results,prepared)])
                report['metrics'][arm]=metrics(truth,q,power,p['blocks']);affected=[7,8,9,12,13]
                report['metrics'][arm]['affected_five_q_auc']=float(roc_auc_score(truth[:,affected].ravel(),(q[:,affected]/(1+q[:,affected])).ravel()))
                report['metrics'][arm]['affected_five_cutoff_errors']=int(np.sum((power[:,affected]>=.5)!=truth[:,affected]))
                diagnostic[arm+'_qratio']=q;diagnostic[arm+'_pratio']=power
            np.savez_compressed(out/'diagnostic.npz',**diagnostic);report['status']='complete'
    except Exception:
        report.update(status='failed',error=traceback.format_exc());raise
    finally:
        report['runtime_seconds']=time.monotonic()-start
        report['artifacts']={str(f.relative_to(out)):digest(f) for f in out.rglob('*') if f.is_file() and f.name!='manifest.json'}
        (out/'manifest.json').write_text(json.dumps(report,indent=2))
        (out/'experiment_record.md').write_text('\n## '+args.run_id+' — '+report['timestamp_utc']+'\n\nStatus '+report['status']+
            '; three-arm36-case continuous river diagnostic with exact12 reuse. Full source/input/protocol/environment/command/result evidence at `'+
            str((out/'manifest.json').relative_to(ROOT))+'`, SHA `'+digest(out/'manifest.json')+'`. Labels decoded: '+str(report['labels_decoded'])+
            '; runtime '+str(report['runtime_seconds'])+' seconds. No automatic promotion or new fitted model.\n')
    print(json.dumps({'status':report['status'],'manifest_sha256':digest(out/'manifest.json')}),flush=True)


if __name__=='__main__':main()
