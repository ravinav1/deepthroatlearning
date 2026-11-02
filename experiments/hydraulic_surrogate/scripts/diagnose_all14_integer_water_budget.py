"""Frozen twelve-case all-unit integrality diagnostic with exact two-case reuse."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
import json,multiprocessing,os
from pathlib import Path
import shlex,shutil,sys,time,traceback
from unittest.mock import patch
for variable in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[variable]='1'
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.diagnose_integer_water_budget import arrays,digest
from scripts.pilot_integer_water_budget import environment
from scripts.probe_all14_integer_water_budget_v2 import worker,identical_arrays,all_slacks_exact_zero
from scripts.diagnose_lp_shop_relaxation import metrics
from scripts.continue_integer_water_budget_cache_v2 import resource_state
from sintef_project.lp_features import HydroCaseData,UNIT_ORDER
NEW_SOURCE=['scripts/diagnose_all14_integer_water_budget.py','tests/test_all14_integer_diagnostic.py','docs/all14_integer_water_budget_diagnostic.md']


def legal_dates(dates) -> bool:
    """Require every selected seven-day target interval to end before2021."""
    values=pd.to_datetime(dates,utc=True)
    return len(values)>0 and bool(np.all(values+pd.Timedelta(days=7)<=pd.Timestamp('2021-01-01',tz='UTC')))


def physical_gates(records: list[dict], ids: list[int]) -> dict:
    """Require the full fixed population; never score an adaptively completed subset."""
    full=[r['case_id']for r in records]==ids
    return {'all_fixed_cases':full,'all_proven':full and all(r['status']=='complete'and r['proof'].get('mip_status')==0 and r['proof'].get('proven_optimal')for r in records),
      'all_lp_replays':full and all(r.get('fixed_lp_all_arrays_bitwise',False)for r in records),
      'all_zero_slacks':full and all(r.get('all_slacks_zero',False)for r in records),
      'all_subset_objectives':full and all(r.get('objective_subset_checks',False)for r in records),
      'all_rss_below_2gib':full and all(0<r['peak_rss_bytes']<2*1024**3 for r in records)}


def freeze(path: Path) -> None:
    """Pin original metadata selection, identical physical controls and audited reuse."""
    original_path=ROOT/'docs/integer_water_budget_protocol.json';original=json.loads(original_path.read_text())
    two_path=ROOT/'docs/all14_integer_water_budget_probe_protocol_v2.json';two=json.loads(two_path.read_text())
    p=json.loads(json.dumps(two));cm=json.loads((ROOT/p['cache']/'manifest.json').read_text())
    p.update(version='all14_integer_water_budget_twelve_case_diagnostic_v1',created_utc=datetime.now(timezone.utc).isoformat(),
       cases=[{'case_id':i,'metadata_position':cm['case_ids'].index(i),'date':cm['case_dates'][cm['case_ids'].index(i)]}for i in original['case_ids']],
       blocks=original['blocks'],source={**p['source'],**{n:digest(ROOT/n)for n in NEW_SOURCE}},
       selection='Exact original twelve metadata IDs and blocks; no case reselection from label errors.',
       scope='Descriptive twelve-case comparison only. Reuse two audited identical all14 results and solve the other ten once at60s; one fresh worker; no retries/fallback or learned promotion.',
       label_rule='Only after every physical gate passes, decode selected targets whose seven-day intervals end by2021-01-01. No scored subset or2021+ targets.',
       resource_minimum_free_percent=30,automatic_promotion=False,
       gate='All twelve fixed cases must pass proven optimality, exact fixed-LP replay, zero absolute slack, subset objectives and measured peakRSS below2GiB before any target decoding. Scores remain descriptive with no acceptance threshold.',
       reuse={'directory':'artifacts/all14-integer-water-budget-probe-001','manifest_sha256':'b29c685f193df82074ac389c5c5f1ebac6502860fb01443f3f3bcf6b6fb990c4',
              'audit':'artifacts/all14-integer-water-budget-result-independent-audit-001/manifest.json','audit_sha256':'97c1ff331fb7ad5fe87c7fd2ebd8de7e00d06c384bbb4fa5d0ee69e04a666e69','case_ids':[1,1826]})
    p['source']['scripts/continue_integer_water_budget_cache_v2.py']=digest(ROOT/'scripts/continue_integer_water_budget_cache_v2.py')
    p['source']['scripts/build_integer_water_budget_cache.py']=digest(ROOT/'scripts/build_integer_water_budget_cache.py')
    p['source']['sintef_project/integer_water_budget_features.py']=digest(ROOT/'sintef_project/integer_water_budget_features.py')
    p['inputs'][str(original_path.relative_to(ROOT))]=digest(original_path);p['inputs'][str(two_path.relative_to(ROOT))]=digest(two_path)
    for case in p['cases']:
        i=case['case_id']
        for base,suffix in [(p['cache'],'inputs.npz'),(p['cache'],'result.npz'),(p['comparison'],'integer.npz'),(p['comparison'],'proof.json')]:
            file=ROOT/base/f'case_{i:04d}_{suffix}';p['inputs'][str(file.relative_to(ROOT))]=digest(file)
    for name in [p['reuse']['directory']+'/manifest.json',p['reuse']['audit']]:p['inputs'][name]=digest(ROOT/name)
    assert legal_dates([c['date']for c in p['cases']])and len(p['cases'])==12
    p['metric_specification']={'existing_proxy_metrics':'Original metrics helper with q/qmin and p/pmin; fixed p/pmin>=0.5 on cutoff',
       'additional':'Full14 and weak3 on-hour counts and mismatch counts; per-case/per-unit oncounts; state>=0.5 counts/mismatch separately',
       'comparisons':['continuous','three_binary','all14_binary'],'interpretation':'Descriptive only on the original reused metadata cases; no score gates or automatic model promotion.'}
    with path.open('x')as f:json.dump(p,f,indent=2)
    print(digest(path),flush=True)


def read_labels(ids: list[int]) -> np.ndarray:
    """Decode only the unchanged twelve pre2021 case schedules."""
    path=ROOT/'sintef_project/data/kernel/Unit_commitment_decisions.csv';meta=pd.read_csv(path,usecols=['Run No','starttime'])
    mask=meta['Run No'].isin(ids).to_numpy();assert meta.loc[mask,'Run No'].tolist()==ids and legal_dates(meta.loc[mask,'starttime'])
    cols=[f'result_committed_{u}_t{h}'for u in UNIT_ORDER for h in range(168)]
    frame=pd.read_csv(path,skiprows=(np.flatnonzero(~mask)+1).tolist(),usecols=['Run No',*cols],dtype={c:'uint8'for c in cols})
    assert frame['Run No'].tolist()==ids;labels=frame[cols].to_numpy().reshape(len(ids),14,168);assert np.isin(labels,[0,1]).all();return labels


def main() -> None:
    """Seal all physical results before opening the fixed descriptive labels."""
    parser=argparse.ArgumentParser();parser.add_argument('--freeze-protocol',action='store_true');parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--protocol',type=Path,default=ROOT/'docs/all14_integer_water_budget_diagnostic_protocol.json')
    parser.add_argument('--protocol-sha256');parser.add_argument('--independent-audit',type=Path);parser.add_argument('--independent-audit-sha256');parser.add_argument('--run-id')
    args=parser.parse_args()
    if args.freeze_protocol:freeze(args.protocol);return
    if args.run_id is None or Path(args.run_id).name!=args.run_id or args.run_id in ['', '.', '..']:raise ValueError('Invalid run ID')
    out=ROOT/'artifacts'/args.run_id;out.mkdir(exist_ok=False);started=time.monotonic()
    report={'status':'initializing','timestamp_utc':datetime.now(timezone.utc).isoformat(),'command':shlex.join([sys.executable,*sys.argv]),
       'protocol_sha256':args.protocol_sha256,'labels_decoded':False,'training':False,'cases':[],'environment':environment(),'automatic_promotion':False}
    try:
        assert digest(args.protocol)==args.protocol_sha256;p=json.loads(args.protocol.read_text());assert p['environment']==environment()
        for section in ['source','data','inputs']:
            for name,h in p[section].items():assert digest(ROOT/name)==h,name
        ids=[c['case_id']for c in p['cases']];assert len(ids)==12 and legal_dates([c['date']for c in p['cases']])
        if not args.preflight:
            assert digest(args.independent_audit)==args.independent_audit_sha256
            a=json.loads(args.independent_audit.read_text());assert a['status']=='passed'and a['protocol_sha256']==args.protocol_sha256
            report['independent_audit_sha256']=args.independent_audit_sha256
        shutil.copy2(args.protocol,out/'protocol.json')
        for name in p['source']:
            target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
        reused=ROOT/p['reuse']['directory'];assert digest(reused/'manifest.json')==p['reuse']['manifest_sha256']
        reuse_a=json.loads((ROOT/p['reuse']['audit']).read_text());assert reuse_a['status']=='passed'and digest(ROOT/p['reuse']['audit'])==p['reuse']['audit_sha256']
        rm=json.loads((reused/'manifest.json').read_text());assert rm['feasibility_passed'];lookup={r['case_id']:r for r in rm['cases']}
        cache=ROOT/p['cache'];assert digest(cache/'manifest.json')==p['cache_manifest_sha256']
        original_read=pd.read_csv;target=(ROOT/'sintef_project/data/kernel/Unit_commitment_decisions.csv').resolve()
        def guarded(path,*a,**kw):
            if isinstance(path,(str,Path))and Path(path).resolve()==target:assert not a and kw=={'usecols':['Run No','starttime']}
            return original_read(path,*a,**kw)
        jobs={};records={}
        with patch.object(pd,'read_csv',guarded):
            data=HydroCaseData(ROOT)
            for selected in p['cases']:
                i=selected['case_id'];index=selected['metadata_position'];assert int(data.cases.iloc[index]['Run No'])==i
                fresh=data.case(index);fresh['natural']=fresh['natural'].astype(np.float64)
                for name,field in [('pmin','p_min'),('start_cost','startcost_const'),('stop_cost','stopcost_const')]:fresh[name]=np.asarray([data.model['generator'][u][field]for u in UNIT_ORDER],dtype=np.float64)
                ip=cache/f'case_{i:04d}_inputs.npz';case=arrays(ip);assert identical_arrays(case,fresh);shutil.copy2(ip,out/ip.name)
                previous=json.loads((ROOT/p['comparison']/f'case_{i:04d}_proof.json').read_text());assert previous['proof']['proven_optimal']and previous['input_sha256']==digest(ip)
                if i in p['reuse']['case_ids']:
                    assert digest(reused/ip.name)==digest(ip);shutil.copytree(reused/f'case_{i:04d}',out/f'case_{i:04d}')
                    record=lookup[i];assert record['environment']==p['environment']and record['proof']['mip_time_limit_seconds']==60.
                    for name,h in record['artifacts'].items():assert digest(out/f'case_{i:04d}'/name)==h
                    assert digest(out/f'case_{i:04d}'/'proof.json')==digest(reused/f'case_{i:04d}'/'proof.json');records[i]=record
                else:jobs[i]=(str(out),i,data.network,case,arrays(cache/f'case_{i:04d}_result.npz'),arrays(ROOT/p['comparison']/f'case_{i:04d}_integer.npz'),previous)
        report.update(fresh_input_reconstruction_byte_exact=True,reused_case_ids=p['reuse']['case_ids'],new_case_ids=list(jobs))
        if args.preflight:report.update(status='passed_preflight_only',solves=0);return
        baseline=resource_state();report['startup_resources']=baseline
        with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn'),max_tasks_per_child=1)as pool:
            for i in ids:
                if i in records:continue
                current=resource_state()
                if current['free_percent']<p['resource_minimum_free_percent']or current['swapouts']>baseline['swapouts']:
                    report.update(status='resource_stopped',resource_stop=current,cases=[records[k]for k in ids if k in records]);return
                records[i]=pool.submit(worker,*jobs[i]).result()
                print(json.dumps({'case_id':i,'status':records[i]['status'],'mip_status':records[i]['proof'].get('mip_status'),'seconds':records[i]['runtime_seconds']}),flush=True)
        report['cases']=[records[i]for i in ids];gates=physical_gates(report['cases'],ids)
        lock={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':args.protocol_sha256,'physical_gates':gates,
          'proof_sha256':{str(i):digest(out/f'case_{i:04d}'/'proof.json')for i in ids}}
        (out/'physical_lock.json').write_text(json.dumps(lock,indent=2)+'\n');report['physical_lock_sha256']=digest(out/'physical_lock.json')
        report.update(status='complete',physical_gates=gates,physical_feasibility_passed=all(gates.values()))
        if not all(gates.values()):return
        report['labels_opened_utc']=datetime.now(timezone.utc).isoformat();labels=read_labels(ids);report['labels_decoded']=True
        weak=[UNIT_ORDER.index(u)for u in ['Hogga_G1','Lio_G1','Haukeli_G1']];scores={};saved={'labels':labels,'case_ids':np.asarray(ids),'units':np.asarray(UNIT_ORDER)}
        for kind in ['continuous','three_binary','all14_binary']:
            physical=[arrays(cache/f'case_{i:04d}_result.npz')if kind=='continuous'else arrays(ROOT/p['comparison']/f'case_{i:04d}_integer.npz')if kind=='three_binary'else arrays(out/f'case_{i:04d}'/'result.npz')for i in ids]
            q=np.asarray([r['flow']/arrays(out/f'case_{i:04d}_inputs.npz')['qmin'][:,None]for i,r in zip(ids,physical)])
            power=np.asarray([r['power']/arrays(out/f'case_{i:04d}_inputs.npz')['pmin'][:,None]for i,r in zip(ids,physical)])
            state=np.asarray([r['state']for r in physical]);hard=power>=.5
            score=metrics(labels,q,power,p['blocks']);score.update(weak3_power_cutoff_hour_errors=int(np.sum(hard[:,weak]!=labels[:,weak])),
              power_cutoff_on_hours=int(hard.sum()),weak3_power_cutoff_on_hours=int(hard[:,weak].sum()),
              case_unit_power_cutoff_on_hours=hard.sum(axis=-1).tolist(),state_cutoff_on_hours=int((state>=.5).sum()),
              state_cutoff_hour_errors=int(np.sum((state>=.5)!=labels)))
            scores[kind]=score;saved.update({kind+'_qratio':q,kind+'_pratio':power,kind+'_state':state})
        np.savez_compressed(out/'diagnostic.npz',**saved);report.update(metrics=scores,target_on_hours=int(labels.sum()),target_weak3_on_hours=int(labels[:,weak].sum()),target_case_unit_on_hours=labels.sum(axis=-1).tolist())
        print(json.dumps({'status':'complete','descriptive':{k:{n:v[n]for n in ['micro_auc_qratio','weak3_micro_auc_qratio','power_cutoff_hour_errors','weak3_power_cutoff_hour_errors']}for k,v in scores.items()}}),flush=True)
    except BaseException:report.update(status='failed',error=traceback.format_exc());raise
    finally:
        report['runtime_seconds']=time.monotonic()-started;report['artifacts']={x.relative_to(out).as_posix():digest(x)for x in out.rglob('*')if x.is_file()and x.name!='manifest.json'}
        (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
        (out/'experiment_record.md').write_text(f"\n## {args.run_id} — {report['timestamp_utc']}\n\nStatus {report['status']}; fixed twelve-case all14 diagnostic, two exact reused results and ten single60s attempts. No retry/fallback; labels opened only after all physical gates. Descriptive source/input/solver/proof/metrics/resource evidence: `artifacts/{args.run_id}/manifest.json`. Command `{report['command']}`. Runtime {report['runtime_seconds']:.3f}s. No automatic model or public promotion.\n")
if __name__=='__main__':main()
