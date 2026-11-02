"""Metadata-spaced label-free MILP timing and full saved-replay pilot."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import platform
import resource
import shlex
import shutil
import sys
import time
import traceback
from unittest.mock import patch

for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[variable] = '1'

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.optimize._highspy import _core

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.diagnose_integer_water_budget import arrays, digest, TARGETS
from sintef_project.integer_water_budget import IntegerWaterBudgetLP
from sintef_project.lp_features import HydroCaseData, UNIT_ORDER

NEW_SOURCE = ['scripts/pilot_integer_water_budget.py', 'docs/integer_water_budget_timing_pilot.md',
              'tests/test_integer_water_budget_pilot.py']


def environment() -> dict:
    """Identify the actual bundled solver, not just its Python wrapper."""
    return {'python': sys.version, 'platform': platform.platform(), 'machine': platform.machine(),
            'cpu_count': os.cpu_count(), 'highs_version': '.'.join(str(getattr(_core, 'HIGHS_VERSION_' + k)) for k in ['MAJOR','MINOR','PATCH']),
            'packages': {p: importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','PyYAML']},
            'threads': {k: os.environ[k] for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']}}


def capture_solve(network, case: dict, targets: list[int], seconds: float = 60.) -> tuple[dict, dict, list[dict]]:
    """Record both existing conditional LP calls without changing their inputs."""
    saved = []
    solver = IntegerWaterBudgetLP(network, len(case['price']), case['energy'].shape[1])
    def recording_linprog(*args, **kwargs):
        result = linprog(*args, **kwargs)
        fields = {'status': np.asarray(result.status), 'success': np.asarray(result.success),
                  'objective': np.asarray(result.fun if result.fun is not None else np.nan)}
        if result.x is not None:
            fields['solution'] = np.asarray(result.x).copy()
        for kind in ['eqlin', 'ineqlin', 'lower', 'upper']:
            for key in ['residual', 'marginals']:
                value = getattr(getattr(result, kind, None), key, None)
                if value is not None:
                    fields[kind + '_' + key] = np.asarray(value).copy()
        saved.append(fields)
        return result
    try:
        with patch('sintef_project.integer_water_budget.linprog', recording_linprog):
            result, proof = solver.solve_integer(case, targets, seconds, 1.)
        return result, proof, saved
    except Exception as error:
        error.saved_lp_calls = saved
        error.saved_integer_arrays = solver.last_integer_arrays
        error.saved_integer_proof = solver.last_integer_metadata
        raise


def freeze(path: Path) -> None:
    """Choose metadata positions and pin all mathematical and execution inputs."""
    original = json.loads((ROOT/'docs/integer_water_budget_protocol.json').read_text())
    cache = ROOT/original['cache']
    m = json.loads((cache/'manifest.json').read_text())
    positions = np.rint(np.linspace(0, len(m['case_ids'])-1, 24)).astype(int).tolist()
    sources = dict(original['source'])
    sources.update({s:digest(ROOT/s) for s in NEW_SOURCE})
    audit = 'artifacts/integer-water-budget-completion-independent-audit-001/manifest.json'
    p = {'version':'integer_water_budget_timing_pilot_v1','created_utc':datetime.now(timezone.utc).isoformat(),
         'source':sources,'data':original['data'],'cache':original['cache'],
         'cache_manifest_sha256':digest(cache/'manifest.json'),'independent_audit':audit,
         'independent_audit_sha256':digest(ROOT/audit),'metadata_positions':positions,
         'case_ids':[m['case_ids'][i] for i in positions], 'case_dates':[m['case_dates'][i] for i in positions],
         'milp_seconds':60.,'conditional_lp_seconds_each':1.,'workers':2,'attempts_per_case':2,
         'targets':TARGETS,'environment':environment(), 'labels_decoded':False,
         'scheduling':'Two spawned workers, max_tasks_per_child=1; each attempt gets a fresh process.',
         'replay':'Preserve every raw MILP output and both complete fixed-state LP outputs; compare arrays bitwise. No retries or fallback.'}
    with path.open('x') as stream:
        json.dump(p,stream,indent=2)
    print(digest(path))


def worker(out_name: str, case_id: int, attempt: int, network, case: dict, control: dict,
           input_sha: str, control_sha: str, seconds: float) -> dict:
    """Run a single independent attempt with no data or target file loading."""
    out = Path(out_name)/f'case_{case_id:04d}'/f'attempt_{attempt}'
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    report = {'case_id':case_id,'attempt':attempt,'timestamp_utc':datetime.now(timezone.utc).isoformat(),
              'status':'initializing','pid':os.getpid(),'input_sha256':input_sha,'control_sha256':control_sha,
              'environment':environment()}
    saved = []
    try:
        result, proof, saved = capture_solve(network,case,[UNIT_ORDER.index(u) for u in TARGETS],seconds)
        np.savez_compressed(out/'result.npz',**result)
        assert result['objective'] <= control['objective'] + 1e-4
        assert len(saved) == 2
        for key in saved[0]:
            np.testing.assert_array_equal(saved[0][key],saved[1][key],err_msg=key)
        scale = 1+np.abs(case['price']).mean()
        report.update(status='complete',proof=proof,objective=float(result['objective']),
                      original_relaxed_objective=float(control['objective']),
                      native_absolute_gap_eur=float((proof['mip_objective_scaled']-proof['mip_dual_bound_scaled'])*scale),
                      all_slacks_zero=all(np.max(result[k]) == 0 for k in ['minimum_slack','maximum_slack','bypass_slack','vest_slack']),
                      fixed_lp_all_arrays_bitwise=True)
    except Exception as error:
        saved = getattr(error,'saved_lp_calls',saved)
        report.update(status='failed',error=traceback.format_exc(),proof=getattr(error,'saved_integer_proof',{}))
        np.savez_compressed(out/'failed_incumbent.npz',**getattr(error,'saved_integer_arrays',{}))
    finally:
        for i, fields in enumerate(saved,1):
            np.savez_compressed(out/f'fixed_lp_{i}.npz',**fields)
        report.update(runtime_seconds=time.monotonic()-start,
                      peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024))
        report['artifacts']={p.name:digest(p) for p in out.iterdir() if p.is_file()}
        (out/'proof.json').write_text(json.dumps(report,indent=2))
    return report


def main() -> None:
    """Run the frozen timing pilot and report failures without any target metrics."""
    parser=argparse.ArgumentParser()
    parser.add_argument('--freeze-protocol',action='store_true')
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--protocol',type=Path,default=ROOT/'docs/integer_water_budget_timing_pilot_protocol.json')
    parser.add_argument('--protocol-sha256')
    parser.add_argument('--run-id')
    args=parser.parse_args()
    if args.freeze_protocol:
        freeze(args.protocol)
        return
    if args.protocol_sha256 is None or args.run_id is None:
        raise ValueError('Explicit protocol hash and unique run ID required')
    assert digest(args.protocol)==args.protocol_sha256
    p=json.loads(args.protocol.read_text())
    assert p['environment']==environment()
    for group in ['source','data']:
        for name,sha in p[group].items():
            assert digest(ROOT/name)==sha,name
    assert digest(ROOT/p['independent_audit'])==p['independent_audit_sha256']
    audit=json.loads((ROOT/p['independent_audit']).read_text())
    assert audit['status']=='passed'
    cache=ROOT/p['cache']
    assert digest(cache/'manifest.json')==p['cache_manifest_sha256']
    cm=json.loads((cache/'manifest.json').read_text())
    out=ROOT/'artifacts'/args.run_id
    out.mkdir(exist_ok=False)
    start=time.monotonic()
    report={'run_id':args.run_id,'timestamp_utc':datetime.now(timezone.utc).isoformat(),'status':'initializing',
            'command':shlex.join([sys.executable,*sys.argv]),'protocol_sha256':args.protocol_sha256,
            'environment':environment(),'labels_decoded':False,'training':False,'kaggle_submission':None,
            'case_ids':p['case_ids'],'attempts':[],'case_replays':[]}
    try:
        shutil.copy2(args.protocol,out/'protocol.json')
        for name in p['source']:
            dest=out/'source'/name
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/name,dest)
        label_path=(ROOT/'sintef_project/data/kernel/Unit_commitment_decisions.csv').resolve()
        old_read=pd.read_csv
        def guarded(path,*positional,**kwargs):
            if isinstance(path,(str,Path)) and Path(path).resolve()==label_path:
                if positional or kwargs!={'usecols':['Run No','starttime']}:
                    raise ValueError('Pilot may read target metadata only')
            return old_read(path,*positional,**kwargs)
        jobs=[]
        with patch.object(pd,'read_csv',guarded):
            data=HydroCaseData(ROOT)
            for index,case_id,date in zip(p['metadata_positions'],p['case_ids'],p['case_dates']):
                assert int(data.cases.iloc[index]['Run No'])==case_id
                assert str(data.cases.iloc[index].starttime)==date
                case=data.case(index)
                case['natural']=np.asarray(case['natural'],dtype=np.float64)
                for name,field in [('pmin','p_min'),('start_cost','startcost_const'),('stop_cost','stopcost_const')]:
                    case[name]=np.asarray([data.model['generator'][u][field] for u in UNIT_ORDER],dtype=np.float64)
                ip,cp=cache/f'case_{case_id:04d}_inputs.npz',cache/f'case_{case_id:04d}_result.npz'
                for path in [ip,cp]:
                    assert digest(path)==cm['artifacts'][path.name]
                old=arrays(ip)
                assert set(old)==set(case)
                for key in case:
                    np.testing.assert_array_equal(case[key],old[key],err_msg=key)
                np.savez_compressed(out/f'case_{case_id:04d}_inputs.npz',**case)
                jobs.append((case_id,case,arrays(cp),digest(ip),digest(cp)))
        if args.preflight:
            report.update(status='passed_preflight_only',input_reconstruction='All 24 cases equal saved inputs bitwise; all frozen hashes and versions passed.')
            return
        with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn'),max_tasks_per_child=1) as pool:
            futures=[]
            for case_id,case,control,ish,csh in jobs:
                for attempt in [1,2]:
                    futures.append(pool.submit(worker,str(out),case_id,attempt,data.network,case,control,ish,csh,60.))
            for future in as_completed(futures):
                value=future.result()
                report['attempts'].append(value)
                print(json.dumps({'completed':len(report['attempts']),'case_id':value['case_id'],'attempt':value['attempt'],
                                  'status':value['status'],'mip_status':value['proof'].get('mip_status'),
                                  'mip_gap':value['proof'].get('mip_gap'),'seconds':value['runtime_seconds']}),flush=True)
        by_case={i:[] for i in p['case_ids']}
        for r in report['attempts']:
            by_case[r['case_id']].append(r)
        for case_id,records in by_case.items():
            records.sort(key=lambda x:x['attempt'])
            row={'case_id':case_id,'both_complete':all(r['status']=='complete' for r in records),
                 'both_proven':all(r['proof'].get('proven_optimal',False) for r in records)}
            if row['both_complete']:
                a,b=[arrays(out/f'case_{case_id:04d}'/f'attempt_{n}'/'result.npz') for n in [1,2]]
                row['different_result_fields']=[k for k in a if not np.array_equal(a[k],b[k])]
                row['all_result_fields_bitwise']=not row['different_result_fields']
                row['integer_states_bitwise']=np.array_equal(a['integer_states'],b['integer_states'])
                row['objective_difference_eur']=float(a['objective']-b['objective'])
            report['case_replays'].append(row)
        times=np.asarray([r['runtime_seconds'] for r in report['attempts']])
        report.update(status='complete',all_attempts_proven=all(r['proof'].get('proven_optimal',False) for r in report['attempts']),
                      all_case_replays_bitwise=all(r.get('all_result_fields_bitwise',False) for r in report['case_replays']),
                      all_slacks_zero=all(r.get('all_slacks_zero',False) for r in report['attempts']),
                      timing={'mean_attempt_seconds':float(times.mean()),'median_attempt_seconds':float(np.median(times)),
                              'p90_attempt_seconds':float(np.quantile(times,.9)),'p95_attempt_seconds':float(np.quantile(times,.95)),
                              'maximum_attempt_seconds':float(times.max()),'sum_attempt_seconds':float(times.sum()),
                              'sum_based_full_double_pass_hours_two_workers':float(times.mean()*3647/3600)},
                      maximum_worker_rss_bytes=max(r['peak_rss_bytes'] for r in report['attempts']))
    except BaseException:
        report.update(status='failed',error=traceback.format_exc())
        raise
    finally:
        report['runtime_seconds']=time.monotonic()-start
        report['artifacts']={f.relative_to(out).as_posix():digest(f) for f in out.rglob('*') if f.is_file() and f.name!='manifest.json'}
        (out/'manifest.json').write_text(json.dumps(report,indent=2))
        (out/'experiment_record.md').write_text(f'\n## {args.run_id} — {report["timestamp_utc"]}\n\nStatus {report["status"]}. Frozen 24-case metadata-spaced label-free timing/replay pilot. Two one-thread workers, two fresh-process MILP attempts per case, 60-second MILP and two 1-second conditional LP caps. Both LP arrays/duals retained. Command `{report["command"]}`. Runtime {report["runtime_seconds"]:.3f} s. Full hashes, environment, statuses, bounds, outputs and replay checks: `artifacts/{args.run_id}/manifest.json`. No labels, model fit, fallback, full cache or submission.\n')


if __name__=='__main__':
    main()
