"""Native-session interruption proof with preserved complete cases, plus a3GiB probe."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,wait
from datetime import datetime,timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def digest(path: Path) -> str:
    """Hash frozen source and proof files."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def partial_worker(directory: str, index: int) -> int:
    """Publish explicitly incomplete evidence, then wait for synthetic interruption."""
    folder=Path(directory)/'cases'/f'partial_case_{index}';folder.mkdir()
    (folder/'partial_attempt.json').write_text(json.dumps({'status':'incomplete','pid':os.getpid(),'index':index}))
    deadline=time.monotonic()+60
    while time.monotonic()<deadline:time.sleep(.1)
    raise TimeoutError('Harness was not interrupted within60seconds')


def coordinator(directory: Path, protocol_sha: str) -> None:
    """Mirror original failure-finalization while preserving one proven case."""
    from scripts.continue_integer_water_budget_cache_v2 import BASE_PROTOCOL_SHA,import_complete_case
    from scripts.build_integer_water_budget_cache import verify_complete_case
    import shutil
    directory.mkdir(exist_ok=False)
    source=next((ROOT/'artifacts/integer-water-budget-full-cache-preflight-001/cases').glob('case_*'))
    complete=verify_complete_case(source,BASE_PROTOCOL_SHA)
    copied=directory/'cases'/source.name;copied.parent.mkdir();shutil.copytree(source,copied)
    original_bytes=(source/'complete.json').read_bytes();case_sha=digest(source/'complete.json')
    initial={'status':'building_input_only','protocol_sha256':BASE_PROTOCOL_SHA,'completed_cases':{str(complete['case_id']):case_sha}}
    (directory/'initial_manifest.json').write_text(json.dumps(initial,indent=2)+'\n')
    workers=[];interrupted=False;started=time.monotonic()
    try:
        with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn'),max_tasks_per_child=1)as pool:
            futures=[pool.submit(partial_worker,str(directory),i)for i in range(2)]
            workers=list(pool._processes.values())
            print('HARNESS_READY',os.getpid(),flush=True)
            wait(futures)
    except KeyboardInterrupt:
        interrupted=True
    finally:
        finalized=dict(initial,status='failed',error='Expected synthetic native-session KeyboardInterrupt')
        (directory/'finalized_manifest.json').write_text(json.dumps(finalized,indent=2)+'\n')
        exited=all(w.exitcode is not None and not w.is_alive()for w in workers)and not multiprocessing.active_children()
        payload_unchanged=(copied/'complete.json').read_bytes()==original_bytes
        for name,h in complete['artifacts'].items():payload_unchanged=payload_unchanged and digest(copied/name)==h
        imported=directory/'resume_simulation';imported.mkdir()
        for case_id,h in finalized['completed_cases'].items():
            src=directory/'cases'/f'case_{int(case_id):04d}'
            if digest(src/'complete.json')!=h:raise ValueError('Finalized completed proof changed')
            import_complete_case(src,imported/src.name,BASE_PROTOCOL_SHA,protocol_sha,digest(directory/'finalized_manifest.json'))
        partials=[p for p in (directory/'cases').glob('partial_case_*')if p.is_dir()]
        no_partial_import=(len(list(imported.iterdir()))==1 and all(not (p/'complete.json').exists()for p in partials))
        report={'status':'passed'if interrupted and exited and payload_unchanged and no_partial_import and len(partials)==2 else 'failed',
          'timestamp_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':protocol_sha,'source_sha256':digest(Path(__file__)),
          'interrupted':interrupted,'coordinator_finalized':True,'all_children_exited':exited,
          'worker_exit_codes':[w.exitcode for w in workers],'complete_case_payloads_unchanged':payload_unchanged,
          'complete_case_count':1,'partial_attempt_count':len(partials),'partial_imported':not no_partial_import,
          'old_manifest_sha256':digest(directory/'finalized_manifest.json'),'original_complete_sha256':case_sha,
          'operation':'Exact supported execution-session write_stdin Ctrl-C; no privileged signaling',
          'runtime_seconds':time.monotonic()-started,'live_cache_interrupted':False}
        report['artifacts']={x.relative_to(directory).as_posix():digest(x)for x in directory.rglob('*')if x.is_file()and x.name!='manifest.json'}
        (directory/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'status':report['status'],'manifest_sha256':digest(directory/'manifest.json'),'children_exited':exited}),flush=True)
    if report['status']!='passed':raise ValueError('Controlled-interruption preservation proof failed')


def allocator(directory: Path) -> None:
    """Touch3GiB and hold only until the proof process records system headroom."""
    size=3*1024**3;buffer=bytearray(size)
    for offset in range(0,size,4096):buffer[offset]=17
    (directory/'allocation_ready.json').write_text(json.dumps({'allocated_bytes':size,'peak_rss_bytes':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}))
    deadline=time.monotonic()+40
    while not (directory/'release_allocation').exists():
        if time.monotonic()>deadline:raise TimeoutError('Allocation release absent')
        time.sleep(.02)
    del buffer


def main() -> None:
    """Validate externally hashed native interruption evidence, then test memory."""
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['proof','coordinator','allocator'],default='proof')
    parser.add_argument('--directory',type=Path);parser.add_argument('--run-id');parser.add_argument('--protocol-sha256')
    parser.add_argument('--harness',type=Path);parser.add_argument('--harness-manifest-sha256')
    args=parser.parse_args()
    if args.mode=='allocator':allocator(args.directory);return
    pp=ROOT/'docs/integer_water_budget_four_worker_protocol_v2.json'
    if digest(pp)!=args.protocol_sha256:raise ValueError('Frozen continuation protocol differs')
    p=json.loads(pp.read_text())
    for name,h in p['source'].items():
        if digest(ROOT/name)!=h:raise ValueError('Frozen source changed: '+name)
    if args.mode=='coordinator':coordinator(args.directory,args.protocol_sha256);return
    if args.run_id is None or Path(args.run_id).name!=args.run_id or args.run_id in ['', '.', '..']:raise ValueError('Invalid immutable run ID')
    out=ROOT/'artifacts'/args.run_id;out.mkdir(exist_ok=False);started=time.monotonic();child=None
    report={'status':'initializing','timestamp_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':args.protocol_sha256,
            'live_cache_interrupted':False,'environment':{'python':sys.version,'platform':platform.platform()}}
    try:
        hp=args.harness/'manifest.json'
        if digest(hp)!=args.harness_manifest_sha256:raise ValueError('External native-session harness hash differs')
        h=json.loads(hp.read_text())
        if (h['status']!='passed'or h['protocol_sha256']!=args.protocol_sha256 or h['source_sha256']!=digest(Path(__file__))
            or not h['coordinator_finalized']or not h['all_children_exited']or not h['complete_case_payloads_unchanged']
            or h['partial_imported']or h['complete_case_count']!=1 or h['partial_attempt_count']!=2):
            raise ValueError('Native-session interruption proof did not preserve required evidence')
        for name,sha in h['artifacts'].items():
            if digest(args.harness/name)!=sha:raise ValueError('Harness artifact changed')
        report.update(interruption_harness_passed=True,harness_manifest_sha256=args.harness_manifest_sha256)
        from scripts.continue_integer_water_budget_cache_v2 import resource_state
        report['before']=resource_state();allocation=out/'headroom';allocation.mkdir()
        with (allocation/'stderr.log').open('w')as errors:
            child=subprocess.Popen([sys.executable,__file__,'--mode','allocator','--directory',str(allocation)],stdout=subprocess.DEVNULL,stderr=errors)
            deadline=time.monotonic()+50
            while not (allocation/'allocation_ready.json').exists():
                if child.poll()is not None or time.monotonic()>deadline:raise RuntimeError('Allocation readiness failed')
                time.sleep(.02)
            report['during']=resource_state();report['allocation']=json.loads((allocation/'allocation_ready.json').read_text())
            (allocation/'release_allocation').write_text('release after measurements\n');code=child.wait(timeout=50);child=None
            if code!=0:raise ValueError('Allocation process failed')
        report['after']=resource_state()
        report['headroom_probe_passed']=(report['allocation']['allocated_bytes']==p['execution']['headroom_probe_bytes']
           and report['allocation']['peak_rss_bytes']>=p['execution']['headroom_probe_bytes']
           and min(report[x]['free_percent']for x in ['before','during','after'])>=p['execution']['headroom_minimum_free_percent']
           and len({report[x]['swapouts']for x in ['before','during','after']})==1)
        if not report['headroom_probe_passed']:raise ValueError('3GiB headroom gate failed')
        report['status']='passed'
    except BaseException:
        report.update(status='failed',error=traceback.format_exc());raise
    finally:
        if child is not None and child.poll()is None:
            (out/'headroom/release_allocation').write_text('cleanup release\n');child.wait(timeout=50)
        report['runtime_seconds']=time.monotonic()-started;report['source_sha256']=digest(Path(__file__))
        report['artifacts']={x.relative_to(out).as_posix():digest(x)for x in out.rglob('*')if x.is_file()and x.name!='manifest.json'}
        (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'status':report['status'],'manifest_sha256':digest(out/'manifest.json'),'seconds':report['runtime_seconds']}),flush=True)
if __name__=='__main__':main()
