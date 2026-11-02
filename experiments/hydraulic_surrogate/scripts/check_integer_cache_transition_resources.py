"""Bounded memory headroom probe and original-coordinator interrupt/drain harness."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, wait
from datetime import datetime,timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import resource
import signal
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def digest(path: Path) -> str:
    """Hash small immutable transition evidence."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def harness_worker(directory: str, index: int) -> int:
    """Wait for a parent-controlled release, then publish completion."""
    out=Path(directory);(out/f'worker_{index}_started').write_text(str(os.getpid()))
    deadline=time.monotonic()+30
    while not (out/'release_workers').exists():
        if time.monotonic()>deadline:raise TimeoutError('Harness worker release missing')
        time.sleep(.02)
    (out/f'worker_{index}_done').write_text(str(os.getpid()))
    return index


def coordinator(directory: Path) -> None:
    """Replicate original Pool context semantics when only the coordinator is interrupted."""
    interrupted=False
    try:
        with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn'),max_tasks_per_child=1)as pool:
            futures=[pool.submit(harness_worker,str(directory),i)for i in range(2)]
            wait(futures)
    except KeyboardInterrupt:
        interrupted=True
    finally:
        complete=all((directory/f'worker_{i}_done').exists()for i in range(2))
        (directory/'coordinator_finalized.json').write_text(json.dumps({'interrupted':interrupted,'all_active_workers_finished':complete}))
    if not interrupted or not complete:raise ValueError('Original drain semantics not established')


def allocator(directory: Path) -> None:
    """Touch all3GiB pages, hold briefly for measurement, then release on request."""
    size=3*1024**3; buffer=bytearray(size)
    for offset in range(0,size,4096):buffer[offset]=17
    rss=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (directory/'allocation_ready.json').write_text(json.dumps({'allocated_bytes':size,'page_touch_stride':4096,'peak_rss_bytes':rss}))
    deadline=time.monotonic()+30
    while not (directory/'release_allocation').exists():
        if time.monotonic()>deadline:raise TimeoutError('Allocation release missing')
        time.sleep(.02)
    del buffer


def wait_files(process: subprocess.Popen, paths: list[Path], seconds: float = 40) -> None:
    """Wait only for bounded synthetic child readiness."""
    deadline=time.monotonic()+seconds
    while not all(p.exists()for p in paths):
        if process.poll()is not None:raise RuntimeError('Synthetic child ended before readiness')
        if time.monotonic()>deadline:raise TimeoutError('Synthetic readiness timeout')
        time.sleep(.02)


def main() -> None:
    """Prove resource and coordinator-drain gates without touching the live cache."""
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['proof','coordinator','allocator'],default='proof')
    parser.add_argument('--directory',type=Path);parser.add_argument('--run-id');parser.add_argument('--protocol-sha256')
    args=parser.parse_args()
    if args.mode=='coordinator':coordinator(args.directory);return
    if args.mode=='allocator':allocator(args.directory);return
    if args.run_id is None or Path(args.run_id).name!=args.run_id or args.run_id in ['', '.', '..']:raise ValueError('Invalid immutable run ID')
    out=ROOT/'artifacts'/args.run_id;out.mkdir(exist_ok=False);started=time.monotonic()
    report={'status':'initializing','timestamp_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':args.protocol_sha256,
            'environment':{'python':sys.version,'platform':platform.platform()},'live_cache_signaled':False,'solver_runs':0}
    child=None
    try:
        protocol_path=ROOT/'docs/integer_water_budget_four_worker_protocol.json'
        if digest(protocol_path)!=args.protocol_sha256:raise ValueError('Frozen continuation protocol hash differs')
        p=json.loads(protocol_path.read_text())
        if digest(Path(__file__))!=p['source']['scripts/check_integer_cache_transition_resources.py']:raise ValueError('Probe source changed')
        from scripts.continue_integer_water_budget_cache import resource_state
        harness=out/'drain_harness';harness.mkdir()
        with (harness/'stderr.log').open('w')as errors:
            child=subprocess.Popen([sys.executable,__file__,'--mode','coordinator','--directory',str(harness)],stdout=subprocess.DEVNULL,stderr=errors)
            wait_files(child,[harness/f'worker_{i}_started'for i in range(2)])
            os.kill(child.pid,signal.SIGINT)
            time.sleep(.1)
            if (harness/'coordinator_finalized.json').exists():raise ValueError('Coordinator finalized before active workers drained')
            (harness/'release_workers').write_text('release after coordinator-only SIGINT\n')
            code=child.wait(timeout=40);child=None
            state=json.loads((harness/'coordinator_finalized.json').read_text())
            if code!=0 or not state['interrupted'] or not state['all_active_workers_finished']:raise ValueError('Synthetic coordinator drain failed')
        report['drain_harness_passed']=True
        report['before']=resource_state();allocation=out/'headroom';allocation.mkdir()
        with (allocation/'stderr.log').open('w')as errors:
            child=subprocess.Popen([sys.executable,__file__,'--mode','allocator','--directory',str(allocation)],stdout=subprocess.DEVNULL,stderr=errors)
            wait_files(child,[allocation/'allocation_ready.json'])
            report['during']=resource_state();report['allocation']=json.loads((allocation/'allocation_ready.json').read_text())
            (allocation/'release_allocation').write_text('release after system measurements\n')
            code=child.wait(timeout=40);child=None
            if code!=0:raise ValueError('Allocation process failed')
        report['after']=resource_state()
        report['headroom_probe_passed']=(report['allocation']['allocated_bytes']==p['execution']['headroom_probe_bytes']
           and report['allocation']['peak_rss_bytes']>=p['execution']['headroom_probe_bytes']
           and min(report[x]['free_percent']for x in ['before','during','after'])>=p['execution']['headroom_minimum_free_percent']
           and len({report[x]['swapouts']for x in ['before','during','after']})==1)
        if not report['headroom_probe_passed']:raise ValueError('3GiB resource gate failed')
        report['status']='passed'
    except BaseException:
        report.update(status='failed',error=traceback.format_exc());raise
    finally:
        if child is not None and child.poll()is None:
            for directory,name in [(out/'drain_harness','release_workers'),(out/'headroom','release_allocation')]:
                if directory.exists():(directory/name).write_text('bounded cleanup release\n')
            try:child.wait(timeout=40)
            except subprocess.TimeoutExpired:child.terminate();child.wait(timeout=5)
        report['runtime_seconds']=time.monotonic()-started;report['source_sha256']=digest(Path(__file__))
        report['artifacts']={x.relative_to(out).as_posix():digest(x)for x in out.rglob('*')if x.is_file()and x.name!='manifest.json'}
        (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'status':report['status'],'manifest_sha256':digest(out/'manifest.json'),'seconds':report['runtime_seconds']}),flush=True)
if __name__=='__main__':main()
