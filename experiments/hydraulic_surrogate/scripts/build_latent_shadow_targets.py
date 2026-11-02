"""Generate one fixed chronological partition of latent-shadow targets."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import shlex
import shutil
import sys
import time
import traceback

for variable in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[variable]='1'

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sintef_project.latent_shadow import FEATURES, GRID, TARGET_UNITS, ShadowInputs, partition_indices, read_selected_labels, soft_targets, uncertainty_summary
from sintef_project.nonlinear_pq import digest


def verify_protocol(expected_sha: str) -> tuple[dict,str]:
    """Fail closed on all frozen source and raw-input pins."""
    path=ROOT/'docs/latent_shadow_learnability_protocol.json'
    if digest(path)!=expected_sha:
        raise ValueError('Explicit learnability protocol SHA mismatch')
    protocol=json.loads(path.read_text())
    for group in ['source','data']:
        for relative,sha in protocol[group].items():
            if digest(ROOT/relative)!=sha:
                raise ValueError(f'Changed {group}: {relative}')
    if protocol['grid']!=GRID.tolist() or protocol['features']!=FEATURES or protocol['units']!=TARGET_UNITS:
        raise ValueError('Frozen family/schema changed')
    reach=ROOT/protocol['reachability_manifest']
    if digest(reach)!=protocol['reachability_manifest_sha256'] or not json.loads(reach.read_text())['gate_passed']:
        raise ValueError('Reachability gate missing or changed')
    return protocol,expected_sha


def main() -> None:
    """Build input-only grids, label-selected target vectors and immutable case files."""
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--partition',choices=['train','screen'],required=True)
    parser.add_argument('--protocol-sha256',required=True)
    parser.add_argument('--training-lock',type=Path)
    parser.add_argument('--training-lock-sha256')
    parser.add_argument('--reuse-targets',type=Path)
    args=parser.parse_args()
    out=ROOT/'artifacts'/args.run_id
    out.mkdir(exist_ok=False)
    started=time.monotonic()
    report={'run_id':args.run_id,'timestamp_utc':datetime.now(timezone.utc).isoformat(),
            'command':shlex.join([sys.executable,*sys.argv]),'partition':args.partition,'status':'initializing',
            'training':'none; deterministic target generation only','case_files':{},'reused_cases':[],
            'environment':{'python':sys.version,'platform':platform.platform(),'machine':platform.machine(),
                           'packages':{n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','PyYAML']}},'kaggle_submission':None}
    try:
        protocol,sha=verify_protocol(args.protocol_sha256)
        report.update(protocol_sha256=sha,source=protocol['source'],data=protocol['data'])
        if args.partition=='screen':
            if not args.training_lock or not args.training_lock_sha256 or digest(args.training_lock)!=args.training_lock_sha256:
                raise ValueError('Screen generation requires explicit completed training-lock SHA')
            lock=json.loads(args.training_lock.read_text())
            fit=ROOT/lock['fit_manifest']
            if lock['status']!='locked' or lock['protocol_sha256']!=sha or digest(fit)!=lock['fit_manifest_sha256']:
                raise ValueError('Training lock is incomplete or changed')
            fm=json.loads(fit.read_text())
            if fm['status']!='complete' or fm['mode']!='fit' or fm['labels_max_year']!=2018:
                raise ValueError('Training fit violates the chronological gate')
            for relative,expected in lock['model_artifacts'].items():
                if digest(ROOT/relative)!=expected:
                    raise ValueError('A locked model changed before screen targets')
            report.update(training_lock=str(args.training_lock),training_lock_sha256=args.training_lock_sha256)
        cache=ROOT/'artifacts/nonlinear-pq-features-001'
        if digest(cache/'manifest.json')!=protocol['cache_manifest_sha256']:
            raise ValueError('Nonlinear cache manifest changed')
        cm=json.loads((cache/'manifest.json').read_text())
        if digest(ROOT/'artifacts/lp-full-001/manifest.json')!=cm['lp_manifest_sha256']:
            raise ValueError('Frozen LP manifest changed, including discharge feature provenance')
        for name in ['features.npz','raw_on_rewards.npz']:
            if digest(cache/name)!=cm['artifacts'][name]:
                raise ValueError('Frozen nonlinear cache changed')
        inputs=ShadowInputs(ROOT,ROOT/'artifacts/lp-full-001')
        indices=partition_indices(inputs.cases,args.partition)
        selected=inputs.cases.iloc[indices]
        report.update(case_ids=selected['Run No'].tolist(),case_dates=selected.starttime.astype(str).tolist(),
                      labels_max_year=int(selected.starttime.dt.year.max()),features=FEATURES,grid=GRID.tolist(),units=TARGET_UNITS)
        expected_max=2018 if args.partition=='train' else 2020
        if report['labels_max_year']!=expected_max or len(set(report['case_ids']))!=len(indices):
            raise ValueError('Partition year or IDs changed')
        for name in protocol['source']:
            target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
        (out/'protocol_before_labels.json').write_text(json.dumps(report,indent=2))
        with np.load(cache/'features.npz',allow_pickle=False) as z:
            saved_features=z['values'][indices]
            if z['case_ids'][indices].tolist()!=report['case_ids']:
                raise ValueError('Nonlinear cache IDs differ')
        with np.load(cache/'raw_on_rewards.npz',allow_pickle=False) as z:
            saved_rewards=z['rewards'][indices]
        # One selected-row parse for the entire partition; no case-by-case CSV reload.
        report['label_decode_started_utc']=datetime.now(timezone.utc).isoformat()
        labels=read_selected_labels(ROOT/'sintef_project/data/kernel/Unit_commitment_decisions.csv',report['case_ids'])
        reuse=None
        if args.reuse_targets:
            reuse=json.loads((args.reuse_targets/'manifest.json').read_text())
            if reuse['status']!='failed' or reuse['partition']!=args.partition or reuse['protocol_sha256']!=sha or reuse['case_ids']!=report['case_ids']:
                raise ValueError('Reuse requires an immutable failed run with exact protocol/partition/IDs')
            report['reuse_manifest_sha256']=digest(args.reuse_targets/'manifest.json')
        for i,index in enumerate(indices):
            case_id=int(selected.iloc[i]['Run No']);name=f'case_{case_id:04d}.npz';path=out/name
            if reuse is not None and name in reuse.get('case_files',{}):
                original=args.reuse_targets/name
                if digest(original)!=reuse['case_files'][name]:
                    raise ValueError('Failed-run retained case hash changed')
                shutil.copy2(original,path)
                with np.load(path,allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['labels'],labels[i])
                    np.testing.assert_array_equal(z['old_features'],saved_features[i])
                    np.testing.assert_array_equal(z['rewards'][:,8,:].T,saved_rewards[i])
                report['reused_cases'].append(case_id)
            else:
                values=inputs.grid_case(int(index))
                np.testing.assert_array_equal(values['old_features'],saved_features[i])
                np.testing.assert_array_equal(values['rewards'][:,8,:].T,saved_rewards[i])
                error=np.sum(values['paths']!=labels[i,:,None,:],axis=-1)
                target=soft_targets(error)
                lo,hi=uncertainty_summary(target)
                temporary=path.with_suffix('.npz.partial')
                with temporary.open('wb') as stream:
                    np.savez_compressed(stream,**values,labels=labels[i],error=error,target=target,
                        acceptable=error<=error.min(axis=-1,keepdims=True)+1,
                        target_lower90=lo,target_upper90=hi,case_id=np.asarray(case_id),grid=GRID,
                        feature_names=np.asarray(FEATURES),units=np.asarray(TARGET_UNITS))
                temporary.rename(path)
            report['case_files'][name]=digest(path)
            if (i+1)%100==0 or i+1==len(indices):
                print('TARGET_CASES',args.partition,i+1,len(indices),'seconds',round(time.monotonic()-started,2),flush=True)
                (out/'progress.json').write_text(json.dumps({k:report[k] for k in ['run_id','partition','protocol_sha256','case_files']},indent=2))
        report.update(status='complete',d0_cache_identity_all_cases=True,labels_decoded_once=True)
    except BaseException:
        report.update(status='failed',error=traceback.format_exc())
        raise
    finally:
        report['runtime_seconds']=time.monotonic()-started
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report['peak_rss_bytes']=int(rss if sys.platform=='darwin' else rss*1024)
        report['artifacts']={str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'}
        (out/'manifest.json').write_text(json.dumps(report,indent=2))
        with (ROOT/'docs/experiments.md').open('a') as stream:
            stream.write(f'\n## {args.run_id} — {report["timestamp_utc"]}\n\nStatus {report["status"]}; latent-shadow {args.partition} target generation only, no fitted model. Command `{report["command"]}`. Fixed17-grid/threeunits/eightfeatures; exact legal inputs, source/cache/sample/IDs and all per-case rewards/paths/soft-target hashes in `artifacts/{args.run_id}/manifest.json`. Max target year {report.get("labels_max_year")}. Runtime {report["runtime_seconds"]:.3f}s, peak RSS {report["peak_rss_bytes"]}bytes. Kaggle and whole14-unit validation absent.\n')


if __name__=='__main__':
    main()
