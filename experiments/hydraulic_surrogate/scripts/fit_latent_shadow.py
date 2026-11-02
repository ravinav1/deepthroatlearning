"""Fit or screen locked small latent-shadow maps on the prescribed early years."""
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
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.build_latent_shadow_targets import verify_protocol
from sintef_project.latent_shadow import FEATURES, GRID, PRIOR, TARGET_UNITS, fit_mapping, predict_distribution, schedule_mixture, soft_targets, uncertainty_summary, validate_partition_metadata
from sintef_project.nonlinear_pq import digest


def load_cache(path: Path, partition: str, protocol_sha: str) -> tuple[dict,dict]:
    """Load only a completed, fully hashed prescribed target partition."""
    manifest=json.loads((path/'manifest.json').read_text())
    if manifest['status']!='complete' or manifest['partition']!=partition or manifest['protocol_sha256']!=protocol_sha:
        raise ValueError('Incomplete or changed chronological target cache')
    metadata=pd.read_csv(ROOT/'sintef_project/data/kernel/Unit_commitment_decisions.csv',usecols=['Run No','starttime'])
    validate_partition_metadata(manifest['case_ids'],manifest['case_dates'],metadata,partition)
    for name,sha in manifest['artifacts'].items():
        if digest(path/name)!=sha:
            raise ValueError(f'Target cache file changed: {name}')
    dates=pd.DatetimeIndex(pd.to_datetime(manifest['case_dates'],utc=True))
    first,last={'train':('2015-01-01','2018-12-25'),'screen':('2019-01-01','2020-12-25')}[partition]
    if dates.min()!=pd.Timestamp(first,tz='UTC') or dates.max()!=pd.Timestamp(last,tz='UTC') or not dates.is_monotonic_increasing:
        raise ValueError('Unexpected target partition date range/order')
    if dates.year.max()!=(2018 if partition=='train' else 2020):
        raise ValueError('Later-year target access forbidden')
    result={k:[] for k in ['features','paths','labels','target','error','acceptable']}
    for case_id in manifest['case_ids']:
        name=f'case_{case_id:04d}.npz'
        if digest(path/name)!=manifest['case_files'][name]:
            raise ValueError('Case hash differs from registry')
        with np.load(path/name,allow_pickle=False) as z:
            if int(z['case_id'])!=case_id or z['feature_names'].tolist()!=FEATURES or z['units'].tolist()!=TARGET_UNITS:
                raise ValueError('Target case schema mismatch')
            np.testing.assert_array_equal(z['grid'],GRID)
            for key in result:
                result[key].append(z[key])
    result={k:np.stack(v) for k,v in result.items()}
    result['dates']=dates
    result['case_ids']=np.asarray(manifest['case_ids'])
    if not np.isfinite(result['features']).all() or not np.isin(result['labels'],[0,1]).all():
        raise ValueError('Invalid features or labels')
    np.testing.assert_array_equal(np.sum(result['paths']!=result['labels'][:,:,None,:],axis=-1),result['error'])
    np.testing.assert_array_equal(result['target'],soft_targets(result['error']))
    np.testing.assert_array_equal(result['acceptable'],result['error']<=result['error'].min(axis=-1,keepdims=True)+1)
    if not np.isin(result['paths'],[0,1]).all():
        raise ValueError('Invalid retained DP path state')
    return result,manifest


def predict_all(x: np.ndarray, family: dict) -> np.ndarray:
    """Call the three input-only maps in the fixed target-unit order."""
    return np.stack([predict_distribution(x[:,j],family[unit]) for j,unit in enumerate(TARGET_UNITS)],axis=1)


def metrics(distribution: np.ndarray, cache: dict, mask: np.ndarray) -> dict:
    """Measure screen targets after inference without feeding labels into a model."""
    pi=distribution[mask]
    truth,paths,target,error,acceptable=[cache[k][mask] for k in ['labels','paths','target','error','acceptable']]
    p=np.stack([schedule_mixture(pi[:,j],paths[:,j]) for j in range(3)],axis=1)
    binary=(p>.5).astype(np.uint8)
    lower,upper=uncertainty_summary(pi)
    expected=pi @ GRID
    distance=np.min(np.where(acceptable,np.abs(expected[:,:,None]-GRID),np.inf),axis=-1)
    coverage=(acceptable&(GRID>=lower[:,:,None])&(GRID<=upper[:,:,None])).any(axis=-1)
    boundary=np.zeros(truth.shape,dtype=bool)
    for i in range(len(truth)):
        for j in range(3):
            for hour in np.flatnonzero(np.diff(truth[i,j].astype(int))!=0)+1:
                boundary[i,j,max(0,hour-3):min(168,hour+4)]=True
    wrong=binary!=truth
    return {'cases':int(mask.sum()),'hour_errors':int(wrong.sum()),
            'per_unit_hour_errors':dict(zip(TARGET_UNITS,wrong.sum(axis=(0,2)).astype(int).tolist())),
            'expected_hour_mismatch':float(np.sum(pi*error)),
            'soft_target_cross_entropy':float(-np.mean(np.sum(target*np.log(np.maximum(pi,1e-300)),axis=-1))),
            'mean_expected_displacement_distance_to_acceptable_set':float(distance.mean()),
            'central90_interval_acceptable_set_coverage':float(coverage.mean()),
            'three_target_pooled_auc':float(roc_auc_score(truth.ravel(),p.ravel())),
            'per_unit_auc':{u:float(roc_auc_score(truth[:,j].ravel(),p[:,j].ravel())) for j,u in enumerate(TARGET_UNITS)},
            'on_fraction_absolute_error':float(np.abs(p.mean(-1)-truth.mean(-1)).mean()),
            'switch_count_absolute_error':float(np.abs(np.sum(np.diff(binary.astype(int),axis=-1)!=0,axis=-1)-np.sum(np.diff(truth.astype(int),axis=-1)!=0,axis=-1)).mean()),
            'errors_near_true_boundaries':int((wrong&boundary).sum()),
            'errors_away_from_true_boundaries':int((wrong&~boundary).sum())}


def main() -> None:
    """Fit early targets, or evaluate an immutable earlier fit on a later screen."""
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['fit','evaluate'],required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--protocol-sha256',required=True)
    parser.add_argument('--train-cache',type=Path,required=True)
    parser.add_argument('--screen-cache',type=Path)
    parser.add_argument('--training-lock',type=Path)
    parser.add_argument('--training-lock-sha256')
    args=parser.parse_args()
    out=ROOT/'artifacts'/args.run_id;out.mkdir(exist_ok=False)
    started=time.monotonic()
    report={'run_id':args.run_id,'timestamp_utc':datetime.now(timezone.utc).isoformat(),
            'command':shlex.join([sys.executable,*sys.argv]),'status':'initializing','mode':args.mode,
            'environment':{'python':sys.version,'platform':platform.platform(),'machine':platform.machine(),
                           'packages':{n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn']}},'seed':4173,'kaggle_submission':None,
            'labels_max_year':2018 if args.mode=='fit' else 2020,'inputs':{},
            'score_scope':'Only three-target DP-family development measures. No whole14-unit production-anchor AUC.'}
    try:
        protocol,sha=verify_protocol(args.protocol_sha256)
        report.update(protocol_sha256=sha,model_config=protocol['model'],source=protocol['source'],data=protocol['data'])
        for name in protocol['source']:
            target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
        report['inputs'][str(args.train_cache/'manifest.json')]=digest(args.train_cache/'manifest.json')
        if args.mode=='fit':
            cache,cm=load_cache(args.train_cache,'train',sha)
            dates=cache['dates'];x=cache['features'];q=cache['target']
            report.update(train_ids=cache['case_ids'].tolist(),train_dates=dates.astype(str).tolist())
            permutation=np.broadcast_to(np.arange(len(dates))[:,None],(len(dates),3)).copy()
            rng=np.random.default_rng(4173)
            seasons=(dates.month.to_numpy()%12)//3
            for year in sorted(set(dates.year)):
                for season in range(4):
                    group=np.flatnonzero((dates.year==year)&(seasons==season))
                    for j in range(3):
                        permutation[group,j]=rng.permutation(group)
            np.save(out/'permutation_indices.npy',permutation)
            families={'real':{},'constant':{},'permuted':{}}
            for j,unit in enumerate(TARGET_UNITS):
                families['real'][unit]=fit_mapping(x[:,j],q[:,j])
                families['constant'][unit]=fit_mapping(x[:,j],q[:,j],constant=True)
                families['permuted'][unit]=fit_mapping(x[:,j],q[permutation[:,j],j])
                print('MAPPING_FIT',unit,{name:families[name][unit]['iterations'] for name in families},flush=True)
            (out/'models.json').write_text(json.dumps(families,indent=2))
            restored=json.loads((out/'models.json').read_text())
            predictions={name:predict_all(x,family) for name,family in families.items()}
            for name in families:
                np.testing.assert_array_equal(predictions[name],predict_all(x,restored[name]))
            np.savez_compressed(out/'training_distributions.npz',**predictions,case_ids=cache['case_ids'],units=np.asarray(TARGET_UNITS))
            report['training_cross_entropy_in_sample']={name:float(-np.mean(np.sum(q*np.log(np.maximum(pi,1e-300)),axis=-1))) for name,pi in predictions.items()}
            report['serialized_coefficient_replay_exact']=True
            report['lock_file_excluded_from_artifact_hash_cycle']='training_lock.json is created only after the completed manifest is sealed.'
        else:
            if not args.screen_cache or not args.training_lock or not args.training_lock_sha256 or digest(args.training_lock)!=args.training_lock_sha256:
                raise ValueError('Evaluation requires explicit screen cache and locked training SHA')
            lock=json.loads(args.training_lock.read_text())
            fit_path=ROOT/lock['fit_manifest']
            if lock['status']!='locked' or lock['protocol_sha256']!=sha or digest(fit_path)!=lock['fit_manifest_sha256']:
                raise ValueError('Locked training manifest changed')
            fit=json.loads(fit_path.read_text())
            if fit['status']!='complete' or fit['mode']!='fit' or fit['labels_max_year']!=2018:
                raise ValueError('Invalid earlier training fit')
            if fit['inputs'][str(args.train_cache/'manifest.json')]!=digest(args.train_cache/'manifest.json'):
                raise ValueError('Training target cache changed after lock')
            for name,expected in lock['model_artifacts'].items():
                if digest(ROOT/name)!=expected:
                    raise ValueError('Locked model artifact changed')
            families=json.loads((fit_path.parent/'models.json').read_text())
            cache,cm=load_cache(args.screen_cache,'screen',sha)
            if cm['training_lock_sha256']!=args.training_lock_sha256:
                raise ValueError('Screen targets were not gated by this training lock')
            report.update(training_lock_sha256=args.training_lock_sha256,screen_ids=cache['case_ids'].tolist(),screen_dates=cache['dates'].astype(str).tolist())
            report['inputs'].update({str(args.training_lock):digest(args.training_lock),str(args.screen_cache/'manifest.json'):digest(args.screen_cache/'manifest.json'),str(fit_path):digest(fit_path)})
            distributions={name:predict_all(cache['features'],family) for name,family in families.items()}
            distributions['prior']=np.broadcast_to(softmax(PRIOR),(len(cache['case_ids']),3,17)).copy()
            distributions['unshifted']=np.zeros_like(distributions['prior']);distributions['unshifted'][:,:,8]=1.
            results={}
            for year in [2019,2020]:
                mask=cache['dates'].year==year
                rows={name:metrics(pi,cache,mask) for name,pi in distributions.items()}
                real,constant,permuted,unshifted=[rows[k] for k in ['real','constant','permuted','unshifted']]
                gates={'hour_error_vs_unshifted':real['hour_errors']<=.9*unshifted['hour_errors'],
                       'hour_error_vs_constant':real['hour_errors']<=.9*constant['hour_errors'],
                       'expected_mismatch_vs_controls':real['expected_hour_mismatch']<min(constant['expected_hour_mismatch'],permuted['expected_hour_mismatch']),
                       'target_cross_entropy_vs_controls':real['soft_target_cross_entropy']<min(constant['soft_target_cross_entropy'],permuted['soft_target_cross_entropy']),
                       'three_target_auc_vs_controls':real['three_target_pooled_auc']>max(constant['three_target_pooled_auc'],permuted['three_target_pooled_auc']),
                       'per_unit_harm_bound':all(real['per_unit_hour_errors'][u]<=1.05*constant['per_unit_hour_errors'][u] for u in TARGET_UNITS)}
                results[str(year)]={'metrics':rows,'gates':gates,'passed':all(gates.values())}
            np.savez_compressed(out/'screen_distributions.npz',**distributions,case_ids=cache['case_ids'],units=np.asarray(TARGET_UNITS))
            probabilities={name:np.stack([schedule_mixture(pi[:,j],cache['paths'][:,j]) for j in range(3)],axis=1) for name,pi in distributions.items()}
            np.savez_compressed(out/'screen_probabilities.npz',**probabilities,labels=cache['labels'],case_ids=cache['case_ids'],units=np.asarray(TARGET_UNITS))
            report.update(results=results,gate_passed=all(row['passed'] for row in results.values()))
            print(json.dumps({'gate_passed':report['gate_passed'],'results':results},indent=2),flush=True)
        report['status']='complete'
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
            stream.write(f'\n## {args.run_id} — {report["timestamp_utc"]}\n\nStatus {report["status"]}; latent-shadow {args.mode}. Command `{report["command"]}`. Three17-state convex linear maps, eight features, fixedsofttargets/prior/ridge and chronological2015–2018training/2019–2020screen. Source/data/cache/splits/coefficients/normalization/permutation/control hashes in `artifacts/{args.run_id}/manifest.json`. Screen gate {report.get("gate_passed")}; scores cover three targets only, no whole14-unit production-anchor claim. Runtime {report["runtime_seconds"]:.3f}s; peakRSS {report["peak_rss_bytes"]}bytes. No Kaggle action.\n')
    if args.mode=='fit' and report['status']=='complete':
        lock={'status':'locked','timestamp_utc':datetime.now(timezone.utc).isoformat(),
              'protocol_sha256':args.protocol_sha256,'fit_manifest':str((out/'manifest.json').relative_to(ROOT)),
              'fit_manifest_sha256':digest(out/'manifest.json'),
              'model_artifacts':{str((out/name).relative_to(ROOT)):digest(out/name) for name in ['models.json','permutation_indices.npy','training_distributions.npz']}}
        with (out/'training_lock.json').open('x') as stream:
            json.dump(lock,stream,indent=2)
        print('TRAINING_LOCK_SHA256',digest(out/'training_lock.json'),flush=True)


if __name__=='__main__':
    main()
