"""One 2021-only direct probability-splice experiment for frozen latent maps."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
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
from sklearn.metrics import roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.build_latent_shadow_targets import verify_protocol
from scripts.fit_latent_shadow import load_cache,predict_all
from sintef_project.latent_shadow import FEATURES,GRID,TARGET_UNITS,ShadowInputs,fit_mapping,read_selected_labels,schedule_mixture,soft_targets
from sintef_project.nonlinear_pq import digest
from sintef_project.postprocess import project_vinje

WEIGHTS=[0.,.025,.05,.1,.25]
TRAIN_CACHE=ROOT/'artifacts/latent-shadow-targets-train-001'
SCREEN_CACHE=ROOT/'artifacts/latent-shadow-targets-screen-001'
ANCHOR=ROOT/'artifacts/lp-context-joint-comparison-001'


def splice(anchor: np.ndarray,probability: np.ndarray,units: list[str],weight: float) -> np.ndarray:
    """Replace only the three fixed target columns with one global weight."""
    if weight not in WEIGHTS or probability.shape!=(len(anchor),3):
        raise ValueError('Unexpected global weight or latent probability shape')
    if not np.isfinite(probability).all() or probability.min()<0 or probability.max()>1:
        raise ValueError('Invalid latent probabilities')
    indices=[units.index(u) for u in TARGET_UNITS]
    result=anchor.astype(np.float64).copy()
    if weight:
        result[:,indices]=(1-weight)*result[:,indices]+weight*probability
    result=project_vinje(result,units)
    other=[j for j,u in enumerate(units) if u not in TARGET_UNITS]
    np.testing.assert_array_equal(result[:,other],anchor[:,other])
    if not weight:
        np.testing.assert_array_equal(result,anchor)
    return result


def select_weight(scores: dict[str,float]) -> float:
    """Require the exact finite grid and prefer lower weights on exact ties."""
    if set(scores)!={str(w) for w in WEIGHTS} or not all(np.isfinite(v) and 0<=v<=1 for v in scores.values()):
        raise ValueError('Invalid complete 2021 score grid')
    return max(WEIGHTS,key=lambda w:scores[str(w)])


def check_frozen_config(expected_sha: str) -> dict:
    """Verify new protocol pins plus every original learnability dependency."""
    path=ROOT/'docs/latent_shadow_splice_protocol.json'
    if digest(path)!=expected_sha:
        raise ValueError('Explicit splice protocol SHA mismatch')
    protocol=json.loads(path.read_text())
    for name,sha in protocol['source'].items():
        if digest(ROOT/name)!=sha:
            raise ValueError(f'Changed splice source: {name}')
    verify_protocol(protocol['learnability_protocol_sha256'])
    for name,sha in protocol['inputs'].items():
        if digest(ROOT/name)!=sha:
            raise ValueError(f'Changed frozen input: {name}')
    learned=json.loads((ROOT/protocol['learnability_result']).read_text())
    if learned['status']!='complete' or not learned['gate_passed']:
        raise ValueError('Early-year predictive screen did not pass')
    if protocol['candidate_weights']!=WEIGHTS or protocol['units']!=TARGET_UNITS or protocol['year']!=2021:
        raise ValueError('2021-only frozen scope changed')
    return protocol


def fit_families(x: np.ndarray,target: np.ndarray,dates: pd.DatetimeIndex) -> tuple[dict,np.ndarray]:
    """Fit the same real/constant/permuted model families on an eligible training prefix."""
    permutation=np.broadcast_to(np.arange(len(dates))[:,None],(len(dates),3)).copy()
    rng=np.random.default_rng(4173)
    seasons=(dates.month.to_numpy()%12)//3
    for year in sorted(set(dates.year)):
        for season in range(4):
            group=np.flatnonzero((dates.year==year)&(seasons==season))
            for j in range(3):
                permutation[group,j]=rng.permutation(group)
    families={'real':{},'constant':{},'permuted':{}}
    for j,unit in enumerate(TARGET_UNITS):
        families['real'][unit]=fit_mapping(x[:,j],target[:,j])
        families['constant'][unit]=fit_mapping(x[:,j],target[:,j],constant=True)
        families['permuted'][unit]=fit_mapping(x[:,j],target[permutation[:,j],j])
    return families,permutation


def main() -> None:
    """Fit only through 2020 and choose one full-matrix 2021 probability splice."""
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--protocol-sha256',required=True)
    args=parser.parse_args()
    out=ROOT/'artifacts'/args.run_id;out.mkdir(exist_ok=False)
    started=time.monotonic()
    report={'run_id':args.run_id,'timestamp_utc':datetime.now(timezone.utc).isoformat(),
            'command':shlex.join([sys.executable,*sys.argv]),'status':'initializing','year':2021,
            'seed':4173,'kaggle_submission':None,'candidate_weights':WEIGHTS,'inputs':{},
            'environment':{'python':sys.version,'platform':platform.platform(),'machine':platform.machine(),
                           'packages':{n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn','PyYAML']}},
            'validation_label_access_utc':None,'training':'Only fixed small latent maps; no new tree/stacking model'}
    try:
        protocol=check_frozen_config(args.protocol_sha256)
        report.update(protocol_sha256=args.protocol_sha256,source=protocol['source'],data=protocol['data'],inputs=protocol['inputs'].copy())
        for name in protocol['source']:
            target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
        earlier,_=load_cache(TRAIN_CACHE,'train',protocol['learnability_protocol_sha256'])
        later,_=load_cache(SCREEN_CACHE,'screen',protocol['learnability_protocol_sha256'])
        raw_metadata=pd.read_csv(ROOT/'sintef_project/data/kernel/Unit_commitment_decisions.csv',usecols=['Run No','starttime'])
        raw_metadata.starttime=pd.to_datetime(raw_metadata.starttime,utc=True)
        training=raw_metadata[raw_metadata.starttime<=pd.Timestamp('2020-12-25',tz='UTC')]
        validation=raw_metadata[raw_metadata.starttime.dt.year==2021]
        if len(training)!=2186 or len(validation)!=365 or training.starttime.max()+pd.Timedelta(days=7)!=validation.starttime.min():
            raise ValueError('Unexpected complete 2021 split or purge')
        reused_ids=np.concatenate([earlier['case_ids'],later['case_ids']])
        missing=training[~training['Run No'].isin(reused_ids)]
        if missing['Run No'].tolist()!=[1456,1457,1458,1459,1460,1461]:
            raise ValueError('Only the six original 2018 purge cases may be newly generated for training')
        cache=ROOT/'artifacts/nonlinear-pq-features-001'
        cm=json.loads((cache/'manifest.json').read_text())
        if digest(cache/'manifest.json')!=protocol['cache_manifest_sha256'] or digest(ROOT/'artifacts/lp-full-001/manifest.json')!=cm['lp_manifest_sha256']:
            raise ValueError('Frozen LP/nonlinear cache changed')
        for name in ['features.npz','raw_on_rewards.npz']:
            if digest(cache/name)!=cm['artifacts'][name]:
                raise ValueError('Frozen nonlinear arrays changed')
        inputs=ShadowInputs(ROOT,ROOT/'artifacts/lp-full-001')
        with np.load(cache/'features.npz',allow_pickle=False) as z:
            saved_features=z['values']
            cached_ids=z['case_ids']
        with np.load(cache/'raw_on_rewards.npz',allow_pickle=False) as z:
            saved_rewards=z['rewards']
        id_to_index={int(v):i for i,v in enumerate(cached_ids)}
        gap_labels=read_selected_labels(ROOT/'sintef_project/data/kernel/Unit_commitment_decisions.csv',missing['Run No'].tolist())
        gap_x,gap_target=[],[]
        for i,case_id in enumerate(missing['Run No']):
            index=id_to_index[int(case_id)]
            values=inputs.grid_case(index)
            np.testing.assert_array_equal(values['old_features'],saved_features[index])
            np.testing.assert_array_equal(values['rewards'][:,8,:].T,saved_rewards[index])
            error=np.sum(values['paths']!=gap_labels[i,:,None,:],axis=-1)
            target=soft_targets(error)
            np.savez_compressed(out/f'train_gap_{case_id:04d}.npz',**values,labels=gap_labels[i],target=target,error=error,case_id=np.asarray(case_id))
            gap_x.append(values['features']);gap_target.append(target)
        all_ids=np.concatenate([reused_ids,missing['Run No'].to_numpy()])
        order=np.argsort(all_ids)
        x=np.concatenate([earlier['features'],later['features'],np.stack(gap_x)])[order]
        target=np.concatenate([earlier['target'],later['target'],np.stack(gap_target)])[order]
        if all_ids[order].tolist()!=training['Run No'].tolist():
            raise ValueError('Combined training IDs mismatch')
        report.update(train_ids=training['Run No'].tolist(),train_dates=training.starttime.astype(str).tolist(),
                      reused_target_cases=len(reused_ids),new_target_case_ids=missing['Run No'].tolist(),maximum_fitted_label_year=2020)
        families,permutation=fit_families(x,target,pd.DatetimeIndex(training.starttime))
        np.save(out/'permutation_indices.npy',permutation)
        (out/'models.json').write_text(json.dumps(families,indent=2))
        restored=json.loads((out/'models.json').read_text())
        train_predictions={name:predict_all(x,family) for name,family in families.items()}
        for name in families:
            np.testing.assert_array_equal(train_predictions[name],predict_all(x,restored[name]))
        np.savez_compressed(out/'training_distributions.npz',**train_predictions,case_ids=training['Run No'].to_numpy(),units=np.asarray(TARGET_UNITS))
        fit_lock={'kind':'latent_shadow_splice_fit_v1','status':'locked','year':2021,
                  'timestamp_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':args.protocol_sha256,
                  'train_ids':report['train_ids'],'train_dates':report['train_dates'],'maximum_fitted_label_year':2020,
                  'source':protocol['source'],'data':protocol['data'],'inputs':protocol['inputs'],
                  'artifacts':{name:digest(out/name) for name in ['models.json','permutation_indices.npy','training_distributions.npz']},
                  'coefficient_replay_exact':True}
        (out/'fit_lock.json').write_text(json.dumps(fit_lock,indent=2))
        report['fit_lock_sha256']=digest(out/'fit_lock.json')
        print('FIT_LOCK',report['fit_lock_sha256'],'reused',len(reused_ids),'new',len(missing),flush=True)
        validation_x,validation_paths=[],[]
        for i,case_id in enumerate(validation['Run No']):
            index=id_to_index[int(case_id)]
            values=inputs.grid_case(index)
            np.testing.assert_array_equal(values['old_features'],saved_features[index])
            np.testing.assert_array_equal(values['rewards'][:,8,:].T,saved_rewards[index])
            np.savez_compressed(out/f'validation_input_{case_id:04d}.npz',**values,case_id=np.asarray(case_id))
            validation_x.append(values['features']);validation_paths.append(values['paths'])
            if (i+1)%100==0:
                print('VALIDATION_INPUTS',i+1,len(validation),flush=True)
        validation_x,validation_paths=np.stack(validation_x),np.stack(validation_paths)
        distributions={name:predict_all(validation_x,family) for name,family in families.items()}
        for name in distributions:
            np.testing.assert_array_equal(distributions[name],predict_all(validation_x,restored[name]))
        raw={name:np.stack([schedule_mixture(pi[:,j],validation_paths[:,j]) for j in range(3)],axis=2).reshape(-1,3)
             for name,pi in distributions.items()}
        np.savez_compressed(out/'raw_validation_2021.npz',**raw,case_ids=validation['Run No'].to_numpy(),units=np.asarray(TARGET_UNITS))
        np.savez_compressed(out/'validation_distributions_2021.npz',**distributions,case_ids=validation['Run No'].to_numpy(),units=np.asarray(TARGET_UNITS))
        report['raw_predictions_saved_before_validation_labels_utc']=datetime.now(timezone.utc).isoformat()
        prediction_lock={'kind':'latent_shadow_splice_predictions_v1','timestamp_utc':report['raw_predictions_saved_before_validation_labels_utc'],
            'protocol_sha256':args.protocol_sha256,'fit_lock_sha256':report['fit_lock_sha256'],
            'case_ids':validation['Run No'].tolist(),'target_units':TARGET_UNITS,'coefficient_replay_exact':True,
            'artifacts':{p.name:digest(p) for p in out.glob('validation_input_*.npz')}}
        prediction_lock['artifacts'].update({name:digest(out/name) for name in ['raw_validation_2021.npz','validation_distributions_2021.npz']})
        (out/'prediction_lock.json').write_text(json.dumps(prediction_lock,indent=2))
        report['prediction_lock_sha256']=digest(out/'prediction_lock.json')
        anchor_path=ANCHOR/'selected_validation_2021.npz'
        report['validation_label_access_utc']=datetime.now(timezone.utc).isoformat()
        with np.load(anchor_path,allow_pickle=False) as z:
            anchor,y,ids,units=z['predictions'],z['labels'],z['case_ids'],z['units'].tolist()
        if (ids.tolist()!=validation['Run No'].tolist() or units!=protocol['anchor_units']
                or anchor.shape!=y.shape or anchor.shape!=(365*168,14)):
            raise ValueError('Anchor labels/IDs/schema mismatch')
        scores={str(w):float(roc_auc_score(y.ravel(),splice(anchor,raw['real'],units,w).ravel())) for w in WEIGHTS}
        chosen=select_weight(scores)
        controls={name:float(roc_auc_score(y.ravel(),splice(anchor,raw[name],units,chosen).ravel())) for name in ['constant','permuted']}
        selected=splice(anchor,raw['real'],units,chosen)
        eligible=bool(chosen>0 and scores[str(chosen)]>max(scores['0.0'],*controls.values()))
        np.savez_compressed(out/'selected_validation_2021.npz',predictions=selected,labels=y,case_ids=ids,units=np.asarray(units))
        lock={'kind':'latent_shadow_splice_selection_v1','year':2021,'timestamp_utc':datetime.now(timezone.utc).isoformat(),
              'protocol_sha256':args.protocol_sha256,'candidate_weights':WEIGHTS,'scores_2021':scores,
              'selected_weight':chosen,'selected_micro_auc':scores[str(chosen)],'anchor_micro_auc':scores['0.0'],
              'control_micro_auc':controls,'eligible_for_2022':eligible,'fit_lock_sha256':report['fit_lock_sha256'],
              'anchor_sha256':digest(anchor_path),'raw_prediction_sha256':digest(out/'raw_validation_2021.npz'),
              'selected_prediction_sha256':digest(out/'selected_validation_2021.npz')}
        (out/'selection_2021.json').write_text(json.dumps(lock,indent=2))
        report.update(status='complete',selected_weight=chosen,selected_micro_auc=scores[str(chosen)],anchor_micro_auc=scores['0.0'],
                      control_micro_auc=controls,eligible_for_2022=eligible,scores_2021=scores,
                      selection_lock_sha256=digest(out/'selection_2021.json'),
                      validation_ids=ids.tolist(),units=units,
                      raw_unit_auc={name:{u:float(roc_auc_score(y[:,units.index(u)],pred[:,j])) for j,u in enumerate(TARGET_UNITS)} for name,pred in raw.items()})
        print(json.dumps({k:report[k] for k in ['status','selected_weight','selected_micro_auc','anchor_micro_auc','control_micro_auc','eligible_for_2022','scores_2021']},indent=2),flush=True)
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
            stream.write(f'\n## {args.run_id} — {report["timestamp_utc"]}\n\nStatus {report["status"]}; strictly2021-only direct three-unit latent-map probability splice. Command `{report["command"]}`. Reuses2180oldtargets and generates6gapcases; fixed maps fit through2020-12-25, no same-rowfeaturestacking. Whole14-unit weights {WEIGHTS}; selected {report.get("selected_weight")}, AUC {report.get("selected_micro_auc")}, same-weightcontrols {report.get("control_micro_auc")}; eligiblefor2022 {report.get("eligible_for_2022")}. Source/data/cache/sample/fit-lock/prediction/selection hashes: `artifacts/{args.run_id}/manifest.json`. Runtime {report["runtime_seconds"]:.3f}s, peakRSS {report["peak_rss_bytes"]}bytes. No2022evaluation orKaggleaction.\n')


if __name__=='__main__':
    main()
