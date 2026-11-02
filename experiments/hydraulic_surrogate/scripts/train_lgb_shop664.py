"""Fixed-round LightGBM664 feature ablation with strict chronological selection."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import resource
import shlex
import shutil
import subprocess
import sys
import time
import traceback
for variable in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[variable] = '2'
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
WEIGHTS = [0., .1, .25, .5]


def sha(path: Path) -> str:
    """Hash immutable source/data bytes in bounded chunks."""
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def environment() -> dict:
    """Record the exact local CPU/LightGBM environment."""
    return {'python': sys.version, 'platform': platform.platform(), 'machine': platform.machine(),
            'cpu_count': os.cpu_count(), 'packages': {name: importlib.metadata.version(name) for name in
            ['numpy','pandas','lightgbm','scikit-learn','scipy','PyYAML']},
            'thread_environment': {name: os.environ[name] for name in
            ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']}}


def peak_rss() -> int:
    """Normalize ru_maxrss bytes across Darwin and Linux."""
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (1 if sys.platform == 'darwin' else 1024)


def memory_free_percent() -> float:
    """Read macOS memory-pressure headroom without allocating a probe."""
    completed = subprocess.run(['memory_pressure','-Q'], capture_output=True, text=True, check=True)
    match = re.search(r'System-wide memory free percentage:\s*(\d+)%', completed.stdout)
    if match is None:
        raise RuntimeError('Cannot establish memory headroom')
    return float(match.group(1))


def folds_and_samples(metadata: pd.DataFrame) -> dict:
    """Recreate the original two-year RNG sequence and six-day purged samples."""
    dates = pd.to_datetime(metadata.starttime, utc=True)
    rng = np.random.default_rng(4173); result = {}
    for year in [2021, 2022]:
        begin = pd.Timestamp(f'{year}-01-01', tz='UTC'); end = pd.Timestamp(f'{year+1}-01-01', tz='UTC')
        train = np.flatnonzero(dates < begin-pd.Timedelta(days=6))
        validation = np.flatnonzero((dates >= begin) & (dates < end))
        assert dates.iloc[train].max() + pd.Timedelta(hours=167) < begin
        rows = (train[:,None]*168 + np.arange(168)).ravel()
        sampled = np.sort(rng.choice(rows, size=int(len(rows)*.4), replace=False))
        result[str(year)] = {'train_cases': train, 'validation_cases': validation,
                            'sample': sampled, 'validation_rows': (validation[:,None]*168+np.arange(168)).ravel()}
    return result


def choose(scores: dict, control_auc: float, selected: float | None = None) -> tuple[float,bool]:
    """Prefer lower tied weights and require strict gain against both controls."""
    weight = max(WEIGHTS, key=lambda w:(scores[str(w)],-w)) if selected is None else selected
    assert weight in WEIGHTS
    return weight, bool(weight > 0 and scores[str(weight)] > max(scores['0.0'], control_auc))


def validate_lock(path: Path, expected: str, protocol: dict, protocol_sha: str,
                  audit_path: Path, audit_sha: str) -> dict:
    """Require independent eligibility before any2022 labels/features are loaded."""
    assert sha(path) == expected and sha(audit_path) == audit_sha
    lock = json.loads(path.read_text()); audit = json.loads(audit_path.read_text())
    assert audit['status'] == 'passed' and audit['selection_lock_sha256'] == expected and audit['eligible_for_2022']
    assert lock['version'] == 'lgb_shop664_selection_lock_v1' and lock['selection_year'] == 2021
    assert lock['protocol_sha256'] == protocol_sha and lock['trainer_sha256'] == protocol['source']['scripts/train_lgb_shop664.py']
    assert lock['candidate_weights'] == WEIGHTS
    chosen, eligible = choose(lock['scores_2021'], lock['control_2021_micro_auc'])
    assert eligible and lock['eligible_for_2022'] and chosen == lock['selected_weight']
    assert lock['selected_2021_micro_auc'] == lock['scores_2021'][str(chosen)]
    assert lock['anchor_2021_micro_auc'] == lock['scores_2021']['0.0']
    assert lock['anchor_prediction_sha256'] == protocol['anchor']['2021']['prediction_sha256']
    assert lock['control_prediction_sha256'] == protocol['control']['2021']['prediction_sha256']
    return lock


def load_reference(record: dict, labels: np.ndarray, ids: np.ndarray, units: list[str]) -> np.ndarray:
    """Verify exact original raw/projected reference identity and alignment."""
    manifest_path, prediction_path = ROOT/record['manifest'], ROOT/record['prediction']
    assert sha(manifest_path) == record['manifest_sha256'] and sha(prediction_path) == record['prediction_sha256']
    manifest = json.loads(manifest_path.read_text())
    assert manifest['status'] == 'complete' and manifest['artifacts'][prediction_path.name] == record['prediction_sha256']
    with np.load(prediction_path, allow_pickle=False) as saved:
        assert saved['case_ids'].tolist() == ids.tolist() and saved['units'].tolist() == units
        assert np.array_equal(saved['labels'],labels)
        values = saved['predictions'].astype(np.float64)
    assert values.shape == labels.shape and np.isfinite(values).all() and 0 <= values.min() <= values.max() <= 1
    return values


def check_closure(protocol: dict) -> tuple[dict,pd.DataFrame,dict]:
    """Check frozen code, all staged inputs and sample bytes without target decoding."""
    assert protocol['version'] == 'lgb_shop664_feature_ablation_v1' and protocol['candidate_weights'] == WEIGHTS
    assert protocol['environment'] == environment()
    for group in ['source','inputs','data']:
        for name, expected in protocol[group].items(): assert sha(ROOT/name) == expected, name
    for name in protocol['independent_prerequisites']:
        assert json.loads((ROOT/name).read_text())['status'] == 'passed', name
    original = json.loads((ROOT/protocol['baseline_manifest']).read_text())
    expected = dict(original['model_parameters']); expected['n_jobs'] = 2
    assert protocol['model_parameters'] == expected
    loader_protocol = json.loads((ROOT/protocol['loader_protocol']).read_text())
    assert protocol['feature_columns'] == loader_protocol['feature_columns']
    assert protocol['units'] == loader_protocol['units'] and len(protocol['feature_columns']) == 664
    stage = ROOT/protocol['stage']; paths = list(stage.rglob('*'))
    assert not any(p.is_symlink() for p in paths)
    assert {p.relative_to(stage).as_posix() for p in paths if p.is_file()} == set(protocol['stage_files'])
    for name, record in protocol['stage_files'].items():
        p = stage/name; assert p.stat().st_size == record['size_bytes'] and sha(p) == record['sha256'], name
    meta = pd.read_csv(ROOT/'sintef_project/data/kernel/Unit_commitment_decisions.csv',usecols=['Run No','starttime'])
    folds = folds_and_samples(meta)
    for year, split in folds.items():
        expected_fold=protocol['folds'][year]
        assert meta.iloc[split['train_cases']]['Run No'].tolist() == expected_fold['train_ids']
        assert meta.iloc[split['validation_cases']]['Run No'].tolist() == expected_fold['validation_ids']
        sample_path=ROOT/protocol['samples'][year]['path']
        assert sha(sample_path) == protocol['samples'][year]['sha256']
        saved=np.load(sample_path,allow_pickle=False)
        assert saved.dtype == split['sample'].dtype and saved.shape == split['sample'].shape
        assert saved.tobytes() == split['sample'].tobytes() and len(saved) == expected_fold['sampled_training_rows']
    return loader_protocol,meta,folds


def main() -> None:
    """Fit one chronological fold only after reviewed closure and resource gates."""
    parser=argparse.ArgumentParser()
    parser.add_argument('--protocol',type=Path,required=True);parser.add_argument('--protocol-sha256',required=True)
    parser.add_argument('--year',type=int,choices=[2021,2022],required=True);parser.add_argument('--run-id',required=True)
    parser.add_argument('--preflight',action='store_true');parser.add_argument('--independent-audit',type=Path)
    parser.add_argument('--independent-audit-sha256');parser.add_argument('--selection-lock',type=Path)
    parser.add_argument('--selection-lock-sha256');parser.add_argument('--lock-audit',type=Path);parser.add_argument('--lock-audit-sha256')
    args=parser.parse_args()
    assert Path(args.run_id).name==args.run_id and args.run_id not in ['','.','..']
    out=ROOT/'artifacts'/args.run_id;out.mkdir(exist_ok=False);started=time.monotonic()
    report={'run_id':args.run_id,'timestamp_utc':datetime.now(timezone.utc).isoformat(),'status':'initializing',
            'command':shlex.join([sys.executable,*sys.argv]),'protocol_sha256':args.protocol_sha256,'year':args.year,
            'environment':environment(),'training':not args.preflight,'kaggle_submission':None,'completed_units':0,'models':{},
            'caveat':'Both chronological years are reused development data; no public-score selection.'}
    try:
        assert sha(args.protocol)==args.protocol_sha256
        protocol=json.loads(args.protocol.read_text());loader_protocol,metadata,splits=check_closure(protocol)
        shutil.copy2(args.protocol,out/'protocol.json')
        for name in protocol['source']:
            dest=out/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
        report.update(exact_staged_input_closure=True,exact_original_sample_bytes=True,preflight_metadata_only=True)
        if args.preflight:
            report.update(status='passed_preflight_only',labels_decoded=False);return
        assert args.independent_audit and sha(args.independent_audit)==args.independent_audit_sha256
        audit=json.loads(args.independent_audit.read_text())
        assert audit['status']=='passed' and audit['protocol_sha256']==args.protocol_sha256
        report['independent_audit_sha256']=args.independent_audit_sha256
        lock=None
        if args.year==2022:
            assert args.selection_lock and args.selection_lock_sha256 and args.lock_audit and args.lock_audit_sha256
            lock=validate_lock(args.selection_lock,args.selection_lock_sha256,protocol,args.protocol_sha256,args.lock_audit,args.lock_audit_sha256)
            report.update(selection_lock_sha256=args.selection_lock_sha256,lock_audit_sha256=args.lock_audit_sha256)
        else:
            assert args.selection_lock is None and args.lock_audit is None
        report['memory_free_percent_before_build']=memory_free_percent()
        if report['memory_free_percent_before_build']<protocol['resources']['minimum_free_percent_before_build']:
            report.update(status='operationally_deferred_resources',training=False);return
        # Import the original byte-pinned loader before any sintef_project modules.
        loader_path=ROOT/'scripts/cuda_xgb664_loader.py'
        spec=importlib.util.spec_from_file_location('lgb664_frozen_feature_loader',loader_path)
        loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
        x,labels,train,mapping,meta,baseline=loader.build_verified_features(ROOT/protocol['stage'],args.year,loader_protocol)
        assert x.shape==(3647*168,664) and x.columns.tolist()==protocol['feature_columns']
        assert train['Run No'].tolist()==metadata['Run No'].tolist()
        units=protocol['units'];split=splits[str(args.year)];sample=split['sample'];validation=split['validation_rows']
        y=np.column_stack([labels[u]for u in units]).astype(np.uint8)
        assert np.isin(y[sample],[0,1]).all() and np.isin(y[validation],[0,1]).all()
        assert max(pd.to_datetime(train.iloc[split['train_cases']].starttime,utc=True))+pd.Timedelta(hours=167)<pd.Timestamp(f'{args.year}-01-01',tz='UTC')
        xt=x.iloc[sample].copy();xv=x.iloc[validation].copy();del x
        report.update(features=protocol['feature_columns'],feature_spec=meta['feature_spec'],units=units,
                      train_ids=protocol['folds'][str(args.year)]['train_ids'],validation_ids=protocol['folds'][str(args.year)]['validation_ids'],
                      sample_sha256=protocol['samples'][str(args.year)]['sha256'],sampled_training_rows=len(sample),
                      model_parameters=protocol['model_parameters'],maximum_decoded_target_year=args.year,peak_rss_after_features_bytes=peak_rss())
        shutil.copy2(ROOT/protocol['samples'][str(args.year)]['path'],out/f'train_sample_{args.year}.npy')
        predictions=np.zeros((len(validation),14),dtype=np.float32)
        for j,unit in enumerate(units):
            free=memory_free_percent()
            if free<protocol['resources']['minimum_free_percent_before_unit']:
                report.update(status='operationally_deferred_resources',memory_free_percent_stopped=free);return
            tick=time.monotonic();model=lgb.LGBMClassifier(**protocol['model_parameters'])
            model.fit(xt,y[sample,j])
            assert model.booster_.num_trees()==350 and model.booster_.num_feature()==664
            predicted=model.predict_proba(xv)[:,1]
            checkpoint=out/f'fold{args.year}_{unit}.txt';model.booster_.save_model(str(checkpoint))
            reloaded=lgb.Booster(model_file=str(checkpoint));replay=reloaded.predict(xv,num_threads=2)
            assert predicted.dtype==replay.dtype and predicted.tobytes()==replay.tobytes(),unit
            predictions[:,j]=predicted
            report['models'][unit]={'checkpoint_sha256':sha(checkpoint),'trees':350,'features':664,
                                  'replay_rows':len(validation),'checkpoint_replay_max_error':0.,
                                  'float64_checkpoint_replay_bitwise':True,'runtime_seconds':time.monotonic()-tick}
            report['completed_units']=j+1
            print(json.dumps({'year':args.year,'unit':unit,'completed_units':j+1,'seconds':report['models'][unit]['runtime_seconds'],'peak_rss_bytes':peak_rss()}),flush=True)
            (out/'progress.json').write_text(json.dumps(report,indent=2))
            del model,reloaded
        np.save(out/'raw_predictions_before_selection.npy',predictions)
        seal={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'completed_units':14,
              'prediction_sha256':sha(out/'raw_predictions_before_selection.npy'),
              'models':{unit:record['checkpoint_sha256']for unit,record in report['models'].items()}}
        (out/'fit_and_prediction_seal.json').write_text(json.dumps(seal,indent=2))
        fold_labels=y[validation];ids=train.iloc[split['validation_cases']]['Run No'].to_numpy()
        np.savez_compressed(out/f'validation_{args.year}.npz',predictions=predictions,labels=fold_labels,case_ids=ids,units=np.asarray(units))
        # Postprocessing runs only after all14 fixed models and saved replays are sealed.
        source=ROOT/'sintef_project/postprocess.py'
        spec=importlib.util.spec_from_file_location('lgb664_frozen_postprocess',source)
        post=importlib.util.module_from_spec(spec);spec.loader.exec_module(post)
        anchor=load_reference(protocol['anchor'][str(args.year)],fold_labels,ids,units)
        control=load_reference(protocol['control'][str(args.year)],fold_labels,ids,units)
        raw=predictions.astype(np.float64)
        weights=WEIGHTS if lock is None else [0.,lock['selected_weight']]
        values={str(w):post.project_vinje((1-w)*anchor+w*raw,units)for w in weights}
        assert np.array_equal(values['0.0'],anchor)
        scores={w:float(roc_auc_score(fold_labels.ravel(),p.ravel()))for w,p in values.items()}
        selected=max(WEIGHTS,key=lambda w:(scores[str(w)],-w))if lock is None else lock['selected_weight']
        matched=post.project_vinje((1-selected)*anchor+selected*control,units)
        control_auc=float(roc_auc_score(fold_labels.ravel(),matched.ravel()))
        selected,eligible=choose(scores,control_auc,selected)
        np.savez_compressed(out/f'selected_validation_{args.year}.npz',predictions=values[str(selected)],labels=fold_labels,case_ids=ids,units=np.asarray(units))
        report.update(status='complete',completed_rounds_per_unit=350,checkpoint_replay_bitwise=True,scores=scores,
                      selected_weight=selected,selected_micro_auc=scores[str(selected)],anchor_micro_auc=scores['0.0'],
                      control_micro_auc=control_auc,beats_anchor_and_control=eligible,
                      raw_micro_auc=float(roc_auc_score(fold_labels.ravel(),predictions.ravel())),fit_and_prediction_seal_sha256=sha(out/'fit_and_prediction_seal.json'))
        if lock is None:
            selection={'version':'lgb_shop664_selection_lock_v1','selection_year':2021,'locked_utc':datetime.now(timezone.utc).isoformat(),
                       'protocol_sha256':args.protocol_sha256,'trainer_sha256':sha(Path(__file__)),'candidate_weights':WEIGHTS,
                       'scores_2021':scores,'selected_weight':selected,'selected_2021_micro_auc':scores[str(selected)],
                       'anchor_2021_micro_auc':scores['0.0'],'control_2021_micro_auc':control_auc,'eligible_for_2022':eligible,
                       'anchor_prediction_sha256':protocol['anchor']['2021']['prediction_sha256'],
                       'control_prediction_sha256':protocol['control']['2021']['prediction_sha256'],
                       'raw_prediction_sha256':sha(out/'validation_2021.npz'),'fit_and_prediction_seal_sha256':sha(out/'fit_and_prediction_seal.json')}
            (out/'selection_2021.json').write_text(json.dumps(selection,indent=2))
            report['selection_lock_sha256']=sha(out/'selection_2021.json')
        else:
            report.update(accepted=eligible,meets_transfer_gate=min(lock['selected_2021_micro_auc'],scores[str(selected)])>=.998)
        print(json.dumps({k:report[k]for k in ['status','selected_weight','selected_micro_auc','anchor_micro_auc','control_micro_auc','beats_anchor_and_control']}),flush=True)
    except BaseException:
        report.update(status='failed',error=traceback.format_exc());raise
    finally:
        report.update(runtime_seconds=time.monotonic()-started,peak_rss_bytes=peak_rss())
        report['artifacts']={p.relative_to(out).as_posix():sha(p)for p in out.rglob('*')if p.is_file()and p.name!='manifest.json'}
        (out/'manifest.json').write_text(json.dumps(report,indent=2))
        (out/'experiment_record.md').write_text(f'\n## {args.run_id} — {report["timestamp_utc"]}\n\nStatus {report["status"]}. Frozen LightGBM664 feature ablation: exact608prefix plus56SHOP, original350trees/31leaves/lr.06/minleaf100/.9featurefraction/L2=1/40%sample,2CPUthreads, no early stopping. Command `{report["command"]}`. Year{args.year}; selectedweight {report.get("selected_weight")}, selectedAUC {report.get("selected_micro_auc")}, SHOPanchor {report.get("anchor_micro_auc")}, matched originalLGB608 {report.get("control_micro_auc")}. Runtime {report["runtime_seconds"]:.3f}s, peakRSS {report["peak_rss_bytes"]}bytes. Full sources/data/splits/samples/config/checkpoints/replays/environment/gates: `artifacts/{args.run_id}/manifest.json`. No final training/submission;2022 requires independent eligible2021lock.\n')


if __name__=='__main__':main()
