"""Synthetic-only eight-round 14-output vector-leaf CPU/serialization check."""
from __future__ import annotations
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[key]='2'
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
import xgboost as xgb

PACKAGES={'numpy':'2.4.6','scipy':'1.17.1','scikit-learn':'1.9.1','joblib':'1.6.0','threadpoolctl':'3.7.0','narwhals':'2.26.0','cloudpickle':'3.1.2','xgboost':'3.4.1'}
ENVIRONMENT_SHA='2094900a418de72fa78499a0ad92b2fab2019c24002f60edfa048d17bec0b92e'
ROOT=Path(__file__).resolve().parents[1]
PARAMETERS=dict(n_estimators=8,max_depth=3,learning_rate=.05,min_child_weight=20,
    reg_lambda=4.,reg_alpha=0.,max_bin=128,subsample=.9,colsample_bytree=.9,
    colsample_bylevel=1.,colsample_bynode=1.,random_state=4173,n_jobs=2,
    objective='binary:logistic',tree_method='hist',device='cpu',eval_metric='logloss',
    grow_policy='depthwise',max_delta_step=0.,validate_parameters=True,verbosity=1,
    multi_strategy='multi_output_tree')


def sha(path:Path)->str:
    """Hash exact synthetic/model artifacts."""
    with path.open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def require(value:bool,message:str)->None:
    """Fail without changing the isolated runtime or silently substituting a version."""
    if not value:raise ValueError(message)


def native_check(booster:xgb.Booster,model_path:Path)->dict:
    """Confirm eight actual shared trees and fourteen-vector leaves in native JSON."""
    config=json.loads(booster.save_config());learner=config['learner']
    require(learner['generic_param']['device']=='cpu','CPU execution required; no GPU claim')
    require('grow_quantile_histmaker' in json.dumps(learner['gradient_booster']),'CPU histogram updater required')
    require(learner['learner_train_param']['multi_strategy']=='multi_output_tree','Shared tree strategy changed')
    require(learner['objective']['name']=='binary:logistic','Binary multi-label objective changed')
    require(int(learner['learner_model_param']['num_target'])==14,'Fourteen targets required')
    require(booster.num_features()==664 and booster.num_boosted_rounds()==8,'Synthetic feature/round count differs')
    saved=json.loads(model_path.read_text());native=saved['learner']['gradient_booster']['model']
    require(int(native['gbtree_model_param']['num_trees'])==8 and len(native['trees'])==8,'Expected eight trees, not112 scalar trees')
    require(all(int(tree['tree_param']['size_leaf_vector'])==14 for tree in native['trees']),
            'Every actual tree must store14-output vector leaves')
    require(int(saved['learner']['learner_model_param']['num_target'])==14,'Saved target count differs')
    return config


def main()->None:
    """Fit synthetic data once or replay the saved model in a distinct process."""
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['fit','replay'],required=True)
    ap.add_argument('--directory',type=Path,required=True);ap.add_argument('--fit-manifest-sha256')
    a=ap.parse_args();out=a.directory;out.mkdir(exist_ok=True)
    manifest_path=out/f'{a.phase}_manifest.json';require(not manifest_path.exists(),'Immutable phase already exists')
    start=time.monotonic();report={'status':'initializing','timestamp_utc':datetime.now(timezone.utc).isoformat(),
        'phase':a.phase,'synthetic_only':True,'competition_data_read':False,'seed':4173,'pid':os.getpid(),
        'command':[sys.executable,*sys.argv],'parameters':PARAMETERS,'future_fold_authorized':False}
    try:
        require(sys.version.split()[0]=='3.14.6','Pinned isolated Python required')
        env_path=ROOT/'artifacts/xgb341-cpu-environment-001/manifest.json'
        require(sha(env_path)==ENVIRONMENT_SHA,'Environment manifest changed')
        prepared=json.loads(env_path.read_text())
        require(str(Path(sys.prefix).resolve())==prepared['prefix'],'Unexpected environment path')
        require(sha(Path(sys.executable).resolve())==prepared['python_executable_sha256'],'Python binary changed')
        for group in ['distribution_records','loaded_native_libraries']:
            for path,expected in prepared[group].items():require(sha(Path(path))==expected,'Installed package/native library changed: '+path)
        require(sha(Path(xgb.build_info()['libxgboost']))==prepared['xgboost_library_sha256'],'XGBoost native library changed')
        packages={name:importlib.metadata.version(name)for name in PACKAGES};require(packages==PACKAGES,'Package versions changed')
        require(sys.prefix!=sys.base_prefix,'Isolated venv required')
        cfg=Path(sys.prefix)/'pyvenv.cfg'
        require('include-system-site-packages = false' in cfg.read_text().lower(),'Unexpected venv')
        require(xgb.build_info().get('USE_CUDA') is False and xgb.build_info().get('USE_OPENMP') is True,'Pinned CPU/OpenMP build required')
        require(platform.machine()=='arm64','Native Apple Silicon required')
        report['environment']={'python':sys.version,'platform':platform.platform(),'packages':packages,
            'build_info':xgb.build_info(),'gpu_capability_tested':False,'venv_configuration':cfg.read_text(),
            'environment_manifest_sha256':ENVIRONMENT_SHA}
        if a.phase=='fit':
            require(not any((out/name).exists()for name in ['synthetic_x.npy','synthetic_y.npy','model.ubj','model.json']),
                    'Existing synthetic fit payload; do not overwrite')
            rng=np.random.default_rng(4173);x=rng.standard_normal((2048,664),dtype=np.float32)
            y=(x[:,:14]+.2*x[:,14:28]+.05*rng.standard_normal((2048,14))>0).astype(np.uint8)
            require(all(np.array_equal(np.unique(y[:,j]),[0,1])for j in range(14)),'Both labels required in every output')
            np.save(out/'synthetic_x.npy',x,allow_pickle=False);np.save(out/'synthetic_y.npy',y,allow_pickle=False)
            model=xgb.XGBClassifier(**PARAMETERS);tick=time.monotonic();model.fit(x,y)
            report['fit_seconds']=time.monotonic()-tick
            booster=model.get_booster();matrix=xgb.DMatrix(x,nthread=2)
            prediction=booster.predict(matrix)
            require(prediction.shape==(2048,14) and np.isfinite(prediction).all() and
                    0<=prediction.min()<=prediction.max()<=1,'Invalid multi-label probability matrix')
            require(model.n_classes_==2 and model.objective=='binary:logistic','Unexpected classification mode')
            booster.save_model(out/'model.ubj');booster.save_model(out/'model.json')
            config=native_check(booster,out/'model.json')
            np.save(out/'fit_predictions.npy',prediction,allow_pickle=False)
            (out/'fit_config.json').write_text(json.dumps(config,indent=2))
            (out/'expanded_parameters.json').write_text(json.dumps(model.get_params(),indent=2))
            report.update(status='complete',shape=[2048,664],target_shape=[2048,14],trees=8,vector_leaf_size=14,
                positive_labels_per_output=y.sum(axis=0).tolist(),probability_range=[float(prediction.min()),float(prediction.max())],
                cpu_execution=True,gpu_capability_tested=False)
            paths=['synthetic_x.npy','synthetic_y.npy','model.ubj','model.json','fit_predictions.npy','fit_config.json','expanded_parameters.json']
        else:
            require(a.fit_manifest_sha256 is not None and sha(out/'fit_manifest.json')==a.fit_manifest_sha256,'Explicit original fit hash required')
            fit=json.loads((out/'fit_manifest.json').read_text())
            require(fit['status']=='complete' and fit['synthetic_only'] and fit['pid']!=os.getpid(),'Separate successful fit process required')
            require(fit['parameters']==PARAMETERS and fit['environment']==report['environment'],'Fit/replay runtime or parameters changed')
            for name,h in fit['artifacts'].items():require(sha(out/name)==h,'Fit payload changed: '+name)
            x=np.load(out/'synthetic_x.npy',allow_pickle=False);y=np.load(out/'synthetic_y.npy',allow_pickle=False)
            require(x.shape==(2048,664) and x.dtype==np.float32 and y.shape==(2048,14),'Synthetic schema changed')
            booster=xgb.Booster();booster.load_model(out/'model.ubj')
            booster.set_param({'device':'cpu','nthread':2,'tree_method':'hist','multi_strategy':'multi_output_tree'})
            replay=booster.predict(xgb.DMatrix(x,nthread=2));config=native_check(booster,out/'model.json')
            original=np.load(out/'fit_predictions.npy',allow_pickle=False)
            require(replay.dtype==original.dtype and replay.shape==original.shape and replay.tobytes()==original.tobytes(),
                    'Fresh-process serialized checkpoint replay changed probabilities')
            other=xgb.Booster();other.load_model(out/'model.json')
            other.set_param({'device':'cpu','nthread':2,'tree_method':'hist','multi_strategy':'multi_output_tree'})
            json_replay=other.predict(xgb.DMatrix(x,nthread=2))
            require(json_replay.dtype==original.dtype and json_replay.tobytes()==original.tobytes(),'JSON replay differs')
            np.save(out/'replay_predictions.npy',replay,allow_pickle=False)
            (out/'replay_config.json').write_text(json.dumps(config,indent=2))
            report.update(status='complete',fit_manifest_sha256=a.fit_manifest_sha256,fit_pid=fit['pid'],
                exact_replay_rows=2048,output_count=14,ubj_and_json_replays_bitwise=True,cpu_execution=True,gpu_capability_tested=False)
            paths=['replay_predictions.npy','replay_config.json']
        report['artifacts']={name:sha(out/name)for name in paths}
    except BaseException:
        report.update(status='failed',error=traceback.format_exc());raise
    finally:
        report.update(runtime_seconds=time.monotonic()-start,source_sha256=sha(Path(__file__)),
            peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024))
        manifest_path.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'phase':a.phase,'status':report['status'],'manifest_sha256':sha(manifest_path)}),flush=True)


if __name__=='__main__':main()
