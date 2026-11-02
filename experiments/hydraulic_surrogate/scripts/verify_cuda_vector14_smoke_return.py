"""Verify one externally pinned synthetic CUDA vector-leaf return without fitting."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path,PurePosixPath
import platform
import shutil
import stat
import sys
import time
import traceback
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUN='cuda-vector14-smoke-001'
FILES={'smoke_cuda_vector14.py','fit.log','replay.log','synthetic_x.npy','synthetic_y.npy',
       'model.ubj','model.json','fit_predictions.npy','replay_predictions.npy','fit_config.json',
       'replay_config.json','expanded_parameters.json','fit_manifest.json','replay_manifest.json','manifest.json'}
PACKAGES={'numpy':'2.1.3','pandas':'2.3.3','xgboost':'3.4.1','scipy':'1.16.3','scikit-learn':'1.6.1','pyyaml':'6.0.3'}


def sha(path:Path)->str:
    """Hash exact bytes in bounded memory."""
    with path.open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def require(condition:bool,message:str)->None:
    """Fail closed on any provenance, schema or numerical mismatch."""
    if not condition:raise ValueError(message)


def safe_import(archive:Path,destination:Path)->Path:
    """Allow precisely the fifteen flat synthetic-smoke files and no symlinks."""
    with zipfile.ZipFile(archive)as z:
        infos=z.infolist();names=[i.filename for i in infos]
        expected={RUN+'/'+name for name in FILES}
        require(len(names)==len(set(names)) and set(names)==expected,'Archive namespace/file allowlist differs')
        require(sum(i.file_size for i in infos)<=32*1024*1024,'Synthetic archive unexpectedly large')
        for info in infos:
            path=PurePosixPath(info.filename)
            require(not path.is_absolute() and '..' not in path.parts and len(path.parts)==2,'Unsafe archive member')
            require(not stat.S_ISLNK(info.external_attr>>16) and not info.is_dir(),'Unexpected archive link/directory')
            require(info.file_size<=16*1024*1024,'Oversized individual smoke payload')
            target=destination/Path(*path.parts);target.parent.mkdir(parents=True,exist_ok=True)
            with z.open(info)as src,target.open('xb')as dst:shutil.copyfileobj(src,dst)
    return destination/RUN


def native_config(config:dict,parameters:dict)->None:
    """Check saved native fit settings and shared CUDA binary objective."""
    learner=config['learner'];generic=learner['generic_param']
    require(generic['device']=='cuda:1' and int(generic['nthread'])==2 and int(generic['seed'])==4173,'Native device/thread/seed differs')
    require(learner['learner_train_param']['multi_strategy']=='multi_output_tree','Scalar tree strategy')
    require(learner['objective']['name']=='binary:logistic','Wrong multi-label objective')
    require(int(learner['learner_model_param']['num_target'])==14 and int(learner['learner_model_param']['num_feature'])==664,'Native dimensions differ')
    require('grow_gpu_hist' in json.dumps(learner['gradient_booster']),'Missing CUDA histogram updater')
    tree=learner['gradient_booster']['tree_train_param']
    for name in ['max_depth','learning_rate','min_child_weight','reg_lambda','reg_alpha','max_bin',
                 'subsample','colsample_bytree','colsample_bylevel','colsample_bynode','max_delta_step']:
        require(abs(float(tree[name])-float(parameters[name]))<=1e-6,'Native parameter differs: '+name)
    require(tree['grow_policy']=='depthwise','Wrong tree growth policy')


def json_forest_prediction(saved:dict,x:np.ndarray)->np.ndarray:
    """Independently evaluate native14-vector leaves using float32 tree traversal."""
    learner=saved['learner'];param=learner['learner_model_param']
    require(int(param['num_target'])==14 and int(param['num_feature'])==664,'Saved dimensions differ')
    require(learner['objective']['name']=='binary:logistic','Saved objective differs')
    forest=learner['gradient_booster']['model'];trees=forest['trees']
    require(int(forest['gbtree_model_param']['num_trees'])==len(trees)==8,'Expected8 shared trees, not112 scalar trees')
    intercept=np.asarray(json.loads(param['base_score']),dtype=np.float32).reshape(-1)
    require(intercept.size in [1,14] and np.isfinite(intercept).all() and np.all((intercept>0)&(intercept<1)),'Invalid intercept')
    margin=np.broadcast_to(-np.log(1/intercept-1),(len(x),14)).astype(np.float32).copy()
    rows=np.arange(len(x))
    for tree in trees:
        tp=tree['tree_param'];nodes=int(tp['num_nodes'])
        require(int(tp['size_leaf_vector'])==14 and 1<=nodes<=15,'Invalid vector size/depth3 node count')
        left=np.asarray(tree['left_children'],dtype=np.int64);right=np.asarray(tree['right_children'],dtype=np.int64)
        feature=np.asarray(tree['split_indices'],dtype=np.int64);threshold=np.asarray(tree['split_conditions'],dtype=np.float32)
        require(all(len(v)==nodes for v in [left,right,feature,threshold]),'Tree array length differs')
        leaf_weights=np.asarray(tree['leaf_weights'],dtype=np.float32).reshape(-1,14)
        require(len(leaf_weights)==int(np.sum(left<0)) and np.isfinite(leaf_weights).all(),'Leaf coefficient schema differs')
        node=np.zeros(len(x),dtype=np.int64)
        for _ in range(3):
            active=left[node]>=0
            if not active.any():break
            current=node[active];f=feature[current]
            require(np.all((f>=0)&(f<664)),'Invalid split feature')
            go_left=x[rows[active],f]<threshold[current]
            node[active]=np.where(go_left,left[current],right[current])
            require(np.all((node>=0)&(node<nodes)),'Invalid child node')
        require(np.all(left[node]<0),'Tree exceeds declared depth3')
        # Native3.4 vector trees store the leaf-vector index in each leaf's right slot.
        index=right[node];require(np.all((index>=0)&(index<len(leaf_weights))),'Invalid leaf index')
        margin+=leaf_weights[index]
    return 1/(1+np.exp(-margin))


def main()->None:
    """Import only hosted-complete smoke evidence; this cannot authorize a real fold."""
    ap=argparse.ArgumentParser();ap.add_argument('--archive',type=Path,required=True)
    for name in ['archive-sha256','manifest-sha256','fit-manifest-sha256','replay-manifest-sha256','run-id']:ap.add_argument('--'+name,required=True)
    a=ap.parse_args();require(Path(a.run_id).name==a.run_id and a.run_id not in ['', '.', '..'],'Invalid output ID')
    out=ROOT/'artifacts'/a.run_id;out.mkdir(exist_ok=False);start=time.monotonic()
    report={'status':'initializing','timestamp_utc':datetime.now(timezone.utc).isoformat(),'training':False,
            'competition_data_read':False,'full_fold_authorized':False,'command':[sys.executable,*sys.argv],
            'environment':{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__}}
    try:
        preparation=ROOT/'artifacts/cuda-vector14-smoke-notebook-001/manifest.json'
        review=ROOT/'artifacts/cuda-vector14-smoke-independent-review-001/manifest.json'
        require(sha(preparation)=='5a83eb71c8f71767beddac87dcb7bf5d730321497ba03a894b60ff95d3c33c93','Preparation changed')
        require(sha(review)=='971cdfb18711dd69b037b79bb57834848d49c8a8891e038ee868282b4f86a006','Source approval changed')
        prep=json.loads(preparation.read_text());approval=json.loads(review.read_text())
        require(approval['status']=='passed_source_review_only' and approval['preparation_manifest_sha256']==sha(preparation),'Wrong source approval')
        for name,h in prep['artifacts'].items():require(sha(preparation.parent/name)==h,'Prepared bytes changed: '+name)
        require(sha(a.archive)==a.archive_sha256,'External hosted archive hash differs')
        directory=safe_import(a.archive,out/'imported')
        manifests={}
        for filename,expected in [('manifest.json',a.manifest_sha256),('fit_manifest.json',a.fit_manifest_sha256),('replay_manifest.json',a.replay_manifest_sha256)]:
            require(sha(directory/filename)==expected,'Externally observed phase hash differs: '+filename)
            m=json.loads((directory/filename).read_text());require(m['status']=='complete' and m['synthetic_only'],'Smoke failed or is not synthetic')
            require(m['competition_data_read'] is False and m['future_fold_authorized'] is False,'Unexpected real-data/fit scope')
            for name,h in m['artifacts'].items():require(name in FILES and sha(directory/name)==h,'Returned artifact changed: '+name)
            manifests[filename]=m
        top,fit,replay=[manifests[k]for k in ['manifest.json','fit_manifest.json','replay_manifest.json']]
        require(sha(directory/'smoke_cuda_vector14.py')==prep['worker_sha256']==fit['source_sha256']==replay['source_sha256'],'Worker source changed')
        require(fit['parameters']==replay['parameters']==approval['parameters'],'Synthetic parameters changed')
        require(top['fit_manifest_sha256']==a.fit_manifest_sha256==replay['fit_manifest_sha256'] and top['replay_manifest_sha256']==a.replay_manifest_sha256,'Phase linkage differs')
        require(fit['pid']!=replay['pid'] and replay['fit_pid']==fit['pid'],'Fresh-process replay missing')
        require(fit['phase']=='fit' and replay['phase']=='replay' and fit['seed']==replay['seed']==4173,'Phase or seed differs')
        require(fit['environment']==replay['environment'] and fit['environment']['packages']==PACKAGES,'Runtime differs')
        require(fit['environment']['python'].split()[0]=='3.13.15' and fit['environment']['build_info']['USE_CUDA'] is True,'Wrong Python/CUDA build')
        require(fit['environment']['gpu']=='Tesla T4, 15360 MiB, 580.178.04\nTesla T4, 15360 MiB, 580.178.04','GPU identity differs')
        require(top['per_phase_timeout_seconds']==120 and fit['cpu_fallback'] is replay['cpu_fallback'] is False,'Timeout/fallback differs')
        x=np.load(directory/'synthetic_x.npy',allow_pickle=False);y=np.load(directory/'synthetic_y.npy',allow_pickle=False)
        rng=np.random.default_rng(4173);expected_x=rng.standard_normal((2048,664),dtype=np.float32)
        expected_y=(expected_x[:,:14]+.2*expected_x[:,14:28]+.05*rng.standard_normal((2048,14))>0).astype(np.uint8)
        require(x.dtype==expected_x.dtype and x.shape==expected_x.shape and x.tobytes()==expected_x.tobytes(),'Synthetic inputs differ')
        require(y.dtype==expected_y.dtype and y.shape==expected_y.shape and y.tobytes()==expected_y.tobytes(),'Synthetic targets differ')
        require(fit['positive_labels_per_output']==y.sum(axis=0).tolist() and all(np.array_equal(np.unique(y[:,j]),[0,1])for j in range(14)),'Missing labels')
        prediction=np.load(directory/'fit_predictions.npy',allow_pickle=False);again=np.load(directory/'replay_predictions.npy',allow_pickle=False)
        require(prediction.dtype==again.dtype==np.float32 and prediction.shape==again.shape==(2048,14) and prediction.tobytes()==again.tobytes(),'Serialized replay mismatch')
        require(np.isfinite(prediction).all() and 0<=prediction.min()<=prediction.max()<=1,'Invalid probabilities')
        require(replay['ubj_and_json_replays_bitwise'] and replay['exact_replay_rows']==2048 and replay['output_count']==14,'Incomplete replay proof')
        require(top['separate_process_replay'] and top['all14_outputs_verified'] and top['actual_tree_count']==fit['trees']==8 and top['actual_vector_leaf_size']==fit['vector_leaf_size']==14,'Missing native model proof')
        config=json.loads((directory/'fit_config.json').read_text());native_config(config,approval['parameters'])
        replay_config=json.loads((directory/'replay_config.json').read_text())
        require(replay_config['learner']['generic_param']['device']=='cuda:1','Replay device differs')
        expanded=json.loads((directory/'expanded_parameters.json').read_text())
        for name,value in approval['parameters'].items():require(expanded[name]==value,'Expanded parameter differs: '+name)
        manual=json_forest_prediction(json.loads((directory/'model.json').read_text()),x)
        error=float(np.max(np.abs(manual-prediction)))
        require(error<=3e-7,'Independent JSON forest arithmetic differs')
        report.update(status='passed',archive_sha256=a.archive_sha256,hosted_manifest_sha256=a.manifest_sha256,
            fit_manifest_sha256=a.fit_manifest_sha256,replay_manifest_sha256=a.replay_manifest_sha256,
            preparation_manifest_sha256=sha(preparation),source_review_sha256=sha(review),native14_vector_leaves_verified=True,
            tree_count=8,synthetic_inputs_regenerated_bitwise=True,fresh_process_serialization_replay_bitwise=True,
            independent_json_prediction_max_error=error,
            caveat='Only bounded synthetic CUDA/serialization capability verified. No competition fold/score,600round model or full-size performance established. No newerUBJ loaded into local olderXGBoost; independent JSON arithmetic used.')
    except BaseException:
        report.update(status='failed',error=traceback.format_exc());raise
    finally:
        shutil.copy2(__file__,out/Path(__file__).name)
        report.update(runtime_seconds=time.monotonic()-start,source_sha256=sha(Path(__file__)))
        (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'status':report['status'],'manifest_sha256':sha(out/'manifest.json')}),flush=True)


if __name__=='__main__':main()
