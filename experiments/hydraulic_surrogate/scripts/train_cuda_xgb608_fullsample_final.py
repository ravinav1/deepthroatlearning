"""Fit the accepted600-tree GPU1 fullsample recipe on the all final training rows."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import sys
import time
import traceback
import warnings

for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[name] = '2'


def digest(path: Path) -> str:
    """Hash a file in bounded memory."""
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def require(condition: bool, message: str) -> None:
    """Reject any recipe or artifact mismatch."""
    if not condition:
        raise ValueError(message)


def load_pinned(path: Path, expected: str, name: str):
    """Import an exact reviewed helper without modifying its source."""
    require(digest(path) == expected, f'Pinned helper changed: {path.name}')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def final_sample(training_rows: int, expected_sha: str, np):
    """Use every canonical training row in original order, with no random subsampling."""
    sample = np.arange(training_rows, dtype=np.int64)
    buffer = io.BytesIO()
    np.save(buffer, sample)
    require(hashlib.sha256(buffer.getvalue()).hexdigest() == expected_sha, 'Full-training all-row array differs from frozen exact bytes')
    return sample


def verify_fresh_smoke(path: Path, expected_sha: str, validation: dict, protocol: dict, code: Path) -> dict:
    """Require a genuine fresh synthetic smoke and the accepted validation runtime."""
    require(digest(path) == expected_sha, 'Fresh smoke differs from explicit execution hash')
    smoke = json.loads(path.read_text())
    require(smoke['status'] == 'complete' and smoke['mode'] == 'smoke'
            and smoke['protocol_sha256'] == protocol['validation_protocol_sha256']
            and smoke['trainer_sha256'] == validation['trainer_sha256']
            and smoke['shared_loader_sha256'] == validation['shared_loader_sha256'], 'Fresh smoke source/protocol changed')
    require(smoke['smoke']['synthetic_only'] is True and smoke['smoke']['shape'] == [2048,608]
            and smoke['smoke']['exact_replay_rows'] == 2048 and smoke['smoke']['rounds'] == 8,
            'Fresh full synthetic smoke and checkpoint replay required')
    require(smoke['cuda'] == protocol['reference_cuda'] and smoke['cuda']['requested_device'] == 'cuda:1'
            and smoke['cuda']['cpu_fallback'] is False and smoke['cuda']['xgboost_build_info']['USE_CUDA'] is True,
            'Fresh smoke runtime differs from accepted validation')
    expected = {'train_cuda_xgb608_fullsample.py': validation['trainer_sha256'],
                'shared_loader.py': validation['shared_loader_sha256'], 'core_loader.py': validation['core_loader_sha256'],
                'protocol.json': protocol['validation_protocol_sha256'],
                'synthetic_smoke.ubj': smoke['smoke']['checkpoint_sha256']}
    require(smoke['artifacts'] == expected and {p.name for p in path.parent.iterdir()} == set(expected) | {'manifest.json'},
            'Fresh smoke file allowlist changed')
    for name, expected_hash in expected.items():
        member = path.parent / name
        require(member.is_file() and not member.is_symlink() and digest(member) == expected_hash,
                'Fresh smoke artifact changed: ' + name)
    helper = load_pinned(code / 'verify_cuda_xgb608_fullsample_return.py', validation['return_verifier_sha256'], 'fullsample_return_helper')
    helper.verify_native_config(smoke['smoke']['native_config'], {**validation['model_parameters'], 'n_estimators':8, 'max_depth':3})
    return smoke


def main() -> None:
    """Train all14 final models, preserving every checkpoint and full-test replay."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--validation-protocol', type=Path, required=True)
    parser.add_argument('--validation-runner', type=Path, required=True)
    parser.add_argument('--shared-loader', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--supplement', type=Path, required=True)
    parser.add_argument('--smoke-manifest', type=Path, required=True)
    parser.add_argument('--smoke-manifest-sha256', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--stage-root', type=Path, default=Path('/kaggle/working/tdt4173-xgb608-fullsample-final-inputs'))
    parser.add_argument('--output-root', type=Path, default=Path('/kaggle/working/xgb608-fullsample-final-runs'))
    args = parser.parse_args()
    out = args.output_root/args.run_id
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    report = {'run_id':args.run_id, 'timestamp_utc':datetime.now(timezone.utc).isoformat(),
              'command':shlex.join([sys.executable,*sys.argv]), 'mode':'final', 'status':'initializing',
              'seed':4173, 'early_stopping':None, 'environment':{'python':sys.version,'platform':platform.platform()},
              'kaggle_submission':None}
    try:
        protocol = json.loads(args.protocol.read_text())
        require(protocol['version']=='cuda_xgb608_fullsample_final_v1' and digest(Path(__file__))==protocol['trainer_sha256'], 'Final trainer/protocol changed')
        require(digest(args.validation_protocol)==protocol['validation_protocol_sha256'], 'Validation protocol changed')
        validation = json.loads(args.validation_protocol.read_text())
        base = load_pinned(args.validation_runner, validation['trainer_sha256'], 'frozen_608_validation_runner')
        loader = load_pinned(args.shared_loader, validation['shared_loader_sha256'], 'frozen_608_feature_loader')
        require(protocol['model_parameters']==validation['model_parameters']==base.PARAMETERS,
                'Final hyperparameters differ from accepted validation recipe')
        accepted = protocol['accepted_validation']
        require(accepted['selected_weight']==.25 and accepted['accepted'] is True
                and all(accepted['selected_scores'][str(y)] > max(accepted['anchor_scores'][str(y)], accepted['control_scores'][str(y)]) for y in [2021,2022]),
                'Strict both-year accepted selection is not recorded')
        smoke = verify_fresh_smoke(args.smoke_manifest, args.smoke_manifest_sha256, validation, protocol, args.shared_loader.parent)
        import numpy as np
        import pandas as pd
        import xgboost as xgb
        report['environment']['packages']={p:importlib.metadata.version(p) for p in ['numpy','pandas','xgboost','scipy','scikit-learn','pyyaml']}
        report['cuda']=base.runtime_environment(xgb)
        require(report['cuda']==smoke['cuda']==protocol['reference_cuda'], 'GPU software/device environment differs from accepted validation')
        report.update(protocol_sha256=digest(args.protocol),trainer_sha256=digest(Path(__file__)),
                      validation_protocol_sha256=digest(args.validation_protocol),validation_runner_sha256=digest(args.validation_runner),
                      shared_loader_sha256=digest(args.shared_loader),core_loader_sha256=digest(args.shared_loader.with_name('train_cuda_tcn.py')),smoke_manifest_sha256=digest(args.smoke_manifest),
                      return_verifier_sha256=validation['return_verifier_sha256'],
                      model_parameters_requested=base.PARAMETERS,accepted_validation=accepted)
        report['bundle'],report['supplement'],feature_stage=loader.stage_inputs(args.bundle,args.supplement,args.stage_root,validation)
        x,labels,train,mapping,meta,baseline=loader.build_verified_features(feature_stage,protocol['training_label_year'],validation)
        require(train['Run No'].tolist()==protocol['train_case_ids'] and mapping['Run No'].tolist()==protocol['test_case_ids'],
                'Full train/test metadata order changed')
        dates=pd.to_datetime(train.starttime,utc=True)
        require(int(dates.dt.year.max())==protocol['training_label_year'], 'Full label prefix does not include every training case')
        require(meta['units']==validation['units'] and x.columns.tolist()==validation['feature_columns'], 'Feature/unit schema changed')
        ntrain=len(train)*168
        sample=final_sample(ntrain,protocol['train_sample_sha256'],np)
        np.save(out/'train_sample_final.npy',sample)
        y=np.column_stack([labels[u] for u in validation['units']]).astype(np.uint8)
        del labels
        require(y.shape==(ntrain,14) and np.isin(y,[0,1]).all(), 'Invalid full training target matrix')
        xtrain,xtest=x.iloc[sample],x.iloc[ntrain:]
        del x
        matrix=xgb.DMatrix(xtest,feature_names=validation['feature_columns'],nthread=2)
        pred=np.empty((len(mapping)*168,14),dtype=np.float32)
        replay=np.empty_like(pred)
        report.update(feature_spec=meta['feature_spec'],features=validation['feature_columns'],units=validation['units'],
                      train_case_ids=protocol['train_case_ids'],test_case_ids=protocol['test_case_ids'],
                      training_rows=ntrain,sampled_training_rows=len(sample),sample_sha256=digest(out/'train_sample_final.npy'),
                      training_dates=[str(dates.min()),str(dates.max())],models={})
        for j,unit in enumerate(validation['units']):
            tick=time.perf_counter()
            model=xgb.XGBClassifier(**base.PARAMETERS)
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter('always')
                model.fit(xtrain,y[sample,j])
                booster=model.get_booster()
                config=json.loads(booster.save_config())
                base.require_gpu_config(config)
                require(booster.num_boosted_rounds()==600 and booster.feature_names==validation['feature_columns'], 'Final boosted rounds/schema changed')
                pred[:,j]=booster.predict(matrix)
                checkpoint=out/f'final_{unit}.ubj'
                model.save_model(checkpoint)
                restored=xgb.Booster()
                restored.load_model(checkpoint)
                restored.set_param({'device':'cuda:1','nthread':2,'tree_method':'hist'})
                replay[:,j]=restored.predict(matrix)
                np.testing.assert_array_equal(replay[:,j],pred[:,j])
                base.require_gpu_config(json.loads(restored.save_config()))
            require(np.isfinite(pred[:,j]).all() and pred[:,j].min()>=0 and pred[:,j].max()<=1, 'Invalid final probabilities')
            config_path=out/f'config_final_{unit}.json'
            config_path.write_text(json.dumps(config,indent=2))
            report['models'][unit]={'checkpoint':checkpoint.name,'checkpoint_sha256':digest(checkpoint),
                'native_config_sha256':digest(config_path),'expanded_parameters':base.clean_json(model.get_params()),
                'checkpoint_replay_max_error':0.,'replay_rows':len(mapping)*168,
                'runtime_seconds':time.perf_counter()-tick,'warnings':[str(w.message) for w in captured]}
            (out/'manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False))
            print('FINAL',unit,flush=True)
            del model,booster,restored
        require(pred.tobytes()==replay.tobytes(),'Final full-test replay bytes differ')
        np.savez_compressed(out/'test_predictions.npz',predictions=pred,case_ids=mapping['Run No'].to_numpy(),units=np.asarray(validation['units'],dtype=str))
        np.save(out/'replay_test_predictions.npy',replay)
        report.update(status='complete',completed_units=14,completed_rounds_per_unit=600,
                      test_predictions_sha256=digest(out/'test_predictions.npz'),replay_sha256=digest(out/'replay_test_predictions.npy'),
                      checkpoint_replay_bitwise=True,postprocessing='none; local nested fixed-anchor blend only')
    except BaseException:
        report['status']='failed';report['error']=traceback.format_exc();raise
    finally:
        for source,name in [(Path(__file__),'train_cuda_xgb608_fullsample_final.py'),(args.protocol,'protocol.json'),
                            (args.validation_protocol,'validation_protocol.json'),(args.validation_runner,'validation_runner.py'),
                            (args.shared_loader,'shared_loader.py'),
                            (args.shared_loader.with_name('train_cuda_tcn.py'),'core_loader.py'),
                            (args.shared_loader.with_name('verify_cuda_xgb608_fullsample_return.py'),'return_verifier.py')]:
            if source.is_file():shutil.copy2(source,out/name)
        report['runtime_seconds']=time.time()-started
        report['artifacts']={p.relative_to(out).as_posix():digest(p) for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'}
        (out/'manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        print(json.dumps({k:report.get(k) for k in ['status','completed_units','runtime_seconds']}),flush=True)


if __name__=='__main__':main()
