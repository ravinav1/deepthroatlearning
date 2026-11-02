"""Verify final GPU artifacts and export the exact accepted nested probability blend."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import sys
import time
import traceback

import numpy as np
import pandas as pd

ROOT=Path(os.environ.get('TDT4173_WORKSPACE_ROOT',Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0,str(ROOT))
from scripts.train_cuda_xgb608_capacity_final import digest,require,verify_fresh_smoke
from sintef_project.postprocess import project_vinje


def compose(anchor:np.ndarray,raw:np.ndarray,units:list[str])->np.ndarray:
    """Preserve the selected intermediate anchor projection before final blending."""
    require(anchor.shape==raw.shape and np.isfinite(raw).all() and raw.min()>=0 and raw.max()<=1
            and np.array_equal(project_vinje(anchor,units),anchor), 'Expected valid raw probabilities and already projected anchor')
    return project_vinje(.75*anchor.astype(np.float64)+.25*raw.astype(np.float64),units)


def submission_frame(pred:np.ndarray,ids:np.ndarray,units:list[str],template:pd.DataFrame)->pd.DataFrame:
    """Map case/hour/unit probabilities into the exact template ID/column order."""
    columns=['Run No']+[f'result_committed_{u}_t{h}' for u in units for h in range(168)]
    require(template.columns.tolist()==columns and len(set(ids.tolist()))==len(ids)
            and set(ids.tolist())==set(template['Run No'].tolist()), 'Submission ID/column schema differs')
    require(pred.shape==(len(ids)*168,len(units)) and np.isfinite(pred).all()
            and pred.min()>=0 and pred.max()<=1,'Invalid submission probabilities')
    wide=pred.reshape(len(ids),168,len(units)).transpose(0,2,1).reshape(len(ids),-1)
    frame=pd.DataFrame(wide,index=pd.Index(ids,name='Run No'),columns=columns[1:])
    return frame.loc[template['Run No']].reset_index()


def verify_anchor(protocol:dict)->np.ndarray:
    """Reconstruct the accepted public0.99640 nested anchor exactly from sealed sources."""
    info=protocol['local_anchor'];directory=ROOT/info['run_path']
    require(digest(directory/'manifest.json')==info['manifest_sha256'], 'Accepted primary export manifest changed')
    primary=json.loads((directory/'manifest.json').read_text())
    require(primary['status']=='complete' and primary['final_full_test_replay_bitwise']
            and primary['anchor_source_reconstruction_bitwise'], 'Accepted primary export was not verified')
    for name,expected in primary['artifacts'].items():
        require(digest(directory/name)==expected, 'Accepted primary artifact changed: '+name)
    proof_path=ROOT/info['verification_path']
    require(digest(proof_path)==info['verification_sha256'], 'Independent primary proof changed')
    proof=json.loads(proof_path.read_text())
    require(proof['status']=='passed' and proof['primary_manifest_sha256']==info['manifest_sha256'], 'Independent primary proof differs')
    original_protocol_path=ROOT/info['original_final_protocol_path']
    original_exporter_path=ROOT/info['original_exporter_path']
    require(digest(original_protocol_path)==info['original_final_protocol_sha256']
            and digest(original_exporter_path)==info['original_exporter_sha256'], 'Original nested anchor source changed')
    original_protocol=json.loads(original_protocol_path.read_text())
    spec=importlib.util.spec_from_file_location('original_primary_exporter',original_exporter_path)
    original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
    context=original.verify_anchor(original_protocol)
    with np.load(ROOT/info['primary_raw_path'],allow_pickle=False) as saved:
        raw=saved['predictions']
        require(saved['case_ids'].tolist()==protocol['test_case_ids'] and saved['units'].tolist()==protocol['units'], 'Primary raw schema differs')
    replay=np.load(ROOT/info['primary_replay_path'],allow_pickle=False)
    require(raw.dtype==replay.dtype==np.float32 and raw.tobytes()==replay.tobytes(), 'Primary raw checkpoint replay differs')
    anchor=np.load(directory/'test_predictions.npy',allow_pickle=False)
    require(digest(directory/'test_predictions.npy')==info['predictions_sha256'], 'Accepted anchor array changed')
    require(project_vinje(.5*context.astype(np.float64)+.5*raw.astype(np.float64),protocol['units']).tobytes()==anchor.tobytes(),
            'Accepted nested primary anchor reconstruction differs')
    return anchor


def main()->None:
    """Import a hash-pinned final download and emit an exact frozen-recipe CSV."""
    parser=argparse.ArgumentParser()
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--protocol-sha256',required=True)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--archive-sha256',required=True)
    parser.add_argument('--manifest-sha256',required=True)
    parser.add_argument('--gpu-run-name',required=True)
    parser.add_argument('--smoke-run-name',required=True)
    parser.add_argument('--smoke-manifest-sha256',required=True)
    parser.add_argument('--run-id',required=True)
    args=parser.parse_args()
    require('/' not in args.run_id and args.run_id not in ['', '.', '..'],'Invalid output run ID')
    out=ROOT/'artifacts'/args.run_id;out.mkdir(exist_ok=False)
    start=time.time()
    report={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'command':shlex.join([sys.executable,*sys.argv]),
            'status':'initializing','training':'none; final GPU artifact verification and fixed nested blend export','kaggle':None}
    try:
        require(digest(args.protocol)==args.protocol_sha256,'Final protocol differs from reviewed package')
        protocol=json.loads(args.protocol.read_text())
        require(digest(Path(__file__))==protocol['exporter_sha256'],'Local exporter differs from reviewed final protocol')
        require(digest(ROOT/'scripts/train_cuda_xgb608_capacity_final.py')==protocol['trainer_sha256'],'Imported local verification helpers changed')
        require(digest(ROOT/'sintef_project/postprocess.py')==protocol['postprocess_sha256'],'Projection source changed')
        validation_path=ROOT/'docs/cuda_xgb608_capacity_protocol.json'
        require(digest(validation_path)==protocol['validation_protocol_sha256'],'Original validation protocol changed')
        validation=json.loads(validation_path.read_text())
        for path,expected in protocol['local_acceptance_inputs'].items():require(digest(ROOT/path)==expected,'Local acceptance evidence changed')
        require(protocol['accepted_validation']['accepted'] is True and protocol['accepted_validation']['selected_weight']==.25,'Candidate not accepted')
        helper_path=ROOT/'scripts/verify_cuda_xgb608_capacity_return.py'
        require(digest(helper_path)==protocol['return_verifier_sha256'],'Reviewed transfer verifier changed')
        spec=importlib.util.spec_from_file_location('frozen_xgb_transfer',helper_path)
        helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        helper.import_archive(args.archive,args.archive_sha256,out/'download',args.gpu_run_name,args.smoke_run_name)
        directory=out/'download'/args.gpu_run_name
        require(digest(directory/'manifest.json')==args.manifest_sha256,'Final manifest differs from hosted hash')
        gpu=json.loads((directory/'manifest.json').read_text())
        require(gpu['status']=='complete' and gpu['mode']=='final' and gpu['completed_units']==14
                and gpu['completed_rounds_per_unit']==1200 and gpu['seed']==4173 and gpu['early_stopping'] is None,'Incomplete final fixed-recipe fit')
        require(gpu['protocol_sha256']==args.protocol_sha256 and gpu['model_parameters_requested']==protocol['model_parameters']
                and gpu['accepted_validation']==protocol['accepted_validation'],'Final recipe/acceptance changed')
        source_hashes={'train_cuda_xgb608_capacity_final.py':protocol['trainer_sha256'],'protocol.json':args.protocol_sha256,
                       'validation_protocol.json':protocol['validation_protocol_sha256'],
                       'validation_runner.py':validation['trainer_sha256'],'shared_loader.py':validation['shared_loader_sha256'],
                       'core_loader.py':validation['core_loader_sha256'],
                       'return_verifier.py':validation['return_verifier_sha256']}
        expected_names=set(source_hashes)|{'train_sample_final.npy','test_predictions.npz','replay_test_predictions.npy'}
        for unit in protocol['units']:expected_names.update([f'final_{unit}.ubj',f'config_final_{unit}.json'])
        files=list(directory.rglob('*'))
        require(not any(p.is_symlink() for p in files)
                and {p.relative_to(directory).as_posix() for p in files if p.is_file()}==expected_names|{'manifest.json'}
                and set(gpu['artifacts'])==expected_names,'Final artifact allowlist changed')
        for name,expected in gpu['artifacts'].items():require(digest(directory/name)==expected,'Final artifact changed: '+name)
        for name,expected in source_hashes.items():require(digest(directory/name)==expected,'Final source/protocol changed: '+name)
        smoke_path=out/'download'/args.smoke_run_name/'manifest.json'
        smoke=verify_fresh_smoke(smoke_path,args.smoke_manifest_sha256,validation,protocol,ROOT/'scripts')
        require(gpu['smoke_manifest_sha256']==args.smoke_manifest_sha256 and gpu['cuda']==smoke['cuda'], 'Final fresh smoke/runtime identity differs')
        require(gpu['train_case_ids']==protocol['train_case_ids'] and gpu['test_case_ids']==protocol['test_case_ids']
                and gpu['training_rows']==len(protocol['train_case_ids'])*168
                and gpu['sampled_training_rows']==protocol['sampled_training_rows'],'Final train/test scope changed')
        require(digest(directory/'train_sample_final.npy')==gpu['sample_sha256']==protocol['train_sample_sha256'],'Full training sample changed')
        require(gpu['features']==validation['feature_columns'] and gpu['units']==protocol['units'],'Final schema changed')
        baseline_path=ROOT/'artifacts/lgb-normalized-lp-context-joint-final-001/manifest.json'
        require(digest(baseline_path)==protocol['baseline_final_manifest_sha256'],'Baseline full-fit reference changed')
        baseline=json.loads(baseline_path.read_text())
        specs=[baseline['feature_spec'],gpu['feature_spec']]
        for item in specs:
            for key in ['lp','lp_joint']:item[key].pop('cache_directory',None)
        require(specs[0]==specs[1],'Final feature formulas/source/cache differ')
        for path,expected in baseline['data'].items():require(digest(ROOT/path)==expected,'Original input bytes changed')
        bundle_path=ROOT/'artifacts/kaggle-neural-bundle-001/payload/bundle_manifest.json'
        require(digest(bundle_path)==validation['bundle_manifest_sha256'] and gpu['bundle']==json.loads(bundle_path.read_text()),'Final staged bundle differs')
        supplement_path=ROOT/'artifacts/cuda-xgb608-package-001/supplement/bundle_manifest.json'
        require(digest(supplement_path)==validation['supplement_manifest_sha256'] and gpu['supplement']==json.loads(supplement_path.read_text()),'Final608 supplement differs')
        require(gpu['core_loader_sha256']==validation['core_loader_sha256'],'Final core loader changed')
        require(set(gpu['models'])==set(protocol['units']),'Incomplete final model records')
        for unit in protocol['units']:
            model=gpu['models'][unit];checkpoint=f'final_{unit}.ubj';config=f'config_final_{unit}.json'
            require(model['checkpoint']==checkpoint and model['checkpoint_sha256']==gpu['artifacts'][checkpoint]
                    and model['native_config_sha256']==gpu['artifacts'][config] and model['checkpoint_replay_max_error']==0.
                    and model['replay_rows']==len(protocol['test_case_ids'])*168,'Final checkpoint/replay proof differs')
            for key,value in protocol['model_parameters'].items():require(model['expanded_parameters'][key]==value,'Final expanded parameters differ')
            helper.verify_native_config(json.loads((directory/config).read_text()),protocol['model_parameters'])
        with np.load(directory/'test_predictions.npz',allow_pickle=False) as saved:
            require(set(saved.files)=={'predictions','case_ids','units'} and saved['case_ids'].tolist()==protocol['test_case_ids']
                    and saved['units'].tolist()==protocol['units'],'Final raw test ordering/schema differs')
            raw=saved['predictions'];ids=saved['case_ids']
        replay=np.load(directory/'replay_test_predictions.npy',allow_pickle=False)
        require(raw.dtype==replay.dtype==np.float32 and raw.shape==replay.shape==(725*168,14)
                and raw.tobytes()==replay.tobytes() and gpu['checkpoint_replay_bitwise'] is True,'Final full-test replay bytes differ')
        require(digest(directory/'test_predictions.npz')==gpu['test_predictions_sha256']
                and digest(directory/'replay_test_predictions.npy')==gpu['replay_sha256'],'Final array hashes disagree')
        require(np.isfinite(raw).all() and raw.min()>=0 and raw.max()<=1,'Invalid raw final probabilities')
        anchor=verify_anchor(protocol)
        pred=compose(anchor,raw,protocol['units'])
        template=pd.read_csv(ROOT/'sintef_project/sample_submission.csv')
        frame=submission_frame(pred,ids,protocol['units'],template)
        require(frame.shape==(725,2353),'Unexpected competition dimensions')
        np.save(out/'test_predictions.npy',pred)
        frame.to_csv(out/'submission.csv',index=False,float_format='%.9g')
        back=pd.read_csv(out/'submission.csv')
        require(back.columns.tolist()==template.columns.tolist() and back['Run No'].equals(template['Run No']),'CSV IDs/schema changed')
        error=float(np.max(np.abs(back.iloc[:,1:].to_numpy()-frame.iloc[:,1:].to_numpy())))
        require(error<=5.1e-10,'CSV serialization changed probabilities')
        report.update(status='complete',protocol_sha256=args.protocol_sha256,hosted_manifest_sha256=args.manifest_sha256,
                      archive_sha256=args.archive_sha256,formula='project_vinje(.75*saved_projected_acceptedGPU608anchor+.25*rawGPUcapacity),float64',
                      validation_scores=protocol['accepted_validation']['selected_scores'],submission_sha256=digest(out/'submission.csv'),
                      shape=list(frame.shape),probability_range=[float(pred.min()),float(pred.max())],csv_readback_max_error=error,
                      final_full_test_replay_bitwise=True,anchor_source_reconstruction_bitwise=True,
                      local_checkpoint_inference=False)
        print(json.dumps({k:report[k] for k in ['status','submission_sha256','validation_scores','csv_readback_max_error']}),flush=True)
    except BaseException:
        report['status']='failed';report['error']=traceback.format_exc();raise
    finally:
        shutil.copy2(__file__,out/'export_cuda_xgb608_capacity_final.py')
        report['runtime_seconds']=time.time()-start
        report['artifacts']={p.relative_to(out).as_posix():digest(p) for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'}
        (out/'manifest.json').write_text(json.dumps(report,indent=2))
        with (ROOT/'docs/experiments.md').open('a') as stream:
            stream.write(f'\n## {args.run_id} — {report["timestamp_utc"]}\n\nStatus:{report["status"]}. Command:`{report["command"]}`. Strict finalGPU artifact/sample/config/checkpoint/fulltestreplay verification and exact nested75%acceptedGPU608anchor/25%capacityGPU blend export. No training or weightretuning. SubmissionSHA:{report.get("submission_sha256")}. Full provenance:`artifacts/{args.run_id}/manifest.json`. NoKaggle result at export.\n')


if __name__=='__main__':main()
