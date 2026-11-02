"""Freeze the accepted original395 GPU final fit and its local export recipe."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shlex
import shutil
import sys
import time
import traceback
import zipfile

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.train_cuda_xgb_final import digest, final_sample, require
from scripts.export_cuda_xgb_final import verify_anchor

VALIDATION_SHA='7317e241c9a579531def94aec898b4cb3ae3192a5f5818062ee12b94db5b9b7c'
LOCK_SHA='315c420fa43a6a046fe97facafd473069de1d6761ad5d0f372617893bf3db040'
SAMPLE_SHA='4e38d45e87950d4d84849224ddb64fa9b5bf7b21d20a7d9b998223ce5e0fa83d'
SMOKE_SHA='f31b812df3cdea770b5f04861608d884faa8bbcaccebbe078a6e95c6736e7e4f'
RETURN_VERIFIER_SHA='4bae40e926be270a750b57eec39b35642315110c500780be6b427727a6bc8de4'


def main()->None:
    """Package only reviewed code and metadata; never fit, upload or submit."""
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',default='cuda-xgb-final-package-001')
    parser.add_argument('--protocol-output',type=Path,default=ROOT/'docs/cuda_xgb_final_protocol.json')
    args=parser.parse_args()
    out=ROOT/'artifacts'/args.run_id;out.mkdir(exist_ok=False)
    started=time.time()
    report={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'status':'initializing',
            'command':shlex.join([sys.executable,*sys.argv]),'training_or_upload':'none',
            'environment':{'python':sys.version,'platform':platform.platform(),
                           'packages':{k:importlib.metadata.version(k) for k in ['numpy','pandas']}},'kaggle':None}
    try:
        validation_path=ROOT/'docs/cuda_xgb_protocol.json'
        require(digest(validation_path)==VALIDATION_SHA,'Original validation protocol changed')
        validation=json.loads(validation_path.read_text())
        required_sources={'scripts/train_cuda_xgb.py':'trainer_sha256','scripts/train_cuda_tcn.py':'shared_loader_sha256',
                          'scripts/launch_cuda_xgb.py':'launcher_sha256','scripts/select_cuda_xgb_blend.py':'local_selector_sha256',
                          'sintef_project/postprocess.py':'postprocess_sha256'}
        for path,key in required_sources.items():require(digest(ROOT/path)==validation[key],'Accepted source changed: '+path)
        lock_path=ROOT/'artifacts/cuda-xgb-select-2021-001/selection_2021.json'
        require(digest(lock_path)==LOCK_SHA,'Selected2021 lock changed')
        lock=json.loads(lock_path.read_text())
        require(lock['training_protocol_sha256']==VALIDATION_SHA and lock['selected_weight']==.25
                and lock['anchor_run']==validation['anchor']['run_id'] and lock['candidate_weights']==validation['candidate_weights'],
                'Selection does not identify the accepted original395 recipe')
        require(.25==max(lock['candidate_weights'],key=lambda w:lock['scores_2021'][str(w)])
                and lock['selected_2021_micro_auc']==lock['scores_2021']['0.25'],'Selection is not fixed2021 argmax')
        acceptance_paths=['artifacts/cuda-xgb-select-2021-001/selection_2021.json']
        scores={};anchors={}
        for year in [2021,2022]:
            selected_path=ROOT/f'artifacts/cuda-xgb-select-{year}-001/manifest.json'
            proof_path=ROOT/f'artifacts/verify-cuda-xgb-{year}-return-001/manifest.json'
            selected=json.loads(selected_path.read_text());proof=json.loads(proof_path.read_text())
            require(selected['status']==proof['status']=='complete' and proof['verification']['models_verified']==14
                    and digest(selected_path)==proof['selection_manifest_sha256']
                    and selected['protocol_sha256']==VALIDATION_SHA and selected['selected_weight']==.25,
                    'Accepted fold verification changed')
            require(selected['selected_micro_auc']>selected['anchor_micro_auc'],'Selected weight did not improve both declared anchors')
            if year==2022:
                require(selected['accepted'] is True and selected['selection_lock_sha256']==LOCK_SHA
                        and set(selected['scores'])=={'0.0','0.25'},'Locked2022 acceptance changed')
            scores[str(year)]=selected['selected_micro_auc'];anchors[str(year)]=selected['anchor_micro_auc']
            acceptance_paths.extend([str(selected_path.relative_to(ROOT)),str(proof_path.relative_to(ROOT))])
        baseline_path=ROOT/'artifacts/lgb-normalized-lp-final-001/manifest.json'
        require(digest(baseline_path)=='b6c55a407649ffee4729f49b7217d0e1233abb165da40c1fc71f6cedb8e995fd','Baseline full fit changed')
        baseline=json.loads(baseline_path.read_text());config=baseline['arguments']
        require(baseline['status']=='complete' and config['years']==[] and config['final'] is True
                and config['sample_fraction']==.4 and config['seed']==4173 and baseline['features']==validation['feature_columns'],
                'Full sample/schema reference is not the original standalone final fit')
        train_path=ROOT/'sintef_project/data/kernel/Unit_commitment_decisions.csv'
        mapping_path=ROOT/'sintef_project/prediction_mapping.csv'
        for path in [train_path,mapping_path]:require(digest(path)==baseline['data'][str(path.relative_to(ROOT))],'Train/test metadata changed')
        train=pd.read_csv(train_path,usecols=['Run No','starttime']);mapping=pd.read_csv(mapping_path)
        require(len(train)==2922 and len(mapping)==725 and train['Run No'].is_unique and mapping['Run No'].is_unique,
                'Unexpected full fit case scope')
        sample_path=baseline_path.parent/'train_sample_final.npy'
        require(digest(sample_path)==baseline['artifacts']['train_sample_final.npy']==SAMPLE_SHA,'Saved full sample changed')
        sample=final_sample(len(train)*168,SAMPLE_SHA,np)
        np.testing.assert_array_equal(sample,np.load(sample_path,allow_pickle=False))
        smoke_path=ROOT/'artifacts/verify-cuda-xgb-2021-return-001/download/cuda-xgb-smoke-001/manifest.json'
        require(digest(smoke_path)==SMOKE_SHA,'Successful original smoke changed')
        helper_path=ROOT/'artifacts/xgb_return_preflight_20261102T144214Z_d148bf91/verify_cuda_xgb_return.py'
        require(digest(helper_path)==RETURN_VERIFIER_SHA,'Independent return verifier changed')
        anchor_path=ROOT/'artifacts/submission-lp-joint-blend-001/manifest.json'
        anchor_array=anchor_path.parent/'test_predictions.npy'
        anchor_proof=ROOT/'artifacts/verify-lp-joint-export-002/manifest.json'
        require(digest(anchor_path)=='10b1d42ff56f464b4f3993c1f80a574d447a824e299305d6a41a805fdf6b7750'
                and digest(anchor_array)=='f8ee08547ff5c302257dfc6e27b1054fa8ab9d978873650dc2e28247ddf0163b',
                'Saved projected joint anchor changed')
        protocol={'version':'cuda_xgb395_final_v1','created_utc':report['timestamp_utc'],
            'trainer_sha256':digest(ROOT/'scripts/train_cuda_xgb_final.py'),
            'exporter_sha256':digest(ROOT/'scripts/export_cuda_xgb_final.py'),
            'validation_protocol_sha256':VALIDATION_SHA,'postprocess_sha256':validation['postprocess_sha256'],
            'return_verifier_sha256':RETURN_VERIFIER_SHA,'model_parameters':validation['model_parameters'],
            'units':validation['units'],'training_label_year':int(pd.to_datetime(train.starttime).dt.year.max()),
            'train_case_ids':train['Run No'].tolist(),'test_case_ids':mapping['Run No'].tolist(),
            'train_sample_sha256':SAMPLE_SHA,'sampled_training_rows':len(sample),
            'sampling':'First default_rng(4173).choice over all2922*168 training rows, size=int(n*.4), sorted; no validation-fold RNG calls; exact NPY SHA required.',
            'baseline_final_manifest_sha256':digest(baseline_path),'smoke_manifest_sha256':SMOKE_SHA,
            'accepted_validation':{'selected_weight':.25,'accepted':True,'selected_scores':scores,'anchor_scores':anchors,
                                   'selection_lock_sha256':LOCK_SHA,'validation_protocol_sha256':VALIDATION_SHA,
                                   'evidence':'Reused chronological development years, six-day purge; no leaderboard claim.'},
            'local_acceptance_inputs':{p:digest(ROOT/p) for p in acceptance_paths},
            'local_anchor':{'run_path':str(anchor_path.parent.relative_to(ROOT)),'manifest_sha256':digest(anchor_path),
                            'predictions_sha256':digest(anchor_array),'verification_path':str(anchor_proof.relative_to(ROOT)),
                            'verification_sha256':digest(anchor_proof)},
            'export_formula':'project_vinje(.75*saved_projected_joint_anchor+.25*rawGPU),float64; intermediate projection must not be flattened.',
            'checkpoint_replay':'Reload all14 savedUBJ files in same hosted CUDA environment; all121800 testrows must be bitwise equal.',
            'full_fit_scope':'All provided2015-2022 training cases; no chronological purge in final fit; test inputs only.',
            'transfer_gate_passed':False,'kaggle_submission':None}
        anchor=verify_anchor(protocol)
        require(anchor.shape==(725*168,14),'Frozen final anchor shape changed')
        payload=out/'payload';payload.mkdir()
        for name in ['train_cuda_xgb_final.py','train_cuda_xgb.py','train_cuda_tcn.py','launch_cuda_xgb.py']:
            shutil.copy2(ROOT/'scripts'/name,payload/name)
        shutil.copy2(validation_path,payload/'cuda_xgb_protocol.json')
        encoded=json.dumps(protocol,indent=2,allow_nan=False)
        (payload/'cuda_xgb_final_protocol.json').write_text(encoded)
        with args.protocol_output.open('x') as stream:stream.write(encoded)
        (payload/'README.md').write_text('This private code package fits the accepted original395 GPU branch only. The original sealed private input bundle and successful original GPU smoke are reused. No anchor predictions are included. Run only the reviewed explicit final command through the offline uv launcher; download all14 checkpoints, raw probabilities, replay array, sources, manifests and log. Local export separately verifies these before applying the nested fixed25% blend. No main-repository transfer is permitted; the .998 validation gate remains unmet.\n')
        entries=[{'path':p.name,'size_bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(payload.iterdir())]
        (payload/'package_manifest.json').write_text(json.dumps({'files':entries,'private_only':True,
            'purpose':'Accepted395 final-fit code/metadata only','raw_data_or_anchor_arrays_included':False},indent=2))
        archive_path=out/'cuda-xgb-final-code.zip'
        with zipfile.ZipFile(archive_path,'x',compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(payload.iterdir()):archive.write(path,path.name)
        with zipfile.ZipFile(archive_path) as archive:
            require(set(archive.namelist())=={e['path'] for e in entries}|{'package_manifest.json'},'Code archive schema changed')
            for entry in entries:
                blob=archive.read(entry['path'])
                require(len(blob)==entry['size_bytes'] and hashlib.sha256(blob).hexdigest()==entry['sha256'],'Code archive readback changed')
        sources={}
        paths=['scripts/train_cuda_xgb_final.py','scripts/export_cuda_xgb_final.py','scripts/build_cuda_xgb_final_package.py',
               'tests/test_cuda_xgb_final.py',*required_sources]
        for relative in paths:
            target=out/'source'/relative;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/relative,target);sources[relative]=digest(target)
        report.update(status='complete',source=sources,protocol_sha256=digest(payload/'cuda_xgb_final_protocol.json'),
                      archive_sha256=digest(archive_path),archive_size_bytes=archive_path.stat().st_size,
                      package_manifest_sha256=digest(payload/'package_manifest.json'),payload_files=entries,
                      local_anchor_reconstruction_bitwise=True,full_sample_sha256=SAMPLE_SHA,
                      accepted_validation=protocol['accepted_validation'],protocol_output=str(args.protocol_output))
        print(json.dumps({k:report[k] for k in ['status','protocol_sha256','archive_sha256','archive_size_bytes']}),flush=True)
    except BaseException:
        report['status']='failed';report['error']=traceback.format_exc();raise
    finally:
        report['runtime_seconds']=time.time()-started
        (out/'manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        with (ROOT/'docs/experiments.md').open('a') as stream:
            stream.write(f'\n## {args.run_id} — {report["timestamp_utc"]}\n\nStatus: {report["status"]}. Command: `{report["command"]}`. Prepare accepted original395 GPU full fit and exact nested local export only; no model fit, upload or submission. Exact baseline full sample SHA `{SAMPLE_SHA}`, first seed4173 40% draw over all training cases. Both purged development folds accepted at frozen25% GPU weight. All anchor source checkpoints/raw arrays and replayed arrays rechecked; projected anchor reconstructed bitwise. Code/protocol ZIP readback checks and all provenance: `artifacts/{args.run_id}/manifest.json`. Main repository transfer gate remains unmet.\n')


if __name__=='__main__':main()
