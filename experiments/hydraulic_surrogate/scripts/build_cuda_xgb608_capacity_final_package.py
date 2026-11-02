"""Seal capacity final code only after externally pinned independent2022 acceptance."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shlex
import shutil
import sys
import time
import traceback

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.train_cuda_xgb608_capacity_final import digest,final_sample,require
from scripts.export_cuda_xgb608_capacity_final import verify_anchor
from scripts.build_cuda_xgb608_package import zip_verified

VALIDATION_SHA='e9172028088440d8a46a5966946c63cb723606ea916833d3c81e5f67ccfae026'
LOCK_SHA='f418296446f832b408ccc2e7ca3aee6825408e11420767252f92785f193665be'
SAMPLE_SHA='4e38d45e87950d4d84849224ddb64fa9b5bf7b21d20a7d9b998223ce5e0fa83d'
SMOKE_SHA='8cda83c2e0753d8e6c796bca1698b96f4074ece445e3354eb6d659666160805e'
CODE=['train_cuda_xgb608_capacity_final.py','train_cuda_xgb608_capacity.py','verify_cuda_xgb608_capacity_return.py',
      'cuda_xgb608_loader.py','train_cuda_tcn.py','launch_cuda_xgb.py','launch_cuda_xgb608_capacity_final_dataset.py']
SOURCE=[*[f'scripts/{name}'for name in CODE], 'scripts/export_cuda_xgb608_capacity_final.py',
        'scripts/build_cuda_xgb608_capacity_final_package.py','tests/test_cuda_xgb608_capacity_final.py',
        'sintef_project/postprocess.py']


def main()->None:
    """Check acceptance, exact full sample and nested anchor before preparing any final protocol."""
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',default='cuda-xgb608-capacity-final-package-001')
    parser.add_argument('--selection-manifest-sha256',required=True)
    parser.add_argument('--return-manifest-sha256',required=True)
    parser.add_argument('--acceptance-audit',type=Path,required=True)
    parser.add_argument('--acceptance-audit-sha256',required=True)
    args=parser.parse_args()
    require(Path(args.run_id).name==args.run_id and args.run_id not in ['', '.', '..'],'Invalid output ID')
    out=ROOT/'artifacts'/args.run_id;out.mkdir(exist_ok=False);started=time.monotonic()
    report={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'status':'initializing','command':shlex.join([sys.executable,*sys.argv]),
            'training':False,'upload':False,'kaggle':None,'environment':{'python':sys.version,'platform':platform.platform()}}
    try:
        val_path=ROOT/'docs/cuda_xgb608_capacity_protocol.json';require(digest(val_path)==VALIDATION_SHA,'Validation protocol changed')
        val=json.loads(val_path.read_text())
        for name,key in [('train_cuda_xgb608_capacity.py','trainer_sha256'),('verify_cuda_xgb608_capacity_return.py','return_verifier_sha256'),
                         ('cuda_xgb608_loader.py','shared_loader_sha256'),('train_cuda_tcn.py','core_loader_sha256'),
                         ('launch_cuda_xgb.py','launcher_sha256'),('select_cuda_xgb608_capacity_blend.py','local_selector_sha256')]:
            require(digest(ROOT/'scripts'/name)==val[key], 'Accepted source changed: '+name)
        require(digest(args.acceptance_audit)==args.acceptance_audit_sha256,'Independent acceptance audit changed')
        audit=json.loads(args.acceptance_audit.read_text())
        require(audit['status']=='passed'and audit['accepted']and audit['selection_lock_sha256']==LOCK_SHA
                and audit['selection_manifest_sha256']==args.selection_manifest_sha256,'Independent2022 acceptance absent')
        lock_path=ROOT/'artifacts/cuda-xgb608-capacity-select-2021-001/selection_2021.json'
        require(digest(lock_path)==LOCK_SHA,'2021 lock changed');lock=json.loads(lock_path.read_text())
        require(lock['selected_weight']==.25 and lock['eligible_for_2022'],'Wrong accepted weight')
        inputs={str(lock_path.relative_to(ROOT)):LOCK_SHA,str(args.acceptance_audit.resolve().relative_to(ROOT)):args.acceptance_audit_sha256}
        scores={};anchors={};controls={}
        for year in [2021,2022]:
            selected_path=ROOT/f'artifacts/cuda-xgb608-capacity-select-{year}-001/manifest.json'
            verified_path=ROOT/f'artifacts/verify-cuda-xgb608-capacity-{year}-return-001/manifest.json'
            selected=json.loads(selected_path.read_text());verified=json.loads(verified_path.read_text())
            require(selected['status']==verified['status']=='complete'and digest(selected_path)==verified['selection_manifest_sha256']
                    and selected['protocol_sha256']==VALIDATION_SHA and selected['selected_weight']==.25,'Fold acceptance closure changed')
            require(selected['selected_micro_auc']>max(selected['anchor_micro_auc'],selected['control_micro_auc']),'Candidate failed anchor/control gate')
            if year==2022:
                require(digest(selected_path)==args.selection_manifest_sha256 and digest(verified_path)==args.return_manifest_sha256
                        and selected['accepted']and selected['selection_lock_sha256']==LOCK_SHA and set(selected['scores'])=={'0.0','0.25'},'Locked2022 acceptance changed')
            scores[str(year)]=selected['selected_micro_auc'];anchors[str(year)]=selected['anchor_micro_auc'];controls[str(year)]=selected['control_micro_auc']
            inputs[str(selected_path.relative_to(ROOT))]=digest(selected_path);inputs[str(verified_path.relative_to(ROOT))]=digest(verified_path)
        baseline_path=ROOT/'artifacts/lgb-normalized-lp-context-joint-final-001/manifest.json'
        require(digest(baseline_path)=='bf14d938dc63270d2b5c1d1b47309dda08540176844b68e3fb049ee80376393f','Original full-fit baseline changed')
        baseline=json.loads(baseline_path.read_text())
        train=pd.read_csv(ROOT/'sintef_project/data/kernel/Unit_commitment_decisions.csv',usecols=['Run No','starttime'])
        mapping=pd.read_csv(ROOT/'sintef_project/prediction_mapping.csv')
        require(len(train)==2922 and len(mapping)==725,'Unexpected full train/test scope')
        for name,expected in baseline['data'].items():require(digest(ROOT/name)==expected,'Original input data changed')
        sample=final_sample(len(train)*168,SAMPLE_SHA,np)
        np.testing.assert_array_equal(sample,np.load(baseline_path.parent/'train_sample_final.npy',allow_pickle=False))
        smoke_path=ROOT/'artifacts/verify-cuda-xgb608-capacity-2022-return-001/download/cuda-xgb608-capacity-smoke-002/manifest.json'
        require(digest(smoke_path)==SMOKE_SHA,'Externally observed successful2022 smoke changed')
        smoke=json.loads(smoke_path.read_text());require(smoke['status']=='complete'and smoke['protocol_sha256']==VALIDATION_SHA,'Bad reference smoke')
        inputs[str(smoke_path.relative_to(ROOT))]=SMOKE_SHA
        anchor_dir=ROOT/'artifacts/submission-cuda-xgb608-context-joint-blend-001'
        require(digest(anchor_dir/'manifest.json')=='6d13425abe03f5aa7ffe36baffd9b6c06e197f28a4b1afec90c79a8313053555','Accepted primary anchor changed')
        proof_path=ROOT/'artifacts/cuda-xgb608-submissions-independent-audit-001/manifest.json'
        require(digest(proof_path)=='4d8bce81e8748dd274968a0a53bf6d5fc274e60b056451369eefd3c41210f19d','Primary independent proof changed')
        original_protocol=ROOT/'docs/cuda_xgb608_final_protocol.json';original_exporter=ROOT/'scripts/export_cuda_xgb608_final.py'
        require(digest(original_protocol)=='a5939edb8ec7f46dbdb4c6edd0c06909ea09ed0ac029709b06945a5c6434683b','Original final protocol changed')
        require(digest(original_exporter)==json.loads(original_protocol.read_text())['exporter_sha256'],'Original exporter changed')
        local_anchor={'run_path':str(anchor_dir.relative_to(ROOT)),'manifest_sha256':digest(anchor_dir/'manifest.json'),
            'predictions_sha256':digest(anchor_dir/'test_predictions.npy'),'verification_path':str(proof_path.relative_to(ROOT)),
            'verification_sha256':digest(proof_path),'original_final_protocol_path':str(original_protocol.relative_to(ROOT)),
            'original_final_protocol_sha256':digest(original_protocol),'original_exporter_path':str(original_exporter.relative_to(ROOT)),
            'original_exporter_sha256':digest(original_exporter),
            'primary_raw_path':str((anchor_dir/'download/cuda-xgb608-final-003/test_predictions.npz').relative_to(ROOT)),
            'primary_replay_path':str((anchor_dir/'download/cuda-xgb608-final-003/replay_test_predictions.npy').relative_to(ROOT))}
        protocol={'version':'cuda_xgb608_capacity_final_v1','created_utc':report['timestamp_utc'],
            'trainer_sha256':digest(ROOT/'scripts/train_cuda_xgb608_capacity_final.py'),
            'exporter_sha256':digest(ROOT/'scripts/export_cuda_xgb608_capacity_final.py'),
            'dataset_launcher_sha256':digest(ROOT/'scripts/launch_cuda_xgb608_capacity_final_dataset.py'),
            'validation_protocol_sha256':VALIDATION_SHA,'return_verifier_sha256':val['return_verifier_sha256'],
            'postprocess_sha256':val['postprocess_sha256'],'model_parameters':val['model_parameters'],'units':val['units'],
            'training_label_year':2022,'train_case_ids':train['Run No'].tolist(),'test_case_ids':mapping['Run No'].tolist(),
            'train_sample_sha256':SAMPLE_SHA,'sampled_training_rows':len(sample),'baseline_final_manifest_sha256':digest(baseline_path),
            'sampling':'Original first seed4173 sorted40% draw over all2922*168 training rows; no validation RNG calls.',
            'reference_smoke_manifest_sha256':SMOKE_SHA,'reference_cuda':smoke['cuda'],
            'smoke_policy':'Execute fresh genuine matching capacity smoke; explicit manifest SHA, exact source/artifacts/nativeconfig/runtime required. No historical smoke restoration.',
            'accepted_validation':{'selected_weight':.25,'accepted':True,'selected_scores':scores,'anchor_scores':anchors,'control_scores':controls,
                'selection_lock_sha256':LOCK_SHA,'validation_protocol_sha256':VALIDATION_SHA,'evidence':'Reused purged chronological development folds; no new public score.'},
            'local_acceptance_inputs':inputs,'local_anchor':local_anchor,
            'export_formula':'project_vinje(.75*saved_projected_acceptedGPU608anchor+.25*rawGPUcapacity),float64',
            'checkpoint_replay':'All14savedUBJ models, full121800testrows, sameCUDAenvironment, bitwise replay.',
            'transfer_gate_passed':min(scores.values())>=.998,'kaggle_submission':None}
        require(verify_anchor(protocol).shape==(725*168,14),'Accepted nested anchor shape differs')
        protocol_path=ROOT/'docs/cuda_xgb608_capacity_final_protocol.json'
        with protocol_path.open('x')as stream:json.dump(protocol,stream,indent=2)
        payload=out/'dataset';code=payload/'code';code.mkdir(parents=True)
        for name in CODE:shutil.copy2(ROOT/'scripts'/name,code/name)
        shutil.copy2(val_path,code/val_path.name);shutil.copy2(protocol_path,code/protocol_path.name)
        entries=[{'path':p.name,'size_bytes':p.stat().st_size,'sha256':digest(p)}for p in sorted(code.iterdir())]
        (code/'package_manifest.json').write_text(json.dumps({'files':entries,'visibility':'PRIVATE','raw_data_or_anchor_arrays_included':False},indent=2))
        archive=out/'cuda-xgb608-capacity-final-private-dataset.zip';zip_verified(payload,archive)
        report.update(status='complete',protocol_sha256=digest(protocol_path),package_manifest_sha256=digest(code/'package_manifest.json'),
                      dataset_archive_sha256=digest(archive),dataset_archive_bytes=archive.stat().st_size,
                      accepted_validation=protocol['accepted_validation'],anchor_reconstruction_bitwise=True,
                      source={name:digest(ROOT/name)for name in SOURCE})
        print(json.dumps({k:report[k]for k in ['status','protocol_sha256','package_manifest_sha256','dataset_archive_sha256','dataset_archive_bytes']}),flush=True)
    except BaseException:
        report.update(status='failed',error=traceback.format_exc());raise
    finally:
        for name in SOURCE:
            if (ROOT/name).is_file():
                dest=out/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
        report['runtime_seconds']=time.monotonic()-started
        report['artifacts']={p.relative_to(out).as_posix():digest(p)for p in out.rglob('*')if p.is_file()and p.name!='manifest.json'}
        (out/'manifest.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
