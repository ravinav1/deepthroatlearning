"""Independently reconstruct both predeclared SHOP664 submission arrays and CSVs."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import platform
import shlex
import sys
import time
import traceback

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.verify_cuda_xgb664_return import verify_native_config


def digest(path:Path)->str:
    """Hash one immutable input or output in bounded memory."""
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):
            value.update(block)
    return value.hexdigest()


def independent_projection(values:np.ndarray,units:list[str])->np.ndarray:
    """Apply generic three-point PAVA with original block sums."""
    out=values.copy()
    indices=[units.index(f'Vinje_G{i}') for i in [1,2,3]]
    for row in np.flatnonzero(np.any(np.diff(values[:,indices],axis=1)>0,axis=1)):
        original=values[row,indices].tolist()
        blocks=[]
        for i in range(3):
            blocks.append([i,i+1,original[i]])
            while len(blocks)>1 and blocks[-2][2]<blocks[-1][2]:
                right,left=blocks.pop(),blocks.pop()
                begin,end=left[0],right[1]
                blocks.append([begin,end,sum(original[begin:end])/(end-begin)])
        for begin,end,value in blocks:
            out[row,np.asarray(indices[begin:end])]=value
    return out


def expected_csv_values(predictions:np.ndarray,ids:np.ndarray,template_ids:np.ndarray,units:list[str])->np.ndarray:
    """Independently map case-major hourly probabilities into template case order."""
    if len(set(ids.tolist()))!=len(ids) or len(set(template_ids.tolist()))!=len(template_ids) or set(ids.tolist())!=set(template_ids.tolist()):
        raise ValueError('Duplicate, missing or foreign case ID')
    if predictions.shape!=(len(ids)*168,len(units)):
        raise ValueError('Unexpected hourly prediction shape')
    lookup={int(case_id):i for i,case_id in enumerate(ids)}
    reordered=predictions.reshape(len(ids),168,len(units))[[lookup[int(case_id)] for case_id in template_ids]]
    return reordered.transpose(0,2,1).reshape(len(ids),168*len(units))


def main()->None:
    """Verify pinned imports and both exact exports without fitting or inference."""
    parser=argparse.ArgumentParser()
    parser.add_argument('--export-directory',type=Path,required=True)
    parser.add_argument('--export-manifest-sha256',required=True)
    parser.add_argument('--protocol',type=Path,default=ROOT/'docs/cuda_xgb664_final_protocol.json')
    parser.add_argument('--protocol-sha256',required=True)
    parser.add_argument('--gpu-run-name',default='cuda-xgb664-final-001')
    parser.add_argument('--run-id',required=True)
    args=parser.parse_args()
    if Path(args.run_id).name!=args.run_id or args.run_id in ['', '.', '..']:
        raise ValueError('Unique audit run ID required')
    out=ROOT/'artifacts'/args.run_id
    out.mkdir(exist_ok=False)
    start=time.monotonic()
    report={'run_id':args.run_id,'timestamp_utc':datetime.now(timezone.utc).isoformat(),'status':'initializing',
            'command':shlex.join([sys.executable,*sys.argv]),'training':False,'local_checkpoint_inference':False,
            'kaggle_submission':None,'environment':{'python':sys.version,'platform':platform.platform()}}
    try:
        export=args.export_directory.resolve()
        assert digest(export/'manifest.json')==args.export_manifest_sha256
        assert digest(args.protocol)==args.protocol_sha256
        p=json.loads(args.protocol.read_text())
        m=json.loads((export/'manifest.json').read_text())
        assert p['version']=='cuda_xgb664_final_v1' and m['status']=='complete' and m['protocol_sha256']==args.protocol_sha256
        assert p['accepted_validation']['accepted'] and p['accepted_validation']['selected_weight']==.25
        for name,sha in m['artifacts'].items():
            assert digest(export/name)==sha,name
        for name,sha in p['local_acceptance_inputs'].items():
            assert digest(ROOT/name)==sha,name
        assert digest(ROOT/'scripts/verify_cuda_xgb664_return.py')==p['return_verifier_sha256']
        assert digest(ROOT/'scripts/export_cuda_xgb664_final.py')==p['exporter_sha256']
        assert digest(ROOT/'scripts/train_cuda_xgb664_final.py')==p['trainer_sha256']
        gpu=export/'download'/args.gpu_run_name
        assert digest(gpu/'manifest.json')==m['hosted_manifest_sha256']
        gm=json.loads((gpu/'manifest.json').read_text())
        assert gm['status']=='complete' and gm['completed_units']==14 and gm['completed_rounds_per_unit']==600
        assert gm['protocol_sha256']==args.protocol_sha256 and gm['accepted_validation']==p['accepted_validation']
        assert gm['train_case_ids']==p['train_case_ids'] and gm['test_case_ids']==p['test_case_ids']
        assert gm['units']==p['units'] and len(gm['features'])==664
        assert digest(gpu/'train_sample_final.npy')==p['train_sample_sha256']==gm['sample_sha256']
        for name,sha in gm['artifacts'].items():
            assert digest(gpu/name)==sha,name
        for unit in p['units']:
            model=gm['models'][unit]
            assert model['checkpoint_replay_max_error']==0 and model['replay_rows']==725*168
            assert model['checkpoint_sha256']==digest(gpu/f'final_{unit}.ubj')
            assert model['native_config_sha256']==digest(gpu/f'config_final_{unit}.json')
            for key,value in p['model_parameters'].items():
                assert model['expanded_parameters'][key]==value,key
            verify_native_config(json.loads((gpu/f'config_final_{unit}.json').read_text()),p['model_parameters'])
        with np.load(gpu/'test_predictions.npz',allow_pickle=False) as z:
            raw,ids,units=z['predictions'],z['case_ids'],z['units'].tolist()
        replay=np.load(gpu/'replay_test_predictions.npy',allow_pickle=False)
        assert raw.dtype==replay.dtype==np.float32 and raw.shape==replay.shape==(725*168,14)
        assert raw.tobytes()==replay.tobytes() and gm['checkpoint_replay_bitwise']
        assert ids.tolist()==p['test_case_ids'] and units==p['units']
        assert np.isfinite(raw).all() and raw.min()>=0 and raw.max()<=1
        anchor_info=p['local_anchor']
        anchor_dir=ROOT/anchor_info['run_path']
        assert digest(anchor_dir/'manifest.json')==anchor_info['manifest_sha256']
        assert digest(ROOT/anchor_info['verification_path'])==anchor_info['verification_sha256']
        assert digest(anchor_dir/'test_predictions.npy')==anchor_info['predictions_sha256']
        anchor=np.load(anchor_dir/'test_predictions.npy',allow_pickle=False)
        assert anchor.dtype==np.float64 and anchor.shape==raw.shape
        assert independent_projection(anchor,units).tobytes()==anchor.tobytes()
        primary=independent_projection(.75*anchor+.25*raw.astype(np.float64),units)
        standalone=independent_projection(raw.astype(np.float64),units)
        template=pd.read_csv(ROOT/'sintef_project/sample_submission.csv')
        assert template.columns.tolist()==['Run No']+[f'result_committed_{u}_t{h}' for u in units for h in range(168)]
        checks={}
        for name,values,array_name,csv_name in [('primary',primary,'test_predictions.npy','submission.csv'),
                                               ('standalone',standalone,'standalone_test_predictions.npy','submission_standalone_projected.csv')]:
            saved=np.load(export/array_name,allow_pickle=False)
            assert values.dtype==saved.dtype==np.float64 and values.shape==saved.shape and values.tobytes()==saved.tobytes()
            assert np.isfinite(values).all() and values.min()>=0 and values.max()<=1
            csv=pd.read_csv(export/csv_name)
            assert csv.shape==template.shape==(725,2353) and csv.columns.tolist()==template.columns.tolist()
            np.testing.assert_array_equal(csv['Run No'].to_numpy(),template['Run No'].to_numpy())
            expected=expected_csv_values(values,ids,template['Run No'].to_numpy(),units)
            actual=csv.iloc[:,1:].to_numpy()
            assert np.isfinite(actual).all() and actual.min()>=0 and actual.max()<=1
            error=float(np.max(np.abs(actual-expected)))
            assert error<=5.1e-10
            checks[name]={'array_bitwise_equal':True,'csv_max_absolute_error':error,
                          'submission_sha256':digest(export/csv_name),'predictions_sha256':digest(export/array_name)}
        assert m['submission_sha256']==checks['primary']['submission_sha256']
        assert m['diagnostic_export']['submission_sha256']==checks['standalone']['submission_sha256']
        assert m['validation_scores']==p['accepted_validation']['selected_scores']
        assert m['diagnostic_export']['validation_scores']==p['diagnostic_export']['validation_scores']
        assert m['diagnostic_export']['automatic_promotion'] is False
        report.update(status='passed',export_manifest_sha256=args.export_manifest_sha256,protocol_sha256=args.protocol_sha256,
                      hosted_manifest_sha256=m['hosted_manifest_sha256'],primary=checks['primary'],standalone=checks['standalone'],
                      formula='Independent generic PAVA of .75 pinned projected GPU608 anchor + .25 raw SHOP664; separate PAVA(raw SHOP664).',
                      accepted_anchor_manifest_sha256=anchor_info['manifest_sha256'],
                      primary_validation_scores=p['accepted_validation']['selected_scores'],
                      standalone_validation_scores=p['diagnostic_export']['validation_scores'],
                      automatic_promotion=False,all_models_and_full_test_replay_verified=True)
    except BaseException:
        report.update(status='failed',error=traceback.format_exc())
        raise
    finally:
        report.update(runtime_seconds=time.monotonic()-start,audit_source_sha256=digest(Path(__file__)))
        (out/'audit.py').write_bytes(Path(__file__).read_bytes())
        (out/'manifest.json').write_text(json.dumps(report,indent=2))
        (out/'experiment_record.md').write_text(f'\n## {args.run_id} — {report["timestamp_utc"]}\n\nIndependent final SHOP664 primary and standalone export audit. Status {report["status"]}; runtime {report["runtime_seconds"]:.3f} s. Command `{report["command"]}`. All source/protocol/checkpoint/sample/raw replay and exported array/CSV evidence is in `artifacts/{args.run_id}/manifest.json`. No fit, weight search, automatic promotion or leaderboard claim.\n')


if __name__=='__main__':
    main()
