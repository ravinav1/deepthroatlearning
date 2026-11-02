"""Export only an independently verified acceptedLGB664 final nested blend."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import importlib.util,json,shlex,shutil,sys,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.train_lgb_shop664_final import check_final_protocol
from scripts.train_lgb_shop664 import sha
from sintef_project.postprocess import project_vinje


def compose(anchor:np.ndarray,raw:np.ndarray,units:list[str],weight:float)->np.ndarray:
    """Preserve the exact already-projected acceptedSHOP anchor before blending."""
    assert anchor.shape==raw.shape and 0<weight<=1
    assert np.isfinite(raw).all()and 0<=raw.min()<=raw.max()<=1
    assert np.array_equal(project_vinje(anchor,units),anchor)
    return project_vinje((1-weight)*anchor.astype(np.float64)+weight*raw.astype(np.float64),units)


def submission_frame(pred:np.ndarray,ids:np.ndarray,units:list[str],template:pd.DataFrame)->pd.DataFrame:
    """Map preserved case-hour-unit arrays into exact template IDs and columns."""
    columns=['Run No']+[f'result_committed_{u}_t{h}'for u in units for h in range(168)]
    assert template.columns.tolist()==columns and len(set(ids.tolist()))==len(ids)
    assert set(ids.tolist())==set(template['Run No'].tolist())
    assert pred.shape==(len(ids)*168,len(units))and np.isfinite(pred).all()and 0<=pred.min()<=pred.max()<=1
    wide=pred.reshape(len(ids),168,len(units)).transpose(0,2,1).reshape(len(ids),-1)
    frame=pd.DataFrame(wide,index=pd.Index(ids,name='Run No'),columns=columns[1:])
    return frame.loc[template['Run No']].reset_index()


def main()->None:
    """Check independently replayed final artifacts before writing one reviewable CSV."""
    ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--protocol-sha256',required=True)
    ap.add_argument('--final-run',type=Path,required=True);ap.add_argument('--final-manifest-sha256',required=True)
    ap.add_argument('--final-audit',type=Path,required=True);ap.add_argument('--final-audit-sha256',required=True);ap.add_argument('--run-id',required=True)
    a=ap.parse_args();assert Path(a.run_id).name==a.run_id and a.run_id not in ['','.','..']
    out=ROOT/'artifacts'/a.run_id;out.mkdir(exist_ok=False);start=time.monotonic()
    report={'status':'initializing','timestamp_utc':datetime.now(timezone.utc).isoformat(),'command':shlex.join([sys.executable,*sys.argv]),'training':False,'kaggle_submission':None}
    try:
        p,v,_=check_final_protocol(a.protocol,a.protocol_sha256)
        assert sha(a.final_run/'manifest.json')==a.final_manifest_sha256 and sha(a.final_audit)==a.final_audit_sha256
        m=json.loads((a.final_run/'manifest.json').read_text());audit=json.loads(a.final_audit.read_text())
        assert audit['status']=='passed'and audit['final_manifest_sha256']==a.final_manifest_sha256
        assert audit['protocol_sha256']==a.protocol_sha256 and audit['all14_full_test_checkpoints_replayed']
        assert m['status']=='complete'and m['protocol_sha256']==a.protocol_sha256 and m['checkpoint_replay_bitwise']
        assert m['completed_units']==14 and m['completed_rounds_per_unit']==350 and m['sampled_training_rows']==196358
        assert m['model_parameters']==p['model_parameters']and m['accepted_validation']==p['accepted_validation']
        assert m['sample_sha256']==p['sample']['sha256']and m['features']==v['feature_columns']
        assert m['train_ids']==p['train_ids']and m['test_ids']==p['test_ids']and m['units']==v['units']
        for name,h in m['artifacts'].items():assert sha(a.final_run/name)==h,name
        for unit,record in m['models'].items():
            assert record['trees']==350 and record['features']==664 and record['replay_rows']==725*168 and record['float64_replay_bitwise']and record['checkpoint_replay_max_error']==0
        raw=np.load(a.final_run/'test_predictions.npy',allow_pickle=False);replay=np.load(a.final_run/'replay_test_predictions.npy',allow_pickle=False)
        assert raw.dtype==replay.dtype==np.float32 and raw.shape==replay.shape==(725*168,14)and raw.tobytes()==replay.tobytes()
        anchor_dir=ROOT/p['anchor']['directory'];anchor_manifest=json.loads((anchor_dir/'manifest.json').read_text())
        assert sha(anchor_dir/'manifest.json')==p['anchor']['manifest_sha256']
        assert anchor_manifest['status']=='complete'and anchor_manifest['final_full_test_replay_bitwise']and anchor_manifest['anchor_source_reconstruction_bitwise']
        for name,h in anchor_manifest['artifacts'].items():assert sha(anchor_dir/name)==h,name
        anchor_proof=ROOT/p['anchor']['audit'];assert sha(anchor_proof)==p['anchor']['audit_sha256']and json.loads(anchor_proof.read_text())['status']=='passed'
        anchor=np.load(anchor_dir/'test_predictions.npy',allow_pickle=False)
        assert sha(anchor_dir/'test_predictions.npy')==p['anchor']['prediction_sha256']and anchor.dtype==np.float64
        combined=compose(anchor,raw,v['units'],p['selected_weight']);ids=np.asarray(p['test_ids'])
        template=pd.read_csv(ROOT/'sintef_project/sample_submission.csv');frame=submission_frame(combined,ids,v['units'],template)
        assert frame.shape==(725,2353)
        frame.to_csv(out/'submission.csv',index=False,float_format='%.9g');np.save(out/'test_predictions.npy',combined)
        readback=pd.read_csv(out/'submission.csv');assert readback.columns.tolist()==template.columns.tolist()and readback['Run No'].tolist()==template['Run No'].tolist()
        values=readback.iloc[:,1:].to_numpy();assert np.isfinite(values).all()and 0<=values.min()<=values.max()<=1
        error=float(np.max(np.abs(values-frame.iloc[:,1:].to_numpy())));assert error<=5.000001e-10
        report.update(status='complete',protocol_sha256=a.protocol_sha256,final_manifest_sha256=a.final_manifest_sha256,
                      final_independent_audit_sha256=a.final_audit_sha256,anchor_manifest_sha256=p['anchor']['manifest_sha256'],
                      selected_weight=p['selected_weight'],formula=p['formula'],shape=list(frame.shape),probability_range=[float(combined.min()),float(combined.max())],
                      csv_readback_max_error=error,submission_sha256=sha(out/'submission.csv'),accepted_validation=p['accepted_validation'])
    except BaseException:report.update(status='failed',error=traceback.format_exc());raise
    finally:
        shutil.copy2(__file__,out/Path(__file__).name)
        if a.protocol.is_file():shutil.copy2(a.protocol,out/'protocol.json')
        report['runtime_seconds']=time.monotonic()-start;report['artifacts']={f.name:sha(f)for f in out.iterdir()if f.is_file()and f.name!='manifest.json'}
        (out/'manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps({'status':report['status'],'submission_sha256':report.get('submission_sha256')}),flush=True)


if __name__=='__main__':main()
