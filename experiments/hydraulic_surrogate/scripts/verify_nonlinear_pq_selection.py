"""Independently verify saved nonlinear-PQ fold artifacts and fixed splices."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback

import numpy as np
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sintef_project.postprocess import project_vinje


def require(value: bool, message: str) -> None:
    """Reject any changed source, provenance, alignment or score."""
    if not value:
        raise ValueError(message)


def digest(path: Path) -> str:
    """Hash a saved file without loading it wholly into memory."""
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            result.update(block)
    return result.hexdigest()


def splice(anchor: np.ndarray, raw: np.ndarray, indices: list[int], weight: float,
           units: list[str]) -> np.ndarray:
    """Independently form the whole matrix in float64, changing only target units."""
    output = np.array(anchor, dtype=np.float64, copy=True)
    for j, index in enumerate(indices):
        output[:, index] = (1-weight) * np.asarray(anchor[:, index], dtype=np.float64) + weight * np.asarray(raw[:, j], dtype=np.float64)
    return project_vinje(output, units)


def main() -> None:
    """Check completed saved artifacts without a model fit or feature reconstruction."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', choices=[2021, 2022], type=int, required=True)
    parser.add_argument('--model-run', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--selection-lock', type=Path)
    parser.add_argument('--selection-lock-sha256')
    args = parser.parse_args()
    out = ROOT/'artifacts'/args.run_id;out.mkdir(exist_ok=False)
    started=time.time()
    result={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'year':args.year,'model_run':args.model_run,
            'command':[sys.executable,*sys.argv],'environment':{'python':sys.version,'platform':platform.platform()},
            'training_or_feature_rebuild':False,'kaggle':None}
    try:
        directory=ROOT/'artifacts'/args.model_run
        manifest=json.loads((directory/'manifest.json').read_text())
        require(manifest['status']=='complete' and manifest['year']==args.year,'Incomplete/wrong fold')
        for name,expected in manifest['artifacts'].items():
            require(digest(directory/name)==expected,f'Changed artifact:{name}')
        protocol=json.loads((directory/'protocol.json').read_text())
        protocol_sha=digest(directory/'protocol.json')
        require(protocol_sha==manifest['protocol_sha256']==digest(ROOT/'docs/nonlinear_pq_protocol.json'),'Protocol changed')
        for name,expected in protocol['source'].items():
            require(digest(ROOT/name)==digest(directory/'source'/name)==expected,f'Changed source:{name}')
        for name,expected in protocol['data'].items():
            require(digest(ROOT/name)==expected,f'Changed data:{name}')
        base=ROOT/'artifacts/lgb-normalized-lp-context-joint-001'
        anchor_dir=ROOT/'artifacts/lp-context-joint-comparison-001'
        cache=ROOT/'artifacts/nonlinear-pq-features-001'
        for path,key in [(base/'manifest.json','base_manifest_sha256'),(anchor_dir/'analysis.json','anchor_analysis_sha256'),
                         (cache/'manifest.json','cache_manifest_sha256'),(cache/'features.npz','cache_features_sha256')]:
            require(digest(path)==protocol[key],f'Changed frozen input:{path}')
        baseline=json.loads((base/'manifest.json').read_text());anchor_manifest=json.loads((anchor_dir/'analysis.json').read_text())
        require(manifest['features']==baseline['features']+protocol['append_columns'] and len(manifest['features'])==614,'614 schema changed')
        require(manifest['model_parameters']==protocol['model_parameters']=={**baseline['model_parameters'],'n_jobs':2},'Model recipe changed')
        require(manifest['fold']==baseline['folds'][str(args.year)],'Recorded chronological IDs/splits changed')
        require(digest(directory/f'train_sample_{args.year}.npy')==protocol['sample_sha256'][str(args.year)]
                ==digest(base/f'train_sample_{args.year}.npy'),'Exact baseline sample changed')
        if args.year==2021:
            lock_path=directory/'selection_2021.json'
            require(digest(lock_path)==manifest['selection_lock_sha256'],'2021 lock changed')
        else:
            require(args.selection_lock is not None and args.selection_lock_sha256 is not None,'Explicit2021lock required')
            lock_path=args.selection_lock
            require(digest(lock_path)==args.selection_lock_sha256==manifest['selection_lock_sha256'],'2022 used another lock')
        lock=json.loads(lock_path.read_text())
        require(lock['year']==2021 and lock['protocol_sha256']==protocol_sha and lock['candidate_weights']==[0.,.1,.25,.5],'Wrong2021lock')
        require(set(lock['scores_2021'])=={str(w) for w in lock['candidate_weights']},'Incomplete frozen2021grid')
        selected=lock['selected_weight']
        require(selected==max(lock['candidate_weights'],key=lambda w:lock['scores_2021'][str(w)])
                and lock['selected_micro_auc']==lock['scores_2021'][str(selected)],'Wrong frozenargmax/tie')
        require(lock['eligible_for_2022']==(selected>0 and lock['selected_micro_auc']>max(lock['anchor_micro_auc'],lock['control_micro_auc'])),'Wrong2021gate')
        anchor_path=anchor_dir/f'selected_validation_{args.year}.npz';control_path=base/f'validation_{args.year}.npz'
        require(digest(anchor_path)==anchor_manifest['artifacts'][anchor_path.name]
                and digest(control_path)==baseline['artifacts'][control_path.name],'Anchor/control data changed')
        with np.load(directory/f'raw_validation_{args.year}.npz',allow_pickle=False) as data:
            raw,labels,ids,units,targets=data['predictions'],data['labels'],data['case_ids'],data['units'].tolist(),data['targets'].tolist()
        indices=[units.index(unit) for unit in targets]
        require(targets==protocol['targets']==['Haukeli_G1','Lio_G1','Hogga_G1'],'Target selection changed')
        require(ids.tolist()==baseline['folds'][str(args.year)]['validation_ids'],'OOF case order changed')
        require(raw.shape==(len(ids)*168,3) and labels.shape==(len(ids)*168,14)
                and np.isfinite(raw).all() and raw.min()>=0 and raw.max()<=1 and np.isin(labels,[0,1]).all(),'Invalid probabilities/labels')
        arrays=[]
        for path in [anchor_path,control_path]:
            with np.load(path,allow_pickle=False) as data:
                require(np.array_equal(data['labels'],labels) and np.array_equal(data['case_ids'],ids)
                        and data['units'].tolist()==units,'Label/ID/unit mismatch')
                arrays.append(data['predictions'])
        anchor,control=arrays
        scores={str(w):float(roc_auc_score(labels.ravel(),splice(anchor,raw,indices,w,units).ravel()))
                for w in (lock['candidate_weights'] if args.year==2021 else [selected])}
        require(scores==manifest['scores'] and (args.year!=2021 or scores==lock['scores_2021']),'Independent grid scores differ')
        rebuilt=splice(anchor,raw,indices,selected,units)
        with np.load(directory/f'selected_validation_{args.year}.npz',allow_pickle=False) as data:
            require(rebuilt.dtype==data['predictions'].dtype and rebuilt.tobytes()==data['predictions'].tobytes(),'Selected splice not byteexact')
        control_auc=float(roc_auc_score(labels.ravel(),splice(anchor,control[:,indices],indices,selected,units).ravel()))
        anchor_auc=float(roc_auc_score(labels.ravel(),anchor.ravel()))
        require(control_auc==manifest['control_micro_auc'] and anchor_auc==manifest['anchor_micro_auc'],'Reference scores differ')
        eligible=selected>0 and scores[str(selected)]>max(control_auc,anchor_auc)
        require(eligible==manifest['beats_anchor_and_control'],'Acceptance differs from fixedgate')
        if args.year==2021:
            require(digest(anchor_path)==lock['anchor_sha256'] and digest(control_path)==lock['control_sha256']
                    and digest(directory/'raw_validation_2021.npz')==lock['raw_prediction_sha256'],'Lock provenancechanged')
        unit_scores={}
        for j,unit in enumerate(targets):
            record=manifest['models'][unit]
            require(record['exact_checkpoint_replay'] is True
                    and digest(directory/f'fold{args.year}_{unit}.txt')==record['checkpoint_sha256'],'Checkpoint/replayproofchanged')
            score=float(roc_auc_score(labels[:,indices[j]],raw[:,j]));require(score==record['unit_auc'],'Unit AUC differs')
            unit_scores[unit]={'new_raw_auc':score,'matched608_auc':float(roc_auc_score(labels[:,indices[j]],control[:,indices[j]]))}
        result.update(status='passed',eligible=eligible,selected_weight=selected,selected_auc=scores[str(selected)],
                      anchor_auc=anchor_auc,control_auc=control_auc,unit_scores=unit_scores,
                      selection_lock_sha256=digest(lock_path),model_manifest_sha256=digest(directory/'manifest.json'),
                      protocol_sha256=protocol_sha,exact_saved_splice=True,checkpoint_replay='Recorded exact replay and checkpoint hashes verified; no new inference')
        print(json.dumps(result,indent=2),flush=True)
    except BaseException:
        result.update(status='failed',error=traceback.format_exc());raise
    finally:
        result['runtime_seconds']=time.time()-started
        shutil.copy2(__file__,out/'verify_nonlinear_pq_selection.py')
        result['verifier_sha256']=digest(Path(__file__))
        (out/'manifest.json').write_text(json.dumps(result,indent=2))
        with (ROOT/'docs/experiments.md').open('a') as stream:
            stream.write(f'\n## {args.run_id} — {result["timestamp_utc"]}\n\nStatus:{result["status"]}. Independentsavedartifact/protocol/source/data/cache/sample/chronologicalID/checkpoint-hash verification; nofit orfeaturematrix. Recomputed fullfixedsplicegrid and sameweightmatched608control; exactselectedarraybytes verified. SelectedAUC{result.get("selected_auc")},anchor{result.get("anchor_auc")},control{result.get("control_auc")},eligible{result.get("eligible")}. Proof/source/command:`artifacts/{args.run_id}/manifest.json`. Runtime{result["runtime_seconds"]:.3f}s. Unitcounterexamples retained; noKaggle submission orscore.\n')


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
