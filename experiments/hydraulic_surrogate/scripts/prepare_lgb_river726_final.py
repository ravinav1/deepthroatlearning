"""Seal726 final only after actual independently accepted chronological folds."""
from datetime import datetime,timezone
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.train_lgb_shop664 import sha,environment
from scripts.train_lgb_river726_final import verify_confirmation,check_final_protocol
from scripts import train_lgb_river726 as fold


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def main() -> None:
    ap=argparse.ArgumentParser()
    for name in ['selection-run','selection-audit','confirmation','confirmation-audit']:ap.add_argument('--'+name,type=Path,required=True)
    for name in ['selection-manifest-sha256','selection-audit-sha256','confirmation-manifest-sha256','confirmation-audit-sha256']:ap.add_argument('--'+name,required=True)
    ap.add_argument('--output',type=Path,default=ROOT/'docs/lgb_river726_final_protocol.json');a=ap.parse_args()
    vp=ROOT/'docs/lgb_river726_protocol.json';assert sha(vp)=='c71d6a026f7ca3e802464df4be7b8874ab8592f54d38d85cd97e2b9040762b7c'
    v=json.loads(vp.read_text());fold.check_closure(v)
    original_path=ROOT/'docs/lgb_shop664_final_protocol.json'
    assert sha(original_path)=='5513e67123a968ded3790ec6bcda898d50d3682685fa7f62020ea25108e322ad'
    original=json.loads(original_path.read_text())
    inputs={**original['inputs'],**v['inputs'],rel(original_path):sha(original_path),rel(vp):sha(vp)}
    accepted={}
    for year,directory,proof,manifest_sha,audit_sha in [
        ('2021',a.selection_run,a.selection_audit,a.selection_manifest_sha256,a.selection_audit_sha256),
        ('2022',a.confirmation,a.confirmation_audit,a.confirmation_manifest_sha256,a.confirmation_audit_sha256)]:
        mp=directory/'manifest.json';assert sha(mp)==manifest_sha and sha(proof)==audit_sha
        m=json.loads(mp.read_text());audit=json.loads(proof.read_text())
        assert m['status']=='complete'and m['year']==int(year)and m['protocol_sha256']==sha(vp)and m['beats_anchor_and_control']
        assert audit['status']=='passed'and audit['candidate_manifest_sha256']==manifest_sha and audit['protocol_sha256']==sha(vp)
        assert audit['eligible_for_2022']if year=='2021'else audit['accepted']and m['accepted']
        accepted[year]={'manifest':rel(mp),'manifest_sha256':manifest_sha,'audit':rel(proof),'audit_sha256':audit_sha,
            'selected_micro_auc':m['selected_micro_auc'],'anchor_micro_auc':m['anchor_micro_auc'],'control_micro_auc':m['control_micro_auc']}
        inputs[rel(mp)]=manifest_sha;inputs[rel(proof)]=audit_sha
    lock_path=a.selection_run/'selection_2021.json';lock_sha=sha(lock_path);inputs[rel(lock_path)]=lock_sha
    lock=fold.validate_lock(lock_path,lock_sha,v,sha(vp),a.selection_audit,a.selection_audit_sha256)
    confirmation=json.loads((a.confirmation/'manifest.json').read_text());assert confirmation['selection_lock_sha256']==lock_sha
    sources={**original['source'],**v['source']}
    for name in ['scripts/train_lgb_river726_final.py','scripts/prepare_lgb_river726_final.py','scripts/export_lgb_river726_final.py',
                 'tests/test_lgb_river726_final.py','docs/lgb_river726_final.md']:sources[name]=sha(ROOT/name)
    p={'version':'lgb_river726_accepted_final_v1','created_utc':datetime.now(timezone.utc).isoformat(),
       'environment':environment(),'source':sources,'inputs':inputs,'validation_protocol':rel(vp),'validation_protocol_sha256':sha(vp),
       'accepted_validation':accepted,'selection_lock':rel(lock_path),'selection_lock_sha256':lock_sha,'selected_weight':lock['selected_weight'],
       'model_parameters':v['model_parameters'],'original_final_manifest':original['original_final_manifest'],
       'sample':original['sample'],'train_ids':original['train_ids'],'test_ids':original['test_ids'],'anchor':original['anchor'],
       'cache':v['cache'],'resources':v['resources'],
       'formula':'project_vinje((1-selected_weight)*saved_projected_acceptedSHOP664_anchor+selected_weight*raw_LGB726),float64',
       'scope':'One unchanged196358-row sample,14x350trees/2CPUthreads, no earlystopping; exact664+audited62features/full121800test-row replay; oneprimary CSV only after independentfinalreplay.',
       'inference':'Same pinnedrecipe with --inference-from and originalmanifestSHA, no refit.'}
    verify_confirmation(p)
    # Exclusive create prevents replacement of an already frozen recipe.
    with a.output.open('x')as stream:json.dump(p,stream,indent=2)
    check_final_protocol(a.output,sha(a.output))
    print(sha(a.output),flush=True)

if __name__=='__main__':main()
