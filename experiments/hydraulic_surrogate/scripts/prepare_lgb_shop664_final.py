"""Seal a final recipe only from actual independently accepted2022 confirmation."""
from datetime import datetime,timezone
import argparse,json
from pathlib import Path
import sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.train_lgb_shop664 import sha,environment
from scripts.train_lgb_shop664_final import verify_confirmation


def main()->None:
    ap=argparse.ArgumentParser();ap.add_argument('--confirmation',type=Path,required=True);ap.add_argument('--confirmation-manifest-sha256',required=True)
    ap.add_argument('--confirmation-audit',type=Path,required=True);ap.add_argument('--confirmation-audit-sha256',required=True);a=ap.parse_args()
    vp=ROOT/'docs/lgb_shop664_protocol.json';assert sha(vp)=='882e8c3124ba834322c8f062a7544e9ca3e9512b385e8830db387c38f7c97d08';v=json.loads(vp.read_text())
    assert sha(a.confirmation/'manifest.json')==a.confirmation_manifest_sha256 and sha(a.confirmation_audit)==a.confirmation_audit_sha256
    confirmation=json.loads((a.confirmation/'manifest.json').read_text());audit=json.loads(a.confirmation_audit.read_text())
    assert confirmation['status']=='complete'and confirmation['year']==2022 and confirmation['accepted']
    assert audit['status']=='passed'and audit['accepted']and audit['candidate_manifest_sha256']==a.confirmation_manifest_sha256
    lock_path=ROOT/'artifacts/lgb-shop664-2021-001/selection_2021.json'
    assert sha(lock_path)=='cc4c7e21967d61bbb6c9139ff32a6da2586a8686156d520f3bccf3eaab28a1b9'
    lock=json.loads(lock_path.read_text());assert confirmation['selection_lock_sha256']==sha(lock_path)
    inputs=dict(v['inputs']);inputs[str(vp.relative_to(ROOT))]=sha(vp);accepted={}
    first=ROOT/'artifacts/lgb-shop664-2021-001';first_audit=ROOT/'artifacts/lgb-shop664-2021-independent-audit-001/manifest.json'
    assert sha(first/'manifest.json')=='88faf9b155271b5ab46488031b9510d5aecbb6a9bbd1abe0c19554808b5bf413'
    assert sha(first_audit)=='31e11af7408082765541ba8cf45d7c4c7bcfc2884d80a664a7d5969961d0e005'
    for year,folder,proof in [('2021',first,first_audit),('2022',a.confirmation,a.confirmation_audit)]:
        mp=folder/'manifest.json';m=json.loads(mp.read_text())
        accepted[year]={'manifest':str(mp.resolve().relative_to(ROOT)),'manifest_sha256':sha(mp),'audit':str(proof.resolve().relative_to(ROOT)),'audit_sha256':sha(proof),
                        'selected_micro_auc':m['selected_micro_auc'],'anchor_micro_auc':m['anchor_micro_auc'],'control_micro_auc':m['control_micro_auc']}
        inputs[str(mp.resolve().relative_to(ROOT))]=sha(mp);inputs[str(proof.resolve().relative_to(ROOT))]=sha(proof)
    original=ROOT/'artifacts/lgb-normalized-lp-context-joint-final-001/manifest.json';om=json.loads(original.read_text());assert om['status']=='complete'
    sample=original.parent/'train_sample_final.npy';assert sha(sample)==om['artifacts'][sample.name]
    anchor=ROOT/'artifacts/submission-cuda-xgb664-anchor-blend-001';am=json.loads((anchor/'manifest.json').read_text())
    assert sha(anchor/'manifest.json')=='dcd1617cb60aa29dce9c9a64bff16fbc25f00ef569912ea1b80c21eea5aa973c'
    assert am['status']=='complete'and am['submission_sha256']=='898d29699cda7aa93cc5b22e410d63eb7fa371ce4afc9f0e9b5a176993e9e96a'
    apath=ROOT/'artifacts/cuda-dual-final-submissions-independent-audit-001/manifest.json'
    assert sha(apath)=='ff88843e5eb8aa1d351ceb36b0640afb08dfc82d2490ecba294e3dacf272aa7d'
    for path in [lock_path,original,sample,anchor/'manifest.json',anchor/'test_predictions.npy',apath]:inputs[str(path.relative_to(ROOT))]=sha(path)
    sources=dict(v['source'])
    for name in ['scripts/train_lgb_shop664_final.py','scripts/export_lgb_shop664_final.py','scripts/prepare_lgb_shop664_final.py','tests/test_lgb_shop664_final.py','docs/lgb_shop664_final.md']:sources[name]=sha(ROOT/name)
    train=pd.read_csv(ROOT/'sintef_project/data/kernel/Unit_commitment_decisions.csv',usecols=['Run No','starttime']);mapping=pd.read_csv(ROOT/'sintef_project/prediction_mapping.csv')
    p={'version':'lgb_shop664_accepted_final_v1','created_utc':datetime.now(timezone.utc).isoformat(),'environment':environment(),'source':sources,'inputs':inputs,
       'validation_protocol':str(vp.relative_to(ROOT)),'validation_protocol_sha256':sha(vp),'accepted_validation':accepted,
       'selection_lock':str(lock_path.relative_to(ROOT)),'selection_lock_sha256':sha(lock_path),'selected_weight':lock['selected_weight'],
       'model_parameters':v['model_parameters'],'original_final_manifest':str(original.relative_to(ROOT)),
       'sample':{'path':str(sample.relative_to(ROOT)),'sha256':sha(sample)},'train_ids':train['Run No'].tolist(),'test_ids':mapping['Run No'].tolist(),
       'anchor':{'directory':str(anchor.relative_to(ROOT)),'manifest_sha256':sha(anchor/'manifest.json'),'prediction_sha256':sha(anchor/'test_predictions.npy'),'audit':str(apath.relative_to(ROOT)),'audit_sha256':sha(apath)},
       'formula':'project_vinje((1-selected_weight)*saved_projected_acceptedSHOP664_anchor+selected_weight*raw_LGB664),float64',
       'scope':'One original196358-row sample,14x350trees/2CPUthreads, no early stopping; exact full121800test-row checkpoint replay; one primary CSV only after independentfinalreplay proof.',
       'resources':v['resources'],'inference':'Same byte-pinned recipe with --inference-from and external original finalmanifestSHA; no refit in replay mode.'}
    verify_confirmation(p)
    output=ROOT/'docs/lgb_shop664_final_protocol.json'
    with output.open('x')as stream:json.dump(p,stream,indent=2)
    print(sha(output),flush=True)


if __name__=='__main__':main()
