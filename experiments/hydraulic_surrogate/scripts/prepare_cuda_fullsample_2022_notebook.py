"""Prepare fixed fullsample2022 confirmation only after independently audited2021 selection."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.prepare_cuda_dual_2022_notebook import replace_once

def sha(p:Path)->str:
    """Hash an immutable source or manifest."""
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main()->None:
    """Create a notebook only, with genuine locked bytes and no remote action."""
    ap=argparse.ArgumentParser();ap.add_argument('--selection-lock',type=Path,required=True);ap.add_argument('--selection-lock-sha256',required=True);ap.add_argument('--audit',type=Path,required=True);ap.add_argument('--audit-sha256',required=True);ap.add_argument('--run-id',required=True);args=ap.parse_args()
    assert Path(args.run_id).name==args.run_id and args.run_id not in ['', '.', '..']
    assert sha(args.selection_lock)==args.selection_lock_sha256 and sha(args.audit)==args.audit_sha256
    lock=json.loads(args.selection_lock.read_text());audit=json.loads(args.audit.read_text())
    assert audit['status']=='passed' and audit['selection_lock_sha256']==args.selection_lock_sha256 and audit['eligible_for_2022']
    assert lock['training_protocol_sha256']=='d5add5941279d980f156015b4b5ca70da2af4b6d8b99c4841050569b02b7255e' and lock['selection_year']==2021 and lock['eligible_for_2022'] and lock['selected_weight']>0
    assert lock['selected_2021_micro_auc']>max(lock['anchor_2021_micro_auc'],lock['control_2021_micro_auc'])
    template=ROOT/'artifacts/cuda-xgb608-fullsample-notebook-001/run_fullsample_2021.py';assert sha(template)=='ac4f0901fadc47653f53d19063f0a82a35210381c113d0a5b831bf25127bbd6b'
    source=template.read_text();source=source.replace('smoke-001','smoke-002').replace('fullsample-2021-001','fullsample-2022-001')
    source=source.replace('FULLSAMPLE2021','FULLSAMPLE2022').replace('smoke and2021 fold','smoke and locked2022 fold')
    source=replace_once(source,"'--mode', 'fit', '--year', '2021', '--run-id', FIT_ID,\n               '--smoke-manifest', str(smoke_path)]","'--mode', 'fit', '--year', '2022', '--run-id', FIT_ID,\n               '--smoke-manifest', str(smoke_path), '--selection-lock', str(lock_path), '--selection-lock-sha256', LOCK_SHA]")
    source=replace_once(source,"manifest.get('year') == 2021","manifest.get('year') == 2022")
    source=replace_once(source,"protocol['sample_rows']['2021'] == 367248","protocol['sample_rows']['2022'] == 428568\n    assert manifest['selection_lock_sha256'] == LOCK_SHA")
    source=replace_once(source,"inspect preserved ZIP/logs, do not select or run2022","inspect preserved ZIP/logs; do not fit or export a final")
    marker="print('PINNED_RUNTIME_VERIFIED', stamp(), gpu, PACKAGES, flush=True)"
    embedded=f"LOCK_SHA = {args.selection_lock_sha256!r}\nAUDIT_SHA = {args.audit_sha256!r}\nLOCK_TEXT = {args.selection_lock.read_text()!r}\nAUDIT_TEXT = {args.audit.read_text()!r}\n"+'''assert hashlib.sha256(LOCK_TEXT.encode()).hexdigest() == LOCK_SHA
assert hashlib.sha256(AUDIT_TEXT.encode()).hexdigest() == AUDIT_SHA
locked, audited = json.loads(LOCK_TEXT), json.loads(AUDIT_TEXT)
assert audited['status'] == 'passed' and audited['selection_lock_sha256'] == LOCK_SHA and audited['eligible_for_2022']
assert locked['training_protocol_sha256'] == PROTOCOL_SHA and locked['selection_year'] == 2021 and locked['eligible_for_2022']
assert locked['selected_weight'] > 0 and locked['selected_2021_micro_auc'] > max(locked['anchor_2021_micro_auc'], locked['control_2021_micro_auc'])
lock_root = ROOT / 'independently-verified-fullsample2021-lock'
lock_root.mkdir(exist_ok=False)
lock_path = lock_root / 'selection_2021.json'
lock_path.write_bytes(LOCK_TEXT.encode())
(lock_root / 'independent_audit.json').write_bytes(AUDIT_TEXT.encode())
assert sha(lock_path) == LOCK_SHA
'''
    source=replace_once(source,marker,embedded+marker);compile(source,'run_fullsample_2022.py','exec')
    out=ROOT/'artifacts'/args.run_id;out.mkdir(exist_ok=False);sp=out/'run_fullsample_2022.py';sp.write_text(source)
    book={'cells':[{'cell_type':'markdown','metadata':{},'source':['# Frozen fullsample2022 confirmation\n','Only the independently accepted2021 fixed weight is used. Fresh genuine smoke002, exact full-row600-tree recipe and checkpoint replays. No weight search or final training.\n']},{'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':source.splitlines(keepends=True)}],'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5};bp=out/'tdt4173-fullsample608-frozen-2022.ipynb';bp.write_text(json.dumps(book,indent=2));shutil.copy2(__file__,out/Path(__file__).name)
    m={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'status':'prepared','training':False,'source_sha256':sha(sp),'notebook_sha256':sha(bp),'builder_sha256':sha(Path(__file__)),'template_sha256':sha(template),'selection_lock_sha256':args.selection_lock_sha256,'audit_sha256':args.audit_sha256,'selected_weight':lock['selected_weight']};(out/'manifest.json').write_text(json.dumps(m,indent=2));print(json.dumps(m,indent=2))
if __name__=='__main__':main()
