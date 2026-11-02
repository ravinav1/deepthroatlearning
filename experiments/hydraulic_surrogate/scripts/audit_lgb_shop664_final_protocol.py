"""Verify the actual accepted LGB664 final protocol against reviewed templates."""
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.train_lgb_shop664_final import check_final_protocol
from scripts.train_lgb_shop664 import sha,environment


def main() -> None:
    """Approve actual acceptance/source/sample closure, without fitting or features."""
    out=ROOT/'artifacts/lgb-shop664-final-protocol-independent-audit-001';out.mkdir(exist_ok=False)
    start=time.monotonic()
    r={'status':'initializing','timestamp_utc':datetime.now(timezone.utc).isoformat(),
       'command':[sys.executable,*sys.argv],'training':False,'feature_build':False,
       'labels_decoded':False,'environment':environment()}
    try:
        path=ROOT/'docs/lgb_shop664_final_protocol.json'
        expected='5513e67123a968ded3790ec6bcda898d50d3682685fa7f62020ea25108e322ad'
        p,v,_=check_final_protocol(path,expected)
        template=ROOT/'artifacts/lgb-shop664-final-template-independent-audit-001/manifest.json'
        assert sha(template)=='0f952fa4e58dc574ea73e77239a8ebb6be1f567a585f445ae83ef0f4930454da'
        t=json.loads(template.read_text());assert t['status']=='passed_template_only'
        for name,h in t['reviewed_source'].items():
            assert sha(ROOT/name)==h==p['source'][name]
            assert sha(template.parent/'source'/name)==h
        assert p['selected_weight']==.1 and p['model_parameters']==t['model_parameters']
        assert p['selection_lock_sha256']=='cc4c7e21967d61bbb6c9139ff32a6da2586a8686156d520f3bccf3eaab28a1b9'
        second=p['accepted_validation']['2022']
        assert second['manifest_sha256']=='fa10113f17fe184559921fd9b248aeb77245aa11ccc89dbd1318855344642adb'
        assert second['audit_sha256']=='35f65d33e51cd3ee88d3b477f5b7cd3fa0e02c90ad9bfaf0bd318013e8a22d76'
        assert second['selected_micro_auc']==.9938717710733546
        sample=np.sort(np.random.default_rng(4173).choice(2922*168,int(2922*168*.4),replace=False))
        buffer=io.BytesIO();np.save(buffer,sample,allow_pickle=False)
        assert hashlib.sha256(buffer.getvalue()).hexdigest()==p['sample']['sha256']==t['original_full_sample_sha256']
        assert p['anchor']['manifest_sha256']==t['anchor_manifest_sha256']
        assert p['resources']==v['resources'] and p['resources']['model_cpu_threads']==2
        preflight=ROOT/'artifacts/lgb-shop664-final-preflight-001/manifest.json'
        assert sha(preflight)=='6403e747ce247ebc33e1bf5ecffc3ee43e3b7dfb636263ab6aa400226082faf9'
        f=json.loads(preflight.read_text());assert f['status']=='passed_preflight_only' and f['protocol_sha256']==expected
        for name,h in f['artifacts'].items():assert sha(preflight.parent/name)==h
        r.update(status='passed',protocol_sha256=expected,template_audit_sha256=sha(template),
                 final_preflight_sha256=sha(preflight),accepted_validation=p['accepted_validation'],
                 selection_lock_sha256=p['selection_lock_sha256'],selected_weight=.1,
                 sample_sha256=p['sample']['sha256'],sample_rows=196358,
                 reviewed_source_unchanged=True,model_parameters=p['model_parameters'],
                 replay_auditor_sha256=sha(ROOT/'scripts/audit_lgb_shop664_final.py'),
                 scope='Actual final protocol/source/acceptance closure approved. Final fit still requires scheduling; actual all14 full-test replay and actual CSV audit precede submission.')
        shutil.copy2(path,out/'protocol.json')
    except BaseException:
        r.update(status='failed',error=traceback.format_exc());raise
    finally:
        shutil.copy2(__file__,out/Path(__file__).name)
        r.update(runtime_seconds=time.monotonic()-start,source_sha256=sha(Path(__file__)))
        (out/'manifest.json').write_text(json.dumps(r,indent=2)+'\n')
        print(json.dumps({'status':r['status'],'manifest_sha256':sha(out/'manifest.json')}),flush=True)


if __name__=='__main__':
    main()
