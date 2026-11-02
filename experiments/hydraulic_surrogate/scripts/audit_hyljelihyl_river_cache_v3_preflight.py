"""Review the minimal v3 no-retry repair and exact audited v2 input import."""
import ast
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.audit_hyljelihyl_river_cost_v2 import sha


def main() -> None:
    out=ROOT/'artifacts/hyljelihyl-river-full-cache-v3-independent-preflight-001';out.mkdir(exist_ok=False);start=time.monotonic()
    r={'status':'initializing','timestamp_utc':datetime.now(timezone.utc).isoformat(),'command':[sys.executable,*sys.argv],
       'optimization_calls':0,'labels_decoded':False,'training':False}
    try:
        pp=ROOT/'docs/hyljelihyl_river_full_cache_protocol_v3.json';p=json.loads(pp.read_text())
        assert sha(pp)=='e4b7bdf6954838739174f767a0a69ab110317fef069e646ac8725ab8cc9e187e'
        r['protocol_sha256']=sha(pp)
        for group in ['source','data']:
            for name,h in p[group].items():assert sha(ROOT/name)==h,name
        c=p['compatible_input_preflight']
        for name in ['protocol','manifest','input_audit','blocked_launch_review']:assert sha(ROOT/c[name])==c[name+'_sha256']
        old=json.loads((ROOT/c['protocol']).read_text());pre=json.loads((ROOT/c['manifest']).read_text());audit=json.loads((ROOT/c['input_audit']).read_text())
        assert c['protocol_sha256']=='cf75e04282b1731d7ec43d0cfcdc96815ff6f961766f79f6bb28e1cde0b3a356'
        assert c['manifest_sha256']=='569aee683a4a79ef2ed909f4c8ec5b747dd46105d4a49866a2df61880919b0b3'
        assert c['input_audit_sha256']=='68bc2d9fbfa19f390b09e55f2e41e7085270a1ecd3012e7d3f6337738367dc32'
        assert audit['status']=='passed'and audit['case_count']==3647 and audit['protocol_sha256']==c['protocol_sha256']
        assert audit['input_preflight_manifest_sha256']==c['manifest_sha256']
        assert json.loads((ROOT/c['blocked_launch_review']).read_text())['status']=='blocked_source_revision'
        assert pre['status']=='passed'and pre['optimization_calls']==0 and pre['protocol_sha256']==c['protocol_sha256']
        assert not pre['labels_decoded']and not pre['training']
        for name,h in pre['artifacts'].items():assert sha((ROOT/c['manifest']).parent/name)==h,name
        for name,h in audit['artifacts'].items():assert sha((ROOT/c['input_audit']).parent/name)==h,name
        ignored={'version','created_utc','source','resume','prior_unrun_preparation','compatible_input_preflight'}
        assert {k:v for k,v in p.items()if k not in ignored}=={k:v for k,v in old.items()if k not in ignored}
        for name,h in old['source'].items():assert p['source'][name]==h
        def functions(name):
            tree=ast.parse((ROOT/name).read_text());return {n.name:ast.dump(n,include_attributes=False)for n in tree.body if isinstance(n,ast.FunctionDef)}
        a=functions('scripts/build_hyljelihyl_river_cache_v2.py');b=functions('scripts/build_hyljelihyl_river_cache_v3.py')
        assert set(a)==set(b)
        changed={name for name in a if a[name]!=b[name]}
        assert changed=={'freeze','verify_protocol','publish_ready_pairs','retained_failures','verify_resume','main'}
        assert a['prepare_inputs']==b['prepare_inputs']and a['full_worker']==b['full_worker']
        # Call the frozen closure guard only; it performs no optimization/input rebuilding.
        from scripts.build_hyljelihyl_river_cache_v3 import verify_protocol
        verify_protocol(p,sha(pp))
        t=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_hyljelihyl_river_cache_v3.py'],cwd=ROOT,text=True,capture_output=True)
        (out/'tests.log').write_text(t.stdout+t.stderr);assert t.returncode==0,t.stderr
        r.update(status='passed',input_preflight_manifest_sha256=c['manifest_sha256'],input_preflight_protocol_sha256=c['protocol_sha256'],
            input_preflight_path=c['manifest'],case_count=3647,previous_input_audit_sha256=c['input_audit_sha256'],synthetic_tests_passed=8,
            source_delta='Only protocol/import guards, durable pair-failure preservation and resource-drain-only resume; unchanged input generator, worker/math/solver/features, receipt verification and resource thresholds.',
            repair_review='Every complete-pair publication exception writes immutable failure with exact proof/payload hash inventory and traceback; retained_failures checks it; parent final status failed and verify_resume blocks retained case/pair failures even if parent list omitted. Operational interruptions cannot use automatic resume.',
            compatibility_review='Full independent3647 reconstruction already passed under v2; exact v2 protocol/preflight/inputaudit/blockedreview and every actual preflight artifact are hash checked. Existing physical source/data/settings and all input/worker function ASTs are identical.',
            limitation='Source/input launch gate only; root authorization and independent actual complete-cache physical proof remain required.')
    except BaseException:r.update(status='failed',error=traceback.format_exc());raise
    finally:
        shutil.copy2(__file__,out/Path(__file__).name);r['runtime_seconds']=time.monotonic()-start
        r['artifacts']={f.name:sha(f)for f in out.iterdir()if f.is_file()and f.name!='manifest.json'}
        (out/'manifest.json').write_text(json.dumps(r,indent=2));print(json.dumps({'status':r['status'],'manifest_sha256':sha(out/'manifest.json'),'seconds':r['runtime_seconds']}),flush=True)

if __name__=='__main__':main()
