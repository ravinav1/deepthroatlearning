"""Independent source/fixture review of the full integer-cache physical auditor."""
from datetime import datetime, timezone
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_integer_water_budget_cache import (
    arrays, audit_case, digest, independent_features, HydroCaseData,
    ShopRelaxationLP, verify_hashes, BASE_SHA, PROTOCOL_SHA, np,
)


def main() -> None:
    """Exercise complete pilot payloads only; never read the running full cache."""
    out=ROOT/'artifacts/integer-cache-auditor-independent-source-review-001'
    out.mkdir(exist_ok=False)
    start=time.monotonic()
    report={'status':'initializing','timestamp_utc':datetime.now(timezone.utc).isoformat(),
            'optimization_calls':0,'live_cache_read':False,'partial_cache_feature_read':False,
            'full_cache_approved':False,'labels_decoded':False,'training':False,
            'command':[sys.executable,*sys.argv]}
    try:
        source=ROOT/'scripts/audit_integer_water_budget_cache.py'
        assert digest(source)=='78ff7efe84b00129bafa112c9a2a6be8149100f99f456313af8178788f9660e9'
        preflight=ROOT/'artifacts/integer-cache-auditor-pilot-preflight-002/manifest.json'
        assert digest(preflight)=='a34def0aac8bd6bce02abd4e22175c0b90020f7f58534eec593cba633df82cb0'
        pm=json.loads(preflight.read_text())
        assert pm['status']=='passed' and len(pm['cases'])==24
        assert pm['all_24_cases_physical_and_replayed'] and all(c['rejected'] for c in pm['corruption_checks'])
        for name,expected in pm['source'].items():
            assert digest(ROOT/name)==expected and digest(preflight.parent/'source'/name)==expected
            dest=out/'reviewed_source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
        ppath=ROOT/'docs/integer_water_budget_four_worker_protocol_v2.json'
        assert digest(ppath)==PROTOCOL_SHA
        p=json.loads(ppath.read_text())
        verify_hashes(ROOT,p['source']);verify_hashes(ROOT,p['data'])
        pilot=ROOT/p['pilot_directory']
        assert digest(pilot/'manifest.json')==p['pilot_manifest_sha256']
        verify_hashes(pilot,p['pilot_files'])
        # Construct a clearly labelled reviewer fixture from the finalized pilot.
        # No source case receipt or feature file from either full-cache run is read.
        folder=out/'reviewer_fixture_case_0001';folder.mkdir()
        ip=pilot/'case_0001_inputs.npz';shutil.copy2(ip,folder/'inputs.npz')
        case=arrays(ip)
        continuous=ROOT/p['continuous_cache']
        assert digest(continuous/'manifest.json')==p['continuous_cache_manifest_sha256']
        cm=json.loads((continuous/'manifest.json').read_text());cp=continuous/'case_0001_result.npz'
        assert digest(cp)==cm['artifacts'][cp.name]
        control=arrays(cp)
        # HydroCaseData's loader reads historical target metadata only.
        from unittest.mock import patch
        import pandas as pd
        target=(ROOT/'sintef_project/data/kernel/Unit_commitment_decisions.csv').resolve()
        original_read=pd.read_csv
        def metadata_only(path,*args,**kwargs):
            if isinstance(path,(str,Path)) and Path(path).resolve()==target:
                assert not args and kwargs=={'usecols':['Run No','starttime']}
            return original_read(path,*args,**kwargs)
        with patch.object(pd,'read_csv',metadata_only):
            data=HydroCaseData(ROOT)
        solver=ShopRelaxationLP(data.network)
        for replica in [1,2]:
            shutil.copytree(pilot/f'case_0001/attempt_{replica}',folder/f'pilot_replica_{replica}')
        result=arrays(folder/'pilot_replica_1/result.npz')
        shutil.copy2(folder/'pilot_replica_1/result.npz',folder/'canonical_result.npz')
        values=independent_features(case,result,control,data.network)
        np.save(folder/'features.npy',values,allow_pickle=False)
        receipt={'status':'complete','case_id':1,'input_sha256':digest(ip),'control_sha256':digest(cp),
                 'accepted_attempts':['pilot_replica_1','pilot_replica_2'],
                 'artifacts':{f.relative_to(folder).as_posix():digest(f) for f in folder.rglob('*') if f.is_file()}}
        rebuilt,checked=audit_case(folder,receipt,case,control,solver,BASE_SHA,p)
        assert rebuilt.tobytes()==values.tobytes() and checked['all_replays_bitwise']
        mutation=copy.deepcopy(receipt);mutation['accepted_attempts']=['pilot_replica_1','pilot_replica_1']
        try:
            audit_case(folder,mutation,case,control,solver,BASE_SHA,p)
        except ValueError as e:
            assert 'Duplicate' in str(e)
        else:
            raise AssertionError('Duplicate replica was accepted')
        changed=copy.deepcopy(p);changed['pilot_case_ids']=[]
        try:
            audit_case(folder,receipt,case,control,solver,BASE_SHA,changed)
        except ValueError as e:
            assert 'Unapproved pilot' in str(e)
        else:
            raise AssertionError('Unapproved pilot case was accepted')
        tests=subprocess.run([sys.executable,'-m','unittest','tests.test_integer_cache_physical_audit'],cwd=ROOT,capture_output=True,text=True)
        (out/'tests.txt').write_text(tests.stdout+tests.stderr);assert tests.returncode==0
        report.update(status='passed_source_review_only',auditor_source_sha256=digest(source),
                      protocol_sha256=PROTOCOL_SHA,preflight_sha256=digest(preflight),
                      pilot_manifest_sha256=p['pilot_manifest_sha256'],focused_tests_passed=4,
                      duplicate_replica_rejected=True,unapproved_pilot_rejected=True,
                      completed_pilot_case_receipt_fixture=checked,
                      review=['Complete3647case gate precedes derived-feature reads; all original source/data/environment/prefix hashes are checked.',
                              'Original copied reuse manifest and636compatible receipt/payload bytes are tied to the externally fixed original manifest.',
                              'All12canonical diagnostic source/result/proof bytes and origins, and24pilot original proof/result/LP bytes are pinned explicitly.',
                              'Two separate process replicas and four LP payloads are replayed; retry60to600seconds is native-status1-only with identical input/protocol/environment and earlier timestamp.',
                              'Frozen physical matrices are rebuilt without optimization. Primal feasibility, integrality, bounds, objectives, LP dual signs/stationarity/complementarity and strong duality are independently checked.',
                              'Native status0 supplies the MILP optimality statement; no extra literal-zero reported-gap criterion is introduced.',
                              'All84features are independently reconstructed; original formula/order/dtype is preserved and final matrix bytes must agree.'],
                      limitations='Source and completed-pilot fixture approval only. The actual3647case cache must finish and pass the complete auditor before any feature package or model fit.')
    except BaseException:
        report.update(status='failed',error=traceback.format_exc());raise
    finally:
        shutil.copy2(__file__,out/Path(__file__).name)
        report.update(runtime_seconds=time.monotonic()-start,source_sha256=digest(Path(__file__)))
        report['artifacts']={f.relative_to(out).as_posix():digest(f) for f in out.rglob('*') if f.is_file()}
        (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'status':report['status'],'manifest_sha256':digest(out/'manifest.json')}),flush=True)


if __name__=='__main__':
    main()
