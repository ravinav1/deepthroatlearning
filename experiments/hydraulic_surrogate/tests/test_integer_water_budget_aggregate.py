"""Eligibility and preservation boundaries for the separate numerical-exception aggregate."""
from copy import deepcopy
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json
import tempfile
import unittest
import pandas as pd
from sintef_project.integer_water_budget_aggregate_features import (
    COMPLETION_SHA,EXCEPTION_AUDIT_SHA,validate_aggregate_manifest,add_integer_water_features)
from sintef_project.integer_water_budget_features import feature_specification
from scripts.build_integer_water_budget_aggregate import copy_directory_exact,inventory,validate_original_final,verify_original_exit,EXIT_COMMAND


def fixture():
    ids=list(range(1,3648));imports=list(range(1,637));ordinary=[cid for cid in ids if cid not in imports and cid!=2012]
    p={'version':'integer_water_budget_aggregate_v1','case_ids':ids,'case_dates':['metadata']*3647,
        'imported_case_ids':imports,'exception_case_ids':[2012],'source':{},'data':{},'environment':{},
        'feature_specification':feature_specification(),'prefix_columns':[],
        'completion_manifest_sha256':COMPLETION_SHA,'exception_actual_audit_sha256':EXCEPTION_AUDIT_SHA,
        'original_protocol_sha256':'original'}
    m={**p,'status':'complete','all_cases_accepted_and_replayed':True,'all_cases_native_strict_gate_passed':False,
        'original_cache_still_ineligible':True,'failures':[],'completed_cases':{str(cid):'receipt'for cid in ids},
        'ordinary_case_ids':ordinary,'ordinary_case_count':3010,'imported_case_count':636,'exception_count':1}
    return m,p


class AggregateTest(unittest.TestCase):
    def test_complete_disjoint_population_and_exact_exception_required(self):
        m,p=fixture();validate_aggregate_manifest(m,p)
        broken=[]
        item=deepcopy(m);del item['completed_cases']['3647'];broken.append(item)
        item=deepcopy(m);item['ordinary_case_ids'][0]=1;broken.append(item)
        item=deepcopy(m);item['exception_case_ids']=[2013];broken.append(item)
        item=deepcopy(m);item['all_cases_native_strict_gate_passed']=True;broken.append(item)
        item=deepcopy(m);item['original_cache_still_ineligible']=False;broken.append(item)
        item=deepcopy(m);item['completion_manifest_sha256']='another result';broken.append(item)
        item=deepcopy(m);item['status']='passed_preflight_only';broken.append(item)
        for item in broken:
            with self.assertRaises(ValueError):validate_aggregate_manifest(item,p)

    def test_direct_loader_rejects_missing_external_audit_before_features(self):
        with self.assertRaisesRegex(ValueError,'Explicit independent aggregate audit'):
            add_integer_water_features(pd.DataFrame(),Path('/missing'),Path('/missing'),'manifest','protocol')

    def test_copy_preserves_original_failed_receipt_and_rejects_inventory_change(self):
        with tempfile.TemporaryDirectory()as temp:
            root=Path(temp);source=root/'source';source.mkdir();(source/'proof.json').write_text('{"status":"failed"}')
            (source/'payload.bin').write_bytes(bytes(range(256)));before=inventory(source)
            copy_directory_exact(source,root/'copy',before)
            self.assertEqual(before,inventory(source));self.assertEqual(before,inventory(root/'copy'))
            self.assertEqual((root/'copy/proof.json').read_bytes(),(source/'proof.json').read_bytes())
            (source/'payload.bin').write_bytes(b'changed')
            with self.assertRaises(ValueError):copy_directory_exact(source,root/'another',before)
            (source/'alias').symlink_to(source/'proof.json')
            with self.assertRaises(ValueError):inventory(source)

    def test_original_must_be_finalized_with_only_two_declared_failed_attempts(self):
        from scripts.build_integer_water_budget_aggregate import IMPORT_SHA
        _,p=fixture()
        old={'status':'incomplete','protocol_sha256':'original','remaining_queued_attempts':0,'graceful_drain':{'requested':False},
            'case_ids':p['case_ids'],'case_dates':p['case_dates'],'completed_cases':{str(cid):'receipt'for cid in p['case_ids']if cid!=2012},
            'failures':[{'case_id':2012,'replica':r,'stage':'attempt_ineligible'}for r in [1,2]],
            'resumed_case_ids':p['imported_case_ids'],'reuse_manifest_sha256':IMPORT_SHA,'labels_decoded':False,'training':False}
        validate_original_final(old,p)
        for field,value in [('status','building_input_only'),('remaining_queued_attempts',1),('graceful_drain',{'requested':True}),
                            ('failures',[{'case_id':2013,'replica':r,'stage':'attempt_ineligible'}for r in [1,2]])]:
            bad={**old,field:value}
            with self.assertRaises(AssertionError):validate_original_final(bad,p)

    def test_exit_requires_exact_pid_lineage_real_scoped_output_and_late_timestamp(self):
        with tempfile.TemporaryDirectory()as temp:
            cache=Path(temp);attempt=cache/'attempt';attempt.mkdir();(attempt/'proof.json').write_text(json.dumps({'pid':123}))
            now=datetime.now(timezone.utc)
            m={'attempts':[{'directory':'attempt'}],'timestamp_utc':now.isoformat(),'runtime_seconds':1.}
            p={'original_cache':str(cache),'original_protocol_sha256':'p','original_coordinator_pid':122,'original_exit_process_command':EXIT_COMMAND}
            o={'status':'passed','original_cache_manifest_sha256':'manifest','original_cache_protocol_sha256':'p',
                'original_cache_path':str(cache),'expected_worker_pids':[123],'expected_coordinator_pid':122,
                'command':EXIT_COMMAND,'returncode':1,'scoped_process_output':'','timestamp_utc':(now+timedelta(seconds=2)).isoformat(),
                'coordinator_exited':True,'all_workers_exited':True}
            verify_original_exit(o,m,p,'manifest')
            for field,value in [('expected_worker_pids',[]),('timestamp_utc',now.isoformat()),('returncode',2),
                                ('original_cache_manifest_sha256','wrong')]:
                with self.assertRaises(AssertionError):verify_original_exit({**o,field:value},m,p,'manifest')
            with self.assertRaises(AssertionError):
                verify_original_exit({**o,'returncode':0,'scoped_process_output':'123 multiprocessing.spawn\n'},m,p,'manifest')


if __name__=='__main__':unittest.main()
