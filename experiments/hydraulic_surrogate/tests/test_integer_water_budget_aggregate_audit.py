"""Independent aggregate audit rejects provenance relabelling and payload changes."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from scripts.audit_integer_water_budget_aggregate import groups,safe,digest,compatibility,BASE_SHA,OLD_SHA,IMPORT_SHA,exit_contract


class IntegerAggregateAuditTest(unittest.TestCase):
    def population(self):
        ids=list(range(1,3648));imports=list(range(1,637));ordinary=[i for i in ids if i not in imports and i!=2012]
        p={'version':'integer_water_budget_aggregate_v1','case_ids':ids,'imported_case_ids':imports,'exception_case_ids':[2012]}
        m={'ordinary_case_ids':ordinary,'imported_case_ids':imports,'exception_case_ids':[2012],
           'ordinary_case_count':3010,'imported_case_count':636,'exception_count':1}
        return m,p

    def test_exact_disjoint_3010_636_one_exception(self):
        m,p=self.population();self.assertEqual(groups(m,p)['exception'],{2012})
        for kind in ['ordinary','imported']:
            bad=copy.deepcopy(m);bad[kind+'_case_ids'][0]=2012
            with self.assertRaises(AssertionError):groups(bad,p)
        for replacement in [[2013],[2012,2013],[]]:
            bad={**m,'exception_case_ids':replacement}
            with self.assertRaises(AssertionError):groups(bad,p)
        bad=copy.deepcopy(m);bad['ordinary_case_ids'][-1]=bad['ordinary_case_ids'][0]
        with self.assertRaises(AssertionError):groups(bad,p)

    def test_metadata_membership_and_declared_counts_not_just_lengths(self):
        m,p=self.population();bad=copy.deepcopy(p);bad['case_ids'][-1]=9999
        with self.assertRaises(AssertionError):groups(m,bad)
        bad={**m,'ordinary_case_count':3011}
        with self.assertRaises(AssertionError):groups(bad,p)
        bad={**p,'version':'ordinary_cache'}
        with self.assertRaises(AssertionError):groups(m,bad)

    def test_safe_relative_paths_reject_aliases_and_symlinks(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);(root/'real').mkdir();(root/'link').symlink_to(root/'real',target_is_directory=True)
            self.assertEqual(safe(root,'original_case/complete.json'),root/'original_case/complete.json')
            for name in ['/tmp/a','../escape','a/../b','a//b','./a','link/file']:
                with self.assertRaises(AssertionError):safe(root,name)

    def test_exit_evidence_is_actual_complete_bound_and_post_run(self):
        p={'original_cache':'artifacts/original','original_coordinator_pid':11014,
           'original_exit_process_command':"pgrep -fl 'multiprocessing.spawn|continue_integer_water_budget_cache_v2'"}
        m={'timestamp_utc':'2026-11-02T20:00:00+00:00','runtime_seconds':60.}
        o={'status':'passed','original_cache_manifest_sha256':'hash','original_cache_protocol_sha256':OLD_SHA,
           'original_cache_path':p['original_cache'],'expected_coordinator_pid':11014,'expected_worker_pids':[2,3],
           'command':p['original_exit_process_command'],'returncode':1,'scoped_process_output':'',
           'timestamp_utc':'2026-11-02T20:01:01+00:00','coordinator_exited':True,'all_workers_exited':True}
        exit_contract(o,m,p,'hash',[3,2,2])
        for key,value in [('original_cache_manifest_sha256','other'),('timestamp_utc','2026-11-02T20:00:59+00:00'),
                          ('expected_worker_pids',[2]),('returncode',0),('coordinator_exited',False)]:
            with self.assertRaises(AssertionError):exit_contract({**o,key:value},m,p,'hash',[2,3])
        with self.assertRaises(AssertionError):exit_contract({**o,'returncode':0,'scoped_process_output':'3 spawn'},m,p,'hash',[2,3])

    def test_unchanged_636_compatibility_transform_and_payload(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);original=root/'old/cases/case_0001';original.mkdir(parents=True)
            folder=root/'copied';folder.mkdir();(original/'result').write_bytes(b'original')
            (folder/'result').write_bytes(b'original')
            old={'status':'complete','protocol_sha256':BASE_SHA,'artifacts':{'result':digest(original/'result')}}
            (original/'complete.json').write_text(json.dumps(old));h=digest(original/'complete.json')
            (folder/'compatible_original_complete.json').write_bytes((original/'complete.json').read_bytes())
            c={'original_protocol_sha256':BASE_SHA,'original_manifest_sha256':IMPORT_SHA,
               'original_complete_sha256':h,'original_complete_path':'compatible_original_complete.json',
               'all_original_payload_bytes_preserved':True}
            r=copy.deepcopy(old);r['protocol_sha256']=OLD_SHA;r['compatibility']=c;r['artifacts']['compatible_original_complete.json']=h
            compatibility(folder,r,1,root/'old',{'completed_cases':{'1':h}})
            bad=copy.deepcopy(r);bad['status']='accepted'
            with self.assertRaises(AssertionError):compatibility(folder,bad,1,root/'old',{'completed_cases':{'1':h}})
            (folder/'result').write_bytes(b'changed')
            with self.assertRaises(AssertionError):compatibility(folder,r,1,root/'old',{'completed_cases':{'1':h}})

if __name__=='__main__':unittest.main()
