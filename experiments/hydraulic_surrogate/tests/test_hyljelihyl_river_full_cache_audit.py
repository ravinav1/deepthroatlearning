"""Independent cache audit rejects plausible but incorrect execution lineage."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from scripts.audit_hyljelihyl_river_full_cache import safe,sha,receipt_paths,phase_lineage,pilot_lineage,check_resource


class FullRiverAuditTest(unittest.TestCase):
    def test_paths_reject_alias_escape_and_absolute(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp)
            self.assertEqual(safe(root,'phases/a.json'),root/'phases/a.json')
            (root/'real').mkdir();(root/'alias').symlink_to(root/'real',target_is_directory=True)
            with self.assertRaises(AssertionError):safe(root,'alias/a.json')
            for name in ['../a','/a','phases/../a','phases//a','./a','']:
                with self.assertRaises(AssertionError):safe(root,name)

    def test_phase_binding_rejects_swapped_replicas_pid_order_and_seal(self):
        r={'origin':{'type':'phase','phase':2,'input_preflight_sha256':'input'},'pids':[11,12],
           'replica_paths':['phases/phase_002/worker_1/case_0003','phases/phase_002/worker_2/case_0003'],
           'input_record':{'case_id':3}}
        launch={'protocol_sha256':'p','input_preflight_sha256':'input','worker_pids':[11,12],'orders':[[3,4],[4,3]]}
        proofs=[{'pid':pid,'protocol_sha256':'p','input_preflight_sha256':'input','input_record':r['input_record']}for pid in [11,12]]
        phase_lineage(r,3,launch,2,proofs,'input','p')
        bad=copy.deepcopy(r);bad['replica_paths'].reverse()
        with self.assertRaises(AssertionError):phase_lineage(bad,3,launch,2,proofs,'input','p')
        for field,value in [('pid',99),('input_preflight_sha256','other'),('input_record',{'case_id':4})]:
            badproof=copy.deepcopy(proofs);badproof[0][field]=value
            with self.assertRaises(AssertionError):phase_lineage(r,3,launch,2,badproof,'input','p')
        badlaunch=copy.deepcopy(launch);badlaunch['orders'][1]=[4]
        with self.assertRaises(AssertionError):phase_lineage(r,3,badlaunch,2,proofs,'input','p')

    def test_receipt_pins_exact_payload_and_first_feature(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);names=['imports/worker_1/case_0001','imports/worker_2/case_0001']
            for name in names:
                folder=root/name;folder.mkdir(parents=True);(folder/'features.npy').write_bytes(b'feature');(folder/'proof.json').write_text('{}')
            r={'status':'complete','case_id':1,'protocol_sha256':'p','input_record':{'case_id':1},
               'replica_paths':names,'features_path':names[0]+'/features.npy','features_sha256':sha(root/names[0]/'features.npy'),
               'pids':[11,12],'replays_byte_exact':True,
               'artifacts':{str(f.relative_to(root)):sha(f)for name in names for f in (root/name).iterdir()}}
            receipt_paths(root,r,1,'p',{'case_id':1})
            for field,value in [('features_path',names[1]+'/features.npy'),('pids',[11,11]),('replica_paths',[names[0],names[0]])]:
                bad={**r,field:value}
                with self.assertRaises(AssertionError):receipt_paths(root,bad,1,'p',{'case_id':1})
            (root/names[0]/'unlisted').write_text('extra')
            with self.assertRaises(AssertionError):receipt_paths(root,r,1,'p',{'case_id':1})

    def test_pilot_allowlist_origin_and_exact_payload(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);names=['imports/worker_1/case_0001','imports/worker_2/case_0001'];artifacts={}
            for i,name in enumerate(names,1):
                folder=root/name;folder.mkdir(parents=True);(folder/'proof.json').write_text(json.dumps({'pid':i}))
                artifacts[f'worker_{i}/case_0001/proof.json']=sha(folder/'proof.json')
            p={'pilot_case_ids':[1],'pilot_manifest_sha256':'pilot','pilot_worker_manifests_sha256':['a','b'],'canonical_case_ids':[1]}
            r={'replica_paths':names,'pids':[1,2],'origin':{'type':'pilot_import','pilot_manifest_sha256':'pilot',
                'worker_manifests_sha256':['a','b'],'canonical_fresh_process_lineage':True}}
            pilot_lineage(root,r,1,p,{'artifacts':artifacts,'worker_pids':[1,2]})
            with self.assertRaises(AssertionError):pilot_lineage(root,r,2,p,{'artifacts':artifacts,'worker_pids':[1,2]})
            bad=copy.deepcopy(r);bad['pids'].reverse()
            with self.assertRaises(AssertionError):pilot_lineage(root,bad,1,p,{'artifacts':artifacts,'worker_pids':[1,2]})
            bad=copy.deepcopy(r);bad['origin']['canonical_fresh_process_lineage']=False
            with self.assertRaises(AssertionError):pilot_lineage(root,bad,1,p,{'artifacts':artifacts,'worker_pids':[1,2]})
            (root/names[1]/'proof.json').write_text('different')
            with self.assertRaises(AssertionError):pilot_lineage(root,r,1,p,{'artifacts':artifacts,'worker_pids':[1,2]})

    def test_memory_and_swap_requirements_are_independent(self):
        good={'free_percent':30,'pressure_derived_free_bytes':3*1024**3,'swapouts':0};check_resource(good,30,0)
        for field,value in [('free_percent',29.9),('pressure_derived_free_bytes',3*1024**3-1),('swapouts',1)]:
            with self.assertRaises(AssertionError):check_resource({**good,field:value},30,0)

if __name__=='__main__':unittest.main()
