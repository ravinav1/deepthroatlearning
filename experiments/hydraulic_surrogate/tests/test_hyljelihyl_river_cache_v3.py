"""Cache eligibility, literal replay, interruption and immutable resume checks."""
import json
import multiprocessing as mp
from pathlib import Path
import tempfile
import time
import unittest
import numpy as np
from scripts.build_hyljelihyl_river_cache_v3 import (
    digest,write_json,verify_pair,publish_receipt,verify_receipt,verify_resume,drain_workers,retained_failures,
    publish_ready_pairs,verify_entry_resources)


def fake_worker(cache_name: str,number: int) -> None:
    """Finish the active synthetic write, then observe the coordinator drain marker."""
    cache=Path(cache_name);(cache/f'active_{number}').write_text('active')
    time.sleep(.15)
    (cache/f'finished_{number}').write_text('finished')
    while not (cache/'STOP_AFTER_ACTIVE').exists():time.sleep(.01)


class RiverCacheTest(unittest.TestCase):
    def pair(self,cache: Path) -> list[Path]:
        paths=[]
        for i in [1,2]:
            path=cache/f'imports/worker_{i}/case_0001';path.mkdir(parents=True);paths.append(path)
            np.savez_compressed(path/'result.npz',x=np.array([-0.,1.]))
            np.savez_compressed(path/'primal_dual.npz',x=np.array([0.,1.]))
            np.savez_compressed(path/'native_output.npz',x=np.array([0.,1.]))
            np.save(path/'features.npy',np.zeros((168,62),dtype=np.float32))
            write_json(path/'proof.json',{'case_id':1,'pid':100+i,'status':'complete','native':{'status':0},
                'kkt':{'bounds':0.},'all_slacks_zero':True,'artifacts':{f.name:digest(f)for f in path.iterdir()}},True)
        return paths

    def pilot_protocol(self,cache: Path,paths: list[Path]) -> tuple[dict,dict]:
        pilot=cache/'pilot';pilot.mkdir()
        for i in [1,2]:
            folder=pilot/f'worker_{i}';folder.mkdir();write_json(folder/'manifest.json',{'status':'complete','pid':100+i})
        write_json(pilot/'manifest.json',{'status':'passed','artifacts':{
            f'worker_{i}/case_0001/'+f.name:digest(f)for i,path in enumerate(paths,1)for f in path.iterdir()}})
        p={'pilot_case_ids':[1],'canonical_case_ids':[1],'pilot_directory':str(pilot),
            'pilot_manifest_sha256':digest(pilot/'manifest.json'),
            'pilot_worker_manifests_sha256':[digest(pilot/f'worker_{i}/manifest.json')for i in [1,2]]}
        origin={'type':'pilot_import','pilot_manifest_sha256':p['pilot_manifest_sha256'],
            'worker_manifests_sha256':p['pilot_worker_manifests_sha256'],'canonical_fresh_process_lineage':True}
        return p,origin

    def test_literal_replay_rejects_signed_zero_and_payload_changes(self):
        with tempfile.TemporaryDirectory()as temp:
            cache=Path(temp);paths=self.pair(cache);verify_pair(*paths,1)
            np.savez_compressed(paths[1]/'result.npz',x=np.array([0.,1.]))
            proof=json.loads((paths[1]/'proof.json').read_text());proof['artifacts']['result.npz']=digest(paths[1]/'result.npz')
            write_json(paths[1]/'proof.json',proof)
            with self.assertRaises(AssertionError):verify_pair(*paths,1)

    def test_immutable_receipt_and_resume_reject_changed_complete(self):
        with tempfile.TemporaryDirectory()as temp:
            cache=Path(temp);paths=self.pair(cache)
            p,origin=self.pilot_protocol(cache,paths)
            publish_receipt(cache,1,paths,'protocol',{'case_id':1},origin)
            verify_receipt(cache,1,'protocol',p,{'case_id':1})
            with self.assertRaises(ValueError):publish_receipt(cache,1,paths,'protocol',{'case_id':1},origin)
            receipt=cache/'cases/case_0001/complete.json';before=receipt.read_bytes()
            m={'status':'incomplete_resource_drain','all_workers_exited':True,'protocol_sha256':'protocol',
                'failures':[],'completed_cases':{'1':digest(receipt)},'artifacts':{str(f.relative_to(cache)):digest(f)for f in cache.rglob('*')if f.is_file()}}
            write_json(cache/'manifest.json',m);h=digest(cache/'manifest.json');verify_resume(cache,h,'protocol',p)
            self.assertEqual(before,receipt.read_bytes())
            (paths[0]/'features.npy').write_bytes(b'corrupted')
            with self.assertRaises(ValueError):verify_resume(cache,h,'protocol',p)

    def test_failed_case_blocks_resume_even_if_parent_omitted_failure(self):
        with tempfile.TemporaryDirectory()as temp:
            cache=Path(temp);failure=cache/'phases/phase_001/worker_1/case_0001/proof.json';failure.parent.mkdir(parents=True)
            write_json(failure,{'status':'failed','native':{'status':1},'error':'native time limit'})
            m={'status':'incomplete_operational','all_workers_exited':True,'protocol_sha256':'p','failures':[],'completed_cases':{},'artifacts':{}}
            write_json(cache/'manifest.json',m)
            self.assertEqual(len(retained_failures(cache)),1)
            with self.assertRaises(AssertionError):verify_resume(cache,digest(cache/'manifest.json'),'p',{})
            m['all_workers_exited']=False;write_json(cache/'manifest.json',m)
            with self.assertRaises(AssertionError):verify_resume(cache,digest(cache/'manifest.json'),'p',{})

    def test_drain_joins_children_and_preserves_complete_bytes(self):
        with tempfile.TemporaryDirectory()as temp:
            cache=Path(temp);receipt=cache/'complete.json';receipt.write_text('immutable complete evidence')
            before=receipt.read_bytes();ctx=mp.get_context('spawn')
            workers=[ctx.Process(target=fake_worker,args=(temp,i))for i in [1,2]]
            for worker in workers:worker.start()
            result=drain_workers(cache,workers,10.)
            self.assertTrue(result['all_workers_exited']);self.assertEqual([w.exitcode for w in workers],[0,0])
            self.assertEqual(before,receipt.read_bytes())
            self.assertTrue(all((cache/f'finished_{i}').exists()for i in [1,2]))

    def test_post_join_scan_publishes_last_pair_once(self):
        with tempfile.TemporaryDirectory()as temp:
            cache=Path(temp);phase=cache/'phases/phase_001';phase.mkdir(parents=True)
            complete={};inputs={1:{'case_id':1}};origin={'type':'phase','phase':1,'input_preflight_sha256':'preflight'}
            publish_ready_pairs(cache,phase,[1],complete,'p',inputs,origin)
            self.assertEqual(complete,{})
            for index,path in enumerate(self.pair(cache),1):
                dest=phase/f'worker_{index}/case_0001';dest.parent.mkdir();path.rename(dest)
                proof=json.loads((dest/'proof.json').read_text());proof.update(protocol_sha256='p',input_preflight_sha256='preflight',input_record=inputs[1])
                write_json(dest/'proof.json',proof)
            publish_ready_pairs(cache,phase,[1],complete,'p',inputs,origin)
            self.assertEqual(set(complete),{1});receipt=cache/'cases/case_0001/complete.json';before=receipt.read_bytes()
            publish_ready_pairs(cache,phase,[1],complete,'p',inputs,origin)
            self.assertEqual(before,receipt.read_bytes())

    def test_second_entry_reading_rechecks_memory_and_swap(self):
        p={'resource_entry_free_percent':30,'minimum_pressure_derived_headroom_bytes':3*1024**3}
        good={'free_percent':50,'pressure_derived_free_bytes':10*1024**3,'swapouts':0}
        verify_entry_resources(good,p,good)
        for field,value in [('free_percent',29),('pressure_derived_free_bytes',2*1024**3),('swapouts',1)]:
            bad={**good,field:value}
            with self.assertRaises(AssertionError):verify_entry_resources(bad,p,good)

    def test_receipt_pointer_and_pilot_lineage_tampering_rejected(self):
        with tempfile.TemporaryDirectory()as temp:
            cache=Path(temp);paths=self.pair(cache);p,origin=self.pilot_protocol(cache,paths)
            good=publish_receipt(cache,1,paths,'p',{'case_id':1},origin);receipt=cache/'cases/case_0001/complete.json'
            verify_receipt(cache,1,'p',p)
            edits=[('features_path','../elsewhere/features.npy'),('features_path',str((paths[0]/'features.npy').resolve())),
                ('features_path',str((paths[1]/'features.npy').relative_to(cache))),('replica_paths',[good['replica_paths'][0]]),
                ('replica_paths',['../../escape',good['replica_paths'][1]]),('features_sha256','wrong')]
            for field,value in edits:
                bad={**good,field:value};write_json(receipt,bad)
                with self.assertRaises(AssertionError):verify_receipt(cache,1,'p',p)
            write_json(receipt,good)
            with self.assertRaises(AssertionError):verify_receipt(cache,1,'p',{**p,'pilot_case_ids':[]})
            bad={**good,'origin':{**origin,'pilot_manifest_sha256':'wrong'}};write_json(receipt,bad)
            with self.assertRaises(AssertionError):verify_receipt(cache,1,'p',p)

    def test_complete_worker_pair_mismatch_is_durable_and_blocks_resume(self):
        with tempfile.TemporaryDirectory()as temp:
            cache=Path(temp);phase=cache/'phases/phase_001';phase.mkdir(parents=True)
            for index,path in enumerate(self.pair(cache),1):
                dest=phase/f'worker_{index}/case_0001';dest.parent.mkdir();path.rename(dest)
                if index==2:np.savez_compressed(dest/'result.npz',x=np.array([0.,1.]))
                proof=json.loads((dest/'proof.json').read_text());proof.update(protocol_sha256='p',input_preflight_sha256='pre',input_record={'case_id':1})
                proof['artifacts']['result.npz']=digest(dest/'result.npz');write_json(dest/'proof.json',proof)
            with self.assertRaises(AssertionError):
                publish_ready_pairs(cache,phase,[1],{},'p',{1:{'case_id':1}},{'type':'phase','phase':1,'input_preflight_sha256':'pre'})
            failures=retained_failures(cache);self.assertEqual(len(failures),1);self.assertEqual(failures[0]['case_id'],1)
            for status in ['incomplete_resource_drain','incomplete_operational']:
                m={'status':status,'all_workers_exited':True,'protocol_sha256':'p','failures':[],'completed_cases':{},'artifacts':{}}
                write_json(cache/'manifest.json',m)
                with self.assertRaises(AssertionError):verify_resume(cache,digest(cache/'manifest.json'),'p',{})


if __name__=='__main__':unittest.main()
