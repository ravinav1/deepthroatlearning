"""Cache eligibility, literal replay, interruption and immutable resume checks."""
import json
import multiprocessing as mp
from pathlib import Path
import tempfile
import time
import unittest
import numpy as np
from scripts.build_hyljelihyl_river_cache import (
    digest,write_json,verify_pair,publish_receipt,verify_receipt,verify_resume,drain_workers,retained_failures)


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
            path=cache/f'replica_{i}';path.mkdir();paths.append(path)
            np.savez_compressed(path/'result.npz',x=np.array([-0.,1.]))
            np.savez_compressed(path/'primal_dual.npz',x=np.array([0.,1.]))
            np.savez_compressed(path/'native_output.npz',x=np.array([0.,1.]))
            np.save(path/'features.npy',np.zeros((168,62),dtype=np.float32))
            write_json(path/'proof.json',{'case_id':1,'pid':100+i,'status':'complete','native':{'status':0},
                'kkt':{'bounds':0.},'all_slacks_zero':True,'artifacts':{f.name:digest(f)for f in path.iterdir()}},True)
        return paths

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
            publish_receipt(cache,1,paths,'protocol',{'case_id':1},{'type':'pilot_import'})
            verify_receipt(cache,1,'protocol',{'case_id':1})
            with self.assertRaises(ValueError):publish_receipt(cache,1,paths,'protocol',{'case_id':1},{'type':'pilot_import'})
            receipt=cache/'cases/case_0001/complete.json';before=receipt.read_bytes()
            m={'status':'incomplete_resource_drain','all_workers_exited':True,'protocol_sha256':'protocol',
                'failures':[],'completed_cases':{'1':digest(receipt)},'artifacts':{str(f.relative_to(cache)):digest(f)for f in cache.rglob('*')if f.is_file()}}
            write_json(cache/'manifest.json',m);h=digest(cache/'manifest.json');verify_resume(cache,h,'protocol')
            self.assertEqual(before,receipt.read_bytes())
            (paths[0]/'features.npy').write_bytes(b'corrupted')
            with self.assertRaises(ValueError):verify_resume(cache,h,'protocol')

    def test_failed_case_blocks_resume_even_if_parent_omitted_failure(self):
        with tempfile.TemporaryDirectory()as temp:
            cache=Path(temp);failure=cache/'phases/phase_001/worker_1/case_0001/proof.json';failure.parent.mkdir(parents=True)
            write_json(failure,{'status':'failed','native':{'status':1},'error':'native time limit'})
            m={'status':'incomplete_operational','all_workers_exited':True,'protocol_sha256':'p','failures':[],'completed_cases':{},'artifacts':{}}
            write_json(cache/'manifest.json',m)
            self.assertEqual(len(retained_failures(cache)),1)
            with self.assertRaises(AssertionError):verify_resume(cache,digest(cache/'manifest.json'),'p')
            m['all_workers_exited']=False;write_json(cache/'manifest.json',m)
            with self.assertRaises(AssertionError):verify_resume(cache,digest(cache/'manifest.json'),'p')

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


if __name__=='__main__':unittest.main()
