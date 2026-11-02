"""Strict mathematical compatibility and byte-preserved old case import checks."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from scripts.continue_integer_water_budget_cache_v2 import check_compatibility,import_complete_case,digest,BASE_PROTOCOL_SHA,EXECUTION_SOURCE

class ContinuationTests(unittest.TestCase):
    def protocols(self):
        old=json.loads(Path('docs/integer_water_budget_full_cache_protocol.json').read_text())
        new=copy.deepcopy(old);new.update(version='new',created_utc='new',workers=4,resume_policy='compatible',
            compatible_base={'path':'docs/integer_water_budget_full_cache_protocol.json','sha256':BASE_PROTOCOL_SHA},execution={})
        new['source'].update({p:'new-source-pin'for p in EXECUTION_SOURCE})
        return old,new
    def test_physical_fields_cannot_change(self):
        old,new=self.protocols();check_compatibility(old,new)
        for field,value in [('first_milp_seconds',61.),('retry_milp_seconds',601.),('solver_threads',2),('feature_specification',{}),('case_ids',[1])]:
            bad=copy.deepcopy(new);bad[field]=value
            with self.assertRaises(ValueError):check_compatibility(old,bad)
        bad=copy.deepcopy(new);bad['source']['sintef_project/integer_water_budget.py']='changed'
        with self.assertRaises(ValueError):check_compatibility(old,bad)
    def test_real_complete_case_bytes_are_preserved(self):
        root=Path('artifacts/integer-water-budget-full-cache-preflight-001/cases')
        source=next(root.glob('case_*'));old=json.loads((source/'complete.json').read_text())
        with tempfile.TemporaryDirectory()as tmp:
            dest=Path(tmp)/source.name
            receipt=import_complete_case(source,dest,BASE_PROTOCOL_SHA,'f'*64,'e'*64)
            self.assertEqual((dest/'compatible_original_complete.json').read_bytes(),(source/'complete.json').read_bytes())
            self.assertEqual(receipt['protocol_sha256'],'f'*64)
            for name,h in old['artifacts'].items():self.assertEqual(digest(dest/name),h)
            self.assertEqual(receipt['compatibility']['original_complete_sha256'],digest(source/'complete.json'))
            (dest/'features.npy').write_bytes(b'bad')
            from scripts.build_integer_water_budget_cache import verify_complete_case
            with self.assertRaises(ValueError):verify_complete_case(dest,'f'*64)
if __name__=='__main__':unittest.main()
