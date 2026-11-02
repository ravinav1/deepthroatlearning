"""Exact completed-case reuse rejects changed, missing and extra payloads."""

import json
from pathlib import Path
import tempfile
import unittest

from scripts.diagnose_hyljelihyl_river_cost_36 import reuse_case, digest


class RiverReuseTests(unittest.TestCase):
    def fixture(self, root):
        source=root/'source';folder=source/'case_0001'/'A_1';folder.mkdir(parents=True)
        (folder/'proof.json').write_bytes(b'{"status":"complete"}\n')
        (source/'case_0001'/'input.npz').write_bytes(b'unchanged synthetic payload\x00')
        manifest={'artifacts':{str(p.relative_to(source)):digest(p) for p in source.rglob('*') if p.is_file()}}
        (source/'manifest.json').write_text(json.dumps(manifest))
        return source,manifest

    def test_copy_preserves_every_original_byte(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source,manifest=self.fixture(root)
            receipt=reuse_case(source,root/'copy',1,manifest)
            self.assertTrue(receipt['all_original_bytes_preserved'])
            self.assertEqual(receipt['original_manifest_sha256'],digest(source/'manifest.json'))
            for name,h in receipt['case_payload_hashes'].items():self.assertEqual(digest(root/'copy'/name),h)

    def test_tampered_missing_and_extra_payload_rejected(self):
        for mutation in ['change','missing','extra']:
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);source,manifest=self.fixture(root);target=source/'case_0001'/'input.npz'
                if mutation=='change':target.write_bytes(b'changed')
                elif mutation=='missing':target.unlink()
                else:(source/'case_0001'/'unexpected').write_bytes(b'extra')
                with self.assertRaises(AssertionError):reuse_case(source,root/'copy',1,manifest)
                self.assertFalse((root/'copy').exists())


if __name__=='__main__':unittest.main()
