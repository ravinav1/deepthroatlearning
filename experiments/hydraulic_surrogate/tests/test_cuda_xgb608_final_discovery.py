"""Ensure final dataset discovery requires exact unique code and bundle identities."""
from pathlib import Path
import shutil
import tempfile
import unittest

from scripts.launch_cuda_xgb608_final_dataset import discover

ROOT=Path(__file__).resolve().parents[1]


class FinalDiscoveryTests(unittest.TestCase):
    def make_input(self,root:Path)->Path:
        code=root/'private-final-dataset'/'code'
        shutil.copytree(ROOT/'artifacts/cuda-xgb608-final-package-001/payload',code)
        bundle=root/'original-private-input';bundle.mkdir()
        shutil.copy2(ROOT/'artifacts/kaggle-neural-bundle-001/payload/bundle_manifest.json',bundle/'bundle_manifest.json')
        extra=root/'private608-supplement';extra.mkdir()
        shutil.copy2(ROOT/'artifacts/cuda-xgb608-package-001/supplement/bundle_manifest.json',extra/'bundle_manifest.json')
        return code

    def test_discovers_extracted_bundle_and_rejects_code_change(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);code=self.make_input(root)
            paths=discover(root)
            self.assertEqual(paths['code'],code.resolve())
            self.assertEqual(paths['bundle'],(root/'original-private-input').resolve())
            self.assertEqual(paths['supplement'],(root/'private608-supplement').resolve())
            with (code/'train_cuda_xgb608_final.py').open('a') as stream:stream.write('\n# changed\n')
            with self.assertRaisesRegex(ValueError,'member changed'):
                discover(root)

    def test_duplicate_protocol_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);code=self.make_input(root)
            other=root/'duplicate';other.mkdir()
            shutil.copy2(code/'cuda_xgb608_final_protocol.json',other/'cuda_xgb608_final_protocol.json')
            with self.assertRaisesRegex(ValueError,'found 2'):
                discover(root)

    def test_missing_supplement_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);self.make_input(root)
            (root/'private608-supplement/bundle_manifest.json').unlink()
            with self.assertRaisesRegex(ValueError,'found 0'):
                discover(root)


if __name__=='__main__':unittest.main()
