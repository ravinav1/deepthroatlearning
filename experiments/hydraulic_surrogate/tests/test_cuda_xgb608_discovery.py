"""Small exact-hash discovery checks, without reading competition inputs."""
from pathlib import Path
import tempfile
import unittest

from scripts.launch_cuda_xgb608_dataset import digest, find_exact


class CUDA608DiscoveryTests(unittest.TestCase):
    def test_exact_hash_nested_path_and_wrong_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good = root/'dataset'/'supplement'/'bundle_manifest.json'
            good.parent.mkdir(parents=True); good.write_text('correct')
            bad = root/'bundle_manifest.json'; bad.write_text('different')
            self.assertEqual(find_exact(root, 'bundle_manifest.json', digest(good)), good.resolve())
            with self.assertRaises(ValueError): find_exact(root, 'bundle_manifest.json', '0'*64)

    def test_ambiguous_matching_copies_fail_and_symlinks_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root/'a'/'bundle_manifest.json';first.parent.mkdir();first.write_text('same')
            second = root/'b'/'bundle_manifest.json';second.parent.mkdir();second.symlink_to(first)
            self.assertEqual(find_exact(root, 'bundle_manifest.json', digest(first)), first.resolve())
            second.unlink();second.write_text('same')
            with self.assertRaises(ValueError):find_exact(root, 'bundle_manifest.json', digest(first))


if __name__ == '__main__':
    unittest.main()
