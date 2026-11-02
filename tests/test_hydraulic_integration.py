"""Tests for artifact restoration and isolation from historical Round pipelines."""
from pathlib import Path
import hashlib
import importlib.util
import io
import json
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("hydraulic_adapter", ROOT / "src/hydraulic_surrogate.py")
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


class ArtifactRestoreTests(unittest.TestCase):
    def test_traversal_and_symlink_escape_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ["../escape", "/absolute", "dir/../../escape", "a\\b"]:
                with self.assertRaises(ValueError):
                    adapter.safe_path(root, name)
            (root / "escape").symlink_to(root.parent, target_is_directory=True)
            with self.assertRaises(ValueError):
                adapter.safe_path(root, "escape/file")

    def test_restore_checks_bytes_and_preserves_conflicts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            content = b"immutable model bytes"
            records = {"run/model.txt": {"size_bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}}
            archive = root / "models.tar.gz"
            with tarfile.open(archive, "w:gz") as bundle:
                member = tarfile.TarInfo("run/model.txt")
                member.size = len(content)
                bundle.addfile(member, io.BytesIO(content))
            adapter.install_archive(archive, root / "restored", records)
            adapter.install_archive(archive, root / "restored", records)
            restored = root / "restored/run/model.txt"
            self.assertEqual(restored.read_bytes(), content)
            restored.write_bytes(b"local experiment")
            with self.assertRaises(ValueError):
                adapter.install_archive(archive, root / "restored", records)
            self.assertEqual(restored.read_bytes(), b"local experiment")

    def test_bad_digest_never_installs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "bad.tar.gz"
            with tarfile.open(archive, "w:gz") as bundle:
                member = tarfile.TarInfo("model")
                member.size = 3
                bundle.addfile(member, io.BytesIO(b"bad"))
            with self.assertRaises(ValueError):
                adapter.install_archive(archive, root / "restored", {"model": {"size_bytes": 3, "sha256": "0" * 64}})
            self.assertFalse((root / "restored/model").exists())

    def test_recipe_preserves_nested_projection_stages(self):
        config = json.loads((ROOT / "configs/hydraulic_surrogate_best.json").read_text())
        self.assertEqual(sum(config["base_weights"].values()), 1.0)
        self.assertEqual([entry["weight"] for entry in config["nested_blends"]], [0.5, 0.25, 0.1])
        self.assertEqual(len(config["components"]), 7)
        self.assertEqual(len(config["test_ids"]), 725)
        self.assertEqual(len(set(config["test_ids"]) & set(config["train_ids"])), 0)
        self.assertTrue(all(entry["repository_path"].startswith(("data/raw/", "configs/")) for entry in config["data"].values()))

    def test_preflight_failure_is_recorded_without_training(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(adapter, "STORE", Path(temporary)):
            with patch.object(adapter, "setup", side_effect=ValueError("missing supplied inputs")):
                with self.assertRaisesRegex(ValueError, "missing supplied inputs"):
                    adapter.reconstruct(False, "failed-preflight")
            manifest = json.loads((Path(temporary) / "reproductions/failed-preflight/manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertFalse(manifest["training"])
            self.assertEqual(manifest["components"], {})
            self.assertIn("missing supplied inputs", manifest["error"])


if __name__ == "__main__":
    unittest.main()
