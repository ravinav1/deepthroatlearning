"""Exercise the real launcher using a deterministic, metadata-only fake Codex CLI."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


REPOSITORY = Path(__file__).resolve().parents[1]
FAKE_CODEX = '''\
import json
import os
from pathlib import Path
import sys
import tomllib

if "--version" in sys.argv:
    print("codex-cli 0.166.0 (test double)")
    raise SystemExit(0)

if "app-server" not in sys.argv:
    Path("launched.json").write_text(json.dumps(sys.argv[1:]))
    raise SystemExit(0)

config = {"agents": {}}
for index, argument in enumerate(sys.argv[:-1]):
    if argument == "-c":
        item = tomllib.loads(sys.argv[index + 1])
        if "agents" in item:
            config["agents"].update(item["agents"])
        else:
            config.update(item)
config["agents"].update(json.loads(os.environ.get("FAKE_CODEX_AGENT_OVERRIDES", "{}")))
for key in json.loads(os.environ.get("FAKE_CODEX_AGENT_OMISSIONS", "[]")):
    config["agents"].pop(key, None)
config.update(json.loads(os.environ.get("FAKE_CODEX_ROOT_OVERRIDES", "{}")))

for line in sys.stdin:
    request = json.loads(line)
    if "id" not in request:
        continue
    method = request["method"]
    if method == "initialize":
        result = {}
    elif method == "config/read":
        result = {
            "config": config,
            "layers": [{
                "name": {"type": "project", "dotCodexFolder": str(Path.cwd() / ".codex")},
                "disabledReason": "untrusted" if os.environ.get("FAKE_CODEX_UNTRUSTED") else None,
            }],
        }
    elif method == "model/list":
        result = {
            "data": [{"model": "gpt-6-astra", "supportedReasoningEfforts": [
                {"reasoningEffort": "ultra"}, {"reasoningEffort": "max"},
            ]}],
            "nextCursor": None,
        }
    else:
        print(json.dumps({"id": request["id"], "error": {"code": -32601}}), flush=True)
        continue
    print(json.dumps({"id": request["id"], "result": result}), flush=True)
'''


@unittest.skipUnless(shutil.which("bash"), "The launcher requires Bash")
@unittest.skipIf(sys.version_info < (3, 11), "The launcher requires Python 3.11+")
class CodexStartupTests(unittest.TestCase):
    """Protect startup gates without credentials, networks, models or training."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="dtl-codex-startup-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        (self.root / "scripts").mkdir()
        shutil.copy(REPOSITORY / "scripts/codex-start.sh", self.root / "scripts/codex-start.sh")
        shutil.copytree(REPOSITORY / ".codex", self.root / ".codex")
        (self.root / "configs").mkdir()
        (self.root / "reports/hydraulic_surrogate").mkdir(parents=True)
        fake_bin = self.root / "bin"
        fake_bin.mkdir()
        (fake_bin / "codex.py").write_text(FAKE_CODEX)
        (fake_bin / "codex").write_text('#!/bin/sh\nexec "$DTL_CODEX_PYTHON" "$0.py" "$@"\n')
        (fake_bin / "codex").chmod(0o755)
        self.environment = {
            key: value for key, value in os.environ.items()
            if not key.startswith(("DTL_CODEX_", "FAKE_CODEX_"))
        }
        self.environment.update({
            "PATH": str(fake_bin) + os.pathsep + os.environ.get("PATH", ""),
            "DTL_CODEX_PYTHON": sys.executable,
            "DTL_CODEX_TIMEOUT": "5",
        })
        self.data_path = self.root / "data/raw/test.csv"
        self.artifact_path = self.root / "models/hydraulic_surrogate/test/model.bin"
        expected = {"size_bytes": 2, "sha256": hashlib.sha256(b"ok").hexdigest()}
        recipe = {"data": {"original": dict(expected, repository_path="data/raw/test.csv")}}
        inventory = {"files": {"test/model.bin": dict(expected, group="best")}}
        (self.root / "configs/hydraulic_surrogate_best.json").write_text(json.dumps(recipe))
        (self.root / "reports/hydraulic_surrogate/artifacts.json").write_text(json.dumps(inventory))
        self.launched = self.root / "launched.json"
        self.lock = self.root / "models/codex-session/session.lock"

    def restore_inputs(self) -> None:
        """Create small files whose expected hashes are in the fixture inventory."""
        for path in (self.data_path, self.artifact_path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"ok")

    def run_launcher(
        self,
        arguments: list[str],
        expected_exit: int,
        *,
        environment: dict[str, str] | None = None,
        launched: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        """Run the shell entry point and assert whether it reached model launch."""
        self.launched.unlink(missing_ok=True)
        result = subprocess.run(
            ["bash", "scripts/codex-start.sh", *arguments],
            cwd=self.root,
            env=dict(self.environment, **(environment or {})),
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(result.returncode, expected_exit, result.stdout + result.stderr)
        self.assertEqual(self.launched.exists(), launched, result.stdout + result.stderr)
        return result

    def read_report(self) -> dict:
        """Read the last preflight report from this test's isolated workspace."""
        paths = list((self.root / "models/codex-session").glob("preflight-*.json"))
        self.assertTrue(paths, "Expected a durable preflight report")
        return json.loads(max(paths, key=lambda path: path.stat().st_mtime_ns).read_text())

    def assert_agent_setting_rejected(self, key: str, value: object) -> None:
        """Verify a changed effective child setting fails closed and is recorded."""
        self.restore_inputs()
        result = self.run_launcher(
            ["start"], 1,
            environment={"FAKE_CODEX_AGENT_OVERRIDES": json.dumps({key: value})},
        )
        self.assertIn("Effective agents." + key, result.stderr)
        report = self.read_report()
        self.assertEqual(report["effective_configuration"]["agents"][key], value)
        self.assertTrue(any("agents." + key in error for error in report["errors"]))
        self.assertFalse(self.lock.exists(), "A failed preflight must release its launcher lock")

    def test_missing_artifacts_fail_check(self) -> None:
        self.run_launcher(["check"], 2)
        self.assertFalse(self.read_report()["data_and_best_complete"])

    def test_source_only_audit_can_start(self) -> None:
        self.run_launcher(["start"], 0, launched=True)
        self.assertIn("Perform the startup audit only", json.loads(self.launched.read_text())[-1])
        self.assertFalse(self.read_report()["data_and_best_complete"])

    def test_missing_artifacts_block_improvement(self) -> None:
        self.run_launcher(["improve"], 2)
        self.assertFalse(self.lock.exists())

    def test_complete_metadata_and_hash_preflight(self) -> None:
        self.restore_inputs()
        self.run_launcher(["check", "--hashes"], 0)
        report = self.read_report()
        self.assertTrue(report["data_and_best_complete"])
        self.assertTrue(report["hashes_checked"])
        self.assertEqual(report["model_catalog_status"], "advertised")
        self.assertFalse(report["model_execution_verified"])

    def test_unadvertised_model_blocks_launch(self) -> None:
        self.run_launcher(["start"], 1, environment={"DTL_CODEX_MODEL": "not-available"})
        self.assertEqual(self.read_report()["model_catalog_status"], "not advertised")

    def test_unadvertised_effort_blocks_launch(self) -> None:
        self.run_launcher(["start"], 1, environment={"DTL_CODEX_EFFORT": "not-available"})
        self.assertEqual(self.read_report()["model_catalog_status"], "not advertised")

    def test_untrusted_project_blocks_launch(self) -> None:
        self.run_launcher(["start"], 1, environment={"FAKE_CODEX_UNTRUSTED": "1"})
        self.assertFalse(self.read_report()["project_config_loaded"])

    def test_explicit_fallback_reaches_root_and_children(self) -> None:
        self.run_launcher(["start"], 0, environment={"DTL_CODEX_EFFORT": "max"}, launched=True)
        arguments = json.loads(self.launched.read_text())
        self.assertIn('model_reasoning_effort="max"', arguments)
        self.assertIn('agents.default_subagent_reasoning_effort="max"', arguments)
        self.assertIn('agents.default_subagent_model="gpt-6-astra"', arguments)
        self.assertEqual(self.read_report()["effective_configuration"]["agents"]["default_subagent_reasoning_effort"], "max")

    def test_complete_improvement_starts_coordinator_only(self) -> None:
        self.restore_inputs()
        self.run_launcher(["improve"], 0, launched=True)
        prompt = json.loads(self.launched.read_text())[-1]
        self.assertIn("bounded, preregistered experiment cycles", prompt)
        self.assertIn("Continue the user's requested ongoing goal work without another permission round", prompt)
        self.assertTrue(self.read_report()["hashes_checked"])

    def test_same_sized_corrupt_artifact_blocks_improvement(self) -> None:
        self.restore_inputs()
        self.artifact_path.write_bytes(b"no")
        self.run_launcher(["improve"], 2)
        self.assertEqual(self.read_report()["best_artifacts"]["invalid"][0]["reason"], "SHA256 mismatch")

    def test_duplicate_lock_is_preserved(self) -> None:
        self.lock.mkdir(parents=True)
        owner = self.lock / "owner.txt"
        owner.write_text("another launcher")
        self.run_launcher(["start"], 73)
        self.assertEqual(owner.read_text(), "another launcher")

    def test_offline_cannot_launch(self) -> None:
        self.run_launcher(["start", "--offline"], 64)
        self.assertFalse(self.lock.exists())

    def test_unsafe_model_identifier_is_rejected(self) -> None:
        self.run_launcher(["check"], 64, environment={"DTL_CODEX_MODEL": "a;touch injected"})
        self.assertFalse((self.root / "injected").exists())

    def test_successful_start_releases_lock(self) -> None:
        self.run_launcher(["start"], 0, launched=True)
        self.assertFalse(self.lock.exists())

    def test_disabled_agents_block_launch(self) -> None:
        self.assert_agent_setting_rejected("enabled", False)

    def test_excessive_agent_concurrency_blocks_launch(self) -> None:
        self.assert_agent_setting_rejected("max_concurrent_threads_per_session", 99)

    def test_different_child_model_blocks_launch(self) -> None:
        self.assert_agent_setting_rejected("default_subagent_model", "gpt-6-luna")

    def test_different_child_effort_blocks_launch(self) -> None:
        self.assert_agent_setting_rejected("default_subagent_reasoning_effort", "low")

    def test_ignored_agent_settings_block_launch(self) -> None:
        for key in (
            "enabled", "max_concurrent_threads_per_session",
            "default_subagent_model", "default_subagent_reasoning_effort",
        ):
            with self.subTest(key=key):
                result = self.run_launcher(
                    ["start"], 1,
                    environment={"FAKE_CODEX_AGENT_OMISSIONS": json.dumps([key])},
                )
                self.assertIn("Effective agents." + key, result.stderr)
                self.assertFalse(self.lock.exists())

    def test_mistyped_agent_values_block_launch(self) -> None:
        for key, value in (("enabled", 1), ("max_concurrent_threads_per_session", "3")):
            with self.subTest(key=key):
                self.assert_agent_setting_rejected(key, value)

    def test_different_effective_root_settings_block_launch(self) -> None:
        for key, value in (("model", "gpt-6-luna"), ("model_reasoning_effort", "low")):
            with self.subTest(key=key):
                result = self.run_launcher(
                    ["start"], 1,
                    environment={"FAKE_CODEX_ROOT_OVERRIDES": json.dumps({key: value})},
                )
                self.assertIn("Effective model/effort differs", result.stderr)


if __name__ == "__main__":
    unittest.main()
