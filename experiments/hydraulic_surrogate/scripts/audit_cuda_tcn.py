"""Independent small-fixture review of the portable CUDA runner; no training."""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import traceback
import types
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Exercise staging boundaries, target-year redaction and training statistics."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    out = ROOT / "artifacts" / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    runner = ROOT / "scripts/train_cuda_tcn.py"
    source = runner.read_bytes()
    module = types.ModuleType("independent_tcn_audit_target")
    module.__file__ = str(runner)
    exec(compile(source, str(runner), "exec"), module.__dict__)
    report = {"run_id": args.run_id, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "status": "running", "source_sha256": hashlib.sha256(source).hexdigest(),
              "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "tests": [], "real_data_read": False, "torch_imported": False, "model_fit": False,
              "GPU_test": False, "cross_platform_numeric_equivalence": "Not established by static review or small CPU fixtures"}
    (out / "reviewed_train_cuda_tcn.py").write_bytes(source)
    try:
        with tempfile.TemporaryDirectory(prefix="tdt4173-tcn-audit-") as temporary:
            root = Path(temporary)
            content = b"test payload only\n"
            manifest = {"payload_files": [{"path": "sintef_project/small.txt", "sha256": hashlib.sha256(content).hexdigest(), "size_bytes": len(content)}]}
            manifest_bytes = json.dumps(manifest).encode()
            manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()

            def make_zip(name, extra=None, altered=False):
                path = root / name
                with zipfile.ZipFile(path, "w") as archive:
                    archive.writestr("bundle_manifest.json", manifest_bytes)
                    archive.writestr("sintef_project/small.txt", b"altered" if altered else content)
                    if extra:
                        archive.writestr(extra, b"unlisted")
                return path

            good = make_zip("good.zip")
            module.stage_bundle(good, root / "staged", manifest_sha, module.digest(good))
            assert (root / "staged/sintef_project/small.txt").read_bytes() == content
            report["tests"].append("valid exact-allowlist ZIP round trip")
            module.stage_bundle(good, root / "staged", manifest_sha, module.digest(good))
            (root / "staged/sintef_project/small.txt").write_bytes(b"changed")
            try:
                module.stage_bundle(good, root / "staged", manifest_sha, module.digest(good))
            except ValueError:
                pass
            else:
                raise AssertionError("Tampered existing stage was accepted")
            report["tests"].append("strict reuse accepts exact bytes and rejects tampering")
            for name in ["../escape", "/absolute", "a/../b", "a\\b", "./a", "."]:
                try:
                    module.safe_relative(name)
                except ValueError:
                    pass
                else:
                    raise AssertionError(f"Unsafe path accepted: {name}")
            report["tests"].append("noncanonical and escaping paths rejected")
            for name, extra, altered in [("extra", "unlisted.py", False), ("traversal", "../escape", False), ("tampered", None, True)]:
                path = make_zip(name + ".zip", extra, altered)
                try:
                    module.stage_bundle(path, root / name, manifest_sha, module.digest(path))
                except ValueError:
                    pass
                else:
                    raise AssertionError(f"Invalid ZIP accepted: {name}")
            assert not (root / "escape").exists()
            report["tests"].append("unlisted/traversal/tampered ZIPs rejected without outside writes")
            path = root / "target.csv"
            path.write_text("Run No,starttime,total_calculation_time,result_a\n1,2020-12-01,1,1\n2,2021-05-01,1,0\n3,2022-06-01,1,INVALID_FUTURE_TARGET\n")
            requests = []

            def reader(path, *args, **kwargs):
                requests.append(kwargs.copy())
                return pd.read_csv(path, *args, **kwargs)

            redacted = module.read_targets_through_year(path, 2021, reader)
            assert requests == [{"usecols": ["Run No", "starttime"]}, {"nrows": 2}]
            np.testing.assert_array_equal(redacted.result_a.to_numpy(), [1, 0, 0])
            assert redacted.starttime.iloc[-1] == "2022-06-01"
            report["tests"].append("2021 target prefix never decodes sentinel2022 target")
            protocol = {"anchor": {"run_id": "fixture-anchor", "analysis_sha256": "a" * 64}}
            lock = {"version": "cuda_tcn_2021_blend_lock_v1", "selection_year": 2021,
                    "training_protocol_sha256": "b" * 64, "anchor_run": "fixture-anchor", "anchor_sha256": "a" * 64,
                    "candidate_neural_weights": [0., .1, .25, .5], "selected_neural_weight": .1,
                    "training_2021_complete": True, "neural_run_manifest_sha256": "c" * 64,
                    "neural_validation_2021_sha256": "d" * 64}
            lock_path = root / "fixture-lock.json"
            lock_path.write_text(json.dumps(lock))
            module.validate_selection_lock(lock_path, module.digest(lock_path), protocol, "b" * 64)
            for field, bad in [("selection_year", 2022), ("training_protocol_sha256", "e" * 64),
                               ("selected_neural_weight", .9), ("training_2021_complete", False)]:
                changed_lock = dict(lock); changed_lock[field] = bad
                lock_path.write_text(json.dumps(changed_lock))
                try:
                    module.validate_selection_lock(lock_path, module.digest(lock_path), protocol, "b" * 64)
                except ValueError:
                    pass
                else:
                    raise AssertionError(f"Invalid selection lock accepted: {field}")
            report["tests"].append("2022 lock rejects wrong year/protocol/weight/incomplete2021")
            cube = np.arange(6 * 4 * 3, dtype=np.float32).reshape(6, 4, 3)
            cube[:, :, 2] = 7.
            train = np.array([0, 2])
            mean, scale = module.training_stats(cube, train)
            expected = cube[train].reshape(-1, 3).astype(np.float64)
            np.testing.assert_allclose(mean, expected.mean(0), atol=1e-12)
            std = expected.std(0); std[std < 1e-6] = 1.
            np.testing.assert_allclose(scale, std, atol=1e-12)
            changed = cube.copy(); changed[[1, 3, 4, 5]] = 1e30
            altered_mean, altered_scale = module.training_stats(changed, train)
            np.testing.assert_array_equal(mean, altered_mean)
            np.testing.assert_array_equal(scale, altered_scale)
            batch = module.normalized_batch(changed, np.array([0, 1]), mean, scale)
            assert batch.shape == (2, 3, 4) and batch.dtype == np.float32 and batch.flags.c_contiguous
            assert np.isfinite(batch).all() and (np.abs(batch) <= 10).all()
            report["tests"].append("training-only float64 moments, constant scale, clipping and NCT layout")
            fake = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
            try:
                module.cuda_environment(fake)
            except ValueError as error:
                assert "CPU fallback is forbidden" in str(error)
            else:
                raise AssertionError("Missing CUDA was accepted")
            report["tests"].append("missing CUDA fails before any CPU model execution")
        protocol_path = ROOT / "docs/cuda_tcn_protocol.json"
        if protocol_path.is_file():
            protocol_bytes = protocol_path.read_bytes()
            protocol = json.loads(protocol_bytes)
            protocol_sha = hashlib.sha256(protocol_bytes).hexdigest()
            assert protocol["trainer_sha256"] == report["source_sha256"]
            original_path = ROOT / "docs/neural_tcn_protocol.json"
            original = json.loads(original_path.read_text())
            assert protocol["supersedes_protocol_sha256"] == hashlib.sha256(original_path.read_bytes()).hexdigest()
            assert protocol["feature_columns"] == original["feature_columns"]
            bundle = ROOT / "artifacts/kaggle-neural-bundle-001/payload/bundle_manifest.json"
            assert protocol["bundle_manifest_sha256"] == hashlib.sha256(bundle.read_bytes()).hexdigest()
            anchor = ROOT / "artifacts/lp-joint-comparison-001/analysis.json"
            assert protocol["anchor"]["analysis_sha256"] == hashlib.sha256(anchor.read_bytes()).hexdigest()
            report["protocol_sha256"] = protocol_sha
            report["notebooks"] = {}
            for name in ["cuda_tcn_smoke.ipynb", "cuda_tcn_review.ipynb"]:
                notebook_path = ROOT / "notebooks" / name
                notebook = json.loads(notebook_path.read_text())
                assert notebook["metadata"]["tdt4173_protocol_sha256"] == protocol_sha
                for cell in notebook["cells"]:
                    if cell["cell_type"] == "code":
                        assert cell["execution_count"] is None and cell["outputs"] == []
                        compile("".join(cell["source"]), name, "exec")
                tree = ast.parse("".join(notebook["cells"][1]["source"]))
                constants = {target.id: ast.literal_eval(node.value) for node in tree.body
                             if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
                             for target in node.targets if isinstance(target, ast.Name)}
                assert constants["TRAINER_TEXT"].encode() == source
                assert constants["PROTOCOL_TEXT"].encode() == protocol_bytes
                if name == "cuda_tcn_smoke.ipynb":
                    assert len(notebook["cells"]) == 4
                else:
                    for index, variable in [(5, "RUN_2021_EXPLICITLY"), (7, "RUN_2022_EXPLICITLY")]:
                        tree = ast.parse("".join(notebook["cells"][index]["source"]))
                        assignments = {target.id: node.value for node in tree.body if isinstance(node, ast.Assign)
                                       for target in node.targets if isinstance(target, ast.Name)}
                        assert ast.literal_eval(assignments[variable]) is False
                report["notebooks"][name] = hashlib.sha256(notebook_path.read_bytes()).hexdigest()
            report["tests"].append("protocol lineage/bundle/anchor hashes and notebook embedded bytes/defaults match")
        report["status"] = "complete"
    except BaseException:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        raise
    finally:
        report["runtime_seconds"] = time.monotonic() - started
        (out / "manifest.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report))


if __name__ == "__main__":
    main()
