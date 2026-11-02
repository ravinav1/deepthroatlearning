"""Create a local allowlisted data/source bundle; perform no remote action."""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shlex
import shutil
import sys
import time
import traceback
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    """Return SHA256 using bounded memory."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    """Copy only reviewed inputs, then verify every ZIP member by streamed hash."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="kaggle-neural-bundle-001")
    args = parser.parse_args()
    out = ROOT / "artifacts" / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    payload = out / "payload"
    payload.mkdir()
    started = time.monotonic()
    report = {"run_id": args.run_id, "timestamp_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
              "command": shlex.join([sys.executable, *sys.argv]), "source": {str(Path(__file__).relative_to(ROOT)): digest(Path(__file__))},
              "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__},
              "remote_upload": False, "remote_execution": False, "model_training": False, "full_features_loaded": False,
              "kaggle": None, "validation_score": None}
    (out / "manifest.json").write_text(json.dumps(report, indent=2))
    try:
        lp_path = ROOT / "artifacts/lp-full-001/manifest.json"
        lp = json.loads(lp_path.read_text())
        protocol_path = ROOT / "docs/neural_tcn_protocol.json"
        protocol = json.loads(protocol_path.read_text())
        assert lp["status"] == "complete" and lp["feature_shape"] == [3647, 168, 78]
        assert protocol["data"] == lp["data"]
        assert len(protocol["feature_columns"]) == 395
        records = {}

        def include(relative: str, expected: str, role: str, expected_from: str) -> None:
            path = ROOT / relative
            assert not Path(relative).is_absolute() and ".." not in Path(relative).parts
            assert path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(ROOT)
            assert digest(path) == expected, f"Source checksum differs: {relative}"
            if relative in records:
                assert records[relative]["sha256"] == expected
                return
            target = payload / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            assert digest(target) == expected
            if path.suffix == ".py":
                ast.parse(target.read_text(), filename=relative)
            records[relative] = {"path": relative, "sha256": expected, "size_bytes": target.stat().st_size,
                                 "role": role, "expected_from": expected_from}

        for relative, expected in lp["data"].items():
            assert Path(relative).suffix in [".csv", ".yaml"]
            include(relative, expected, "raw_input_or_submission_schema", "artifacts/lp-full-001/manifest.json:data")
        for relative, expected in lp["source"].items():
            include(relative, expected, "strict_lp_loader_source_closure", "artifacts/lp-full-001/manifest.json:source")
        for relative in ["sintef_project/context_features.py", "sintef_project/normalized.py",
                         "sintef_project/postprocess.py", "sintef_project/kaggle_metric.py"]:
            include(relative, protocol["feature_source"][relative], "normalized395_feature_or_scoring_dependency",
                    "docs/neural_tcn_protocol.json:feature_source")
        include("scripts/repair_lp_cache_schema.py", lp["schema_repair"]["source_sha256"], "schema_repair_provenance",
                "artifacts/lp-full-001/manifest.json:schema_repair")
        include("artifacts/lp-full-001/repair_lp_cache_schema.py", lp["artifacts"]["repair_lp_cache_schema.py"],
                "cached_schema_repair_provenance", "artifacts/lp-full-001/manifest.json:artifacts")
        include("artifacts/lp-full-001/features.npz", lp["artifacts"]["features.npz"], "frozen_input_only_lp_features",
                "artifacts/lp-full-001/manifest.json:artifacts")
        include("artifacts/lp-full-001/manifest.json", digest(lp_path), "unchanged_lp_provenance", "source_file_sha256")
        include("docs/neural_tcn_protocol.json", digest(protocol_path), "frozen_proposed_neural_protocol", "source_file_sha256")
        include(protocol["baseline_manifest"], protocol["baseline_manifest_sha256"], "protocol_referenced_feature_provenance",
                "docs/neural_tcn_protocol.json:baseline_manifest_sha256")
        # Inspect only the feature header and small metadata arrays, not its numeric payload.
        feature_path = payload / "artifacts/lp-full-001/features.npz"
        with zipfile.ZipFile(feature_path) as archive, archive.open("features.npy") as member:
            version = np.lib.format.read_magic(member)
            assert version == (1, 0)
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(member)
            assert shape == (3647, 168, 78) and dtype == np.dtype("float32") and not fortran_order
        with np.load(feature_path, allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive["case_ids"], np.arange(1, 3648))
            assert archive["columns"].tolist() == lp["feature_names"]
            assert len(archive["units"]) == 14
        bundle = {"version": "kaggle_neural_inputs_v1", "timestamp_utc": report["timestamp_utc"],
                  "payload_files": [records[key] for key in sorted(records)],
                  "lp_manifest_sha256": digest(lp_path), "protocol_sha256": digest(protocol_path),
                  "feature_columns_count": 395, "lp_feature_shape": [3647, 168, 78],
                  "extraction_layout": "Archive root is repository root; preserve all relative paths before using strict LP loader.",
                  "status": "Local data/source bundle only. No portable trainer, GPU smoke result, upload or remote execution included.",
                  "protocol_note": "The frozen proposed protocol retains its original anchor and status text. Later authorized protocol changes require separate provenance; do not silently rewrite this input.",
                  "excluded_categories": ["credentials", "environment_directories", "checkpoints", "predictions", "per_case_LP_solutions",
                                          "unrelated_scripts", "other_experiment_caches", "private_repository_snapshots"],
                  "strict_loader_note": "Original LP manifest retains full historical artifact inventory. Only features.npz and root data/source closure are needed by add_lp_features; omitted case archives are not silently represented as bundled."}
        inner_manifest = payload / "bundle_manifest.json"
        inner_manifest.write_text(json.dumps(bundle, indent=2))
        expected = {record["path"]: record for record in bundle["payload_files"]}
        expected["bundle_manifest.json"] = {"path": "bundle_manifest.json", "sha256": digest(inner_manifest), "size_bytes": inner_manifest.stat().st_size}
        archive_path = out / "kaggle-neural-inputs.zip"
        with zipfile.ZipFile(archive_path, "w", allowZip64=True, compresslevel=6) as archive:
            for relative in sorted(expected):
                info = zipfile.ZipInfo(relative, date_time=(2026, 1, 1, 0, 0, 0))
                info.external_attr = 0o100644 << 16
                info.compress_type = zipfile.ZIP_STORED if relative.endswith(".npz") else zipfile.ZIP_DEFLATED
                with (payload / relative).open("rb") as source, archive.open(info, "w", force_zip64=True) as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
        checked = []
        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()
            assert len(names) == len(set(names)) and set(names) == set(expected)
            for relative in names:
                info = archive.getinfo(relative)
                assert info.file_size == expected[relative]["size_bytes"]
                assert not relative.startswith("/") and ".." not in Path(relative).parts
                with archive.open(relative) as handle:
                    got = hashlib.file_digest(handle, "sha256").hexdigest()
                assert got == expected[relative]["sha256"], relative
                checked.append(relative)
        report.update(status="complete", bundle_manifest_sha256=digest(inner_manifest), payload_files=bundle["payload_files"],
                      payload_file_count=len(records), zip_member_count=len(expected),
                      uncompressed_payload_bytes=sum(record["size_bytes"] for record in records.values()),
                      archive={"path": str(archive_path.relative_to(ROOT)), "sha256": digest(archive_path), "size_bytes": archive_path.stat().st_size},
                      readback={"all_members_sha256_verified": True, "exact_allowlist_verified": True, "checked_members": checked,
                                "lp_feature_header_and_metadata_verified": True, "full_numeric_feature_array_decoded": False})
        print(json.dumps({key: report[key] for key in ["status", "payload_file_count", "zip_member_count", "uncompressed_payload_bytes", "archive"]}))
    except BaseException:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        raise
    finally:
        report["runtime_seconds"] = time.monotonic() - started
        (out / "manifest.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
