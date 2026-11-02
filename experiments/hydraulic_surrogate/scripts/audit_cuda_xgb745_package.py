"""Independently inspect the sealed additive LPv2 package without fitting."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SHA = "4e8aaf0a32c7899cac0b0c95927af6b0bb764d49847069fcf4af5a5c8092b41e"
ZIP_SHA = "c01101fa9a6756fb78ec5f70c16b47e21a95ea22711a0fb66ab8f26a142a9894"


def sha(path: Path) -> str:
    """Hash exact original or packaged bytes."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: Path) -> dict:
    """Read a local JSON record without decoding prediction or target arrays."""
    return json.loads(path.read_text())


def main() -> None:
    """Verify source closure, package members, fixed controls and inherited recipe."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    assert Path(args.run_id).name == args.run_id and args.run_id not in ("", ".", "..")
    out = ROOT / "artifacts" / args.run_id
    out.mkdir(exist_ok=False)
    start = time.monotonic()
    report = {"status": "running", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "command": [sys.executable, *sys.argv], "protocol_sha256": PROTOCOL_SHA,
              "training": False, "labels_decoded": False, "upload": False,
              "environment": {"python": sys.version, "platform": platform.platform()}}
    try:
        package = ROOT / "artifacts/cuda-xgb745-package-002"
        m = read(package / "manifest.json")
        p_path = ROOT / "docs/cuda_xgb745_protocol.json"
        assert sha(p_path) == PROTOCOL_SHA == m["protocol_sha256"]
        p = read(p_path)
        old = read(ROOT / "docs/cuda_xgb664_protocol.json")
        assert sha(ROOT / "docs/cuda_xgb664_protocol.json") == "4e09f0fe59973d7a9c2c5e5792feb30c29b8af04a67c27bafbe1386ae9da8182"
        assert m["status"] == "complete" and m["training"] is False and m["upload"] is False
        for name, expected in m["source"].items():
            assert sha(ROOT / name) == expected == sha(package / "source" / name), name
        for name, expected in m["artifacts"].items():
            assert sha(package / name) == expected, name
        for key in ("model_parameters", "seed", "candidate_weights", "units", "folds", "sample_sha256",
                    "sampling", "feature_source_sha256", "baseline_manifest_sha256", "core_loader_sha256",
                    "launcher_sha256", "postprocess_sha256", "bundle_manifest_sha256", "bundle_archive_sha256",
                    "supplement_manifest_sha256", "supplement_archive_sha256", "prior_loader_sha256",
                    "shop_bundle_manifest_sha256", "shop_bundle_archive_sha256", "shop_cache_manifest_sha256",
                    "shop_features_sha256"):
            assert p[key] == old[key], key
        assert p["prior664_loader_sha256"] == old["shared_loader_sha256"]
        assert p["feature_columns"][:664] == old["feature_columns"]
        assert len(p["feature_columns"]) == len(set(p["feature_columns"])) == 745
        cache = ROOT / p["v2_cache_directory"]
        assert sha(cache / "manifest.json") == p["v2_cache_manifest_sha256"]
        cm = read(cache / "manifest.json")
        assert cm["status"] == "complete" and cm["feature_shape"] == [3647, 168, 81]
        assert cm["feature_names"] == p["feature_columns"][664:]
        assert cm["math"] == p["feature_ablation"]["lp_v2"]
        assert len(cm["cases"]) == 3647 and len(cm["artifacts"]) == 3653
        for case in cm["cases"]:
            assert case["status"] == 0
            for key in ("max_minimum_slack_mm3", "max_capacity_slack_mm3", "max_bypass_slack_mm3", "max_vest_slack_mm3"):
                assert case[key] == 0
        for group in ("source", "data"):
            for name, expected in cm[group].items():
                assert sha(ROOT / name) == expected
        for name, expected in cm["artifacts"].items():
            assert sha(cache / name) == expected
        existing = {}
        for path, expected in [(ROOT / "artifacts/kaggle-neural-bundle-001/payload/bundle_manifest.json", p["bundle_manifest_sha256"]),
                               (ROOT / "artifacts/cuda-xgb608-package-001/supplement/bundle_manifest.json", p["supplement_manifest_sha256"]),
                               (ROOT / p["shop_bundle_manifest_path"], p["shop_bundle_manifest_sha256"])]:
            assert sha(path) == expected
            for member in read(path)["payload_files"]:
                assert member["path"] not in existing
                existing[member["path"]] = member["sha256"]
        supplement = read(ROOT / p["v2_bundle_manifest_path"])
        assert sha(ROOT / p["v2_bundle_manifest_path"]) == p["v2_bundle_manifest_sha256"]
        assert supplement["contains_labels"] is False and supplement["contains_anchor_predictions"] is False
        assert len(supplement["payload_files"]) == 4
        for member in supplement["payload_files"]:
            assert member["path"] not in existing
            assert sha(ROOT / member["path"]) == member["sha256"]
        archive = package / "cuda-xgb745-private-dataset.zip"
        assert sha(archive) == ZIP_SHA == m["dataset_archive_sha256"]
        assert archive.stat().st_size == 54218767 == m["dataset_archive_bytes"]
        dataset = package / "dataset"
        manifest = read(dataset / "dataset_manifest.json")
        assert manifest["required_visibility"] == "PRIVATE" and manifest["protocol_sha256"] == PROTOCOL_SHA
        names = {x["path"] for x in manifest["files"]} | {"dataset_manifest.json"}
        with zipfile.ZipFile(archive) as z:
            assert len(z.namelist()) == len(names) and set(z.namelist()) == names
            for name in names:
                assert not Path(name).is_absolute() and ".." not in Path(name).parts
                assert z.read(name) == (dataset / name).read_bytes()
            for member in manifest["files"]:
                assert hashlib.sha256(z.read(member["path"])).hexdigest() == member["sha256"]
        for year in ("2021", "2022"):
            for role in ("anchor", "control"):
                record = p[role][year]
                path = ROOT / record["prediction"]
                assert sha(path) == record["prediction_sha256"]
                assert sha(ROOT / record["manifest"]) == record["manifest_sha256"]
            assert f"cuda-xgb664-select-{year}-001" in p["anchor"][year]["prediction"]
            assert f"cuda-xgb664-{year}-001/validation_{year}.npz" in p["control"][year]["prediction"]
        feature_path = ROOT / "artifacts/cuda-xgb745-feature-preflight-001/manifest.json"
        assert sha(feature_path) == "9c90dff0df4a232c53dc56410130e200625da9deaa49361a586ae28a95d46413"
        feature = read(feature_path)
        assert feature["status"] == "passed" and feature["shape"] == [612696, 745]
        assert feature["protocol_sha256"] == PROTOCOL_SHA and feature["dataset_archive_sha256"] == ZIP_SHA
        assert feature["appended81_bitwise"] and feature["shop56_bitwise"]
        assert feature["maximum_decoded_target_year"] == 2021 and feature["training"] is False
        assert sha(feature_path.parent / "audit.py") == feature["source_sha256"]
        assert feature["feature_spec"]["lp_v2"]["mathematics"] == p["feature_ablation"]["lp_v2"]
        suite = read(ROOT / "artifacts/verification-suite-20261102-1939/manifest.json")
        assert suite["status"] == "passed" and suite["tests_passed"] == 272 and suite["test_sources_unchanged"]
        assert suite["test_sources"]["tests/test_cuda_xgb745.py"] == m["source"]["tests/test_cuda_xgb745.py"]
        report.update(status="passed", package_manifest_sha256=sha(package / "manifest.json"),
                      archive_sha256=ZIP_SHA, archive_size_bytes=archive.stat().st_size,
                      verified_original_cache_artifacts=3653, original_bundle_payloads_disjoint=True,
                      source_hashes=m["source"], fixed_sample_and_fold_recipe_equal=True,
                      producer_feature_preflight_sha256=sha(feature_path),
                      producer_observed_feature_peak_rss_bytes=feature["peak_rss_bytes"],
                      tests_proof_sha256=sha(ROOT / "artifacts/verification-suite-20261102-1939/manifest.json"),
                      review="Reviewed trainer/selector/verifier deltas from frozen664: only feature extension, input bundle, schema and current anchor/control change. Parameters, chronology, CUDA guard and full checkpoint replay preserved.",
                      scope="Independent sealed package and source review, including review of the producer's actual745 construction proof. No second heavy feature rebuild. Fresh hosted CUDA smoke still required.")
    except BaseException:
        report.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        shutil.copy2(__file__, out / Path(__file__).name)
        report["audit_source_sha256"] = sha(Path(__file__))
        report["runtime_seconds"] = time.monotonic() - start
        (out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"status": report["status"], "manifest_sha256": sha(out / "manifest.json"),
                          "runtime_seconds": report["runtime_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
