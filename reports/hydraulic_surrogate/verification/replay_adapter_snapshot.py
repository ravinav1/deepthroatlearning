"""Portable entry point for the preserved hydraulic-surrogate experiments.

Run from any directory. Original research code and manifests stay byte-identical;
this adapter maps data/artifacts into the existing repository conventions.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import gzip
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "experiments/hydraulic_surrogate"
STORE = ROOT / "models/hydraulic_surrogate"
INDEX = ROOT / "reports/hydraulic_surrogate/artifacts.json"
CONFIG = ROOT / "configs/hydraulic_surrogate_best.json"


def digest(path: Path) -> str:
    """Return a bounded-memory SHA256 digest."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path) -> dict:
    """Read a UTF-8 JSON object."""
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    """Keep validation active even when Python assertions are disabled."""
    if not condition:
        raise ValueError(message)


def safe_path(root: Path, name: str) -> Path:
    """Reject archive traversal, absolute paths, and escaping existing symlinks."""
    relative = PurePosixPath(name)
    require(bool(name) and not relative.is_absolute() and ".." not in relative.parts
            and "\\" not in name, f"Unsafe relative path: {name!r}")
    path = root.joinpath(*relative.parts)
    require(path.resolve().is_relative_to(root.resolve()), f"Path escapes destination: {name}")
    return path


def verify_file(path: Path, record: dict) -> None:
    """Require both the recorded size and content hash."""
    require(path.is_file(), f"Missing {path}; fetch the appropriate artifact group first")
    require(path.stat().st_size == record["size_bytes"] and digest(path) == record["sha256"],
            f"Artifact changed: {path}")


def install_archive(archive: Path, destination: Path, records: dict) -> None:
    """Install only enumerated regular files, checking all bytes before publication."""
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".restore-", dir=destination) as temporary:
        stage = Path(temporary)
        seen = set()
        with tarfile.open(archive, "r:gz") as bundle:
            for member in bundle:
                require(member.isfile() and member.name in records and member.name not in seen,
                        f"Unlisted, duplicate, or non-regular archive member: {member.name}")
                require(member.size == records[member.name]["size_bytes"], "Archive size mismatch")
                output = safe_path(stage, member.name)
                output.parent.mkdir(parents=True, exist_ok=True)
                with bundle.extractfile(member) as source, output.open("xb") as target:
                    shutil.copyfileobj(source, target)
                verify_file(output, records[member.name])
                seen.add(member.name)
        require(seen == set(records), "Archive is missing indexed files")
        # Check conflicts before installing any files. Never replace local work.
        for name, record in records.items():
            target = safe_path(destination, name)
            if target.exists():
                verify_file(target, record)
        for name in records:
            target = safe_path(destination, name)
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(stage / name, target)


def download_asset(asset: dict) -> Path:
    """Cache a release asset only after its committed size/hash checks pass."""
    downloads = STORE / ".downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    archive = safe_path(downloads, asset["name"])
    if not archive.exists():
        temporary = archive.with_suffix(archive.suffix + ".partial")
        request = urllib.request.Request(asset["url"], headers={"User-Agent": "deepthroatlearning-artifacts"})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        verify_file(temporary, asset)
        os.replace(temporary, archive)
    verify_file(archive, asset)
    return archive


def artifact_index(group: str, download: bool = False) -> dict:
    """Load optional large inventories from hash-pinned release assets."""
    index = read_json(INDEX)
    for inventory in index.get("inventories", []):
        if group not in {"all", inventory["group"]}:
            continue
        path = download_asset(inventory) if download else STORE / ".downloads" / inventory["name"]
        verify_file(path, inventory)
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            entries = json.load(stream)
        require(not set(entries) & set(index["files"]), "Overlapping inventories")
        index["files"].update(entries)
    return index


def fetch_artifacts(group: str) -> None:
    """Download and verify release assets without credentials or machine-local paths."""
    index = artifact_index(group, download=True)
    assets = [asset for asset in index["assets"] if group == "all" or asset["group"] == group]
    require(bool(assets), f"Unknown artifact group: {group}")
    for asset in assets:
        archive = download_asset(asset)
        records = {name: record for name, record in index["files"].items() if record["asset"] == asset["name"]}
        install_archive(archive, STORE, records)
        print(f"Restored {asset['name']} ({len(records)} verified files)", flush=True)


def link_path(source: Path, destination: Path) -> None:
    """Create a local relative link, preserving existing files and rejecting conflicts."""
    if destination.is_symlink():
        require(destination.resolve() == source.resolve(), f"Existing link differs: {destination}")
        return
    if destination.exists():
        require(source.is_file() and destination.is_file() and digest(source) == digest(destination),
                f"Existing path differs: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(os.path.relpath(source, destination.parent), target_is_directory=source.is_dir())


def setup() -> None:
    """Verify supplied inputs and map the archived layout to this repository."""
    config = read_json(CONFIG)
    for original, record in config["data"].items():
        source = ROOT / record["repository_path"]
        verify_file(source, record)
        link_path(source, PACKAGE / original)
    STORE.mkdir(parents=True, exist_ok=True)
    link_path(STORE, PACKAGE / "artifacts")
    print("Verified supplied inputs; configured repository-local data and artifact paths.")


def import_data(source: Path) -> None:
    """Copy an extracted course sintef_project directory after verifying all inputs."""
    config = read_json(CONFIG)
    pending = []
    for original, record in config["data"].items():
        incoming = source / original.removeprefix("sintef_project/")
        verify_file(incoming, record)
        destination = ROOT / record["repository_path"]
        if destination.exists():
            verify_file(destination, record)
        else:
            pending.append((incoming, destination))
    for incoming, destination in pending:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(incoming, destination)
    setup()


def verify(group: str = "best") -> None:
    """Check frozen source and restored artifact integrity."""
    snapshot = read_json(ROOT / "reports/hydraulic_surrogate/source_snapshot.json")
    for name, expected in snapshot["files"].items():
        require(digest(PACKAGE / name) == expected, f"Frozen research source changed: {name}")
    index = artifact_index(group)
    assets = {asset["name"] for asset in index["assets"] if group == "all" or asset["group"] == group}
    count = 0
    for name, record in index["files"].items():
        if record["asset"] in assets:
            verify_file(STORE / name, record)
            count += 1
    print(f"Verified {len(snapshot['files'])} source files and {count} artifact files.")


def prepare_stage() -> Path:
    """Rebuild the exact frozen 664-feature stage from local raw data and saved caches."""
    protocol = read_json(PACKAGE / "docs/lgb_shop664_protocol.json")
    stage = STORE / ".stages/shop664"
    for name, record in protocol["stage_files"].items():
        source = PACKAGE / name
        verify_file(source, record)
        destination = stage / name
        if destination.exists():
            verify_file(destination, record)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    return stage


def load_predictions(directory: Path):
    """Read a saved hourly probability array without permitting pickle payloads."""
    import numpy as np
    if (directory / "test_predictions.npy").exists():
        return np.load(directory / "test_predictions.npy", allow_pickle=False)
    with np.load(directory / "test_predictions.npz", allow_pickle=False) as archive:
        return archive["predictions"]


def reconstruct(replay: bool, run_id: str) -> None:
    """Reconstruct all nested projections and optionally replay every component model."""
    require(Path(run_id).name == run_id and run_id not in {"", ".", ".."}, "Invalid run ID")
    output = STORE / "reproductions" / run_id
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"status": "running", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "command": sys.argv, "replay_checkpoints": replay, "training": False,
              "components": {}, "python": sys.version,
              "original_workspace_required": False}
    try:
        config = read_json(CONFIG)
        report.update(config_sha256=digest(CONFIG), adapter_sha256=digest(Path(__file__)),
                      artifact_index_sha256=digest(INDEX), cpu_replay_policy=config["cpu_replay"])
        setup()
        verify("best")
        import numpy as np
        import pandas as pd
        report["packages"] = {}
        for name in ["numpy", "pandas", "lightgbm", "xgboost", "scipy", "scikit-learn"]:
            try:
                report["packages"][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                report["packages"][name] = None
        sys.path.insert(0, str(PACKAGE))
        test = None
        if replay:
            stage = prepare_stage()
            spec = importlib.util.spec_from_file_location("relocated664", PACKAGE / "scripts/cuda_xgb664_loader.py")
            loader = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(loader)
            protocol = read_json(PACKAGE / "docs/cuda_xgb664_protocol.json")
            x, labels, train, mapping, metadata, baseline = loader.build_verified_features(stage, 2022, protocol)
            require(train["Run No"].tolist() == config["train_ids"], "Training case IDs changed")
            require(mapping["Run No"].tolist() == config["test_ids"], "Test case IDs changed")
            test = x.iloc[len(train) * 168:].copy()
            del x, labels
            report["stage"] = str(stage.relative_to(ROOT))
        from sintef_project.postprocess import project_vinje
        units = config["units"]
        values = {}
        for component in config["components"]:
            directory = STORE / component["directory"]
            manifest = read_json(directory / "manifest.json")
            require(digest(directory / "manifest.json") == component["manifest_sha256"], "Component manifest changed")
            expected = load_predictions(directory)
            require(expected.shape == (725 * 168, 14), "Wrong component array shape")
            require(np.isfinite(expected).all() and 0 <= expected.min() <= expected.max() <= 1, "Invalid probabilities")
            prediction = expected
            component_report = {"manifest_sha256": component["manifest_sha256"],
                                "checkpoint_replay_bitwise": None, "models_replayed": 0}
            report["components"][component["name"]] = component_report
            if replay:
                columns = manifest["features"]
                features = test.loc[:, columns]
                prediction = np.empty(expected.shape, dtype=expected.dtype)
                if component["library"] == "xgboost":
                    import xgboost as xgb
                    matrix = xgb.DMatrix(features, feature_names=columns, nthread=2)
                for j, unit in enumerate(units):
                    checkpoint = directory / f"final_{unit}.{component['extension']}"
                    require(digest(checkpoint) == manifest["artifacts"][checkpoint.name], "Checkpoint changed")
                    if component["library"] == "lightgbm":
                        import lightgbm as lgb
                        model = lgb.Booster(model_file=str(checkpoint))
                        prediction[:, j] = model.predict(features, num_threads=2)
                    else:
                        import xgboost as xgb
                        model = xgb.Booster()
                        model.load_model(checkpoint)
                        model.set_param({"device": "cpu", "nthread": 2})
                        prediction[:, j] = model.predict(matrix)
                component_report.update({
                    "checkpoint_replay_bitwise": prediction.tobytes() == expected.tobytes(),
                    "max_abs_error": float(np.max(np.abs(prediction.astype(np.float64) - expected))),
                    "different_values": int(np.count_nonzero(prediction != expected)),
                    "models_replayed": 14,
                })
                if component["library"] == "lightgbm":
                    require(prediction.dtype == expected.dtype and component_report["checkpoint_replay_bitwise"],
                            f"LightGBM checkpoint replay differs for {component['name']}")
                else:
                    # The original replay was CUDA, not CPU. Keep its immutable
                    # probabilities for byte-exact export after a separately
                    # reported CPU portability check. Never call this bitwise replay.
                    tolerance = config["cpu_replay"]["xgboost_max_abs_error"]
                    require(0 <= tolerance <= 1e-5, "CPU portability bound exceeds reviewed limit")
                    require(np.isfinite(prediction).all() and 0 <= prediction.min() <= prediction.max() <= 1
                            and component_report["max_abs_error"] <= tolerance,
                            f"CPU XGBoost predictions exceed the portability bound for {component['name']}")
                    component_report.update(cpu_portability_passed=True, absolute_tolerance=tolerance,
                                            export_uses_original_cuda_probabilities=True)
                    prediction = expected
            values[component["name"]] = prediction.astype(np.float64)
            print(f"Verified {component['name']}" + (" (14 checkpoints)" if replay else " (saved predictions)"), flush=True)
        combined = np.zeros((725 * 168, 14), dtype=np.float64)
        for name, weight in config["base_weights"].items():
            combined += weight * values[name]
        combined = project_vinje(combined, units)
        for step in config["nested_blends"]:
            weight = step["weight"]
            combined = project_vinje((1 - weight) * combined + weight * values[step["component"]], units)
        archived = np.load(STORE / config["best_directory"] / "test_predictions.npy", allow_pickle=False)
        require(combined.tobytes() == archived.tobytes(), "Nested blend reconstruction differs")
        template = pd.read_csv(ROOT / "data/raw/sample_submission.csv")
        columns = ["Run No"] + [f"result_committed_{unit}_t{hour}" for unit in units for hour in range(168)]
        require(template.columns.tolist() == columns, "Submission column order changed")
        matrix = combined.reshape(725, 168, 14).transpose(0, 2, 1).reshape(725, 2352)
        frame = pd.DataFrame(matrix, index=pd.Index(config["test_ids"], name="Run No"), columns=columns[1:])
        frame = frame.loc[template["Run No"]].reset_index()
        frame.to_csv(output / "submission.csv", index=False, float_format="%.9g")
        require(digest(output / "submission.csv") == config["submission_sha256"], "Exact submitted CSV not reproduced")
        report.update(status="passed", submission_sha256=config["submission_sha256"],
                      nested_blend_bitwise=True, exact_csv_bytes=True, shape=list(frame.shape))
        print(f"Exact best submission: {output / 'submission.csv'}", flush=True)
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["runtime_seconds"] = time.monotonic() - started
        (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")


def main() -> None:
    """Dispatch setup, artifact recovery, exact replay, and archived research commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("setup")
    command = sub.add_parser("import-data")
    command.add_argument("--source", type=Path, required=True,
                         help="Extracted sintef_project directory with data/, sample_submission.csv and prediction_mapping.csv")
    for name in ["fetch-artifacts", "verify"]:
        command = sub.add_parser(name)
        command.add_argument("--group", choices=["best", "research", "physics", "all"], default="best")
    for name in ["export-best", "replay-best"]:
        command = sub.add_parser(name)
        command.add_argument("--run-id", required=True)
    command = sub.add_parser("run")
    command.add_argument("script")
    command.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command == "setup":
        setup()
    elif args.command == "import-data":
        import_data(args.source)
    elif args.command == "fetch-artifacts":
        fetch_artifacts(args.group)
    elif args.command == "verify":
        verify(args.group)
    elif args.command in {"export-best", "replay-best"}:
        reconstruct(args.command == "replay-best", args.run_id)
    else:
        script = safe_path(PACKAGE / "scripts", args.script)
        require(script.is_file() and script.suffix == ".py", "Choose an existing research Python script")
        setup()
        raise SystemExit(subprocess.call([sys.executable, str(script), *args.arguments], cwd=PACKAGE))


if __name__ == "__main__":
    main()
