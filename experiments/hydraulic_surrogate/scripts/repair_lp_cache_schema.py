"""Convert trusted manifest column names to safe Unicode archive metadata.

The numeric LP features are read without pickle and remain byte-for-byte equal.
This repairs pandas3 Index serialization in the frozen pool_lp_v1 cache builder.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import sys

import numpy as np


def digest(path: Path) -> str:
    """Return SHA-256."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Rewrite only metadata encoding and preserve complete repair provenance."""
    root = Path(__file__).resolve().parents[1]
    cache_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "artifacts/lp-full-001"
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    path = cache_dir / "features.npz"
    before_hash = digest(path)
    if before_hash != manifest["artifacts"]["features.npz"]:
        raise ValueError("Archive checksum changed before schema repair")
    with np.load(path, allow_pickle=False) as archive:
        values = archive["features"]
        ids = archive["case_ids"]
        units = archive["units"]
    numeric_hash = hashlib.sha256(values.tobytes()).hexdigest()
    replacement = cache_dir / "features_repaired.npz"
    np.savez_compressed(replacement, features=values, case_ids=ids,
                        columns=np.asarray(manifest["feature_names"], dtype=str), units=np.asarray(units, dtype=str))
    with np.load(replacement, allow_pickle=False) as checked:
        if hashlib.sha256(checked["features"].tobytes()).hexdigest() != numeric_hash:
            raise ValueError("Numeric features changed during schema repair")
        if checked["columns"].tolist() != manifest["feature_names"]:
            raise ValueError("Column metadata mismatch")
    replacement.replace(path)
    copied = cache_dir / "repair_lp_cache_schema.py"
    shutil.copy2(Path(__file__), copied)
    manifest["schema_repair"] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "reason": "pandas3 Index yielded object-dtype column names; safe allow_pickle=False load rejected them",
        "source_sha256": digest(Path(__file__)), "old_archive_sha256": before_hash,
        "new_archive_sha256": digest(path), "unchanged_numeric_feature_sha256": numeric_hash,
        "procedure": "Read only numeric features, IDs and Unicode units with allow_pickle=False; reconstruct Unicode columns from recorded manifest; rewrite NPZ; verify numeric-byte hash unchanged",
    }
    manifest["artifacts"]["features.npz"] = digest(path)
    manifest["artifacts"][copied.name] = digest(copied)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print("SCHEMA REPAIR PASS", manifest["schema_repair"]["new_archive_sha256"])


if __name__ == "__main__":
    main()
