"""Low-memory complete-cache/order/normalization verification for the joint LP."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shlex
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from sintef_project.features import _lookup
from sintef_project.lp_features import HydroCaseData
from sintef_project.lp_joint_plant import add_lp_joint_features
from sintef_project.normalized import add_normalized_features


def main() -> None:
    """Verify every cached row with only price/hour base columns in memory."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="lp-joint-verification-001")
    args = parser.parse_args()
    out = ROOT / "artifacts" / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    tick = time.monotonic()
    data = HydroCaseData(ROOT)
    hours = np.tile(np.arange(168, dtype=np.int16), len(data.cases))
    timestamps = pd.DatetimeIndex(np.repeat(pd.DatetimeIndex(data.cases.starttime).array, 168)) + pd.to_timedelta(hours, unit="h")
    x = pd.DataFrame({"horizon_hour": hours, "price": _lookup(data.price, timestamps, "joint-cache verification price")[:, 0]})
    x, spec = add_lp_joint_features(x, ROOT)
    assert x.shape == (3647 * 168, 89)
    assert list(x.columns[2:]) == spec["columns"]
    normalized, normalization = add_normalized_features(x, ROOT)
    assert normalized.shape == (3647 * 168, 90)
    for name in spec["columns"]:
        assert np.isfinite(x[name].to_numpy()).all(), name
        assert np.array_equal(x[name].to_numpy(), normalized[name].to_numpy()), name
        if name.startswith("lpjoint_on_fraction_"):
            assert x[name].min() >= -1e-6 and x[name].max() <= 1 + 1e-6, name
    for plant, count in [("Tokke", 4), ("Vinje", 3)]:
        names = [name for name in spec["columns"] if name.startswith(f"lpjoint_on_fraction_{plant}_")]
        expected = x[names].sum(axis=1).to_numpy()
        actual = x[f"lpjoint_active_unit_count_{plant}"].to_numpy()
        np.testing.assert_allclose(expected, actual, atol=1e-6)
        assert actual.min() >= -1e-6 and actual.max() <= count + 1e-6
    report = {"status": "passed", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "command": shlex.join([sys.executable, *sys.argv]), "runtime_seconds": time.monotonic() - tick,
              "checks": ["complete612696 case-hour order", "strict source/data/cache hashes", "exact87-column feature order", "all finite", "dimensionless columns unchanged by normalization", "joint on-fraction bounds and plant-count identities"],
              "lightweight_raw_shape": list(x.shape), "lightweight_normalized_shape": list(normalized.shape),
              "expected_full_training_shape": [612696, 404],
              "full_training_shape_note": "Expected from231base+72physics+13context+87joint+1normalization; complete large feature frame deliberately not built in this memory-bounded check.",
              "joint_spec": spec, "normalization_spec": normalization,
              "environment": {name: importlib.metadata.version(name) for name in ["numpy", "pandas"]},
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    shutil.copy2(Path(__file__), out / Path(__file__).name)
    (out / "manifest.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({key: report[key] for key in ["status", "runtime_seconds", "lightweight_raw_shape", "expected_full_training_shape"]}))


if __name__ == "__main__":
    main()
