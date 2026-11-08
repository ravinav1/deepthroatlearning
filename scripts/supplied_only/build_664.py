"""Rebuild the frozen 664-column supplied-only feature table from the research modules and restored caches."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

CLONE = Path(__file__).resolve().parents[2]
ROOT = CLONE / "experiments/hydraulic_surrogate"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, REPORTS, InputLedger  # noqa: E402
from sintef_project.features import build_features  # noqa: E402
from sintef_project.physics import add_physics_features  # noqa: E402
from sintef_project.context_features import add_context_features  # noqa: E402
from sintef_project.lp_features import add_lp_features  # noqa: E402
from sintef_project.lp_context import add_lp_context_features  # noqa: E402
from sintef_project.normalized import add_normalized_features  # noqa: E402
from sintef_project.lp_joint_plant import add_lp_joint_features  # noqa: E402
from sintef_project.lp_shop_features import add_shop_features  # noqa: E402


def main() -> None:
    tick = time.monotonic()
    protocol = json.loads((ROOT / "docs/lgb_shop664_protocol.json").read_text())
    expected = protocol["feature_columns"]
    x, labels, train, mapping, meta = build_features(ROOT)
    x, _ = add_physics_features(x, ROOT)
    x, _ = add_context_features(x, ROOT)
    x, _ = add_lp_features(x, ROOT)
    x, _ = add_lp_context_features(x, ROOT)
    x, _ = add_normalized_features(x, ROOT)
    x, _ = add_lp_joint_features(x, ROOT)
    assert x.columns.tolist() == expected[:608], "608 prefix order differs from the frozen protocol"
    x, shop_spec = add_shop_features(x, ROOT, ROOT / "artifacts/lp-shop-full-001")
    assert x.columns.tolist() == expected and x.shape == (3647 * 168, 664)
    values = x.to_numpy(dtype=np.float32)
    assert np.isfinite(values).all()
    out = OUT / "features664"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "features664.npy", values)
    y = np.column_stack([labels[u] for u in meta["units"]]).astype(np.uint8)
    np.save(out / "labels_train.npy", y)
    cases = pd.concat([train, mapping], ignore_index=True)
    cases["starttime"] = pd.to_datetime(cases["starttime"], utc=True)
    cases.to_csv(out / "cases.csv", index=False)
    (out / "columns.json").write_text(json.dumps({"columns": expected, "units": meta["units"], "shop_cache": shop_spec["manifest_sha256"],
                                                 "n_train_cases": int(len(train)), "n_test_cases": int(len(mapping)),
                                                 "seconds": time.monotonic() - tick}, indent=1))
    print("built", values.shape, "labels", y.shape, "cases", len(cases), f"{time.monotonic() - tick:.0f}s")


if __name__ == "__main__":
    main()
