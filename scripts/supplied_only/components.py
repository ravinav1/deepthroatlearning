"""Assemble per-fold validation arrays of every supplied-only component and verify recorded AUCs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments/hydraulic_surrogate"))
from common import ARTIFACTS, CLONE, OUT, REPORTS, InputLedger, micro_auc, validation_npz  # noqa: E402
from sintef_project.postprocess import project_vinje  # noqa: E402

EXPECTED = {
    2021: {"anchor": 0.9951042081969695, "canonical_blend": 0.9952596524075921, "canonical_raw": 0.9942803435801887},
    2022: {"anchor": 0.9938717710733546, "canonical_blend": 0.9940509264719491},
}


def main() -> None:
    ledger = InputLedger()
    for year in (2021, 2022):
        src = {
            "anchor": ARTIFACTS / f"lgb-shop664-{year}-001/selected_validation_{year}.npz",
            "lgb664_raw": ARTIFACTS / f"lgb-shop664-{year}-001/validation_{year}.npz",
            "canonical_blend": ARTIFACTS / f"physics-trick-canonical115-{year}-001/validation_{year}.npz",
            "startup142_blend": ARTIFACTS / f"provided-startup142-{year}-001/validation_{year}.npz",
            "busbar144_blend": ARTIFACTS / f"provided-busbar144-{year}-001/validation_{year}.npz",
            "lp_context_joint": ARTIFACTS / f"lp-context-joint-comparison-001/selected_validation_{year}.npz",
            "gpu608_raw": CLONE / f"models/supplied_only/gpu/cuda-xgb608-{year}-001/validation_{year}.npz",
            "gpu664_raw": CLONE / f"models/supplied_only/gpu/cuda-xgb664-{year}-001/validation_{year}.npz",
        }
        arrays, labels, case_ids = {}, None, None
        for name, path in src.items():
            z = validation_npz(path, ledger)
            arrays[name] = z["predictions"].astype(np.float64)
            if labels is None:
                labels, case_ids = z["labels"], z["case_ids"]
            assert np.array_equal(labels, z["labels"]) and np.array_equal(case_ids, z["case_ids"]), name
        for name, run in [("canonical_raw", f"physics-trick-canonical115-{year}-001"),
                          ("startup142_raw", f"provided-startup142-{year}-001"),
                          ("busbar144_raw", f"provided-busbar144-{year}-001")]:
            path = ledger.add(ARTIFACTS / run / "raw_predictions.npy")
            arrays[name] = np.load(path).astype(np.float64)
            assert arrays[name].shape == labels.shape
        scores = {k: micro_auc(labels, v) for k, v in arrays.items()}
        scores["canonical_raw"] = micro_auc(labels, project_vinje(arrays["canonical_raw"]))
        scores["canonical_raw_unprojected"] = micro_auc(labels, arrays["canonical_raw"])
        rebuilt = project_vinje(0.75 * arrays["anchor"] + 0.25 * arrays["canonical_raw"])
        scores["rebuilt_canonical_blend"] = micro_auc(labels, rebuilt)
        for k, v in EXPECTED[year].items():
            assert abs(scores[k] - v) < 1e-9, (year, k, scores[k], v)
        assert abs(scores["rebuilt_canonical_blend"] - EXPECTED[year]["canonical_blend"]) < 1e-9
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(OUT / f"components_{year}.npz", labels=labels, case_ids=case_ids, **arrays)
        (REPORTS / f"components_{year}_scores.json").write_text(json.dumps(scores, indent=1))
        print(year, json.dumps({k: round(v, 7) for k, v in scores.items()}, indent=1))
    ledger.write(REPORTS / "components_inputs.json")


if __name__ == "__main__":
    main()
