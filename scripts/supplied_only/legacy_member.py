"""L5: blend the retrained Round10-style separate-generator LightGBM into the current candidate.

Candidate base = L3b recipe: P(0.65 A0 + 0.25 startup142_raw + 0.10 busbar144_raw).
Legacy member = mean over available seeds of the old-lineage predictions (113+2 features,
economic rescaling, per-generator models). Weight grid selected on 2021, checked on 2022.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blend import load_fold, plant_blend  # noqa: E402
from common import CLONE, HORIZON, PLANTS, REPORTS, UNITS, micro_auc  # noqa: E402

L3B = {p: {"startup142_raw": 0.25, "busbar144_raw": 0.1} for p in PLANTS}
GRID = [0, .05, .1, .15, .2, .3]


def legacy_array(year: int, seeds: list[int], case_ids: np.ndarray) -> np.ndarray:
    columns = [f"result_committed_{u}_t{h}" for u in UNITS for h in range(HORIZON)]
    arrays = []
    for seed in seeds:
        path = CLONE / f"data/processed/round10_legacy_so_seed{seed}_{year}_validation_predictions.csv"
        frame = pd.read_csv(path).set_index("Run No").loc[case_ids, columns]
        arrays.append(frame.to_numpy(dtype=np.float64).reshape(len(case_ids), len(UNITS), HORIZON).transpose(0, 2, 1).reshape(-1, len(UNITS)))
    return np.mean(arrays, axis=0)


BASE_MODE = "l3b"


def candidate(fold: dict, legacy: np.ndarray, w: float) -> np.ndarray:
    base = fold["anchor"] if BASE_MODE == "anchor" else plant_blend(fold["anchor"], {k: fold[k] for k in ("startup142_raw", "busbar144_raw")}, L3B)
    return plant_blend(base, {"legacy": legacy}, {p: {"legacy": w} for p in PLANTS}) if w else base


def main() -> None:
    global BASE_MODE
    args = [a for a in sys.argv[1:]]
    if "--base-anchor" in args:
        BASE_MODE = "anchor"; args.remove("--base-anchor")
    seeds = [int(s) for s in args] or [299]
    tag = "" if BASE_MODE == "l3b" else "_on_anchor"
    f21 = load_fold(2021)
    leg21 = legacy_array(2021, seeds, f21["case_ids"])
    standalone21 = micro_auc(f21["labels"], leg21)
    scores21 = {str(w): micro_auc(f21["labels"], candidate(f21, leg21, w)) for w in GRID}
    pick = max(GRID, key=lambda w: (scores21[str(w)], -w))
    sel = {"seeds": seeds, "legacy_standalone_2021": standalone21, "scores_2021": scores21, "selected_weight": pick}
    (REPORTS / f"l5_legacy_selection_2021{tag}.json").write_text(json.dumps(sel, indent=1))
    print("2021 standalone", standalone21, "grid", scores21, "pick", pick)
    f22 = load_fold(2022)
    leg22 = legacy_array(2022, seeds, f22["case_ids"])
    base22 = micro_auc(f22["labels"], candidate(f22, leg22, 0)); cand22 = micro_auc(f22["labels"], candidate(f22, leg22, pick))
    ev = {"legacy_standalone_2022": micro_auc(f22["labels"], leg22), "base_2022": base22, "candidate_2022": cand22,
          "gain_2022": cand22 - base22, "gain_2021": scores21[str(pick)] - scores21["0"],
          "accepted": bool(pick > 0 and scores21[str(pick)] > scores21["0"] and cand22 > base22)}
    (REPORTS / f"l5_legacy_evaluation_2022{tag}.json").write_text(json.dumps(ev, indent=1))
    print("2022", ev)


if __name__ == "__main__":
    main()
