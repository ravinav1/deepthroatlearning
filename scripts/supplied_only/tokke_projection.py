"""L6: Tokke partial-order projection test on the current supplied-only best, 2021 select, 2022 check."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blend import load_fold, plant_blend, project_tokke  # noqa: E402
from common import PLANTS, REPORTS, micro_auc  # noqa: E402

BASE = {p: {"canonical_raw": 0.25} for p in PLANTS}
MIX = [0, .25, .5, 1.0]


def scores(year: int) -> dict:
    fold = load_fold(year)
    base = plant_blend(fold["anchor"], {"canonical_raw": fold["canonical_raw"]}, BASE)
    proj = project_tokke(base)
    viol = float(np.mean((base[:, 1] < base[:, 3]) | (base[:, 3] < base[:, 4]) | (base[:, 3] < base[:, 2])))
    out = {str(m): micro_auc(fold["labels"], (1 - m) * base + m * proj) for m in MIX}
    return {"year": year, "violation_fraction": viol, "scores": out}


def main() -> None:
    s21 = scores(2021)
    pick = max(MIX, key=lambda m: (s21["scores"][str(m)], -m))
    s22 = scores(2022)
    res = {"2021": s21, "selected_mix": pick, "2022": s22,
           "accepted": bool(pick > 0 and s21["scores"][str(pick)] > s21["scores"]["0"] and s22["scores"][str(pick)] > s22["scores"]["0"])}
    (REPORTS / "l6_tokke_projection.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
