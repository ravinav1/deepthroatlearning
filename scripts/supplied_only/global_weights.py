"""L3b: one global weight per raw component (canonical115, startup142, busbar144) on the anchor.

Full grid selected on 2021, locked, evaluated unchanged on 2022.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blend import load_fold, plant_blend  # noqa: E402
from common import PLANTS, REPORTS, micro_auc  # noqa: E402

COMPONENTS = ["canonical_raw", "startup142_raw", "busbar144_raw"]
GRID = [[0, .1, .25, .5], [0, .1, .25, .5], [0, .1, .25]]
BASE = {p: {"canonical_raw": 0.25} for p in PLANTS}


def score(fold, combo):
    raws = {k: fold[k] for k in COMPONENTS}
    weights = {p: {k: v for k, v in zip(COMPONENTS, combo) if v} for p in PLANTS}
    return micro_auc(fold["labels"], plant_blend(fold["anchor"], raws, weights))


def main() -> None:
    f21 = load_fold(2021)
    table = []
    for combo in itertools.product(*GRID):
        if sum(combo) > 0.75:
            continue
        table.append((score(f21, combo), -sum(combo), combo))
    table.sort(key=lambda t: (-t[0], -t[1]))
    base21 = score(f21, (0.25, 0, 0))
    best = table[0]
    sel = {"base_2021": base21, "selected": dict(zip(COMPONENTS, best[2])), "selected_2021": best[0],
           "top10": [(round(s, 9), c) for s, _, c in table[:10]]}
    (REPORTS / "l3b_global_weights_selection_2021.json").write_text(json.dumps(sel, indent=1))
    f22 = load_fold(2022)
    base22 = score(f22, (0.25, 0, 0)); cand22 = score(f22, best[2])
    ev = {"base_2022": base22, "candidate_2022": cand22, "gain_2022": cand22 - base22, "gain_2021": best[0] - base21,
          "accepted": bool(best[0] > base21 and cand22 > base22)}
    (REPORTS / "l3b_global_weights_evaluation_2022.json").write_text(json.dumps(ev, indent=1))
    print("2021 base", base21, "selected", sel["selected"], best[0]); print("top5", sel["top10"][:5]); print("2022", ev)


if __name__ == "__main__":
    main()
