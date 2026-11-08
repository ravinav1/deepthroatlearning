"""L3: per-plant blend weights over supplied-only raw components, 2021 select, 2022 check."""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blend import load_fold, plant_blend  # noqa: E402
from common import PLANTS, REPORTS, micro_auc  # noqa: E402

COMPONENTS = ["canonical_raw", "startup142_raw", "busbar144_raw"]
GRID = {"canonical_raw": [0, .1, .25, .5], "startup142_raw": [0, .1, .25], "busbar144_raw": [0, .1, .25]}
MAX_TOTAL = 0.75
BASE = {p: {"canonical_raw": 0.25} for p in PLANTS}  # current supplied-only best


def search(year: int) -> dict:
    fold = load_fold(year)
    labels, anchor = fold["labels"], fold["anchor"]
    raws = {k: fold[k] for k in COMPONENTS}
    weights = {p: dict(BASE[p]) for p in PLANTS}
    best = micro_auc(labels, plant_blend(anchor, raws, weights))
    base_score = best
    history = [{"pass": 0, "score": best, "weights": json.loads(json.dumps(weights))}]
    combos = [dict(zip(COMPONENTS, c)) for c in itertools.product(*(GRID[k] for k in COMPONENTS))
              if sum(c) <= MAX_TOTAL]
    for pass_no in (1, 2):
        improved = False
        for plant in PLANTS:
            candidates = []
            for combo in combos:
                trial = {p: dict(weights[p]) for p in PLANTS}
                trial[plant] = {k: v for k, v in combo.items() if v}
                score = micro_auc(labels, plant_blend(anchor, raws, trial))
                candidates.append((score, -sum(combo.values()), combo))
            candidates.sort(key=lambda t: (-t[0], -t[1]))
            top_score, _, top = candidates[0]
            if top_score > best + 1e-12:
                best = top_score; weights[plant] = {k: v for k, v in top.items() if v}; improved = True
            history.append({"pass": pass_no, "plant": plant, "best_so_far": best,
                            "chosen": weights[plant], "top3": [(round(s, 9), c) for s, _, c in candidates[:3]]})
        if not improved:
            break
    return {"year": year, "base_score": base_score, "selected_score": best, "weights": weights, "history": history}


def evaluate(year: int, weights: dict) -> dict:
    fold = load_fold(year)
    raws = {k: fold[k] for k in COMPONENTS}
    base = micro_auc(fold["labels"], plant_blend(fold["anchor"], raws, BASE))
    cand = micro_auc(fold["labels"], plant_blend(fold["anchor"], raws, weights))
    return {"year": year, "base_score": base, "candidate_score": cand, "gain": cand - base}


def main() -> None:
    sel = search(2021)
    lock = REPORTS / "l3_plant_weights_selection_2021.json"
    lock.write_text(json.dumps(sel, indent=1))
    print("2021 base", sel["base_score"], "selected", sel["selected_score"], "gain", sel["selected_score"] - sel["base_score"])
    print(json.dumps(sel["weights"]))
    ev = evaluate(2022, sel["weights"])
    ev["accepted"] = bool(sel["selected_score"] > sel["base_score"] and ev["gain"] > 0)
    (REPORTS / "l3_plant_weights_evaluation_2022.json").write_text(json.dumps(ev, indent=1))
    print("2022", ev)


if __name__ == "__main__":
    main()
