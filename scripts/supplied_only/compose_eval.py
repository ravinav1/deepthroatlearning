"""Evaluate composed supplied-only recipes on both development folds (2021 selects, 2022 checks)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blend import load_fold, plant_blend, project_tokke  # noqa: E402
from common import OUT, PLANTS, REPORTS, micro_auc  # noqa: E402
from legacy_member import legacy_array  # noqa: E402


def startup(year: int, seeds: list[int]) -> np.ndarray:
    arrays = []
    for s in seeds:
        d = OUT / (f"startup16_{year}" if s == 4173 else f"startup16_{year}_seed{s}")
        arrays.append(np.load(d / "raw_predictions.npy"))
    return np.mean(arrays, axis=0)


def recipe(fold: dict, su: np.ndarray, w_su: float, leg: np.ndarray | None, w_leg: float, tokke: bool) -> np.ndarray:
    out = plant_blend(fold["anchor"], {"su": su}, {p: {"su": w_su} for p in PLANTS})
    if leg is not None and w_leg:
        out = plant_blend(out, {"leg": leg}, {p: {"leg": w_leg} for p in PLANTS})
    return project_tokke(out) if tokke else out


def main() -> None:
    seeds = [int(s) for s in sys.argv[1:]] or [4173]
    results = {}
    for year in (2021, 2022):
        fold = load_fold(year); su = startup(year, seeds); leg = legacy_array(year, [299, 673], fold["case_ids"])
        results[year] = {}
        for w_su in (.1, .25, .5):
            for w_leg in (0, .05):
                for tokke in (False, True):
                    key = f"su{w_su}_leg{w_leg}_tokke{int(tokke)}"
                    results[year][key] = micro_auc(fold["labels"], recipe(fold, su, w_su, leg, w_leg, tokke))
        results[year]["anchor"] = micro_auc(fold["labels"], fold["anchor"])
    pick = max((k for k in results[2021] if k != "anchor"), key=lambda k: results[2021][k])
    summary = {"seeds": seeds, "selected_on_2021": pick, "2021": results[2021], "2022": results[2022],
               "gain_2021": results[2021][pick] - results[2021]["anchor"], "gain_2022": results[2022][pick] - results[2022]["anchor"]}
    tag = "_".join(map(str, seeds))
    (REPORTS / f"compose_eval_seeds{tag}.json").write_text(json.dumps(summary, indent=1))
    print("selected", pick, "2021", results[2021][pick], "2022", results[2022][pick], "gains", summary["gain_2021"], summary["gain_2022"])
    for k in sorted(results[2021], key=lambda k: -results[2021][k])[:8]:
        print(f"{k:28s} 2021 {results[2021][k]:.7f}  2022 {results[2022][k]:.7f}")


if __name__ == "__main__":
    main()
