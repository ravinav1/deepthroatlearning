"""L4 evaluation: canonical115 seed-bag mean against the single seed and on top of the L3b recipe."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blend import load_fold, plant_blend  # noqa: E402
from common import OUT, PLANTS, REPORTS, micro_auc  # noqa: E402
from sintef_project.postprocess import project_vinje  # noqa: E402

L3B = {p: {"startup142_raw": 0.25, "busbar144_raw": 0.1} for p in PLANTS}


def main() -> None:
    year = int(sys.argv[1]); seeds = [int(s) for s in sys.argv[2:]]
    fold = load_fold(year)
    raws = {4173: fold["canonical_raw"]}
    for s in seeds:
        raws[s] = np.load(OUT / f"seed_bag/canonical115_{year}_seed{s}/raw_predictions.npy")
    per_seed = {str(s): micro_auc(fold["labels"], project_vinje(r)) for s, r in raws.items()}
    bag = np.mean(list(raws.values()), axis=0)
    out = {"year": year, "per_seed_projected_auc": per_seed, "bag_projected_auc": micro_auc(fold["labels"], project_vinje(bag))}
    # on the plain anchor at the historical 0.25 weight, single seed versus bag
    single = plant_blend(fold["anchor"], {"c": fold["canonical_raw"]}, {p: {"c": 0.25} for p in PLANTS})
    bagged = plant_blend(fold["anchor"], {"c": bag}, {p: {"c": 0.25} for p in PLANTS})
    out["anchor_plus_single_0.25"] = micro_auc(fold["labels"], single)
    out["anchor_plus_bag_0.25"] = micro_auc(fold["labels"], bagged)
    # on top of L3b
    base = plant_blend(fold["anchor"], {k: fold[k] for k in ("startup142_raw", "busbar144_raw")}, L3B)
    out["l3b_base"] = micro_auc(fold["labels"], base)
    out["l3b_plus_bag"] = {str(w): micro_auc(fold["labels"], plant_blend(base, {"c": bag}, {p: {"c": w} for p in PLANTS})) for w in (.1, .25)}
    (REPORTS / f"l4_seed_bag_{year}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
