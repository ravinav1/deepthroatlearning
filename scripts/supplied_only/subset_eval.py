"""Compare startup-style variants on each fold overall and on the high-volatility half of cases."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blend import load_fold  # noqa: E402
from common import OUT, REPORTS, micro_auc  # noqa: E402
from sintef_project.postprocess import project_vinje  # noqa: E402


def main() -> None:
    tags = sys.argv[1:] or ["startup16", "startup16vw"]
    cols = json.loads((OUT / "features664/columns.json").read_text())["columns"]
    x = np.load(OUT / "features664/features664.npy", mmap_mode="r"); j = cols.index("norm_price_week_std")
    week_std = np.asarray(x[:, j]).reshape(3647, 168)[:, 0]
    out = {}
    for year, (a, b) in {2021: (2192, 2557), 2022: (2557, 2922)}.items():
        fold = load_fold(year); labels, anchor = fold["labels"], fold["anchor"]
        std = week_std[a:b]; hi = np.repeat(std >= np.median(std), 168)
        out[year] = {"anchor_all": micro_auc(labels, anchor), "anchor_hi": micro_auc(labels[hi], anchor[hi]),
                     "anchor_lo": micro_auc(labels[~hi], anchor[~hi])}
        for tag in tags:
            raw = np.load(OUT / f"{tag}_{year}/raw_predictions.npy")
            p = project_vinje(0.75 * anchor + 0.25 * raw)
            out[year][tag] = {"all": micro_auc(labels, p), "hi": micro_auc(labels[hi], p[hi]), "lo": micro_auc(labels[~hi], p[~hi]),
                              "standalone_hi": micro_auc(labels[hi], project_vinje(raw)[hi])}
    (REPORTS / f"subset_eval_{'_'.join(tags)}.json").write_text(json.dumps(out, indent=1))
    for year in out:
        print(year, "anchor all/hi/lo", round(out[year]["anchor_all"], 7), round(out[year]["anchor_hi"], 7), round(out[year]["anchor_lo"], 7))
        for tag in tags:
            r = out[year][tag]; print(f"  {tag:14s} all {r['all']:.7f} hi {r['hi']:.7f} lo {r['lo']:.7f}  (standalone hi {r['standalone_hi']:.7f})")


if __name__ == "__main__":
    main()
