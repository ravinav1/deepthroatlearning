"""D1 diagnostic: how often do overlapping cases agree on the same calendar hour?

Reads only the supplied label file. Writes reports/supplied_only/d1_overlap_agreement.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HORIZON, UNITS, REPORTS, InputLedger, load_labels_array  # noqa: E402


def main() -> None:
    ledger = InputLedger()
    cases, labels = load_labels_array(ledger)  # (n, 14, 168)
    starts = cases["starttime"]
    start_hours = ((starts - starts.iloc[0]).dt.total_seconds() / 3600).round().astype(int).to_numpy()
    n = len(cases)
    by_offset: dict[int, dict] = {}
    for d in range(1, 7):
        offset = 24 * d
        agree = np.zeros(len(UNITS)); total = np.zeros(len(UNITS))
        pos_pairs = np.zeros(len(UNITS)); pos_agree = np.zeros(len(UNITS))
        # case i (start s) and case j with start s + 24d overlap on hours [24d, 168) of i == [0, 168-24d) of j
        lookup = {h: i for i, h in enumerate(start_hours)}
        for i in range(n):
            j = lookup.get(start_hours[i] + offset)
            if j is None:
                continue
            a = labels[i, :, offset:]
            b = labels[j, :, :HORIZON - offset]
            agree += (a == b).sum(axis=1); total += a.shape[1]
            pos = (a == 1) | (b == 1)
            pos_pairs += pos.sum(axis=1); pos_agree += ((a == 1) & (b == 1)).sum(axis=1)
        by_offset[d] = {
            "agreement_rate": float(agree.sum() / total.sum()),
            "agreement_by_unit": {u: float(agree[k] / total[k]) for k, u in enumerate(UNITS)},
            "positive_jaccard": float(pos_agree.sum() / max(pos_pairs.sum(), 1)),
            "positive_jaccard_by_unit": {u: float(pos_agree[k] / max(pos_pairs[k], 1)) for k, u in enumerate(UNITS)},
            "compared_hours": int(total.sum()),
        }
    # Agreement as a function of horizon position of the earlier case (end-of-horizon effects).
    pos_profile = []
    d = 1; offset = 24
    lookup = {h: i for i, h in enumerate(start_hours)}
    agree_h = np.zeros(HORIZON - offset); total_h = np.zeros(HORIZON - offset)
    for i in range(n):
        j = lookup.get(start_hours[i] + offset)
        if j is None:
            continue
        a = labels[i, :, offset:]; b = labels[j, :, :HORIZON - offset]
        agree_h += (a == b).sum(axis=0); total_h += a.shape[0]
    pos_profile = (agree_h / total_h).tolist()
    base_rate = float(labels.mean())
    result = {"cases": int(n), "base_on_rate": base_rate, "by_day_offset": by_offset,
              "day1_agreement_by_hour_of_earlier_case": pos_profile,
              "note": "Agreement between the two overlapping cases on the same calendar hour; labels only."}
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "d1_overlap_agreement.json").write_text(json.dumps(result, indent=1))
    ledger.write(REPORTS / "d1_overlap_agreement_inputs.json")
    for d, r in by_offset.items():
        print(f"offset {d}d: agreement {r['agreement_rate']:.5f}  positive-jaccard {r['positive_jaccard']:.5f}")
    print("day-1 agreement by earlier-case hour (every 12h):", [round(x, 4) for x in pos_profile[::12]])
    print("worst units at 1d:", sorted(by_offset[1]["agreement_by_unit"].items(), key=lambda kv: kv[1])[:4])


if __name__ == "__main__":
    main()
