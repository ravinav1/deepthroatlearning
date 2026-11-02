"""Read-only diagnostics of the saved Round5 ensemble; writes a new audit report."""

import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

from pathlib import Path
import sys
import json
import hashlib
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from round6_evaluate import load_cache
from round2_diagnostics import PairDiagnostics, generator_pairs, transition_mask
from round5_compare import official_metric


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wrong_count(y, p):
    positive = p[y == 1]
    negative = np.sort(p[y == 0])
    return float(np.sum(len(negative) - .5 * (
        np.searchsorted(negative, positive, side="left") +
        np.searchsorted(negative, positive, side="right"))))


def main():
    output = ROOT / "reports/audit_20261017_diagnostics.json"
    assert not output.exists()
    records, sources = [], {}
    for year, mode in ((2021, "native"), (2022, "native"), (2022, "frozen")):
        path = ROOT / f"data/processed/round6_{mode}_{year}_baseline.npz"
        arrays, metadata = load_cache(path, year, mode)
        sources[str(path.relative_to(ROOT))] = digest(path)
        y = arrays["labels"].reshape(365, 14, 168)
        p = (.5 * arrays["round3"] + .25 * arrays["old299"] + .25 * arrays["old673"]).reshape(y.shape)
        columns = metadata["columns"]
        generators = [columns[g * 168].removeprefix("result_committed_").removesuffix("_t0") for g in range(14)]
        starts = pd.DatetimeIndex(pd.to_datetime(arrays["starttime"], utc=True))
        auc = float(roc_auc_score(y.ravel(), p.ravel()))
        solution, submission = pd.DataFrame(y.reshape(365, -1), columns=columns), pd.DataFrame(p.reshape(365, -1), columns=columns)
        for frame in (solution, submission):
            frame.insert(0, "Run No", arrays["ids"])
        assert auc == official_metric()(solution, submission)
        pairs = PairDiagnostics(y, p)
        assert abs(wrong_count(y.ravel(), p.ravel()) - pairs.wrong_pairs) < 1e-5
        matrix, composition = generator_pairs(y, p, generators)
        incident = pairs.incident.reshape(y.shape)
        per_generator = []
        for g, name in enumerate(generators):
            mask = np.zeros_like(y, dtype=bool)
            mask[:, g, :] = True
            per_generator.append({"generator": name, **pairs.subset(mask),
                "allocated_wrong_pair_share": float(incident[:, g, :].sum() / (2 * pairs.wrong_pairs)),
                "weekly_on_hours_mae": float(np.abs(p[:, g].sum(axis=1) - y[:, g].sum(axis=1)).mean()),
                "weekly_on_hours_bias": float((p[:, g].sum(axis=1) - y[:, g].sum(axis=1)).mean())})
        assert abs(sum(row["allocated_wrong_pair_share"] for row in per_generator) - 1) < 1e-12
        transition = transition_mask(y, radius=2)
        slices = {"near_transition_2h": pairs.subset(transition), "outside_transition_2h": pairs.subset(~transition)}
        for lo, hi in ((0, 24), (24, 144), (144, 168)):
            mask = np.broadcast_to((np.arange(168) >= lo) & (np.arange(168) < hi), y.shape)
            slices[f"horizon_{lo}_{hi}"] = pairs.subset(mask)
        same_week_wrong = sum(wrong_count(y[i].ravel(), p[i].ravel()) for i in range(len(y)))
        same_week_possible = sum(int(y[i].sum()) * (2352 - int(y[i].sum())) for i in range(len(y)))
        same_week_auc = 1 - same_week_wrong / same_week_possible
        monthly = []
        for month in range(1, 13):
            mask = np.broadcast_to((starts.month == month)[:, None, None], y.shape)
            monthly.append({"month": month, **pairs.subset(mask),
                "allocated_wrong_pair_share": float(incident[mask].sum() / (2 * pairs.wrong_pairs))})
        worst = []
        case_error = incident.sum(axis=(1, 2)) / (2 * pairs.wrong_pairs)
        for i in np.argsort(case_error)[-10:][::-1]:
            case_y, case_p = y[i].ravel(), p[i].ravel()
            worst.append({"case_id": int(arrays["ids"][i]), "starttime": str(starts[i]),
                "allocated_wrong_pair_share": float(case_error[i]),
                "on_fraction": float(case_y.mean()), "mean_probability": float(case_p.mean()),
                "within_case_auc": float(roc_auc_score(case_y, case_p)) if len(np.unique(case_y)) == 2 else None})
        records.append({"year": year, "mode": mode, "cases": len(y), "micro_auc": auc,
            "pair_composition": composition,
            "generator_pair_matrix": matrix,
            "generators": per_generator, "slices": slices, "months": monthly,
            "same_case_pair_auc": same_week_auc,
            "same_case_pair_availability_fraction": same_week_possible / pairs.total_pairs,
            "same_case_wrong_pair_fraction": same_week_wrong / pairs.wrong_pairs,
            "cross_case_pair_auc": 1 - (pairs.wrong_pairs - same_week_wrong) / (pairs.total_pairs - same_week_possible),
            "worst_cases_by_allocated_pair_error": worst,
            "source_metadata": {k: metadata[k] for k in ("round5_micro_roc_auc", "competition_ids_overlap")}})
        print(json.dumps({"year": year, "mode": mode, "micro_auc": auc,
            "worst_generators_by_allocated_error": sorted(per_generator, key=lambda row: row["allocated_wrong_pair_share"], reverse=True)[:5],
            "slices": slices, "same_case_pair_auc": same_week_auc}, indent=2), flush=True)
    sources[str(Path(__file__).relative_to(ROOT))] = digest(Path(__file__))
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "evaluations": records,
        "source_sha256": sources, "official_metric_verified": True, "fitting_performed": False,
        "definitions": {
            "allocated_wrong_pair_share": "Half of every wrong/tied positive-negative pair is allocated to each endpoint. Shares across disjoint generator/month groups sum to one.",
            "incident_wrong_pair_share": "Fraction of wrong/tied pairs touching the slice; overlapping slices cannot be summed.",
            "same_case_pair_auc": "AUC restricted to positive-negative pairs in the same case, weighted by available pairs; diagnostic, not the competition metric.",
            "limitations": "Reused historical development years; labels define diagnostic slices only. Pair errors include half-weight ties. No model or calibration fitting, no test labels."}}
    with output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
