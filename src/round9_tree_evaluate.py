"""Compare a fixed Round9 feature ablation using cached historical predictions."""

import argparse
import json
import os
from datetime import datetime, timezone

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from features import ROOT
from plant_blend import read_predictions
from round6_evaluate import load_cache
from round5_compare import official_metric
from round3_frozen_evaluation import digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", nargs="+", required=True)
    parser.add_argument("--year", type=int, default=2021)
    parser.add_argument("--frozen", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    assert len(args.predictions) in (1, 2)
    output = ROOT / args.output
    assert output.stem.startswith("round9_") and not output.exists()
    mode = "frozen" if args.frozen else "native"
    path = ROOT / f"data/processed/round6_{mode}_{args.year}_baseline.npz"
    arrays, metadata = load_cache(path, args.year, mode)
    sources = dict(metadata["source_sha256"])
    sources[str(path.relative_to(ROOT))] = digest(path)
    columns, ids = metadata["columns"], pd.Series(arrays["ids"], name="Run No")
    predictions = []
    for relative in args.predictions:
        predictions.append(read_predictions(ROOT / relative, columns, ids))
        sources[relative] = digest(ROOT / relative)
    single = len(predictions) == 1
    baseline = .5 * arrays["round3"] + .5 * arrays["old299"] if single else (
        .5 * arrays["round3"] + .25 * arrays["old299"] + .25 * arrays["old673"])
    candidate = .5 * arrays["round3"] + .5 * np.mean(predictions, axis=0)
    starts = pd.DatetimeIndex(pd.to_datetime(arrays["starttime"], utc=True))
    subsets = {"all": np.ones(len(ids), dtype=bool)}
    if args.year == 2022:
        subsets["jan7_onward"] = starts >= pd.Timestamp("2022-01-07", tz="UTC")
    score = official_metric()
    rows = []
    for subset, mask in subsets.items():
        labels = arrays["labels"][mask]
        result = {"subset": subset, "cases": int(mask.sum())}
        for name, values in (("baseline", baseline[mask]), ("candidate", candidate[mask])):
            observed = float(roc_auc_score(labels.ravel(), values.ravel()))
            solution = pd.DataFrame(labels, columns=columns)
            submission = pd.DataFrame(values, columns=columns)
            for frame in (solution, submission):
                frame.insert(0, "Run No", ids.to_numpy()[mask])
            assert score(solution, submission) == observed
            result[f"{name}_micro_roc_auc"] = observed
        result["delta"] = result["candidate_micro_roc_auc"] - result["baseline_micro_roc_auc"]
        rows.append(result)
    quarters = []
    for quarter in range(1, 5):
        mask = starts.quarter == quarter
        labels = arrays["labels"][mask].ravel()
        quarters.append({"quarter": quarter, "delta": float(roc_auc_score(labels, candidate[mask].ravel())
                                                            - roc_auc_score(labels, baseline[mask].ravel()))})
    sources["src/round9_tree_evaluate.py"] = digest(ROOT / "src/round9_tree_evaluate.py")
    assert all(digest(ROOT / relative) == expected for relative, expected in sources.items())
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "year": args.year, "mode": mode,
              "comparison": "matched_single_seed299" if single else "actual_Round5_uniform_seed_pair",
              "predictions": args.predictions, "results": rows, "quarterly": quarters,
              "source_sha256": sources, "source_hashes_unchanged": True, "official_metric_verified": True,
              "competition_ids_overlap": metadata["competition_ids_overlap"],
              "limitations": "Reused historical development years. Pooled micro-AUC selection; no claim of independent holdout evidence."}
    write_json(output, report)
    print(json.dumps({"comparison": report["comparison"], "results": rows}, indent=2), flush=True)


if __name__ == "__main__":
    main()
