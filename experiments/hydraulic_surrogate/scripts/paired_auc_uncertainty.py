"""Measure descriptive uncertainty without changing a frozen candidate recipe."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import sys
import time
import traceback

for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ[key] = "1"
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sintef_project.auc_uncertainty import WeightedAUC, circular_block_weights


def digest(path: Path) -> str:
    """Hash the exact observed artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Use paired 28-day circular bootstrap weights on complete daily cases."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=200)
    args = parser.parse_args()
    if args.replicates < 2:
        raise ValueError("At least two replicates required")
    out = ROOT / "artifacts" / args.run_id
    out.mkdir(exist_ok=False)
    started = time.time()
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "command": shlex.join([sys.executable, *sys.argv]), "seed": 4173,
              "block_days": 28, "replicates": args.replicates,
              "method": "Paired circular moving-block bootstrap of whole daily cases; fixed models and predictions; percentile 95% interval of candidate-minus-baseline pooled micro AUC",
              "limitations": "Descriptive conditional uncertainty only. Reused development years and adaptive candidate selection are not corrected. Training variability and cross-year distribution shift are not measured. No weights, models or acceptance rule are tuned with this report.",
              "environment": {"python": sys.version, "platform": platform.platform(),
                              "packages": {k: importlib.metadata.version(k) for k in ["numpy", "pandas", "scikit-learn"]}},
              "inputs": {}, "source": {}, "kaggle": None}
    try:
        for relative in ["scripts/paired_auc_uncertainty.py", "sintef_project/auc_uncertainty.py"]:
            path = ROOT / relative
            report["source"][relative] = digest(path)
            target = out / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        arrays = []
        ids = units = labels = None
        for path in [args.baseline, args.candidate]:
            report["inputs"][str(path)] = digest(path)
            with np.load(path, allow_pickle=False) as data:
                if ids is None:
                    ids, units, labels = data["case_ids"], data["units"].tolist(), data["labels"]
                elif not (np.array_equal(ids, data["case_ids"]) and units == data["units"].tolist()
                          and np.array_equal(labels, data["labels"])):
                    raise ValueError("Labels, case IDs or generator order differ")
                p = data["predictions"].astype(np.float64)
                if p.shape != (len(ids) * 168, len(units)) or p.shape != labels.shape:
                    raise ValueError("Incomplete case-hour-unit arrays")
                if ((p < 0) | (p > 1)).any():
                    raise ValueError("Expected probability predictions")
                arrays.append(p)
        metadata_path = ROOT / "sintef_project/data/kernel/Unit_commitment_decisions.csv"
        report["inputs"][str(metadata_path.relative_to(ROOT))] = digest(metadata_path)
        metadata = pd.read_csv(metadata_path, usecols=["Run No", "starttime"]).set_index("Run No")
        dates = pd.DatetimeIndex(pd.to_datetime(metadata.loc[ids, "starttime"], utc=True))
        if len(ids) < 28 or not np.all(np.diff(dates.as_unit("ns").asi8) == 86_400_000_000_000):
            raise ValueError("Block resampling requires consecutive daily cases")
        prepared = [WeightedAUC.prepare(labels, p, len(ids)) for p in arrays]
        point = np.array([p.score(np.ones(len(ids))) for p in prepared])
        reference = [roc_auc_score(labels.ravel(), p.ravel()) for p in arrays]
        np.testing.assert_allclose(point, reference, atol=1e-13, rtol=0)
        rng = np.random.default_rng(4173)
        values = np.empty((args.replicates, 2))
        for index in range(args.replicates):
            weights = circular_block_weights(rng, len(ids), 28)
            values[index] = [p.score(weights) for p in prepared]
        difference = values[:, 1] - values[:, 0]
        np.savez_compressed(out / "bootstrap.npz", auc=values, difference=difference)
        report.update(status="complete", case_ids=ids.tolist(), units=units,
                      dates=[str(dates[0]), str(dates[-1])],
                      baseline_auc=float(point[0]), candidate_auc=float(point[1]),
                      difference=float(point[1] - point[0]),
                      difference_percentile_95=np.quantile(difference, [.025, .975]).tolist(),
                      bootstrap_fraction_positive=float(np.mean(difference > 0)),
                      bootstrap_standard_deviation=float(difference.std(ddof=1)))
        print(json.dumps({k: report[k] for k in ["difference", "difference_percentile_95", "bootstrap_fraction_positive"]}), flush=True)
    except BaseException:
        report.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        report["runtime_seconds"] = time.time() - started
        report["artifacts"] = {str(p.relative_to(out)): digest(p) for p in out.rglob("*") if p.is_file()}
        (out / "manifest.json").write_text(json.dumps(report, indent=2))
        with (ROOT / "docs/experiments.md").open("a") as handle:
            handle.write(f'\n## {args.run_id} — {report["timestamp_utc"]}\n\n'
                         f'Status: {report["status"]}. Command: `{report["command"]}`. '
                         f'Runtime {report["runtime_seconds"]:.2f}s. Fixed-model paired28-day circular case bootstrap, seed4173, {args.replicates}replicates; no fitting, selection, early stopping or Kaggle submission. '
                         f'Difference {report.get("difference")}; descriptive95%percentile interval {report.get("difference_percentile_95")}. '
                         'Adaptive research reuse, training uncertainty and distribution shift are not accounted for. '
                         f'Full source/input/artifact hashes, IDs, dates, versions and bootstrap draws: `artifacts/{args.run_id}/manifest.json`.\n')


if __name__ == "__main__":
    main()
