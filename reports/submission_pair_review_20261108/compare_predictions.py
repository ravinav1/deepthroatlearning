"""Compare existing submitted probabilities without loading labels or scoring AUC."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CANDIDATES = {
    "3": ("models/submission_pair_review_20261108/so-bag2-anchor-startup16-001.csv", "f7b0b52fc54e93e1db6171fdf85a7955486c3439897911291eef98c70a4f716a"),
    "5": ("models/submission_pair_review_20261108/so-primary-anchor-startup16-001.csv", "a45f8001a87afbc67b17d95fb39c9f1982593d24f91887d5747fec1f3c367ceb"),
    "2": ("models/survivor_combo_20261104/deployment/submission.csv", "324863d5689285ef8a56caa64772d756cfbda314e9e793ecb1042a2f439b2821"),
    "17": ("models/hydraulic_surrogate/submission-lgb-shop664-anchor-blend-001/submission.csv", "1c1d173816127997e711d54c8269406830666af66b6f4eb6d24f4f227b70d496"),
    "19": ("models/hydraulic_surrogate/submission-cuda-xgb664-anchor-blend-001/submission_standalone_projected.csv", "4c96e9f69459417fd138af7d33f5339275bb5ef29ffdd59f0a4590ee8f60ae42"),
    "25": ("models/hydraulic_surrogate/submission-lp-context-joint-blend-001/submission.csv", "f7907cce9d7348a1c77959e64d5e6099303123846c0648b37c816bb495411fcf"),
    "32": ("submissions/submission_round5_lgb_generators.csv", "5b736f51f14028eaee5382124314fd4c368b72f546116d0a6ac053fafa086a29"),
}


def sha256(path: Path) -> str:
    """Hash an exact input file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def diagnostics(a: np.ndarray, b: np.ndarray) -> dict:
    """Describe probability differences; no labels are accepted."""
    a, b = a.ravel(), b.ravel()
    delta = np.abs(a - b)
    result = {
        "n_outputs": len(a),
        "pearson": float(np.corrcoef(a, b)[0, 1]),
        "spearman": float(np.corrcoef(rankdata(a), rankdata(b))[0, 1]),
        "mean_absolute_difference": float(delta.mean()),
        "p95_absolute_difference": float(np.quantile(delta, 0.95)),
        "p99_absolute_difference": float(np.quantile(delta, 0.99)),
        "threshold_0_5_disagreement_fraction": float(np.mean((a >= 0.5) != (b >= 0.5))),
        "fraction_absolute_difference_gt_0_01": float(np.mean(delta > 0.01)),
        "fraction_absolute_difference_gt_0_05": float(np.mean(delta > 0.05)),
    }
    return result


def main() -> None:
    """Validate allowed prediction files and write descriptive comparisons."""
    result_path = OUT / "prediction_diagnostics.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    template = pd.read_csv(ROOT / "data/raw/sample_submission.csv")
    mapping = pd.read_csv(ROOT / "data/raw/prediction_mapping.csv")
    assert np.array_equal(template["Run No"], mapping["Run No"])
    assert template.shape == (725, 2353)
    units = [template.columns[1 + i * 168].removeprefix("result_committed_").removesuffix("_t0") for i in range(14)]
    groups = {"all": np.ones(725, dtype=bool), "2023_public": mapping.starttime.str.startswith("2023").to_numpy(), "2024_private": mapping.starttime.str.startswith("2024").to_numpy()}
    hashes = {}
    for number, (relative, expected) in CANDIDATES.items():
        actual = sha256(ROOT / relative)
        assert actual == expected, (number, actual, expected)
        hashes[relative] = actual

    def load(number: str) -> np.ndarray:
        frame = pd.read_csv(ROOT / CANDIDATES[number][0])
        assert frame.columns.tolist() == template.columns.tolist()
        assert np.array_equal(frame["Run No"], template["Run No"])
        values = frame.iloc[:, 1:].to_numpy(dtype=np.float64)
        assert np.isfinite(values).all() and values.min() >= 0 and values.max() <= 1
        return values.reshape(725, 14, 168)

    reference = load("3")
    comparisons = {}
    for number in CANDIDATES:
        if number == "3":
            continue
        candidate = load(number)
        comparisons[number] = {}
        for group, mask in groups.items():
            a, b = reference[mask], candidate[mask]
            values = diagnostics(a, b)
            uncertain = (a > 0.1) & (a < 0.9)
            values["reference_0_1_to_0_9_subset"] = diagnostics(a[uncertain], b[uncertain])
            values["per_unit"] = {unit: diagnostics(a[:, i, :], b[:, i, :]) for i, unit in enumerate(units)}
            comparisons[number][group] = values
        summary = comparisons[number]["2024_private"]
        print(json.dumps({"versus_3": number, "2024_pearson": summary["pearson"], "2024_spearman": summary["spearman"], "2024_mean_abs_diff": summary["mean_absolute_difference"], "2024_threshold_disagreement": summary["threshold_0_5_disagreement_fraction"], "2024_uncertain_pearson": summary["reference_0_1_to_0_9_subset"]["pearson"]}), flush=True)
    report = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "kind": "label-free diagnostics, not performance evaluation",
        "protocol_sha256": sha256(OUT / "protocol.json"),
        "script_sha256": sha256(Path(__file__)),
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "command": "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 .venv-hydraulic/bin/python reports/submission_pair_review_20261108/compare_predictions.py",
        "environment": {"os": platform.platform(), "python": sys.version, "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "machine": platform.machine()},
        "input_hashes": hashes | {"data/raw/prediction_mapping.csv": sha256(ROOT / "data/raw/prediction_mapping.csv"), "data/raw/sample_submission.csv": sha256(ROOT / "data/raw/sample_submission.csv")},
        "n_cases": {name: int(mask.sum()) for name, mask in groups.items()},
        "reference_catalogue_number": "3",
        "comparisons": comparisons,
        "labels_read": False,
        "new_auc_evaluations": 0,
        "new_2022_looks": 0,
        "limitation": "Prediction differences do not establish error complementarity or superiority on private labels. Candidate 4 is absent and has not been replaced by a proxy.",
    }
    result_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
