"""Fit one shrinkage calibrator on early2021 and freeze before later evaluation."""

import os

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

from datetime import datetime, timezone
import json
import warnings

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from features import ROOT
from plant_blend import metrics
from round3_frozen_evaluation import digest, write_json
from round5_compare import official_metric
from round6_evaluate import load_cache


PREFIX = "round9_group_calibration"
CLIP = 1e-6


def now():
    return datetime.now(timezone.utc).isoformat()


def original(arrays):
    return .5 * arrays["round3"] + .25 * arrays["old299"] + .25 * arrays["old673"]


def features(predictions, generator_indices, count):
    values = np.empty((predictions.size, count + 1), dtype=np.float64)
    values[:, 0] = logit(np.clip(predictions.ravel(), CLIP, 1-CLIP))
    values[:, 1:] = np.tile(np.eye(count)[generator_indices], (len(predictions), 1))
    return values


def apply(predictions, slope, offsets, generator_indices):
    return expit(slope * logit(np.clip(predictions, CLIP, 1-CLIP)) + offsets[generator_indices][None, :])


def evaluate(arrays, metadata, slope, offsets, generator_indices, generators, subset_masks=None):
    baseline = original(arrays)
    candidate = apply(baseline, slope, offsets, generator_indices)
    labels, columns = arrays["labels"], metadata["columns"]
    starts = pd.DatetimeIndex(pd.to_datetime(arrays["starttime"], utc=True))
    masks = subset_masks or {"all": np.ones(len(labels), dtype=bool)}
    if subset_masks is None and metadata["year"] == 2022:
        masks["jan7_onward"] = starts >= pd.Timestamp("2022-01-07", tz="UTC")
    rows = []
    score = official_metric()
    for subset, mask in masks.items():
        before = metrics(labels[mask], baseline[mask], generators)
        after = metrics(labels[mask], candidate[mask], generators)
        truth = pd.DataFrame(labels[mask], columns=columns)
        pred = pd.DataFrame(candidate[mask], columns=columns)
        truth.insert(0, "Run No", arrays["ids"][mask])
        pred.insert(0, "Run No", arrays["ids"][mask])
        official = score(truth, pred)
        assert official == after["micro_roc_auc"]
        # A positive common slope with a generator-specific offset preserves
        # ordering within each generator, including between its different weeks.
        if slope > 0:
            assert all(abs(after["generator_auc"][g]-before["generator_auc"][g]) < 1e-9 for g in generators)
        rows.append({"year": metadata["year"], "mode": metadata["mode"], "subset": subset,
                     "cases": int(mask.sum()), "first_case_start": str(starts[mask].min()),
                     "last_case_start": str(starts[mask].max()),
                     "baseline": before, "candidate": after,
                     "delta": after["micro_roc_auc"]-before["micro_roc_auc"],
                     "official_micro_roc_auc": official})
    return rows


def main():
    protocol_path = ROOT / f"reports/{PREFIX}_protocol.json"
    fit_path = ROOT / f"reports/{PREFIX}_fit.json"
    selection_path = ROOT / f"reports/{PREFIX}_selection.json"
    report_path = ROOT / f"reports/{PREFIX}_results.json"
    assert not any(path.exists() for path in (protocol_path, fit_path, selection_path, report_path))
    protocol = {"created_utc": now(), "baseline": ".5 Round3 + .25 original Round5 seed299 + .25 original Round5 seed673",
                "method": "Binary ridge logistic regression: one common logit slope +14generator offsets",
                "C": .001, "fit_intercept": False, "max_iter": 300, "solver": "lbfgs", "threads": 1,
                "probability_clip": CLIP, "require_positive_slope": True,
                "fit_cases": "2021 starts whose start+7days <=2021-07-01UTC (Jan1-Jun24)",
                "holdout_cases": "2021-07-01UTC through2021-12-31UTC",
                "minimum_holdout_micro_auc_gain": .00005,
                "minimum_native_2022_gain_all_and_jan7": .00005,
                "minimum_frozen_2022_gain_all_and_jan7": 0,
                "refit_after_2021_holdout": False, "fit_2022_labels": False, "retries": False,
                "source_code_sha256": digest(ROOT / "src/round9_group_calibration.py")}
    write_json(protocol_path, protocol)
    sources = {str(protocol_path.relative_to(ROOT)): digest(protocol_path)}

    def read(year, mode):
        path = ROOT / f"data/processed/round6_{mode}_{year}_baseline.npz"
        arrays, metadata = load_cache(path, year, mode)
        sources[str(path.relative_to(ROOT))] = digest(path)
        sources.update(metadata["source_sha256"])
        return arrays, metadata

    arrays, metadata = read(2021, "native")
    columns = metadata["columns"]
    generator_names = [c.removeprefix("result_committed_").rsplit("_t", 1)[0] for c in columns]
    generators = list(dict.fromkeys(generator_names))
    assert len(generators) == 14
    generator_indices = np.array([generators.index(g) for g in generator_names])
    starts = pd.DatetimeIndex(pd.to_datetime(arrays["starttime"], utc=True))
    boundary = pd.Timestamp("2021-07-01", tz="UTC")
    fit_mask = starts + pd.Timedelta(days=7) <= boundary
    holdout = starts >= boundary
    assert fit_mask.sum() == 175 and holdout.sum() == 184 and (~(fit_mask|holdout)).sum() == 6
    x = features(original(arrays)[fit_mask], generator_indices, len(generators))
    y = arrays["labels"][fit_mask].ravel()
    model = LogisticRegression(C=.001, fit_intercept=False, max_iter=300, solver="lbfgs")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x, y)
    slope = float(model.coef_[0, 0])
    offsets = model.coef_[0, 1:]
    fit = {"frozen_utc": now(), "protocol_sha256": digest(protocol_path),
           "slope": slope, "offsets": dict(zip(generators, offsets.tolist())),
           "fit_cases": int(fit_mask.sum()), "fit_observations": int(y.size),
           "first_fit_start": str(starts[fit_mask].min()), "last_fit_start": str(starts[fit_mask].max()),
           "fit_target_end_boundary": str((starts[fit_mask]+pd.Timedelta(days=7)).max()),
           "held_out_cases": int(holdout.sum()), "excluded_overlap_case_starts": int((~(fit_mask|holdout)).sum()),
           "iterations": model.n_iter_.tolist(), "convergence_warnings": [str(w.message) for w in caught],
           "source_sha256": sources.copy()}
    write_json(fit_path, fit)
    sources[str(fit_path.relative_to(ROOT))] = digest(fit_path)
    del x, y, model
    screen = evaluate(arrays, metadata, slope, offsets, generator_indices, generators,
                      {"h2_holdout": holdout})
    screen_passed = slope > 0 and not caught and screen[0]["delta"] >= protocol["minimum_holdout_micro_auc_gain"]
    write_json(selection_path, {"locked_utc": now(), "fit_sha256": digest(fit_path),
                               "screen": screen, "screen_passed": screen_passed})
    sources[str(selection_path.relative_to(ROOT))] = digest(selection_path)
    rows = screen.copy()
    passed = screen_passed
    if screen_passed:
        for mode in ("native", "frozen"):
            arrays, metadata = read(2022, mode)
            assert metadata["columns"] == columns
            result = evaluate(arrays, metadata, slope, offsets, generator_indices, generators)
            rows.extend(result)
            passed = passed and all(row["delta"] >= protocol[f"minimum_{mode}_2022_gain_all_and_jan7"] for row in result)
    assert all(digest(ROOT / path) == value for path, value in sources.items())
    write_json(report_path, {"finished_utc": now(), "protocol": protocol, "fit": fit,
                            "screen_passed": screen_passed, "all_gates_passed": passed,
                            "confirmation_evaluated": screen_passed, "results": rows,
                            "source_sha256": sources, "source_hashes_unchanged": True,
                            "official_metric_verified": True, "competition_label_cases": 0,
                            "limitations": "Years are reused development data. Calibration only changes cross-generator ranking "
                                           "when slope is positive; within-generator order is preserved except possible clipped ties. "
                                           "Native/frozen2022 share labels. Coefficients are never refitted after H1training."})
    print(json.dumps({"slope": slope, "screen_passed": screen_passed, "all_gates_passed": passed,
                      "results": [{k: row[k] for k in ("year", "mode", "subset", "delta")} for row in rows]}, indent=2))


if __name__ == "__main__":
    main()
