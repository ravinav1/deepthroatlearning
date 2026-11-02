"""Screen two fixed Vinje generator-order corrections; no fitting or CSV export."""

import json
import os
from datetime import datetime, timezone

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import pandas as pd
from scipy.optimize import isotonic_regression
from sklearn.metrics import roc_auc_score

from features import ROOT
from round3_frozen_evaluation import digest, write_json
from round5_compare import official_metric
from round6_evaluate import EXPECTED_ROUND5, load_cache


PREFIX = "round9_unit_order"
GENERATORS = ["Vinje_G1", "Vinje_G2", "Vinje_G3"]


def project_three(values):
    """Equal-weight Euclidean projection onto x1 >= x2 >= x3."""
    x = np.asarray(values, dtype=float)
    flat = x.reshape(-1, 3)
    out = flat.copy()
    a, b, c = flat.T
    ab = a < b
    ab_mean = (a + b) / 2
    out[ab, :2] = ab_mean[ab, None]
    all_from_ab = ab & (ab_mean < c)
    bc = ~ab & (b < c)
    bc_mean = (b + c) / 2
    out[bc, 1:] = bc_mean[bc, None]
    all_from_bc = bc & (a < bc_mean)
    all_three = all_from_ab | all_from_bc
    out[all_three] = flat[all_three].mean(axis=1, keepdims=True)
    return out.reshape(x.shape)


def transform(values, indices, method):
    candidate = values.copy()
    vinje = np.stack([values[:, index] for index in indices], axis=-1)
    corrected = (np.sort(vinje, axis=-1)[..., ::-1] if method == "sort"
                 else project_three(vinje))
    for unit, index in enumerate(indices):
        candidate[:, index] = corrected[..., unit]
    other = np.ones(values.shape[1], dtype=bool)
    other[np.concatenate(indices)] = False
    np.testing.assert_array_equal(candidate[:, other], values[:, other])
    np.testing.assert_allclose(corrected.sum(axis=-1), vinje.sum(axis=-1), rtol=0, atol=5e-16)
    assert (np.diff(corrected, axis=-1) <= 0).all()
    assert np.isfinite(candidate).all() and ((candidate >= 0) & (candidate <= 1)).all()
    return candidate


def ordering(values):
    # values: case, hour, generator. Count adjacent pair and case-hour violations.
    bad = values[..., :-1] < values[..., 1:]
    return {"case_hours": int(bad.shape[0] * bad.shape[1]),
            "adjacent_pairs": int(bad.size),
            "violating_adjacent_pairs": int(bad.sum()),
            "violating_case_hours": int(bad.any(axis=-1).sum()),
            "case_hour_violation_fraction": float(bad.any(axis=-1).mean())}


def main():
    started = datetime.now(timezone.utc)
    protocol_path = ROOT / f"reports/{PREFIX}_protocol.json"
    report_path = ROOT / f"reports/{PREFIX}_results.json"
    lock_path = ROOT / f"reports/{PREFIX}_selection_2021.json"
    for path in (protocol_path, report_path, lock_path):
        assert not path.exists(), path
    protocol = {
        "created_utc": started.isoformat(),
        "baseline": "Original Round5: .5 Round3 + .25 relative LightGBM seed299 + .25 seed673",
        "generators": GENERATORS,
        "hypothesis": "Vinje labels nearly respect G1 >= G2 >= G3; remove inconsistent probabilities.",
        "variants": ["Sort the three probabilities descending at each case/hour.",
                     "Equal-weight Euclidean nonincreasing isotonic projection at each case/hour."],
        "selection": "Best pooled micro-AUC among baseline, sort and isotonic on native 2021; baseline wins ties.",
        "screen_gate": "At least +0.00005 pooled micro-AUC vs original Round5 on 2021.",
        "confirmation_gate": "Native 2022 all and Jan7+ gain >=0.00005; frozen 2022 all and Jan7+ gain >=0.",
        "no_adaptation": "Only selected correction evaluated on 2022; no retries, strengths or alternate selections.",
        "training": "No models fitted. Training-label diagnostic excludes weeks crossing 2021 boundary.",
        "scope": "Vinje only. Other generator probabilities stay bit-identical. No Tokke correction.",
    }
    write_json(protocol_path, protocol)
    source_hashes = {}

    def remember(path):
        source_hashes[str(path.relative_to(ROOT))] = digest(path)

    for path in (protocol_path, ROOT / "src/round9_unit_order.py", ROOT / "src/round6_evaluate.py",
                 ROOT / "src/round5_compare.py", ROOT / "src/round3_frozen_evaluation.py",
                 ROOT / "evaluation/kaggle_metric.py", ROOT / "data/raw/kernel/Unit_commitment_decisions.csv",
                 ROOT / "data/raw/prediction_mapping.csv"):
        remember(path)

    # Check the vectorized projection against an independent PAVA implementation.
    probes = np.random.default_rng(4173).random((300, 3))
    probes = np.vstack([probes, [[0, 0, 0], [1, 1, 1], [0, 1, 0], [1, 0, 1], [0, 0, 1]]])
    reference = np.stack([isotonic_regression(row, increasing=False).x for row in probes])
    np.testing.assert_allclose(project_three(probes), reference, rtol=0, atol=3e-16)

    columns = [f"result_committed_{generator}_t{hour}" for generator in GENERATORS for hour in range(168)]
    cases = pd.read_csv(ROOT / "data/raw/kernel/Unit_commitment_decisions.csv",
                        usecols=["Run No", "starttime", *columns])
    starts = pd.to_datetime(cases.starttime, utc=True)
    training = starts + pd.Timedelta(hours=168) <= pd.Timestamp("2021-01-01", tz="UTC")
    test_ids = pd.read_csv(ROOT / "data/raw/prediction_mapping.csv", usecols=["Run No"])["Run No"]
    assert not cases["Run No"].isin(test_ids).any()
    training_labels = cases.loc[training, columns].to_numpy().reshape(-1, 3, 168).transpose(0, 2, 1)
    training_diagnostic = {"cases": int(training.sum()), "latest_start": str(starts[training].max()),
                           "latest_target_end": str(starts[training].max() + pd.Timedelta(hours=168)),
                           **ordering(training_labels)}

    metric = official_metric()
    evaluations = []
    artifacts = {}

    def evaluate(year, mode, methods):
        cache_path = ROOT / f"data/processed/round6_{mode}_{year}_baseline.npz"
        arrays, metadata = load_cache(cache_path, year, mode)
        remember(cache_path)
        source_hashes.update(metadata["source_sha256"])
        output_columns = metadata["columns"]
        indices = [np.array([output_columns.index(f"result_committed_{generator}_t{hour}")
                             for hour in range(168)]) for generator in GENERATORS]
        labels = arrays["labels"]
        baseline = .5 * arrays["round3"] + .25 * arrays["old299"] + .25 * arrays["old673"]
        base_auc = float(roc_auc_score(labels.ravel(), baseline.ravel()))
        assert abs(base_auc - EXPECTED_ROUND5[mode, year]) < 1e-12
        variants = {"baseline": baseline}
        variants.update({name: transform(baseline, indices, name) for name in methods})
        starts_ = pd.DatetimeIndex(pd.to_datetime(arrays["starttime"], utc=True))
        subsets = {"all": np.ones(len(labels), dtype=bool)}
        if year == 2022:
            subsets["jan7_onward"] = starts_ >= pd.Timestamp("2022-01-07", tz="UTC")
        rows = []
        for subset, mask in subsets.items():
            baseline_score = float(roc_auc_score(labels[mask].ravel(), baseline[mask].ravel()))
            for name, prediction in variants.items():
                observed = float(roc_auc_score(labels[mask].ravel(), prediction[mask].ravel()))
                truth = pd.DataFrame(labels[mask], columns=output_columns)
                submission = pd.DataFrame(prediction[mask], columns=output_columns)
                for frame in (truth, submission):
                    frame.insert(0, "Run No", arrays["ids"][mask])
                assert metric(truth, submission) == observed
                rows.append({"year": year, "mode": mode, "subset": subset, "method": name,
                             "cases": int(mask.sum()), "micro_roc_auc": observed,
                             "delta_vs_baseline": observed - baseline_score,
                             "vinje_generator_auc": {generator: float(roc_auc_score(labels[mask][:, index].ravel(),
                                                                                   prediction[mask][:, index].ravel()))
                                                      for generator, index in zip(GENERATORS, indices)}})
        artifact = ROOT / f"data/processed/{PREFIX}_{mode}_{year}.npz"
        with artifact.open("xb") as handle:
            np.savez_compressed(handle, ids=arrays["ids"], columns=np.asarray(output_columns),
                                **{name: values for name, values in variants.items() if name != "baseline"})
        artifacts[str(artifact.relative_to(ROOT))] = digest(artifact)
        result = {"year": year, "mode": mode, "results": rows,
                  "label_ordering": ordering(np.stack([labels[:, i] for i in indices], axis=-1)),
                  "baseline_prediction_ordering": ordering(np.stack([baseline[:, i] for i in indices], axis=-1))}
        evaluations.append(result)
        print(json.dumps(result), flush=True)
        return result

    screening = evaluate(2021, "native", ("sort", "isotonic"))
    best = max(screening["results"], key=lambda row: row["micro_roc_auc"])
    selected = best["method"]
    screen_passed = selected != "baseline" and best["delta_vs_baseline"] >= 0.00005
    selection = {"created_utc": datetime.now(timezone.utc).isoformat(), "selected": selected,
                 "screen_gate_passed": screen_passed, "results": screening["results"],
                 "protocol_sha256": digest(protocol_path), "source_sha256": source_hashes,
                 "prediction_sha256": artifacts.copy()}
    write_json(lock_path, selection)
    remember(lock_path)
    confirmation_passed = False
    if screen_passed:
        native = evaluate(2022, "native", (selected,))
        frozen = evaluate(2022, "frozen", (selected,))
        confirmation_passed = all(row["delta_vs_baseline"] >= (0.00005 if row["mode"] == "native" else 0)
                                  for evaluation in (native, frozen) for row in evaluation["results"]
                                  if row["method"] == selected)
    assert all(digest(ROOT / path) == expected for path, expected in source_hashes.items())
    result = {"started_utc": started.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat(),
              "elapsed_seconds": (datetime.now(timezone.utc) - started).total_seconds(),
              "training_label_diagnostic": training_diagnostic, "selected": selected,
              "screen_gate_passed": screen_passed, "confirmation_gate_passed": confirmation_passed,
              "evaluations": evaluations, "source_sha256": source_hashes, "source_hashes_unchanged": True,
              "prediction_sha256": artifacts, "official_metric_verified": True,
              "non_vinje_predictions_identical": True, "probabilities_valid": True,
              "vinje_hourly_probability_sum_preserved": True, "isotonic_scipy_checks": len(probes),
              "competition_labels_used": False, "models_fitted": 0, "submissions_created": 0,
              "decision": "Eligible for export review" if confirmation_passed else "Reject; keep original Round5.",
              "limitations": "Historical years are reused development data. Ordering is an empirical pattern, not proven universal. "
                             "The two 2022 modes share labels. No parameter/strength selected after 2022."}
    write_json(report_path, result)
    print(json.dumps({"selected": selected, "passed": confirmation_passed, "report": str(report_path)}), flush=True)


if __name__ == "__main__":
    main()
