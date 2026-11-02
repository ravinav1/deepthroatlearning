"""One Lio weekly-hours regressor and one fixed hourly probability correction.

Uses existing exogenous features and saved Round5 ensemble predictions.
Run with .venv/bin/python src/check_round8_weekly_hours.py.
"""

import json
import os
from datetime import datetime, timezone

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score

from features import ROOT, HOURS, load_data
from round5_lightgbm import generator_features
from round6_evaluate import EXPECTED_ROUND5, load_cache
from round5_compare import official_metric
from round3_frozen_evaluation import digest, write_json


CONFIG = ROOT / "configs/round8_weekly_hours.json"
REPORT = ROOT / "reports/round8_weekly_hours_20261017.json"


def weekly_summaries(hourly, starts, training_mask):
    hourly = hourly.drop(columns=["horizon_hour", "hour_of_day", "day_of_week", "month", "generator_id"])
    values = hourly.to_numpy().reshape(len(starts), HOURS, -1)
    minimum, maximum = values.min(axis=1), values.max(axis=1)
    varying = (maximum[training_mask] != minimum[training_mask]).any(axis=0)
    means, stds = values.mean(axis=1), values.std(axis=1)
    columns = {}
    for index, name in enumerate(hourly.columns):
        columns[f"{name}_mean"] = means[:, index]
        if varying[index]:
            columns[f"{name}_min"] = minimum[:, index]
            columns[f"{name}_max"] = maximum[:, index]
            columns[f"{name}_std"] = stds[:, index]
    month = starts.dt.month.to_numpy()
    columns["start_month_sin"] = np.sin(2 * np.pi * month / 12)
    columns["start_month_cos"] = np.cos(2 * np.pi * month / 12)
    result = pd.DataFrame(columns, index=starts.index).astype("float32")
    # Freeze this input-only selection before screening; use the same columns later.
    keep = result.loc[training_mask].nunique() > 1
    return result.loc[:, keep]


def shift_to_fraction(probabilities, fractions, epsilon):
    logits = logit(np.clip(probabilities, epsilon, 1 - epsilon))
    targets = np.clip(fractions, epsilon, 1 - epsilon)
    low, high = np.full(len(targets), -40.0), np.full(len(targets), 40.0)
    for _ in range(50):
        middle = (low + high) / 2
        below = expit(logits + middle[:, None]).mean(axis=1) < targets
        low = np.where(below, middle, low)
        high = np.where(below, high, middle)
    offsets = (low + high) / 2
    corrected = expit(logits + offsets[:, None])
    assert np.max(np.abs(corrected.mean(axis=1) - targets)) < 1e-10
    order = np.argsort(probabilities, axis=1)
    assert (np.diff(np.take_along_axis(corrected, order, axis=1), axis=1) >= 0).all()
    return corrected, offsets, targets


def main():
    assert not REPORT.exists()
    config = json.loads(CONFIG.read_text())
    deadline = datetime.fromisoformat(config["deadline_utc"])
    source_hashes = {}

    def remember(path):
        source_hashes[str(path.relative_to(ROOT))] = digest(path)
        return path

    def check_time(env=None):
        if datetime.now(timezone.utc) >= deadline:
            raise TimeoutError("The 20-minute weekly-hours pilot has ended.")

    check_time.before_iteration = True
    check_time.order = 0
    check_time()
    for path in [CONFIG, ROOT / "src/check_round8_weekly_hours.py", ROOT / config["source_feature_config"],
                 ROOT / "src/features.py", ROOT / "src/tree_preprocessing.py", ROOT / "src/topology_features.py",
                 ROOT / "src/price_features.py", ROOT / "src/round5_lightgbm.py",
                 ROOT / "src/round6_evaluate.py", ROOT / "evaluation/kaggle_metric.py",
                 ROOT / "configs/Tokke_Vinje_topology.yaml"]:
        remember(path)
    for folder in ("data/raw/kernel", "data/raw/extended"):
        for path in sorted((ROOT / folder).glob("*.csv")):
            remember(path)
    remember(ROOT / "data/raw/prediction_mapping.csv")
    remember(ROOT / "data/raw/sample_submission.csv")

    cases, columns, generators, inputs, topology = load_data(use_extended=True)
    test_ids = pd.read_csv(ROOT / "data/raw/prediction_mapping.csv", usecols=["Run No"])["Run No"]
    assert not cases["Run No"].isin(test_ids).any()
    generator = config["generator"]
    generator_id = generators.index(generator)
    lio_columns = [f"result_committed_{generator}_t{hour}" for hour in range(HOURS)]
    output_indices = [columns.index(column) for column in lio_columns]
    target = cases[lio_columns].to_numpy().mean(axis=1)
    pre2021 = (cases.starttime + pd.Timedelta(hours=HOURS) <= pd.Timestamp("2021-01-01", tz="UTC")).to_numpy()
    feature_config = json.loads((ROOT / config["source_feature_config"]).read_text())
    # Only identifiers/timestamps for alignment enter this function, never label columns.
    hourly = generator_features(cases[["Run No", "starttime"]], generator, generator_id, inputs, topology, feature_config)
    features = weekly_summaries(hourly, cases.starttime, pre2021)
    del hourly
    assert np.isfinite(features.to_numpy()).all()
    print(f"Weekly features: {features.shape[1]}; pre-2021 training cases: {int(pre2021.sum())}", flush=True)
    model_reports, evaluations = [], []

    def fit(year):
        check_time()
        boundary = pd.Timestamp(f"{year}-01-01", tz="UTC")
        mask = (cases.starttime + pd.Timedelta(hours=HOURS) <= boundary).to_numpy()
        model_path = ROOT / f"models/round8_lio_weekly_hours_pre{year}.txt"
        assert not model_path.exists()
        model = lgb.LGBMRegressor(**config["model"])
        model.fit(features.loc[mask], target[mask], callbacks=[check_time])
        model.booster_.save_model(str(model_path))
        cohort = cases.loc[mask, ["Run No", "starttime"]]
        model_reports.append({"train_before": year, "cases": int(mask.sum()),
                              "first_case_start": str(cohort.starttime.min()),
                              "last_case_start": str(cohort.starttime.max()),
                              "latest_target_end": str(cohort.starttime.max() + pd.Timedelta(hours=HOURS)),
                              "competition_id_overlap": 0, "trees": model.booster_.num_trees(),
                              "model_path": str(model_path.relative_to(ROOT)), "model_sha256": digest(model_path)})
        return model

    score = official_metric()

    def evaluate(model, year, mode):
        check_time()
        cache_path = ROOT / f"data/processed/round6_{mode}_{year}_baseline.npz"
        arrays, metadata = load_cache(cache_path, year, mode)
        remember(cache_path)
        source_hashes.update(metadata["source_sha256"])
        assert metadata["columns"] == columns
        selected = cases.set_index("Run No").loc[arrays["ids"]]
        indices = pd.Index(cases["Run No"]).get_indexer(arrays["ids"])
        assert (indices >= 0).all()
        np.testing.assert_array_equal(selected[columns].to_numpy(), arrays["labels"])
        baseline = .5 * arrays["round3"] + .25 * arrays["old299"] + .25 * arrays["old673"]
        assert abs(roc_auc_score(arrays["labels"].ravel(), baseline.ravel()) - EXPECTED_ROUND5[mode, year]) < 1e-12
        fractions = model.predict(features.iloc[indices])
        corrected, offsets, clipped = shift_to_fraction(baseline[:, output_indices], fractions, config["fraction_epsilon"])
        candidate = baseline.copy()
        candidate[:, output_indices] = corrected
        other = [i for i in range(len(columns)) if i not in output_indices]
        np.testing.assert_array_equal(candidate[:, other], baseline[:, other])
        starts = pd.DatetimeIndex(pd.to_datetime(arrays["starttime"], utc=True))
        masks = {"all": np.ones(len(starts), dtype=bool)}
        if year == 2022:
            masks["jan7_onward"] = starts >= pd.Timestamp("2022-01-07", tz="UTC")
        rows = []
        for subset, mask in masks.items():
            labels = arrays["labels"][mask]
            true_hours = labels[:, output_indices].sum(axis=1)
            result = {"mode": mode, "year": year, "subset": subset, "cases": int(mask.sum())}
            for name, values in (("baseline", baseline[mask]), ("candidate", candidate[mask])):
                observed = float(roc_auc_score(labels.ravel(), values.ravel()))
                solution = pd.DataFrame(labels, columns=columns)
                submission = pd.DataFrame(values, columns=columns)
                for frame in (solution, submission):
                    frame.insert(0, "Run No", arrays["ids"][mask])
                assert score(solution, submission) == observed
                error = values[:, output_indices].sum(axis=1) - true_hours
                result[name] = {"micro_roc_auc": observed,
                                "lio_auc": float(roc_auc_score(labels[:, output_indices].ravel(), values[:, output_indices].ravel())),
                                "weekly_hours_mae": float(np.abs(error).mean()),
                                "weekly_hours_rmse": float(np.sqrt(np.mean(error ** 2))),
                                "weekly_hours_bias": float(error.mean()),
                                "weeks_error_over_48_hours": int((np.abs(error) > 48).sum())}
            result["micro_auc_gain"] = result["candidate"]["micro_roc_auc"] - result["baseline"]["micro_roc_auc"]
            result["weekly_hours_mae_reduction"] = result["baseline"]["weekly_hours_mae"] - result["candidate"]["weekly_hours_mae"]
            result["passes_gate"] = result["micro_auc_gain"] > 0 and result["weekly_hours_mae_reduction"] > 0
            rows.append(result)
            print(json.dumps(result), flush=True)
        artifact = ROOT / f"data/processed/round8_lio_weekly_hours_{mode}_{year}.npz"
        with artifact.open("xb") as handle:
            np.savez_compressed(handle, ids=arrays["ids"], raw_predicted_fraction=fractions,
                                clipped_predicted_fraction=clipped, logit_offset=offsets,
                                corrected_lio=corrected, true_lio_hours=arrays["labels"][:, output_indices].sum(axis=1))
        return {"mode": mode, "year": year, "results": rows,
                "predictions_path": str(artifact.relative_to(ROOT)), "predictions_sha256": digest(artifact),
                "clipped_cases": int((fractions != clipped).sum()), "other_generators_unchanged": True,
                "within_week_no_rank_inversions": True, "official_metric_verified": True}

    model2021 = fit(2021)
    screen = evaluate(model2021, 2021, "native")
    evaluations.append(screen)
    screen_passed = all(row["passes_gate"] for row in screen["results"])
    if screen_passed:
        evaluations.append(evaluate(model2021, 2022, "frozen"))
        # A failed frozen check ends the pilot before another fit.
        if all(row["passes_gate"] for row in evaluations[-1]["results"]):
            evaluations.append(evaluate(fit(2022), 2022, "native"))
    passed = len(evaluations) == 3 and all(row["passes_gate"] for evaluation in evaluations for row in evaluation["results"])
    check_time()
    assert all(digest(ROOT / path) == expected for path, expected in source_hashes.items())
    finished = datetime.now(timezone.utc)
    report = {"config": config, "finished_utc": finished.isoformat(),
              "elapsed_minutes": (finished - datetime.fromisoformat(config["started_utc"])).total_seconds() / 60,
              "feature_names": features.columns.tolist(), "weekly_feature_count": len(features.columns),
              "feature_selection_training_cases": int(pre2021.sum()), "models": model_reports,
              "evaluations": evaluations, "passes_all_gates": passed,
              "decision": "Eligible for further confirmation/export" if passed else "Stop; keep Round5. This fixed weekly-total correction failed its gate.",
              "submissions_created": 0, "competition_id_overlap": 0,
              "source_sha256": source_hashes, "source_hashes_unchanged": True,
              "limitations": "One fixed model and full-strength correction only. Reused historical development years. "
                             "Weekly-total accuracy does not imply micro-AUC gains. A constant case logit offset preserves within-week order "
                             "apart from numerical ties, but can change ranking across cases and generators. "
                             "No base-model predictions or labels were used as input features; therefore no in-sample stacking inputs are present."}
    write_json(REPORT, report)
    print(json.dumps({"decision": report["decision"], "elapsed_minutes": report["elapsed_minutes"]}), flush=True)


if __name__ == "__main__":
    main()
