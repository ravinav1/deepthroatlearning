"""Cached prediction diagnosis and paired block bootstrap of pooled micro AUC.

The bootstrap sorts each model's scores once. Each draw gives every output of a
case the same repetition count, so dependent hours/generators stay together.
No model loading or fitting occurs here.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from features import feeding_reservoirs
from plant_blend import column_groups, read_predictions


ROOT = Path(__file__).resolve().parents[1]
HOURS = 168


def auc(labels, probabilities):
    y = np.asarray(labels).ravel()
    if not y.size or y.min() == y.max():
        return None
    return float(roc_auc_score(y, np.asarray(probabilities).ravel()))


def wrong_pair_count(labels, probabilities):
    """Number of misordered positive/negative pairs, counting ties as half."""
    y, p = np.asarray(labels).ravel(), np.asarray(probabilities).ravel()
    negatives = np.sort(p[y == 0])
    positives = p[y == 1]
    lower = np.searchsorted(negatives, positives, side="left")
    upper = np.searchsorted(negatives, positives, side="right")
    return float(np.sum(len(negatives) - .5 * (lower + upper)))


class WeightedAUC:
    """Exact pooled AUC for arbitrary nonnegative counts of whole cases."""

    def __init__(self, labels, probabilities):
        y, p = np.asarray(labels), np.asarray(probabilities)
        assert y.shape == p.shape and y.ndim >= 2
        self.cases = len(y)
        flat_p = p.ravel()
        order = np.argsort(flat_p, kind="stable")
        scores = flat_p[order]
        self.group_starts = np.r_[0, np.flatnonzero(np.diff(scores)) + 1]
        self.labels = y.ravel()[order].astype(bool)
        width = y.size // len(y)
        self.case_indices = (order // width).astype(np.int32)

    def __call__(self, counts):
        counts = np.asarray(counts, dtype=np.float64)
        assert counts.shape == (self.cases,) and np.all(counts >= 0)
        weights = counts[self.case_indices]
        pos = np.add.reduceat(weights * self.labels, self.group_starts)
        neg = np.add.reduceat(weights * ~self.labels, self.group_starts)
        denominator = pos.sum() * neg.sum()
        if not denominator:
            return float("nan")
        neg_below = np.cumsum(neg) - neg
        return float(np.sum(pos * (neg_below + .5 * neg)) / denominator)


def paired_block_bootstrap(labels, reference, candidate, block_lengths=(7, 14, 28),
                           repetitions=120, seed=4173, starts=None):
    """Paired circular moving-block bootstrap, one row per daily-start case.

    Each block contains adjacent case-start dates, including every output from
    every sampled case. Circular wrapping gives all days equal inclusion chance.
    The final sampled block is truncated to keep the original case count.
    Intervals describe uncertainty conditional on these reused predictions;
    they do not remove validation-selection bias or certify future performance.
    """
    labels, reference, candidate = map(np.asarray, (labels, reference, candidate))
    assert labels.shape == reference.shape == candidate.shape
    if starts is not None:
        day = pd.DatetimeIndex(pd.to_datetime(starts, utc=True)).as_unit("ns").asi8
        assert np.all(np.diff(day) == pd.Timedelta(days=1).value), "Cases must be consecutive daily starts."
    functions = WeightedAUC(labels, reference), WeightedAUC(labels, candidate)
    count = len(labels)
    exact = [fn(np.ones(count)) for fn in functions]
    for value, probabilities in zip(exact, (reference, candidate)):
        assert abs(value - auc(labels, probabilities)) < 1e-12
    results = []
    for length in block_lengths:
        assert 1 <= length <= count
        rng = np.random.default_rng(seed + length)
        deltas = []
        for _ in range(repetitions):
            beginnings = rng.integers(0, count, size=(count + length - 1) // length)
            sampled = ((beginnings[:, None] + np.arange(length)) % count).ravel()[:count]
            counts = np.bincount(sampled, minlength=count)
            values = [fn(counts) for fn in functions]
            deltas.append(values[1] - values[0])
        deltas = np.asarray(deltas)
        assert np.isfinite(deltas).all()
        results.append({
            "block_days": length, "repetitions": repetitions,
            "reference_auc": exact[0], "candidate_auc": exact[1],
            "observed_delta": exact[1] - exact[0],
            "bootstrap_delta_mean": float(deltas.mean()),
            "bootstrap_delta_std": float(deltas.std(ddof=1)),
            "percentile_interval_95": np.quantile(deltas, [.025, .975]).tolist(),
            "fraction_positive_draws": float(np.mean(deltas > 0)),
        })
    return results


class PairDiagnostics:
    def __init__(self, labels, probabilities):
        self.y, self.p = np.asarray(labels).ravel(), np.asarray(probabilities).ravel()
        self.positive_count = int(self.y.sum())
        self.negative_count = len(self.y) - self.positive_count
        self.total_pairs = self.positive_count * self.negative_count
        self.wrong_pairs = wrong_pair_count(self.y, self.p)
        positive = self.y == 1
        sorted_neg = np.sort(self.p[~positive])
        sorted_pos = np.sort(self.p[positive])
        self.incident = np.empty(len(self.y), dtype=np.float64)
        self.incident[positive] = len(sorted_neg) - .5 * (
            np.searchsorted(sorted_neg, self.p[positive], side="left") +
            np.searchsorted(sorted_neg, self.p[positive], side="right"))
        self.incident[~positive] = .5 * (
            np.searchsorted(sorted_pos, self.p[~positive], side="left") +
            np.searchsorted(sorted_pos, self.p[~positive], side="right"))
        assert abs(self.incident.sum() / 2 - self.wrong_pairs) < 1e-5
        assert abs(1 - self.wrong_pairs / self.total_pairs - auc(self.y, self.p)) < 1e-12

    def subset(self, mask):
        mask = np.asarray(mask).ravel()
        assert mask.dtype == bool and mask.shape == self.y.shape
        y, p = self.y[mask], self.p[mask]
        positive, negative = int(y.sum()), len(y) - int(y.sum())
        # Summing incident counts double-counts pairs entirely within the group.
        wrong = float(self.incident[mask].sum() - wrong_pair_count(y, p))
        possible = positive * self.negative_count + negative * self.positive_count - positive * negative
        wrong_share = wrong / self.wrong_pairs if self.wrong_pairs else 0.
        pair_share = possible / self.total_pairs
        return {
            "outputs": len(y), "output_fraction": len(y) / len(self.y),
            "on_fraction": float(y.mean()) if len(y) else None,
            "mean_probability": float(p.mean()) if len(p) else None,
            "micro_roc_auc": auc(y, p),
            "incident_wrong_pair_share": wrong_share,
            "incident_available_pair_share": pair_share,
            "error_enrichment_vs_pair_availability": wrong_share / pair_share if pair_share else None,
            "incident_pair_error_rate": wrong / possible if possible else None,
        }


def transition_mask(labels, radius=2):
    transitions = np.zeros_like(labels, dtype=bool)
    transitions[:, :, 1:] = labels[:, :, 1:] != labels[:, :, :-1]
    near = transitions.copy()
    for lag in range(1, radius + 1):
        near[:, :, lag:] |= transitions[:, :, :-lag]
        near[:, :, :-lag] |= transitions[:, :, lag:]
    return near


def generator_pairs(labels, probabilities, generators):
    total_p = int(labels.sum())
    total_pairs = total_p * (labels.size - total_p)
    total_wrong = wrong_pair_count(labels, probabilities)
    rows = []
    for neg_index, neg_name in enumerate(generators):
        neg = np.sort(probabilities[:, neg_index][labels[:, neg_index] == 0])
        for pos_index, pos_name in enumerate(generators):
            pos = probabilities[:, pos_index][labels[:, pos_index] == 1]
            lower = np.searchsorted(neg, pos, side="left")
            upper = np.searchsorted(neg, pos, side="right")
            wrong = float(np.sum(len(neg) - .5 * (lower + upper)))
            possible = len(pos) * len(neg)
            rows.append({
                "positive_generator": pos_name, "negative_generator": neg_name,
                "same_generator": pos_name == neg_name,
                "wrong_pairs": wrong, "available_pairs": possible,
                "wrong_pair_share": wrong / total_wrong,
                "available_pair_share": possible / total_pairs,
                "pair_error_rate": wrong / possible if possible else None,
                "error_enrichment": (wrong / total_wrong) / (possible / total_pairs) if possible else None,
            })
    assert abs(sum(row["wrong_pairs"] for row in rows) - total_wrong) < 1e-5
    assert sum(row["available_pairs"] for row in rows) == total_pairs
    summary = {}
    for name, same in (("within_generator", True), ("cross_generator", False)):
        wrong = sum(row["wrong_pairs"] for row in rows if row["same_generator"] == same)
        possible = sum(row["available_pairs"] for row in rows if row["same_generator"] == same)
        summary[name] = {"wrong_pair_share": wrong / total_wrong,
                         "available_pair_share": possible / total_pairs,
                         "pair_error_rate": wrong / possible,
                         "error_enrichment": (wrong / total_wrong) / (possible / total_pairs)}
    return rows, summary


def load_year(config, year):
    columns = pd.read_csv(ROOT / "data/raw/sample_submission.csv", nrows=0).columns[1:].tolist()
    generators = list(dict.fromkeys(c.removeprefix("result_committed_").rsplit("_t", 1)[0] for c in columns))
    assert columns == [f"result_committed_{g}_t{h}" for g in generators for h in range(HOURS)]
    truth = pd.read_csv(ROOT / "data/raw/kernel/Unit_commitment_decisions.csv",
                        usecols=["Run No", "starttime", *columns], dtype={c: "uint8" for c in columns})
    truth["starttime"] = pd.to_datetime(truth.starttime, utc=True)
    test_ids = pd.read_csv(ROOT / "data/raw/prediction_mapping.csv")["Run No"]
    assert not truth["Run No"].isin(test_ids).any()
    cases = truth.loc[truth.starttime.dt.year == year].sort_values("starttime").reset_index(drop=True)
    sources, probabilities = {}, {}
    for name, prefix in config["prefixes"].items():
        path = ROOT / f"data/processed/{prefix}_{year}_validation_predictions.csv"
        probabilities[name] = read_predictions(path, columns, cases["Run No"])
        sources[name] = {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    recipe = json.loads((ROOT / config["recipe"]).read_text())
    groups = column_groups(columns)
    winner = np.empty_like(probabilities["xgb_rescaled"])
    predecessor = np.empty_like(winner)
    for plant, indices in groups.items():
        weights = recipe["plant_weights"][plant]
        winner[:, indices] = sum(weight * probabilities[name][:, indices] for name, weight in weights.items())
        neural = weights.get("transformer", 0.)
        predecessor[:, indices] = ((1 - neural) * probabilities["xgb_rescaled"][:, indices] +
                                   neural * probabilities["transformer"][:, indices])
    shape = (len(cases), len(generators), HOURS)
    labels = cases[columns].to_numpy().reshape(shape)
    expected = json.loads((ROOT / "reports/goal_lightgbm_comparison.json").read_text())
    expected_auc = next(row["micro_roc_auc"] for row in expected["results"]
                        if row["year"] == year and row["variant"] == "candidate_equal_tree_mean")
    assert abs(auc(labels, winner) - expected_auc) < 1e-12
    boundary = pd.Timestamp(f"{year}-01-01", tz="UTC")
    ends = truth.starttime + pd.Timedelta(hours=HOURS)
    train = truth.loc[ends <= boundary]
    split = {"training_cases": len(train), "validation_cases": len(cases),
             "purged_cases": int(((truth.starttime < boundary) & (ends > boundary)).sum()),
             "last_training_start": str(train.starttime.max()),
             "last_training_end_exclusive": str((train.starttime + pd.Timedelta(hours=HOURS)).max()),
             "first_validation_start": str(cases.starttime.min()),
             "validation_status": "Reused historical development year; not independent holdout."}
    for prefix in config["prefixes"].values():
        report = json.loads((ROOT / f"reports/{prefix}_{year}_metrics.json").read_text())
        for key in ("training_cases", "validation_cases", "purged_cases", "last_training_start", "first_validation_start"):
            assert report[key] == split[key], (prefix, key, report[key], split[key])
    return cases, generators, labels, winner.reshape(shape), predecessor.reshape(shape), sources, split


def regimes(cases, generators):
    starts = cases.starttime
    def read(name):
        frame = pd.read_csv(ROOT / f"data/raw/kernel/{name}.csv")
        frame["date"] = pd.to_datetime(frame.date, utc=True)
        return frame.set_index("date")
    prices = read("Historical_day_ahead_price_2015_2025")
    volumes = read("Historical_volume_2015_2024")
    values = read("Synthetic_water_value_2015_2024")
    topology = yaml.safe_load((ROOT / "configs/Tokke_Vinje_topology.yaml").read_text())
    timestamps = pd.DatetimeIndex(np.repeat(starts.array, HOURS)) + pd.to_timedelta(np.tile(np.arange(HOURS), len(cases)), unit="h")
    price = prices.loc[timestamps].iloc[:, 0].to_numpy().reshape(-1, HOURS)
    scale = np.maximum(np.mean(np.abs(price), axis=1), 10.)
    fill, margin = [], []
    for name in generators:
        reservoirs = feeding_reservoirs(topology, name.rsplit("_G", 1)[0])
        capacity = sum(topology["model"]["reservoir"][r]["max_vol"] for r in reservoirs)
        volume = volumes.loc[starts, reservoirs].sum(axis=1, skipna=False).to_numpy()
        water_value = values.loc[starts + pd.Timedelta(hours=HOURS), reservoirs].mean(axis=1, skipna=False).to_numpy()
        fill.append(np.repeat((volume / capacity)[:, None], HOURS, axis=1))
        margin.append((price - water_value[:, None]) / scale[:, None])
    fill, margin = np.stack(fill, axis=1), np.stack(margin, axis=1)
    return {"initial_local_fill": (fill, [-np.inf, .25, .75, np.inf], ["below_25pct", "25_to_75pct", "above_75pct"]),
            "normalized_price_minus_local_terminal_water_value":
                (margin, [-np.inf, -.1, .1, np.inf], ["below_minus_0.1", "within_plusminus_0.1", "above_0.1"])}


def diagnose(cases, generators, labels, winner, predecessor, radius):
    pairs = PairDiagnostics(labels, winner)
    shape = labels.shape
    rows, pair_summary = generator_pairs(labels, winner, generators)
    result = {"micro_roc_auc": auc(labels, winner), "predecessor_auc": auc(labels, predecessor),
              "generator_pair_composition": pair_summary, "by_generator": [], "by_start_month": [],
              "by_horizon_day": [], "regime_slices": {}}
    for index, name in enumerate(generators):
        mask = np.broadcast_to(np.arange(len(generators))[None, :, None] == index, shape)
        result["by_generator"].append({"generator": name, **pairs.subset(mask),
                                       "predecessor_auc": auc(labels[:, index], predecessor[:, index])})
    special = np.array([name in ("Lio_G1", "Haukeli_G1") for name in generators])[None, :, None]
    result["lio_haukeli_union"] = pairs.subset(np.broadcast_to(special, shape))
    near = transition_mask(labels, radius)
    result["transition_regions"] = {
        "definition": f"Within +/-{radius} hours of true-label state change; centered at the first new-state hour, no inferred preceding label at t0. Diagnostic only.",
        "near": pairs.subset(near), "away": pairs.subset(~near)}
    for month in range(1, 13):
        mask = np.broadcast_to((cases.starttime.dt.month.to_numpy() == month)[:, None, None], shape)
        result["by_start_month"].append({"month": month, **pairs.subset(mask), "predecessor_auc": auc(labels[mask], predecessor[mask])})
    for day in range(7):
        mask = np.broadcast_to((np.arange(HOURS) // 24 == day)[None, None, :], shape)
        result["by_horizon_day"].append({"day": day + 1, **pairs.subset(mask), "predecessor_auc": auc(labels[mask], predecessor[mask])})
    for name, (values, boundaries, names) in regimes(cases, generators).items():
        result["regime_slices"][name] = []
        for low, high, label in zip(boundaries[:-1], boundaries[1:], names):
            mask = (values >= low) & (values < high)
            result["regime_slices"][name].append({"regime": label, **pairs.subset(mask),
                                                 "predecessor_auc": auc(labels[mask], predecessor[mask])})
    return result, rows


def self_check():
    rng = np.random.default_rng(17)
    labels = rng.integers(0, 2, size=(12, 3, 4), dtype=np.uint8)
    probabilities = np.round(rng.random(labels.shape), 1)
    weighted = WeightedAUC(labels, probabilities)
    for counts in (np.ones(12), rng.integers(0, 4, size=12)):
        repeated = np.repeat(np.arange(12), counts.astype(int))
        assert abs(weighted(counts) - auc(labels[repeated], probabilities[repeated])) < 1e-12
    diagnostic = PairDiagnostics(labels, probabilities)
    mask = np.zeros_like(labels, dtype=bool)
    mask[:, 0, :] = True
    incident = diagnostic.subset(mask)
    all_y, all_p, all_m = labels.ravel(), probabilities.ravel(), mask.ravel()
    wrong = possible = 0
    for i in np.flatnonzero(all_y == 1):
        for j in np.flatnonzero(all_y == 0):
            if all_m[i] or all_m[j]:
                possible += 1
                wrong += float(all_p[i] < all_p[j]) + .5 * float(all_p[i] == all_p[j])
    assert abs(incident["incident_pair_error_rate"] - wrong / possible) < 1e-12
    rows, _ = generator_pairs(labels, probabilities, ["a", "b", "c"])
    assert abs(sum(row["wrong_pairs"] for row in rows) - diagnostic.wrong_pairs) < 1e-12
    dates = pd.Series(pd.date_range("2021-01-01", periods=12, tz="UTC")).dt.as_unit("us")
    for row in paired_block_bootstrap(labels, probabilities, probabilities, [3], 5, starts=dates):
        assert row["percentile_interval_95"] == [0., 0.]
    return {"weighted_auc_matches_explicit_case_repetition_with_ties": True,
            "incident_pair_counts_match_brute_force": True,
            "generator_pair_partition_sums_to_global": True,
            "identical_prediction_paired_bootstrap_is_zero": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/round2_diagnostics.json")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    checks = self_check()
    if args.self_check:
        print(json.dumps(checks, indent=2))
        return
    config = json.loads((ROOT / args.config).read_text())
    output, pair_output = ROOT / config["report"], ROOT / config["pair_report"]
    assert not output.exists() and not pair_output.exists(), "Keep earlier reports intact."
    report = {"config": config, "checks": checks, "results": [],
              "interpretation": [
                  "AUCs sliced by generator/month/horizon are subset metrics; they do not add to pooled micro AUC.",
                  "Incident error shares count each pair once per slice, and overlap between different slices.",
                  "Compare wrong-pair share with available-pair share before calling a group disproportionately difficult.",
                  "Price-water-value margin is a proxy using the same local-reservoir mean as the existing features, not the optimizer's full opportunity cost.",
                  "Both years have been repeatedly used for development; bootstrap intervals do not remove model-selection bias.",
                  "All-case resampling retains 14 generators and 168 hours. Moving blocks additionally retain dependence between daily-start overlapping weeks."
              ]}
    all_pairs = []
    for year in config["years"]:
        cases, generators, labels, winner, predecessor, sources, split = load_year(config, year)
        diagnosis, pair_rows = diagnose(cases, generators, labels, winner, predecessor, config["transition_radius_hours"])
        print(f"{year} final AUC={diagnosis['micro_roc_auc']:.9f}; pairs={diagnosis['generator_pair_composition']}", flush=True)
        diagnosis.update({"year": year, "prediction_sources": sources, "split": split})
        diagnosis["winner_minus_predecessor_block_bootstrap"] = paired_block_bootstrap(
            labels, predecessor, winner, config["block_lengths"], config["bootstrap_repetitions"],
            config["seed"], cases.starttime)
        report["results"].append(diagnosis)
        all_pairs.extend({"year": year, **row} for row in pair_rows)
        print(f"{year} bootstrap: {diagnosis['winner_minus_predecessor_block_bootstrap']}", flush=True)
    with output.open("x") as file:
        file.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    pd.DataFrame(all_pairs).to_csv(pair_output, index=False, mode="x")
    print(f"Saved {output.relative_to(ROOT)} and {pair_output.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
