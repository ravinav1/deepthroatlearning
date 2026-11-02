"""Descriptive temporal error audit of the frozen round2 winning ensemble.

No training or threshold selection. Run with .venv/bin/python src/round3_error_analysis.py.
Label-derived runs and transition masks are diagnostics, never input features.
"""

import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import hashlib
import json
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from features import ROOT, HOURS, load_data, feeding_reservoirs
from plant_blend import column_groups, read_predictions
from round2_diagnostics import PairDiagnostics, transition_mask

THRESHOLDS = (.3, .5, .7)
RECIPE = "configs/round2_submission_depth8_haukeli_weighted.json"
PREFIXES = {
    "xgb_rescaled": "goal_xgboost_rescaled",
    "lightgbm_rescaled": "goal_lightgbm_rescaled",
    "transformer": "transformer_rescaled",
    "xgb_depth8": "round2_xgb_depth8",
    "haukeli_specialist": "round2_xgb_haukeli",
    "transformer_weighted": "round2_transformer_task_weights",
}
CATEGORIES = ["missed_interior_ON_run", "extra_interior_ON_run", "censored_no_overlap_run",
              "other_error_near_transition_6h", "other_error_away_from_transition_6h"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def auc(y, p):
    return float(roc_auc_score(y.ravel(), p.ravel())) if np.unique(y).size > 1 else None


def runs(sequence, state=1):
    a = np.asarray(sequence) == state
    edges = np.diff(np.r_[False, a, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def matching(real, pred, limit=6):
    """Monotone one-to-one matches: maximize matches, then minimize absolute lag."""
    n, m = len(real), len(pred)
    scores = np.zeros((n + 1, m + 1, 2), dtype=int)
    choice = np.zeros((n + 1, m + 1), dtype=np.int8)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            candidates = [(tuple(scores[i - 1, j]), 1), (tuple(scores[i, j - 1]), 2)]
            distance = abs(int(real[i - 1] - pred[j - 1]))
            if distance <= limit:
                previous = scores[i - 1, j - 1]
                candidates.append(((previous[0] + 1, previous[1] - distance), 3))
            score, step = max(candidates, key=lambda item: item[0])
            scores[i, j], choice[i, j] = score, step
    pairs = []
    i, j = n, m
    while i and j:
        step = choice[i, j]
        if step == 3:
            pairs.append((int(real[i - 1]), int(pred[j - 1])))
            i, j = i - 1, j - 1
        elif step == 1:
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


def current_predictions(cases, columns, year):
    config = json.loads((ROOT / RECIPE).read_text())
    groups, components, sources = column_groups(columns), {}, {}
    for name, prefix in PREFIXES.items():
        path = ROOT / f"data/processed/{prefix}_{year}_validation_predictions.csv"
        # Historical specialist caches are full width; only Haukeli receives its weight.
        part = read_predictions(path, columns, cases["Run No"])
        components[name] = part
        sources[name] = {"path": str(path.relative_to(ROOT)), "sha256": digest(path)}
    result = np.zeros_like(components["transformer"])
    for plant, coefficients in config["plant_weights"].items():
        assert abs(sum(coefficients.values()) - 1) < 1e-12
        for name, weight in coefficients.items():
            result[:, groups[plant]] += weight * components[name][:, groups[plant]]
    expected = json.loads((ROOT / config["historical_report"]).read_text())
    target = next(row["micro_roc_auc"] for row in expected["results"]
                  if row["year"] == year and row["variant"] == config["historical_variant"])
    assert abs(auc(cases[columns].to_numpy(), result) - target) < 1e-12
    return result.reshape(len(cases), 14, HOURS), sources


def threshold_diagnostics(y, p, threshold):
    predicted = p >= threshold
    wrong = predicted != y
    near = transition_mask(y, radius=6)
    category = np.full(y.shape, -1, dtype=np.int8)
    run_counts = {name: 0 for name in ("true_interior_ON", "predicted_interior_ON", "missed_interior_ON",
                                      "extra_interior_ON", "missed_censored_ON", "extra_censored_ON")}
    events = {direction: {"true": 0, "predicted": 0, "lags": []} for direction in ("ON", "OFF")}
    run_records = []
    for case in range(len(y)):
        for generator in range(y.shape[1]):
            truth, guess = y[case, generator], predicted[case, generator]
            true_on, pred_on = runs(truth), runs(guess)
            for lo, hi in true_on:
                censored = lo == 0 or hi == HOURS
                run_counts["true_interior_ON"] += int(not censored)
                if not guess[lo:hi].any():
                    run_counts["missed_censored_ON" if censored else "missed_interior_ON"] += 1
                    category[case, generator, lo:hi] = 2 if censored else 0
                    run_records.append((case, generator, "missed", int(lo), int(hi), censored))
            for lo, hi in pred_on:
                censored = lo == 0 or hi == HOURS
                run_counts["predicted_interior_ON"] += int(not censored)
                if not truth[lo:hi].any():
                    run_counts["extra_censored_ON" if censored else "extra_interior_ON"] += 1
                    category[case, generator, lo:hi] = 2 if censored else 1
                    run_records.append((case, generator, "extra", int(lo), int(hi), censored))
            true_change, pred_change = np.diff(truth.astype(int)), np.diff(guess.astype(int))
            for direction, sign in (("ON", 1), ("OFF", -1)):
                real = np.flatnonzero(true_change == sign) + 1
                observed = np.flatnonzero(pred_change == sign) + 1
                pairs = matching(real, observed)
                events[direction]["true"] += len(real)
                events[direction]["predicted"] += len(observed)
                events[direction]["lags"].extend(b - a for a, b in pairs)
    category[wrong & (category == -1) & near] = 3
    category[wrong & (category == -1) & ~near] = 4
    assert np.array_equal(category >= 0, wrong)
    events_report = {}
    for direction, counts in events.items():
        lag = np.asarray(counts.pop("lags"), dtype=int)
        events_report[direction] = {
            **counts, "matched": len(lag), "unmatched_true": counts["true"] - len(lag),
            "unmatched_predicted": counts["predicted"] - len(lag),
            "early": int((lag < 0).sum()), "exact": int((lag == 0).sum()), "late": int((lag > 0).sum()),
            "matched_median_lag_hours": float(np.median(lag)) if len(lag) else None,
            "lag_histogram": {str(i): int((lag == i).sum()) for i in range(-6, 7)},
        }
    total_wrong = int(wrong.sum())
    per_case_error = wrong.reshape(len(y), -1).mean(axis=1)
    return {
        "threshold": threshold, "error_fraction": float(wrong.mean()), "incorrect_outputs": total_wrong,
        "case_error_fraction_quantiles": dict(zip(["q10", "median", "q90"], np.quantile(per_case_error, [.1, .5, .9]).tolist())),
        "false_positive_outputs": int((wrong & (y == 0)).sum()), "false_negative_outputs": int((wrong & (y == 1)).sum()),
        "error_decomposition": {name: {"outputs": int((category == i).sum()),
                                      "share_of_incorrect_outputs": float((category == i).sum() / total_wrong)}
                                for i, name in enumerate(CATEGORIES)},
        "ON_runs": run_counts, "transition_matching": events_report,
        "entire_week_missed_ON_case_generators": int(((y == 1).all(axis=2) & (predicted == 0).all(axis=2)).sum()),
        "entire_week_extra_ON_case_generators": int(((y == 0).all(axis=2) & predicted.all(axis=2)).sum()),
    }, run_records


def input_context(cases, generators, inputs, topology):
    timestamps = pd.DatetimeIndex(np.repeat(cases.starttime.array, HOURS)) + pd.to_timedelta(np.tile(np.arange(HOURS), len(cases)), unit="h")
    price = inputs["price"].loc[timestamps].iloc[:, 0].to_numpy().reshape(-1, HOURS)
    rank = (pd.DataFrame(price).rank(axis=1, method="average").to_numpy() - 1) / (HOURS - 1)
    scale = np.maximum(np.abs(price).mean(axis=1), 10.)
    margin = []
    for generator in generators:
        reservoirs = feeding_reservoirs(topology, generator.rsplit("_G", 1)[0])
        value = inputs["water_value"].loc[cases.starttime + pd.Timedelta(hours=HOURS), reservoirs].mean(axis=1).to_numpy()
        margin.append((price - value[:, None]) / scale[:, None])
    margin = np.stack(margin, axis=1)
    favourable_length = np.zeros_like(margin, dtype=int)
    censored_block = np.zeros_like(margin, dtype=bool)
    for c in range(len(cases)):
        for g in range(len(generators)):
            for lo, hi in runs(margin[c, g] > 0):
                favourable_length[c, g, lo:hi] = hi - lo
                censored_block[c, g, lo:hi] = lo == 0 or hi == HOURS
    return price, np.broadcast_to(rank[:, None, :], margin.shape), margin, favourable_length, censored_block


def probability_diagnostics(y, p):
    pairs = PairDiagnostics(y, p)
    near2, near6 = transition_mask(y, 2), transition_mask(y, 6)
    confident = ((y == 1) & (p <= .1)) | ((y == 0) & (p >= .9))
    return {
        "micro_roc_auc": auc(y, p), "outputs": y.size,
        "mean_binary_cross_entropy": float(-np.mean(y * np.log(np.clip(p, 1e-12, 1)) + (1-y) * np.log(np.clip(1-p, 1e-12, 1)))),
        "slices": {name: pairs.subset(mask) for name, mask in {
            "within_2h_true_transition": near2, "away_from_2h_true_transition": ~near2,
            "within_6h_true_transition": near6, "away_from_6h_true_transition": ~near6,
            "confident_wrong": confident, "confident_wrong_away_from_6h_transition": confident & ~near6,
        }.items()},
    }


def input_diagnostics(y, p, rank, margin, length, censored):
    pairs = PairDiagnostics(y, p)
    masks = {"rank_bottom20": rank < .2, "rank_middle60": (rank >= .2) & (rank <= .8), "rank_top20": rank > .8,
             "unfavourable": length == 0, "favourable_1to3h": (length >= 1) & (length <= 3),
             "favourable_4to12h": (length >= 4) & (length <= 12), "favourable_13to24h": (length >= 13) & (length <= 24),
             "favourable_over24h": length > 24,
             "favourable_1to12h_margin_0to0.1": (length > 0) & (length <= 12) & (margin <= .1),
             "favourable_over12h_margin_0to0.1": (length > 12) & (margin > 0) & (margin <= .1),
             "favourable_1to12h_nonedge_margin_0to0.1": (length > 0) & (length <= 12) & ~censored & (margin <= .1),
             "favourable_over12h_nonedge_margin_0to0.1": (length > 12) & ~censored & (margin > 0) & (margin <= .1)}
    return {name: {**pairs.subset(mask), "contributing_cases": int(mask.reshape(len(y), -1).any(axis=1).sum())}
            for name, mask in masks.items()}



def matched_input_comparison(y, p, margin, length, censored):
    """Pair short/long blocks within the same case, generator and narrow margin bin."""
    loss = -(y * np.log(np.clip(p, 1e-12, 1)) + (1-y) * np.log(np.clip(1-p, 1e-12, 1)))
    error = (p >= .5) != y
    rows = []
    for case in range(len(y)):
        for generator in range(y.shape[1]):
            for lo, hi in zip((0, .025, .05, .075), (.025, .05, .075, .1)):
                eligible = (margin[case, generator] > lo) & (margin[case, generator] <= hi) & ~censored[case, generator]
                short = eligible & (length[case, generator] > 0) & (length[case, generator] <= 12)
                long = eligible & (length[case, generator] > 12)
                if short.sum() >= 3 and long.sum() >= 3:
                    rows.append({"case": case, "generator": generator,
                                 "delta_bce": float(loss[case, generator, short].mean() - loss[case, generator, long].mean()),
                                 "delta_error_at_0.5": float(error[case, generator, short].mean() - error[case, generator, long].mean())})
    frame = pd.DataFrame(rows)
    # Cases, not dependent hours, are equally weighted in this descriptive summary.
    by_case = frame.groupby("case")[["delta_bce", "delta_error_at_0.5"]].mean()
    return {"strata": len(frame), "cases": len(by_case),
            "definition": "Short1–12h minus longer12h+ nonedge favourable blocks, matching case, generator and normalized-margin bins of width0.025 in(0,0.1]; at least3hours in each group. Strata averaged within case, then cases equally weighted. Descriptive; labels/class composition are not matched, and overlapping cases are dependent.",
            "case_mean_delta_bce": float(by_case.delta_bce.mean()),
            "case_median_delta_bce": float(by_case.delta_bce.median()),
            "fraction_cases_positive_delta_bce": float((by_case.delta_bce > 0).mean()),
            "case_mean_delta_error_at_0.5": float(by_case["delta_error_at_0.5"].mean())}


def timing_plot(results, output):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    colors = ["#b44c46", "#d88944", "#7b73a1", "#509b9b", "#616975"]
    labels = ["Missed interior ON run", "Extra interior ON run", "Censored no-overlap run", "Other error within 6h", "Other error beyond 6h"]
    for row, scope in enumerate(("all", "Lio_G1")):
        entries = [(year, item) for year in (2021, 2022) for item in results[str(year)][scope]["thresholds"]]
        ax = axes[row, 0]
        x, bottom = np.arange(len(entries)), np.zeros(len(entries))
        for category, color, label in zip(CATEGORIES, colors, labels):
            values = [100 * item["error_decomposition"][category]["outputs"] / results[str(year)][scope]["probability"]["outputs"] for year, item in entries]
            ax.bar(x, values, bottom=bottom, color=color, label=label)
            bottom += values
        ax.set_xticks(x, [f"{year}\np≥{item['threshold']}" for year, item in entries])
        ax.set_ylabel("Incorrect outputs (%)")
        ax.set_title(f"{'All generators' if scope == 'all' else 'Lio'}: descriptive error decomposition")
        if row == 0:
            ax.legend(fontsize=8, loc="upper left")
        ax = axes[row, 1]
        for year, color in ((2021, "#2873a8"), (2022, "#b66527")):
            item = next(t for t in results[str(year)][scope]["thresholds"] if t["threshold"] == .5)
            for direction, style in (("ON", "-"), ("OFF", "--")):
                event = item["transition_matching"][direction]
                values = [100 * event["lag_histogram"][str(h)] / event["true"] for h in range(-6, 7)]
                ax.plot(range(-6, 7), values, linestyle=style, marker=".", color=color, label=f"{year} switch {direction}")
        ax.axvline(0, color="gray", linewidth=.8)
        ax.set_xlabel("Predicted switch − true switch (hours; negative = early)")
        ax.set_ylabel("Share of all true switches (%)")
        ax.set_title("One-to-one matching within ±6h; threshold 0.5")
        ax.legend(fontsize=8)
    fig.suptitle("Current 0.99053 recipe: temporal error audit on reused historical years\nThreshold errors describe behaviour; micro ROC-AUC remains the evaluation metric", fontsize=14)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def example_plot(examples, output):
    fig, axes = plt.subplots(len(examples), 2, figsize=(14, 3.1 * len(examples)), constrained_layout=True)
    for row, example in enumerate(examples):
        ax, price_ax = axes[row]
        h = np.arange(HOURS)
        ax.step(h, example["truth"], where="post", color="black", label="Solver ON/OFF", linewidth=1.5)
        ax.plot(h, example["probability"], color="#2873a8", label="Current probability")
        ax.axhline(.5, color="gray", linestyle=":", linewidth=.8)
        ax.set(ylim=(-.05, 1.05), xlim=(0, 167), ylabel="ON probability / label", title=f"{example['generator']}, {example['starttime'][:10]}, case {example['case_id']}\n{example['selection_reason']}")
        ax.legend(fontsize=8, loc="upper right")
        price_ax.plot(h, example["price"], color="#2873a8", label="Supplied price")
        price_ax.axhline(example["local_water_value"], color="#b66527", linestyle="--", label="Local terminal WV proxy")
        price_ax.set(ylabel="EUR/MWh", xlim=(0, 167), title="Inputs define candidate favourable blocks; they do not force ON")
        price_ax.legend(fontsize=8, loc="upper right")
        for target in (ax, price_ax):
            target.set_xlabel("Hour within this case (0–167)")
            target.grid(alpha=.15)
    fig.suptitle("Illustrative high-error cases selected deterministically from 2022; not representative samples", fontsize=13)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main():
    output = ROOT / "reports/round3_errors.json"
    figures = [ROOT / f"reports/figures/round3_errors_{name}.png" for name in ("timing", "examples")]
    assert not output.exists() and all(not p.exists() for p in figures)
    cases_all, columns, generators, inputs, topology = load_data()
    competition_ids = pd.read_csv(ROOT / "data/raw/prediction_mapping.csv")["Run No"]
    assert not cases_all["Run No"].isin(competition_ids).any()
    results, sources, examples = {}, {}, []
    for year in (2021, 2022):
        cases = cases_all.loc[cases_all.starttime.dt.year == year].reset_index(drop=True)
        y = cases[columns].to_numpy().reshape(len(cases), len(generators), HOURS)
        p, sources[str(year)] = current_predictions(cases, columns, year)
        price, rank, margin, length, edge = input_context(cases, generators, inputs, topology)
        year_results = {}
        for scope, indices in (("all", list(range(14))), ("Lio_G1", [generators.index("Lio_G1")])):
            a, b = y[:, indices], p[:, indices]
            threshold_results = [threshold_diagnostics(a, b, t)[0] for t in THRESHOLDS]
            year_results[scope] = {
                "probability": probability_diagnostics(a, b), "thresholds": threshold_results,
                "input_slices": input_diagnostics(a, b, rank[:, indices], margin[:, indices], length[:, indices], edge[:, indices]),
                "matched_input_comparison": matched_input_comparison(a, b, margin[:, indices], length[:, indices], edge[:, indices]),
            }
        year_results["generator_auc"] = {g: auc(y[:, i], p[:, i]) for i, g in enumerate(generators)}
        results[str(year)] = year_results
        print(json.dumps({"year": year, "micro_auc": year_results["all"]["probability"]["micro_roc_auc"],
                          "threshold_summary": [(s, [(t['threshold'], t['error_fraction']) for t in year_results[s]['thresholds']]) for s in ('all', 'Lio_G1')]}, indent=2), flush=True)
        if year == 2022:
            _, records = threshold_diagnostics(y, p, .5)
            lio = generators.index("Lio_G1")
            # Longest fully missed interior Lio ON run; ties resolve by earliest case/hour.
            missed = [r for r in records if r[1] == lio and r[2] == "missed" and not r[5]]
            case, gen, _, lo, hi, _ = max(missed, key=lambda r: (r[4] - r[3], -r[0], -r[3]))
            choices = [(case, gen, f"Longest completely missed interior Lio ON run: {hi-lo}h (p≥0.5)")]
            # A non-overlapping Lio week with greatest probability loss away from true transitions.
            away = ~transition_mask(y, 6)
            loss = -(y * np.log(np.clip(p, 1e-12, 1)) + (1-y) * np.log(np.clip(1-p, 1e-12, 1)))
            scores = (loss[:, lio] * away[:, lio]).sum(axis=1)
            scores[np.abs(np.arange(len(cases)) - case) < 7] = -np.inf
            choices.append((int(scores.argmax()), lio, "Largest Lio loss >6h from transitions, excluding overlapping example weeks"))
            # Non-Lio case with largest sum of probability loss, avoiding repeated Lio diagnosis.
            loss_by_case_generator = loss.sum(axis=2)
            loss_by_case_generator[:, lio] = -np.inf
            case, gen = np.unravel_index(loss_by_case_generator.argmax(), loss_by_case_generator.shape)
            choices.append((int(case), int(gen), "Largest non-Lio case/generator probability loss in 2022"))
            for case, gen, reason in choices:
                scale = max(float(np.abs(price[case]).mean()), 10.)
                examples.append({"year": year, "case_id": int(cases.iloc[case]["Run No"]),
                                 "starttime": str(cases.iloc[case].starttime), "generator": generators[gen],
                                 "selection_reason": reason, "truth": y[case, gen].tolist(), "probability": p[case, gen].tolist(),
                                 "price": price[case].tolist(), "local_water_value": float(price[case, 0] - margin[case, gen, 0] * scale)})
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "recipe": RECIPE, "recipe_sha256": digest(ROOT / RECIPE),
        "method": {
            "status": "Descriptive EDA on reused 2021/2022 development predictions, not independent evidence or tuning.",
            "dependence": "All cases retain 14 generators x168h. Cases overlap daily; no hourly independence or significance claims. Counts are per-case occurrences, not unique calendar events.",
            "probability_primary": "Micro ROC-AUC and misordered positive/negative pairs. Incident pair shares overlap across slices and cannot be added.",
            "binary_thresholds": list(THRESHOLDS), "binary_note": "Fixed descriptive thresholds only. No threshold is chosen for deployment; differences may reflect calibration.",
            "events": "Match same-direction true/predicted transitions one-to-one in temporal order within6h, maximizing matches then minimizing absolute delay. A matched event can be associated with a fragmented run. Events at observed week edges are not imputed.",
            "runs": "A missed interior ON run has both boundaries within hours0..167 and no predicted ON anywhere inside it; extra run is analogous. Edge-censored no-overlap runs are separately identified. No hard minimum dwell is imposed.",
            "decomposition": "Mutually exclusive incorrect-hour categories: missed interior ON, extra interior ON, no-overlap censored run, remaining error within±6h of true transition, remaining error farther away.",
            "favourable": "Input-only proxy: hourly supplied price > mean terminal water value of directly feeding reservoirs. This is not a verified optimizer switching threshold. Length truncated to supplied168h; edge-touching blocks also reported separately.",
            "rank": "Average tied price rank within each168h supplied week, normalized to[0,1].",
            "input_interpretation": "Slice AUC/rates are descriptive and confounded by generator, season, class composition and economic margin. They do not establish feature benefit.",
        },
        "sources": sources,
        "input_hashes": {path: digest(ROOT / path) for path in (
            "data/raw/kernel/Unit_commitment_decisions.csv", "data/raw/sample_submission.csv", "data/raw/prediction_mapping.csv",
            "data/raw/kernel/Historical_day_ahead_price_2015_2025.csv", "data/raw/kernel/Synthetic_water_value_2015_2024.csv",
            "configs/Tokke_Vinje_topology.yaml")},
        "analysis_script_sha256": digest(ROOT / "src/round3_error_analysis.py"),
        "interpretation": {
            "primary_candidate": "Input-defined favourable-block duration has a stable short-block error concentration. Same-case/generator/narrow-margin comparisons reduce obvious composition differences, but do not prove a causal or predictive gain. Test features with a controlled fit.",
            "price_rank": "No stable rank-only error concentration; weekly rank remains a cheap transformation/invariance hypothesis, not an EDA-established remedy.",
            "timing": "Many matched transitions are exact; early/late direction is not stable between years, especially Lio. No universal time shift or hard dwell rule is justified.",
            "lio": "Large Lio errors remain away from transitions. Negative-margin proxy hours contain both substantial ON prevalence and most within-Lio wrong-pair incidence, but also almost all available pairs, so this is not enrichment evidence. Positive-margin duration is not a demonstrated Lio-specific explanation.",
            "generalization_scope": "This audit uses separate native historical fits for2021and2022. The latest pre2021 frozen models improve2022 vs the older frozen baseline, includingLio, as documented in reports/round3_frozen_ensemble_comparison.json. Native-refit Lio regression does not establish a universal second-year transfer failure.",
            "confident_slice": "Confident-wrong slices use labels by construction; their within-slice AUC is tautological. Output prevalence and incident wrong-pair mass describe concentration; they are never features.",
            "examples": "Deterministically selected high-error illustrations, not a representative sample. Price/local-WV plots alone do not explain reservoir constraints or solver decisions."
        },
        "results": results, "examples": examples,
        "figures": [str(p.relative_to(ROOT)) for p in figures],
    }
    timing_plot(results, figures[0])
    example_plot(examples, figures[1])
    with output.open("x") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
