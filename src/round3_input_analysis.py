"""Input-only, season-matched case analysis. No labels or model fitting."""

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, wasserstein_distance
import yaml

from features import feeding_reservoirs


ROOT = Path(__file__).resolve().parents[1]
HOURS = 168


def read_inputs():
    raw = ROOT / "data/raw"
    # Deliberately do not load commitment columns, even for historical cases.
    historical = pd.read_csv(raw / "kernel/Unit_commitment_decisions.csv",
                             usecols=["Run No", "starttime"])
    competition = pd.read_csv(raw / "prediction_mapping.csv")
    assert set(historical["Run No"]).isdisjoint(competition["Run No"])
    cases = pd.concat([historical, competition], ignore_index=True)
    cases["starttime"] = pd.to_datetime(cases["starttime"], utc=True)
    cases = cases.sort_values("starttime").reset_index(drop=True)
    names = {
        "price": "Historical_day_ahead_price_2015_2025.csv",
        "volume": "Historical_volume_2015_2024.csv",
        "water_value": "Synthetic_water_value_2015_2024.csv",
        "inflow": "Historical_inflow_1958_2025.csv",
    }
    inputs = {}
    paths = [raw / "kernel/Unit_commitment_decisions.csv", raw / "prediction_mapping.csv"]
    for key, filename in names.items():
        path = raw / "kernel" / filename
        frame = pd.read_csv(path)
        frame["date"] = pd.to_datetime(frame["date"], utc=True)
        inputs[key] = frame.set_index("date")
        paths.append(path)
    topology_path = ROOT / "configs/Tokke_Vinje_topology.yaml"
    topology = yaml.safe_load(topology_path.read_text())
    paths.append(topology_path)
    return cases, inputs, topology, paths


def episode_summary(mask):
    """Case summaries of input-defined positive blocks, clipped to the supplied week."""
    longest, count, weighted_length, boundary_positive = [], [], [], []
    for row in mask:
        changes = np.diff(np.r_[False, row, False].astype("int8"))
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        lengths = ends - starts
        longest.append(lengths.max(initial=0))
        count.append(len(lengths))
        weighted_length.append(np.sum(lengths ** 2) / max(1, lengths.sum()))
        boundary_positive.append(bool(row[0] or row[-1]))
    return {
        "longest_block_h": np.asarray(longest),
        "blocks_per_week": np.asarray(count),
        "positive_hour_mean_block_h": np.asarray(weighted_length),
        "boundary_positive": np.asarray(boundary_positive, dtype=float),
    }


def construct_summaries(cases, inputs, topology):
    starts = cases["starttime"]
    stamps = pd.DatetimeIndex(np.repeat(starts.array, HOURS))
    stamps += pd.to_timedelta(np.tile(np.arange(HOURS), len(cases)), unit="h")
    price = inputs["price"].loc[stamps].iloc[:, 0].to_numpy().reshape(-1, HOURS)
    scale = np.maximum(np.abs(price).mean(axis=1), 10.)
    ranks = (rankdata(price, axis=1, method="average") - 1) / (HOURS - 1)
    frame = cases.to_dict("series")
    frame["year"] = starts.dt.year
    frame["month"] = starts.dt.month
    frame["day"] = starts.dt.day
    frame["week_price_mean_eur_mwh"] = price.mean(axis=1)
    frame["week_price_scale_eur_mwh"] = scale
    frame["week_price_sd_scaled"] = price.std(axis=1) / scale
    frame["week_price_mean_abs_hourly_change_scaled"] = np.abs(np.diff(price, axis=1)).mean(axis=1) / scale
    frame["negative_price_fraction"] = (price < 0).mean(axis=1)
    frame["price_rank_mean_abs_hourly_change"] = np.abs(np.diff(ranks, axis=1)).mean(axis=1)
    frame["price_rank_lag24_correlation"] = [np.corrcoef(row[:-24], row[24:])[0, 1] for row in ranks]
    for name, value in episode_summary(ranks >= .75).items():
        frame[f"top_quartile_{name}"] = value
    plant_reservoirs = {}
    for plant in topology["model"]["plant"]:
        reservoirs = feeding_reservoirs(topology, plant)
        plant_reservoirs[plant] = reservoirs
        volume = inputs["volume"].loc[starts, reservoirs].sum(axis=1).to_numpy()
        capacity = sum(topology["model"]["reservoir"][r]["max_vol"] for r in reservoirs)
        water_value = inputs["water_value"].loc[starts + pd.Timedelta(hours=HOURS), reservoirs].mean(axis=1).to_numpy()
        flow = inputs["inflow"].loc[stamps, reservoirs].sum(axis=1).to_numpy().reshape(-1, HOURS)
        margin = (price - water_value[:, None]) / scale[:, None]
        prefix = plant + "_"
        frame[prefix + "initial_fill_fraction"] = volume / capacity
        frame[prefix + "week_natural_inflow_capacity_fraction"] = flow.sum(axis=1) * 3600 / 1e6 / capacity
        frame[prefix + "terminal_water_value_scaled"] = water_value / scale
        frame[prefix + "mean_price_water_margin_scaled"] = margin.mean(axis=1)
        frame[prefix + "positive_margin_fraction"] = (margin > 0).mean(axis=1)
        frame[prefix + "near_margin_fraction"] = (np.abs(margin) < .1).mean(axis=1)
        frame[prefix + "all_positive_week"] = (margin > 0).all(axis=1).astype(float)
        frame[prefix + "all_nonpositive_week"] = (margin <= 0).all(axis=1).astype(float)
        for name, value in episode_summary(margin > 0).items():
            frame[prefix + name] = value
    # Individual reservoir inputs avoid masking system shifts by averaging plants.
    for reservoir in inputs["volume"].columns:
        capacity = topology["model"]["reservoir"][reservoir]["max_vol"]
        frame[reservoir + "_reservoir_initial_fill_fraction"] = inputs["volume"].loc[starts, reservoir].to_numpy() / capacity
        flow = inputs["inflow"].loc[stamps, reservoir].to_numpy().reshape(-1, HOURS)
        frame[reservoir + "_reservoir_week_natural_inflow_capacity_fraction"] = flow.sum(axis=1) * 3600 / 1e6 / capacity
    frame = pd.DataFrame(frame)
    assert np.isfinite(frame.select_dtypes(include="number")).all().all()
    return frame, plant_reservoirs


def describe(values):
    return {"mean": float(np.mean(values)), "q10": float(np.quantile(values, .1)),
            "median": float(np.median(values)), "q90": float(np.quantile(values, .9))}


def compare_months(reference, target, columns):
    result = {}
    for column in columns:
        details = []
        for month in range(1, 13):
            past = reference.loc[reference.month == month, column].to_numpy()
            future = target.loc[target.month == month, column].to_numpy()
            q01, q25, q75, q99 = np.quantile(past, [.01, .25, .75, .99])
            spread = q75 - q25
            # Sparse constant features can have zero IQR. Their raw difference and
            # tail frequencies remain informative; do not invent a standardized score.
            distance = wasserstein_distance(past, future)
            details.append({
                "month": month, "reference_n": len(past), "target_n": len(future),
                "reference_median": float(np.median(past)), "target_median": float(np.median(future)),
                "reference_iqr": float(spread), "wasserstein_raw": float(distance),
                "wasserstein_reference_iqr": float(distance / spread) if spread > 1e-10 else None,
                "target_outside_reference_p01_p99": float(np.mean((future < q01) | (future > q99))),
            })
        valid = [d["wasserstein_reference_iqr"] for d in details if d["wasserstein_reference_iqr"] is not None]
        result[column] = {
            "equal_month_mean_wasserstein_reference_iqr": float(np.mean(valid)) if valid else None,
            "nonzero_iqr_months": len(valid),
            "equal_month_mean_outside_reference_p01_p99": float(np.mean([d["target_outside_reference_p01_p99"] for d in details])),
            "months": details,
        }
    return result


def plot_months(frame, output):
    specs = [
        ("week_price_mean_eur_mwh", "Weekly mean price", "EUR/MWh"),
        ("week_price_sd_scaled", "Within-week price variation", "SD / weekly price scale"),
        ("price_rank_mean_abs_hourly_change", "Movement in within-week price rank", "Mean absolute hourly rank change"),
        ("Lio_positive_margin_fraction", "Lio: hours with price above water value", "Fraction of supplied week"),
        ("Lio_longest_block_h", "Lio: longest positive-margin block", "Hours, clipped at week boundaries"),
        ("Lio_initial_fill_fraction", "Lio: initial reservoir storage", "Fraction of nominal capacity"),
    ]
    colors = {2021: "#4c78a8", 2022: "#e45756", 2023: "#f2a541", 2024: "#1b9e77"}
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for ax, (column, title, unit) in zip(axes.ravel(), specs):
        reference = frame.loc[frame.year <= 2020].groupby("month")[column]
        month = np.arange(1, 13)
        ax.fill_between(month, reference.quantile(.1), reference.quantile(.9), color="#b0b0b0", alpha=.23,
                        label="2015–20 case 10–90%")
        ax.plot(month, reference.median(), color="#777777", linewidth=1.6, linestyle="--", label="2015–20 median")
        for year, color in colors.items():
            ax.plot(month, frame.loc[frame.year == year].groupby("month")[column].median(),
                    color=color, linewidth=1.7, marker="o", markersize=3, label=str(year))
        ax.set(title=title, ylabel=unit, xticks=month, xlabel="Case-start month")
        ax.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=6, frameon=False)
    fig.suptitle("Input distributions by season: 2023 public and 2024 private inputs\n"
                 "Matched Jan 1–Dec 25 case starts; 29 February excluded in every year", fontsize=14)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_shift(comparison, output):
    labels = {
        "week_price_mean_eur_mwh": "Weekly mean price (raw)",
        "week_price_sd_scaled": "Price SD / weekly scale",
        "week_price_mean_abs_hourly_change_scaled": "Hourly price change / scale",
        "price_rank_mean_abs_hourly_change": "Hourly rank change",
        "top_quartile_longest_block_h": "Longest top-quartile block",
        "Lio_terminal_water_value_scaled": "Lio terminal water value / scale",
        "Lio_positive_margin_fraction": "Lio positive-margin share",
        "Lio_longest_block_h": "Lio longest positive block",
        "Lio_initial_fill_fraction": "Lio initial fill",
        "Lio_week_natural_inflow_capacity_fraction": "Lio weekly natural inflow / capacity",
        "Haukeli_positive_margin_fraction": "Haukeli positive-margin share",
        "Haukeli_initial_fill_fraction": "Haukeli initial fill",
        "Haukeli_week_natural_inflow_capacity_fraction": "Haukeli weekly natural inflow / capacity",
        "Tokke_positive_margin_fraction": "Tokke positive-margin share",
        "Tokke_initial_fill_fraction": "Tokke initial fill",
        "Tokke_week_natural_inflow_capacity_fraction": "Tokke weekly natural inflow / capacity",
    }
    array = np.array([[comparison[str(year)][key]["equal_month_mean_wasserstein_reference_iqr"]
                       for year in [2021, 2022, 2023, 2024]] for key in labels], dtype=float)
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    im = ax.imshow(array, cmap="YlOrRd", aspect="auto", vmin=0, vmax=3)
    ax.set(xticks=range(4), xticklabels=["2021", "2022", "2023 public", "2024 private"],
           yticks=range(len(labels)), yticklabels=list(labels.values()))
    for row in range(array.shape[0]):
        for col in range(array.shape[1]):
            value = array[row, col]
            label = f"{value:.2f}" if np.isfinite(value) else "constant"
            ax.text(col, row, label, ha="center", va="center", color="white" if value > 1.7 else "black")
    ax.set_title("Month-matched input shift relative to 2015–20\n"
                 "Wasserstein distance / reference IQR, averaged equally over months", pad=18)
    fig.colorbar(im, ax=ax, label="Distribution distance (reference IQR units; colour clipped at 3)", shrink=.85)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main():
    report_path = ROOT / "reports/round3_inputs_analysis.json"
    figures = [ROOT / "reports/figures/round3_inputs_seasonality.png",
               ROOT / "reports/figures/round3_inputs_shift.png"]
    cases, inputs, topology, paths = read_inputs()
    frame, reservoirs = construct_summaries(cases, inputs, topology)
    # Private cases end on Dec 25. Match that calendar support across every year,
    # also removing Feb 29 rather than confounding year effects with a leap day.
    matched = frame.loc[~((frame.month == 12) & (frame.day > 25)) &
                        ~((frame.month == 2) & (frame.day == 29))].copy()
    columns = [c for c in frame if c not in ["Run No", "starttime", "year", "month", "day"]]
    references = {"2015_2020": matched.loc[matched.year <= 2020],
                  "2021_2022": matched.loc[matched.year.isin([2021, 2022])],
                  "2023": matched.loc[matched.year == 2023]}
    comparisons = {name: {str(year): compare_months(reference, matched.loc[matched.year == year], columns)
                          for year in ([2021, 2022, 2023, 2024] if name == "2015_2020" else
                                       [2023, 2024] if name == "2021_2022" else [2024])}
                   for name, reference in references.items()}
    annual = {str(year): {"cases": len(group), "features": {c: describe(group[c]) for c in columns}}
              for year, group in matched.loc[matched.year >= 2021].groupby("year")}
    annual["2015_2020"] = {"cases": len(references["2015_2020"]),
                            "features": {c: describe(references["2015_2020"][c]) for c in columns}}
    plot_months(matched, figures[0])
    plot_shift(comparisons["2015_2020"], figures[1])
    # Keep the report compact: all features retain aggregate shift measures;
    # monthly detail is available for the focused price and Lio diagnostics.
    detailed = ["week_price_mean_eur_mwh", "week_price_sd_scaled",
                "week_price_mean_abs_hourly_change_scaled", "price_rank_mean_abs_hourly_change",
                "top_quartile_longest_block_h", "Lio_positive_margin_fraction",
                "Lio_longest_block_h", "Lio_initial_fill_fraction",
                "Lio_week_natural_inflow_capacity_fraction", "Haukeli_positive_margin_fraction"]
    monthly = {str(year): {name: [float(v) for v in matched.loc[matched.year == year].groupby("month")[name].median()]
                          for name in detailed} for year in [2021, 2022, 2023, 2024]}
    for reference in comparisons.values():
        for year_comparison in reference.values():
            for feature_comparison in year_comparison.values():
                feature_comparison.pop("months")
    report = {
        "purpose": "Input-only seasonal distribution analysis; no ON/OFF labels loaded and no model fitting.",
        "cases_by_year_all": {str(y): int(n) for y, n in frame.groupby("year").size().items()},
        "cases_by_year_calendar_matched": {str(y): int(n) for y, n in matched.groupby("year").size().items()},
        "calendar_matching": "Jan 1 through Dec 25 case starts in every year; February 29 removed. Equal weighting of monthly distribution distances.",
        "case_horizon": "168 supplied hourly prices/inflows; volume only at case start; water value at start+168h.",
        "price_scale": "max(mean(abs(168 prices)),10), identical to existing economic rescaling.",
        "favourable_proxy": "Strict price > mean feeding-reservoir terminal water value. This ignores transfers, constraints, efficiency/start costs and is not a feasibility or optimality rule.",
        "block_censoring": "Lengths and distance concepts stop at supplied week boundaries; a positive edge is censored, not a known switch.",
        "rank_definition": "Average tie rank within each 168-hour case, scaled (rank-1)/167. Marginal near-uniformity follows by construction and does not establish predictive stability.",
        "shift_metric": "Descriptive month-matched Wasserstein/reference IQR; zero-IQR months omitted only from standardized distance, count recorded. P01/P99 exceedance also reported. No IID p-values because daily cases overlap.",
        "feeding_reservoirs": reservoirs,
        "annual_calendar_matched_summaries": annual,
        "monthly_case_medians_january_to_december": monthly,
        "month_matched_comparisons": comparisons,
        "findings": [
            "Private-year 2024 weekly normalized price SD has median 0.399, versus 0.288 in 2023 and 0.219 in 2022. Existing monetary rescaling does not remove the volatility shift.",
            "Rank temporal structure is steadier: median absolute adjacent rank movement is 0.0746/0.0724/0.0745/0.0731 in 2024/23/22/21, and median longest top-quartile blocks 15/14/14/15 hours. These are sequence-shape summaries, not the trivially uniform rank marginal.",
            "Lio 2024 median positive-margin share is 0.661, versus 0.244 in 2023 and 0.024 in 2022. Median longest positive block is 42/14/3 hours. Initial fill is 0.905/0.736/0.782, and natural-inflow/capacity median 0.030/0.013/0.009. These are input regimes, not inferred operating labels.",
            "Haukeli initial fill is exactly 0.5 throughout. Entirely nonpositive-margin weeks fall from 53.2% in 2023 to 1.95% in 2024; median weekly inflow/capacity rises 0.141 to 0.338. A public-year specialist can face substantially different private-year economics and inflow.",
            "2024 normalized price SD and hourly change lie outside monthly 2021–22 p01/p99 ranges for 35.4% and 39.9% of cases, compared with 8.6% for adjacent rank movement. These descriptive tail frequencies ignore dependence for inference and are not hypothesis-test significance.",
            "Positive Lio blocks touch either horizon boundary in 57.9% of 2024 cases versus 36.2% in 2023. Block features must signal horizon censoring and never read prices or realized storage outside the supplied case."
        ],
        "candidate_recommendations": [
            "Test within-week and remaining-week midranks alongside current prices, water values and scaled margins. The rank level alone does not establish stability; differences in sequence shape remain meaningful.",
            "Separately test input-defined positive-margin block duration, elapsed/remaining hours and distance to the next/previous favourable hour, with horizon censor flags. A positive margin is an economic proxy, not guaranteed feasibility or an ON label."
        ],
        "not_new": "Fixed 6/12/24-hour integrated price-water-value margins with reference power and start/stop costs are already implemented and previously tested. Do not treat these as a new block-duration experiment.",
        "limits": [
            "No performance benefit is established by input-only EDA; evaluate candidates chronologically and keep original monetary features.",
            "All daily weeks overlap, and month-start grouping still includes later-month hours. No IID p-values or effective-sample-size claims are used.",
            "One-year reference IQRs may be tiny, inflating standardized distances. Use raw summaries and nonzero-IQR-month counts; zero-IQR comparisons are missing rather than an artificial standardized zero.",
            "This analysis uses public/private inputs to identify shifts, not test outcomes. It does not infer private accuracy or choose thresholds using hidden labels.",
            "Starting storage divided by nominal maximum can exceed 1 in the supplied data; values are retained, not silently clipped."
        ],
        "figures": [str(p.relative_to(ROOT)) for p in figures],
        "input_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(report_path), "matched_counts": report["cases_by_year_calendar_matched"],
                      "feeding_reservoirs": reservoirs}, indent=2))


if __name__ == "__main__":
    main()
