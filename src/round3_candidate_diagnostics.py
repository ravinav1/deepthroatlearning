"""Fixed-mask, paired diagnosis of one XGBoost change in the current ensemble.

Example (after candidate predictions exist):
  .venv/bin/python src/round3_candidate_diagnostics.py \
    --prefix round3_xgb_NAME --variant candidate_mean --years 2021 2022 \
    --output reports/round3_NAME_diagnostics.json

This diagnoses a previously selected variant; it does not choose blend weights.
"""

import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import argparse
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from features import HOURS, ROOT, load_data
from plant_blend import read_predictions
from round2_compare import BASELINE_PREFIXES, combine
from round2_diagnostics import PairDiagnostics, paired_block_bootstrap, transition_mask
from round3_compare import NEW_PREFIXES, RECIPE, load_current_year
from round3_error_analysis import input_context


def fixed_masks(labels, generators, margin, duration, censored):
    near2, near6 = transition_mask(labels, 2), transition_mask(labels, 6)
    lio = np.zeros_like(labels, dtype=bool)
    lio[:, generators.index("Lio_G1"), :] = True
    short = (duration >= 1) & (duration <= 12) & ~censored
    return {
        "all": np.ones_like(labels, dtype=bool),
        "within_2h_true_transition": near2,
        "away_from_6h_true_transition": ~near6,
        "Lio_G1": lio,
        "Lio_within_2h_true_transition": lio & near2,
        "Lio_away_from_6h_true_transition": lio & ~near6,
        "input_favourable_1to12h_nonedge": short,
        "input_favourable_1to12h_nonedge_margin_0to0.1": short & (margin > 0) & (margin <= .1),
    }


def compare_masks(labels, baseline, candidate, masks):
    diagnostics = PairDiagnostics(labels, baseline), PairDiagnostics(labels, candidate)
    rows = {}
    fields = ("micro_roc_auc", "incident_pair_error_rate")
    for name, mask in masks.items():
        reference, changed = (diagnostic.subset(mask) for diagnostic in diagnostics)
        # All labels, membership and pair denominators are identical in this comparison.
        assert reference["outputs"] == changed["outputs"]
        assert reference["incident_available_pair_share"] == changed["incident_available_pair_share"]
        rows[name] = {
            "outputs": reference["outputs"], "output_fraction": reference["output_fraction"],
            "on_fraction": reference["on_fraction"],
            "incident_available_pair_share": reference["incident_available_pair_share"],
            "baseline": {key: reference[key] for key in fields},
            "candidate": {key: changed[key] for key in fields},
            "delta_candidate_minus_baseline": {
                key: changed[key] - reference[key] if changed[key] is not None and reference[key] is not None else None
                for key in fields
            },
        }
    return rows


def replace_depth8(components, columns, weights, candidate, variant):
    changed = components.copy()
    changed["xgb_depth8"] = (candidate if variant == "candidate_replace"
                             else .5 * (components["xgb_depth8"] + candidate))
    return combine(changed, columns, weights)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--variant", choices=["candidate_mean", "candidate_replace"], required=True)
    parser.add_argument("--years", nargs="+", type=int, default=[2021, 2022], choices=[2021, 2022])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = ROOT / args.output
    assert not output.exists(), f"Report already exists: {output}"
    _, _, _, inputs, topology = load_data()
    results, sources = [], {}
    for year in args.years:
        recipe, columns, cases, generators, components, baseline = load_current_year(year)
        path = ROOT / f"data/processed/{args.prefix}_{year}_validation_predictions.csv"
        new = read_predictions(path, columns, cases["Run No"])
        candidate = replace_depth8(components, columns, recipe["plant_weights"], new, args.variant)
        for prefix in [*BASELINE_PREFIXES.values(), *NEW_PREFIXES.values(), args.prefix]:
            source = ROOT / f"data/processed/{prefix}_{year}_validation_predictions.csv"
            sources[str(source.relative_to(ROOT))] = digest(source)
        shape = (len(cases), len(generators), HOURS)
        labels = cases[columns].to_numpy().reshape(shape)
        baseline, candidate = baseline.reshape(shape), candidate.reshape(shape)
        _, _, margin, duration, censored = input_context(cases, generators, inputs, topology)
        masks = fixed_masks(labels, generators, margin, duration, censored)
        subsets = {"all_cases": np.ones(len(cases), dtype=bool)}
        if year == 2022:
            subsets["jan7_onward"] = (cases.starttime >= pd.Timestamp("2022-01-07", tz="UTC")).to_numpy()
        for name, keep in subsets.items():
            slices = compare_masks(labels[keep], baseline[keep], candidate[keep],
                                   {key: mask[keep] for key, mask in masks.items()})
            uncertainty = paired_block_bootstrap(
                labels[keep], baseline[keep], candidate[keep], block_lengths=(14,),
                repetitions=200, seed=4173, starts=cases.loc[keep, "starttime"],
            )[0]
            results.append({"year": year, "subset": name, "cases": int(keep.sum()),
                            "slices": slices, "paired_whole_case_bootstrap": uncertainty})
            print(f"{year} {name}: delta AUC {uncertainty['observed_delta']:+.9f}; "
                  f"14-day interval {uncertainty['percentile_interval_95']}", flush=True)
    for relative in (RECIPE, "src/round3_candidate_diagnostics.py", "src/round3_error_analysis.py",
                     "src/round3_compare.py", "src/round2_diagnostics.py",
                     "data/raw/kernel/Unit_commitment_decisions.csv", "data/raw/sample_submission.csv",
                     "data/raw/prediction_mapping.csv", "data/raw/kernel/Historical_day_ahead_price_2015_2025.csv",
                     "data/raw/kernel/Synthetic_water_value_2015_2024.csv", "configs/Tokke_Vinje_topology.yaml"):
        sources[relative] = digest(ROOT / relative)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "recipe": RECIPE,
        "candidate_prefix": args.prefix, "selected_variant": args.variant,
        "method": {
            "change": "Replace only xgb_depth8, or average old/new within that same contribution. Preserve all plant coefficients and other models; Haukeli receives no depth8 contribution.",
            "interpretation": "Reused historical development years, conditional on selected models and variant. No fitting or weight selection here. Native2022 and any separately reported frozen2022 results use the same labels, not independent replications.",
            "slices": "Membership fixed from common truth or supplied inputs before comparing probabilities. Label-derived masks are diagnostics only. Lio slice AUC is within Lio; its incident rate includes pairs with either endpoint in Lio, including other generators.",
            "pair_rate": "Misordered positive/negative pairs touching the slice divided by all possible pairs touching the slice; ties count one half. Lower is better. Slice AUC is conditional on both endpoints inside the slice; higher is better. Slice incidences overlap and cannot be summed.",
            "input_blocks": "Supplied price above mean local terminal water value; positive blocks1–12h with neither edge touching the168h horizon. Proxy only, not a verified switching rule. Narrow-margin slice additionally uses(price-WV)/max(mean(abs(weeklyprice)),10) in(0,0.1].",
            "bootstrap": "200 paired circular14-day blocks of daily-start cases. Every sampled case retains all14x168outputs. Percentile95% intervals are conditional and do not correct repeated model selection; 200 draws give approximate interval endpoints.",
            "purge": "Jan7+ removes six2022case starts overlapping2021selection horizons. It does not remove previous reuse of2022.",
        },
        "source_sha256": sources, "results": results,
    }
    with output.open("x") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    print(f"Saved {output.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
