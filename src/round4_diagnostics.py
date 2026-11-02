"""Diagnose a locked component change against the round3 ensemble from caches.

Example:
  .venv/bin/python src/round4_diagnostics.py --component lightgbm_rescaled \
    --prefix round4_lgb_ranks --variant candidate_replace --years 2021 \
    --output reports/round4_lgb_ranks_native_2021_diagnostics.json
Use --frozen --years 2022 --subsets jan7_onward for pre-2021 models.
No models are loaded, fitted, or selected. Every numerical worker uses one thread.
"""

import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from features import HOURS, ROOT
from plant_blend import read_predictions
from round2_compare import combine
from round2_diagnostics import (
    WeightedAUC, auc, paired_block_bootstrap, transition_mask,
)
from round3_candidate_diagnostics import compare_masks
from round4_compare import COMPONENTS, EXPECTED, FROZEN, NATIVE, RECIPE


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def masked_block_bootstrap(labels, reference, candidate, mask, starts,
                           repetitions=200, block_days=14, seed=4173):
    """Use established exact weighted AUC with whole-case counts on a fixed mask.

    Masked outputs are represented as one-output rows for WeightedAUC. Their
    weights are mapped back to the original daily cases, preserving dependence.
    The circular draws exactly match paired_block_bootstrap for the same seed.
    """
    assert labels.shape == reference.shape == candidate.shape == mask.shape
    assert mask.dtype == bool
    count = len(labels)
    dates = pd.DatetimeIndex(starts).as_unit("ns").asi8
    assert np.all(np.diff(dates) == pd.Timedelta(days=1).value)
    source_cases = np.broadcast_to(
        np.arange(count)[:, None, None], labels.shape,
    )[mask]
    selected_labels = labels[mask, None]
    functions = [WeightedAUC(selected_labels, p[mask, None])
                 for p in (reference, candidate)]
    exact = [fn(np.ones(len(source_cases))) for fn in functions]
    for value, probabilities in zip(exact, (reference, candidate)):
        expected = auc(labels[mask], probabilities[mask])
        assert expected is not None and abs(value - expected) < 1e-12
    rng = np.random.default_rng(seed + block_days)
    deltas = []
    for _ in range(repetitions):
        beginnings = rng.integers(0, count, size=(count + block_days - 1) // block_days)
        sampled = ((beginnings[:, None] + np.arange(block_days)) % count).ravel()[:count]
        counts = np.bincount(sampled, minlength=count)
        weights = counts[source_cases]
        values = [fn(weights) for fn in functions]
        deltas.append(values[1] - values[0])
    deltas = np.asarray(deltas)
    valid = np.isfinite(deltas)
    retained = deltas[valid]
    return {
        "block_days": block_days, "repetitions": repetitions,
        "valid_draws": int(valid.sum()), "invalid_single_class_draws": int((~valid).sum()),
        "reference_auc": exact[0], "candidate_auc": exact[1],
        "observed_delta": exact[1] - exact[0],
        "bootstrap_delta_mean": float(retained.mean()) if len(retained) else None,
        "bootstrap_delta_std": float(retained.std(ddof=1)) if len(retained) > 1 else None,
        "percentile_interval_95": np.quantile(retained, [.025, .975]).tolist() if len(retained) else None,
        "fraction_positive_draws": float((retained > 0).mean()) if len(retained) else None,
    }


def masks_for(labels, generators):
    near2, near6 = transition_mask(labels, 2), transition_mask(labels, 6)
    masks = {"all": np.ones_like(labels, dtype=bool),
             "within_2h_true_transition": near2,
             "away_from_6h_true_transition": ~near6}
    for index, name in enumerate(generators):
        mask = np.zeros_like(labels, dtype=bool)
        mask[:, index, :] = True
        masks[name] = mask
    lio = masks["Lio_G1"]
    masks["Lio_within_2h_true_transition"] = lio & near2
    masks["Lio_away_from_6h_true_transition"] = lio & ~near6
    return masks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", required=True, choices=COMPONENTS)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--variant", required=True, choices=("candidate_replace", "candidate_mean"))
    parser.add_argument("--years", nargs="+", type=int, required=True, choices=(2021, 2022))
    parser.add_argument("--frozen", action="store_true")
    parser.add_argument("--subsets", nargs="+", choices=("all", "jan7_onward"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = ROOT / args.output
    assert output.parent == ROOT / "reports" and output.stem.startswith("round4_")
    assert not output.exists()
    assert Path(args.prefix).name == args.prefix and args.prefix.startswith("round4_")
    assert len(set(args.years)) == len(args.years)
    assert not args.frozen or args.years == [2022]
    mode = "frozen" if args.frozen else "native"
    prefixes = FROZEN if args.frozen else NATIVE
    recipe = json.loads((ROOT / RECIPE).read_text())
    columns = pd.read_csv(ROOT / "data/raw/sample_submission.csv", nrows=0).columns[1:].tolist()
    generators = list(dict.fromkeys(c.removeprefix("result_committed_").rsplit("_t", 1)[0]
                                    for c in columns))
    assert len(columns) == 2352 and len(generators) == 14
    truth_path = ROOT / "data/raw/kernel/Unit_commitment_decisions.csv"
    truth = pd.read_csv(truth_path, usecols=["Run No", "starttime", *columns],
                        dtype={c: "uint8" for c in columns})
    truth["starttime"] = pd.to_datetime(truth.starttime, utc=True)
    test_ids = pd.read_csv(ROOT / "data/raw/prediction_mapping.csv", usecols=["Run No"])["Run No"]
    assert truth["Run No"].is_unique and not truth["Run No"].isin(test_ids).any()
    protected = {p: digest(ROOT / p) for p in (
        RECIPE, "src/round4_diagnostics.py", "src/round4_compare.py",
        "src/round2_compare.py", "src/round2_diagnostics.py",
        "src/round3_candidate_diagnostics.py", "data/raw/sample_submission.csv",
        "data/raw/prediction_mapping.csv", str(truth_path.relative_to(ROOT)),
    )}
    results = []
    for year in args.years:
        cases = truth.loc[truth.starttime.dt.year == year].sort_values("starttime").reset_index(drop=True)
        assert len(cases) == 365

        def read(prefix):
            path = ROOT / f"data/processed/{prefix}_{year}_validation_predictions.csv"
            values = read_predictions(path, columns, cases["Run No"])
            protected[str(path.relative_to(ROOT))] = digest(path)
            return values

        components = {name: read(prefix) for name, prefix in prefixes.items()}
        baseline = combine(components, columns, recipe["plant_weights"])
        new = read(args.prefix)
        changed = components.copy()
        changed[args.component] = (new if args.variant == "candidate_replace"
                                   else .5 * (new + components[args.component]))
        candidate = combine(changed, columns, recipe["plant_weights"])
        shape = (len(cases), len(generators), HOURS)
        labels = cases[columns].to_numpy().reshape(shape)
        baseline, candidate = baseline.reshape(shape), candidate.reshape(shape)
        observed = auc(labels, baseline)
        expected = EXPECTED[mode, year]
        assert abs(observed - expected) < (1e-9 if args.frozen else 1e-12)
        masks = masks_for(labels, generators)
        subsets = args.subsets or (["all", "jan7_onward"] if year == 2022 else ["all"])
        assert year == 2022 or subsets == ["all"]
        for subset in subsets:
            keep = (np.ones(len(cases), dtype=bool) if subset == "all" else
                    (cases.starttime >= pd.Timestamp("2022-01-07", tz="UTC")).to_numpy())
            y, old, new = labels[keep], baseline[keep], candidate[keep]
            selected_masks = {name: mask[keep] for name, mask in masks.items()}
            starts = cases.loc[keep, "starttime"]
            slices = compare_masks(y, old, new, selected_masks)
            uncertainty = paired_block_bootstrap(
                y, old, new, block_lengths=(14,), repetitions=200, seed=4173, starts=starts,
            )[0]
            for name, mask in selected_masks.items():
                if name == "all":
                    continue
                slices[name]["paired_whole_case_bootstrap"] = masked_block_bootstrap(
                    y, old, new, mask, starts,
                )
            results.append({"mode": mode, "year": year, "subset": subset,
                            "cases": int(keep.sum()), "slices": slices,
                            "paired_whole_case_bootstrap": uncertainty})
            print(f"{mode} {year} {subset}: delta {uncertainty['observed_delta']:+.9f}; "
                  f"95% interval {uncertainty['percentile_interval_95']}", flush=True)
    assert all(digest(ROOT / p) == h for p, h in protected.items())
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "recipe": RECIPE,
        "component": args.component, "candidate_prefix": args.prefix,
        "selected_variant": args.variant, "cpu_threads": 1, "new_fitting": False,
        "method": {
            "change": "Only the explicitly named component changes; all plant weights and other components remain fixed.",
            "bootstrap": "200 paired circular 14-day blocks, seed 4173, identical case-count draws across models and slices. Every sampled case retains its outputs. Slice masks are fixed before scores are compared. Percentile intervals condition on selected predictions and do not correct repeated development selection; endpoints are approximate.",
            "slice_bootstrap": "Existing WeightedAUC computes exact masked pooled AUC using the original whole-case repetition counts. Invalid single-class draws, if any, are counted explicitly and excluded only for that slice interval. Positive-draw fraction is not a p-value or a future-improvement probability.",
            "slice_metrics": "Within-slice AUC and incident pair error rate use existing PairDiagnostics. The incident rate includes cross-generator pairs with either endpoint in a slice; overlapping slice incidences must not be summed. Slice bootstrap intervals concern within-slice AUC only.",
            "transition_masks": "Transitions come from the common truth with radii 2 and 6 hours. Masks are diagnostics, never model inputs or selection rules.",
            "interpretation": "Both years are reused development data. Frozen and native 2022 are deployment scenarios on the same labels, not independent replications. Jan7+ excludes six starts overlapping 2021 selection horizons.",
        },
        "source_sha256": protected, "source_hashes_unchanged": True, "results": results,
    }
    with output.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(f"Saved {output.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
