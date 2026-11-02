"""Compare fixed component replacements against the 0.99068 round3 recipe.

Native: --replace lightgbm_rescaled=round4_lgb_ranks --years 2021 2022
Frozen: --frozen --replace lightgbm_rescaled=round4_frozen_lgb_ranks
Prefixes omit the year; --output is always required. No model libraries loaded.
For a mixed variant, --mean-component transformer_weighted makes that slot use
its fixed old/new mean in candidate_replace. candidate_mean always means every
named component uses its old/new mean, preserving the original CLI behavior.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import pandas as pd

from features import ROOT
from plant_blend import metrics, read_predictions
from round2_compare import combine
from round3_frozen_evaluation import digest, prediction_path, write_json


RECIPE = "configs/round3_submission_temporal_ranks.json"
NATIVE = {
    "xgb_rescaled": "goal_xgboost_rescaled",
    "lightgbm_rescaled": "goal_lightgbm_rescaled",
    "transformer": "transformer_rescaled",
    "xgb_depth8": "round3_xgb_ranks",
    "haukeli_specialist": "round2_xgb_haukeli",
    "transformer_weighted": "round2_transformer_task_weights",
}
FROZEN = {
    "xgb_rescaled": "round2_frozen_xgb",
    "lightgbm_rescaled": "round2_frozen_lgb",
    "transformer": "round2_frozen_transformer",
    "xgb_depth8": "round3_frozen_xgb_ranks",
    "haukeli_specialist": "round3_frozen_haukeli_specialist",
    "transformer_weighted": "round3_frozen_transformer_weighted",
}
EXPECTED = {("native", 2021): .9889840634255942,
            ("native", 2022): .988606657825117,
            ("frozen", 2022): .9864608919020703}
COMPONENTS = ("lightgbm_rescaled", "transformer_weighted")


def official_metric():
    spec = importlib.util.spec_from_file_location("round4_provided_metric", ROOT / "evaluation/kaggle_metric.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replace", action="append", default=[], metavar="COMPONENT=PREFIX")
    parser.add_argument("--mean-component", action="append", default=[], choices=COMPONENTS,
                        help="Use the named replacement's 50/50 old/new mean in candidate_replace; candidate_mean is unchanged.")
    parser.add_argument("--years", nargs="+", type=int, default=None, choices=[2021, 2022])
    parser.add_argument("--frozen", action="store_true")
    parser.add_argument("--identity-check", action="store_true", help="Use the two original component caches as replacements.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--save-prefix")
    args = parser.parse_args()
    mode = "frozen" if args.frozen else "native"
    years = args.years or ([2022] if args.frozen else [2021])
    assert len(set(years)) == len(years) and (not args.frozen or years == [2022])
    output = ROOT / args.output
    assert output.stem.startswith("round4_") and not output.exists(), output
    replacements = dict(item.split("=", 1) for item in args.replace)
    assert len(replacements) == len(args.replace) and set(replacements).issubset(COMPONENTS)
    assert not (args.identity_check and replacements)
    prefixes = FROZEN if args.frozen else NATIVE
    if args.identity_check:
        replacements = {name: prefixes[name] for name in COMPONENTS}
    else:
        assert all(Path(prefix).name == prefix and prefix.startswith("round4_") for prefix in replacements.values())
    assert len(set(args.mean_component)) == len(args.mean_component), "Name each mean component once."
    assert set(args.mean_component).issubset(replacements), "Each mean component needs a --replace source."
    if args.save_prefix:
        assert Path(args.save_prefix).name == args.save_prefix and args.save_prefix.startswith("round4_")
        for year in years:
            for variant in ("candidate_replace", "candidate_mean"):
                assert not prediction_path(f"{args.save_prefix}_{variant}_{year}").exists()
    recipe = json.loads((ROOT / RECIPE).read_text())
    columns = pd.read_csv(ROOT / "data/raw/sample_submission.csv", nrows=0).columns[1:].tolist()
    generators = list(dict.fromkeys(c.removeprefix("result_committed_").rsplit("_t", 1)[0] for c in columns))
    truth_path = ROOT / "data/raw/kernel/Unit_commitment_decisions.csv"
    truth = pd.read_csv(truth_path, usecols=["Run No", "starttime", *columns],
                        dtype={c: "uint8" for c in columns})
    truth["starttime"] = pd.to_datetime(truth.starttime, utc=True)
    test_ids = pd.read_csv(ROOT / "data/raw/prediction_mapping.csv", usecols=["Run No"])["Run No"]
    assert not truth["Run No"].isin(test_ids).any() and truth["Run No"].is_unique
    protected = {str(p.relative_to(ROOT)): digest(p) for p in
                 (ROOT / RECIPE, truth_path, ROOT / "evaluation/kaggle_metric.py")}
    score, rows, saved, reconstruction = official_metric(), [], {}, []
    for year in years:
        cases = truth.loc[truth.starttime.dt.year == year].sort_values("starttime").reset_index(drop=True)
        assert len(cases) == 365 and len(columns) == 2352

        def read(prefix):
            path = prediction_path(f"{prefix}_{year}")
            values = read_predictions(path, columns, cases["Run No"])
            protected[str(path.relative_to(ROOT))] = digest(path)
            return values

        components = {name: read(prefix) for name, prefix in prefixes.items()}
        baseline = combine(components, columns, recipe["plant_weights"])
        independent = np.column_stack([
            sum(weight * components[name][:, i] for name, weight in recipe["plant_weights"][
                column.removeprefix("result_committed_").rsplit("_G", 1)[0]].items())
            for i, column in enumerate(columns)
        ])
        assert np.array_equal(baseline, independent)
        observed = metrics(cases[columns].to_numpy(), baseline, generators)["micro_roc_auc"]
        expected = EXPECTED[mode, year]
        assert abs(observed - expected) < 1e-9 if args.frozen else observed == expected
        reconstruction.append({"year": year, "recorded_micro_roc_auc": expected,
                               "csv_reconstructed_micro_roc_auc": observed,
                               "difference_from_recorded": observed - expected})
        changed, averaged = components.copy(), components.copy()
        for name, prefix in replacements.items():
            candidate = read(prefix)
            averaged[name] = .5 * (components[name] + candidate)
            changed[name] = averaged[name] if name in args.mean_component else candidate
        variants = {"baseline": baseline}
        if replacements:
            variants.update(candidate_replace=combine(changed, columns, recipe["plant_weights"]),
                            candidate_mean=combine(averaged, columns, recipe["plant_weights"]))
        if args.identity_check:
            assert all(np.array_equal(value, baseline) for value in variants.values())
        subsets = {"all": np.ones(len(cases), dtype=bool)}
        if year == 2022:
            subsets["jan7_onward"] = (cases.starttime >= pd.Timestamp("2022-01-07", tz="UTC")).to_numpy()
            assert subsets["jan7_onward"].sum() == 359
        for subset, mask in subsets.items():
            reference = metrics(cases.loc[mask, columns].to_numpy(), baseline[mask], generators)
            for variant, values in variants.items():
                result = metrics(cases.loc[mask, columns].to_numpy(), values[mask], generators)
                frame = pd.DataFrame(values[mask], columns=columns, index=cases.index[mask])
                frame.insert(0, "Run No", cases.loc[mask, "Run No"])
                official = score(cases.loc[mask, ["Run No", *columns]], frame)
                assert official == result["micro_roc_auc"]
                rows.append({"mode": mode, "year": year, "subset": subset, "variant": variant,
                             "cases": int(mask.sum()), **result, "official_micro_roc_auc": official,
                             "delta_vs_baseline": result["micro_roc_auc"] - reference["micro_roc_auc"],
                             "generator_delta_vs_baseline": {g: result["generator_auc"][g] - reference["generator_auc"][g] for g in generators}})
        if args.save_prefix:
            for variant, values in variants.items():
                if variant == "baseline":
                    continue
                path = prediction_path(f"{args.save_prefix}_{variant}_{year}")
                frame = pd.DataFrame(values, columns=columns)
                frame.insert(0, "Run No", cases["Run No"])
                frame.to_csv(path, index=False, mode="x")
                saved[str(path.relative_to(ROOT))] = digest(path)
    assert all(digest(ROOT / path) == expected for path, expected in protected.items())
    write_json(output, {
        "baseline_config": RECIPE, "baseline_components": prefixes, "plant_weights": recipe["plant_weights"],
        "replacements": replacements, "identity_check_passed": args.identity_check,
        "candidate_replace_component_modes": {name: "old_new_mean" if name in args.mean_component else "replacement"
                                              for name in replacements},
        "baseline_score_reconstruction": reconstruction,
        "serialization_note": "Frozen round3 scoring originally blended the new XGBoost float32 output before CSV serialization. CSV reconstruction uses its stored decimal values as float64 consistently for all variants; the recorded-vs-reconstructed AUC difference is asserted below 1e-9. Native scores must match exactly.",
        "independent_baseline_reconstruction_exact": True, "official_metric_verified": True,
        "competition_ids_overlap": 0, "source_hashes": protected, "saved_predictions": saved,
        "variant_definitions": {"candidate_replace": ("Use the fixed 50/50 old/new mean for " + ", ".join(args.mean_component) + "; replace every other named component with its new predictions.")
                                                      if args.mean_component else "Replace each named component with its new predictions.",
                                "candidate_mean": "Use a fixed 50/50 old/new mean within each named component."},
        "validation_note": "All plant coefficients remain fixed. Both years are reused development data. Select on 2021 before 2022 confirmation. Frozen inference uses only pre-2021 models and stored scalers. Jan7+ excludes six starts overlapping 2021 selection horizons.",
        "results": rows,
    })
    print(json.dumps([{k: r[k] for k in ("mode", "year", "subset", "variant", "micro_roc_auc", "delta_vs_baseline")} for r in rows], indent=2))


if __name__ == "__main__":
    main()
