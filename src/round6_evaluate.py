"""Evaluate fixed Round6 LightGBM changes against the actual Round5 ensemble.

Build reusable arrays:
  --year 2021 --build-cache data/processed/round6_native_2021_baseline.npz --output reports/round6_baseline_2021.json
Screen one seed:
  --baseline-cache data/processed/round6_native_2021_baseline.npz --candidate299 data/processed/NAME_validation_predictions.csv --output reports/round6_NAME_screen.json
Evaluate a locked pair: add --candidate673, optionally --bootstrap 200 and --save-blend.
For frozen 2022, pass both --year 2022 and --frozen when building/using its cache.
No fitting, model-library loading, weight search, or modification of old artifacts.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from features import ROOT
from plant_blend import metrics, read_predictions
from round2_compare import combine
from round2_diagnostics import paired_block_bootstrap
from round5_compare import EXPECTED, FROZEN, NATIVE, RECIPE, official_metric
from round3_frozen_evaluation import digest, prediction_path, write_json


ROUND5_RECIPE = "configs/round5_submission_lgb_generators.json"
EXPECTED_ROUND5 = {("native", 2021): .9903748458953463,
                   ("native", 2022): .9894979113990909,
                   ("frozen", 2022): .9870830517775449}


def remember(path, sources):
    path = Path(path)
    sources[str(path.relative_to(ROOT))] = digest(path)
    return path


def build_baseline(year, mode):
    sources = {}
    recipe = json.loads(remember(ROOT / RECIPE, sources).read_text())
    round5 = json.loads(remember(ROOT / ROUND5_RECIPE, sources).read_text())
    assert all(weights == {"incumbent": .5, "separate_lgb": .5}
               for weights in round5["plant_weights"].values())
    template = remember(ROOT / "data/raw/sample_submission.csv", sources)
    columns = pd.read_csv(template, nrows=0).columns[1:].tolist()
    assert len(columns) == 2352
    truth_path = remember(ROOT / "data/raw/kernel/Unit_commitment_decisions.csv", sources)
    truth = pd.read_csv(truth_path, usecols=["Run No", "starttime", *columns],
                        dtype={column: "uint8" for column in columns})
    truth["starttime"] = pd.to_datetime(truth.starttime, utc=True)
    mapping = remember(ROOT / "data/raw/prediction_mapping.csv", sources)
    test_ids = pd.read_csv(mapping, usecols=["Run No"])["Run No"]
    assert truth["Run No"].is_unique and not truth["Run No"].isin(test_ids).any()
    cases = truth.loc[truth.starttime.dt.year == year].sort_values("starttime").reset_index(drop=True)
    assert len(cases) == 365

    def read(prefix):
        return read_predictions(remember(prediction_path(f"{prefix}_{year}"), sources),
                                columns, cases["Run No"])

    prefixes = FROZEN if mode == "frozen" else NATIVE
    components = {name: read(prefix) for name, prefix in prefixes.items()}
    round3 = combine(components, columns, recipe["plant_weights"])
    labels = cases[columns].to_numpy()
    observed = float(roc_auc_score(labels.ravel(), round3.ravel()))
    assert abs(observed - EXPECTED[mode, year]) < 1e-9
    prefix = "round5_frozen_lgb_peer_relative" if mode == "frozen" else "round5_lgb_peer_relative"
    old299, old673 = read(f"{prefix}_seed299"), read(f"{prefix}_seed673")
    baseline = .5 * round3 + .25 * old299 + .25 * old673
    baseline_auc = float(roc_auc_score(labels.ravel(), baseline.ravel()))
    assert abs(baseline_auc - EXPECTED_ROUND5[mode, year]) < 1e-9
    remember(ROOT / "evaluation/kaggle_metric.py", sources)
    metadata = {"year": year, "mode": mode, "columns": columns,
                "source_sha256": sources, "competition_ids_overlap": 0,
                "round3_micro_roc_auc": observed, "round5_micro_roc_auc": baseline_auc}
    arrays = {"labels": labels, "round3": round3, "old299": old299, "old673": old673,
              "ids": cases["Run No"].to_numpy(), "starttime": cases.starttime.astype(str).to_numpy(dtype=str)}
    return arrays, metadata


def load_cache(path, year, mode):
    with np.load(path, allow_pickle=False) as cache:
        metadata = json.loads(str(cache["metadata"].item()))
        arrays = {name: cache[name] for name in ("labels", "round3", "old299", "old673", "ids", "starttime")}
    assert metadata["year"] == year and metadata["mode"] == mode
    assert arrays["labels"].shape == arrays["round3"].shape == arrays["old299"].shape == arrays["old673"].shape == (365, 2352)
    assert all(digest(ROOT / relative) == expected for relative, expected in metadata["source_sha256"].items())
    return arrays, metadata


def training_summary(path, prediction_path_, sources):
    report = json.loads(remember(ROOT / path, sources).read_text())
    assert report["prediction_sha256"] == digest(prediction_path_)
    fields = ("training_cases", "evaluation_cases", "first_training_start", "last_training_start",
              "training_micro_roc_auc", "sampled_training_micro_roc_auc", "lightgbm_micro_roc_auc",
              "training_auc_scope", "training_diagnostic_scope", "training_validation_auc_gap",
              "sampling_rule", "training_case_ids_sha256")
    result = {key: report[key] for key in fields if key in report}
    result["per_generator"] = {
        generator: {key: values[key] for key in ("training_auc", "sampled_training_auc", "training_log_loss",
                                                "auc", "validation_auc", "validation_log_loss", "trees")
                    if key in values}
        for generator, values in report["generator_models"].items()
    }
    for values in result["per_generator"].values():
        if "training_auc" in values and "validation_auc" in values:
            values["training_validation_auc_gap"] = values["training_auc"] - values["validation_auc"]
    result["note"] = "Training scores are supplied diagnostics, not independent validation; no models are replayed here."
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--year", type=int, choices=[2021, 2022], default=2021)
    parser.add_argument("--frozen", action="store_true")
    parser.add_argument("--build-cache", help="Exclusively write reconstructed baseline arrays to this NPZ path.")
    parser.add_argument("--baseline-cache", help="Reuse a baseline cache and check its source hashes.")
    parser.add_argument("--candidate299")
    parser.add_argument("--candidate673")
    parser.add_argument("--report299")
    parser.add_argument("--report673")
    parser.add_argument("--bootstrap", type=int, choices=[0, 200], default=0)
    parser.add_argument("--save-blend", help="Optional complete wide CSV of the candidate ensemble.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    mode = "frozen" if args.frozen else "native"
    assert not args.frozen or args.year == 2022
    assert not (args.build_cache and args.baseline_cache)
    assert args.candidate299 or (args.build_cache and not args.candidate673)
    assert not args.candidate673 or args.candidate299
    assert not args.bootstrap or args.candidate673, "Bootstrap only the locked paired candidate."
    assert not args.save_blend or args.candidate299
    output = ROOT / args.output
    assert output.stem.startswith(("round6_", "round10_")) and not output.exists()
    for relative in (args.build_cache, args.save_blend):
        if relative:
            assert not (ROOT / relative).exists(), relative
    arrays, metadata = (load_cache(ROOT / args.baseline_cache, args.year, mode) if args.baseline_cache
                        else build_baseline(args.year, mode))
    sources = metadata["source_sha256"].copy()
    if args.baseline_cache:
        remember(ROOT / args.baseline_cache, sources)
    if args.build_cache:
        cache_path = ROOT / args.build_cache
        with cache_path.open("xb") as handle:
            np.savez_compressed(handle, metadata=np.asarray(json.dumps(metadata)), **arrays)
        remember(cache_path, sources)
    columns = metadata["columns"]
    generators = list(dict.fromkeys(column.removeprefix("result_committed_").rsplit("_t", 1)[0]
                                    for column in columns))
    ids = pd.Series(arrays["ids"], name="Run No")
    starts = pd.DatetimeIndex(pd.to_datetime(arrays["starttime"], utc=True))
    labels, round3, old299, old673 = (arrays[name] for name in ("labels", "round3", "old299", "old673"))
    baseline = .5 * round3 + .25 * old299 + .25 * old673
    variants = {"baseline": baseline}
    training = {}
    if args.candidate299:
        new299_path = remember(ROOT / args.candidate299, sources)
        new299 = read_predictions(new299_path, columns, ids)
        variants["baseline_single_seed299"] = .5 * round3 + .5 * old299
        variants["candidate_single_seed299"] = .5 * round3 + .5 * new299
        variants["candidate"] = .5 * round3 + .25 * new299 + .25 * old673
        if args.report299:
            training["seed299"] = training_summary(args.report299, new299_path, sources)
    if args.candidate673:
        new673_path = remember(ROOT / args.candidate673, sources)
        new673 = read_predictions(new673_path, columns, ids)
        variants["candidate"] = .5 * round3 + .25 * new299 + .25 * new673
        if args.report673:
            training["seed673"] = training_summary(args.report673, new673_path, sources)
    score, rows, bootstrap, quarterly = official_metric(), [], [], []
    subsets = {"all": np.ones(len(ids), dtype=bool)}
    if args.year == 2022:
        subsets["jan7_onward"] = starts >= pd.Timestamp("2022-01-07", tz="UTC")
    for subset, mask in subsets.items():
        reference = metrics(labels[mask], baseline[mask], generators)
        single_reference = (metrics(labels[mask], variants["baseline_single_seed299"][mask], generators)
                            if args.candidate299 else None)
        for variant, values in variants.items():
            result = metrics(labels[mask], values[mask], generators)
            solution = pd.DataFrame(labels[mask], columns=columns)
            submission = pd.DataFrame(values[mask], columns=columns)
            solution.insert(0, "Run No", ids.to_numpy()[mask])
            submission.insert(0, "Run No", ids.to_numpy()[mask])
            official = score(solution, submission)
            assert official == result["micro_roc_auc"]
            base = single_reference if "single_seed299" in variant else reference
            rows.append({"mode": mode, "year": args.year, "subset": subset, "variant": variant,
                         "cases": int(mask.sum()), **result, "official_micro_roc_auc": official,
                         "reference_variant": "baseline_single_seed299" if "single_seed299" in variant else "baseline",
                         "delta_vs_baseline": result["micro_roc_auc"] - base["micro_roc_auc"],
                         "generator_delta_vs_baseline": {g: result["generator_auc"][g] - base["generator_auc"][g]
                                                         for g in generators}})
        if args.bootstrap:
            bootstrap.append({"subset": subset, "results": paired_block_bootstrap(
                labels[mask], baseline[mask], variants["candidate"][mask], block_lengths=(14,),
                repetitions=args.bootstrap, seed=4173, starts=starts[mask])})
    if args.candidate299:
        for quarter in range(1, 5):
            mask = starts.quarter == quarter
            before = float(roc_auc_score(labels[mask].ravel(), baseline[mask].ravel()))
            after = float(roc_auc_score(labels[mask].ravel(), variants["candidate"][mask].ravel()))
            quarterly.append({"quarter": quarter, "cases": int(mask.sum()), "baseline_micro_roc_auc": before,
                              "candidate_micro_roc_auc": after, "delta_vs_baseline": after - before})
    saved = {}
    if args.save_blend:
        frame = pd.DataFrame(variants["candidate"], columns=columns)
        frame.insert(0, "Run No", ids)
        frame.to_csv(ROOT / args.save_blend, index=False, mode="x")
        saved[args.save_blend] = digest(ROOT / args.save_blend)
    for relative in ("src/round6_evaluate.py", "src/round5_compare.py", "src/round2_compare.py",
                     "src/plant_blend.py", "src/round2_diagnostics.py"):
        remember(ROOT / relative, sources)
    assert all(digest(ROOT / path) == expected for path, expected in sources.items())
    write_json(output, {
        "created_utc": datetime.now(timezone.utc).isoformat(), "mode": mode, "year": args.year,
        "candidate299": args.candidate299, "candidate673": args.candidate673,
        "baseline_cache": args.baseline_cache or args.build_cache, "baseline_metadata": metadata,
        "candidate_definition": (".5*Round3 + .25*new299 + .25*new673" if args.candidate673
                                 else ".5*Round3 + .25*new299 + .25*old673 (swap299 only)"),
        "baseline_definition": ".5*Round3 + .25*old299 + .25*old673; fixed Round5 incumbent",
        "screen_single_seed_definition": "Compare .5*Round3+.5*new299 against .5*Round3+.5*old299.",
        "results": rows, "quarterly": quarterly, "paired_block_bootstrap": bootstrap,
        "training_vs_validation": training, "source_sha256": sources, "source_hashes_unchanged": True,
        "saved_predictions": saved, "official_metric_verified": True, "competition_ids_overlap": 0,
        "limitations": "Historical years are reused development data. No blend weights or features are selected here. "
                       "Frozen2022 and native2022 share labels. Jan7+ removes six starts overlapping2021 selection horizons. "
                       "When enabled, bootstrap uses200 paired draws of circular14-day blocks of whole cases, conditional on supplied predictions; "
                       "it does not correct model-selection bias. Generator AUCs are diagnostics; selection metric is pooled micro AUC.",
    })
    print(json.dumps([{key: row[key] for key in ("mode", "year", "subset", "variant", "micro_roc_auc", "delta_vs_baseline")}
                      for row in rows], indent=2), flush=True)


if __name__ == "__main__":
    main()
