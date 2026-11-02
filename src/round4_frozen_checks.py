"""Shared artifact/cohort checks for the two separate frozen inference CLIs."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from features import HOURS, ROOT, load_data
from plant_blend import metrics, read_predictions
from round3_frozen_evaluation import digest, prediction_path, write_json


def prepare(kind, identity_source, identity_reference):
    parser = argparse.ArgumentParser(description=f"Replay a saved pre-2021 {kind} on 2022; no fitting.")
    parser.add_argument("--source", help="Saved config/report/model stem including _2021.")
    parser.add_argument("--output-prefix", help="Unique round4 output stem including _2022.")
    parser.add_argument("--identity-check", action="store_true", help="Replay the original component against its frozen cache.")
    if kind == "lightgbm":
        parser.add_argument("--num-iteration", type=int, help="Use a fixed tree prefix after replaying the full saved model.")
    args = parser.parse_args()
    source = args.source or (identity_source if args.identity_check else None)
    assert source and Path(source).name == source and source.endswith("_2021")
    assert source == identity_source if args.identity_check else source.startswith("round4_")
    num_iteration = getattr(args, "num_iteration", None)
    assert num_iteration is None or (num_iteration > 0 and not args.identity_check)
    prefix = args.output_prefix or (f"round4_frozen_{kind}_identity_2022" if args.identity_check
                                   else f"round4_frozen_{source.removeprefix('round4_').removesuffix('_2021')}_2022")
    assert Path(prefix).name == prefix and prefix.startswith("round4_") and prefix.endswith("_2022")
    output, report_path = prediction_path(prefix), ROOT / f"reports/{prefix}_metrics.json"
    assert not output.exists() and not report_path.exists(), "Choose an unused output prefix."
    source_report_path, config_path = ROOT / f"reports/{source}_metrics.json", ROOT / f"configs/{source}.json"
    source_report = json.loads(source_report_path.read_text())
    config = source_report["config"]
    assert json.loads(config_path.read_text()) == config
    assert config["validation_start"] == "2021-01-01" and config["validation_end"] == "2022-01-01"
    model_path = ROOT / (f"models/lightgbm_{source}.txt" if kind == "lightgbm" else f"models/{source}.pt")
    replay_path = prediction_path(source)
    protected = {str(p.relative_to(ROOT)): digest(p) for p in (model_path, config_path, source_report_path, replay_path)}
    cases, columns, generators, inputs, topology = load_data(use_extended=config.get("use_extended", False))
    boundary = pd.Timestamp("2021-01-01", tz="UTC")
    train = cases.loc[cases.starttime + pd.Timedelta(hours=HOURS) <= boundary]
    selection = cases.loc[cases.starttime.dt.year == 2021].reset_index(drop=True)
    evaluation = cases.loc[cases.starttime.dt.year == 2022].reset_index(drop=True)
    assert len(train) == source_report["training_cases"] == 2186
    assert str(train.starttime.max()) == source_report["last_training_start"]
    assert (train.starttime + pd.Timedelta(hours=HOURS)).max() <= boundary
    assert len(selection) == source_report["validation_cases"] == len(evaluation) == 365
    assert not train["Run No"].isin(evaluation["Run No"]).any()
    competition_ids = pd.read_csv(ROOT / "data/raw/prediction_mapping.csv", usecols=["Run No"])["Run No"]
    assert not cases["Run No"].isin(competition_ids).any()
    cohort_hash = hashlib.sha256(train[["Run No", "starttime"]].to_csv(index=False).encode()).hexdigest()
    if source_report.get("training_case_ids_sha256"):
        assert cohort_hash == source_report["training_case_ids_sha256"]
    evidence = {"source": source, "source_config": config, "source_hashes": protected,
                "training_cases": len(train), "last_training_start": str(train.starttime.max()),
                "last_training_horizon_end": str((train.starttime + pd.Timedelta(hours=HOURS)).max()),
                "training_cohort_sha256_reconstructed": cohort_hash,
                "competition_ids_overlap": 0, "new_fitting": False, "scaler_refitted": False,
                "cpu_threads": 1, "evaluation_year": 2022, "evaluation_cases": len(evaluation)}
    if kind == "lightgbm":
        evidence["requested_num_iteration"] = num_iteration
    context = (selection, evaluation, columns, generators, inputs, topology)

    def finish(predict_cases, tolerance, predict_replay=None):
        replay_cases = selection.iloc[[0, 60, 120, 180, 240, 300, 364]].reset_index(drop=True)
        cached = read_predictions(replay_path, columns, selection["Run No"])
        replay = (predict_replay or predict_cases)(replay_cases)
        assert replay.columns.tolist() == ["Run No", *columns]
        assert replay["Run No"].equals(replay_cases["Run No"])
        difference = float(np.abs(replay[columns].to_numpy() - cached[[0, 60, 120, 180, 240, 300, 364]]).max())
        assert difference < tolerance, difference
        evidence.update({"source_replay_cases": replay_cases["Run No"].tolist(),
                         "source_replay_max_absolute_difference": difference})
        print(f"{source}: source-year replay passed ({difference:.3g}); frozen 2022 inference", flush=True)
        started = time.monotonic()
        predictions = predict_cases(evaluation)
        evidence["inference_seconds"] = time.monotonic() - started
        assert predictions.columns.tolist() == ["Run No", *columns]
        assert predictions["Run No"].equals(evaluation["Run No"])
        values = predictions[columns].to_numpy()
        assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
        if args.identity_check:
            identity_path = prediction_path(identity_reference)
            protected[str(identity_path.relative_to(ROOT))] = digest(identity_path)
            reference = read_predictions(identity_path, columns, evaluation["Run No"])
            frozen_difference = float(np.abs(values - reference).max())
            assert frozen_difference < tolerance, frozen_difference
            evidence.update({"identity_check_passed": True, "frozen_reference_max_absolute_difference": frozen_difference})
        rows = []
        for subset, mask in (("all", np.ones(len(evaluation), dtype=bool)),
                             ("jan7_onward", (evaluation.starttime >= pd.Timestamp("2022-01-07", tz="UTC")).to_numpy())):
            rows.append({"subset": subset, "cases": int(mask.sum()), **metrics(evaluation.loc[mask, columns].to_numpy(), values[mask], generators)})
        evidence["results"] = rows
        assert all(digest(ROOT / path) == expected for path, expected in protected.items())
        if not args.identity_check:
            predictions.to_csv(output, index=False, mode="x")
            evidence.update({"prediction_path": str(output.relative_to(ROOT)), "prediction_sha256": digest(output)})
        evidence["validation_note"] = "Pre-2021 model and stored scaler remain frozen. Both years are reused development data. Jan7+ excludes six starts overlapping 2021 checkpoint selection horizons."
        write_json(report_path, evidence)
        print(json.dumps({"report": str(report_path.relative_to(ROOT)), "identity_check": args.identity_check,
                          "results": [{k: row[k] for k in ("subset", "cases", "micro_roc_auc")} for row in rows]}, indent=2))

    return source_report, config, model_path, context, evidence, finish
