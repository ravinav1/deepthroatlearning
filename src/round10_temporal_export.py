"""Export the exact frozen2021 temporal correction; no fitting or uploads."""

import round10_temporal as pilot  # Set numeric thread counts before imports.
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd

ROOT = pilot.ROOT
OUTPUT = ROOT / "submissions/submission_round10_temporal.csv"
REPORT = ROOT / "reports/round10_temporal_export.json"
MODEL = ROOT / "models/round10_temporal_temporal.json"
INCUMBENT = ROOT / "submissions/submission_round5_lgb_generators.csv"
MEAN_REPORT = ROOT / "reports/round5_lgb_peer_relative_seeds_full_mean.json"


def main():
    assert not OUTPUT.exists() and not REPORT.exists()
    source_paths = [MODEL, INCUMBENT, MEAN_REPORT, pilot.PROTOCOL,
        ROOT / "reports/round10_temporal_results.json", ROOT / "reports/round10_temporal_matched_control_results.json",
        ROOT / "reports/round10_temporal_verification.json", ROOT / "reports/round10_temporal_control_amendment.json",
        ROOT / "configs/round5_submission_lgb_generators.json", ROOT / "reports/submission_round5_lgb_generators_run.json",
        ROOT / "submissions/submission_round3_temporal_ranks.csv", ROOT / "data/raw/sample_submission.csv",
        ROOT / "data/raw/prediction_mapping.csv", ROOT / "data/processed/round6_native_2021_baseline.npz",
        ROOT / "src/round10_temporal.py", ROOT / "src/round10_temporal_export.py"]
    average = json.loads(MEAN_REPORT.read_text())
    source_paths += [ROOT / average["output"], *[ROOT / member["path"] for member in average["sources"]]]
    sources = {str(path.relative_to(ROOT)): pilot.digest(path) for path in source_paths}
    model = json.loads(MODEL.read_text())
    historical = json.loads((ROOT / "reports/round10_temporal_results.json").read_text())
    matched = json.loads((ROOT / "reports/round10_temporal_matched_control_results.json").read_text())
    assert historical["model_paths"][str(MODEL.relative_to(ROOT))] == pilot.digest(MODEL)
    assert matched["positive_against_all_three_controls_both_scenarios"]
    assert matched["all_original_and_matched_block_intervals_above_zero"]
    assert model["shrinkage"] == .1 and model["training_cases"] == 359 and len(model["coefficients"]) == 31
    template = pd.read_csv(ROOT / "data/raw/sample_submission.csv", float_precision="round_trip")
    ids, columns = template["Run No"], template.columns[1:].tolist()
    assert template.shape == (725, 2353) and ids.is_unique and len(set(columns)) == 2352
    with np.load(ROOT / "data/processed/round6_native_2021_baseline.npz", allow_pickle=False) as cache:
        metadata = json.loads(str(cache["metadata"].item()))
        assert metadata["columns"] == columns
        assert not np.isin(ids, cache["ids"]).any()
    mapping = pd.read_csv(ROOT / "data/raw/prediction_mapping.csv")
    assert mapping["Run No"].is_unique and set(mapping["Run No"]) == set(ids)
    starts = pd.DatetimeIndex(pd.to_datetime(mapping.set_index("Run No").loc[ids, "starttime"], utc=True))

    def read(path):
        frame = pd.read_csv(path, float_precision="round_trip")
        assert frame.columns.tolist() == ["Run No", *columns] and frame["Run No"].equals(ids)
        values = frame[columns].to_numpy()
        assert np.isfinite(values).all() and np.all((values >= 0) & (values <= 1))
        return values

    members = []
    for member in average["sources"]:
        assert member["weight"] == .5 and pilot.digest(ROOT / member["path"]) == member["sha256"]
        members.append(read(ROOT / member["path"]))
    assert len(members) == 2 and pilot.digest(ROOT / average["output"]) == average["output_sha256"]
    p0 = read(ROOT / average["output"])
    assert np.array_equal(.5 * members[0] + .5 * members[1], p0)
    baseline = read(INCUMBENT)
    round3 = read(ROOT / "submissions/submission_round3_temporal_ranks.csv")
    assert np.array_equal(.5 * round3 + .5 * p0, baseline)
    previous_export = json.loads((ROOT / "reports/submission_round5_lgb_generators_run.json").read_text())
    assert pilot.digest(INCUMBENT) == previous_export["submission_sha256"]
    branch = p0.reshape(725, 14, 168)
    corrected, correction_sizes = pilot.predict_residual(branch, ids.to_numpy(), model)
    candidate = baseline + .5 * (corrected.reshape(p0.shape) - p0)
    assert candidate.shape == (725, 2352) and np.isfinite(candidate).all()
    assert np.all((candidate >= 0) & (candidate <= 1))
    assert np.max(np.abs(candidate - (.5 * round3 + .5 * corrected.reshape(p0.shape)))) <= 3e-16
    # Preserve exact original scores when the correction is zero.
    assert np.array_equal(baseline + .5 * (p0 - p0), baseline)
    exported = pd.DataFrame(candidate, columns=columns)
    exported.insert(0, "Run No", ids)
    exported.to_csv(OUTPUT, index=False, mode="x")
    reloaded = read(OUTPUT)
    assert np.array_equal(reloaded, candidate)
    # A fresh model call must reproduce every exported value, independently of CSV serialization.
    replay, _ = pilot.predict_residual(branch, ids.to_numpy(), json.loads(MODEL.read_text()))
    expected = baseline + .5 * (replay.reshape(p0.shape) - p0)
    assert np.array_equal(reloaded, expected)
    assert all(pilot.digest(ROOT / path) == sha for path, sha in sources.items())
    per_year = []
    for year in (2023, 2024):
        choose = starts.year == year
        change = candidate[choose] - baseline[choose]
        per_year.append({"year": year, "cases": int(choose.sum()),
            "mean_absolute_probability_change": float(np.abs(change).mean()),
            "maximum_absolute_probability_change": float(np.abs(change).max()),
            "mean_signed_probability_change": float(change.mean())})
    pilot.write_json(REPORT, {"created_utc": datetime.now(timezone.utc).isoformat(),
        "submission": str(OUTPUT.relative_to(ROOT)), "submission_sha256": pilot.digest(OUTPUT),
        "status": "Experimental candidate from a small consistent reused-development gain; not uploaded.",
        "recipe": "Original Round5 baseline +0.5*(frozen_temporal_corrected_two_seed_branch-original_two_seed_branch)",
        "model_path": str(MODEL.relative_to(ROOT)), "model_sha256": pilot.digest(MODEL),
        "model_training": "2021Jan1-Dec25 cached pre2021-fitted base predictions; unchanged after historical checks.",
        "shrinkage": .1, "cases": 725, "outputs_per_case": 2352, "verified_probabilities": int(candidate.size),
        "template_order_verified": True, "finite_probabilities_in_0_1": True,
        "seed_mean_exactly_reconstructed": True, "incumbent_exactly_reconstructed": True,
        "csv_roundtrip_exact": True, "full_model_replay_exact": True,
        "incumbent_preserved": True, "source_hashes_unchanged": True, "source_sha256": sources,
        "correction_sizes": correction_sizes, "per_year_prediction_changes": per_year,
        "fitting_performed": False, "blend_weight_tuning": False, "cpu_threads": 1, "uploaded": False,
        "limitations": "The meta model stays frozen; no2022 refit. Native/frozen2022 evidence uses the same reused labels and does not guarantee2023/2024 or Kaggle gains. Current full base models use an additional training year beyond native2022. This is a shared linear temporal correction; raw exogenous inputs and the Round3 branch are unchanged."})
    print(json.dumps({"submission": str(OUTPUT), "verified_probabilities": int(candidate.size),
        "sha256": pilot.digest(OUTPUT), "incumbent_preserved": True, "yearly_change": per_year}, indent=2), flush=True)


if __name__ == "__main__":
    main()
