"""Compare three Round5 blend weights using saved predictions only.

Run once with .venv/bin/python src/check_round7_blend_weights.py.
Select on 2021 before opening either 2022 cache. Preserve existing artifacts.
"""

import json
import os
from datetime import datetime, timezone

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from features import ROOT
from round6_evaluate import EXPECTED_ROUND5, load_cache
from round5_compare import official_metric
from round3_frozen_evaluation import digest, write_json


STARTED = "2026-10-17T18:09:44+00:00"
DEADLINE = "2026-10-17T18:24:44+00:00"
SELECTION = ROOT / "reports/round7_blend_weight_selection_20261017.json"
REPORT = ROOT / "reports/round7_blend_weights_20261017.json"
WEIGHTS = (0.25, 0.5, 0.75)


def check_time():
    if datetime.now(timezone.utc) >= datetime.fromisoformat(DEADLINE):
        raise TimeoutError("The 15-minute blend check has ended.")


def probabilities(arrays, weight):
    return ((1 - weight) * arrays["round3"]
            + (weight / 2) * arrays["old299"]
            + (weight / 2) * arrays["old673"])


def main():
    assert not SELECTION.exists() and not REPORT.exists()
    source_hashes = {}
    score = official_metric()

    def read(year, mode):
        check_time()
        path = ROOT / f"data/processed/round6_{mode}_{year}_baseline.npz"
        arrays, metadata = load_cache(path, year, mode)
        source_hashes.update(metadata["source_sha256"])
        source_hashes[str(path.relative_to(ROOT))] = digest(path)
        ids = arrays["ids"]
        assert len(np.unique(ids)) == len(ids)
        assert metadata["competition_ids_overlap"] == 0
        assert np.isin(arrays["labels"], (0, 1)).all()
        return arrays, metadata

    def auc(arrays, metadata, weight, mask):
        values = probabilities(arrays, weight)[mask]
        labels = arrays["labels"][mask]
        assert np.isfinite(values).all() and ((0 <= values) & (values <= 1)).all()
        observed = float(roc_auc_score(labels.ravel(), values.ravel()))
        solution = pd.DataFrame(labels, columns=metadata["columns"])
        submission = pd.DataFrame(values, columns=metadata["columns"])
        for frame in (solution, submission):
            frame.insert(0, "Run No", arrays["ids"][mask])
        assert score(solution, submission) == observed
        return observed

    arrays, metadata = read(2021, "native")
    all_cases = np.ones(len(arrays["ids"]), dtype=bool)
    trials = [{"lightgbm_weight": weight, "micro_roc_auc": auc(arrays, metadata, weight, all_cases)}
              for weight in WEIGHTS]
    reference = next(row["micro_roc_auc"] for row in trials if row["lightgbm_weight"] == .5)
    assert abs(reference - EXPECTED_ROUND5["native", 2021]) < 1e-12
    for row in trials:
        row["delta_vs_round5"] = row["micro_roc_auc"] - reference
    selected = max(trials, key=lambda row: (row["micro_roc_auc"], -abs(row["lightgbm_weight"] - .5)))
    weight = selected["lightgbm_weight"]
    source_hashes["src/check_round7_blend_weights.py"] = digest(ROOT / "src/check_round7_blend_weights.py")
    selection = {
        "locked_utc": datetime.now(timezone.utc).isoformat(),
        "started_utc": STARTED, "deadline_utc": DEADLINE,
        "definition": "(1-w)*Round3 + (w/2)*original_Round5_seed299 + (w/2)*original_Round5_seed673",
        "selection_year": 2021, "candidate_weights": list(WEIGHTS),
        "trials": trials, "selected_lightgbm_weight": weight,
        "selection_rule": "Highest pooled 2021 micro ROC-AUC; exact ties prefer the current 0.5 weight.",
        "confirmation_gate": {"native_2022_min_gain": .0002, "frozen_2022_min_gain": 0,
                              "required_subsets": ["all", "jan7_onward"]},
        "stop_rule": "Keep Round5 if 0.5 wins or any confirmation gate fails; no fallback or extra weights.",
        "2022_candidate_evaluated_before_lock": False,
        "source_sha256": dict(source_hashes),
    }
    write_json(SELECTION, selection)
    source_hashes[str(SELECTION.relative_to(ROOT))] = digest(SELECTION)
    print(json.dumps({"selection": trials, "locked_weight": weight}), flush=True)

    confirmations = []
    if weight != .5:
        for mode in ("native", "frozen"):
            arrays, metadata = read(2022, mode)
            starts = pd.DatetimeIndex(pd.to_datetime(arrays["starttime"], utc=True))
            subsets = {"all": np.ones(len(starts), dtype=bool),
                       "jan7_onward": starts >= pd.Timestamp("2022-01-07", tz="UTC")}
            for subset, mask in subsets.items():
                check_time()
                before = auc(arrays, metadata, .5, mask)
                after = auc(arrays, metadata, weight, mask)
                if subset == "all":
                    assert abs(before - EXPECTED_ROUND5[mode, 2022]) < 1e-12
                threshold = .0002 if mode == "native" else 0
                row = {"mode": mode, "year": 2022, "subset": subset, "cases": int(mask.sum()),
                       "baseline_micro_roc_auc": before, "candidate_micro_roc_auc": after,
                       "delta_vs_round5": after - before, "required_gain": threshold,
                       "passes_gate": after - before >= threshold}
                confirmations.append(row)
                print(json.dumps(row), flush=True)

    passed = weight != .5 and all(row["passes_gate"] for row in confirmations)
    check_time()
    assert all(digest(ROOT / path) == expected for path, expected in source_hashes.items())
    finished = datetime.now(timezone.utc)
    report = {
        "started_utc": STARTED, "finished_utc": finished.isoformat(),
        "elapsed_minutes": (finished - datetime.fromisoformat(STARTED)).total_seconds() / 60,
        "selection_file": str(SELECTION.relative_to(ROOT)),
        "selected_lightgbm_weight": weight, "trials_2021": trials,
        "confirmation": confirmations, "passes_all_gates": passed,
        "decision": "Eligible for one submission export" if passed else "Stop; keep Round5 (user-reported public AUC 0.99172)",
        "base_models_trained": 0, "submissions_created": 0,
        "competition_label_access": False, "official_metric_verified": True,
        "source_sha256": source_hashes, "source_hashes_unchanged": True,
        "limitations": "2021 and 2022 are reused development years. Native and frozen 2022 share labels. "
                       "The gain threshold is a practical stopping rule, not a significance test. "
                       "Only the locked 2021 choice is evaluated on 2022; this is a global weight test, not plant-specific tuning.",
    }
    write_json(REPORT, report)
    print(json.dumps({"passes_all_gates": passed, "decision": report["decision"],
                      "elapsed_minutes": report["elapsed_minutes"]}), flush=True)


if __name__ == "__main__":
    main()
