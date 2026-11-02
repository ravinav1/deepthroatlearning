"""One predeclared pass over plant blend weights using saved predictions."""

from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from features import ROOT
from plant_blend import column_groups, metrics
from round3_frozen_evaluation import digest, write_json
from round5_compare import official_metric
from round6_evaluate import load_cache


PREFIX = "round9_plant_blend"


def now():
    return datetime.now(timezone.utc).isoformat()


def auc(labels, predictions):
    return float(roc_auc_score(labels.ravel(), predictions.ravel()))


def blend(arrays, groups, weights):
    result = arrays["round3"].copy()
    for plant, indices in groups.items():
        weight = weights[plant]
        result[:, indices] = ((1 - weight) * arrays["round3"][:, indices]
                              + .5 * weight * arrays["old299"][:, indices]
                              + .5 * weight * arrays["old673"][:, indices])
    return result


def evaluate(arrays, metadata, groups, weights):
    labels = arrays["labels"]
    baseline = blend(arrays, groups, dict.fromkeys(groups, .5))
    candidate = blend(arrays, groups, weights)
    columns = metadata["columns"]
    generators = list(dict.fromkeys(column.removeprefix("result_committed_").rsplit("_t", 1)[0]
                                    for column in columns))
    starts = pd.DatetimeIndex(pd.to_datetime(arrays["starttime"], utc=True))
    masks = {"all": np.ones(len(labels), dtype=bool)}
    if metadata["year"] == 2022:
        masks["jan7_onward"] = starts >= pd.Timestamp("2022-01-07", tz="UTC")
    rows = []
    score = official_metric()
    for subset, mask in masks.items():
        before = metrics(labels[mask], baseline[mask], generators)
        after = metrics(labels[mask], candidate[mask], generators)
        truth = pd.DataFrame(labels[mask], columns=columns)
        pred = pd.DataFrame(candidate[mask], columns=columns)
        truth.insert(0, "Run No", arrays["ids"][mask])
        pred.insert(0, "Run No", arrays["ids"][mask])
        official = score(truth, pred)
        assert official == after["micro_roc_auc"]
        rows.append({"year": metadata["year"], "mode": metadata["mode"], "subset": subset,
                     "cases": int(mask.sum()), "baseline": before, "candidate": after,
                     "delta": after["micro_roc_auc"] - before["micro_roc_auc"],
                     "official_micro_roc_auc": official,
                     "generator_auc_deltas": {g: after["generator_auc"][g] - before["generator_auc"][g]
                                              for g in generators}})
    quarters = []
    for quarter in range(1, 5):
        mask = starts.quarter == quarter
        before, after = auc(labels[mask], baseline[mask]), auc(labels[mask], candidate[mask])
        quarters.append({"quarter": quarter, "baseline": before, "candidate": after, "delta": after-before})
    return {"results": rows, "quarters": quarters}


def main():
    protocol_path = ROOT / f"reports/{PREFIX}_protocol.json"
    selection_path = ROOT / f"reports/{PREFIX}_selection.json"
    report_path = ROOT / f"reports/{PREFIX}_results.json"
    assert not any(path.exists() for path in (protocol_path, selection_path, report_path))
    protocol = {
        "created_utc": now(),
        "baseline": ".5 Round3 + .25 original Round5 seed299 + .25 original Round5 seed673",
        "candidate": "Per plant: (1-w)*Round3 + w*mean(original Round5 seeds299,673)",
        "selection": "2021 pooled micro AUC, one coordinate pass, plant order is first appearance in template columns",
        "initial_weight": .5, "weight_candidates": [.25, .5, .75],
        "minimum_coordinate_gain": .00002, "minimum_final_2021_gain": .00005,
        "minimum_native_2022_gain_all_and_jan7": .00005,
        "minimum_frozen_2022_gain_all_and_jan7": 0,
        "no_retries": True, "no_model_fitting": True,
        "confirmation": "Lock weights on disk before loading either 2022 candidate evaluation",
        "source_code_sha256": digest(ROOT / "src/round9_plant_blend.py"),
    }
    write_json(protocol_path, protocol)
    sources = {str(protocol_path.relative_to(ROOT)): digest(protocol_path)}

    def read(year, mode):
        path = ROOT / f"data/processed/round6_{mode}_{year}_baseline.npz"
        arrays, metadata = load_cache(path, year, mode)
        sources[str(path.relative_to(ROOT))] = digest(path)
        sources.update(metadata["source_sha256"])
        return arrays, metadata

    arrays, metadata = read(2021, "native")
    columns = metadata["columns"]
    groups = column_groups(metadata["columns"])
    weights = dict.fromkeys(groups, .5)
    baseline_auc = auc(arrays["labels"], blend(arrays, groups, weights))
    current_auc = baseline_auc
    trials = []
    for plant in groups:
        plant_trials = []
        for weight in protocol["weight_candidates"]:
            trial_weights = weights.copy()
            trial_weights[plant] = weight
            value = auc(arrays["labels"], blend(arrays, groups, trial_weights))
            plant_trials.append({"plant": plant, "weight": weight, "auc": value,
                                 "delta_vs_current": value-current_auc})
        best = max(plant_trials, key=lambda trial: trial["auc"])
        accepted = best["auc"] - current_auc >= protocol["minimum_coordinate_gain"]
        if accepted:
            weights[plant], current_auc = best["weight"], best["auc"]
        trials.append({"plant": plant, "trials": plant_trials, "accepted": accepted,
                       "selected_weight": weights[plant], "current_auc": current_auc})
    screen_passed = current_auc - baseline_auc >= protocol["minimum_final_2021_gain"]
    selection = {"locked_utc": now(), "protocol_sha256": digest(protocol_path),
                 "plant_order": list(groups), "weights": weights,
                 "baseline_auc": baseline_auc, "selected_auc": current_auc,
                 "delta": current_auc-baseline_auc, "screen_passed": screen_passed,
                 "trials": trials, "source_sha256": sources.copy()}
    write_json(selection_path, selection)
    sources[str(selection_path.relative_to(ROOT))] = digest(selection_path)
    results = [evaluate(arrays, metadata, groups, weights)]
    passed = screen_passed
    if screen_passed:
        for mode in ("native", "frozen"):
            arrays, metadata = read(2022, mode)
            assert metadata["columns"] == columns
            result = evaluate(arrays, metadata, groups, weights)
            results.append(result)
            minimum = protocol[f"minimum_{mode}_2022_gain_all_and_jan7"]
            passed = passed and all(row["delta"] >= minimum for row in result["results"])
    assert all(digest(ROOT / path) == value for path, value in sources.items())
    write_json(report_path, {"finished_utc": now(), "protocol": protocol,
                            "selected_weights": weights, "screen_passed": screen_passed,
                            "confirmation_evaluated": screen_passed, "all_gates_passed": passed,
                            "evaluations": results, "source_sha256": sources,
                            "source_hashes_unchanged": True, "official_metric_verified": True,
                            "competition_label_cases": 0,
                            "limitations": "Historical years are reused development data; native and frozen2022 share labels. "
                                           "Per-generator AUCs and quarters are diagnostics, not additive microAUC contributions."})
    print(json.dumps({"weights": weights, "all_gates_passed": passed,
                      "results": [{k: row[k] for k in ("year", "mode", "subset", "delta")}
                                  for result in results for row in result["results"]]}, indent=2))


if __name__ == "__main__":
    main()
