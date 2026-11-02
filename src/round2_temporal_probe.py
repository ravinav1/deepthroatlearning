"""Small cached-prediction experiment: does weak temporal smoothing help?"""

import json

import numpy as np

from features import HOURS, ROOT
from plant_blend import metrics
from round2_compare import combine, load_year


def smooth(probabilities, strength):
    values = probabilities.reshape(len(probabilities), -1, HOURS)
    padded = np.pad(values, ((0, 0), (0, 0), (1, 1)), mode="edge")
    neighbours = .5 * (padded[..., :-2] + padded[..., 2:])
    return ((1 - strength) * values + strength * neighbours).reshape(probabilities.shape)


def main():
    output = ROOT / "reports/round2_temporal_probe.json"
    assert not output.exists()
    rows = []
    selected = 0.0
    for year in [2021, 2022]:
        config, columns, cases, components = load_year(year)
        reference = combine(components, columns, config["plant_weights"])
        labels = cases[columns].to_numpy()
        generators = list(dict.fromkeys(c.removeprefix("result_committed_").rsplit("_t", 1)[0]
                                        for c in columns))
        # Choose on2021 only; read2022 after fixing the strength, including no change.
        strengths = [0.0, 0.1, 0.25] if year == 2021 else list(dict.fromkeys([0.0, selected]))
        for strength in strengths:
            values = smooth(reference, strength)
            assert values.shape == reference.shape and np.isfinite(values).all()
            result = metrics(labels, values, generators)
            rows.append({"year": year, "strength": strength, **result})
        if year == 2021:
            selected = max(rows, key=lambda row: row["micro_roc_auc"])["strength"]
    report = {
        "hypothesis": "Weak adjacent-hour persistence can improve noisy commitment probabilities.",
        "method": "(1-a)*p[t]+a*(p[t-1]+p[t+1])/2; edge replication within each case/generator only.",
        "selection_year": 2021, "confirmation_year": 2022,
        "candidate_strengths": [0.0, 0.1, 0.25], "selected_strength": selected,
        "limitations": "Reused development years. This probes smoothing, not the potential of all structured models. Real one-hour ON/OFF runs remain allowed.",
        "results": rows,
    }
    with output.open("x") as file:
        json.dump(report, file, indent=2)
        file.write("\n")
    for row in rows:
        print(row["year"], row["strength"], row["micro_roc_auc"])


if __name__ == "__main__":
    main()
