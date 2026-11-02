"""Compare optional feature groups on saved validation predictions."""

import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from features import ROOT, HOURS


def main():
    truth = pd.read_csv(ROOT / "data/raw/kernel/Unit_commitment_decisions.csv").set_index("Run No")
    rows = []
    for year in [2021, 2022]:
        names = {
            "xgboost_price": f"xgboost_price_{year}",
            "xgboost_water": f"xgboost_water_{year}",
            "xgboost_economics": f"xgboost_economics_{year}",
            "lightgbm_original": f"topology_{year}",
            "lightgbm_price": f"lightgbm_price_{year}",
        }
        frames = {key: pd.read_csv(ROOT / f"data/processed/{name}_validation_predictions.csv").set_index("Run No")
                  for key, name in names.items()}
        reference = frames["xgboost_price"]
        columns = reference.columns.tolist()
        generators = list(dict.fromkeys(c.removeprefix("result_committed_").rsplit("_t", 1)[0] for c in columns))
        for frame in frames.values():
            assert frame.index.equals(reference.index) and frame.columns.tolist() == columns
        labels = truth.loc[reference.index, columns].to_numpy().reshape(-1, len(generators), HOURS)
        switch = np.zeros_like(labels, dtype=bool)
        switch[:, :, 1:] = labels[:, :, 1:] != labels[:, :, :-1]
        near = np.zeros_like(switch)
        for delta in range(-2, 3):
            if delta < 0:
                near[:, :, :delta] |= switch[:, :, -delta:]
            elif delta > 0:
                near[:, :, delta:] |= switch[:, :, :-delta]
            else:
                near |= switch
        frames["current_blend"] = .75 * reference + .25 * frames["lightgbm_original"]
        frames["both_price_blend"] = .75 * reference + .25 * frames["lightgbm_price"]
        for name, frame in frames.items():
            p = frame.to_numpy().reshape(labels.shape)
            rows.append({"year": year, "model": name,
                         "micro_roc_auc": float(roc_auc_score(labels.ravel(), p.ravel())),
                         "near_switch_auc": float(roc_auc_score(labels[near], p[near])),
                         "near_switch_error_at_0_5": float(np.mean((p[near] >= .5) != labels[near])),
                         "generator_auc": {g: float(roc_auc_score(labels[:, i].ravel(), p[:, i].ravel()))
                                           for i, g in enumerate(generators)}})
    (ROOT / "reports/operating_feature_comparison.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(pd.DataFrame(rows).pivot(index="model", columns="year", values="micro_roc_auc").round(6).to_string())


if __name__ == "__main__":
    main()
