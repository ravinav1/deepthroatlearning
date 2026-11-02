"""Compare river features against the current XGBoost and blend."""

import json

import pandas as pd
from sklearn.metrics import roc_auc_score

from features import ROOT


def main():
    truth = pd.read_csv(ROOT / "data/raw/kernel/Unit_commitment_decisions.csv").set_index("Run No")
    rows = []
    for year in [2021, 2022]:
        frames = {
            name: pd.read_csv(ROOT / f"data/processed/{stem}_{year}_validation_predictions.csv").set_index("Run No")
            for name, stem in {"price": "xgboost_price", "river": "xgboost_river", "lightgbm": "topology"}.items()
        }
        reference = frames["price"]
        for frame in frames.values():
            assert frame.index.equals(reference.index) and frame.columns.equals(reference.columns)
        labels = truth.loc[reference.index, reference.columns]
        frames["current_blend"] = .75 * frames["price"] + .25 * frames["lightgbm"]
        frames["river_blend"] = .75 * frames["river"] + .25 * frames["lightgbm"]
        generators = list(dict.fromkeys(c.removeprefix("result_committed_").rsplit("_t", 1)[0]
                                       for c in reference.columns))
        for name, frame in frames.items():
            generator_auc = {}
            for generator in generators:
                columns = [c for c in reference.columns if c.startswith(f"result_committed_{generator}_t")]
                generator_auc[generator] = float(roc_auc_score(
                    labels[columns].to_numpy().ravel(), frame[columns].to_numpy().ravel()))
            rows.append({"year": year, "model": name,
                         "micro_roc_auc": float(roc_auc_score(labels.to_numpy().ravel(), frame.to_numpy().ravel())),
                         "generator_auc": generator_auc})
    (ROOT / "reports/river_feature_comparison.json").write_text(json.dumps(rows, indent=2) + "\n")
    table = pd.DataFrame([{"year": r["year"], "model": r["model"], "auc": r["micro_roc_auc"],
                           "Lio": r["generator_auc"]["Lio_G1"], "Haukeli": r["generator_auc"]["Haukeli_G1"]}
                          for r in rows])
    print(table.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
