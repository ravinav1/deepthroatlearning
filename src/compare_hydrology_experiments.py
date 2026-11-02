"""Compare separate hydrology feature groups against the price baseline."""

import json

import pandas as pd
from sklearn.metrics import roc_auc_score

from features import ROOT


def main():
    truth = pd.read_csv(ROOT / "data/raw/kernel/Unit_commitment_decisions.csv").set_index("Run No")
    rows = []
    for year in [2020, 2021, 2022]:
        names = ["price", "hydraulic"] if year == 2020 else ["price", "profiles", "hydraulic", "budget"]
        frames = {
            name: pd.read_csv(ROOT / f"data/processed/xgboost_{name}_{year}_validation_predictions.csv").set_index("Run No")
            for name in names
        }
        lightgbm = None
        plant = None
        if year != 2020:
            lightgbm = pd.read_csv(ROOT / f"data/processed/topology_{year}_validation_predictions.csv").set_index("Run No")
            plant = pd.read_csv(ROOT / f"data/processed/plant_commitment_{year}_validation_predictions.csv").set_index("Run No")
        reference = frames["price"]
        for frame in frames.values():
            assert frame.index.equals(reference.index) and frame.columns.equals(reference.columns)
        if lightgbm is not None:
            assert lightgbm.index.equals(reference.index) and lightgbm.columns.equals(reference.columns)
            assert plant.index.equals(reference.index) and plant.columns.equals(reference.columns)
        labels = truth.loc[reference.index, reference.columns]
        generators = list(dict.fromkeys(c.removeprefix("result_committed_").rsplit("_t", 1)[0]
                                       for c in reference.columns))
        for name, frame in frames.items():
            variants = [(name, frame)]
            if lightgbm is not None:
                blend = .75 * frame + .25 * lightgbm
                variants.append((f"{name}_blend", blend))
                if name in ["price", "hydraulic"]:
                    variants.append((f"{name}_plant_blend", .5 * blend + .5 * plant))
            for variant, predictions in variants:
                by_generator = {}
                for generator in generators:
                    columns = [c for c in reference.columns if c.startswith(f"result_committed_{generator}_t")]
                    by_generator[generator] = float(roc_auc_score(labels[columns].to_numpy().ravel(),
                                                                 predictions[columns].to_numpy().ravel()))
                rows.append({"year": year, "model": variant,
                             "micro_roc_auc": float(roc_auc_score(labels.to_numpy().ravel(), predictions.to_numpy().ravel())),
                             "generator_auc": by_generator})
    (ROOT / "reports/hydrology_feature_comparison.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(pd.DataFrame(rows).pivot(index="model", columns="year", values="micro_roc_auc").round(6).to_string())


if __name__ == "__main__":
    main()
