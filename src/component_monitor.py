"""Prepare a frozen validation ensemble with exactly one replaceable component.

Only cached probabilities are read. This module does not load model libraries or
fit anything; its output is used only to select sequence checkpoints.
"""

import numpy as np
import pandas as pd


def frozen_component_monitor(options, validation, columns, root):
    """Return ``constant + replacement_weight * new_component_predictions``.

    ``prediction_paths`` names the historical component caches and
    ``plant_weights`` holds the complete fixed recipe. The replaced component's
    historical cache can be supplied for provenance, but is never read here.
    """
    assert options["mode"] == "replace_component"
    paths = options["prediction_paths"]
    plant_weights = options["plant_weights"]
    replacement = options["replace_component"]
    assert validation["Run No"].is_unique and not validation["Run No"].isna().any(), \
        "Monitor validation case identifiers must be unique and nonmissing."
    assert len(columns) == len(set(columns)), "Monitor output columns must be unique."
    groups = {}
    for index, column in enumerate(columns):
        plant = column.removeprefix("result_committed_").rsplit("_G", 1)[0]
        groups.setdefault(plant, []).append(index)
    assert plant_weights.keys() == groups.keys(), "Monitor recipe must cover precisely the output plants."
    referenced = {name for amounts in plant_weights.values() for name in amounts}
    assert replacement in referenced, "Replacement component must occur in the fixed recipe."
    assert referenced - {replacement} <= paths.keys(), "Missing frozen component prediction path."
    for amounts in plant_weights.values():
        weights = np.asarray(list(amounts.values()), dtype=float)
        assert np.isfinite(weights).all() and ((weights >= 0) & (weights <= 1)).all(), \
            "Monitor component weights must be finite probabilities."
        assert abs(weights.sum() - 1) < 1e-12, "Monitor plant weights must sum to one."

    components = {}
    for name in sorted(referenced - {replacement}):
        frame = pd.read_csv(root / paths[name], float_precision="round_trip")
        assert frame.columns.tolist() == ["Run No", *columns], "Monitor columns must match model outputs."
        assert frame["Run No"].is_unique and set(frame["Run No"]) == set(validation["Run No"]), \
            "Monitor predictions must cover precisely these validation cases."
        values = frame.set_index("Run No").loc[validation["Run No"], columns].to_numpy()
        assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all(), \
            "Monitor cached predictions must be finite probabilities."
        components[name] = values

    constant = np.zeros((len(validation), len(columns)), dtype=np.float64)
    replacement_weights = np.zeros(len(columns), dtype=np.float64)
    for plant, amounts in plant_weights.items():
        indices = groups[plant]
        replacement_weights[indices] = amounts.get(replacement, 0.0)
        for name, weight in amounts.items():
            if name != replacement:
                constant[:, indices] += weight * components[name][:, indices]
    return constant, replacement_weights
