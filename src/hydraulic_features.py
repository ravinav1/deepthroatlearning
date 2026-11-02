"""Initial water levels and tunnel head differences, without simulating flows."""

import numpy as np


def initial_hydraulics(starts, inputs, topology):
    """Use only supplied start volumes; these levels stay fixed across the week."""
    result = {}
    levels = {}
    for name, reservoir in topology["model"]["reservoir"].items():
        curve = reservoir["vol_head"]
        volume = inputs["volume"].loc[starts, name].to_numpy()
        # Retain curve points above the normal maximum reservoir volume.
        levels[name] = np.interp(volume, curve["x"], curve["y"])
        result[f"initial_level_{name}_m"] = levels[name]

    for name, tunnel in topology["model"]["tunnel"].items():
        source = next(
            link["from"] for link in topology["connections"]
            if link["to"] == name and link["to_type"] == "tunnel"
            and link["from_type"] == "reservoir"
        )
        result[f"initial_tunnel_source_submergence_{name}_m"] = (
            levels[source] - tunnel["start_height"]
        )
        targets = [
            link["to"] for link in topology["connections"]
            if link["from"] == name and link["from_type"] == "tunnel"
            and link["to_type"] == "reservoir"
        ]
        for target in targets:
            # Signed differences allow reverse pressure; they do not imply actual flow.
            result[f"initial_tunnel_level_difference_{name}_m"] = levels[source] - levels[target]
            result[f"initial_tunnel_target_submergence_{name}_m"] = (
                levels[target] - tunnel["end_height"]
            )
    return result
