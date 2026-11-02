"""One frozen-flow update of Tokke/Vinje shared hydraulic losses.

Keep initial reservoir heads and terminal values fixed. Subtract only the
additional loss omitted by unit-alone PQ curves. This is an approximation at
the preceding LP dispatch, not a coupled nonlinear optimization or derivative.
"""

import numpy as np

from sintef_project.lp_features import UNIT_ORDER, unit_curve
from sintef_project.lp_head_update import hourly_segments, shared_loss_correction


def update_case_shared_losses(data, index: int, case: dict, previous: dict,
                              plants: tuple[str, ...] = ("Tokke", "Vinje")) -> tuple[dict, dict]:
    """Rebuild hourly PQ curves with frozen extra shared loss; no head trajectory."""
    if not set(plants).issubset({"Tokke", "Vinje"}):
        raise ValueError("This isolated ablation supports only Tokke and Vinje")
    g, k = case["energy"].shape
    horizon = len(case["price"])
    result = dict(case)
    for key in ["energy", "widths"]:
        result[key] = hourly_segments(case[key], g, k, horizon)
    for key in ["qmin", "qmax", "pmax"]:
        result[key] = np.repeat(case[key][:, None], horizon, axis=1)
    initial = data.volume.loc[data.cases.iloc[index].starttime]

    def level(name):
        curve = data.model["reservoir"][name]["vol_head"]
        return float(np.interp(initial[name], curve["x"], curve["y"]))

    upstream = {"Tokke": "Vinjevatn", "Vinje": "Vaamarvatn"}
    downstream = {"Tokke": "Bandak", "Vinje": "Vinjevatn"}
    details = {}
    for plant_name in plants:
        plant = data.model["plant"][plant_name]
        gross_head = level(upstream[plant_name]) - max(float(plant["outlet_line"]), level(downstream[plant_name]))
        corrections = shared_loss_correction(data, previous, plant_name)
        for unit, correction in corrections.items():
            unit_index = UNIT_ORDER.index(unit)
            if (correction < -1e-8).any():
                raise ValueError("Additional shared loss must be nonnegative")
            adjusted_head = gross_head - correction
            if (adjusted_head <= 0).any():
                raise ValueError("Frozen shared loss exceeds available gross head")
            # Exact repeated heads occur when prior dispatch loads the same unit
            # combinations; reuse identical curves without rounding the heads.
            unique, inverse = np.unique(adjusted_head, return_inverse=True)
            curves = [unit_curve(data.model["generator"][unit], plant, float(head), segments=k) for head in unique]
            for key in ["energy", "widths"]:
                result[key][unit_index] = np.stack([curve[key] for curve in curves])[inverse].T
            for key in ["qmin", "qmax", "pmax"]:
                result[key][unit_index] = np.array([curve[key] for curve in curves])[inverse]
            details[unit] = {"initial_gross_head": gross_head, "additional_loss": correction,
                             "adjusted_gross_head": adjusted_head, "unique_curves": len(unique)}
    return result, details


def precise_conservation(case: dict, result: dict, network) -> float:
    """Check the exact supplied float values using float64 accumulation throughout."""
    sea = result["flow"][network.unit_down < 0].sum(dtype=np.float64)
    sea += result["bypass"][network.bypass_down < 0].sum(dtype=np.float64)
    sea += result["spill"][network.spill_down < 0].sum(dtype=np.float64)
    return float(result["storage"][:, -1].sum(dtype=np.float64) + sea
                 - np.asarray(case["initial"], dtype=np.float64).sum()
                 - np.asarray(case["natural"], dtype=np.float64).sum())
