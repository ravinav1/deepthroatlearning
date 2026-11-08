"""Blend helpers shared by the no-fit levers."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments/hydraulic_surrogate"))
from common import PLANTS, UNIT_PLANT, UNITS, OUT  # noqa: E402
from sintef_project.postprocess import project_vinje  # noqa: E402

PLANT_COLUMNS = {p: [j for j, up in enumerate(UNIT_PLANT) if up == p] for p in PLANTS}


def load_fold(year: int) -> dict:
    z = np.load(OUT / f"components_{year}.npz")
    return {k: z[k] for k in z.files}


def plant_blend(anchor: np.ndarray, raws: dict[str, np.ndarray], weights: dict[str, dict[str, float]]) -> np.ndarray:
    """out = P((1 - sum_k w[p,k]) * anchor + sum_k w[p,k] * raw_k), weights per plant."""
    out = anchor.copy()
    for plant, cols in PLANT_COLUMNS.items():
        w = weights.get(plant, {})
        total = sum(w.values())
        assert 0 <= total <= 1, (plant, w)
        mix = (1 - total) * anchor[:, cols]
        for name, value in w.items():
            if value:
                mix = mix + value * raws[name][:, cols]
        out[:, cols] = mix
    return project_vinje(out)


TOKKE = [UNITS.index(f"Tokke_G{i}") for i in (1, 2, 3, 4)]


def project_tokke(p: np.ndarray, iterations: int = 50) -> np.ndarray:
    """Approximate L2 projection onto G1 >= G3 >= G4 and G3 >= G2 (no G2/G4 order).

    Alternating pool-adjacent-violators on the two chains until no violation.
    Sum of the four probabilities is not exactly preserved; values stay in [0, 1].
    """
    values = np.asarray(p, dtype=np.float64).copy()
    g1, g2, g3, g4 = (values[:, TOKKE[i]].copy() for i in range(4))
    for _ in range(iterations):
        changed = False
        # chain G1 >= G3 >= G4
        for a, b in ((g1, g3), (g3, g4)):
            bad = a < b
            if bad.any():
                m = (a[bad] + b[bad]) / 2; a[bad] = m; b[bad] = m; changed = True
        bad = g3 < g2
        if bad.any():
            m = (g3[bad] + g2[bad]) / 2; g3[bad] = m; g2[bad] = m; changed = True
        if not changed:
            break
    for i, g in enumerate((g1, g2, g3, g4)):
        values[:, TOKKE[i]] = g
    return np.clip(values, 0, 1)
