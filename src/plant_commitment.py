"""Turn plant count and unit-combination probabilities into ON probabilities."""

import numpy as np

from features import HOURS


def plant_labels(cases, generators):
    columns = [f"result_committed_{generator}_t{hour}"
               for generator in generators for hour in range(HOURS)]
    return cases[columns].to_numpy().reshape(-1, len(generators), HOURS).transpose(0, 2, 1).reshape(-1, len(generators))


def unit_probabilities(count_probabilities, pattern_probabilities, patterns, units):
    # A pattern is a bit mask: e.g. 5 = G1 and G3 ON.
    bits = ((np.asarray(patterns)[:, None] >> np.arange(units)) & 1)
    counts = bits.sum(axis=1)
    joint = np.zeros_like(pattern_probabilities, dtype="float64")
    for count in range(units + 1):
        selected = counts == count
        # Normalize the pattern model within each count, then weight by P(count).
        conditional = pattern_probabilities[:, selected].astype("float64")
        conditional /= conditional.sum(axis=1, keepdims=True)
        joint[:, selected] = count_probabilities[:, count, None] * conditional
    return joint @ bits
