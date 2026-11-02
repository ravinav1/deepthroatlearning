"""Training-only scaling for sequence inputs, with opt-in stable accumulation."""

import numpy as np


def fit_sequence_scaler(features, stable=False):
    """Return channel means/stds; default preserves the original float32 math."""
    if stable:
        mean = features.mean(axis=(0, 2), keepdims=True, dtype=np.float64).astype(np.float32)
        std = features.std(axis=(0, 2), keepdims=True, dtype=np.float64).astype(np.float32)
    else:
        mean = features.mean(axis=(0, 2), keepdims=True)
        std = features.std(axis=(0, 2), keepdims=True)
    std[std < 1e-6] = 1
    return mean, std
