"""Leakage-safe preprocessing and portable forward inference for CPU MLPs."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.special import expit


def build_without_later_targets(root: Path, builder: Callable, cutoff: str = '2022-01-01') -> tuple:
    """Give an input-only builder metadata plus target rows strictly before cutoff.

    Later target fields are zero sentinels and must be discarded by the caller.
    This interception is local; no raw file or frozen feature source is changed.
    """
    target = (root / 'sintef_project/data/kernel/Unit_commitment_decisions.csv').resolve()
    original_read = pd.read_csv

    def read_csv(path: object, *args: object, **kwargs: object) -> pd.DataFrame:
        if isinstance(path, (str, Path)) and Path(path).resolve() == target:
            if args or kwargs:
                raise ValueError('Frozen feature target read unexpectedly changed')
            metadata = original_read(target, usecols=['Run No', 'starttime'])
            keep = pd.to_datetime(metadata.starttime, utc=True) < pd.Timestamp(cutoff, tz='UTC')
            count = int(keep.sum())
            if count == 0 or not np.array_equal(keep.to_numpy(), np.arange(len(metadata)) < count):
                raise ValueError('Expected a nonempty chronological target prefix')
            prior = original_read(target, nrows=count)
            if prior.columns[:3].tolist() != ['Run No', 'starttime', 'total_calculation_time']:
                raise ValueError('Unexpected target CSV metadata columns')
            later = pd.DataFrame(np.zeros((len(metadata) - count, len(prior.columns) - 3), dtype=np.uint8),
                                 columns=prior.columns[3:])
            later.insert(0, 'total_calculation_time', 0.)
            later.insert(0, 'starttime', metadata.starttime.iloc[count:].to_numpy())
            later.insert(0, 'Run No', metadata['Run No'].iloc[count:].to_numpy())
            return pd.concat([prior, later], ignore_index=True)
        return original_read(path, *args, **kwargs)

    with patch.object(pd, 'read_csv', read_csv):
        return builder(root)


def fit_standardization(training: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute float64 population mean/std from supplied training rows only."""
    values = np.asarray(training)
    if values.ndim != 2 or not all(values.shape) or not np.isfinite(values).all():
        raise ValueError('Expected a nonempty finite training matrix')
    mean = values.mean(axis=0, dtype=np.float64)
    standard_deviation = values.std(axis=0, dtype=np.float64, ddof=0)
    scale = np.where(standard_deviation == 0., 1., standard_deviation)
    return mean, standard_deviation, scale


def standardize(values: np.ndarray, mean: np.ndarray, scale: np.ndarray,
                chunk_rows: int = 8192) -> np.ndarray:
    """Use frozen float64 statistics, clip to [-10,10], and emit float32 rows."""
    values = np.asarray(values)
    if (values.ndim != 2 or mean.shape != (values.shape[1],) or scale.shape != mean.shape
            or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any()
            or chunk_rows <= 0):
        raise ValueError('Invalid normalization inputs')
    result = np.empty(values.shape, dtype=np.float32)
    for start in range(0, len(values), chunk_rows):
        block = values[start:start + chunk_rows].astype(np.float64)
        if not np.isfinite(block).all():
            raise ValueError('Nonfinite input features')
        block -= mean
        block /= scale
        np.clip(block, -10., 10., out=block)
        result[start:start + chunk_rows] = block
    return result


def portable_predict(values: np.ndarray, coefficients: Sequence[np.ndarray],
                     intercepts: Sequence[np.ndarray]) -> np.ndarray:
    """Run dense ReLU hidden layers and independent logistic output units."""
    if len(coefficients) != len(intercepts) or len(coefficients) < 2:
        raise ValueError('Expected matching hidden/output layer weights and intercepts')
    activation = np.asarray(values, dtype=np.float32)
    if activation.ndim != 2 or not np.isfinite(activation).all():
        raise ValueError('Expected finite 2-D normalized inputs')
    for index, (weights, bias) in enumerate(zip(coefficients, intercepts)):
        if (weights.dtype != np.float32 or bias.dtype != np.float32
                or weights.ndim != 2 or bias.shape != (weights.shape[1],)
                or activation.shape[1] != weights.shape[0]):
            raise ValueError('Unexpected portable layer schema')
        activation = activation @ weights
        activation += bias
        if index + 1 == len(coefficients):
            expit(activation, out=activation)
        else:
            np.maximum(activation, 0., out=activation)
    return activation
