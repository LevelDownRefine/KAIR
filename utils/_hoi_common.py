"""Validation shared by the higher-order connectivity utilities."""

from itertools import combinations
from numbers import Integral

import numpy as np


def positive_int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def timeseries(data, min_regions=2, min_samples=2, nonconstant=True):
    if np.iscomplexobj(data):
        raise ValueError("data must be real")
    x = np.asarray(data, dtype=float)
    if x.ndim != 2 or x.shape[0] < min_samples or x.shape[1] < min_regions:
        raise ValueError(
            f"data must have shape (time, regions), with at least "
            f"{min_samples} samples and {min_regions} regions"
        )
    if not np.isfinite(x).all():
        raise ValueError("data must be finite; handle missing/censored samples before analysis")
    if nonconstant and np.any(np.ptp(x, axis=0) == 0):
        raise ValueError("constant regional time series are not supported")
    return x


def multiplets(n_regions, order, indices=None):
    if indices is None:
        return np.array(list(combinations(range(n_regions), order)), dtype=int).reshape(-1, order)
    a = np.asarray(indices)
    if a.ndim != 2 or a.shape[1] != order or not np.issubdtype(a.dtype, np.integer):
        raise ValueError(f"indices must be an integer array with shape (n, {order})")
    if np.any(a < 0) or np.any(a >= n_regions) or np.any(np.diff(a, axis=1) <= 0):
        raise ValueError("each multiplet must contain increasing, distinct, in-range indices")
    if len(np.unique(a, axis=0)) != len(a):
        raise ValueError("duplicate multiplets are not allowed")
    return a.astype(int, copy=True)


def standardize(x):
    centered = x - x.mean(axis=0)
    scale = centered.std(axis=0)
    if np.any(scale == 0) or not np.isfinite(scale).all():
        raise ValueError("cannot standardize constant or numerically overflowing signals")
    return centered / scale
