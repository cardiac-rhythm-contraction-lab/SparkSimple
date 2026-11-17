"""sparksimple.detection.smooth — Temporal smoothing (MATLAB smooth equivalent).

Implements moving-average smoothing matching MATLAB Curve Fitting Toolbox ``smooth(y, w)``.

MATLAB smooth behavior
----------------------
- Handles both odd and even window sizes ``w`` with exact precision:
  - For odd ``w`` (half-span k = (w - 1) // 2): window is x[i - m : i + m + 1] where
    m = min(i, k, N - 1 - i).
  - For even ``w``: MATLAB truncates symmetrically near boundaries using
    m = min(i, w // 2, N - 1 - i, w // 2 - 1).
"""

from __future__ import annotations

import numpy as np


def smooth_moving_average_1d(x: np.ndarray, w: int) -> np.ndarray:
    """Apply moving average smoothing to a 1D vector matching MATLAB ``smooth(x, w)``.

    Parameters
    ----------
    x : np.ndarray, shape (T,)
        Input 1D temporal signal.
    w : int
        Smoothing window size. If w <= 1, returns a copy of x.

    Returns
    -------
    np.ndarray, shape (T,)
        Smoothed 1D signal with the same dtype as x (or float64).
    """
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape {x_arr.shape}")

    n = len(x_arr)
    if n == 0 or w <= 1:
        return x_arr.copy()

    y = np.empty(n, dtype=np.float64)

    if w % 2 == 1:
        k = (w - 1) // 2
        for i in range(n):
            m = min(i, k, n - 1 - i)
            y[i] = np.mean(x_arr[i - m : i + m + 1])
    else:
        kl = w // 2
        kr = w // 2 - 1
        for i in range(n):
            ml = min(i, kl)
            mr = min(n - 1 - i, kr)
            m = min(ml, mr)
            y[i] = np.mean(x_arr[max(0, i - m) : min(n, i + m + 1)])

    return y.astype(x.dtype if x.dtype.kind == "f" else np.float64)


def smooth_volume_time(volume: np.ndarray, w: int) -> np.ndarray:
    """Apply MATLAB-faithful temporal moving average smoothing to a 3D volume.

    Smoothing is applied along the time axis (axis 2 of shape (Y, X, T)).

    Parameters
    ----------
    volume : np.ndarray, shape (rows, cols, frames)
        Input 3D volume (Y, X, T).
    w : int
        Smoothing window size in frames. If w <= 1, returns copy of volume.

    Returns
    -------
    np.ndarray, shape (rows, cols, frames)
        Temporally smoothed volume, same dtype as input.
    """
    vol = np.asarray(volume)
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume (rows, cols, frames), got shape {vol.shape}")

    n_rows, n_cols, n_frames = vol.shape
    if n_frames == 0 or w <= 1:
        return vol.copy()

    out = np.empty_like(vol, dtype=np.float64)

    for y in range(n_rows):
        for x in range(n_cols):
            out[y, x, :] = smooth_moving_average_1d(vol[y, x, :], w)

    return out.astype(vol.dtype)
