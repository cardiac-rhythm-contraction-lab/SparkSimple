"""sparksimple.detection.threshold — Noise threshold estimation and volume thresholding.

Mirrors MATLAB ``sparkDetect.m`` lines 190–198.
"""

from __future__ import annotations

import numpy as np


def get_mask_center_pixel_coords(
    mask: np.ndarray,
) -> tuple[int, int, int, int]:
    """Calculate cell mask centroid pixel coordinates matching MATLAB ``sparkDetect.m`` line 190–191.

    MATLAB reference:
        [ax1 ax2] = find(mk == 1);
        y_center_1based = round(mean(ax1));
        x_center_1based = round(mean(ax2));

    Parameters
    ----------
    mask : np.ndarray
        2D binary cell mask of shape ``(Y, X)``.

    Returns
    -------
    tuple[int, int, int, int]
        ``(y_0based, x_0based, y_1based, x_1based)``
    """
    if mask.ndim != 2:
        raise ValueError(f"Mask must be a 2D array, got ndim={mask.ndim}")

    y_indices, x_indices = np.where(mask == 1)
    if len(y_indices) == 0:
        # Fallback to volume spatial center if mask is empty
        y_0based = mask.shape[0] // 2
        x_0based = mask.shape[1] // 2
        return y_0based, x_0based, y_0based + 1, x_0based + 1

    # Convert to 1-based indexing as in MATLAB find()
    y_1based_coords = y_indices + 1
    x_1based_coords = x_indices + 1

    # MATLAB round() rounds half-integers away from zero: floor(val + 0.5)
    y_1based = int(np.floor(np.mean(y_1based_coords) + 0.5))
    x_1based = int(np.floor(np.mean(x_1based_coords) + 0.5))

    # Convert back to 0-based indexing for Python array lookup
    y_0based = y_1based - 1
    x_0based = x_1based - 1

    # Clamp bounds for safety
    y_0based = max(0, min(mask.shape[0] - 1, y_0based))
    x_0based = max(0, min(mask.shape[1] - 1, x_0based))

    return y_0based, x_0based, y_1based, x_1based


def estimate_noise_threshold(
    v3: np.ndarray,
    mask: np.ndarray,
    multiplier: float = 2.22,
) -> tuple[float, tuple[int, int]]:
    """Estimate noise threshold from center temporal trace matching MATLAB ``sparkDetect.m`` lines 190–194.

    MATLAB reference:
        [ax1 ax2] = find(mk == 1);
        aux6 = reshape(V3(round(mean(ax1)), round(mean(ax2)), :), 1, temps);
        umbral = mad(aux6) * 2.22; % mean absolute deviation * 2.22

    Parameters
    ----------
    v3 : np.ndarray
        3D volume after 3D convolution of shape ``(Y, X, T)``.
    mask : np.ndarray
        2D binary cell mask of shape ``(Y, X)``.
    multiplier : float
        Noise factor multiplier (default: 2.22 from MATLAB line 194).

    Returns
    -------
    tuple[float, tuple[int, int]]
        ``(threshold, (y_center_0based, x_center_0based))``
    """
    if v3.ndim != 3:
        raise ValueError(f"v3 must be a 3D volume, got ndim={v3.ndim}")

    y_0b, x_0b, _, _ = get_mask_center_pixel_coords(mask)

    # Extract 1D temporal trace at centroid pixel
    trace = v3[y_0b, x_0b, :].astype(np.float64)

    # MATLAB mad(X) computes mean absolute deviation around mean: mean(abs(X - mean(X)))
    mad_val = float(np.mean(np.abs(trace - np.mean(trace))))
    threshold = float(mad_val * multiplier)

    return threshold, (y_0b, x_0b)


def apply_threshold(v3: np.ndarray, umbral: float) -> np.ndarray:
    """Zero out all voxels in volume below threshold matching MATLAB line 198.

    MATLAB reference:
        V3(V3 < umbral) = 0;

    Parameters
    ----------
    v3 : np.ndarray
        3D volume of shape ``(Y, X, T)``.
    umbral : float
        Noise threshold value.

    Returns
    -------
    np.ndarray
        Thresholded 3D volume of shape ``(Y, X, T)`` with dtype ``float64``.
    """
    v3_out = v3.copy().astype(np.float64)
    v3_out[v3_out < umbral] = 0.0
    return v3_out
