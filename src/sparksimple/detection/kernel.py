"""sparksimple.detection.kernel — 2D/3D merge kernel generation.

Mirrors MATLAB ``gaussiana2d.m`` and kernel setup in ``sparkDetect.m`` lines 174–179.
"""

from __future__ import annotations

import math
import numpy as np


def build_gaussian_2d(n: int) -> np.ndarray:
    """Build a 2D Gaussian kernel of size (n, n) matching MATLAB ``gaussiana2d.m``.

    MATLAB reference (gaussiana2d.m):
        sig1 = N * 0.2; sig2 = N * 0.2;
        for ii = 1:N
            for jj = 1:N
                B(ii,jj) = sqrt(exp(-((ii-ceil(N/2))^2)/sig1^2) * exp(-((jj-ceil(N/2))^2)/sig2^2));
            end
        end

    Parameters
    ----------
    n : int
        Size of the square 2D kernel (N x N).

    Returns
    -------
    np.ndarray
        2D Gaussian kernel of shape ``(n, n)`` with dtype ``float64``.
    """
    n_int = int(n)
    if n_int <= 0:
        raise ValueError(f"Kernel size n must be positive, got {n}")

    sig = n_int * 0.2
    c = float(math.ceil(n_int / 2.0))  # 1-indexed center coordinate in MATLAB

    b = np.zeros((n_int, n_int), dtype=np.float64)
    sig_sq = sig * sig

    for ii in range(n_int):
        row_dist_sq = ((ii + 1) - c) ** 2
        term1 = math.exp(-row_dist_sq / sig_sq)
        for jj in range(n_int):
            col_dist_sq = ((jj + 1) - c) ** 2
            term2 = math.exp(-col_dist_sq / sig_sq)
            b[ii, jj] = math.sqrt(term1 * term2)

    return b


def build_merge_kernel_3d(
    merge_distance_um: float,
    merge_time_ms: float,
    dx_um: float,
    dt_ms: float,
) -> np.ndarray:
    """Build and normalise the 3D merge kernel matching ``sparkDetect.m`` lines 174–179.

    MATLAB reference:
        t3 = ceil(2 * dm / DX); % spatial kernel dimension (t3 x t3)
        w3 = ceil(2 * tm / DT); % temporal kernel dimension (w3 frames)
        B = gaussiana2d(t3);
        for ii = 2:w3, B(:,:,ii) = B(:,:,1); end
        B = B / sum(sum(sum(B)));

    Parameters
    ----------
    merge_distance_um : float
        Spatial merge distance in micrometres (dm in MATLAB).
    merge_time_ms : float
        Temporal merge time in milliseconds (tm in MATLAB).
    dx_um : float
        Spatial pixel resolution (µm/pixel).
    dt_ms : float
        Temporal frame resolution (ms/frame).

    Returns
    -------
    np.ndarray
        3D merge kernel of shape ``(t3, t3, w3)`` with dtype ``float64``,
        normalised so that ``sum(kernel) == 1.0``.
    """
    if dx_um <= 0 or dt_ms <= 0:
        raise ValueError(f"dx_um and dt_ms must be positive, got dx={dx_um}, dt={dt_ms}")
    if merge_distance_um <= 0 or merge_time_ms <= 0:
        raise ValueError(
            f"merge_distance_um and merge_time_ms must be positive, got "
            f"dm={merge_distance_um}, tm={merge_time_ms}"
        )

    t3 = int(math.ceil(2.0 * merge_distance_um / dx_um))
    w3 = int(math.ceil(2.0 * merge_time_ms / dt_ms))

    b_2d = build_gaussian_2d(t3)
    b_3d = np.repeat(b_2d[:, :, np.newaxis], w3, axis=2)

    total_sum = np.sum(b_3d)
    if total_sum > 0:
        b_3d = b_3d / total_sum

    return b_3d
