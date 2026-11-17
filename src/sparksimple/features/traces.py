"""sparksimple.features.traces — ROI trace extraction and temporal snippet cropping.

Ports MATLAB measureFeatures.m trace extraction logic:
- Spatial ROI averaging excluding out-of-mask pixels
- Temporal window cropping around candidate peak frame
"""

from __future__ import annotations

import numpy as np


def extract_spatial_roi_trace(
    volume: np.ndarray,
    mask: np.ndarray,
    center_y: int,
    center_x: int,
    marg_x_px: int,
) -> np.ndarray:
    """Extract 1D time trace averaged over spatial ROI excluding out-of-mask pixels.

    Mirrors MATLAB ``measureFeatures.m`` lines 121–126:
        aux = volum(oy:fy, ox:fx, 1:c0);
        aux2 = mask(oy:fy, ox:fx);
        aux = reshape(aux, [size(aux,1)*size(aux,2), size(aux,3)]);
        aux2 = find(aux2 == 0);
        aux(aux2, :) = [];
        ts = mean(aux, 1)';

    Parameters
    ----------
    volume : np.ndarray
        3D float64 re-normalised volume of shape (nY, nX, nT).
    mask : np.ndarray
        2D binary cell mask of shape (nY, nX).
    center_y : int
        0-based Y coordinate of candidate centroid.
    center_x : int
        0-based X coordinate of candidate centroid.
    marg_x_px : int
        Spatial window half-width in pixels (round(roiR / DX)).

    Returns
    -------
    timesignal : np.ndarray
        1D float64 array of shape (nT,) containing spatial average across frames.
    """
    if volume.ndim != 3:
        raise ValueError(f"volume must be 3D, got shape {volume.shape}")
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {mask.shape}")

    ny, nx, nt = volume.shape

    oy = max(0, center_y - marg_x_px)
    fy = min(ny, center_y + marg_x_px + 1)
    ox = max(0, center_x - marg_x_px)
    fx = min(nx, center_x + marg_x_px + 1)

    vol_roi = volume[oy:fy, ox:fx, :]  # shape (h, w, nt)
    mask_roi = mask[oy:fy, ox:fx]      # shape (h, w)

    h, w, _ = vol_roi.shape
    vol_flat = vol_roi.reshape((h * w, nt), order="F")
    mask_flat = mask_roi.ravel(order="F")

    valid_mask = mask_flat > 0
    valid_pixels = vol_flat[valid_mask, :]
    timesignal = np.mean(valid_pixels, axis=0)

    return timesignal.astype(np.float64)


def crop_temporal_snippet(
    timesignal: np.ndarray,
    center_t: int,
    marg_t_frames: tuple[int, int] | int,
) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    """Crop temporal snippet around candidate peak frame.

    Mirrors MATLAB ``measureFeatures.m`` lines 67–68, 128–129:
        ot = max(1, c1 - margT(1));
        ft = min(c1 + margT(2), c0);
        cut = ts(ot:ft);

    Parameters
    ----------
    timesignal : np.ndarray
        1D float64 signal array of shape (nT,).
    center_t : int
        0-based T frame index of candidate centroid.
    marg_t_frames : tuple[int, int] | int
        Temporal margin in frames (before_peak, after_peak). If int, applied symmetrically.

    Returns
    -------
    cut : np.ndarray
        1D float64 snippet trace.
    timeinterval_1b : np.ndarray
        1D int array of 1-based frame numbers corresponding to ``cut``.
    peak_idx_rel : int
        0-based index of candidate peak relative to ``cut``.
    ot_1b : int
        1-based start frame of cut (ot in MATLAB).
    ft_1b : int
        1-based end frame of cut (ft in MATLAB).
    """
    nt = len(timesignal)

    if isinstance(marg_t_frames, (tuple, list)):
        marg_before, marg_after = int(marg_t_frames[0]), int(marg_t_frames[1])
    else:
        marg_before = marg_after = int(marg_t_frames)

    ot_0b = max(0, center_t - marg_before)
    ft_0b = min(nt, center_t + marg_after + 1)

    cut = timesignal[ot_0b:ft_0b].astype(np.float64)
    ot_1b = ot_0b + 1
    ft_1b = ft_0b

    timeinterval_1b = np.arange(ot_1b, ft_1b + 1, dtype=int)
    peak_idx_rel = center_t - ot_0b

    return cut, timeinterval_1b, peak_idx_rel, ot_1b, ft_1b
