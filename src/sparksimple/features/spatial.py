"""sparksimple.features.spatial — Spatial feature helpers (membrane distance & FWHM).

Ports MATLAB functions:
- md = bwdist(1 - mk) * DX (distance transform to cell mask boundary)
- Blob4Diameters.m (2D spatial FWHM estimation from Gaussian profile fits)
"""

from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi
from scipy.signal import firls, upfirdn
from functools import lru_cache
from sparksimple.features.robust import bisquare_fit, LocalFitFailure


class EmptyDiameter(LocalFitFailure):
    """MATLAB tallaDiametre returns [] when no strict crossing exists."""


def distance_to_membrane(mask: np.ndarray, dx_um: float) -> np.ndarray:
    """Calculate Euclidean distance to cell mask boundary in micrometres (µm).

    Mirrors MATLAB ``md = bwdist(1 - mk) * DX``.

    Parameters
    ----------
    mask : np.ndarray
        2D binary or numeric cell mask (1 inside cell, 0 outside).
    dx_um : float
        Spatial pixel resolution in µm/pixel.

    Returns
    -------
    dist_map : np.ndarray
        2D float64 array of same shape as ``mask`` containing Euclidean distance
        to nearest background (non-mask) pixel in µm.
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {mask.shape}")

    mask_bool = mask.astype(bool)
    dist_px = ndi.distance_transform_edt(mask_bool)
    return dist_px * float(dx_um)


def blobfit(s: np.ndarray) -> np.ndarray:
    """Fit 1D normalized profile with Gaussian model matching MATLAB ``blobfit``.

    Parameters
    ----------
    s : np.ndarray
        1D profile vector.

    Returns
    -------
    ft : np.ndarray
        1D Gaussian fit evaluated at 1-based pixel indices [1..len(s)].
    """
    s_arr = np.asarray(s, dtype=np.float64)
    length = len(s_arr)
    s_min = float(np.min(s_arr))
    s_max = float(np.max(s_arr))

    rng = s_max - s_min
    if rng == 0:
        s_norm = np.zeros(length, dtype=np.float64)
    else:
        s_norm = (s_arr - s_min) / rng

    x = np.arange(1, length + 1, dtype=np.float64)
    c = length / 2.0

    def gaus_model(x_val, m, s_param):
        return np.exp(-((x_val - m) ** 2) / (2.0 * (s_param ** 2)))

    p0 = [c, 1.0]

    if rng == 0 or length <= 2:
        raise LocalFitFailure("constant_or_short_spatial_profile")
    fit = bisquare_fit(gaus_model, x, s_norm, p0, [0.0, 0.0],
                      [float(length), float(length)/2.0])
    m_fit, s_fit = fit.params

    ft = np.exp(-((x - m_fit) ** 2) / (2.0 * (s_fit ** 2)))
    return ft


@lru_cache(maxsize=1)
def _interp10_filter():
    # interp(ft,10): order 80, passband 0..0.05 (Nyquist units),
    # five image rejection bands. The final band is clipped at Nyquist.
    bands = [0.0, 0.05]
    desired = [10.0, 10.0]
    for k in range(1, 6):
        bands.extend([k/5.0-0.05, min(1.0,k/5.0+0.05)])
        desired.extend([0.0, 0.0])
    return firls(81, bands, desired, fs=2.0)


def interp10(ft):
    """FIR interpolation with endpoint slope reflection, length N*10.

    Matches MATLAB interp default n=4, cutoff=.5 and delay compensation.
    Input must contain at least nine samples.
    """
    ft = np.asarray(ft, dtype=float)
    if len(ft) < 9:
        raise LocalFitFailure("interp_requires_nine_samples")
    extended = np.r_[2*ft[0]-ft[8:0:-1], ft,
                     2*ft[-1]-ft[-2:-10:-1]]
    filtered = upfirdn(_interp10_filter(), extended, up=10)
    return filtered[120:120+10*len(ft)]


def talla_diametre(ft: np.ndarray, height: float = 0.5) -> float:
    fti = interp10(ft)
    gg = fti-height
    crossings = np.flatnonzero(gg[:-1]*gg[1:] < 0)
    if not len(crossings):
        raise EmptyDiameter("no_strict_halfheight_crossing")
    # MATLAB x=[10:length(fti)+9]/10; first crossing index, last+1.
    # One crossing yields .1 px in the actual MATLAB source.
    return float((crossings[-1]+1-crossings[0])/10.0)


def measure_blob_fwhm(
    im_roi_2d: np.ndarray,
    dx_um: float,
    height: float = 0.5,
) -> float:
    """Measure 2D spatial FWHM (in µm) matching MATLAB ``Blob4Diameters.m``.

    Parameters
    ----------
    im_roi_2d : np.ndarray
        2D square float64 image ROI around candidate peak.
    dx_um : float
        Spatial resolution in µm/pixel.
    height : float
        Height cut fraction (default: 0.5 for FWHM).

    Returns
    -------
    fwhm_um : float
        Full Width at Half Maximum in µm.
    """
    if im_roi_2d.ndim != 2:
        raise ValueError(f"im_roi_2d must be 2D, got shape {im_roi_2d.shape}")

    ny, nx = im_roi_2d.shape
    if ny != nx:
        raise ValueError(f"im_roi_2d must be square, got shape ({ny}, {nx})")

    c = int(np.floor(ny/2.0+0.5))-1

    # Profile 1: Horizontal line through center
    s1 = im_roi_2d[c, :]

    # Profile 2: Vertical line through center
    s2 = im_roi_2d[:, c]

    # At gr=1 MATLAB's two diagonal masks reduce to single pixels.
    # It removes zeros and replaces empty (NaN) means with zero.
    s3 = np.diag(im_roi_2d).copy()
    s4 = np.diag(np.flipud(im_roi_2d)).copy()
    s3[~np.isfinite(s3)] = 0.0
    s4[~np.isfinite(s4)] = 0.0

    ft1 = blobfit(s1)
    ft2 = blobfit(s2)
    ft3 = blobfit(s3)
    ft4 = blobfit(s4)

    # MATLAB concatenation [D1,D2,D3*sqrt(2),D4*sqrt(2)] drops [].
    # A failed fit still propagates; an empty crossing result only omits
    # that profile. mean([]) is NaN when every diameter is empty.
    diameters = []
    for fitted, factor in zip((ft1, ft2, ft3, ft4), (1., 1., np.sqrt(2.), np.sqrt(2.))):
        try:
            diameters.append(talla_diametre(fitted, height=height) * factor)
        except EmptyDiameter:
            continue
    d_mean = np.mean(diameters) if diameters else np.nan
    fwhm_um = float(d_mean * dx_um)

    return fwhm_um
