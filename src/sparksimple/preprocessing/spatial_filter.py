"""sparksimple.preprocessing.spatial_filter — Gaussian spatial smoothing (MATLAB ``fs``).

Mirrors the spatial filter applied while building the volume in
``mfiles/sparkDetect.m`` (lines 63-81) and the kernel builder in
``mfiles/gaussiana2d.m``.

MATLAB reference (sparkDetect.m)
--------------------------------
    fs=round(fs/DX);
    fs=fs+mod(fs,2)+1;
    GF=gaussiana2d(fs);
    for ii=1:temps
        ...
        if(fs>0)
            Im=conv2(double(Im),GF,'same');
        end
        volum(:,:,ii)=Im;
    end

Two behaviours of that block are easy to misread and are reproduced here
deliberately:

1. The ``if(fs>0)`` guard is **always true**. The preceding
   ``fs=fs+mod(fs,2)+1`` forces ``fs >= 1`` regardless of the value the user
   supplied, so ``fs = 0`` does not switch the convolution off.
2. What actually makes ``fs = 0`` a no-op is the kernel: it becomes
   ``gaussiana2d(1) == [[1.0]]``, and convolving with that is the identity.

The kernel is **not normalised** (``gaussiana2d`` has no division by its sum).
The resulting intensity gain is harmless because Fo is estimated from the same
smoothed volume, so it cancels in ΔF/F₀ = (F - Fo) / Fo.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import convolve2d


def matlab_round(value: float) -> int:
    """Round half away from zero, matching MATLAB ``round``.

    ``numpy.round`` rounds halves to even, so it cannot be used here:
    ``round(0.5)`` is ``1`` in MATLAB but ``0`` in NumPy, and ``round(2.5)`` is
    ``3`` in MATLAB but ``2`` in NumPy.

    Parameters
    ----------
    value : float
        Value to round.

    Returns
    -------
    int
        Rounded value.
    """
    if value >= 0:
        return int(math.floor(value + 0.5))
    return int(math.ceil(value - 0.5))


def gaussiana2d(n: int) -> np.ndarray:
    """Build the ``n``×``n`` Gaussian kernel matching ``mfiles/gaussiana2d.m``.

    MATLAB reference:
        B=zeros(N,N);
        sig1=N*.2;
        sig2=N*.2;
        for ii=1:N
        for jj=1:N
            B(ii,jj)=sqrt(exp(-((ii-ceil(N/2))^2)/sig1/sig1)
                         *exp(-((jj-ceil(N/2))^2)/sig2/sig2));
        end
        end

    The kernel is separable, so the product-then-square-root form above is
    computed here as the outer product of a single 1-D profile.

    Parameters
    ----------
    n : int
        Kernel side length in pixels. Must be >= 1.

    Returns
    -------
    np.ndarray, shape (n, n), float64
        Unnormalised Gaussian kernel. ``gaussiana2d(1)`` returns ``[[1.0]]``.
    """
    if n < 1:
        raise ValueError(f"Kernel size n must be >= 1, got {n}")

    sigma = n * 0.2
    centre = math.ceil(n / 2)

    # MATLAB ii/jj are 1-based
    offsets = np.arange(1, n + 1, dtype=np.float64) - centre
    profile = np.exp(-(offsets**2) / sigma / sigma)

    return np.sqrt(np.outer(profile, profile))


def spatial_kernel_size_px(spatial_smooth_um: float, dx_um: float) -> int:
    """Convert the ``fs`` smoothing width from µm to the MATLAB kernel size in pixels.

    MATLAB reference (sparkDetect.m lines 63-64):
        fs=round(fs/DX);
        fs=fs+mod(fs,2)+1;

    The result is always odd and always >= 1, which is why the later
    ``if(fs>0)`` guard never rejects anything.

    Parameters
    ----------
    spatial_smooth_um : float
        Smoothing width in micrometres (MATLAB ``fs`` before conversion).
        Must be >= 0.
    dx_um : float
        Pixel size in micrometres (MATLAB ``DX``). Must be > 0.

    Returns
    -------
    int
        Odd kernel side length in pixels. Returns 1 when ``spatial_smooth_um``
        rounds to 0 pixels, which makes the convolution an identity.
    """
    if spatial_smooth_um < 0:
        raise ValueError(
            f"spatial_smooth_um must be >= 0, got {spatial_smooth_um}"
        )
    if dx_um <= 0:
        raise ValueError(f"dx_um must be > 0, got {dx_um}")

    fs_px = matlab_round(spatial_smooth_um / dx_um)
    return fs_px + (fs_px % 2) + 1


def apply_spatial_smoothing(
    stack: np.ndarray,
    spatial_smooth_um: float,
    dx_um: float,
) -> tuple[np.ndarray, int]:
    """Convolve every frame of a stack with the MATLAB Gaussian kernel.

    Replicates ``sparkDetect.m`` lines 63-81. Must be applied **before** the
    cell mask is computed, because MATLAB derives the mask from the already
    smoothed volume (``mk = cellMask4(sum(volum,3))``, line 87).

    Parameters
    ----------
    stack : np.ndarray, shape (rows, cols, frames)
        Raw image stack in (Y, X, T) order.
    spatial_smooth_um : float
        Smoothing width in micrometres (MATLAB ``fs``).
    dx_um : float
        Pixel size in micrometres (MATLAB ``DX``).

    Returns
    -------
    smoothed : np.ndarray, shape (rows, cols, frames)
        Smoothed stack. When the kernel reduces to 1×1 the input is returned
        unchanged (same dtype, same values), matching the MATLAB no-op.
    kernel_px : int
        The kernel side length that was used, for traceability.
    """
    if stack.ndim != 3:
        raise ValueError(
            f"stack must be 3-D (rows, cols, frames), got shape {stack.shape}"
        )

    kernel_px = spatial_kernel_size_px(spatial_smooth_um, dx_um)

    # gaussiana2d(1) is [[1.0]] and conv2 with it is the identity. Short-circuit
    # so the default configuration is bit-for-bit unchanged.
    if kernel_px == 1:
        return stack, kernel_px

    kernel = gaussiana2d(kernel_px)

    # MATLAB conv2(..., 'same') pads with zeros and keeps the centred window.
    # The kernel side is always odd, so the centring is unambiguous.
    smoothed = np.empty(stack.shape, dtype=np.float64)
    for t in range(stack.shape[2]):
        smoothed[:, :, t] = convolve2d(
            stack[:, :, t].astype(np.float64),
            kernel,
            mode="same",
            boundary="fill",
            fillvalue=0.0,
        )

    return smoothed.astype(stack.dtype, copy=False), kernel_px
