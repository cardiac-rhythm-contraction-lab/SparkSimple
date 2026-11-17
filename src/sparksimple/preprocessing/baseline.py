"""sparksimple.preprocessing.baseline — Baseline (Fo) estimation functions.

Ports the baseline block of ``mfiles/sparkDetect.m`` (lines 93–131):

    1. ``apply_signal_sign``   — line 94-96:  volum = ss * volum  (+ neg offset)
    2. ``estimate_baseline_global`` — lines 102–114: Fo = quantile(masked_pixels, .05)
    3. ``estimate_baseline_forced`` — lines 120–130: use pre-supplied Fo value

All functions are pure (no side-effects, no I/O) and operate on NumPy arrays.

QUANTILE METHOD — MATLAB-VALIDATED
----------------------------------
MATLAB's default ``quantile`` behaviour for the SparkSimple2 use case matches
NumPy's ``method='hazen'`` (Hyndman-Fan type 5), not ``method='linear'``.
This was verified against MATLAB R2025b using controlled reference vectors:

- ``[2 5 6 10 11 13]`` quartiles -> ``[5 8 11]``
- ``[2 4 6 8 10 12 14]`` quartiles -> ``[4.5 8 11.5]``
- ``1:200`` at ``q=0.05`` -> ``10.5``
- ``1:200`` at ``q=0.10`` -> ``20.5``

These values match ``np.quantile(..., method='hazen')`` exactly.

FLOAT PRECISION NOTE
---------------------
The image loader (``io.tiff_loader``) returns float32 arrays to match MATLAB's
implicit uint16→double pipeline.  Fo is computed in float64 to avoid quantile
precision loss, then the caller decides the output dtype.
"""

from __future__ import annotations

import numpy as np


# ── Signal sign application ───────────────────────────────────────────────────

def apply_signal_sign(
    volume: np.ndarray,
    signal_sign: int,
) -> np.ndarray:
    """Apply signal sign and, if negative, shift the volume so minimum≈0.

    MATLAB equivalent (sparkDetect.m lines 94–96)::

        volum = ss * volum;
        if (ss < 0), volum = volum - quantile(volum(:), [.1]); end

    Parameters
    ----------
    volume : np.ndarray, shape (rows, cols, frames), float
        Raw image stack in (Y, X, T) order.
    signal_sign : int
        +1 for normal Ca²⁺ signals, -1 for inverted (ratiometric) signals.
        Must be 1 or -1.

    Returns
    -------
    np.ndarray
        Volume after sign application.  Same shape and dtype as input.

    Raises
    ------
    ValueError
        If ``signal_sign`` is not 1 or -1.

    Notes
    -----
    For ``signal_sign == -1`` the MATLAB code subtracts the 10th percentile
    of the *entire* volume (all pixels, all frames flattened) to bring the
    distribution back to a positive range.  This uses the same quantile
    method as ``estimate_baseline_global`` and carries the same validation
    caveat (see module docstring).
    """
    if signal_sign not in (1, -1):
        raise ValueError(f"signal_sign must be 1 or -1, got {signal_sign!r}")

    result = signal_sign * volume.astype(np.float64)

    if signal_sign < 0:
        # MATLAB: volum = volum - quantile(volum(:), [.1])
        offset = np.quantile(result, 0.10, method="hazen")
        result = result - offset

    return result.astype(volume.dtype)


# ── Global baseline estimation (auto mode) ────────────────────────────────────

def estimate_baseline_global(
    volume: np.ndarray,
    mask: np.ndarray,
    quantile: float = 0.05,
) -> float:
    """Estimate Fo as a global scalar from the 5th percentile of masked pixels.

    MATLAB equivalent (sparkDetect.m lines 102–114)::

        vk = volum;
        for ii = 1:temps, vk(:,:,ii) = mk; end
        vk = volum .* vk;
        vk = reshape(vk, numel(vk), 1);
        vk(vk == 0) = [];
        estimatedFo = quantile(vk, .05);

    The MATLAB code:
      1. Replicates the mask along time to build a 3-D mask.
      2. Zeros out pixels outside the mask.
      3. Flattens the volume to 1-D.
      4. Removes zeros (i.e., the out-of-mask pixels).
      5. Takes the 5th percentile of the remaining values.

    Parameters
    ----------
    volume : np.ndarray, shape (rows, cols, frames), float
        Image stack *after* ``apply_signal_sign``.
    mask : np.ndarray, shape (rows, cols), bool or 0/1 int
        Cell mask.  True/1 pixels are included; False/0 are excluded.
        Typically ``np.ones((rows, cols), dtype=bool)`` for simulated data.
    quantile : float
        Quantile to use as baseline estimate.
        MATLAB default: 0.05  (sparkDetect.m line 114).
        Comes from ``DetectionConfig.baseline_quantile``.

    Returns
    -------
    float
        Estimated Fo scalar (float64 precision, regardless of input dtype).

    Raises
    ------
    ValueError
        If no in-mask pixels are found (mask is all zeros).

    Notes
    -----
    QUANTILE METHOD — MATLAB-VALIDATED:
    ``np.quantile(..., method='hazen')`` matches the default MATLAB
    ``quantile`` behaviour used by SparkSimple2.  Validation was performed
    against MATLAB R2025b with controlled reference vectors and recorded in
    ``docs/progress.md``.
    """
    if not (0.0 < quantile < 1.0):
        raise ValueError(
            f"quantile must be in (0, 1), got {quantile}"
        )

    mask_bool = mask.astype(bool)

    # Replicate mask along time axis and extract in-mask pixels
    # Equivalent to: vk = volum .* mk (broadcast mask over frames)
    # Then flatten and remove zeros
    vol_f64 = volume.astype(np.float64)

    # shape (rows, cols, frames) * (rows, cols, 1)  → broadcast
    masked_volume = vol_f64 * mask_bool[:, :, np.newaxis]

    # Flatten and remove zeros (= out-of-mask pixels)
    flat = masked_volume.ravel()
    in_mask = flat[flat != 0.0]

    if in_mask.size == 0:
        raise ValueError(
            "No in-mask pixels found. Check that the mask contains at least "
            "one True/1 pixel and that the volume is not all-zero inside the mask."
        )

    fo = float(np.quantile(in_mask, quantile, method="hazen"))
    return fo


# ── Forced baseline (pre-supplied Fo) ─────────────────────────────────────────

def estimate_baseline_forced(forced_value: float) -> float:
    """Return a pre-supplied Fo value unchanged.

    MATLAB equivalent (sparkDetect.m lines 313–314)::

        else
            Fo = forcedbaseline;
        end

    This trivial wrapper exists so that the preprocessing pipeline can call
    a uniform interface regardless of ``baseline_mode``.

    Parameters
    ----------
    forced_value : float
        Pre-computed baseline intensity value.

    Returns
    -------
    float
        The same value, as float64.

    Raises
    ------
    ValueError
        If ``forced_value`` is not positive (Fo ≤ 0 would cause NaN after
        normalisation — caught here to give a clear error message).
    """
    if forced_value <= 0:
        raise ValueError(
            f"forced_value must be > 0 to avoid division by zero during "
            f"normalisation. Got {forced_value!r}."
        )
    return float(forced_value)
