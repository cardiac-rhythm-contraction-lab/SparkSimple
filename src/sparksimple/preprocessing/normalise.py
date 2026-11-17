"""sparksimple.preprocessing.normalise — Volume normalisation functions.

Ports the normalisation block of ``mfiles/sparkDetect.m`` (lines 115–130):

    optB = 1  →  (F - Fo) / Fo    ``mode="df_f0"``   (MATLAB default)
    optB = 0  →   F / Fo          ``mode="f_f0"``

All functions accept Fo as either a scalar ``float`` or a 2-D NumPy array
with shape ``(rows, cols)``.  NumPy broadcasting makes the time loop
unnecessary, but the result is numerically identical to the MATLAB per-frame
loop used for R2016b compatibility.

MATLAB code (sparkDetect.m lines 115–130)::

    if (optB == 0)
        volum = (volum) ./ estimatedFo;
    else
        volum = (volum - estimatedFo) ./ estimatedFo;
    end
    % — forced baseline branch (lines 121–130) —
    if (optB == 0)
        volum = (volum) ./ forcedbaseline;
    else
        for ii = 1:temps
            volum(:,:,ii) = (volum(:,:,ii) - forcedbaseline) ./ forcedbaseline;
        end
    end
"""

from __future__ import annotations

import numpy as np


# ── Core normalisation functions ──────────────────────────────────────────────

def normalise_df_f0(
    volume: np.ndarray,
    fo: float | np.ndarray,
) -> np.ndarray:
    """Normalise as ΔF/F₀ = (F − Fo) / Fo.

    MATLAB equivalent (sparkDetect.m line 118, optB=1)::

        volum = (volum - estimatedFo) ./ estimatedFo;

    Parameters
    ----------
    volume : np.ndarray, shape (rows, cols, frames), float
        Image stack after ``apply_signal_sign``.
    fo : float or np.ndarray of shape (rows, cols)
        Baseline estimate.  A scalar is broadcast over all pixels and frames;
        a 2-D array is broadcast over frames only, matching the per-pixel
        spatial Fo map used in the re-estimation phase.

    Returns
    -------
    np.ndarray
        Normalised volume, same shape as input, dtype float32.

    Raises
    ------
    ValueError
        If *fo* is a scalar ≤ 0 (would produce NaN/Inf silently).
    """
    _validate_fo(fo)
    vol_f64 = volume.astype(np.float64)

    if isinstance(fo, np.ndarray):
        # 2-D spatial map: (rows, cols) → (rows, cols, 1) for broadcasting
        fo_bc = fo[:, :, np.newaxis].astype(np.float64)
    else:
        fo_bc = float(fo)

    return ((vol_f64 - fo_bc) / fo_bc).astype(np.float32)


def normalise_f_f0(
    volume: np.ndarray,
    fo: float | np.ndarray,
) -> np.ndarray:
    """Normalise as F/F₀.

    MATLAB equivalent (sparkDetect.m line 116, optB=0)::

        volum = (volum) ./ estimatedFo;

    Parameters
    ----------
    volume : np.ndarray, shape (rows, cols, frames), float
        Image stack after ``apply_signal_sign``.
    fo : float or np.ndarray of shape (rows, cols)
        Baseline estimate.

    Returns
    -------
    np.ndarray
        Normalised volume, same shape as input, dtype float32.
    """
    _validate_fo(fo)
    vol_f64 = volume.astype(np.float64)

    if isinstance(fo, np.ndarray):
        fo_bc = fo[:, :, np.newaxis].astype(np.float64)
    else:
        fo_bc = float(fo)

    return (vol_f64 / fo_bc).astype(np.float32)


def normalise_volume(
    volume: np.ndarray,
    fo: float | np.ndarray,
    mode: str,
) -> np.ndarray:
    """Dispatch to the appropriate normalisation function based on *mode*.

    Parameters
    ----------
    volume : np.ndarray
        Image stack after ``apply_signal_sign``.
    fo : float or np.ndarray
        Baseline estimate (scalar or 2-D spatial map).
    mode : str
        ``"df_f0"`` → ΔF/F₀  (MATLAB ``optB=1``, the default).
        ``"f_f0"``  → F/F₀   (MATLAB ``optB=0``).

    Returns
    -------
    np.ndarray
        Normalised volume, dtype float32.

    Raises
    ------
    ValueError
        If *mode* is not ``"df_f0"`` or ``"f_f0"``.
    """
    if mode == "df_f0":
        return normalise_df_f0(volume, fo)
    elif mode == "f_f0":
        return normalise_f_f0(volume, fo)
    else:
        raise ValueError(
            f"mode must be 'df_f0' or 'f_f0', got {mode!r}. "
            "Corresponds to MATLAB optB=1 or optB=0 respectively."
        )


# ── Inverse (denormalise) — for testing and debugging ─────────────────────────

def denormalise_df_f0(
    volume_norm: np.ndarray,
    fo: float | np.ndarray,
) -> np.ndarray:
    """Reverse the ΔF/F₀ normalisation: F = volume_norm * Fo + Fo.

    MATLAB equivalent (sparkDetect.m line 259)::

        volum = volum .* estimatedFo; volum = volum + estimatedFo;

    Used internally for testing roundtrip consistency and for the
    re-estimation block (phase 4+, out of scope here).
    """
    vol_f64 = volume_norm.astype(np.float64)
    if isinstance(fo, np.ndarray):
        fo_bc = fo[:, :, np.newaxis].astype(np.float64)
    else:
        fo_bc = float(fo)
    return (vol_f64 * fo_bc + fo_bc).astype(np.float32)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _validate_fo(fo: float | np.ndarray) -> None:
    """Raise ValueError if *fo* would cause NaN/Inf during normalisation."""
    if isinstance(fo, np.ndarray):
        if np.any(fo <= 0):
            raise ValueError(
                "fo contains non-positive values; normalisation would produce "
                "NaN or Inf.  Check that the baseline estimate is valid."
            )
    else:
        if float(fo) <= 0:
            raise ValueError(
                f"fo must be > 0 to avoid division by zero or Inf. Got {fo!r}."
            )
