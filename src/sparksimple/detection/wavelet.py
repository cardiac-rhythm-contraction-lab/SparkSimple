"""sparksimple.detection.wavelet — Continuous Wavelet Transform (CWT) & V2 volume construction.

Implements scale selection, 1D/3D CWT with 'gaus2' (Mexican Hat) wavelet, and V2 volume
reconstruction matching MATLAB sparkDetect.m lines 137–163.

MATLAB CWT parity notes
-----------------------
MATLAB classic ``cwt(ySIG, scales, 'gaus2')`` uses the integrated wavelet ``intwave('gaus2', 10)``
and computes continuous wavelet coefficients for scale ``a`` via:
    j = 1 + floor((0 : (a/stepSIG)*xMaxWAV) / ((a/stepSIG)*stepWAV))
    f = conj(fliplr(val_WAV(j)))
    coefs = -sqrt(a) * wkeep1(diff(conv(ySIG, f)), lenSIG)

This module implements that exact numerical algorithm, achieving ~1e-14 float64 parity
with MATLAB R2025b.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sparksimple.config import DetectionConfig, PipelineConfig
from sparksimple.detection._gaus2_table import GAUS2_VAL_WAV, GAUS2_XWAV
from sparksimple.detection.smooth import smooth_volume_time


@dataclass
class WaveletResult:
    """Output container for the wavelet transform stage (Phase 5).

    Attributes
    ----------
    v2 : np.ndarray, shape (rows, cols, frames), float32
        Transformed 3D volume (sum of CWT coefficients across scales).
        Corresponds to MATLAB V2 after line 160.
    scales : np.ndarray, 1D, int
        Wavelet scales used for the CWT.
    filter_window_frames : int
        Temporal smoothing window width in frames applied prior to CWT.
    is_simulated : bool
        Whether simulated (.ata) scale selection logic was used.
    """

    v2: np.ndarray
    scales: np.ndarray
    filter_window_frames: int
    is_simulated: bool

    def __repr__(self) -> str:
        rows, cols, frames = self.v2.shape
        return (
            f"WaveletResult("
            f"shape=({rows}×{cols}×{frames}), "
            f"scales={self.scales.tolist()}, "
            f"filter_window_frames={self.filter_window_frames}, "
            f"is_simulated={self.is_simulated})"
        )


def wkeep1(x: np.ndarray, k: int) -> np.ndarray:
    """Extract central k elements of a 1D vector x matching MATLAB ``wkeep1(x, k)``."""
    n = len(x)
    first = int(np.fix((n - k) / 2))
    return x[first : first + k]


def get_wavelet_scales(
    dt_ms: float,
    is_simulated: bool = False,
    cfg: DetectionConfig | None = None,
    *,
    input_format: str = "tiff",
) -> np.ndarray:
    """Determine the CWT scales array matching MATLAB sparkDetect.m lines 139–143.

    MATLAB logic:
        if (espuntdata == 1)
            scala = [ceil(12/DT) : ceil(48/DT)];
        else
            scala = [5:10];
        end

    Parameters
    ----------
    dt_ms : float
        Frame sampling interval in milliseconds (DT).
    is_simulated : bool, default False
        Simulation provenance only; does not select scales.
    input_format : str
        "tiff" (default): frame scales 5..10; "ata": ceil(12/DT)..ceil(48/DT).
    cfg : DetectionConfig, optional
        Configuration containing scale settings. Uses defaults if None.

    Returns
    -------
    np.ndarray, 1D, int
        Array of scale integers.
    """
    if dt_ms <= 0:
        raise ValueError(f"dt_ms must be positive, got {dt_ms}")

    if cfg is None:
        cfg = DetectionConfig()

    if input_format not in ("tiff", "ata"):
        raise ValueError("input_format must be tiff or ata")
    if input_format == "ata":
        low_ms, high_ms = cfg.wavelet_scales_sim_dt_range_ms
        s_min = int(np.ceil(low_ms / dt_ms))
        s_max = int(np.ceil(high_ms / dt_ms))
        if s_min > s_max:
            s_min, s_max = s_max, s_min
        return np.arange(s_min, s_max + 1, dtype=int)

    s_min, s_max = cfg.wavelet_scales_real
    return np.arange(s_min, s_max + 1, dtype=int)


def cwt_gaus2_1d(sig: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Compute Continuous Wavelet Transform for a 1D trace matching MATLAB ``cwt(sig, scales, 'gaus2')``.

    Parameters
    ----------
    sig : np.ndarray, shape (T,)
        Input 1D temporal trace.
    scales : np.ndarray, 1D
        Wavelet scale integers/floats.

    Returns
    -------
    np.ndarray, shape (num_scales, T), float64
        CWT coefficients for each scale.
    """
    sig_raw = np.asarray(sig)
    if sig_raw.ndim != 1:
        raise ValueError(f"Expected 1D signal vector, got shape {sig_raw.shape}")

    sig_arr = sig_raw.astype(np.float64)
    scales_arr = np.asarray(scales, dtype=np.float64)
    len_sig = len(sig_arr)
    nb_scales = len(scales_arr)

    val_wav = GAUS2_VAL_WAV
    x_wav = GAUS2_XWAV
    x_wav_zero = x_wav - x_wav[0]
    x_max_wav = x_wav_zero[-1]
    step_wav = x_wav[1] - x_wav[0]
    step_sig = 1.0

    coefs = np.zeros((nb_scales, len_sig), dtype=np.float64)

    for k, a in enumerate(scales_arr):
        a_sig = a / step_sig
        max_idx = int(np.floor(a_sig * x_max_wav))
        grid = np.arange(0, max_idx + 1, dtype=np.float64)
        j = 1 + np.floor(grid / (a_sig * step_wav)).astype(int) - 1

        f = np.conj(val_wav[j][::-1])

        conv_res = np.convolve(sig_arr, f, mode="full")
        diff_res = np.diff(conv_res)
        kept = wkeep1(diff_res, len_sig)
        coefs[k, :] = -np.sqrt(a) * kept

    return coefs


def cwt_gaus2_volume(
    volume_norm: np.ndarray,
    scales: np.ndarray,
    filter_window_frames: int,
) -> np.ndarray:
    """Compute the 3D transformed volume V2 matching MATLAB sparkDetect.m lines 144–161.

    Parameters
    ----------
    volume_norm : np.ndarray, shape (rows, cols, frames)
        Normalised input volume.
    scales : np.ndarray, 1D
        CWT scale values.
    filter_window_frames : int
        Temporal smoothing window width in frames.

    Returns
    -------
    np.ndarray, shape (rows, cols, frames), float32
        Transformed volume V2.
    """
    vol = np.asarray(volume_norm, dtype=np.float64)
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D volume (rows, cols, frames), got shape {vol.shape}")

    n_rows, n_cols, n_frames = vol.shape
    v2_out = np.zeros((n_rows, n_cols, n_frames), dtype=np.float64)

    # Step 1: Temporal smoothing across entire volume along axis 2
    if filter_window_frames > 1:
        vol_smooth = smooth_volume_time(vol, filter_window_frames)
    else:
        vol_smooth = vol.copy()

    # Step 2: Subtract temporal mean per pixel trace
    mean_trace = np.mean(vol_smooth, axis=2, keepdims=True)
    vol_centered = vol_smooth - mean_trace

    # Step 3 & 4: Trace-by-trace CWT computation matching MATLAB exactly
    scales_arr = np.asarray(scales, dtype=np.float64)

    for y in range(n_rows):
        for x in range(n_cols):
            sig = vol_centered[y, x, :]
            c = cwt_gaus2_1d(sig, scales_arr)
            v2_out[y, x, :] = np.sum(c, axis=0)

    return v2_out.astype(np.float32)


def compute_v2(
    volume_norm: np.ndarray,
    dt_ms: float,
    is_simulated: bool = False,
    cfg: PipelineConfig | None = None,
    *,
    input_format: str = "tiff",
) -> WaveletResult:
    """High-level entry point for Phase 5 (wavelet transform & V2 volume construction).

    Parameters
    ----------
    volume_norm : np.ndarray, shape (rows, cols, frames)
        Normalised input volume (output of preprocessing).
    dt_ms : float
        Frame interval in milliseconds (DT).
    is_simulated : bool, default False
        Simulation provenance; TIFF always uses frame scales.
    input_format : str
        Explicit acquisition format ("tiff" or "ata").
    cfg : PipelineConfig, optional
        Pipeline configuration object.

    Returns
    -------
    WaveletResult
        Contains V2 volume, scales, filter_window_frames, and is_simulated flag.
    """
    if cfg is None:
        cfg = PipelineConfig()

    det_cfg = cfg.detection

    # 1. Determine filter window in frames
    # MATLAB: w = filtime; if (w==0), w=1; end
    w_frames = max(1, int(round(det_cfg.filter_window_frames)))

    # 2. Get scales
    scales = get_wavelet_scales(dt_ms=dt_ms, is_simulated=is_simulated, cfg=det_cfg, input_format=input_format)

    # 3. Compute V2
    v2 = cwt_gaus2_volume(
        volume_norm=volume_norm,
        scales=scales,
        filter_window_frames=w_frames,
    )

    return WaveletResult(
        v2=v2,
        scales=scales,
        filter_window_frames=w_frames,
        is_simulated=is_simulated,
    )
