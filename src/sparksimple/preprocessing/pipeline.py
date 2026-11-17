"""sparksimple.preprocessing.pipeline — Preprocessing orchestrator.

Connects image loading, signal-sign application, cell mask computation, baseline
estimation and volume normalisation into a single ``preprocess()`` call.

This module contains only orchestration logic. All numerical computation
lives in ``cell_mask.py``, ``baseline.py`` and ``normalise.py``.

MATLAB correspondence
---------------------
``sparkDetect.m`` lines 83–131 (first preprocessing block & cellMask4).
The re-estimation of Fo that depends on wavelet detection results
(lines 254–315) is **out of scope** for this phase.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sparksimple.config import PipelineConfig
from sparksimple.preprocessing.baseline import (
    apply_signal_sign,
    estimate_baseline_global,
    estimate_baseline_forced,
)
from sparksimple.preprocessing.cell_mask import cell_mask4
from sparksimple.preprocessing.normalise import normalise_volume
from sparksimple.preprocessing.spatial_filter import apply_spatial_smoothing


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class PreprocessingResult:
    """All outputs of the preprocessing stage.

    Attributes
    ----------
    volume_raw : np.ndarray, shape (rows, cols, frames), float32
        Volume *after* ``apply_signal_sign`` but *before* normalisation.
        Corresponds to the MATLAB ``volum`` array immediately after line 96.
    volume_norm : np.ndarray, shape (rows, cols, frames), float32
        Normalised volume (ΔF/F₀ or F/F₀ depending on ``mode``).
        Corresponds to MATLAB ``volum`` after line 118 or 116.
    fo : float
        Scalar baseline estimate used for normalisation.
    mask : np.ndarray, shape (rows, cols), dtype bool
        Cell mask. ``True`` pixels were included in the baseline estimate.
    baseline_mode : str
        ``"auto"`` or ``"forced"`` — from ``PipelineConfig.baseline_mode``.
    normalisation_mode : str
        ``"df_f0"`` or ``"f_f0"`` — from ``DetectionConfig.normalisation_mode``.
    selected_channel : int | None
        The channel index that was loaded. ``None`` if the stack was already
        channel-selected before calling ``preprocess()``.
    mask_source : str
        Source of the mask: ``"external"``, ``"cell_mask4"``, or ``"ones_fallback"``.
    spatial_smooth_px : int
        Side length of the Gaussian kernel applied before masking (MATLAB ``fs``
        after conversion to pixels). ``1`` means no smoothing was applied.
    """

    volume_raw: np.ndarray
    volume_norm: np.ndarray
    fo: float
    mask: np.ndarray
    baseline_mode: str
    normalisation_mode: str
    selected_channel: int | None = None
    mask_source: str = "cell_mask4"
    spatial_smooth_px: int = 1

    def __repr__(self) -> str:
        rows, cols, frames = self.volume_raw.shape
        return (
            f"PreprocessingResult("
            f"shape=({rows}×{cols}×{frames}), "
            f"fo={self.fo:.4g}, "
            f"baseline_mode={self.baseline_mode!r}, "
            f"norm_mode={self.normalisation_mode!r}, "
            f"mask_source={self.mask_source!r})"
        )


# ── Public API ────────────────────────────────────────────────────────────────

def preprocess(
    stack: np.ndarray,
    cfg: PipelineConfig,
    mask: np.ndarray | None = None,
    auto_cell_mask: bool = True,
    selected_channel: int | None = None,
    dx_um: float | None = None,
) -> PreprocessingResult:
    """Run the full preprocessing pipeline on an image stack.

    Parameters
    ----------
    stack : np.ndarray, shape (rows, cols, frames), float
        Raw image stack, typically the output of ``load_image_stack``.
        Must be in (Y, X, T) order, dtype float32 or float64.
    cfg : PipelineConfig
        Pipeline configuration.
    mask : np.ndarray of shape (rows, cols), bool/int, optional
        Explicit cell mask. If provided, used as-is without recalculation
        (``mask_source="external"``).
    auto_cell_mask : bool, default True
        Controls mask calculation when ``mask`` is None:

        - ``True`` (default for real experimental data, MATLAB ``espuntdata == 0``):
          Computes spatial cell mask using ``cell_mask4(sum(stack, axis=2))``
          (``mask_source="cell_mask4"``).
        - ``False`` (for simulated / point-scan data, MATLAB ``espuntdata == 1``):
          Uses full-frame ``np.ones((rows, cols), dtype=bool)``
          (``mask_source="ones_fallback"``).
    selected_channel : int | None
        For traceability only — channel index used to build the stack.
    dx_um : float | None
        Pixel size in micrometres (MATLAB ``DX``). Required only when
        ``cfg.detection.spatial_smooth_um`` requests spatial smoothing.

    Returns
    -------
    PreprocessingResult
        Contains both signed-raw and normalised volumes, Fo estimate, mask,
        and execution metadata.
    """
    if stack.ndim != 3:
        raise ValueError(
            f"stack must be 3-D (rows, cols, frames), got shape {stack.shape}"
        )

    n_rows, n_cols, _ = stack.shape

    # ── 0. Spatial smoothing ──────────────────────────────────────────────────
    # MATLAB sparkDetect.m lines 63-81: applied while the volume is built, so it
    # must happen *before* the mask, which line 87 derives from the already
    # smoothed volume.
    spatial_smooth_px = 1
    spatial_smooth_um = cfg.detection.spatial_smooth_um
    if spatial_smooth_um:
        if dx_um is None:
            raise ValueError(
                "dx_um is required when cfg.detection.spatial_smooth_um is set "
                f"(got spatial_smooth_um={spatial_smooth_um})."
            )
        stack, spatial_smooth_px = apply_spatial_smoothing(
            stack, spatial_smooth_um, dx_um
        )

    # ── 1. Mask resolution (explicit & traceable) ──────────────────────────────
    if mask is not None:
        if mask.shape != (n_rows, n_cols):
            raise ValueError(
                f"mask shape {mask.shape} does not match stack spatial "
                f"dimensions ({n_rows}, {n_cols})."
            )
        effective_mask = mask.astype(bool)
        mask_source = "external"
    elif auto_cell_mask:
        # Replicates MATLAB sparkDetect.m line 87: mk = cellMask4(sum(volum,3))
        sum_2d = np.sum(stack, axis=2)
        effective_mask = cell_mask4(sum_2d)
        mask_source = "cell_mask4"
    else:
        # Replicates MATLAB sparkDetect.m line 89: mk = ones(size(volum(:,:,1)))
        effective_mask = np.ones((n_rows, n_cols), dtype=bool)
        mask_source = "ones_fallback"

    # ── 2. Apply signal sign ──────────────────────────────────────────────────
    volume_signed = apply_signal_sign(stack, cfg.detection.signal_sign)

    # ── 3. Estimate Fo ────────────────────────────────────────────────────────
    if cfg.baseline_mode == "auto":
        fo = estimate_baseline_global(
            volume_signed,
            effective_mask,
            quantile=cfg.detection.baseline_quantile,
        )
    elif cfg.baseline_mode == "forced":
        if cfg.forced_baseline is None:
            raise ValueError(
                "cfg.forced_baseline must be set when cfg.baseline_mode == 'forced'."
            )
        fo = estimate_baseline_forced(cfg.forced_baseline)
    else:
        raise ValueError(
            f"Unknown cfg.baseline_mode: {cfg.baseline_mode!r}. "
            "Must be 'auto' or 'forced'."
        )

    # ── 4. Normalise ──────────────────────────────────────────────────────────
    volume_norm = normalise_volume(
        volume_signed,
        fo,
        mode=cfg.detection.normalisation_mode,
    )

    return PreprocessingResult(
        volume_raw=volume_signed,
        volume_norm=volume_norm,
        fo=fo,
        mask=effective_mask,
        baseline_mode=cfg.baseline_mode,
        normalisation_mode=cfg.detection.normalisation_mode,
        selected_channel=selected_channel,
        mask_source=mask_source,
        spatial_smooth_px=spatial_smooth_px,
    )
