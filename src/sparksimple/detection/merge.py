"""sparksimple.detection.merge — Phase 6A 3D Merge and Thresholding pipeline orchestrator.

Mirrors MATLAB ``sparkDetect.m`` lines 173–198.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from sparksimple.config import DetectionConfig
from sparksimple.detection.convolution import conv3_matlab
from sparksimple.detection.kernel import build_merge_kernel_3d
from sparksimple.detection.threshold import apply_threshold, estimate_noise_threshold


@dataclass
class MergeResult:
    """Dataclass holding outputs of the Phase 6A 3D merge & thresholding stage.

    Attributes
    ----------
    v3_conv : np.ndarray
        3D volume after 3D merge kernel convolution, shape ``(Y, X, T)``, dtype ``float64``.
    umbral : float
        Calculated noise threshold scalar.
    v3_thresholded : np.ndarray
        3D volume after setting values below ``umbral`` to 0.0, shape ``(Y, X, T)``, dtype ``float64``.
    kernel : np.ndarray
        3D merge kernel used for convolution, shape ``(t3, t3, w3)``, dtype ``float64``.
    centroid_pixel : tuple[int, int]
        0-based ``(y, x)`` pixel coordinates of cell mask centroid used for noise trace.
    """

    v3_conv: np.ndarray
    umbral: float
    v3_thresholded: np.ndarray
    kernel: np.ndarray
    centroid_pixel: tuple[int, int]


def apply_3d_merge_and_threshold(
    v2: np.ndarray,
    mask: np.ndarray,
    dx_um: float,
    dt_ms: float,
    merge_distance_um: float | None = None,
    merge_time_ms: float | None = None,
    cfg: DetectionConfig | None = None,
) -> MergeResult:
    """Orchestrate Phase 6A: 3D merge kernel generation, convolution, thresholding.

    Parameters
    ----------
    v2 : np.ndarray
        3D CWT volume output from Phase 5 of shape ``(Y, X, T)``.
    mask : np.ndarray
        2D binary cell mask of shape ``(Y, X)``.
    dx_um : float
        Spatial resolution (µm/pixel).
    dt_ms : float
        Temporal resolution (ms/frame).
    merge_distance_um : float | None
        Spatial merge distance (µm).  If None, taken from ``cfg.merge_distance_um``.
    merge_time_ms : float | None
        Temporal merge time (ms).  If None, taken from ``cfg.merge_time_ms``.
    cfg : DetectionConfig | None
        Optional configuration object.

    Returns
    -------
    MergeResult
    """
    if cfg is not None:
        if merge_distance_um is None:
            merge_distance_um = cfg.merge_distance_um
        if merge_time_ms is None:
            merge_time_ms = cfg.merge_time_ms

    if merge_distance_um is None or merge_time_ms is None:
        raise ValueError(
            "merge_distance_um and merge_time_ms must be provided either directly "
            "or via DetectionConfig"
        )

    # 1. Build 3D merge kernel
    kernel = build_merge_kernel_3d(
        merge_distance_um=merge_distance_um,
        merge_time_ms=merge_time_ms,
        dx_um=dx_um,
        dt_ms=dt_ms,
    )

    # 2. Check kernel size against volume size (matching sparkDetect.m line 180)
    if kernel.shape[0] > v2.shape[0] or kernel.shape[1] > v2.shape[1]:
        max_merge_distance_um = min(v2.shape[0], v2.shape[1]) * dx_um / 2.0
        raise ValueError(
            f"Image spatial size {v2.shape[:2]} is smaller than merge kernel spatial "
            f"size {kernel.shape[:2]} (merge_distance_um={merge_distance_um}, "
            f"dx_um={dx_um}). Pass --merge-distance-um with a value below "
            f"~{max_merge_distance_um:.2f} um to fit this image."
        )

    # 3. 3D FFT convolution
    v3_conv = conv3_matlab(v2, kernel)

    # 4. Estimate noise threshold
    umbral, centroid_pixel = estimate_noise_threshold(v3_conv, mask)

    # 5. Apply thresholding
    v3_thresholded = apply_threshold(v3_conv, umbral)

    return MergeResult(
        v3_conv=v3_conv,
        umbral=umbral,
        v3_thresholded=v3_thresholded,
        kernel=kernel,
        centroid_pixel=centroid_pixel,
    )
