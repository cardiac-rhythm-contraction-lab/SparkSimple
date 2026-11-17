"""sparksimple.features.extract — Main feature extraction pipeline orchestrator.

Ports MATLAB measureFeatures.m & sparkFeatures.m.
"""

from __future__ import annotations

import numpy as np
import scipy.signal as signal

from sparksimple.detection.kernel import build_gaussian_2d
from sparksimple.features.confidence import compute_feature_confidence
from sparksimple.features.models import FeatureExtractionResult, SparkFeature
from sparksimple.features.metrics import compute_trace_parameters
from sparksimple.features.robust import LocalFitFailure
from sparksimple.features.spatial import distance_to_membrane, measure_blob_fwhm
from sparksimple.features.traces import crop_temporal_snippet, extract_spatial_roi_trace


def extract_features(
    volume: np.ndarray,
    ix_matrix: np.ndarray,
    mask: np.ndarray,
    dx_um: float,
    dt_ms: float,
    roi_r_um: float = 1.5,
    marg_t_ms: tuple[float, float] | float = (160.0, 320.0),
) -> FeatureExtractionResult:
    """Extract candidate spark features matching MATLAB ``measureFeatures.m``.

    Parameters
    ----------
    volume : np.ndarray
        3D float64 re-normalised volume of shape (nY, nX, nT).
        (Post-Phase 7 baseline re-estimation output).
    ix_matrix : np.ndarray
        3D float64 candidate index matrix IX from RegMax3D (Phase 6B).
    mask : np.ndarray
        2D binary cell mask of shape (nY, nX) (Phase 4).
    dx_um : float
        Spatial resolution in µm/pixel.
    dt_ms : float
        Temporal resolution in ms/frame.
    roi_r_um : float
        Spatial ROI radius parameter in µm. Default: 1.5 µm.
    marg_t_ms : tuple[float, float] | float
        Temporal margin in ms [before_peak, after_peak]. Default: (160.0, 320.0).

    Returns
    -------
    FeatureExtractionResult
        Dataclass containing list of SparkFeature objects, count, and distance_map.
    """
    if volume.ndim != 3:
        raise ValueError(f"volume must be 3D, got shape {volume.shape}")
    if ix_matrix.ndim != 3 or ix_matrix.shape != volume.shape:
        raise ValueError(f"ix_matrix shape {ix_matrix.shape} must match volume shape {volume.shape}")
    if mask.ndim != 2 or mask.shape != volume.shape[:2]:
        raise ValueError(f"mask shape {mask.shape} must match volume spatial shape {volume.shape[:2]}")

    ny, nx, nt = volume.shape

    # 1. Distance transform to membrane / mask boundary
    distance_map = distance_to_membrane(mask, dx_um)

    # 2. Window size and kernel setup matching measureFeatures.m
    marg_x_px = int(np.floor(0.5 + 1.0 * roi_r_um / dx_um))
    afegit = int(np.floor(0.5 + 1.5 * roi_r_um / dx_um))
    if afegit < 4:
        afegit = 4

    convs = int(np.floor(0.5 + 1.0 / dx_um))
    if convs % 2 == 0:
        convs -= 1
    if convs < 1:
        convs = 1

    gaussian_2d_kernel = build_gaussian_2d(convs)

    # Parse temporal margins
    if isinstance(marg_t_ms, (tuple, list)):
        marg_t_before_ms, marg_t_after_ms = float(marg_t_ms[0]), float(marg_t_ms[1])
    else:
        marg_t_before_ms = marg_t_after_ms = float(marg_t_ms)

    marg_t_before_frames = int(np.floor(0.5 + marg_t_before_ms / dt_ms))
    marg_t_after_frames = int(np.floor(0.5 + marg_t_after_ms / dt_ms))

    # 3. Process candidate peaks in descending threshold order
    vals = np.sort(np.unique(ix_matrix))[::-1]
    vals = vals[vals > 0]  # Exclude zero background

    features_list: list[SparkFeature] = []

    for thresh in vals:
        indices = np.flatnonzero((ix_matrix == thresh).ravel(order="F"))
        y_indices, x_indices, t_indices = np.unravel_index(indices, ix_matrix.shape, order="F")

        for y_0b, x_0b, t_0b in zip(y_indices, x_indices, t_indices):
            y_0b, x_0b, t_0b = int(y_0b), int(x_0b), int(t_0b)

            px = x_0b + 1
            py = y_0b + 1
            pt = t_0b + 1

            dist2memb = float(distance_map[y_0b, x_0b])

            # Spatial FWHM estimation (Blob4Diameters)
            oy2 = max(0, y_0b - afegit)
            fy2 = min(ny, y_0b + afegit + 1)
            ox2 = max(0, x_0b - afegit)
            fx2 = min(nx, x_0b + afegit + 1)

            t_margin_fwhm_before = int(np.floor(0.5 + marg_t_before_frames / 3.0))
            t_margin_fwhm_after = int(np.floor(0.5 + marg_t_after_frames / 3.0))
            ot2 = max(0, t_0b - t_margin_fwhm_before)
            ft2 = min(nt, t_0b + t_margin_fwhm_after + 1)

            tr0 = np.sum(volume[oy2:fy2, ox2:fx2, ot2:ft2], axis=2).astype(np.float64)

            # Smooth 2D image with Gaussian kernel
            if gaussian_2d_kernel.size > 1:
                tr0 = signal.convolve2d(tr0, gaussian_2d_kernel, mode="same")

            bg = float(np.min(tr0))
            tr0 = tr0 - bg
            tr0[tr0 < 0.0] = 0.0
            tr0_max = float(np.max(tr0))
            if tr0_max > 0.0:
                tr0 = tr0 / tr0_max

            # MATLAB pads only non-square cropped ROIs; an already square
            # boundary crop is kept at its actual size.
            if tr0.shape[0] != tr0.shape[1]:
                falta_top = max(0, -(y_0b - afegit))
                falta_bottom = max(0, (y_0b + afegit + 1) - ny)
                falta_left = max(0, -(x_0b - afegit))
                falta_right = max(0, (x_0b + afegit + 1) - nx)

                minv = float(np.min(tr0))
                tr0 = np.pad(
                    tr0,
                    ((falta_top, falta_bottom), (falta_left, falta_right)),
                    mode="constant",
                    constant_values=minv,
                )

            try:
                fwhm = measure_blob_fwhm(tr0, dx_um=dx_um)
            except LocalFitFailure:
                fwhm = 0.0

            # Signal Trace Extraction & Cropping
            timesignal = extract_spatial_roi_trace(volume, mask, y_0b, x_0b, marg_x_px)
            cut, timeinterval_1b, peak_idx_rel, ot_1b, ft_1b = crop_temporal_snippet(
                timesignal, t_0b, (marg_t_before_frames, marg_t_after_frames)
            )

            t_axis_ms = timeinterval_1b.astype(np.float64) * dt_ms

            # Parameter fitting (sparkParametersFast2)
            params = compute_trace_parameters(cut, t_axis_ms, peak_idx_rel)
            _peak_in_cut = int(params.get("peak_idx_cut", peak_idx_rel))
            _peak_in_cut = max(0, min(_peak_in_cut, len(timeinterval_1b) - 1))
            pt_refined = int(timeinterval_1b[_peak_in_cut])
            confidence = compute_feature_confidence(
                timesignal=timesignal,
                timeinterval_1b=timeinterval_1b,
                candidate_pt_1b=pt,
                nt=nt,
                dist2memb_um=dist2memb,
                roi_r_um=roi_r_um,
                amp=float(params["amp"]),
                baseline=float(params["BL"]),
                tau=float(params["tau"]),
                r2=float(params["r2"]),
                ror=float(params["ror"]),
                t2p=float(params["t2p"]),
                fdhm=float(params["fdhm"]),
                tau_fit_method=str(params.get("tau_fit_method", "not_recorded")),
                decay_points=float(params.get("decay_points", 0.0)),
            )

            spk_feat = SparkFeature(
                px=px,
                py=py,
                pt=pt,
                x_0b=x_0b,
                y_0b=y_0b,
                t_0b=t_0b,
                dist2memb=dist2memb,
                fwhm=fwhm,
                timesignal=timesignal,
                timeinterval=timeinterval_1b,
                r2=params["r2"],
                amp=params["amp"],
                tau=params["tau"],
                ror=params["ror"],
                t2p=params["t2p"],
                fdhm=params["fdhm"],
                AMP=params["AMP"],
                BL=params["BL"],
                mass=params["mass"],
                tau_fit_method=str(params.get("tau_fit_method", "not_recorded")),
                tau_fit_fallback_reason=str(params.get("tau_fit_fallback_reason", "not_recorded")),
                tau_lsq=float(params.get("tau_lsq", np.nan)),
                r2_lsq=float(params.get("r2_lsq", np.nan)),
                tau_robust=float(params.get("tau_robust", np.nan)),
                r2_robust_unweighted=float(params.get("r2_robust_unweighted", np.nan)),
                decay_points=float(params.get("decay_points", 0.0)),
                decay_start_ms=float(params.get("decay_start_ms", np.nan)),
                decay_end_ms=float(params.get("decay_end_ms", np.nan)),
                pt_refined=pt_refined,
                decay_a=float(params.get("decay_a", np.nan)),
                decay_e=float(params.get("decay_e", np.nan)),
                decay_f=float(params.get("decay_f", np.nan)),
                temporal_window_truncated_start=confidence.temporal_window_truncated_start,
                temporal_window_truncated_end=confidence.temporal_window_truncated_end,
                boundary_sensitive_roi=confidence.boundary_sensitive_roi,
                dominant_peak_frame=confidence.dominant_peak_frame,
                peak_shift_frames=confidence.peak_shift_frames,
                multi_peak_count=confidence.multi_peak_count,
                shoulder_peak_count=confidence.shoulder_peak_count,
                multi_peak_or_overlapping_event=confidence.multi_peak_or_overlapping_event,
                broad_overlap_event=confidence.broad_overlap_event,
                late_secondary_peak_event=confidence.late_secondary_peak_event,
                amp_confidence=confidence.amp_confidence,
                tau_confidence=confidence.tau_confidence,
                ror_confidence=confidence.ror_confidence,
                t2p_confidence=confidence.t2p_confidence,
                fdhm_confidence=confidence.fdhm_confidence,
                peak_timing_confidence=confidence.peak_timing_confidence,
                peak_confidence=confidence.peak_confidence,
                feature_confidence_flags=confidence.feature_confidence_flags,
            )

            features_list.append(spk_feat)

    return FeatureExtractionResult(
        features=features_list,
        count=len(features_list),
        distance_map=distance_map,
    )
