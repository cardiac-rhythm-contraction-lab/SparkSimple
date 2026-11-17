"""sparksimple.features.models — Dataclasses for extracted spark features.

Ports MATLAB output structure spkF from measureFeatures.m & sparkFeatures.m.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class SparkFeature:
    """Feature record for a single detected calcium spark candidate.

    Mirrors elements of MATLAB ``spkF`` struct array from ``measureFeatures.m``
    and ``sparkFeatures.m``.

    Attributes
    ----------
    px : int
        1-based column (X) index of candidate centroid.
    py : int
        1-based row (Y) index of candidate centroid.
    pt : int
        1-based frame (T) index of candidate centroid.
    x_0b : int
        0-based X coordinate.
    y_0b : int
        0-based Y coordinate.
    t_0b : int
        0-based T coordinate.
    dist2memb : float
        Distance to cell membrane / mask boundary in micrometres (µm).
        MATLAB: ``spkF(ii).dist2memb = md(py, px) * DX``.
    fwhm : float
        Spatial Full Width at Half Maximum in micrometres (µm).
        MATLAB: ``spkF(ii).FWHM = 2 * R``.
    timesignal : np.ndarray
        1D float64 array of extracted signal trace across all frames 1..c0.
        MATLAB: ``spkF(ii).timesignal = ts``.
    timeinterval : np.ndarray
        1D int array of 1-based frame indices [ot..ft] corresponding to the cut.
        MATLAB: ``spkF(ii).timeinterval = [ot:ft]``.
    r2 : float
        R² goodness-of-fit for the exponential decay model (NaN if fit failed).
        MATLAB: ``spkF(ii).r2``.
    amp : float
        Spark amplitude ΔF/F₀ above baseline (AMP - BL).
        MATLAB: ``spkF(ii).amp``.
    tau : float
        Decay time constant τ in milliseconds (ms).
        MATLAB: ``spkF(ii).tau``.
    ror : float
        Rate of Rise (uprise slope) in ΔF/F₀ per ms (kHzF).
        MATLAB: ``spkF(ii).ror``.
    t2p : float
        Time-to-peak in milliseconds (ms).
        MATLAB: ``spkF(ii).t2p``.
    fdhm : float
        Full Duration at Half Maximum in milliseconds (ms).
        MATLAB: ``spkF(ii).FDHM``.
    AMP : float
        Peak spark signal intensity (max ΔF/F₀).
        MATLAB: ``spkF(ii).AMP``.
    BL : float
        Pre-spark baseline signal level (ΔF/F₀).
        MATLAB: ``spkF(ii).BL``.
    mass : float
        Spark signal mass (integral over time window in ms × ΔF/F₀).
        MATLAB: ``spkF(ii).mass``.
    tau_fit_method : str
        Decay fit method selected by the Python tau fit policy.
    tau_fit_fallback_reason : str
        Reason the robust fit was not selected, or ``"none"``.
    tau_lsq : float
        Tau from bounded least-squares decay fit.
    r2_lsq : float
        Unweighted R² from bounded least-squares decay fit.
    tau_robust : float
        Tau from the MATLAB-style robust bisquare approximation.
    r2_robust_unweighted : float
        Unweighted R² from the robust approximation.
    decay_points : float
        Number of points used for decay fitting.
    decay_start_ms : float
        First time point used for decay fitting.
    decay_end_ms : float
        Last time point used for decay fitting.
    feature_confidence_flags : str
        Semicolon-separated diagnostic flags. These describe feature reliability
        and do not change spark acceptance.
    tau_confidence : str
        Confidence label for tau interpretation, usually ``"high"`` or ``"low"``.
    ror_confidence : str
        Confidence label for rate-of-rise interpretation.
    peak_confidence : str
        Confidence label for peak timing interpretation.
    """

    px: int
    py: int
    pt: int
    x_0b: int
    y_0b: int
    t_0b: int
    dist2memb: float
    fwhm: float
    timesignal: np.ndarray
    timeinterval: np.ndarray
    r2: float
    amp: float
    tau: float
    ror: float
    t2p: float
    fdhm: float
    AMP: float
    BL: float
    mass: float
    pt_refined: int = 0
    decay_a: float = float("nan")
    decay_e: float = float("nan")
    decay_f: float = float("nan")
    tau_fit_method: str = "not_attempted"
    tau_fit_fallback_reason: str = "not_recorded"
    tau_lsq: float = float("nan")
    r2_lsq: float = float("nan")
    tau_robust: float = float("nan")
    r2_robust_unweighted: float = float("nan")
    decay_points: float = 0.0
    decay_start_ms: float = float("nan")
    decay_end_ms: float = float("nan")
    temporal_window_truncated_start: bool = False
    temporal_window_truncated_end: bool = False
    boundary_sensitive_roi: bool = False
    dominant_peak_frame: int = 0
    peak_shift_frames: int = 0
    multi_peak_count: int = 0
    shoulder_peak_count: int = 0
    multi_peak_or_overlapping_event: bool = False
    broad_overlap_event: bool = False
    late_secondary_peak_event: bool = False
    amp_confidence: str = "not_evaluated"
    tau_confidence: str = "not_evaluated"
    ror_confidence: str = "not_evaluated"
    t2p_confidence: str = "not_evaluated"
    fdhm_confidence: str = "not_evaluated"
    peak_timing_confidence: str = "not_evaluated"
    peak_confidence: str = "not_evaluated"
    feature_confidence_flags: str = "not_evaluated"


@dataclass
class FeatureExtractionResult:
    """Output container for Phase 8 feature extraction pipeline.

    Attributes
    ----------
    features : list[SparkFeature]
        List of feature records for all processed candidates.
    count : int
        Number of spark candidates processed.
    distance_map : np.ndarray
        2D float64 image of Euclidean distance to cell mask boundary in µm.
    """

    features: list[SparkFeature]
    count: int
    distance_map: np.ndarray
