"""Feature confidence diagnostics for accepted or rejected spark candidates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks, peak_prominences


@dataclass(frozen=True)
class FeatureConfidenceDiagnostics:
    """Per-feature confidence metadata kept separate from spark acceptance."""

    temporal_window_truncated_start: bool
    temporal_window_truncated_end: bool
    boundary_sensitive_roi: bool
    dominant_peak_frame: int
    peak_shift_frames: int
    multi_peak_count: int
    shoulder_peak_count: int
    multi_peak_or_overlapping_event: bool
    broad_overlap_event: bool
    late_secondary_peak_event: bool
    amp_confidence: str
    tau_confidence: str
    ror_confidence: str
    t2p_confidence: str
    fdhm_confidence: str
    peak_timing_confidence: str
    peak_confidence: str
    feature_confidence_flags: str


def _safe_cut(timesignal: np.ndarray, timeinterval_1b: np.ndarray) -> np.ndarray:
    if timesignal.size == 0 or timeinterval_1b.size == 0:
        return np.asarray([], dtype=np.float64)
    start = max(0, int(timeinterval_1b[0]) - 1)
    end = min(len(timesignal), int(timeinterval_1b[-1]))
    return np.asarray(timesignal[start:end], dtype=np.float64)


def _dominant_peak_frame(cut: np.ndarray, timeinterval_1b: np.ndarray, candidate_pt_1b: int) -> int:
    if cut.size == 0 or timeinterval_1b.size == 0 or not np.isfinite(cut).any():
        return int(candidate_pt_1b)
    return int(timeinterval_1b[int(np.nanargmax(cut))])


def _count_relevant_peaks(cut: np.ndarray, amp: float, baseline: float) -> int:
    if cut.size < 3 or not np.isfinite(amp) or amp <= 0:
        return 0
    prominence = max(0.15 * float(amp), 0.03)
    height = float(baseline) + 0.45 * float(amp) if np.isfinite(baseline) else None
    peaks, _ = find_peaks(cut, height=height, prominence=prominence, distance=3)
    return int(len(peaks))


def _count_shoulder_peaks(cut: np.ndarray, amp: float, baseline: float) -> int:
    if cut.size < 3 or not np.isfinite(amp) or amp <= 0:
        return 0
    prominence = max(0.08 * float(amp), 0.02)
    height = float(baseline) + 0.25 * float(amp) if np.isfinite(baseline) else None
    peaks, _ = find_peaks(cut, height=height, prominence=prominence, distance=2)
    return int(len(peaks))


def _candidate_is_secondary_peak(
    cut: np.ndarray,
    peak_rel_idx: int,
    amp: float,
    baseline: float,
) -> bool:
    if cut.size < 3 or peak_rel_idx <= 0 or peak_rel_idx >= cut.size - 1:
        return False
    if not np.isfinite(amp) or amp <= 0:
        return False
    candidate_value = float(cut[peak_rel_idx])
    local_peak = candidate_value >= float(cut[peak_rel_idx - 1]) and candidate_value >= float(cut[peak_rel_idx + 1])
    high_enough = candidate_value >= float(baseline) + 0.25 * float(amp)
    if not (local_peak and high_enough):
        return False
    try:
        prominence = float(peak_prominences(cut, np.asarray([peak_rel_idx]))[0][0])
    except Exception:
        prominence = 0.0
    return prominence >= max(0.08 * float(amp), 0.02)


def _confidence_from_score(score: int) -> str:
    if score >= 2:
        return "low"
    if score == 1:
        return "caution"
    return "high"


def compute_feature_confidence(
    *,
    timesignal: np.ndarray,
    timeinterval_1b: np.ndarray,
    candidate_pt_1b: int,
    nt: int,
    dist2memb_um: float,
    roi_r_um: float,
    amp: float,
    baseline: float,
    tau: float,
    r2: float,
    ror: float,
    t2p: float,
    fdhm: float,
    tau_fit_method: str,
    decay_points: float,
) -> FeatureConfidenceDiagnostics:
    """Compute diagnostic confidence flags without changing spark acceptance."""
    timeinterval = np.asarray(timeinterval_1b, dtype=int)
    trace = np.asarray(timesignal, dtype=np.float64)
    cut = _safe_cut(trace, timeinterval)

    truncated_start = bool(timeinterval.size > 0 and int(timeinterval[0]) == 1)
    truncated_end = bool(timeinterval.size > 0 and int(timeinterval[-1]) >= int(nt))
    boundary_sensitive = bool(np.isfinite(dist2memb_um) and float(dist2memb_um) < float(roi_r_um))

    dominant_peak = _dominant_peak_frame(cut, timeinterval, candidate_pt_1b)
    peak_shift = int(dominant_peak - int(candidate_pt_1b))
    multi_peak_count = _count_relevant_peaks(cut, amp, baseline)
    shoulder_peak_count = _count_shoulder_peaks(cut, amp, baseline)
    multi_peak = multi_peak_count >= 2
    peak_rel_idx = int(candidate_pt_1b) - int(timeinterval[0]) if timeinterval.size else 0
    candidate_secondary = _candidate_is_secondary_peak(cut, peak_rel_idx, amp, baseline)
    late_secondary = bool(candidate_secondary and peak_shift <= -4)
    broad_overlap = False
    if cut.size > 0 and 0 <= peak_rel_idx < cut.size and np.isfinite(amp) and amp > 0:
        candidate_value = float(cut[peak_rel_idx])
        broad_overlap = bool(abs(peak_shift) > 3 and candidate_value >= float(baseline) + 0.75 * float(amp))
    shoulder_or_two_rise = shoulder_peak_count >= 3 or late_secondary or broad_overlap

    flags: list[str] = []
    if truncated_start:
        flags.append("temporal_window_truncated_start")
    if truncated_end:
        flags.append("temporal_window_truncated_end")
    if boundary_sensitive:
        flags.append("boundary_sensitive_roi")
    if abs(peak_shift) > 3:
        flags.append("peak_shift_suspected")
    if multi_peak:
        flags.append("multi_peak_or_overlapping_event")
    if shoulder_or_two_rise and not multi_peak:
        flags.append("shoulder_or_two_rise_suspected")
    if broad_overlap and not multi_peak:
        flags.append("broad_overlap_suspected")

    tau_hard_issue = bool(
        not np.isfinite(tau)
        or not np.isfinite(r2)
        or str(tau_fit_method) == "invalid"
        or float(decay_points) < 5
        or (truncated_end and abs(peak_shift) > 3)
        or broad_overlap
        or late_secondary
    )
    tau_soft_issue = bool(
        (np.isfinite(r2) and float(r2) < 0.5)
        or truncated_end
        or multi_peak
        or shoulder_or_two_rise
        or abs(peak_shift) >= 2
    )

    ror_score = 0
    if not np.isfinite(ror):
        ror_score += 2
    if truncated_start:
        ror_score += 1
    if multi_peak or broad_overlap or abs(peak_shift) > 3 or shoulder_peak_count >= 5:
        ror_score += 2
    elif shoulder_or_two_rise or abs(peak_shift) >= 2:
        ror_score += 1

    peak_score = 0
    if abs(peak_shift) > 3 or late_secondary:
        peak_score += 2
    elif abs(peak_shift) >= 2 or shoulder_or_two_rise:
        peak_score += 1
    if multi_peak or broad_overlap:
        peak_score += 1

    t2p_score = peak_score
    if truncated_start:
        t2p_score += 1

    fdhm_score = 0
    if not np.isfinite(fdhm):
        fdhm_score += 2
    if truncated_start or truncated_end:
        fdhm_score += 1
    if late_secondary or shoulder_peak_count >= 3:
        fdhm_score += 2
    elif multi_peak or broad_overlap or abs(peak_shift) >= 2:
        fdhm_score += 1

    amp_score = 0
    if not np.isfinite(amp) or amp <= 0:
        amp_score += 2

    amp_confidence = _confidence_from_score(amp_score)
    tau_confidence = "low" if tau_hard_issue else ("caution" if tau_soft_issue else "high")
    ror_confidence = _confidence_from_score(ror_score)
    peak_confidence = _confidence_from_score(peak_score)
    t2p_confidence = _confidence_from_score(t2p_score)
    fdhm_confidence = _confidence_from_score(fdhm_score)

    if tau_confidence == "low":
        flags.append("tau_low_confidence")
    elif tau_confidence == "caution":
        flags.append("tau_caution")
    if ror_confidence == "low":
        flags.append("ror_low_confidence")
    elif ror_confidence == "caution":
        flags.append("ror_caution")
    if peak_confidence == "low":
        flags.append("peak_timing_low_confidence")
    elif peak_confidence == "caution":
        flags.append("peak_timing_caution")
    if t2p_confidence == "low":
        flags.append("t2p_low_confidence")
    elif t2p_confidence == "caution":
        flags.append("t2p_caution")
    if fdhm_confidence == "low":
        flags.append("fdhm_low_confidence")
    elif fdhm_confidence == "caution":
        flags.append("fdhm_caution")
    if amp_confidence == "low":
        flags.append("amp_low_confidence")

    return FeatureConfidenceDiagnostics(
        temporal_window_truncated_start=truncated_start,
        temporal_window_truncated_end=truncated_end,
        boundary_sensitive_roi=boundary_sensitive,
        dominant_peak_frame=dominant_peak,
        peak_shift_frames=peak_shift,
        multi_peak_count=multi_peak_count,
        shoulder_peak_count=shoulder_peak_count,
        multi_peak_or_overlapping_event=multi_peak,
        broad_overlap_event=broad_overlap,
        late_secondary_peak_event=late_secondary,
        amp_confidence=amp_confidence,
        tau_confidence=tau_confidence,
        ror_confidence=ror_confidence,
        t2p_confidence=t2p_confidence,
        fdhm_confidence=fdhm_confidence,
        peak_timing_confidence=peak_confidence,
        peak_confidence=peak_confidence,
        feature_confidence_flags=";".join(flags) if flags else "none",
    )
