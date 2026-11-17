"""sparksimple.export.matlab_compat — MATLAB .mat file serialization for zData2.mat and zMetaData.mat."""

from __future__ import annotations

from typing import Any
import numpy as np
import scipy.io

from sparksimple.roi.models import ROIRecord


def save_zdata2_mat(output_path: str, sparks: list[Any], rois: list[ROIRecord]) -> None:
    """Save spkF and ROIs to zData2.mat with exact MATLAB struct array layout.

    Args:
        output_path: Path to output zData2.mat file.
        sparks: List of FilteredSpark, SparkFeature, or dict records.
        rois: List of ROIRecord objects.
    """

    def get_attr(s: Any, attr: str, default: Any = 0) -> Any:
        if hasattr(s, attr):
            return getattr(s, attr)
        if hasattr(s, "feature") and hasattr(getattr(s, "feature"), attr):
            return getattr(getattr(s, "feature"), attr)
        if isinstance(s, dict):
            return s.get(attr, default)
        return default

    spk_dtype = [
        ("px", "O"),
        ("py", "O"),
        ("pt", "O"),
        ("FWHM", "O"),
        ("timesignal", "O"),
        ("timeinterval", "O"),
        ("r2", "O"),
        ("amp", "O"),
        ("tau", "O"),
        ("ror", "O"),
        ("t2p", "O"),
        ("FDHM", "O"),
        ("AMP", "O"),
        ("BL", "O"),
        ("mass", "O"),
        ("dist2memb", "O"),
        ("id", "O"),
        ("good", "O"),
        ("fail", "O"),
        ("dist2spk", "O"),
        ("roi", "O"),
        ("temporal_window_truncated_start", "O"),
        ("temporal_window_truncated_end", "O"),
        ("boundary_sensitive_roi", "O"),
        ("dominant_peak_frame", "O"),
        ("peak_shift_frames", "O"),
        ("multi_peak_count", "O"),
        ("shoulder_peak_count", "O"),
        ("multi_peak_or_overlapping_event", "O"),
        ("broad_overlap_event", "O"),
        ("late_secondary_peak_event", "O"),
        ("amp_confidence", "O"),
        ("tau_confidence", "O"),
        ("ror_confidence", "O"),
        ("t2p_confidence", "O"),
        ("fdhm_confidence", "O"),
        ("peak_timing_confidence", "O"),
        ("peak_confidence", "O"),
        ("feature_confidence_flags", "O"),
    ]

    spk_arr = np.zeros((1, len(sparks)), dtype=spk_dtype)
    for i, s in enumerate(sparks):
        good_val = 1 if getattr(s, "good", True) else 0
        fail_val = str(getattr(s, "fail", "no"))
        spk_id = int(getattr(s, "id", i + 1))

        spk_arr[0, i]["px"] = int(get_attr(s, "px"))
        spk_arr[0, i]["py"] = int(get_attr(s, "py"))
        spk_arr[0, i]["pt"] = int(get_attr(s, "pt"))

        fwhm_val = get_attr(s, "fwhm")
        if fwhm_val is None or fwhm_val == 0:
            fwhm_val = get_attr(s, "FWHM")
        spk_arr[0, i]["FWHM"] = float(fwhm_val)

        ts = get_attr(s, "timesignal")
        if isinstance(ts, np.ndarray):
            spk_arr[0, i]["timesignal"] = ts.reshape(-1, 1) if ts.ndim == 1 else ts
        else:
            spk_arr[0, i]["timesignal"] = np.array([])

        ti = get_attr(s, "timeinterval")
        if isinstance(ti, np.ndarray):
            spk_arr[0, i]["timeinterval"] = ti.reshape(1, -1) if ti.ndim == 1 else ti
        else:
            spk_arr[0, i]["timeinterval"] = np.array([])

        spk_arr[0, i]["r2"] = float(get_attr(s, "r2"))
        spk_arr[0, i]["amp"] = float(get_attr(s, "amp"))
        spk_arr[0, i]["tau"] = float(get_attr(s, "tau"))
        spk_arr[0, i]["ror"] = float(get_attr(s, "ror"))
        spk_arr[0, i]["t2p"] = float(get_attr(s, "t2p"))

        fdhm_val = get_attr(s, "fdhm")
        if fdhm_val is None or fdhm_val == 0:
            fdhm_val = get_attr(s, "FDHM")
        spk_arr[0, i]["FDHM"] = float(fdhm_val)

        spk_arr[0, i]["AMP"] = float(get_attr(s, "AMP"))

        bl_val = get_attr(s, "BL")
        if bl_val is None or bl_val == 0:
            bl_val = get_attr(s, "bl")
        spk_arr[0, i]["BL"] = float(bl_val)

        spk_arr[0, i]["mass"] = float(get_attr(s, "mass"))
        spk_arr[0, i]["dist2memb"] = float(get_attr(s, "dist2memb"))
        spk_arr[0, i]["id"] = spk_id
        spk_arr[0, i]["good"] = good_val
        spk_arr[0, i]["fail"] = fail_val
        spk_arr[0, i]["dist2spk"] = float(get_attr(s, "dist2spk", 0.0))
        spk_arr[0, i]["roi"] = int(get_attr(s, "roi", 0))
        spk_arr[0, i]["temporal_window_truncated_start"] = int(bool(get_attr(s, "temporal_window_truncated_start", False)))
        spk_arr[0, i]["temporal_window_truncated_end"] = int(bool(get_attr(s, "temporal_window_truncated_end", False)))
        spk_arr[0, i]["boundary_sensitive_roi"] = int(bool(get_attr(s, "boundary_sensitive_roi", False)))
        spk_arr[0, i]["dominant_peak_frame"] = int(get_attr(s, "dominant_peak_frame", 0))
        spk_arr[0, i]["peak_shift_frames"] = int(get_attr(s, "peak_shift_frames", 0))
        spk_arr[0, i]["multi_peak_count"] = int(get_attr(s, "multi_peak_count", 0))
        spk_arr[0, i]["shoulder_peak_count"] = int(get_attr(s, "shoulder_peak_count", 0))
        spk_arr[0, i]["multi_peak_or_overlapping_event"] = int(bool(get_attr(s, "multi_peak_or_overlapping_event", False)))
        spk_arr[0, i]["broad_overlap_event"] = int(bool(get_attr(s, "broad_overlap_event", False)))
        spk_arr[0, i]["late_secondary_peak_event"] = int(bool(get_attr(s, "late_secondary_peak_event", False)))
        spk_arr[0, i]["amp_confidence"] = str(get_attr(s, "amp_confidence", "not_evaluated"))
        spk_arr[0, i]["tau_confidence"] = str(get_attr(s, "tau_confidence", "not_evaluated"))
        spk_arr[0, i]["ror_confidence"] = str(get_attr(s, "ror_confidence", "not_evaluated"))
        spk_arr[0, i]["t2p_confidence"] = str(get_attr(s, "t2p_confidence", "not_evaluated"))
        spk_arr[0, i]["fdhm_confidence"] = str(get_attr(s, "fdhm_confidence", "not_evaluated"))
        spk_arr[0, i]["peak_timing_confidence"] = str(get_attr(s, "peak_timing_confidence", "not_evaluated"))
        spk_arr[0, i]["peak_confidence"] = str(get_attr(s, "peak_confidence", "not_evaluated"))
        spk_arr[0, i]["feature_confidence_flags"] = str(get_attr(s, "feature_confidence_flags", "not_evaluated"))

    roi_dtype = [
        ("signal", "O"),
        ("px", "O"),
        ("py", "O"),
        ("Nsparks", "O"),
        ("FWHM", "O"),
        ("r2", "O"),
        ("amp", "O"),
        ("tau", "O"),
        ("ror", "O"),
        ("t2p", "O"),
        ("FDHM", "O"),
        ("AMP", "O"),
        ("BL", "O"),
        ("dist2memb", "O"),
        ("mass", "O"),
        ("id", "O"),
        ("Nspk", "O"),
        ("dist2closest", "O"),
    ]

    if len(rois) > 0:
        rois_arr = np.zeros((1, len(rois)), dtype=roi_dtype)
        for i, r in enumerate(rois):
            sig = r.signal
            rois_arr[0, i]["signal"] = sig.reshape(-1, 1) if sig.ndim == 1 else sig
            rois_arr[0, i]["px"] = int(r.px)
            rois_arr[0, i]["py"] = int(r.py)
            rois_arr[0, i]["Nsparks"] = int(r.Nsparks)
            rois_arr[0, i]["FWHM"] = float(r.FWHM)
            rois_arr[0, i]["r2"] = float(r.r2)
            rois_arr[0, i]["amp"] = float(r.amp)
            rois_arr[0, i]["tau"] = float(r.tau)
            rois_arr[0, i]["ror"] = float(r.ror)
            rois_arr[0, i]["t2p"] = float(r.t2p)
            rois_arr[0, i]["FDHM"] = float(r.FDHM)
            rois_arr[0, i]["AMP"] = float(r.AMP)
            rois_arr[0, i]["BL"] = float(r.BL)
            rois_arr[0, i]["dist2memb"] = float(r.dist2memb)
            rois_arr[0, i]["mass"] = float(r.mass)
            rois_arr[0, i]["id"] = int(r.id)
            rois_arr[0, i]["Nspk"] = int(r.Nspk)
            rois_arr[0, i]["dist2closest"] = float(r.dist2closest)
    else:
        rois_arr = np.zeros((1, 0), dtype=roi_dtype)

    scipy.io.savemat(output_path, {"spkF": spk_arr, "ROIs": rois_arr})


def save_zmetadata_mat(
    output_path: str,
    fo: np.ndarray | float,
    mask: np.ndarray,
    temps: int,
    dx_um: float,
    dt_ms: float,
    roi_r_um: float,
    execution_metadata_json: str | None = None,
) -> None:
    """Save experiment parameters and baseline metadata to zMetaData.mat."""
    payload = {
            "fo": fo,
            "mask": mask,
            "temps": float(temps),
            "DX": float(dx_um),
            "DT": float(dt_ms),
            "roiR": float(roi_r_um),
    }
    if execution_metadata_json is not None:
        payload["SparkSimpleRunMetadataJSON"] = execution_metadata_json
    scipy.io.savemat(output_path, payload)
