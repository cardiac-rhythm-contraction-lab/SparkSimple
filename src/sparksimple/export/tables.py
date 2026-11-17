"""sparksimple.export.tables — CSV table formatting for accepted and rejected spark exports.

Ports mfiles/guardaDades.m & mfiles/muntaCSV.m.
"""

from __future__ import annotations

from typing import Any
import numpy as np
from sparksimple.roi.models import ROIRecord


def _format_val(v: Any) -> str:
    """Format numerical value or string for CSV cell matching MATLAB muntaCSV.m output."""
    if v is None:
        return ""
    if isinstance(v, (int, np.integer)):
        return str(v)
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return "NaN"
        if np.isinf(v):
            return "Inf"
        # Format floats with compact general representation
        g_str = f"{v:.6g}"
        return g_str
    return str(v)


def _format_csv_row(cols: list[Any]) -> str:
    """Format a list of column values into a comma-terminated CSV line."""
    return ",".join(_format_val(c) for c in cols) + ",\n"


def generate_accepted_csv(
    sparks: list[Any],
    rois: list[ROIRecord],
    fo_map: np.ndarray | float,
    mask: np.ndarray,
    dx_um: float,
    dt_ms: float,
    temps: int,
    output_path: str,
) -> None:
    """Generate the main yData_<Title>.csv output file matching MATLAB guardaDades.m & muntaCSV.m.

    Args:
        sparks: List of spark records (FilteredSpark, SparkFeature, or dict).
        rois: List of ROIRecord objects from cluster_rois.
        fo_map: Baseline F0 map or scalar.
        mask: 2D cell mask array.
        dx_um: Spatial pixel resolution in um.
        dt_ms: Frame rate in ms.
        temps: Total number of time frames.
        output_path: Output file path for CSV.
    """

    def get_attr(s: Any, attr: str, default: Any = 0) -> Any:
        if hasattr(s, attr):
            return getattr(s, attr)
        if hasattr(s, "feature") and hasattr(getattr(s, "feature"), attr):
            return getattr(getattr(s, "feature"), attr)
        if isinstance(s, dict):
            return s.get(attr, default)
        return default

    # Partition accepted sparks and sort by pt candidate frame index
    accepted_sparks = [s for s in sparks if get_attr(s, "good", True)]

    # Sort accepted sparks by pt frame index
    accepted_sparks.sort(key=lambda s: int(get_attr(s, "pt")))

    n_candidates = len(sparks)
    n_accepted = len(accepted_sparks)
    duration_s = max(0, temps - 1) * dt_ms / 1000.0

    in_mask_pixels = int(np.sum(mask == 1))
    cell_area_um2 = in_mask_pixels * dx_um * dx_um

    if isinstance(fo_map, np.ndarray):
        fo_in_mask = fo_map[mask == 1]
        mean_fo = float(np.mean(fo_in_mask)) if len(fo_in_mask) > 0 else float(np.mean(fo_map))
        std_fo = float(np.std(fo_in_mask, ddof=1)) if len(fo_in_mask) > 1 else 0.0
    else:
        mean_fo = float(fo_map)
        std_fo = 0.0

    spk_freq = n_accepted / duration_s if duration_s > 0 else 0.0
    spk_dens_um2 = n_accepted / duration_s / cell_area_um2 if (duration_s > 0 and cell_area_um2 > 0) else 0.0
    spk_dens_mm2 = spk_dens_um2 * 1e6

    lines = []

    # Blank line
    lines.append(",\n")

    # Header metadata labels (offset by 18 commas in MATLAB layout)
    meta_header_prefix = "," * 18
    lines.append(
        meta_header_prefix
        + "Pixel size(um),Frame rate(ms),Total duration(s),Cell area(um&sup2),Cell Fo,Spark candidates,Surviving sparks,Spk Freq (spk/s),spark_density_per_um2_s,spark_density_per_mm2_s,\n"
    )
    lines.append(",\n")

    # Metadata values
    fo_str = f"{mean_fo:.1f} \xbf {std_fo:.1f}"
    lines.append(
        meta_header_prefix
        + f"{dx_um},{dt_ms},{duration_s:.1f},{cell_area_um2:.0f},{fo_str},{n_candidates},{n_accepted},{spk_freq:.4f},{spk_dens_um2:.7g},{spk_dens_mm2:.7g},\n"
    )
    lines.append(",\n")

    # Section 2: Mean values & STDs
    lines.append("mean values & stds,\n")
    lines.append(",\n")

    # Build matrix of dades for accepted sparks to compute column means and stds
    # Columns in dades: [id, px, py, pt, pT, dist2memb, dist2spk, AMP, amp, BL, ror, mass, t2p, FDHM, FWHM, tau, fldr, subfldr, roi]
    dades_rows = []
    for s in accepted_sparks:
        fwhm_v = get_attr(s, "fwhm")
        if fwhm_v is None or fwhm_v == 0:
            fwhm_v = get_attr(s, "FWHM")
        fdhm_v = get_attr(s, "fdhm")
        if fdhm_v is None or fdhm_v == 0:
            fdhm_v = get_attr(s, "FDHM")
        bl_v = get_attr(s, "BL")
        if bl_v is None or bl_v == 0:
            bl_v = get_attr(s, "bl")

        spk_id = int(getattr(s, "id", 0))
        pt_val = int(get_attr(s, "pt"))

        row = [
            spk_id,
            int(get_attr(s, "px")),
            int(get_attr(s, "py")),
            pt_val,
            pt_val,  # pT
            float(get_attr(s, "dist2memb")),
            float(get_attr(s, "dist2spk", 0.0)),
            float(get_attr(s, "AMP")),
            float(get_attr(s, "amp")),
            float(bl_v),
            float(get_attr(s, "ror")),
            float(get_attr(s, "mass")),
            float(get_attr(s, "t2p")),
            float(fdhm_v),
            float(fwhm_v),
            float(get_attr(s, "tau")),
            1,  # fldr
            1,  # subfldr
            int(get_attr(s, "roi", 0)),
        ]
        dades_rows.append(row)

    if len(dades_rows) > 0:
        dades_arr = np.array(dades_rows, dtype=np.float64)
        means = np.nanmean(dades_arr, axis=0)
        stds = np.nanstd(dades_arr, axis=0, ddof=1) if len(dades_rows) > 1 else np.zeros_like(means)
        means[:4] = 0.0
        stds[:4] = 0.0
        lines.append(_format_csv_row(list(means)))
        lines.append(_format_csv_row(list(stds)))
    else:
        lines.append(_format_csv_row([0] * 19))
        lines.append(_format_csv_row([0] * 19))

    lines.append(",\n")

    # Table Header for Accepted Sparks (no trailing comma — matches MATLAB muntaCSV string arg)
    lines.append("ID,Xpix,Ypix,Tframe,Tframe,dist2memb,dist2closest,AMP,amp,BL,RoR,mass,t2p,FDHM,FWHM,tau,fldr,subfldr,ROI\n")
    # MATLAB muntaCSV.m inserts a sep line between each argument (string or matrix)
    lines.append(",\n")

    for row in dades_rows:
        lines.append(_format_csv_row(row))

    lines.append(",\n")

    # Section 3: ROI Data (string labels have no trailing comma in MATLAB format)
    lines.append("Roi Data\n")
    lines.append(",\n")
    lines.append("ID,Xpix,Ypix,Tframe,dist2memb,dist2closest,AMP,amp,BL,RoR,mass,t2p,FDHM,FWHM,tau,#sparks\n")
    # Separator between ROI header string and ROI data matrix
    lines.append(",\n")

    for r in rois:
        r_row = [
            r.id,
            r.px,
            r.py,
            0,
            r.dist2memb,
            r.dist2closest,
            r.AMP,
            r.amp,
            r.BL,
            r.ror,
            r.mass,
            r.t2p,
            r.FDHM,
            r.FWHM,
            r.tau,
            r.Nspk,
        ]
        lines.append(_format_csv_row(r_row))

    lines.append(",\n")

    # Section 4: ROI time signal (label is a plain string, no trailing comma)
    lines.append("ROI time signal\n")
    # headr2 in MATLAB is built by concatenating 'time(ms),' + 'ROI1,' + ... so it ends with ','
    # _format_csv_row appends a comma after each element, matching MATLAB's headr2 construction
    roi_headers = ["time(ms)"] + [f"ROI{r.id}" for r in rois]
    lines.append(_format_csv_row(roi_headers))

    if len(rois) > 0:
        signal_length = len(rois[0].signal)
        for t_step in range(signal_length):
            t_ms = t_step * dt_ms
            row_vals = [t_ms] + [float(r.signal[t_step]) for r in rois]
            lines.append(_format_csv_row(row_vals))

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def generate_rejected_csv(
    sparks: list[Any],
    fo_map: np.ndarray | float,
    mask: np.ndarray,
    dx_um: float,
    dt_ms: float,
    temps: int,
    output_path: str,
) -> None:
    """Generate zDataRejected_<Title>.csv output file matching MATLAB guardaDades.m & muntaCSV.m.

    Args:
        sparks: List of spark records.
        fo_map: Baseline F0 map or scalar.
        mask: 2D cell mask array.
        dx_um: Spatial pixel resolution in um.
        dt_ms: Frame rate in ms.
        temps: Total number of time frames.
        output_path: Output file path for CSV.
    """

    def get_attr(s: Any, attr: str, default: Any = 0) -> Any:
        if hasattr(s, attr):
            return getattr(s, attr)
        if hasattr(s, "feature") and hasattr(getattr(s, "feature"), attr):
            return getattr(getattr(s, "feature"), attr)
        if isinstance(s, dict):
            return s.get(attr, default)
        return default

    rejected_sparks = [s for s in sparks if not get_attr(s, "good", True)]
    rejected_sparks.sort(key=lambda s: int(get_attr(s, "pt")))

    n_candidates = len(sparks)
    n_rejected = len(rejected_sparks)
    duration_s = max(0, temps - 1) * dt_ms / 1000.0

    in_mask_pixels = int(np.sum(mask == 1))
    cell_area_um2 = in_mask_pixels * dx_um * dx_um

    if isinstance(fo_map, np.ndarray):
        fo_in_mask = fo_map[mask == 1]
        mean_fo = float(np.mean(fo_in_mask)) if len(fo_in_mask) > 0 else float(np.mean(fo_map))
        std_fo = float(np.std(fo_in_mask, ddof=1)) if len(fo_in_mask) > 1 else 0.0
    else:
        mean_fo = float(fo_map)
        std_fo = 0.0

    freq_rejected = n_rejected / duration_s if duration_s > 0 else 0.0
    dens_rejected = n_rejected / duration_s / cell_area_um2 if (duration_s > 0 and cell_area_um2 > 0) else 0.0

    lines = []
    lines.append(",\n")

    meta_header_prefix = "," * 18
    lines.append(
        meta_header_prefix
        + "Pixel size(um),Frame rate(ms),Total duration(s),Cell area(um&sup2),Cell Fo,Spark candidates,Rejected sparks,Rejected spark frequency(spk/s),Rejected spark density(spk/s/um²),\n"
    )
    lines.append(",\n")

    fo_str = f"{mean_fo:.1f} \xbf {std_fo:.1f}"
    lines.append(
        meta_header_prefix
        + f"{dx_um},{dt_ms},{duration_s:.1f},{cell_area_um2:.0f},{fo_str},{n_candidates},{n_rejected},{freq_rejected:.4f},{dens_rejected:.7f},\n"
    )
    lines.append(",\n")

    # Header string — no trailing comma in MATLAB format
    lines.append("ID,Xpix,Ypix,Tframe,Tframe,dist2memb,AMP,amp,BL,RoR,mass,t2p,FDHM,FWHM,tau\n")

    for s in rejected_sparks:
        fwhm_v = get_attr(s, "fwhm")
        if fwhm_v is None or fwhm_v == 0:
            fwhm_v = get_attr(s, "FWHM")
        fdhm_v = get_attr(s, "fdhm")
        if fdhm_v is None or fdhm_v == 0:
            fdhm_v = get_attr(s, "FDHM")
        bl_v = get_attr(s, "BL")
        if bl_v is None or bl_v == 0:
            bl_v = get_attr(s, "bl")

        spk_id = int(getattr(s, "id", 0))
        pt_val = int(get_attr(s, "pt"))

        row = [
            spk_id,
            int(get_attr(s, "px")),
            int(get_attr(s, "py")),
            pt_val,
            pt_val,  # pT
            float(get_attr(s, "dist2memb")),
            float(get_attr(s, "AMP")),
            float(get_attr(s, "amp")),
            float(bl_v),
            float(get_attr(s, "ror")),
            float(get_attr(s, "mass")),
            float(get_attr(s, "t2p")),
            float(fdhm_v),
            float(fwhm_v),
            float(get_attr(s, "tau")),
        ]
        lines.append(_format_csv_row(row))

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
