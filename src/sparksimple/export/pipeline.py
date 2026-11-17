"""sparksimple.export.pipeline — Phase 10 orchestrator for ROI clustering and output export.

Ties together ROI clustering (clusterROIs.m) and file export (guardaDades.m / muntaCSV.m).
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from sparksimple.roi.cluster import cluster_rois
from sparksimple.roi.models import ROIRecord, ClusteringResult
from sparksimple.export.tables import generate_accepted_csv, generate_rejected_csv
from sparksimple.export.matlab_compat import save_zdata2_mat, save_zmetadata_mat
from sparksimple.export.models import ExportPaths, ExportSummary


@dataclass
class Phase10Result:
    """Container for Phase 10 ROI clustering + export outputs."""

    clustering: ClusteringResult
    summary: ExportSummary
    paths: ExportPaths


def run_phase10(
    sparks: list[Any],
    volume: np.ndarray,
    mask: np.ndarray,
    fo_map: np.ndarray | float,
    dx_um: float,
    dt_ms: float,
    roi_r_um: float,
    output_dir: str,
    title: str,
    accepted_csv_prefix: str = "yData",
    rejected_csv_prefix: str = "zDataRejected",
    report: bool = False,
    generate_film: bool = False,
    clustering: ClusteringResult | None = None,
    execution_metadata: dict[str, Any] | None = None,
) -> Phase10Result:
    """Execute Phase 10: ROI clustering and output export.

    Args:
        sparks: List of FilteredSpark records from Phase 9.
        volume: 3D normalised volume of shape (Y, X, T).
        mask: 2D cell mask of shape (Y, X).
        fo_map: Baseline F0 map or scalar float.
        dx_um: Spatial pixel resolution in micrometres.
        dt_ms: Frame interval in milliseconds.
        roi_r_um: ROI clustering radius cutoff in micrometres.
        output_dir: Directory for output files.
        title: Experiment title used for file naming.
        accepted_csv_prefix: Prefix for accepted spark CSV filename.
        rejected_csv_prefix: Prefix for rejected spark CSV filename.

    Returns:
        Phase10Result with clustering result, summary stats, and output file paths.
    """
    _, _, temps = volume.shape
    os.makedirs(output_dir, exist_ok=True)

    # ROI clustering.  A precomputed clustering result is accepted so the
    # MATLAB-style `sp=5` entry point can redo export without rerunning ROIs.
    if clustering is None:
        clustering = cluster_rois(sparks, volume, mask, dx_um, dt_ms, roi_r_um)
    rois = clustering.rois

    # Compute summary statistics
    n_total = len(sparks)
    n_accepted = sum(1 for s in sparks if getattr(s, "good", True))
    n_rejected = n_total - n_accepted
    duration_s = max(0, temps - 1) * dt_ms / 1000.0
    in_mask_px = int(np.sum(mask == 1))
    cell_area_um2 = in_mask_px * dx_um * dx_um

    if isinstance(fo_map, np.ndarray):
        fo_in_mask = fo_map[mask == 1]
        mean_fo = float(np.mean(fo_in_mask)) if len(fo_in_mask) > 0 else 0.0
        std_fo = float(np.std(fo_in_mask, ddof=1)) if len(fo_in_mask) > 1 else 0.0
    else:
        mean_fo = float(fo_map)
        std_fo = 0.0

    spk_freq = n_accepted / duration_s if duration_s > 0 else 0.0
    spk_density_per_um2_s = (
        n_accepted / duration_s / cell_area_um2
        if (duration_s > 0 and cell_area_um2 > 0)
        else 0.0
    )
    spk_density_per_mm2_s = spk_density_per_um2_s * 1e6

    summary = ExportSummary(
        n_candidates=n_total,
        n_accepted=n_accepted,
        n_rejected=n_rejected,
        spark_freq=spk_freq,
        spark_density=spk_density_per_mm2_s,
        spark_density_per_um2_s=spk_density_per_um2_s,
        spark_density_per_mm2_s=spk_density_per_mm2_s,
        cell_area_um2=cell_area_um2,
        duration_s=duration_s,
        mean_fo=mean_fo,
        std_fo=std_fo,
    )

    # File paths
    accepted_csv_path = os.path.join(output_dir, f"{accepted_csv_prefix}_{title}.csv")
    rejected_csv_path = os.path.join(output_dir, f"{rejected_csv_prefix}_{title}.csv")
    zdata2_path = os.path.join(output_dir, "zData2.mat")
    zmetadata_path = os.path.join(output_dir, "zMetaData.mat")

    # Write CSV exports
    generate_accepted_csv(sparks, rois, fo_map, mask, dx_um, dt_ms, temps, accepted_csv_path)
    generate_rejected_csv(sparks, fo_map, mask, dx_um, dt_ms, temps, rejected_csv_path)

    # Write MAT exports
    save_zdata2_mat(zdata2_path, sparks, rois)
    save_zmetadata_mat(
        zmetadata_path, fo_map, mask, temps, dx_um, dt_ms, roi_r_um,
        execution_metadata_json=json.dumps(execution_metadata, sort_keys=True) if execution_metadata else None,
    )

    html_report_path = None
    if report:
        from sparksimple.report import generate_report
        result_tmp = Phase10Result(
            clustering=clustering,
            summary=summary,
            paths=ExportPaths(
                accepted_csv=accepted_csv_path,
                rejected_csv=rejected_csv_path,
                zdata2_mat=zdata2_path,
                zmetadata_mat=zmetadata_path,
            ),
        )
        html_path = generate_report(
            result=result_tmp,
            output_dir=output_dir,
            title=title,
            dx_um=dx_um,
            dt_ms=dt_ms,
            roi_r_um=roi_r_um,
            volume=volume,
            mask=mask,
            fo_map=fo_map if isinstance(fo_map, np.ndarray) else None,
            generate_film=generate_film,
        )
        html_report_path = str(html_path)

    paths = ExportPaths(
        accepted_csv=accepted_csv_path,
        rejected_csv=rejected_csv_path,
        zdata2_mat=zdata2_path,
        zmetadata_mat=zmetadata_path,
        html_report=html_report_path,
    )

    return Phase10Result(clustering=clustering, summary=summary, paths=paths)
