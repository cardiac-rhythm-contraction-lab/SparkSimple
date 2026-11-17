"""Data models for file export summary and paths."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ExportSummary:
    """Summary statistics for exported experiment results."""

    n_candidates: int
    n_accepted: int
    n_rejected: int
    spark_freq: float  # spk / s
    spark_density: float  # Backward-compatible alias: accepted spk / s / mm^2
    cell_area_um2: float
    duration_s: float
    mean_fo: float
    std_fo: float
    spark_density_per_um2_s: float = 0.0
    spark_density_per_mm2_s: float = 0.0


@dataclass
class ExportPaths:
    """Paths to exported output files."""

    accepted_csv: str
    rejected_csv: str
    zdata2_mat: str
    zmetadata_mat: str
    html_report: Optional[str] = None
    execution_metadata: Optional[str] = None
