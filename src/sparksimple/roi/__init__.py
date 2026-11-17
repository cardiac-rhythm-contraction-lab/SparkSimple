"""sparksimple.roi — ROI clustering and per-ROI summary metric extraction.

Ports mfiles/clusterROIs.m.
"""

from sparksimple.roi.models import ROIRecord, ClusteringResult
from sparksimple.roi.cluster import cluster_rois

__all__ = [
    "ROIRecord",
    "ClusteringResult",
    "cluster_rois",
]
