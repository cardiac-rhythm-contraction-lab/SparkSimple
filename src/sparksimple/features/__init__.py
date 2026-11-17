"""sparksimple.features — Spark feature extraction.

Ports MATLAB measureFeatures.m & sparkParametersFast2.m.
Features: px, py, pt, dist2memb, FWHM, timesignal, timeinterval,
          amp, tau, ror, t2p, FDHM, AMP, BL, mass, r2.
"""

from sparksimple.features.extract import extract_features
from sparksimple.features.models import FeatureExtractionResult, SparkFeature
from sparksimple.features.spatial import distance_to_membrane, measure_blob_fwhm
from sparksimple.features.traces import crop_temporal_snippet, extract_spatial_roi_trace
from sparksimple.features.metrics import compute_trace_parameters
from sparksimple.features.confidence import FeatureConfidenceDiagnostics, compute_feature_confidence

__all__ = [
    "SparkFeature",
    "FeatureExtractionResult",
    "FeatureConfidenceDiagnostics",
    "distance_to_membrane",
    "measure_blob_fwhm",
    "extract_spatial_roi_trace",
    "crop_temporal_snippet",
    "compute_trace_parameters",
    "compute_feature_confidence",
    "extract_features",
]
