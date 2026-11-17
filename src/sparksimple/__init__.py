"""sparksimple — Python port of SparkSimple2.

Ca²⁺ spark detection and characterisation pipeline.
Migrated from MATLAB (SparkSimple2), stage by stage, with regression tests
against the original MATLAB implementation.

Core pipeline stages
--------------------
Phase 1-2  I/O and metadata
    from sparksimple.config import PipelineConfig
    from sparksimple.models import ExperimentMetadata
    from sparksimple.io import load_image_stack, load_metadata

Phase 3    Baseline estimation and normalisation
    from sparksimple.preprocessing import preprocess, PreprocessingResult

Phase 4    Cell mask segmentation
    from sparksimple.preprocessing.cell_mask import cell_mask4

Phase 5    Wavelet transform (V2)
    from sparksimple.detection import compute_v2

Phase 6A-B  Merge kernel, conv3, thresholding, RegMax3D
    from sparksimple.detection import apply_3d_merge_and_threshold, detect_candidates

Phase 7    Post-detection Fo re-estimation
    from sparksimple.detection import reestimate_fo

Phase 8    Spark feature extraction
    from sparksimple.features import extract_features

Phase 9    Physiological filtering
    from sparksimple.filtering import filter_sparks

Phase 10   ROI clustering and CSV/MAT export
    from sparksimple.roi import cluster_rois
    from sparksimple.export import run_phase10

Full pipeline orchestrator
    from sparksimple.pipeline import run_pipeline, PipelineRunResult
"""

from sparksimple.pipeline import run_pipeline, PipelineRunResult

__version__ = "2.1.0"

__all__ = [
    "run_pipeline",
    "PipelineRunResult",
]
