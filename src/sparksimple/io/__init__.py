"""sparksimple.io — I/O utilities for SparkSimple2 datasets.

Public API
----------
    load_image_stack(folder, channel=None)  →  (np.ndarray, int)
    load_metadata(folder)                   →  ExperimentMetadata
"""

from sparksimple.io.tiff_loader import discover_channel_labels, load_image_stack
from sparksimple.io.metadata import (
    AcquisitionMetadataError,
    load_metadata,
    validate_stack_against_metadata,
)
from sparksimple.models import ExperimentMetadata

__all__ = [
    "load_image_stack",
    "discover_channel_labels",
    "load_metadata",
    "validate_stack_against_metadata",
    "AcquisitionMetadataError",
    "ExperimentMetadata",
]
