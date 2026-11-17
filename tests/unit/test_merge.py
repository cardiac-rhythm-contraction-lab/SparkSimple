"""Unit tests for Phase 6A orchestrator apply_3d_merge_and_threshold (sparksimple.detection.merge)."""

import numpy as np
import pytest

from sparksimple.config import DetectionConfig
from sparksimple.detection.merge import MergeResult, apply_3d_merge_and_threshold


def test_apply_3d_merge_and_threshold_full_flow():
    """Verify full end-to-end flow of Phase 6A pipeline on synthetic volume."""
    v2 = np.ones((20, 20, 30), dtype=np.float64) * 0.1
    # Add a spark-like candidate peak
    v2[10, 10, 15] = 5.0

    mask = np.zeros((20, 20), dtype=np.int32)
    mask[5:15, 5:15] = 1

    cfg = DetectionConfig(merge_distance_um=2.0, merge_time_ms=10.0)

    res = apply_3d_merge_and_threshold(
        v2=v2,
        mask=mask,
        dx_um=0.5,
        dt_ms=2.0,
        cfg=cfg,
    )

    assert isinstance(res, MergeResult)
    assert res.v3_conv.shape == (20, 20, 30)
    assert res.v3_thresholded.shape == (20, 20, 30)
    assert res.kernel.shape == (8, 8, 10)
    assert res.umbral > 0.0

    # Spark peak region should remain non-zero in thresholded volume
    assert res.v3_thresholded[10, 10, 15] > 0.0


def test_apply_3d_merge_and_threshold_missing_config():
    v2 = np.zeros((10, 10, 10))
    mask = np.ones((10, 10))
    with pytest.raises(ValueError, match="must be provided"):
        apply_3d_merge_and_threshold(v2, mask, dx_um=0.5, dt_ms=2.0)
