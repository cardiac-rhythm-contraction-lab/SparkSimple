"""Unit test for top-level extract_features orchestrator."""

import numpy as np
import pytest

from sparksimple.features import SparkFeature, extract_features


def test_extract_features_end_to_end():
    """Verify extract_features on synthetic volume and candidate IX matrix."""
    ny, nx, nt = 32, 32, 60
    volume = np.zeros((ny, nx, nt), dtype=np.float64)

    # Place a synthetic spark at (16, 16, 30)
    yy, xx, tt = np.ogrid[:ny, :nx, :nt]
    spark = 0.8 * np.exp(-((yy - 16) ** 2 + (xx - 16) ** 2) / 4.0) * np.exp(-((tt - 30) ** 2) / 100.0)
    volume += spark

    mask = np.ones((ny, nx), dtype=np.float64)

    # IX matrix marking candidate at (16, 16, 30)
    ix_matrix = np.zeros((ny, nx, nt), dtype=np.float64)
    ix_matrix[16, 16, 30] = 50.0

    res = extract_features(
        volume=volume,
        ix_matrix=ix_matrix,
        mask=mask,
        dx_um=0.5,
        dt_ms=2.0,
        roi_r_um=1.5,
        marg_t_ms=(40.0, 40.0),
    )

    assert res.count == 1
    feat = res.features[0]

    assert isinstance(feat, SparkFeature)
    assert feat.x_0b == 16
    assert feat.y_0b == 16
    assert feat.t_0b == 30
    assert feat.px == 17
    assert feat.py == 17
    assert feat.pt == 31
    assert feat.amp > 0.0
    assert feat.fwhm > 0.0
    assert len(feat.timesignal) == nt
    assert feat.feature_confidence_flags != "not_evaluated"
    assert feat.amp_confidence in {"high", "caution", "low"}
    assert feat.tau_confidence in {"high", "caution", "low"}
    assert feat.ror_confidence in {"high", "caution", "low"}
    assert feat.t2p_confidence in {"high", "caution", "low"}
    assert feat.fdhm_confidence in {"high", "caution", "low"}
    assert feat.peak_timing_confidence in {"high", "caution", "low"}


def test_extract_features_flags_temporal_edges_and_boundary():
    """Feature confidence flags are diagnostic and do not reject the spark."""
    ny, nx, nt = 16, 16, 20
    volume = np.zeros((ny, nx, nt), dtype=np.float64)
    volume[2, 2, 0] = 1.0
    volume[2, 2, 4] = 0.7

    mask = np.ones((ny, nx), dtype=np.float64)
    mask[:2, :] = 0
    mask[:, :2] = 0

    ix_matrix = np.zeros((ny, nx, nt), dtype=np.float64)
    ix_matrix[2, 2, 0] = 10.0

    res = extract_features(
        volume=volume,
        ix_matrix=ix_matrix,
        mask=mask,
        dx_um=0.5,
        dt_ms=10.0,
        roi_r_um=1.5,
        marg_t_ms=(160.0, 320.0),
    )

    feat = res.features[0]
    assert feat.temporal_window_truncated_start is True
    assert feat.boundary_sensitive_roi is True
    assert "temporal_window_truncated_start" in feat.feature_confidence_flags
    assert "boundary_sensitive_roi" in feat.feature_confidence_flags
    assert feat.ror_confidence in {"caution", "low"}
