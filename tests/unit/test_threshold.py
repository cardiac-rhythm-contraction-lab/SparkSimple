"""Unit tests for noise threshold estimation and volume thresholding (sparksimple.detection.threshold)."""

import numpy as np
import pytest

from sparksimple.detection.threshold import (
    apply_threshold,
    estimate_noise_threshold,
    get_mask_center_pixel_coords,
)


def test_get_mask_center_pixel_coords():
    """Verify mask centroid pixel indexing matching MATLAB find() round(mean())."""
    mask = np.zeros((10, 10), dtype=np.int32)
    # Mask active region from rows 2..6 (1-based: 3..7) and cols 4..8 (1-based: 5..9)
    mask[2:7, 4:9] = 1

    # 1-based row indices: [3, 4, 5, 6, 7] -> mean = 5.0 -> round = 5 -> 0-based index 4
    # 1-based col indices: [5, 6, 7, 8, 9] -> mean = 7.0 -> round = 7 -> 0-based index 6
    y_0b, x_0b, y_1b, x_1b = get_mask_center_pixel_coords(mask)

    assert (y_0b, x_0b) == (4, 6)
    assert (y_1b, x_1b) == (5, 7)


def test_estimate_noise_threshold_synthetic():
    """Verify noise threshold computation from trace mean absolute deviation."""
    v3 = np.zeros((10, 10, 100), dtype=np.float64)
    mask = np.zeros((10, 10), dtype=np.int32)
    mask[4, 4] = 1

    # Inject Gaussian-like trace with known MAD at centroid (4, 4)
    np.random.seed(42)
    noise_trace = np.random.normal(loc=0.0, scale=1.0, size=100)
    v3[4, 4, :] = noise_trace

    expected_mad = np.mean(np.abs(noise_trace - np.mean(noise_trace)))
    expected_threshold = expected_mad * 2.22

    umbral, centroid_pixel = estimate_noise_threshold(v3, mask, multiplier=2.22)

    assert centroid_pixel == (4, 4)
    assert pytest.approx(umbral, rel=1e-12) == expected_threshold


def test_apply_threshold():
    """Verify zeroing of values below threshold."""
    v3 = np.array([[[0.1, 0.5, 1.2], [0.3, 0.9, 0.0]]], dtype=np.float64)
    umbral = 0.5

    out = apply_threshold(v3, umbral)

    expected = np.array([[[0.0, 0.5, 1.2], [0.0, 0.9, 0.0]]], dtype=np.float64)
    np.testing.assert_array_equal(out, expected)
    assert out.dtype == np.float64
