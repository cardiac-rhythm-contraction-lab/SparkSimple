"""Unit tests for 2D and 3D merge kernel generation (sparksimple.detection.kernel)."""

import math
import numpy as np
import pytest

from sparksimple.detection.kernel import build_gaussian_2d, build_merge_kernel_3d


def test_build_gaussian_2d_shape_and_symmetry():
    """Verify shape, symmetry, and peak location of 2D Gaussian kernel."""
    n = 5
    b = build_gaussian_2d(n)

    assert b.shape == (5, 5)
    assert b.dtype == np.float64

    # Peak should be at (2, 2) for N=5 (1-based center = ceil(5/2) = 3 -> 0-based index 2)
    max_pos = np.unravel_index(np.argmax(b), b.shape)
    assert max_pos == (2, 2)

    # Symmetry checks
    np.testing.assert_allclose(b, b.T)  # Square symmetry
    np.testing.assert_allclose(b, np.flipud(b))  # Vertical symmetry
    np.testing.assert_allclose(b, np.fliplr(b))  # Horizontal symmetry


def test_build_gaussian_2d_even_size():
    """Verify even-sized 2D Gaussian kernel generation."""
    n = 6
    b = build_gaussian_2d(n)

    assert b.shape == (6, 6)
    # 1-based center is ceil(6/2) = 3 -> 0-based index 2
    assert b[2, 2] > b[1, 1]
    assert b[2, 2] > b[3, 3]


def test_build_gaussian_2d_invalid_size():
    with pytest.raises(ValueError, match="Kernel size n must be positive"):
        build_gaussian_2d(0)


def test_build_merge_kernel_3d_shape_and_normalisation():
    """Verify 3D kernel dimensions and sum normalisation."""
    merge_distance_um = 2.0
    merge_time_ms = 10.0
    dx_um = 0.5
    dt_ms = 2.0

    # t3 = ceil(2 * 2.0 / 0.5) = 8
    # w3 = ceil(2 * 10.0 / 2.0) = 10
    k3d = build_merge_kernel_3d(
        merge_distance_um=merge_distance_um,
        merge_time_ms=merge_time_ms,
        dx_um=dx_um,
        dt_ms=dt_ms,
    )

    assert k3d.shape == (8, 8, 10)
    assert k3d.dtype == np.float64
    assert math.isclose(np.sum(k3d), 1.0, abs_tol=1e-12)

    # Every slice along temporal axis (dim 2) should be identical before normalisation (and after scaling)
    for frame_idx in range(1, 10):
        np.testing.assert_allclose(k3d[:, :, frame_idx], k3d[:, :, 0])


def test_build_merge_kernel_3d_invalid_args():
    with pytest.raises(ValueError):
        build_merge_kernel_3d(0, 10, 0.5, 2.0)
    with pytest.raises(ValueError):
        build_merge_kernel_3d(2.0, 10, -1.0, 2.0)
