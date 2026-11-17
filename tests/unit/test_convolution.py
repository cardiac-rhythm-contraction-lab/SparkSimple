"""Unit tests for conv3_matlab 3D FFT convolution (sparksimple.detection.convolution)."""

import numpy as np
import pytest

from sparksimple.detection.convolution import conv3_matlab


def test_conv3_matlab_shape_and_delta_response():
    """Verify output shape and response to a 3D Dirac delta impulse."""
    a = np.zeros((20, 20, 30), dtype=np.float64)
    # Impulse at center
    a[10, 10, 15] = 1.0

    b = np.ones((3, 3, 5), dtype=np.float64) / 45.0

    out = conv3_matlab(a, b)

    assert out.shape == (20, 20, 30)
    assert out.dtype == np.float64

    # Peak of convolved response should be at (9, 9, 13) matching MATLAB conv3.m padding behavior
    max_pos = np.unravel_index(np.argmax(out), out.shape)
    assert max_pos == (9, 9, 13)
    assert pytest.approx(np.sum(out), rel=1e-12) == 1.0


def test_conv3_matlab_invalid_inputs():
    a_2d = np.zeros((10, 10))
    b_3d = np.zeros((3, 3, 3))
    with pytest.raises(ValueError, match="Both A and B must be 3D arrays"):
        conv3_matlab(a_2d, b_3d)

    a_small = np.zeros((2, 2, 2))
    b_large = np.zeros((5, 5, 5))
    with pytest.raises(ValueError, match="must be >= kernel size"):
        conv3_matlab(a_small, b_large)
