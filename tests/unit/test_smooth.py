"""Unit tests for sparksimple.detection.smooth."""

import numpy as np
import pytest

from sparksimple.detection.smooth import smooth_moving_average_1d, smooth_volume_time


class TestSmooth1D:
    def test_matlab_exact_values(self):
        """Test smooth matches MATLAB output on known input vector.

        MATLAB command:
            smooth([1, 2, 5, 10, 8, 4, 2, 1], 5)
        Output:
            [1.0, 2.66666667, 5.2, 5.8, 5.8, 5.0, 2.33333333, 1.0]
        """
        x = np.array([1.0, 2.0, 5.0, 10.0, 8.0, 4.0, 2.0, 1.0], dtype=np.float64)
        expected = np.array(
            [1.0, 8.0 / 3.0, 26.0 / 5.0, 29.0 / 5.0, 29.0 / 5.0, 5.0, 7.0 / 3.0, 1.0],
            dtype=np.float64,
        )
        res = smooth_moving_average_1d(x, 5)
        np.testing.assert_allclose(res, expected, rtol=1e-12, atol=1e-12)

    def test_even_window_supported(self):
        """Test even window size behavior matches MATLAB."""
        x = np.array([1.0, 2.0, 5.0, 10.0, 8.0, 4.0, 2.0, 1.0], dtype=np.float64)
        res = smooth_moving_average_1d(x, 4)
        assert len(res) == len(x)

    def test_window_one_returns_copy(self):
        x = np.array([1.0, 5.0, 2.0], dtype=np.float32)
        res = smooth_moving_average_1d(x, 1)
        np.testing.assert_array_equal(res, x)
        assert res is not x

    def test_constant_signal_unchanged(self):
        x = np.full(20, 5.0, dtype=np.float64)
        res = smooth_moving_average_1d(x, 7)
        np.testing.assert_allclose(res, x, rtol=1e-12)

    def test_invalid_ndim_raises(self):
        with pytest.raises(ValueError, match="Expected 1D array"):
            smooth_moving_average_1d(np.ones((5, 5)), 3)


class TestSmoothVolume3D:
    def test_matches_1d_per_pixel(self):
        """Verify smooth_volume_time matches 1D per pixel across all (y, x)."""
        ny, nx, nt = 4, 5, 20
        np.random.seed(42)
        vol = np.random.uniform(0.0, 10.0, size=(ny, nx, nt)).astype(np.float64)
        w = 7

        vol_smooth = smooth_volume_time(vol, w)

        for y in range(ny):
            for x in range(nx):
                trace_1d = vol[y, x, :]
                expected_1d = smooth_moving_average_1d(trace_1d, w)
                np.testing.assert_allclose(
                    vol_smooth[y, x, :],
                    expected_1d,
                    rtol=1e-12,
                    atol=1e-12,
                    err_msg=f"Mismatch at pixel ({y}, {x})",
                )

    def test_short_temporal_dimension(self):
        vol = np.random.uniform(0.0, 5.0, size=(2, 2, 3))
        vol_smooth = smooth_volume_time(vol, 7)
        assert vol_smooth.shape == (2, 2, 3)

    def test_window_one_returns_copy(self):
        vol = np.ones((3, 3, 5), dtype=np.float32)
        vol_smooth = smooth_volume_time(vol, 1)
        np.testing.assert_array_equal(vol_smooth, vol)

    def test_invalid_ndim_raises(self):
        with pytest.raises(ValueError, match="Expected 3D volume"):
            smooth_volume_time(np.ones((5, 5)), 3)
