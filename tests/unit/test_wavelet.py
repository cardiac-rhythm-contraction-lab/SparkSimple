"""Unit tests for sparksimple.detection.wavelet."""

import numpy as np
import pytest

from sparksimple.config import DetectionConfig, PipelineConfig
from sparksimple.detection.wavelet import (
    WaveletResult,
    compute_v2,
    cwt_gaus2_1d,
    cwt_gaus2_volume,
    get_wavelet_scales,
)


class TestGetWaveletScales:
    def test_real_experiment_scales(self):
        scales = get_wavelet_scales(dt_ms=2.0, is_simulated=False)
        np.testing.assert_array_equal(scales, np.array([5, 6, 7, 8, 9, 10]))

    def test_ata_experiment_scales(self):
        # DT = 4.0 ms, range 12..48 ms -> ceil(12/4)=3, ceil(48/4)=12
        scales = get_wavelet_scales(dt_ms=4.0, input_format="ata")
        np.testing.assert_array_equal(scales, np.arange(3, 13))

    def test_custom_config_scales(self):
        cfg = DetectionConfig(
            wavelet_scales_real=(3, 7),
            wavelet_scales_sim_dt_range_ms=(10.0, 20.0),
        )
        scales_real = get_wavelet_scales(dt_ms=2.0, is_simulated=False, cfg=cfg)
        np.testing.assert_array_equal(scales_real, np.array([3, 4, 5, 6, 7]))

    def test_nonpositive_dt_raises(self):
        with pytest.raises(ValueError, match="dt_ms must be positive"):
            get_wavelet_scales(dt_ms=0.0)


class TestCwtGaus2:
    def test_1d_output_shape(self):
        sig = np.random.randn(50)
        scales = np.array([5, 6, 7])
        c = cwt_gaus2_1d(sig, scales)
        assert c.shape == (3, 50)

    def test_1d_zero_signal_gives_zeros(self):
        sig = np.zeros(40)
        scales = np.array([5, 6, 7])
        c = cwt_gaus2_1d(sig, scales)
        np.testing.assert_array_equal(c, 0.0)

    def test_1d_invalid_ndim_raises(self):
        with pytest.raises(ValueError, match="Expected 1D signal"):
            cwt_gaus2_1d(np.ones((10, 10)), np.array([5]))


class TestCwtVolume:
    def test_volume_shape_and_dtype(self):
        vol = np.random.uniform(0.0, 1.0, size=(5, 6, 30)).astype(np.float32)
        scales = np.array([5, 6, 7])
        v2 = cwt_gaus2_volume(vol, scales, filter_window_frames=3)
        assert v2.shape == (5, 6, 30)
        assert v2.dtype == np.float32

    def test_volume_matches_1d_trace_loop(self):
        ny, nx, nt = 3, 4, 40
        np.random.seed(42)
        vol = np.random.uniform(-1.0, 2.0, size=(ny, nx, nt)).astype(np.float64)
        scales = np.array([5, 6, 7])
        w = 5

        v2_fast = cwt_gaus2_volume(vol, scales, filter_window_frames=w)

        # Replicate trace-by-trace logic from sparkDetect.m lines 144-160
        v2_ref = np.zeros((ny, nx, nt), dtype=np.float32)
        from sparksimple.detection.smooth import smooth_moving_average_1d

        for y in range(ny):
            for x in range(nx):
                sig = vol[y, x, :]
                sig = smooth_moving_average_1d(sig, w)
                sig = sig - np.mean(sig)
                C = cwt_gaus2_1d(sig, scales)
                v2_ref[y, x, :] = np.sum(C, axis=0).astype(np.float32)

        np.testing.assert_allclose(v2_fast, v2_ref, rtol=1e-5, atol=1e-5)

    def test_invalid_ndim_raises(self):
        with pytest.raises(ValueError, match="Expected 3D volume"):
            cwt_gaus2_volume(np.ones((10, 10)), np.array([5]), 3)


class TestComputeV2:
    def test_compute_v2_orchestrator(self):
        vol = np.random.uniform(0.0, 2.0, size=(4, 4, 50)).astype(np.float32)
        cfg = PipelineConfig()
        res = compute_v2(vol, dt_ms=2.0, is_simulated=False, cfg=cfg)

        assert isinstance(res, WaveletResult)
        assert res.v2.shape == (4, 4, 50)
        assert res.v2.dtype == np.float32
        np.testing.assert_array_equal(res.scales, np.array([5, 6, 7, 8, 9, 10]))
        assert res.filter_window_frames == 1
        assert not res.is_simulated
        assert "WaveletResult" in repr(res)
