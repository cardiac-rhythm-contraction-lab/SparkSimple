"""Unit tests for sparksimple.preprocessing.spatial_filter."""

import math

import numpy as np
import pytest

from sparksimple.config import PipelineConfig
from sparksimple.preprocessing.pipeline import preprocess
from sparksimple.preprocessing.spatial_filter import (
    apply_spatial_smoothing,
    gaussiana2d,
    matlab_round,
    spatial_kernel_size_px,
)


def gaussiana2d_reference(n: int) -> np.ndarray:
    """Literal transcription of mfiles/gaussiana2d.m, used as test oracle."""
    b = np.zeros((n, n), dtype=np.float64)
    sig1 = n * 0.2
    sig2 = n * 0.2
    centre = math.ceil(n / 2)
    for ii in range(1, n + 1):
        for jj in range(1, n + 1):
            b[ii - 1, jj - 1] = math.sqrt(
                math.exp(-((ii - centre) ** 2) / sig1 / sig1)
                * math.exp(-((jj - centre) ** 2) / sig2 / sig2)
            )
    return b


class TestMatlabRound:
    def test_half_rounds_away_from_zero(self):
        """MATLAB round(0.5)=1 and round(2.5)=3, unlike NumPy's round-half-to-even."""
        assert matlab_round(0.5) == 1
        assert matlab_round(2.5) == 3
        assert matlab_round(-0.5) == -1
        assert matlab_round(-2.5) == -3

    def test_differs_from_numpy_on_halves(self):
        assert matlab_round(2.5) != int(np.round(2.5))

    def test_ordinary_values(self):
        assert matlab_round(4.4) == 4
        assert matlab_round(4.6) == 5
        assert matlab_round(0.0) == 0


class TestGaussiana2d:
    def test_unit_kernel_is_identity_element(self):
        """gaussiana2d(1) must be exactly [[1.0]] — this is what makes fs=0 a no-op."""
        np.testing.assert_array_equal(gaussiana2d(1), np.array([[1.0]]))

    @pytest.mark.parametrize("n", [1, 3, 5, 7, 9, 11])
    def test_matches_matlab_reference(self, n):
        np.testing.assert_allclose(
            gaussiana2d(n), gaussiana2d_reference(n), rtol=1e-14, atol=1e-14
        )

    def test_peak_at_matlab_centre(self):
        """MATLAB centre is ceil(N/2) in 1-based indexing."""
        n = 7
        kernel = gaussiana2d(n)
        centre0 = math.ceil(n / 2) - 1
        assert kernel[centre0, centre0] == pytest.approx(1.0)
        assert kernel.max() == pytest.approx(1.0)

    def test_not_normalised(self):
        """The MATLAB kernel has no division by its sum; the gain cancels in dF/F0."""
        assert gaussiana2d(7).sum() > 1.0

    def test_rejects_zero_size(self):
        with pytest.raises(ValueError, match="must be >= 1"):
            gaussiana2d(0)


class TestSpatialKernelSizePx:
    def test_zero_um_gives_unit_kernel(self):
        """MATLAB fs=0 -> round(0)=0 -> 0+mod(0,2)+1 = 1."""
        assert spatial_kernel_size_px(0.0, 0.1) == 1

    def test_matlab_arithmetic(self):
        """fs=0.5um, DX=0.1um -> round(5)=5 -> 5+mod(5,2)+1 = 7."""
        assert spatial_kernel_size_px(0.5, 0.1) == 7

    def test_even_pixel_count_becomes_odd(self):
        """fs=0.4um, DX=0.1um -> round(4)=4 -> 4+mod(4,2)+1 = 5."""
        assert spatial_kernel_size_px(0.4, 0.1) == 5

    @pytest.mark.parametrize("fs_um", [0.0, 0.1, 0.25, 0.4, 0.5, 1.0, 2.0])
    def test_always_odd(self, fs_um):
        assert spatial_kernel_size_px(fs_um, 0.1) % 2 == 1

    def test_rejects_negative_width(self):
        with pytest.raises(ValueError, match="spatial_smooth_um must be >= 0"):
            spatial_kernel_size_px(-1.0, 0.1)

    def test_rejects_non_positive_dx(self):
        with pytest.raises(ValueError, match="dx_um must be > 0"):
            spatial_kernel_size_px(1.0, 0.0)


class TestApplySpatialSmoothing:
    def test_zero_width_is_exact_identity(self):
        """The MATLAB default fs=0 must leave the stack bit-for-bit unchanged."""
        rng = np.random.default_rng(0)
        stack = rng.random((6, 5, 4)).astype(np.float32)
        out, kernel_px = apply_spatial_smoothing(stack, 0.0, 0.1)
        assert kernel_px == 1
        np.testing.assert_array_equal(out, stack)
        assert out.dtype == stack.dtype

    def test_smoothing_changes_values_and_preserves_shape(self):
        rng = np.random.default_rng(1)
        stack = rng.random((8, 7, 3)).astype(np.float64)
        out, kernel_px = apply_spatial_smoothing(stack, 0.5, 0.1)
        assert kernel_px == 7
        assert out.shape == stack.shape
        assert not np.allclose(out, stack)

    def test_matches_scipy_conv2_same_per_frame(self):
        """Every frame must equal conv2(frame, GF, 'same') with zero padding."""
        from scipy.signal import convolve2d

        rng = np.random.default_rng(2)
        stack = rng.random((9, 8, 2))
        out, kernel_px = apply_spatial_smoothing(stack, 0.3, 0.1)
        kernel = gaussiana2d(kernel_px)
        for t in range(stack.shape[2]):
            expected = convolve2d(
                stack[:, :, t], kernel, mode="same", boundary="fill", fillvalue=0.0
            )
            np.testing.assert_allclose(out[:, :, t], expected, rtol=1e-12, atol=1e-12)

    def test_impulse_response_reproduces_kernel(self):
        """A centred impulse must convolve to the kernel itself."""
        stack = np.zeros((11, 11, 1))
        stack[5, 5, 0] = 1.0
        out, kernel_px = apply_spatial_smoothing(stack, 0.5, 0.1)
        kernel = gaussiana2d(kernel_px)
        half = kernel_px // 2
        window = out[5 - half : 5 + half + 1, 5 - half : 5 + half + 1, 0]
        np.testing.assert_allclose(window, kernel, rtol=1e-12, atol=1e-12)

    def test_rejects_non_3d_stack(self):
        with pytest.raises(ValueError, match="must be 3-D"):
            apply_spatial_smoothing(np.zeros((4, 4)), 0.5, 0.1)


class TestPreprocessIntegration:
    @staticmethod
    def _stack():
        rng = np.random.default_rng(3)
        return (rng.random((12, 10, 5)) + 1.0).astype(np.float32)

    def test_default_config_applies_no_smoothing(self):
        cfg = PipelineConfig()
        assert cfg.detection.spatial_smooth_um is None
        res = preprocess(self._stack(), cfg=cfg, dx_um=0.1)
        assert res.spatial_smooth_px == 1

    def test_default_config_needs_no_dx(self):
        """dx_um stays optional as long as smoothing is off."""
        res = preprocess(self._stack(), cfg=PipelineConfig())
        assert res.spatial_smooth_px == 1

    def test_smoothing_requires_dx(self):
        cfg = PipelineConfig()
        cfg.detection.spatial_smooth_um = 0.5
        with pytest.raises(ValueError, match="dx_um is required"):
            preprocess(self._stack(), cfg=cfg, dx_um=None)

    def test_smoothing_is_applied_and_reported(self):
        stack = self._stack()
        cfg = PipelineConfig()
        cfg.detection.spatial_smooth_um = 0.5

        baseline = preprocess(stack, cfg=PipelineConfig(), dx_um=0.1)
        smoothed = preprocess(stack, cfg=cfg, dx_um=0.1)

        assert smoothed.spatial_smooth_px == 7
        assert not np.allclose(smoothed.volume_raw, baseline.volume_raw)

    def test_mask_is_derived_from_smoothed_volume(self):
        """MATLAB computes cellMask4 on the already smoothed volume (line 87)."""
        stack = self._stack()
        cfg = PipelineConfig()
        cfg.detection.spatial_smooth_um = 0.5

        res = preprocess(stack, cfg=cfg, dx_um=0.1)
        expected, _ = apply_spatial_smoothing(stack, 0.5, 0.1)
        reference = preprocess(expected, cfg=PipelineConfig(), dx_um=0.1)

        np.testing.assert_array_equal(res.mask, reference.mask)
