"""tests/unit/test_preprocessing_baseline.py

Unit tests for sparksimple.preprocessing.baseline.

These tests verify the behaviour of individual functions using synthetic
(hand-crafted) arrays.  They do NOT constitute MATLAB regression validation —
they only confirm that the Python implementation is internally consistent
and matches the documented algorithm logic.

MATLAB Regression Note
----------------------
The quantile method used here was validated against MATLAB R2025b with
controlled reference vectors.  These tests still operate on synthetic arrays,
but the expected quantile values now reflect MATLAB-verified behaviour rather
than a provisional hypothesis.
"""

from __future__ import annotations

import numpy as np
import pytest

from sparksimple.preprocessing.baseline import (
    apply_signal_sign,
    estimate_baseline_global,
    estimate_baseline_forced,
)


# ── apply_signal_sign ─────────────────────────────────────────────────────────

class TestApplySignalSign:

    def _make_volume(self, fill: float = 100.0) -> np.ndarray:
        """3×4×5 volume filled with *fill*, dtype float32."""
        return np.full((3, 4, 5), fill, dtype=np.float32)

    def test_positive_sign_unchanged(self):
        """ss=+1: volume must not change."""
        vol = self._make_volume(200.0)
        result = apply_signal_sign(vol, 1)
        np.testing.assert_array_equal(result, vol)

    def test_positive_sign_dtype_preserved(self):
        """ss=+1: dtype must be preserved."""
        vol = self._make_volume()
        result = apply_signal_sign(vol, 1)
        assert result.dtype == np.float32

    def test_negative_sign_negates(self):
        """ss=-1: after negation+offset the result must be ≥ 0."""
        vol = self._make_volume(100.0)
        result = apply_signal_sign(vol, -1)
        # With a uniform volume, -vol + quantile(10%) of -vol = 0 exactly
        assert result.min() >= 0.0

    def test_negative_sign_shifts_by_10th_percentile(self):
        """ss=-1: offset is quantile(volum*-1, 0.10) of the negated volume."""
        rng = np.random.default_rng(42)
        vol = rng.uniform(50, 200, size=(10, 10, 20)).astype(np.float32)
        negated = -vol.astype(np.float64)
        expected_offset = np.quantile(negated, 0.10, method="hazen")
        expected = (negated - expected_offset).astype(np.float32)
        result = apply_signal_sign(vol, -1)
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_invalid_sign_raises(self):
        """Signal sign other than ±1 must raise ValueError."""
        vol = self._make_volume()
        with pytest.raises(ValueError, match="signal_sign"):
            apply_signal_sign(vol, 0)

    def test_invalid_sign_2_raises(self):
        with pytest.raises(ValueError, match="signal_sign"):
            apply_signal_sign(self._make_volume(), 2)


# ── estimate_baseline_global ──────────────────────────────────────────────────

class TestEstimateBaselineGlobal:

    def _uniform_stack(
        self, value: float, rows: int = 5, cols: int = 5, frames: int = 10
    ) -> np.ndarray:
        return np.full((rows, cols, frames), value, dtype=np.float32)

    def _all_ones_mask(self, rows: int = 5, cols: int = 5) -> np.ndarray:
        return np.ones((rows, cols), dtype=bool)

    # -- Basic behaviour --

    def test_uniform_volume_returns_value(self):
        """Uniform volume: quantile of a constant array == the constant."""
        vol = self._uniform_stack(300.0)
        mask = self._all_ones_mask()
        fo = estimate_baseline_global(vol, mask, quantile=0.05)
        assert fo == pytest.approx(300.0, rel=1e-5)

    def test_returns_float(self):
        """Baseline must be a Python float, not an array."""
        vol = self._uniform_stack(100.0)
        mask = self._all_ones_mask()
        fo = estimate_baseline_global(vol, mask)
        assert isinstance(fo, float)

    # -- Mask exclusion --

    def test_masked_pixels_excluded(self):
        """Pixels where mask=0 must not influence the estimate.

        Setup: half of the pixels have value 1 (in-mask), the other half
        have value 1000 (out-of-mask).  The Fo should be ≈ 1, not 500.
        """
        rows, cols, frames = 4, 4, 5
        vol = np.ones((rows, cols, frames), dtype=np.float32)
        # out-of-mask pixels = rows 2-3
        vol[2:, :, :] = 1000.0
        mask = np.zeros((rows, cols), dtype=bool)
        mask[:2, :] = True   # only first two rows are in-mask

        fo = estimate_baseline_global(vol, mask, quantile=0.05)
        assert fo == pytest.approx(1.0, rel=1e-4)

    def test_all_zero_mask_raises(self):
        """A mask of all-zeros means no in-mask pixels → ValueError."""
        vol = self._uniform_stack(100.0)
        mask = np.zeros((5, 5), dtype=bool)
        with pytest.raises(ValueError, match="No in-mask pixels"):
            estimate_baseline_global(vol, mask)

    # -- Quantile parameter --

    def test_quantile_05_on_sorted_distribution(self):
        """
        Verify quantile logic with a controlled distribution.

        Stack contains values 1..100 (over the 100 in-mask pixels × 1 frame).
        The 5th percentile of integers 1..100 by MATLAB-compatible Hazen
        interpolation is 5.5.
        """
        # Build a 10×10×1 stack with values 1..100
        flat = np.arange(1, 101, dtype=np.float32)
        vol = flat.reshape(10, 10, 1)
        mask = np.ones((10, 10), dtype=bool)

        fo = estimate_baseline_global(vol, mask, quantile=0.05)
        assert fo == pytest.approx(5.5, rel=1e-5)

    def test_quantile_50_is_median(self):
        """quantile=0.5 must return the median."""
        flat = np.arange(1, 101, dtype=np.float32)
        vol = flat.reshape(10, 10, 1)
        mask = np.ones((10, 10), dtype=bool)
        fo = estimate_baseline_global(vol, mask, quantile=0.50)
        assert fo == pytest.approx(np.median(flat), rel=1e-5)

    def test_invalid_quantile_zero_raises(self):
        with pytest.raises(ValueError, match="quantile"):
            estimate_baseline_global(
                self._uniform_stack(100.0), self._all_ones_mask(), quantile=0.0
            )

    def test_invalid_quantile_one_raises(self):
        with pytest.raises(ValueError, match="quantile"):
            estimate_baseline_global(
                self._uniform_stack(100.0), self._all_ones_mask(), quantile=1.0
            )

    # -- Float64 precision --

    def test_computation_in_float64(self):
        """Fo must be float64 regardless of float32 input."""
        vol = np.full((5, 5, 5), 123.456, dtype=np.float32)
        mask = np.ones((5, 5), dtype=bool)
        fo = estimate_baseline_global(vol, mask)
        # Python float is always 64-bit
        assert isinstance(fo, float)


# ── estimate_baseline_forced ──────────────────────────────────────────────────

class TestEstimateBaselineForced:

    def test_returns_forced_value(self):
        fo = estimate_baseline_forced(500.0)
        assert fo == pytest.approx(500.0)

    def test_returns_float(self):
        fo = estimate_baseline_forced(100)
        assert isinstance(fo, float)

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="> 0"):
            estimate_baseline_forced(0.0)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="> 0"):
            estimate_baseline_forced(-100.0)
