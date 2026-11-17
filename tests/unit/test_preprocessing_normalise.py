"""tests/unit/test_preprocessing_normalise.py

Unit tests for sparksimple.preprocessing.normalise.

These tests verify the normalisation functions against analytically
computable expected values using synthetic arrays.  They do NOT validate
MATLAB numerical equivalence — they confirm internal Python correctness.
"""

from __future__ import annotations

import numpy as np
import pytest

from sparksimple.preprocessing.normalise import (
    normalise_df_f0,
    normalise_f_f0,
    normalise_volume,
    denormalise_df_f0,
    _validate_fo,
)


# ── normalise_df_f0 ───────────────────────────────────────────────────────────

class TestNormaliseDfF0:

    def test_scalar_fo(self):
        """(F - Fo) / Fo with scalar Fo."""
        vol = np.full((3, 4, 5), 200.0, dtype=np.float32)
        result = normalise_df_f0(vol, fo=100.0)
        # (200 - 100) / 100 = 1.0
        np.testing.assert_allclose(result, 1.0, rtol=1e-6)

    def test_output_dtype_float32(self):
        """Output must be float32 regardless of computation precision."""
        vol = np.ones((2, 2, 2), dtype=np.float32)
        result = normalise_df_f0(vol, fo=1.0)
        assert result.dtype == np.float32

    def test_output_shape_unchanged(self):
        vol = np.ones((7, 8, 10), dtype=np.float32)
        result = normalise_df_f0(vol, fo=1.0)
        assert result.shape == vol.shape

    def test_array_fo_2d(self):
        """2-D spatial Fo map: each pixel normalised by its own Fo."""
        rows, cols, frames = 2, 2, 3
        vol = np.full((rows, cols, frames), 200.0, dtype=np.float32)
        fo = np.array([[100.0, 50.0], [200.0, 400.0]])
        result = normalise_df_f0(vol, fo)
        expected = np.zeros((rows, cols, frames), dtype=np.float32)
        for r in range(rows):
            for c in range(cols):
                expected[r, c, :] = (200.0 - fo[r, c]) / fo[r, c]
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_fo_equal_to_f_gives_zero(self):
        """When F == Fo everywhere: (F - Fo)/Fo = 0."""
        vol = np.full((3, 3, 5), 150.0, dtype=np.float32)
        result = normalise_df_f0(vol, fo=150.0)
        np.testing.assert_allclose(result, 0.0, atol=1e-7)

    def test_negative_fo_raises(self):
        vol = np.ones((2, 2, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="fo must be > 0"):
            normalise_df_f0(vol, fo=-1.0)

    def test_zero_fo_raises(self):
        vol = np.ones((2, 2, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="fo must be > 0"):
            normalise_df_f0(vol, fo=0.0)

    def test_array_fo_with_nonpositive_raises(self):
        vol = np.ones((2, 2, 2), dtype=np.float32)
        fo = np.array([[100.0, -5.0], [200.0, 50.0]])
        with pytest.raises(ValueError, match="non-positive"):
            normalise_df_f0(vol, fo)


# ── normalise_f_f0 ────────────────────────────────────────────────────────────

class TestNormaliseFF0:

    def test_scalar_fo(self):
        """F / Fo with scalar Fo."""
        vol = np.full((3, 4, 5), 200.0, dtype=np.float32)
        result = normalise_f_f0(vol, fo=100.0)
        # 200 / 100 = 2.0
        np.testing.assert_allclose(result, 2.0, rtol=1e-6)

    def test_output_dtype_float32(self):
        vol = np.ones((2, 2, 2), dtype=np.float32)
        result = normalise_f_f0(vol, fo=1.0)
        assert result.dtype == np.float32

    def test_array_fo_2d(self):
        rows, cols, frames = 2, 2, 3
        vol = np.full((rows, cols, frames), 100.0, dtype=np.float32)
        fo = np.array([[50.0, 25.0], [100.0, 200.0]])
        result = normalise_f_f0(vol, fo)
        expected = np.zeros((rows, cols, frames), dtype=np.float32)
        for r in range(rows):
            for c in range(cols):
                expected[r, c, :] = 100.0 / fo[r, c]
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_fo_equal_to_f_gives_one(self):
        """When F == Fo everywhere: F/Fo = 1."""
        vol = np.full((3, 3, 5), 80.0, dtype=np.float32)
        result = normalise_f_f0(vol, fo=80.0)
        np.testing.assert_allclose(result, 1.0, rtol=1e-6)

    def test_zero_fo_raises(self):
        vol = np.ones((2, 2, 2), dtype=np.float32)
        with pytest.raises(ValueError):
            normalise_f_f0(vol, fo=0.0)


# ── normalise_volume dispatch ─────────────────────────────────────────────────

class TestNormaliseVolume:

    def _vol(self) -> np.ndarray:
        return np.full((3, 3, 4), 200.0, dtype=np.float32)

    def test_dispatch_df_f0(self):
        """mode='df_f0' must produce (F-Fo)/Fo."""
        result = normalise_volume(self._vol(), fo=100.0, mode="df_f0")
        np.testing.assert_allclose(result, 1.0, rtol=1e-6)

    def test_dispatch_f_f0(self):
        """mode='f_f0' must produce F/Fo."""
        result = normalise_volume(self._vol(), fo=100.0, mode="f_f0")
        np.testing.assert_allclose(result, 2.0, rtol=1e-6)

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="mode"):
            normalise_volume(self._vol(), fo=100.0, mode="raw")


# ── denormalise_df_f0 (roundtrip) ─────────────────────────────────────────────

class TestDenormaliseDfF0:

    def test_roundtrip_scalar_fo(self):
        """normalise then denormalise must recover the original volume."""
        rng = np.random.default_rng(7)
        vol = rng.uniform(100, 500, size=(5, 5, 20)).astype(np.float32)
        fo = 200.0
        norm = normalise_df_f0(vol, fo)
        recovered = denormalise_df_f0(norm, fo)
        np.testing.assert_allclose(recovered, vol, rtol=1e-5)

    def test_roundtrip_array_fo(self):
        """Roundtrip with a 2-D spatial Fo map."""
        rng = np.random.default_rng(13)
        rows, cols, frames = 6, 8, 15
        vol = rng.uniform(50, 400, size=(rows, cols, frames)).astype(np.float32)
        fo = rng.uniform(10, 100, size=(rows, cols)).astype(np.float64)
        norm = normalise_df_f0(vol, fo)
        recovered = denormalise_df_f0(norm, fo)
        np.testing.assert_allclose(recovered, vol, rtol=1e-5)


# ── _validate_fo ──────────────────────────────────────────────────────────────

class TestValidateFo:

    def test_positive_scalar_ok(self):
        _validate_fo(100.0)  # must not raise

    def test_zero_scalar_raises(self):
        with pytest.raises(ValueError):
            _validate_fo(0.0)

    def test_positive_array_ok(self):
        _validate_fo(np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_array_with_zero_raises(self):
        with pytest.raises(ValueError, match="non-positive"):
            _validate_fo(np.array([[0.0, 1.0]]))
