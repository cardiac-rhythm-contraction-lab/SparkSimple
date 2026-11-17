"""tests/unit/test_preprocessing_pipeline.py

Unit tests for sparksimple.preprocessing.pipeline (the ``preprocess`` function
and ``PreprocessingResult`` dataclass).

These tests verify orchestration logic and data flow.  All numerical
correctness is delegated to the unit tests in test_preprocessing_baseline.py
and test_preprocessing_normalise.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from sparksimple.config import PipelineConfig, DetectionConfig
from sparksimple.preprocessing.pipeline import preprocess, PreprocessingResult


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_stack(
    rows: int = 10,
    cols: int = 12,
    frames: int = 20,
    fill: float = 200.0,
) -> np.ndarray:
    """Return a uniform float32 stack."""
    return np.full((rows, cols, frames), fill, dtype=np.float32)


def _default_cfg() -> PipelineConfig:
    return PipelineConfig()


# ── PreprocessingResult structure ─────────────────────────────────────────────

class TestPreprocessingResultType:

    def test_returns_preprocessing_result(self):
        result = preprocess(_make_stack(), _default_cfg())
        assert isinstance(result, PreprocessingResult)

    def test_volume_raw_shape(self):
        stack = _make_stack(8, 9, 15)
        result = preprocess(stack, _default_cfg())
        assert result.volume_raw.shape == (8, 9, 15)

    def test_volume_norm_shape(self):
        stack = _make_stack(8, 9, 15)
        result = preprocess(stack, _default_cfg())
        assert result.volume_norm.shape == (8, 9, 15)

    def test_volume_raw_dtype_float32(self):
        result = preprocess(_make_stack(), _default_cfg())
        assert result.volume_raw.dtype == np.float32

    def test_volume_norm_dtype_float32(self):
        result = preprocess(_make_stack(), _default_cfg())
        assert result.volume_norm.dtype == np.float32

    def test_fo_is_float(self):
        result = preprocess(_make_stack(), _default_cfg())
        assert isinstance(result.fo, float)

    def test_fo_is_positive(self):
        result = preprocess(_make_stack(), _default_cfg())
        assert result.fo > 0

    def test_mask_is_bool_array(self):
        result = preprocess(_make_stack(), _default_cfg())
        assert result.mask.dtype == bool

    def test_mask_shape(self):
        stack = _make_stack(8, 9, 15)
        result = preprocess(stack, _default_cfg())
        assert result.mask.shape == (8, 9)

    def test_baseline_mode_recorded(self):
        result = preprocess(_make_stack(), _default_cfg())
        assert result.baseline_mode == "auto"

    def test_normalisation_mode_recorded(self):
        result = preprocess(_make_stack(), _default_cfg())
        assert result.normalisation_mode == "df_f0"


# ── Default mask (np.ones) ────────────────────────────────────────────────────

class TestDefaultMask:

    def test_no_mask_gives_all_ones(self):
        """Without an explicit mask, all pixels are in-mask (MATLAB line 89)."""
        stack = _make_stack(5, 6, 10)
        result = preprocess(stack, _default_cfg(), mask=None)
        assert result.mask.all()

    def test_explicit_mask_respected(self):
        stack = _make_stack(5, 6, 10)
        mask = np.zeros((5, 6), dtype=bool)
        mask[2:4, 2:4] = True   # small region in-mask
        result = preprocess(stack, _default_cfg(), mask=mask)
        np.testing.assert_array_equal(result.mask, mask)

    def test_mask_shape_mismatch_raises(self):
        stack = _make_stack(5, 6, 10)
        bad_mask = np.ones((4, 6), dtype=bool)   # wrong rows
        with pytest.raises(ValueError, match="mask shape"):
            preprocess(stack, _default_cfg(), mask=bad_mask)


# ── Forced baseline ───────────────────────────────────────────────────────────

class TestForcedBaseline:

    def test_forced_baseline_uses_supplied_value(self):
        cfg = PipelineConfig(baseline_mode="forced", forced_baseline=150.0)
        result = preprocess(_make_stack(fill=200.0), cfg)
        assert result.fo == pytest.approx(150.0)

    def test_forced_baseline_mode_recorded(self):
        cfg = PipelineConfig(baseline_mode="forced", forced_baseline=150.0)
        result = preprocess(_make_stack(), cfg)
        assert result.baseline_mode == "forced"

    def test_forced_baseline_normalisation_correct(self):
        """With Fo=100 and F=200: (200-100)/100 = 1.0."""
        cfg = PipelineConfig(baseline_mode="forced", forced_baseline=100.0)
        stack = _make_stack(fill=200.0)
        result = preprocess(stack, cfg)
        np.testing.assert_allclose(result.volume_norm, 1.0, rtol=1e-5)

    def test_forced_baseline_missing_raises(self):
        """PipelineConfig itself should raise if forced_baseline is None."""
        with pytest.raises(ValueError, match="forced_baseline"):
            PipelineConfig(baseline_mode="forced", forced_baseline=None)


# ── Normalisation mode propagation ────────────────────────────────────────────

class TestNormalisationMode:

    def test_df_f0_mode(self):
        """mode='df_f0': volume_norm = (F - Fo) / Fo."""
        cfg = PipelineConfig(
            baseline_mode="forced",
            forced_baseline=100.0,
            detection=DetectionConfig(normalisation_mode="df_f0"),
        )
        stack = _make_stack(fill=200.0)
        result = preprocess(stack, cfg)
        # (200 - 100) / 100 = 1.0
        np.testing.assert_allclose(result.volume_norm, 1.0, rtol=1e-5)

    def test_f_f0_mode(self):
        """mode='f_f0': volume_norm = F / Fo."""
        cfg = PipelineConfig(
            baseline_mode="forced",
            forced_baseline=100.0,
            detection=DetectionConfig(normalisation_mode="f_f0"),
        )
        stack = _make_stack(fill=200.0)
        result = preprocess(stack, cfg)
        # 200 / 100 = 2.0
        np.testing.assert_allclose(result.volume_norm, 2.0, rtol=1e-5)


# ── Signal sign propagation ───────────────────────────────────────────────────

class TestSignalSign:

    def test_positive_sign_volume_raw_unchanged(self):
        """ss=+1: volume_raw must equal the input stack."""
        stack = _make_stack(fill=300.0)
        cfg = PipelineConfig()
        result = preprocess(stack, cfg)
        np.testing.assert_array_equal(result.volume_raw, stack)

    def test_negative_sign_volume_raw_is_positive(self):
        """ss=-1: after negation+offset, the MATLAB-compatible 10th quantile is 0.

        MATLAB line 96: ``volum = volum - quantile(volum(:), [.1])``
        This shifts the distribution so that the Hazen 10th quantile is 0,
        meaning ~90% of values are ≥ 0 and ~10% can be slightly negative.
        The test checks finite output and that the distribution is shifted
        toward 0 (not that all values are non-negative).

        We use a non-uniform volume so that baseline estimation does not fail
        on an all-zero distribution.
        """
        rng = np.random.default_rng(17)
        stack = rng.uniform(100, 300, size=(5, 6, 10)).astype(np.float32)
        cfg = PipelineConfig(
            detection=DetectionConfig(
                signal_sign=-1,
                normalisation_mode="df_f0",
            ),
            baseline_mode="forced",
            forced_baseline=100.0,   # use forced so baseline estimation is bypassed
        )
        result = preprocess(stack, cfg)
        # All values must be finite
        assert np.all(np.isfinite(result.volume_raw))
        # After negation + MATLAB-compatible 10th-quantile offset,
        # the Hazen 10th quantile should be ~0.
        assert np.quantile(result.volume_raw, 0.10, method="hazen") == pytest.approx(
            0.0, abs=1e-4
        )


# ── selected_channel traceability ────────────────────────────────────────────

class TestSelectedChannel:

    def test_selected_channel_none_by_default(self):
        result = preprocess(_make_stack(), _default_cfg())
        assert result.selected_channel is None

    def test_selected_channel_stored(self):
        result = preprocess(_make_stack(), _default_cfg(), selected_channel=2)
        assert result.selected_channel == 2


# ── Input validation ──────────────────────────────────────────────────────────

class TestInputValidation:

    def test_2d_stack_raises(self):
        bad_stack = np.ones((10, 10), dtype=np.float32)
        with pytest.raises(ValueError, match="3-D"):
            preprocess(bad_stack, _default_cfg())

    def test_4d_stack_raises(self):
        bad_stack = np.ones((10, 10, 5, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="3-D"):
            preprocess(bad_stack, _default_cfg())


# ── repr ──────────────────────────────────────────────────────────────────────

class TestRepr:

    def test_repr_contains_shape(self):
        result = preprocess(_make_stack(10, 12, 20), _default_cfg())
        r = repr(result)
        assert "10" in r and "12" in r and "20" in r

    def test_repr_contains_mode(self):
        result = preprocess(_make_stack(), _default_cfg())
        assert "df_f0" in repr(result)
