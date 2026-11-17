"""tests/unit/test_config.py — Unit tests for sparksimple.config dataclasses.

Tests cover:
- Default instantiation (MATLAB-confirmed defaults present, unknowns are None)
- Override of individual fields
- Validation logic in __post_init__
- Composition of PipelineConfig from sub-configs
"""

import pytest
from sparksimple.config import (
    DetectionConfig,
    FilteringConfig,
    ROIConfig,
    PipelineConfig,
)


# ── DetectionConfig ───────────────────────────────────────────────────────────

class TestDetectionConfig:

    def test_default_filter_window(self):
        """ft = 1 (no smoothing) is the SparkSimple2.m default passed to sparkDetect.m.
        The old filtime=40 comment in sparkDetect.m was in milliseconds and is now
        commented out; current SparkSimple2.m line 14 sets ft=1."""
        cfg = DetectionConfig()
        assert cfg.filter_window_frames == 1

    def test_filter_window_ms_deprecation_warning(self):
        """Passing filter_window_ms should issue a DeprecationWarning and set filter_window_frames."""
        with pytest.deprecated_call():
            cfg = DetectionConfig(filter_window_ms=20.0)
        assert cfg.filter_window_frames == 20

        with pytest.deprecated_call():
            val = cfg.filter_window_ms
        assert val == 20.0

    def test_default_wavelet_scales_real(self):
        """scala = [5:10] for real data (sparkDetect.m line 142)."""
        cfg = DetectionConfig()
        assert cfg.wavelet_scales_real == (5, 10)

    def test_default_wavelet_sim_range(self):
        """scala = [ceil(12/DT):ceil(48/DT)] for simulated data (line 140)."""
        cfg = DetectionConfig()
        assert cfg.wavelet_scales_sim_dt_range_ms == (12.0, 48.0)

    def test_default_noise_connectivity(self):
        """nc = 100 is the SparkSimple2.m default for RegMax3D max-candidate ceiling
        (``tope`` in RegMax3D.m).  Note: voxel connectivity is always 26 in both
        MATLAB bwlabeln and scipy.ndimage.label — this field controls the early-stop
        count, not the neighbourhood structure."""
        cfg = DetectionConfig()
        assert cfg.noise_connectivity == 100

    def test_default_signal_sign(self):
        """ss = 1 is the normal case (line 94)."""
        cfg = DetectionConfig()
        assert cfg.signal_sign == 1

    def test_default_baseline_quantile(self):
        """quantile(vk, .05) used in sparkDetect.m line 114."""
        cfg = DetectionConfig()
        assert cfg.baseline_quantile == 0.05

    def test_default_normalisation_mode(self):
        """optB = 1  →  (F - Fo) / Fo  in sparkDetect.m line 3."""
        cfg = DetectionConfig()
        assert cfg.normalisation_mode == "df_f0"

    def test_default_merge_parameters(self):
        """dm=3 µm, tm=30 ms are the SparkSimple2.m defaults (lines 17-18)."""
        cfg = DetectionConfig()
        assert cfg.spatial_smooth_um is None
        assert cfg.merge_distance_um == 3.0
        assert cfg.merge_time_ms == 30.0

    def test_override_signal_sign(self):
        cfg = DetectionConfig(signal_sign=-1)
        assert cfg.signal_sign == -1

    def test_invalid_signal_sign_raises(self):
        with pytest.raises(ValueError, match="signal_sign"):
            DetectionConfig(signal_sign=0)

    def test_invalid_normalisation_mode_raises(self):
        with pytest.raises(ValueError, match="normalisation_mode"):
            DetectionConfig(normalisation_mode="raw")

    def test_invalid_baseline_quantile_raises(self):
        with pytest.raises(ValueError, match="baseline_quantile"):
            DetectionConfig(baseline_quantile=1.5)

    def test_invalid_baseline_quantile_zero_raises(self):
        with pytest.raises(ValueError, match="baseline_quantile"):
            DetectionConfig(baseline_quantile=0.0)


# ── FilteringConfig ───────────────────────────────────────────────────────────

class TestFilteringConfig:

    def test_all_defaults_none(self):
        """All thresholds depend on the experiment — no defaults from MATLAB source."""
        cfg = FilteringConfig()
        assert cfg.min_amplitude is None
        assert cfg.max_baseline is None
        assert cfg.min_decay_r2 is None
        assert cfg.t2p_range_ms is None
        assert cfg.tau_range_ms is None
        assert cfg.fdhm_range_ms is None

    def test_override_min_amplitude(self):
        cfg = FilteringConfig(min_amplitude=0.15)
        assert cfg.min_amplitude == 0.15

    def test_override_ranges(self):
        cfg = FilteringConfig(tau_range_ms=(10.0, 300.0))
        assert cfg.tau_range_ms == (10.0, 300.0)


# ── ROIConfig ─────────────────────────────────────────────────────────────────

class TestROIConfig:

    def test_default_roi_radius_is_1_5(self):
        """roiR=1.5 µm is the MATLAB SparkSimple2.m default."""
        cfg = ROIConfig()
        assert cfg.roi_radius_um == 1.5

    def test_override_roi_radius(self):
        cfg = ROIConfig(roi_radius_um=2.0)
        assert cfg.roi_radius_um == 2.0


# ── PipelineConfig ────────────────────────────────────────────────────────────

class TestPipelineConfig:

    def test_default_creates_sub_configs(self):
        cfg = PipelineConfig()
        assert isinstance(cfg.detection, DetectionConfig)
        assert isinstance(cfg.filtering, FilteringConfig)
        assert isinstance(cfg.roi, ROIConfig)

    def test_default_baseline_mode(self):
        cfg = PipelineConfig()
        assert cfg.baseline_mode == "auto"

    def test_default_channel_is_none(self):
        cfg = PipelineConfig()
        assert cfg.channel is None

    def test_sub_configs_are_independent(self):
        """Two PipelineConfigs must not share mutable sub-config instances."""
        cfg_a = PipelineConfig()
        cfg_b = PipelineConfig()
        cfg_a.detection.filter_window_frames = 99
        assert cfg_b.detection.filter_window_frames == 1

    def test_forced_baseline_mode_requires_value(self):
        with pytest.raises(ValueError, match="forced_baseline"):
            PipelineConfig(baseline_mode="forced", forced_baseline=None)

    def test_forced_baseline_mode_with_value(self):
        cfg = PipelineConfig(baseline_mode="forced", forced_baseline=500.0)
        assert cfg.forced_baseline == 500.0

    def test_invalid_baseline_mode_raises(self):
        with pytest.raises(ValueError, match="baseline_mode"):
            PipelineConfig(baseline_mode="unknown")

    def test_negative_channel_raises(self):
        with pytest.raises(ValueError, match="channel"):
            PipelineConfig(channel=-1)

    def test_override_nested_detection(self):
        cfg = PipelineConfig(
            detection=DetectionConfig(signal_sign=-1, filter_window_frames=20)
        )
        assert cfg.detection.signal_sign == -1
        assert cfg.detection.filter_window_frames == 20
