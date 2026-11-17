"""Regression tests for the execution-audit contracts.

These tests intentionally cover technical output contracts, not detector
calibration.  They make NC saturation, units, channel ambiguity and ROI output
visible to a caller.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import tifffile

from sparksimple.config import DetectionConfig
from sparksimple.detection.local_max import detect_candidates
from sparksimple.features.models import SparkFeature
from sparksimple.filtering.models import FilteredSpark, RejectionReason
from sparksimple.roi.cluster import cluster_rois
from sparksimple.io.tiff_loader import load_image_stack


def _accepted(px: int, py: int) -> FilteredSpark:
    feature = SparkFeature(
        px=px, py=py, pt=3, x_0b=px - 1, y_0b=py - 1, t_0b=2,
        dist2memb=1.0, fwhm=1.0, timesignal=np.ones(4),
        timeinterval=np.arange(1, 5), r2=1.0, amp=1.0, tau=1.0,
        ror=1.0, t2p=1.0, fdhm=1.0, AMP=1.0, BL=0.0, mass=1.0,
    )
    return FilteredSpark(id=1, good=True, fail=RejectionReason.ACCEPTED, feature=feature)


def test_candidate_limit_is_reported_when_count_reaches_nc():
    volume = np.zeros((9, 9, 9), dtype=float)
    volume[2, 2, 2] = 9.0
    volume[6, 6, 6] = 8.0
    with pytest.warns(RuntimeWarning, match="candidate limit"):
        result = detect_candidates(volume, max_candidates=1, temporal_split_frac=0.0)
    assert result.count == 1
    assert result.max_candidates == 1
    assert result.candidate_limit_reached is True


def test_filter_window_deprecation_mentions_frames_and_effective_default():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        DetectionConfig(filter_window_ms=20)
    assert "1 frame" in str(caught[0].message)
    assert "frames" in str(caught[0].message)


def test_single_accepted_spark_has_matlab_empty_roi_array():
    sparks = [_accepted(3, 3)]
    result = cluster_rois(
        sparks, np.ones((8, 8, 4)), np.ones((8, 8)),
        dx_um=0.2, dt_ms=10.0, roi_r_um=1.0,
    )
    assert sparks[0].roi == 1
    assert len(result.rois) == 0


def test_roi_window_is_2r_plus_1_and_clipped_at_border():
    sparks = [_accepted(4, 4), _accepted(4, 4)]
    volume = np.arange(9 * 9 * 2, dtype=float).reshape(9, 9, 2)
    result = cluster_rois(sparks, volume, np.ones((9, 9)), 1.0, 10.0, 2.0)
    # The central ROI contains rows/columns 2..6 (1-based): 5x5 pixels.
    expected = volume[1:6, 1:6, :].mean(axis=(0, 1))
    assert np.allclose(result.rois[0].signal, expected)


def test_temporal_split_zero_is_strict_no_prepass():
    volume = np.zeros((1, 1, 15), dtype=float)
    volume[0, 0, [3, 4, 5, 6, 7, 8, 9]] = [3, 7, 10, 8, 9, 7, 3]
    strict = detect_candidates(volume, max_candidates=10, temporal_split_frac=0.0)
    extended = detect_candidates(volume, max_candidates=10, temporal_split_frac=0.3)
    # ``0`` is deliberately the strict MATLAB/no-prepass mode; the extension
    # may alter component topology, but must never be implicit in this mode.
    assert strict.count >= 1
    assert extended.count >= 1


def test_multichannel_loading_requires_an_explicit_channel(tmp_path):
    for frame in (1, 2):
        for channel, intensity in (("ch00", 10), ("ch01", 20), ("ch02", 30)):
            tifffile.imwrite(tmp_path / f"frame{frame}_{channel}.tif", np.full((3, 3), intensity, np.uint16))
    with pytest.raises(ValueError, match="explicit channel"):
        load_image_stack(tmp_path)
    stack, selected = load_image_stack(tmp_path, channel="ch01")
    assert selected == 1
    assert np.all(stack == 20)


def test_density_is_exported_in_unambiguous_um2_and_mm2_units(tmp_path):
    from sparksimple.export.pipeline import run_phase10
    spark = _accepted(3, 3)
    out = run_phase10(
        [spark], np.ones((10, 10, 162)), np.ones((10, 10)), np.ones((10, 10)),
        dx_um=0.86, dt_ms=100.0, roi_r_um=1.5, output_dir=str(tmp_path), title="density",
    )
    assert out.summary.duration_s == pytest.approx(16.1)
    assert out.summary.spark_density_per_um2_s == pytest.approx(1 / (16.1 * 73.96))
    assert out.summary.spark_density_per_mm2_s == pytest.approx(out.summary.spark_density_per_um2_s * 1e6)
    assert out.summary.spark_density == pytest.approx(out.summary.spark_density_per_mm2_s)
    assert "spark_density_per_um2_s" in (tmp_path / "yData_density.csv").read_text()
