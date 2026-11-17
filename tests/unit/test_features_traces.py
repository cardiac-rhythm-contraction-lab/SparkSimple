"""Unit tests for ROI trace extraction and snippet cropping."""

import numpy as np
import pytest

from sparksimple.features.traces import crop_temporal_snippet, extract_spatial_roi_trace


def test_extract_spatial_roi_trace_mask_filtering():
    """Verify extract_spatial_roi_trace correctly excludes out-of-mask pixels."""
    ny, nx, nt = 10, 10, 20
    vol = np.ones((ny, nx, nt), dtype=np.float64)

    # Pixel (5, 5) has signal 10.0, pixel (5, 6) has signal 20.0
    vol[5, 5, :] = 10.0
    vol[5, 6, :] = 20.0

    mask = np.zeros((ny, nx), dtype=np.float64)
    mask[5, 5] = 1.0  # Only (5,5) is inside cell mask

    # Extract trace with margX = 2
    ts = extract_spatial_roi_trace(vol, mask, center_y=5, center_x=5, marg_x_px=2)

    assert len(ts) == nt
    # Because (5,6) is outside mask (mask=0), only (5,5) is included => mean is 10.0
    np.testing.assert_allclose(ts, 10.0)


def test_crop_temporal_snippet_bounds():
    """Verify crop_temporal_snippet cropping windows and 1-based indexing."""
    timesignal = np.linspace(1.0, 100.0, 100)
    center_t = 50  # 0-based frame 50 (value 51.0)

    marg_frames = (10, 20)  # [50-10, 50+20] => [40, 70] inclusive in 0-based => indices 40..70 (length 31)

    cut, t_interval, peak_rel, ot_1b, ft_1b = crop_temporal_snippet(
        timesignal, center_t, marg_frames
    )

    assert len(cut) == 31
    assert ot_1b == 41
    assert ft_1b == 71
    assert t_interval[0] == 41
    assert t_interval[-1] == 71
    assert peak_rel == 10
    assert cut[peak_rel] == timesignal[center_t]
