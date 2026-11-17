"""Unit tests for sparksimple.detection.fo_reestimate."""

import numpy as np
import pytest

from sparksimple.detection.fo_reestimate import (
    FoReestimateResult,
    apply_border_fo_adjustment,
    build_fo_smoothing_kernel_3d,
    reestimate_fo,
    should_reestimate_fo,
)


def test_should_reestimate_fo():
    """Verify MATLAB sum(size(forcedbaseline)) == 2 semantics."""
    assert should_reestimate_fo(None) is True
    assert should_reestimate_fo(0) is True
    assert should_reestimate_fo(0.0) is True
    assert should_reestimate_fo(100.0) is True
    assert should_reestimate_fo(np.float64(100.0)) is True

    # 0D or 1x1 array
    assert should_reestimate_fo(np.array(100.0)) is True
    assert should_reestimate_fo(np.ones((1, 1))) is True

    # 2D spatial matrix (Y+X != 2)
    assert should_reestimate_fo(np.ones((10, 10))) is False
    assert should_reestimate_fo(np.ones((150, 52))) is False


def test_build_fo_smoothing_kernel_3d():
    """Verify 3D smoothing kernel construction."""
    dx_um = 0.3
    dt_ms = 5.0
    kernel = build_fo_smoothing_kernel_3d(dx_um=dx_um, dt_ms=dt_ms)

    expected_spatial = int(np.ceil(2.0 / dx_um))  # ceil(2 / 0.3) = 7
    expected_suptemp = int(np.ceil(68.0 / dt_ms))  # ceil(68 / 5) = 14

    assert kernel.shape == (expected_spatial, expected_spatial, expected_suptemp)
    assert np.isclose(np.sum(kernel), 1.0, atol=1e-12)


def test_apply_border_fo_adjustment():
    """Verify border max propagation."""
    rows, cols = 20, 20
    fo_map = np.ones((rows, cols), dtype=np.float64) * 100.0
    fo_map[5, 5] = 200.0  # high interior value

    mask = np.zeros((rows, cols), dtype=bool)
    mask[3:15, 3:15] = True  # cell mask blob

    foo, border_mask = apply_border_fo_adjustment(fo_map, mask, dx_um=0.3, roi_r_um=1.5)

    assert foo.shape == (rows, cols)
    assert border_mask.shape == (rows, cols)
    assert np.any(border_mask)  # border pixels identified
    # Border pixels near (5,5) should pick up the 200.0 value
    assert foo[4, 4] == 200.0


def test_reestimate_fo_basic():
    """Test reestimate_fo pipeline on a synthetic 3D volume."""
    rng = np.random.default_rng(42)
    rows, cols, frames = 15, 15, 30
    volume_raw = 100.0 + rng.normal(0, 2.0, size=(rows, cols, frames))

    # Add candidate peak
    v3_thresholded = np.zeros((rows, cols, frames), dtype=np.float64)
    v3_thresholded[7, 7, 15] = 5.0
    volume_raw[7, 7, 15] += 50.0

    mask = np.ones((rows, cols), dtype=bool)

    res = reestimate_fo(
        volume_raw=volume_raw,
        v3_thresholded=v3_thresholded,
        mask=mask,
        estimated_fo=100.0,
        dx_um=0.3,
        dt_ms=5.0,
        roi_r_um=1.5,
    )

    assert isinstance(res, FoReestimateResult)
    assert res.fo_map.shape == (rows, cols)
    assert res.volume_renormalised.shape == (rows, cols, frames)
    assert res.candidate_exclusion_mask[7, 7, 15] is np.bool_(True)
    assert np.all(res.fo_map > 0.0)


def test_reestimate_fo_forced_2d():
    """Verify 2D matrix forced baseline skips re-estimation."""
    rows, cols, frames = 10, 10, 20
    volume_raw = np.ones((rows, cols, frames), dtype=np.float64) * 100.0
    v3_thresholded = np.zeros((rows, cols, frames), dtype=np.float64)
    mask = np.ones((rows, cols), dtype=bool)

    forced_2d = np.ones((rows, cols), dtype=np.float64) * 120.0

    res = reestimate_fo(
        volume_raw=volume_raw,
        v3_thresholded=v3_thresholded,
        mask=mask,
        estimated_fo=100.0,
        dx_um=0.3,
        dt_ms=5.0,
        baseline_mode="forced",
        forced_baseline_value=forced_2d,
    )

    assert np.array_equal(res.fo_map, forced_2d)
    # df/f0: (100 - 120) / 120 = -1/6
    assert np.allclose(res.volume_renormalised, -1.0 / 6.0)
