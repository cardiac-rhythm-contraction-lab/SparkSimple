"""Unit tests for Phase 6B 3D local maximum detection (RegMax3D port)."""

import numpy as np
import pytest

from sparksimple.detection.local_max import (
    Candidate,
    CandidateResult,
    _matlab_round_pos,
    detect_candidates,
    extract_candidates,
    regmax_3d,
)


def test_matlab_round_pos():
    """Verify MATLAB-faithful rounding logic for positive numbers."""
    assert _matlab_round_pos(0.4) == 0
    assert _matlab_round_pos(0.5) == 1
    assert _matlab_round_pos(1.4) == 1
    assert _matlab_round_pos(1.5) == 2
    assert _matlab_round_pos(2.5) == 3
    assert _matlab_round_pos(3.0) == 3


def test_regmax_3d_zero_volume():
    """Verify regmax_3d returns zero matrix for all-zero volume."""
    vol = np.zeros((10, 10, 10), dtype=np.float64)
    ix = regmax_3d(vol, max_candidates=50)
    assert ix.shape == (10, 10, 10)
    assert np.all(ix == 0.0)


def test_regmax_3d_single_peak():
    """Verify regmax_3d detects a single synthetic peak at expected coordinate."""
    vol = np.zeros((16, 16, 16), dtype=np.float64)
    y_c, x_c, t_c = 8, 8, 8

    # Create 3D Gaussian blob
    yy, xx, tt = np.ogrid[:16, :16, :16]
    dist2 = (yy - y_c) ** 2 + (xx - x_c) ** 2 + (tt - t_c) ** 2
    vol = 100.0 * np.exp(-dist2 / 4.0)

    ix = regmax_3d(vol, max_candidates=50)
    cands = extract_candidates(ix, v3=vol)

    assert len(cands) == 1
    cand = cands[0]
    assert (cand.y_0b, cand.x_0b, cand.t_0b) == (8, 8, 8)
    assert (cand.y_1b, cand.x_1b, cand.t_1b) == (9, 9, 9)
    assert cand.intensity > 99.0
    assert cand.step_level > 0.0


def test_regmax_3d_two_peaks_ranking():
    """Verify regmax_3d detects two peaks and ranks higher peak with larger step level."""
    vol = np.zeros((20, 20, 20), dtype=np.float64)
    yy, xx, tt = np.ogrid[:20, :20, :20]

    # Peak 1: high (val ~100) at (5, 5, 5)
    p1 = 100.0 * np.exp(-((yy - 5) ** 2 + (xx - 5) ** 2 + (tt - 5) ** 2) / 4.0)
    # Peak 2: lower (val ~70) at (14, 14, 14)
    p2 = 70.0 * np.exp(-((yy - 14) ** 2 + (xx - 14) ** 2 + (tt - 14) ** 2) / 4.0)
    vol = p1 + p2

    ix = regmax_3d(vol, max_candidates=50)
    cands = extract_candidates(ix, v3=vol)

    assert len(cands) == 2
    # Candidates are sorted by step_level descending
    assert (cands[0].y_0b, cands[0].x_0b, cands[0].t_0b) == (5, 5, 5)
    assert (cands[1].y_0b, cands[1].x_0b, cands[1].t_0b) == (14, 14, 14)
    assert cands[0].step_level >= cands[1].step_level


def test_regmax_3d_max_candidates_cap():
    """Verify regmax_3d respects max_candidates limit."""
    vol = np.zeros((40, 40, 40), dtype=np.float64)

    # Add 5 separate peaks with spatial extent (sigma=2.0)
    centers = [(8, 8, 8), (8, 30, 8), (30, 8, 8), (30, 30, 8), (20, 20, 30)]
    yy, xx, tt = np.ogrid[:40, :40, :40]
    for idx, (y, x, t) in enumerate(centers):
        amp = 100.0 - idx * 10.0
        vol += amp * np.exp(-((yy - y) ** 2 + (xx - x) ** 2 + (tt - t) ** 2) / 8.0)

    # Threshold background noise to zero, as done in Phase 6A (V3_thresh)
    vol[vol < 10.0] = 0.0

    # Cap max candidates to 50 (MATLAB default nc)
    ix = regmax_3d(vol, max_candidates=50)
    cands = extract_candidates(ix, v3=vol)

    assert len(cands) == 5
    assert np.count_nonzero(ix > 0) == 5


def test_regmax_3d_invalid_dim():
    """Verify regmax_3d raises ValueError on non-3D input."""
    with pytest.raises(ValueError, match="3-dimensional"):
        regmax_3d(np.zeros((10, 10)))


def test_detect_candidates_orchestrator():
    """Verify detect_candidates orchestrator function returns CandidateResult."""
    vol = np.zeros((15, 15, 15), dtype=np.float64)
    vol[7, 7, 7] = 50.0
    vol[7, 7, 6] = 45.0

    res = detect_candidates(vol, max_candidates=10)
    assert isinstance(res, CandidateResult)
    assert res.count == len(res.candidates)
    assert res.max_candidates == 10
    assert res.count >= 1
