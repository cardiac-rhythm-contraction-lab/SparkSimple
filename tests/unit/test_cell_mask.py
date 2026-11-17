"""Unit tests for sparksimple.preprocessing.cell_mask (cell_mask4 algorithm)."""

import numpy as np
import pytest

from sparksimple.preprocessing.cell_mask import (
    _histc_matlab,
    _kmeans2_1d,
    _find_derivative_threshold,
    _keep_largest_component,
    _fill_internal_holes,
    cell_mask4,
)


def test_histc_matlab_exact_semantics():
    """Verify _histc_matlab matches MATLAB histc behavior on controlled vector."""
    edges = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # x contains values:
    # < 0.0 (ignored: -0.1)
    # [0.0, 0.2): 0.0, 0.1 (2 count)
    # [0.2, 0.4): 0.2 (1 count)
    # [0.4, 0.6): 0.4, 0.5, 0.59 (3 count)
    # [0.6, 0.8): none (0 count)
    # [0.8, 1.0): 0.8, 0.9 (2 count)
    # == 1.0: 1.0, 1.0 (2 count in last bin)
    # > 1.0 (ignored: 1.2)
    x = np.array([-0.1, 0.0, 0.1, 0.2, 0.4, 0.5, 0.59, 0.8, 0.9, 1.0, 1.0, 1.2])

    h = _histc_matlab(x, edges)
    expected = np.array([2, 1, 3, 0, 2, 2])
    np.testing.assert_array_equal(h, expected)


def test_kmeans2_1d_bimodal():
    """Verify _kmeans2_1d calculates midpoint threshold between two clusters."""
    # Bimodal distribution around 10.0 and 50.0
    np.random.seed(42)
    c1 = np.random.normal(loc=10.0, scale=1.0, size=500)
    c2 = np.random.normal(loc=50.0, scale=1.0, size=500)
    vals = np.concatenate([c1, c2])

    thresh = _kmeans2_1d(vals)
    # Midpoint should be approximately (10 + 50) / 2 = 30
    assert 25.0 < thresh < 35.0


def test_find_derivative_threshold():
    """Verify _find_derivative_threshold identifies local inflection zero crossing."""
    edges = np.linspace(0, 1, 100)
    # Create a smooth bimodal histogram with a clear valley around bin 40
    x = np.arange(len(edges))
    h = 100 * np.exp(-((x - 20) ** 2) / 50) + 150 * np.exp(-((x - 70) ** 2) / 100)

    thresh = _find_derivative_threshold(h, edges)
    assert thresh is not None
    assert 0.2 <= thresh <= 0.6


def test_keep_largest_component():
    """Verify _keep_largest_component isolates the largest 8-connected blob."""
    img = np.zeros((20, 20), dtype=bool)
    # Small blob (3x3 = 9 pixels)
    img[2:5, 2:5] = True
    # Large blob (6x6 = 36 pixels)
    img[10:16, 10:16] = True

    cleaned = _keep_largest_component(img)

    assert np.count_nonzero(cleaned) == 36
    assert np.all(cleaned[10:16, 10:16])
    assert not np.any(cleaned[2:5, 2:5])


def test_fill_internal_holes():
    """Verify _fill_internal_holes fills internal background holes not touching borders."""
    mask = np.zeros((30, 30), dtype=bool)
    # Create solid square cell
    mask[5:25, 5:25] = True
    # Punch internal hole inside cell
    mask[10:15, 10:15] = False

    filled = _fill_internal_holes(mask)

    # Internal hole should be filled back to True
    assert np.all(filled[5:25, 5:25])
    # Outside background should remain False
    assert not np.any(filled[0:5, :])
    assert not np.any(filled[25:, :])


def test_cell_mask4_synthetic_cell():
    """Test end-to-end cell_mask4 on a synthetic cell image."""
    ny, nx = 60, 60
    y, x = np.ogrid[:ny, :nx]
    center_y, center_x = 30, 30

    # Elliptical cell body
    dist_sq = ((y - center_y) / 18.0) ** 2 + ((x - center_x) / 12.0) ** 2
    cell_body = dist_sq <= 1.0

    # Image stack sum: bright cell (200..255) vs dark background (10..30)
    img = np.zeros((ny, nx), dtype=np.float64)
    np.random.seed(123)
    bg_noise = np.random.uniform(10, 30, size=(ny, nx))
    cell_signal = np.random.uniform(150, 255, size=(ny, nx))

    img[~cell_body] = bg_noise[~cell_body]
    img[cell_body] = cell_signal[cell_body]

    # Add small noise blob outside cell
    img[2:5, 2:5] = 220.0

    # Add internal hole inside cell
    img[28:32, 28:32] = 15.0

    mask = cell_mask4(img)

    # 1. Mask shape matches input
    assert mask.shape == (ny, nx)
    assert mask.dtype == bool

    # 2. Cell interior is detected as True
    assert np.mean(mask[15:25, 25:35]) > 0.95

    # 3. Background is detected as False
    assert np.mean(mask[0:10, 0:10]) == 0.0

    # 4. Small noise blob outside cell was filtered out
    assert not np.any(mask[2:5, 2:5])

    # 5. Internal hole inside cell was filled
    assert np.all(mask[28:32, 28:32])
