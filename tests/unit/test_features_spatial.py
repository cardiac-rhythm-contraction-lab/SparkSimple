"""Unit tests for spatial feature helpers (distance_to_membrane and measure_blob_fwhm)."""

import numpy as np
import pytest

from sparksimple.features.spatial import distance_to_membrane, measure_blob_fwhm


def test_distance_to_membrane():
    """Verify distance_to_membrane matches Euclidean distance transform scaled by DX."""
    mask = np.zeros((20, 20), dtype=np.float64)
    mask[5:15, 5:15] = 1.0  # 10x10 square cell mask inside 20x20

    dx_um = 0.5
    dist_map = distance_to_membrane(mask, dx_um)

    # Outside mask distance should be 0.0
    assert dist_map[0, 0] == 0.0
    assert dist_map[4, 4] == 0.0

    # Perimeter inside mask should have distance 1 * dx = 0.5
    assert dist_map[5, 5] == 0.5
    assert dist_map[5, 10] == 0.5

    # Center pixel (9, 9) should be 5 pixels away from boundary => 2.5 um
    assert pytest.approx(dist_map[9, 9], abs=1e-3) == 2.5


def test_measure_blob_fwhm_gaussian():
    """Verify measure_blob_fwhm on synthetic 2D Gaussian image."""
    a = 15
    y, x = np.ogrid[:a, :a]
    sigma_px = 2.0
    im = np.exp(-((y - 7) ** 2 + (x - 7) ** 2) / (2.0 * sigma_px ** 2))

    dx_um = 0.4
    # Analytical FWHM = 2 * sigma * sqrt(2 * ln(2)) * dx_um = 2 * 2.0 * 1.17741 * 0.4 = 1.8838 um
    expected_fwhm = 2.0 * sigma_px * np.sqrt(2.0 * np.log(2.0)) * dx_um

    fwhm = measure_blob_fwhm(im, dx_um=dx_um)
    assert pytest.approx(fwhm, rel=0.08) == expected_fwhm
