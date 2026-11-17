"""tests/regression/test_regression_baseline.py

Synthetic regression tests for the preprocessing baseline/normalisation block.

STATUS: PARTIALLY MATLAB-VALIDATED
----------------------------------
The quantile method used in preprocessing was validated against MATLAB R2025b
using controlled reference vectors.  These tests still use synthetic arrays,
but the expected values for quantile-related cases now reflect direct MATLAB
outputs rather than a provisional assumption.

These tests are still not a full end-to-end MATLAB regression suite for real
volumes.  They should be treated as a mix of:

- MATLAB-validated specification tests for quantile behaviour
- algorithm conformance tests for the remaining preprocessing formulas

See docs/progress.md for the current validation scope and remaining risks.
"""

from __future__ import annotations

import numpy as np
import pytest

from sparksimple.preprocessing.baseline import (
    apply_signal_sign,
    estimate_baseline_global,
)
from sparksimple.preprocessing.normalise import normalise_df_f0


# ── R3: Quantile method — synthetic validation ────────────────────────────────

class TestQuantileMethodSynthetic:
    """
    Validate quantile computation on a fully controlled synthetic case.

    The algorithm (sparkDetect.m lines 102-114) flattens the masked volume
    and takes quantile(flat, 0.05).  Here we construct a case where the
    expected quantile is analytically computable.

    MATLAB NOTE: the expected values in this class were checked directly
    against MATLAB R2025b and match NumPy ``method='hazen'``.
    """

    def test_5th_percentile_of_arithmetic_sequence(self):
        """
        Synthetic case: values = 1, 2, ..., 200 (flat in-mask pixels).

        MATLAB R2025b returns 10.5 for quantile(1:200, 0.05).
        This matches NumPy ``method='hazen'``.
        """
        flat = np.arange(1, 201, dtype=np.float32)
        vol = flat.reshape(10, 20, 1)
        mask = np.ones((10, 20), dtype=bool)

        fo = estimate_baseline_global(vol, mask, quantile=0.05)

        expected = 10.5
        assert fo == pytest.approx(expected, rel=1e-6), (
            f"Got fo={fo}, expected={expected}. "
            "This expected value was verified against MATLAB R2025b."
        )

    def test_quantile_excludes_out_of_mask_zeros(self):
        """
        Verify that mask exclusion works: in-mask values are 10..100,
        out-of-mask values are 0 (after multiplication by mask).

        The 5th percentile should be computed only over 10..100.
        """
        rows, cols, frames = 5, 5, 4
        vol = np.zeros((rows, cols, frames), dtype=np.float32)

        # In-mask pixels: rows 0-2 get values ascending from 10 to 100
        # Shape of in-mask region: (3, cols, frames) = (3, 5, 4) = 60 elements
        in_mask_vals = np.linspace(10, 100, 3 * cols * frames)
        vol[:3, :, :] = in_mask_vals.reshape(3, cols, frames).astype(np.float32)

        # Out-of-mask pixels: rows 3-4 get very high values (must be ignored)
        vol[3:, :, :] = 10000.0

        mask = np.zeros((rows, cols), dtype=bool)
        mask[:3, :] = True

        fo = estimate_baseline_global(vol, mask, quantile=0.05)

        # Reference: quantile of in-mask values only
        in_mask_flat = vol[:3, :, :].ravel()
        expected = float(np.quantile(in_mask_flat, 0.05, method="hazen"))
        assert fo == pytest.approx(expected, rel=1e-6)

    def test_uniform_volume_5th_percentile(self):
        """Uniform volume: any quantile returns the same constant."""
        vol = np.full((8, 8, 10), 250.0, dtype=np.float32)
        mask = np.ones((8, 8), dtype=bool)
        fo = estimate_baseline_global(vol, mask, quantile=0.05)
        assert fo == pytest.approx(250.0, rel=1e-5)


# ── R4: Normalisation formula — synthetic validation ──────────────────────────

class TestNormalisationFormulaSynthetic:
    """
    Validate ΔF/F₀ normalisation on a fully controlled synthetic case.

        MATLAB formula (sparkDetect.m line 118):
        volum = (volum - estimatedFo) ./ estimatedFo

    These values are independently computable and serve as a specification
    conformance check.
    """

    def test_df_f0_known_values(self):
        """
        volume = 500, Fo = 200  →  (500 - 200) / 200 = 1.5
        """
        vol = np.full((4, 4, 10), 500.0, dtype=np.float32)
        result = normalise_df_f0(vol, fo=200.0)
        np.testing.assert_allclose(result, 1.5, rtol=1e-5)

    def test_df_f0_below_baseline_is_negative(self):
        """
        When F < Fo: (F - Fo)/Fo < 0.
        volume = 50, Fo = 100  →  (50 - 100)/100 = -0.5
        """
        vol = np.full((3, 3, 5), 50.0, dtype=np.float32)
        result = normalise_df_f0(vol, fo=100.0)
        np.testing.assert_allclose(result, -0.5, rtol=1e-5)

    def test_df_f0_spatial_fo_map(self):
        """
        Per-pixel Fo: each pixel normalised by its own baseline.
        vol[i,j,:] = 300; fo[i,j] varies.
        Expected: (300 - fo[i,j]) / fo[i,j] for each pixel.
        """
        rows, cols, frames = 3, 3, 8
        vol = np.full((rows, cols, frames), 300.0, dtype=np.float32)
        fo = np.array([
            [100.0, 150.0, 200.0],
            [100.0, 150.0, 200.0],
            [100.0, 150.0, 200.0],
        ])
        result = normalise_df_f0(vol, fo)
        expected = np.zeros((rows, cols, frames), dtype=np.float32)
        for r in range(rows):
            for c in range(cols):
                expected[r, c, :] = (300.0 - fo[r, c]) / fo[r, c]
        np.testing.assert_allclose(result, expected, rtol=1e-5)


# ── Negative signal sign — synthetic validation ───────────────────────────────

class TestNegativeSignalSignSynthetic:
    """
    Validate the negative signal handling (sparkDetect.m line 94-96):

        volum = ss * volum;
        if (ss < 0), volum = volum - quantile(volum(:), [.1]); end
    """

    def test_negative_sign_min_is_zero_for_uniform(self):
        """
        Uniform volume negated: all values identical → quantile(v, .1) == v
        → offset == min → result == 0.
        """
        vol = np.full((5, 5, 10), 300.0, dtype=np.float32)
        result = apply_signal_sign(vol, -1)
        np.testing.assert_allclose(result, 0.0, atol=1e-6)

    def test_negative_sign_preserves_relative_ordering(self):
        """
        After negation+offset the relative ordering of pixels is preserved
        (negation reverses order; offset is a global constant).
        High values become small, low values become large.
        """
        # vol[0] = 100 (low), vol[-1] = 200 (high)
        vol = np.linspace(100, 200, 50).reshape(5, 5, 2).astype(np.float32)
        result = apply_signal_sign(vol, -1)
        # After negation: -200 (most negative), -100 (least negative)
        # After offset: smallest values come from originally largest
        # i.e., result[0,0,0] > result[-1,-1,-1]
        assert result[0, 0, 0] > result[-1, -1, -1]
