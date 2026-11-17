"""tests/integration/test_io.py — Integration tests for sparksimple.io.

These tests load real data from the filesystem and verify that the stack shape
is consistent with what the metadata reports — not against hardcoded values.

Skipped automatically if the data folders are missing (via conftest fixtures).
"""

import numpy as np
import pytest

from sparksimple.io import load_image_stack, load_metadata


# ── Simulated experiment ──────────────────────────────────────────────────────

class TestSimulatedExperiment:
    """Load the single-channel simulated dataset (Cabcnew*.tif)."""

    def test_stack_loads(self, require_sim_data):
        stack, ch = load_image_stack(require_sim_data)
        assert stack is not None
        assert isinstance(stack, np.ndarray)

    def test_stack_dtype(self, require_sim_data):
        stack, _ = load_image_stack(require_sim_data)
        assert stack.dtype == np.float32

    def test_stack_ndim(self, require_sim_data):
        stack, _ = load_image_stack(require_sim_data)
        assert stack.ndim == 3  # (rows, cols, frames)

    def test_stack_shape_matches_metadata(self, require_sim_data):
        """Stack dimensions must be consistent with the XML metadata."""
        meta = load_metadata(require_sim_data)
        stack, _ = load_image_stack(require_sim_data)

        n_rows, n_cols, n_frames = stack.shape
        assert n_rows == meta.n_pixels_y, (
            f"rows {n_rows} != metadata n_pixels_y {meta.n_pixels_y}"
        )
        assert n_cols == meta.n_pixels_x, (
            f"cols {n_cols} != metadata n_pixels_x {meta.n_pixels_x}"
        )
        assert n_frames == meta.n_frames, (
            f"frames {n_frames} != metadata n_frames {meta.n_frames}"
        )

    def test_channel_returned_is_zero(self, require_sim_data):
        """Single-channel dataset → channel 0 always selected."""
        _, ch = load_image_stack(require_sim_data)
        assert ch == 0

    def test_no_nan_values(self, require_sim_data):
        stack, _ = load_image_stack(require_sim_data)
        assert not np.any(np.isnan(stack))

    def test_no_all_zeros_frames(self, require_sim_data):
        """No frame should be entirely zero (would indicate a loading error)."""
        stack, _ = load_image_stack(require_sim_data)
        frame_sums = stack.sum(axis=(0, 1))
        assert np.all(frame_sums > 0), (
            f"Frames with zero sum: {np.where(frame_sums == 0)[0].tolist()}"
        )

    def test_forced_channel_zero(self, require_sim_data):
        """Forced channel=0 must succeed and return the same shape."""
        stack_auto, _ = load_image_stack(require_sim_data)
        stack_forced, ch = load_image_stack(require_sim_data, channel=0)
        assert ch == 0
        assert stack_auto.shape == stack_forced.shape


# ── Real experiment ───────────────────────────────────────────────────────────

class TestRealExperiment:
    """Load the three-channel real dataset (MVM_..._chNN.tif)."""

    def test_stack_loads(self, require_real_data):
        stack, ch = load_image_stack(require_real_data, channel="ch01")
        assert stack is not None
        assert ch == 1

    def test_stack_dtype(self, require_real_data):
        stack, _ = load_image_stack(require_real_data, channel="ch01")
        assert stack.dtype == np.float32

    def test_stack_ndim(self, require_real_data):
        stack, _ = load_image_stack(require_real_data, channel="ch01")
        assert stack.ndim == 3

    def test_stack_shape_matches_metadata(self, require_real_data):
        """Shape must agree with what the XML says — no hardcoded sizes."""
        meta = load_metadata(require_real_data)
        stack, _ = load_image_stack(require_real_data, channel="ch01")

        n_rows, n_cols, n_frames = stack.shape
        assert n_rows == meta.n_pixels_y
        assert n_cols == meta.n_pixels_x
        assert n_frames == meta.n_frames

    def test_multichannel_requires_explicit_channel(self, require_real_data):
        """A calcium analysis must never guess which recorded channel to use."""
        with pytest.raises(ValueError, match="explicit channel"):
            load_image_stack(require_real_data)

    def test_no_nan_values(self, require_real_data):
        stack, _ = load_image_stack(require_real_data, channel="ch01")
        assert not np.any(np.isnan(stack))

    def test_forced_channel_out_of_range_raises(self, require_real_data):
        meta = load_metadata(require_real_data)
        with pytest.raises(ValueError, match="channel"):
            load_image_stack(require_real_data, channel=meta.n_channels + 10)

    def test_forced_channel_zero(self, require_real_data):
        meta = load_metadata(require_real_data)
        stack, ch = load_image_stack(require_real_data, channel=0)
        assert ch == 0
        n_rows, n_cols, n_frames = stack.shape
        assert n_rows == meta.n_pixels_y
        assert n_cols == meta.n_pixels_x
        assert n_frames == meta.n_frames

    def test_different_channels_differ(self, require_real_data):
        """Channels 0 and 1 should not be pixel-identical (they are different dyes)."""
        stack_0, _ = load_image_stack(require_real_data, channel=0)
        stack_1, _ = load_image_stack(require_real_data, channel=1)
        assert not np.allclose(stack_0, stack_1), (
            "Channel 0 and channel 1 are identical — loading may be broken"
        )
