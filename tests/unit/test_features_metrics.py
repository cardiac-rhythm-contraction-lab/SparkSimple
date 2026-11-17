"""Unit tests for trace parameter metrics (compute_trace_parameters)."""

import numpy as np
import pytest

from sparksimple.features.metrics import compute_trace_parameters


def test_compute_trace_parameters_synthetic_spark():
    """Verify compute_trace_parameters on synthetic spark trace."""
    dt_ms = 2.0
    t_frames = np.arange(0, 100)
    t_axis_ms = t_frames * dt_ms

    baseline_val = 0.1
    amp_val = 0.8
    tau_val = 30.0

    trace = baseline_val * np.ones_like(t_axis_ms)
    peak_frame = 40  # t = 80 ms

    # Linear uprise from t=60ms (frame 30) to t=80ms (frame 40)
    uprise_len = 10
    trace[peak_frame - uprise_len : peak_frame] = baseline_val + np.linspace(0.0, amp_val, uprise_len)

    # Exponential decay after peak
    t_decay = t_axis_ms[peak_frame:] - t_axis_ms[peak_frame]
    trace[peak_frame:] = baseline_val + amp_val * np.exp(-t_decay / tau_val)

    cut = trace
    peak_idx_rel = peak_frame

    params = compute_trace_parameters(cut, t_axis_ms, peak_idx_rel)

    assert pytest.approx(params["AMP"], abs=1e-2) == baseline_val + amp_val
    assert pytest.approx(params["BL"], abs=1e-2) == baseline_val
    assert pytest.approx(params["amp"], abs=1e-2) == amp_val
    assert pytest.approx(params["tau"], rel=0.1) == tau_val
    assert params["ror"] > 0.0
    assert params["t2p"] > 0.0
    assert params["fdhm"] > 0.0
    assert params["r2"] > 0.8
    assert params["mass"] > 0.0
