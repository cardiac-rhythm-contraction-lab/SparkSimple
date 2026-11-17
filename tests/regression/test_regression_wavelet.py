"""Regression tests for wavelet transform and V2 construction against MATLAB sparkDetect.m."""

import os
import shutil
import subprocess

import numpy as np
import pytest
import scipy.io

from sparksimple.config import PipelineConfig
from sparksimple.detection.smooth import smooth_moving_average_1d
from sparksimple.detection.wavelet import compute_v2

def _matlab_regression_enabled() -> bool:
    """Opt-in only: never spawn/probe MATLAB unless explicitly requested.

    Merely checking "is matlab on PATH" by running `matlab -batch ...` can
    itself launch MATLAB and trigger an interactive license/sign-in prompt
    (a browser tab) on machines where MATLAB is installed but not configured
    for non-interactive/network-license use -- and the test body runs a real
    MATLAB CLI invocation regardless. So this never touches MATLAB at all
    unless a human explicitly sets RUN_MATLAB_REGRESSION_TESTS=1.
    """
    return os.environ.get("RUN_MATLAB_REGRESSION_TESTS") == "1"


pytestmark = pytest.mark.skipif(
    not _matlab_regression_enabled(),
    reason="MATLAB regression test skipped by default (set RUN_MATLAB_REGRESSION_TESTS=1 to opt in)",
)


MFILES_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "mfiles",
    )
)


def test_smooth_matlab_parity(tmp_path, matlab_reference_mfiles):
    MFILES_DIR = str(matlab_reference_mfiles)
    """Verify smooth_moving_average_1d matches MATLAB smooth(sig, w) exactly."""
    np.random.seed(42)
    sig = np.random.uniform(0.0, 10.0, size=50)
    w = 7

    py_smooth = smooth_moving_average_1d(sig, w)

    mat_in_file = tmp_path / "smooth_in.mat"
    mat_out_file = tmp_path / "smooth_out.mat"
    scipy.io.savemat(str(mat_in_file), {"sig": sig, "w": int(w)})

    in_posix = str(mat_in_file).replace("\\", "/")
    out_posix = str(mat_out_file).replace("\\", "/")

    cmd = (
        f"data = load('{in_posix}'); "
        f"sm = smooth(double(data.sig), double(data.w(1))); "
        f"save('{out_posix}', 'sm');"
    )

    subprocess.run(
        ["matlab", "-batch", cmd],
        capture_output=True,
        text=True,
        check=True,
    )

    mat_res = scipy.io.loadmat(str(mat_out_file))
    mat_smooth = mat_res["sm"].squeeze()

    max_diff = np.max(np.abs(py_smooth - mat_smooth))
    print(f"MATLAB vs Python smooth max diff: {max_diff:.6e}")
    np.testing.assert_allclose(py_smooth, mat_smooth, rtol=1e-12, atol=1e-12)


def test_v2_matlab_parity(tmp_path, matlab_reference_mfiles):
    MFILES_DIR = str(matlab_reference_mfiles)

    """Verify compute_v2 output matches MATLAB sparkDetect.m (lines 137-163) for 3D volume."""
    # Create controlled synthetic 3D volume (Y=8, X=8, T=60)
    ny, nx, nt = 8, 8, 60
    np.random.seed(123)
    vol = np.random.uniform(-0.5, 1.5, size=(ny, nx, nt)).astype(np.float64)
    dt_ms = 2.0
    w_ms = 40.0
    w = int(round(w_ms))
    scala = np.array([5, 6, 7, 8, 9, 10], dtype=np.float64)

    # 1. Run Python compute_v2
    cfg = PipelineConfig()
    cfg.detection.filter_window_frames = w
    res = compute_v2(vol, dt_ms=dt_ms, is_simulated=False, cfg=cfg)
    py_v2 = res.v2

    # 2. Run MATLAB CWT block
    mat_in_file = tmp_path / "v2_in.mat"
    mat_out_file = tmp_path / "v2_out.mat"
    scipy.io.savemat(
        str(mat_in_file),
        {"volum": vol, "scala": scala, "w": int(w)},
    )

    in_posix = str(mat_in_file).replace("\\", "/")
    out_posix = str(mat_out_file).replace("\\", "/")

    matlab_script = (
        f"data = load('{in_posix}'); "
        f"volum = double(data.volum); "
        f"scala = double(data.scala); "
        f"w = double(data.w(1)); "
        f"[a, b, temps] = size(volum); "
        f"V2 = zeros(size(volum)); "
        f"for ii = 1:a; "
        f"  for jj = 1:b; "
        f"    sig = volum(ii, jj, :); "
        f"    sig = permute(sig, [3 1 2]); "
        f"    sig = smooth(sig, w); "
        f"    sig = sig - mean(sig); "
        f"    C = cwt(sig, scala, 'gaus2'); "
        f"    C = sum(C, 1); "
        f"    V2(ii, jj, :) = permute(C, [1 3 2]); "
        f"  end; "
        f"end; "
        f"save('{out_posix}', 'V2');"
    )

    subprocess.run(
        ["matlab", "-batch", matlab_script],
        capture_output=True,
        text=True,
        check=True,
    )

    # 3. Compare outputs
    mat_res = scipy.io.loadmat(str(mat_out_file))
    mat_v2 = mat_res["V2"].astype(np.float32)

    max_diff = np.max(np.abs(py_v2 - mat_v2))
    corr = np.corrcoef(py_v2.ravel(), mat_v2.ravel())[0, 1]

    print(f"MATLAB vs Python V2 max diff: {max_diff:.6e}, correlation: {corr:.8f}")

    # Assert high correlation and low absolute difference
    assert corr > 0.999
    np.testing.assert_allclose(py_v2, mat_v2, rtol=1e-3, atol=1e-3)
