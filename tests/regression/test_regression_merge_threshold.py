"""Regression test validating Phase 6A (3D merge kernel, conv3, thresholding) against MATLAB R2025b CLI.

Mirrors mfiles/sparkDetect.m lines 173–198, mfiles/gaussiana2d.m, and mfiles/conv3.m.
"""

import os
from pathlib import Path
import subprocess
import tempfile
import numpy as np
import pytest
import scipy.io as sio

from sparksimple.detection.convolution import conv3_matlab
from sparksimple.detection.kernel import build_merge_kernel_3d
from sparksimple.detection.merge import apply_3d_merge_and_threshold
from sparksimple.detection.threshold import (
    apply_threshold,
    estimate_noise_threshold,
    get_mask_center_pixel_coords,
)


def _matlab_regression_enabled() -> bool:
    """Opt-in only: never spawn/probe MATLAB unless explicitly requested.

    Merely checking "is matlab on PATH" by running `matlab -batch ...` can
    itself launch MATLAB and trigger an interactive license/sign-in prompt
    (a browser tab) on machines where MATLAB is installed but not configured
    for non-interactive/network-license use -- and the test body below runs
    a real MATLAB CLI invocation regardless. So this never touches MATLAB at
    all unless a human explicitly sets RUN_MATLAB_REGRESSION_TESTS=1.
    """
    return os.environ.get("RUN_MATLAB_REGRESSION_TESTS") == "1"


# Skip regression test unless explicitly opted in (see _matlab_regression_enabled)
pytestmark = pytest.mark.skipif(
    not _matlab_regression_enabled(),
    reason="MATLAB regression test skipped by default (set RUN_MATLAB_REGRESSION_TESTS=1 to opt in)",
)


def test_regression_merge_threshold_against_matlab(tmp_path: Path, matlab_reference_mfiles: Path):
    """Compare Phase 6A outputs directly against MATLAB CLI execution."""
    mfiles_dir = matlab_reference_mfiles
    assert mfiles_dir.exists(), f"mfiles directory not found at {mfiles_dir}"

    # 1. Create synthetic test input volume V2 and cell mask
    np.random.seed(123)
    ny, nx, nt = 32, 32, 50
    v2_input = np.random.uniform(0.0, 1.0, size=(ny, nx, nt)).astype(np.float64)
    # Add a couple of synthetic spark peaks
    v2_input[15, 15, 20:25] += 5.0
    v2_input[22, 10, 30:35] += 4.0

    mask_input = np.zeros((ny, nx), dtype=np.float64)
    mask_input[4:28, 4:28] = 1.0  # Square cell mask

    dm_um = 2.5
    tm_ms = 15.0
    dx_um = 0.5
    dt_ms = 3.0

    # 2. Save test inputs to MAT file for MATLAB CLI
    mat_in_path = tmp_path / "input_stage6a.mat"
    mat_out_path = tmp_path / "output_stage6a.mat"

    sio.savemat(
        str(mat_in_path),
        {
            "V2": v2_input,
            "mk": mask_input,
            "dm": float(dm_um),
            "tm": float(tm_ms),
            "DX": float(dx_um),
            "DT": float(dt_ms),
        },
    )

    # 3. Construct MATLAB batch script
    matlab_cmd = f"""
    addpath('{mfiles_dir.as_posix()}');
    load('{mat_in_path.as_posix()}');
    
    t3 = ceil(2 * dm / DX);
    w3 = ceil(2 * tm / DT);
    
    B = gaussiana2d(t3);
    for ii = 2:w3
        B(:,:,ii) = B(:,:,1);
    end
    B = B / sum(sum(sum(B)));
    
    V3 = conv3(V2, B);
    
    [ax1, ax2] = find(mk == 1);
    y_center_1b = round(mean(ax1));
    x_center_1b = round(mean(ax2));
    aux6 = reshape(V3(y_center_1b, x_center_1b, :), 1, size(V3, 3));
    
    umbral = mad(aux6) * 2.22;
    
    V3_thresh = V3;
    V3_thresh(V3_thresh < umbral) = 0;
    
    save('{mat_out_path.as_posix()}', 'B', 'V3', 'y_center_1b', 'x_center_1b', 'umbral', 'V3_thresh');
    """

    res = subprocess.run(
        ["matlab", "-batch", matlab_cmd],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert res.returncode == 0, f"MATLAB execution failed: {res.stderr}\n{res.stdout}"
    assert mat_out_path.exists(), "MATLAB output file was not created"

    # 4. Load MATLAB reference outputs
    mat_out = sio.loadmat(str(mat_out_path))
    matlab_b = mat_out["B"]
    matlab_v3 = mat_out["V3"]
    matlab_y1b = int(mat_out["y_center_1b"].item())
    matlab_x1b = int(mat_out["x_center_1b"].item())
    matlab_umbral = float(mat_out["umbral"].item())
    matlab_v3_thresh = mat_out["V3_thresh"]

    # 5. Execute Python Phase 6A implementation
    py_res = apply_3d_merge_and_threshold(
        v2=v2_input,
        mask=mask_input,
        dx_um=dx_um,
        dt_ms=dt_ms,
        merge_distance_um=dm_um,
        merge_time_ms=tm_ms,
    )

    y_0b, x_0b, py_y1b, py_x1b = get_mask_center_pixel_coords(mask_input)

    # 6. Parity Assertions and Measurements

    # a. Kernel Parity
    kernel_max_diff = np.max(np.abs(py_res.kernel - matlab_b))
    print(f"\n[MATLAB Parity] Kernel Max Abs Diff: {kernel_max_diff:.6e}")
    assert kernel_max_diff < 1e-15, f"Kernel mismatch: {kernel_max_diff}"

    # b. Mask Centroid Pixel Parity
    assert (py_y1b, py_x1b) == (matlab_y1b, matlab_x1b), (
        f"Centroid 1-based pixel mismatch: Python=({py_y1b}, {py_x1b}), "
        f"MATLAB=({matlab_y1b}, {matlab_x1b})"
    )
    assert (y_0b, x_0b) == (py_res.centroid_pixel), "Internal centroid indexing mismatch"

    # c. 3D Convolved Volume (conv3_matlab) Parity
    v3_max_diff = np.max(np.abs(py_res.v3_conv - matlab_v3))
    v3_rel_err = v3_max_diff / np.max(np.abs(matlab_v3))
    v3_corr = np.corrcoef(py_res.v3_conv.ravel(), matlab_v3.ravel())[0, 1]

    print(f"[MATLAB Parity] V3 Conv Max Abs Diff: {v3_max_diff:.6e}")
    print(f"[MATLAB Parity] V3 Conv Rel Error:   {v3_rel_err:.6e}")
    print(f"[MATLAB Parity] V3 Conv Correlation: {v3_corr:.12f}")

    assert v3_corr > 0.999999999, f"Low correlation in convolved volume: {v3_corr}"

    # d. Noise Threshold Scalar Parity
    umbral_abs_diff = abs(py_res.umbral - matlab_umbral)
    print(f"[MATLAB Parity] Umbral Abs Diff:     {umbral_abs_diff:.6e}")
    assert umbral_abs_diff < 1e-12, f"Threshold mismatch: diff={umbral_abs_diff}"

    # e. Thresholded Volume Parity (binary mask and numerical accuracy)
    py_zero_mask = py_res.v3_thresholded == 0.0
    matlab_zero_mask = matlab_v3_thresh == 0.0
    mask_parity = np.array_equal(py_zero_mask, matlab_zero_mask)
    assert mask_parity, "Zero-threshold mask parity failed between Python and MATLAB"

    v3_thresh_max_diff = np.max(np.abs(py_res.v3_thresholded - matlab_v3_thresh))
    print(f"[MATLAB Parity] V3 Thresh Max Abs Diff: {v3_thresh_max_diff:.6e}")
    assert v3_thresh_max_diff < 1e-12, f"Thresholded volume mismatch: {v3_thresh_max_diff}"
