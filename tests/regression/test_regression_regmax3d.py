"""Regression test validating Phase 6B (RegMax3D and IX matrix construction) against MATLAB R2025b CLI.

Mirrors mfiles/sparkDetect.m line 199 (IX = RegMax3D(V3, nc)) and mfiles/RegMax3D.m.
"""

import os
from pathlib import Path
import subprocess
import numpy as np
import pytest
import scipy.io as sio

from sparksimple.detection.local_max import (
    Candidate,
    detect_candidates,
    extract_candidates,
    regmax_3d,
)
from sparksimple.detection.merge import apply_3d_merge_and_threshold


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


# Skip regression test unless explicitly opted in (see _matlab_regression_enabled)
pytestmark = pytest.mark.skipif(
    not _matlab_regression_enabled(),
    reason="MATLAB regression test skipped by default (set RUN_MATLAB_REGRESSION_TESTS=1 to opt in)",
)


def test_regression_regmax3d_against_matlab(tmp_path: Path, matlab_reference_mfiles: Path):
    """Compare RegMax3D Python implementation directly against MATLAB CLI execution."""
    mfiles_dir = matlab_reference_mfiles
    assert mfiles_dir.exists(), f"mfiles directory not found at {mfiles_dir}"

    # 1. Create synthetic 3D V2 volume and cell mask (similar to Phase 6A test volume)
    np.random.seed(456)
    ny, nx, nt = 32, 32, 50
    v2_input = np.random.uniform(0.0, 0.5, size=(ny, nx, nt)).astype(np.float64)

    # Add three distinct synthetic spark peaks with spatial extent
    yy, xx, tt = np.ogrid[:ny, :nx, :nt]
    spark1 = 8.0 * np.exp(-((yy - 12) ** 2 + (xx - 12) ** 2 + (tt - 20) ** 2) / 4.0)
    spark2 = 6.0 * np.exp(-((yy - 20) ** 2 + (xx - 22) ** 2 + (tt - 35) ** 2) / 4.0)
    spark3 = 4.5 * np.exp(-((yy - 8) ** 2 + (xx - 24) ** 2 + (tt - 10) ** 2) / 4.0)
    v2_input += spark1 + spark2 + spark3

    mask_input = np.zeros((ny, nx), dtype=np.float64)
    mask_input[2:30, 2:30] = 1.0  # Cell mask

    dm_um = 2.5
    tm_ms = 15.0
    dx_um = 0.5
    dt_ms = 3.0
    nc = 50  # max candidates

    # Run Phase 6A Python pipeline to obtain V3_thresholded
    stage6a_res = apply_3d_merge_and_threshold(
        v2=v2_input,
        mask=mask_input,
        dx_um=dx_um,
        dt_ms=dt_ms,
        merge_distance_um=dm_um,
        merge_time_ms=tm_ms,
    )
    v3_thresh = stage6a_res.v3_thresholded

    # 2. Save test volume to MAT file for MATLAB CLI
    mat_in_path = tmp_path / "input_stage6b.mat"
    mat_out_path = tmp_path / "output_stage6b.mat"

    sio.savemat(
        str(mat_in_path),
        {
            "V3": v3_thresh,
            "nc": float(nc),
        },
    )

    # 3. Construct MATLAB batch script calling RegMax3D
    matlab_cmd = f"""
    addpath('{mfiles_dir.as_posix()}');
    load('{mat_in_path.as_posix()}');
    
    IX = RegMax3D(V3, nc);
    
    save('{mat_out_path.as_posix()}', 'IX');
    """

    res = subprocess.run(
        ["matlab", "-batch", matlab_cmd],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert res.returncode == 0, f"MATLAB execution failed: {res.stderr}\n{res.stdout}"
    assert mat_out_path.exists(), "MATLAB output file was not created"

    # 4. Load MATLAB reference IX matrix
    mat_out = sio.loadmat(str(mat_out_path))
    matlab_ix = mat_out["IX"].astype(np.float64)

    # 5. Execute Python Phase 6B implementation with temporal split disabled so
    #    the result matches MATLAB RegMax3D exactly (temporal_split_frac=0.3 is
    #    an intentional Python extension that departs from MATLAB, tested separately).
    py_res = detect_candidates(v3_thresh, max_candidates=nc, temporal_split_frac=0.0)
    py_ix = py_res.ix_matrix

    # 6. Parity Assertions and Measurements

    # a. IX Matrix dimensions
    assert py_ix.shape == matlab_ix.shape, f"Shape mismatch: Py={py_ix.shape}, Matlab={matlab_ix.shape}"

    # b. Non-zero Candidate Count
    py_cands_count = np.count_nonzero(py_ix > 0)
    matlab_cands_count = np.count_nonzero(matlab_ix > 0)
    print(f"\n[MATLAB Parity] Python candidates count: {py_cands_count}")
    print(f"[MATLAB Parity] MATLAB candidates count: {matlab_cands_count}")

    assert py_cands_count == matlab_cands_count, (
        f"Candidate count mismatch: Python={py_cands_count}, MATLAB={matlab_cands_count}"
    )

    # c. Binary mask of candidate locations
    py_mask = py_ix > 0
    matlab_mask = matlab_ix > 0
    mask_equal = np.array_equal(py_mask, matlab_mask)
    assert mask_equal, "Candidate spatial locations mismatch between Python and MATLAB"

    # d. Exact numerical step_level values in IX matrix
    ix_max_diff = np.max(np.abs(py_ix - matlab_ix))
    print(f"[MATLAB Parity] IX Matrix Max Abs Diff: {ix_max_diff:.6e}")
    assert ix_max_diff < 1e-12, f"IX matrix numerical mismatch: max diff = {ix_max_diff}"

    # e. Candidate list 1-based and 0-based coordinate parity
    py_coords_1b = set((c.y_1b, c.x_1b, c.t_1b) for c in py_res.candidates)

    matlab_y1b, matlab_x1b, matlab_t1b = np.nonzero(matlab_ix > 0)
    matlab_coords_1b = set(
        (int(y) + 1, int(x) + 1, int(t) + 1)
        for y, x, t in zip(matlab_y1b, matlab_x1b, matlab_t1b)
    )

    assert py_coords_1b == matlab_coords_1b, (
        f"1-based candidate coordinates mismatch!\nPython: {py_coords_1b}\nMATLAB: {matlab_coords_1b}"
    )
