"""Regression tests for cell_mask4 against MATLAB mfiles/cellMask4.m."""

import os
import shutil
import subprocess

import numpy as np
import pytest
import scipy.io

from sparksimple.preprocessing.cell_mask import cell_mask4


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


def test_cell_mask4_matlab_parity(tmp_path, matlab_reference_mfiles):
    MFILES_DIR = str(matlab_reference_mfiles)

    """Verify cell_mask4 output matches MATLAB cellMask4.m pixel-by-pixel."""

    # 1. Create a controlled synthetic 2D cell image (60x80)
    ny, nx = 60, 80
    y, x = np.ogrid[:ny, :nx]
    center_y, center_x = 30, 40

    dist_sq = ((y - center_y) / 20.0) ** 2 + ((x - center_x) / 25.0) ** 2
    cell_region = dist_sq <= 1.0

    img = np.zeros((ny, nx), dtype=np.float64)
    # Background around 20, cell interior around 200
    np.random.seed(42)
    img[~cell_region] = np.random.uniform(10.0, 30.0, size=np.count_nonzero(~cell_region))
    img[cell_region] = np.random.uniform(150.0, 240.0, size=np.count_nonzero(cell_region))

    # Add external noise blob and internal hole to test full pipeline
    img[2:6, 2:6] = 230.0
    img[28:32, 38:42] = 15.0

    # 2. Run Python cell_mask4
    py_mask = cell_mask4(img)

    # 3. Save matrix to temporary MAT file to avoid command-line length limits on Windows
    mat_in_file = tmp_path / "test_img_in.mat"
    mat_out_file = tmp_path / "test_mask_out.mat"
    scipy.io.savemat(str(mat_in_file), {"mk_in": img})

    mfiles_posix = MFILES_DIR.replace("\\", "/")
    in_posix = str(mat_in_file).replace("\\", "/")
    out_posix = str(mat_out_file).replace("\\", "/")

    matlab_script = (
        f"addpath('{mfiles_posix}'); "
        f"data = load('{in_posix}'); "
        f"mask_out = cellMask4(data.mk_in); "
        f"save('{out_posix}', 'mask_out');"
    )

    subprocess.run(
        ["matlab", "-batch", matlab_script],
        capture_output=True,
        text=True,
        check=True,
    )

    # 4. Load MATLAB output
    mat_res = scipy.io.loadmat(str(mat_out_file))
    mat_mask = mat_res["mask_out"].astype(bool)

    # 5. Parity metrics
    pixel_diff = np.count_nonzero(py_mask != mat_mask)
    intersection = np.count_nonzero(py_mask & mat_mask)
    union = np.count_nonzero(py_mask | mat_mask)
    iou = intersection / union if union > 0 else 1.0

    print(f"MATLAB vs Python mask IoU: {iou:.6f}")
    print(f"Pixel differences: {pixel_diff} / {ny * nx}")

    # Assert 100% exact binary agreement (IoU == 1.0)
    assert iou == 1.0
    assert pixel_diff == 0
