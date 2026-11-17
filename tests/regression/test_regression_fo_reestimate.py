"""Regression test for Phase 7 post-detection Fo re-estimation against MATLAB CLI."""

import os
from pathlib import Path
import subprocess
import tempfile
import numpy as np
import pytest
from scipy.io import loadmat, savemat

from sparksimple.detection.fo_reestimate import reestimate_fo


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


pytestmark = pytest.mark.skipif(
    not _matlab_regression_enabled(),
    reason="MATLAB regression test skipped by default (set RUN_MATLAB_REGRESSION_TESTS=1 to opt in)",
)


def test_regression_fo_reestimate_matlab_cli(tmp_path: Path, matlab_reference_mfiles: Path):
    """Verify 100% numerical parity of reestimate_fo against MATLAB CLI sparkDetect.m lines 252–314."""
    mfiles_dir = matlab_reference_mfiles
    assert mfiles_dir.exists(), f"mfiles directory not found at {mfiles_dir}"

    # Generate reproducible synthetic 3D test volume and mask
    rng = np.random.default_rng(42)
    rows, cols, frames = 25, 25, 50
    volume_raw = 100.0 + rng.normal(0, 3.0, size=(rows, cols, frames))

    # Add candidate events in V3 (thresholded candidate volume)
    v3_thresholded = np.zeros((rows, cols, frames), dtype=np.float64)
    v3_thresholded[10, 10, 20] = 5.0
    v3_thresholded[15, 12, 35] = 8.0

    # Mask with circular blob
    y, x = np.ogrid[:rows, :cols]
    center_y, center_x = 12, 12
    dist_from_center = np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)
    mask = (dist_from_center <= 9.0).astype(np.float64)

    dx_um = 0.3
    dt_ms = 5.0
    roi_r_um = 1.5
    estimated_fo = 100.0

    # Save inputs to temporary MAT file for MATLAB execution
    mat_in = tmp_path / "inputs.mat"
    mat_out = tmp_path / "outputs.mat"

    savemat(
        str(mat_in),
        {
            "volum_raw": volume_raw,
            "V3": v3_thresholded,
            "mk": mask,
            "DX": dx_um,
            "DT": dt_ms,
            "roiR": roi_r_um,
            "estimatedFo": estimated_fo,
            "forcedbaseline": 0.0,
            "optB": 1,
        },
    )

    # Run MATLAB snippet mirroring sparkDetect.m lines 254–312 exactly
    matlab_cmd = f"""
    addpath('{mfiles_dir.as_posix()}');
    load('{mat_in.as_posix()}');
    volum = volum_raw;
    suptemp = ceil(68/DT);
    B = gaussiana2d(ceil(2/DX));
    for ii=2:suptemp, B(:,:,ii) = B(:,:,1); end
    B = B / sum(sum(sum(B)));
    vaux = conv3(volum, B);
    vaux(V3 > 0) = -1;
    a = size(mk, 1); b = size(mk, 2); temps = size(volum, 3);
    reestimatedFo = zeros(size(mk));
    for ii=1:a, for jj=1:b,
      vec = reshape(vaux(ii,jj,:), [temps, 1]); vec(vec==-1) = [];
      if isempty(vec), reestimatedFo(ii,jj) = estimatedFo;
      else, reestimatedFo(ii,jj) = min(vec) + std(vec)*0.5; end
    end; end
    Fo = reestimatedFo;
    ra = round(roiR/DX); ra = ra - mod(ra, 2) + 1; ker = ones(ra, ra);
    aux = mk - imerode(mk, ker); gg = find(aux > 0); mkz = zeros(size(mk));
    Foo = Fo; tots = length(gg);
    for ii=1:tots,
      thismk = mkz; thismk(gg(ii)) = 1; thismk = conv2(thismk, ker, 'same');
      Foo(gg(ii)) = max(max(thismk .* Fo));
    end
    Fo = Foo;
    volum_renorm = zeros(size(volum));
    for ii=1:temps, volum_renorm(:,:,ii) = (volum(:,:,ii) - Fo) ./ Fo; end
    save('{mat_out.as_posix()}', 'Fo', 'volum_renorm', 'vaux', 'aux');
    """

    res = subprocess.run(
        ["matlab", "-batch", matlab_cmd],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert res.returncode == 0, f"MATLAB execution failed: {res.stderr}\nOutput: {res.stdout}"
    assert mat_out.exists(), "MATLAB did not produce outputs.mat"

    mat_data = loadmat(str(mat_out))
    mat_fo = mat_data["Fo"]
    mat_vol_renorm = mat_data["volum_renorm"]
    mat_aux = mat_data["aux"] > 0

    # Compute Python result
    py_res = reestimate_fo(
        volume_raw=volume_raw,
        v3_thresholded=v3_thresholded,
        mask=mask,
        estimated_fo=estimated_fo,
        dx_um=dx_um,
        dt_ms=dt_ms,
        roi_r_um=roi_r_um,
        baseline_mode="auto",
        normalisation_mode="df_f0",
    )

    # 1. Border adjustment mask boolean parity
    assert np.array_equal(py_res.border_adjustment_mask, mat_aux)

    # 2. Fo map parity
    max_abs_diff_fo = np.max(np.abs(py_res.fo_map - mat_fo))
    rel_err_fo = max_abs_diff_fo / np.max(np.abs(mat_fo))
    print(f"\n[MATLAB Parity] Fo map max abs diff: {max_abs_diff_fo:.6e}, rel err: {rel_err_fo:.6e}")
    assert max_abs_diff_fo < 1e-12, f"Fo map diff {max_abs_diff_fo} exceeds threshold"

    # 3. Volume renormalised parity
    max_abs_diff_vol = np.max(np.abs(py_res.volume_renormalised - mat_vol_renorm))
    rel_err_vol = max_abs_diff_vol / np.max(np.abs(mat_vol_renorm))
    print(f"[MATLAB Parity] Volume renormalised max abs diff: {max_abs_diff_vol:.6e}, rel err: {rel_err_vol:.6e}")
    assert max_abs_diff_vol < 1e-12, f"Volume renormalised diff {max_abs_diff_vol} exceeds threshold"
