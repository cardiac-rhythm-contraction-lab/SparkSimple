"""Regression test validating Phase 8 (Feature Extraction) against MATLAB R2025b CLI.

Mirrors mfiles/sparkFeatures.m and mfiles/measureFeatures.m.
"""

import os
from pathlib import Path
import subprocess
import numpy as np
import pytest
import scipy.io as sio

from sparksimple.detection.local_max import detect_candidates
from sparksimple.features import extract_features


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


def test_regression_features_against_matlab(tmp_path: Path, matlab_reference_mfiles: Path):
    """Compare Feature Extraction Python implementation directly against MATLAB CLI execution."""
    mfiles_dir = matlab_reference_mfiles
    assert mfiles_dir.exists(), f"mfiles directory not found at {mfiles_dir}"

    # 1. Create synthetic 3D volume and cell mask
    ny, nx, nt = 32, 32, 70
    volume = np.random.uniform(0.0, 0.05, size=(ny, nx, nt)).astype(np.float64)

    # Add two synthetic sparks with temporal uprise and exponential decay
    yy, xx = np.ogrid[:ny, :nx]

    # Spark 1 at (12, 12, 30)
    spark1_space = np.exp(-((yy - 12) ** 2 + (xx - 12) ** 2) / 4.0)
    spark1_time = np.zeros(nt)
    spark1_time[20:30] = np.linspace(0.0, 0.8, 10)
    spark1_time[30:] = 0.8 * np.exp(-(np.arange(30, nt) - 30) / 15.0)
    volume += spark1_space[:, :, np.newaxis] * spark1_time[np.newaxis, np.newaxis, :]

    # Spark 2 at (20, 22, 45)
    spark2_space = np.exp(-((yy - 20) ** 2 + (xx - 22) ** 2) / 4.0)
    spark2_time = np.zeros(nt)
    spark2_time[35:45] = np.linspace(0.0, 0.6, 10)
    spark2_time[45:] = 0.6 * np.exp(-(np.arange(45, nt) - 45) / 20.0)
    volume += spark2_space[:, :, np.newaxis] * spark2_time[np.newaxis, np.newaxis, :]

    mask = np.zeros((ny, nx), dtype=np.float64)
    mask[2:30, 2:30] = 1.0

    dx_um = 0.5
    dt_ms = 2.0
    roi_r_um = 1.5
    marg_t_ms = (80.0, 80.0)

    # Construct IX candidate matrix (marking peak locations)
    v3_thresholded = np.zeros_like(volume)
    v3_thresholded[12, 12, 30] = 1.0
    v3_thresholded[20, 22, 45] = 0.8
    cand_res = detect_candidates(v3_thresholded, max_candidates=10)
    ix_matrix = cand_res.ix_matrix

    # 2. Save test data to MAT files matching MATLAB sparkFeatures requirements
    zdata0_path = tmp_path / "zData0.mat"
    zdata1_path = tmp_path / "zData1.mat"
    zmetadata_path = tmp_path / "zMetaData.mat"
    ys_path = tmp_path / "ys.mat"

    sio.savemat(str(zdata0_path), {"volum": volume, "yesno": 0.0})
    sio.savemat(str(zdata1_path), {"IX": ix_matrix})
    sio.savemat(
        str(zmetadata_path),
        {
            "fo": mask,
            "mask": mask,
            "temps": float(nt),
            "DX": float(dx_um),
            "DT": float(dt_ms),
            "roiR": float(roi_r_um),
        },
    )
    sio.savemat(str(ys_path), {"ys": 6.0})

    (tmp_path / "spkFeat").mkdir(parents=True, exist_ok=True)

    # 3. Execute MATLAB CLI
    matlab_cmd = f"""
    addpath('{mfiles_dir.as_posix()}');
    cd('{tmp_path.as_posix()}');
    
    volum = load('{zdata0_path.as_posix()}', 'volum').volum;
    IX = load('{zdata1_path.as_posix()}', 'IX').IX;
    
    DX = {dx_um};
    DT = {dt_ms};
    roiR = {roi_r_um};
    margT = [{marg_t_ms[0]} {marg_t_ms[1]}];
    Rfol = '{tmp_path.as_posix()}';
    if ~exist([Rfol '/spkFeat'], 'dir'), mkdir([Rfol '/spkFeat']); end
    mask = load([Rfol '/zMetaData.mat'], 'mask').mask;
    
    md = bwdist(1 - mask);
    spkF = measureFeatures(volum, IX, DX, DT, Rfol, roiR, margT);
    
    for ii = 1:length(spkF)
        spkF(ii).dist2memb = md(spkF(ii).py, spkF(ii).px) * DX;
    end
    
    save('{tmp_path.as_posix()}/zData2.mat', 'spkF');
    """

    res = subprocess.run(
        ["matlab", "-batch", matlab_cmd],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert res.returncode == 0, f"MATLAB execution failed: {res.stderr}\n{res.stdout}"

    mat_out = sio.loadmat(str(tmp_path / "zData2.mat"), squeeze_me=True, struct_as_record=False)
    matlab_spkf = mat_out["spkF"]

    if not isinstance(matlab_spkf, np.ndarray):
        matlab_spkf = [matlab_spkf]

    # 4. Execute Python implementation
    py_res = extract_features(
        volume=volume,
        ix_matrix=ix_matrix,
        mask=mask,
        dx_um=dx_um,
        dt_ms=dt_ms,
        roi_r_um=roi_r_um,
        marg_t_ms=marg_t_ms,
    )

    # 5. Assert parity between Python and MATLAB
    assert py_res.count == len(matlab_spkf), f"Candidate count mismatch: Py={py_res.count}, MATLAB={len(matlab_spkf)}"

    print(f"\n[MATLAB Parity] Candidate count matched: {py_res.count}")

    for idx, (py_f, mat_f) in enumerate(zip(py_res.features, matlab_spkf)):
        # Coordinates
        assert py_f.px == mat_f.px, f"Spark #{idx+1} px mismatch: Py={py_f.px}, MATLAB={mat_f.px}"
        assert py_f.py == mat_f.py, f"Spark #{idx+1} py mismatch: Py={py_f.py}, MATLAB={mat_f.py}"
        assert py_f.pt == mat_f.pt, f"Spark #{idx+1} pt mismatch: Py={py_f.pt}, MATLAB={mat_f.pt}"

        # Distance to membrane
        dist_diff = abs(py_f.dist2memb - mat_f.dist2memb)
        print(f"[Spark #{idx+1}] dist2memb Py={py_f.dist2memb:.4f}, MATLAB={mat_f.dist2memb:.4f}, diff={dist_diff:.6e}")
        assert dist_diff < 1e-12

        # FWHM spatial width
        fwhm_diff = abs(py_f.fwhm - mat_f.FWHM)
        print(f"[Spark #{idx+1}] FWHM Py={py_f.fwhm:.4f}, MATLAB={mat_f.FWHM:.4f}, diff={fwhm_diff:.6e}")
        assert fwhm_diff < 0.15  # Numerical curve_fit tolerance

        # Time signal array
        ts_mat = np.asarray(mat_f.timesignal, dtype=np.float64).ravel()
        ts_diff = np.max(np.abs(py_f.timesignal - ts_mat))
        print(f"[Spark #{idx+1}] timesignal max diff={ts_diff:.6e}")
        assert ts_diff < 1e-12

        # Scalar feature metrics
        amp_diff = abs(py_f.amp - mat_f.amp)
        amp_max_diff = abs(py_f.AMP - mat_f.AMP)
        bl_diff = abs(py_f.BL - mat_f.BL)
        mass_diff = abs(py_f.mass - mat_f.mass)

        print(f"[Spark #{idx+1}] amp Py={py_f.amp:.4f}, MATLAB={mat_f.amp:.4f}, diff={amp_diff:.6e}")
        print(f"[Spark #{idx+1}] AMP Py={py_f.AMP:.4f}, MATLAB={mat_f.AMP:.4f}, diff={amp_max_diff:.6e}")
        print(f"[Spark #{idx+1}] BL Py={py_f.BL:.4f}, MATLAB={mat_f.BL:.4f}, diff={bl_diff:.6e}")
        print(f"[Spark #{idx+1}] mass Py={py_f.mass:.4f}, MATLAB={mat_f.mass:.4f}, diff={mass_diff:.6e}")

        assert amp_diff < 1e-3
        assert amp_max_diff < 1e-12
        assert bl_diff < 1e-3
        assert mass_diff < 1e-3
