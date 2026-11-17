"""tests/regression/test_regression_roi_clustering.py — ROI clustering parity tests.

Verifies Python cluster_rois outputs against MATLAB clusterROIs.m reference data
stored in tests/data/*/Matlab_Result/zData2.mat.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import scipy.io as sio
import pytest

from sparksimple.features.models import SparkFeature
from sparksimple.filtering.models import FilteredSpark
from sparksimple.roi.cluster import cluster_rois


REAL_DATA_DIR = Path(__file__).parents[1] / "data" / "Real_Experiment" / "Matlab_Result"
SIM_DATA_DIR = Path(__file__).parents[1] / "data" / "Simulated_Experiment" / "Matlab_Result"


def _load_sparks_from_mat(zdata2_path: Path, zmetadata_path: Path) -> tuple:
    """Load FilteredSpark list and metadata from MATLAB result files.

    Returns (sparks, DX, DT, roiR, mask, volum_shape).
    """
    mat2 = sio.loadmat(str(zdata2_path), squeeze_me=True)
    meta = sio.loadmat(str(zmetadata_path), squeeze_me=True)

    spkF_mat = mat2["spkF"]
    DX = float(meta["DX"])
    DT = float(meta["DT"])
    roiR = float(meta["roiR"])
    mask = meta["mask"]

    sparks = []
    for i, s in enumerate(spkF_mat):
        good = bool(int(s["good"]))
        fail = str(s["fail"])

        fwhm_val = float(s["FWHM"])
        ts = np.asarray(s["timesignal"]).ravel()
        ti = np.asarray(s["timeinterval"]).ravel()

        feat = SparkFeature(
            px=int(s["px"]),
            py=int(s["py"]),
            pt=int(s["pt"]),
            x_0b=int(s["px"]) - 1,
            y_0b=int(s["py"]) - 1,
            t_0b=int(s["pt"]) - 1,
            dist2memb=float(s["dist2memb"]),
            fwhm=fwhm_val,
            timesignal=ts,
            timeinterval=ti,
            r2=float(s["r2"]),
            amp=float(s["amp"]),
            tau=float(s["tau"]),
            ror=float(s["ror"]),
            t2p=float(s["t2p"]),
            fdhm=float(s["FDHM"]),
            AMP=float(s["AMP"]),
            BL=float(s["BL"]),
            mass=float(s["mass"]),
        )
        sparks.append(FilteredSpark(id=i + 1, good=good, fail=fail, feature=feat))

    return sparks, DX, DT, roiR, mask


@pytest.mark.skipif(
    not (REAL_DATA_DIR / "zData2.mat").exists(),
    reason="Real experiment zData2.mat not found",
)
def test_regression_roi_clustering_real_experiment():
    """Verify ROI count, ROI spark counts, and spark→ROI assignment for real experiment."""
    zdata2 = REAL_DATA_DIR / "zData2.mat"
    zmeta = REAL_DATA_DIR / "zMetaData.mat"

    sparks, DX, DT, roiR, mask = _load_sparks_from_mat(zdata2, zmeta)
    mat2 = sio.loadmat(str(zdata2), squeeze_me=True)
    mat_rois = mat2["ROIs"]

    # Build a dummy volume (cluster_rois needs shape, actual values are for signal extraction)
    fov_y, fov_x = mask.shape
    temps = int(sio.loadmat(str(REAL_DATA_DIR / "zMetaData.mat"), squeeze_me=True)["temps"])
    volume = np.zeros((fov_y, fov_x, temps), dtype=np.float64)

    result = cluster_rois(sparks, volume, mask, dx_um=DX, dt_ms=DT, roi_r_um=roiR)
    py_rois = result.rois

    # ROI COUNT must match
    mat_roi_count = len(mat_rois) if hasattr(mat_rois, "__len__") else 0
    assert len(py_rois) == mat_roi_count, (
        f"ROI count mismatch: Python {len(py_rois)} vs MATLAB {mat_roi_count}"
    )

    # For each ROI, verify Nspk matches (ROIs may be in different order)
    mat_nspk_sorted = sorted([int(r["Nspk"]) for r in mat_rois], reverse=True)
    py_nspk_sorted = sorted([r.Nspk for r in py_rois], reverse=True)
    assert py_nspk_sorted == mat_nspk_sorted, (
        f"ROI spark counts mismatch: Python {py_nspk_sorted} vs MATLAB {mat_nspk_sorted}"
    )

    # Accepted sparks must all have non-zero ROI IDs
    for spk in sparks:
        if spk.good:
            roi_id = getattr(spk.feature, "roi", None)
            assert roi_id is not None and roi_id > 0, (
                f"Accepted spark id={spk.id} has invalid roi={roi_id}"
            )


@pytest.mark.skipif(
    not (SIM_DATA_DIR / "zData2.mat").exists(),
    reason="Simulated experiment zData2.mat not found",
)
def test_regression_roi_clustering_simulated_experiment():
    """Verify ROI count, multi-spark grouping, and dist2spk for simulated experiment."""
    zdata2 = SIM_DATA_DIR / "zData2.mat"
    zmeta = SIM_DATA_DIR / "zMetaData.mat"

    sparks, DX, DT, roiR, mask = _load_sparks_from_mat(zdata2, zmeta)
    mat2 = sio.loadmat(str(zdata2), squeeze_me=True)
    mat_rois = mat2["ROIs"]

    meta_sq = sio.loadmat(str(zmeta), squeeze_me=True)
    temps = int(meta_sq["temps"])
    fov_y, fov_x = mask.shape
    volume = np.zeros((fov_y, fov_x, temps), dtype=np.float64)

    result = cluster_rois(sparks, volume, mask, dx_um=DX, dt_ms=DT, roi_r_um=roiR)
    py_rois = result.rois

    # ROI count
    mat_roi_count = len(mat_rois) if hasattr(mat_rois, "__len__") else 0
    assert len(py_rois) == mat_roi_count, (
        f"ROI count mismatch: Python {len(py_rois)} vs MATLAB {mat_roi_count}"
    )

    # The largest ROI must have Nspk=2 (sparks 3 and 11 share a release site at ~0.3um)
    max_nspk = max(r.Nspk for r in py_rois)
    assert max_nspk == 2, f"Expected largest ROI to have 2 sparks, got {max_nspk}"

    # Verify that the 2-spark ROI groups the correct pair
    roi_2spk = next(r for r in py_rois if r.Nspk == 2)
    # Centroid of sparks 3 (py=143,px=802) and 11 (py=140,px=802) → rounded mean: y=142, x=802
    assert roi_2spk.px == 802, f"2-spark ROI X centroid wrong: {roi_2spk.px}"
    assert roi_2spk.py == 142, f"2-spark ROI Y centroid wrong: {roi_2spk.py}"

    # Sparks 3 and 11 (1-based IDs 3 and 11) should be in the same ROI
    spark3_roi = getattr(sparks[2].feature, "roi", None)  # 0-indexed: spark id=3 → index 2
    spark11_roi = getattr(sparks[10].feature, "roi", None)  # spark id=11 → index 10
    assert spark3_roi is not None and spark3_roi > 0
    assert spark11_roi is not None and spark11_roi > 0
    assert spark3_roi == spark11_roi, (
        f"Sparks 3 and 11 should be in same ROI but got {spark3_roi} and {spark11_roi}"
    )

    # The largest ROI gets ROI id=1
    assert roi_2spk.id == 1, f"Largest ROI should have id=1, got {roi_2spk.id}"

    # Verify dist2spk for the 2-spark pair
    # Distance = 3 pixels * DX = 3 * 0.1 = 0.3 um
    spk3_d2s = getattr(sparks[2].feature, "dist2spk", None)
    spk11_d2s = getattr(sparks[10].feature, "dist2spk", None)
    assert spk3_d2s is not None
    assert spk11_d2s is not None
    assert abs(spk3_d2s - 0.3) < 1e-6, f"Spark 3 dist2spk={spk3_d2s:.6f}, expected 0.3"
    assert abs(spk11_d2s - 0.3) < 1e-6, f"Spark 11 dist2spk={spk11_d2s:.6f}, expected 0.3"


@pytest.mark.skipif(
    not (SIM_DATA_DIR / "zData2.mat").exists(),
    reason="Simulated experiment zData2.mat not found",
)
def test_regression_roi_dist2closest_simulated():
    """Verify dist2closest for the multi-spark ROI against MATLAB reference."""
    zdata2 = SIM_DATA_DIR / "zData2.mat"
    zmeta = SIM_DATA_DIR / "zMetaData.mat"

    sparks, DX, DT, roiR, mask = _load_sparks_from_mat(zdata2, zmeta)
    mat2 = sio.loadmat(str(zdata2), squeeze_me=True)
    mat_rois = mat2["ROIs"]

    meta_sq = sio.loadmat(str(zmeta), squeeze_me=True)
    temps = int(meta_sq["temps"])
    fov_y, fov_x = mask.shape
    volume = np.zeros((fov_y, fov_x, temps), dtype=np.float64)

    result = cluster_rois(sparks, volume, mask, dx_um=DX, dt_ms=DT, roi_r_um=roiR)
    py_rois = result.rois

    # Build MATLAB ROI → dist2closest lookup (keyed by Nspk to avoid ordering issues)
    mat_d2c_by_nspk: dict[int, list[float]] = {}
    for r in mat_rois:
        nspk = int(r["Nspk"])
        d2c_val = r["dist2closest"]
        d2c = float(d2c_val) if np.ndim(d2c_val) == 0 else float(d2c_val[0])
        mat_d2c_by_nspk.setdefault(nspk, []).append(d2c)

    # Verify Python's 2-spark ROI dist2closest matches MATLAB reference
    if 2 in mat_d2c_by_nspk:
        mat_d2c_2spk = mat_d2c_by_nspk[2][0]
        py_roi_2spk = next(r for r in py_rois if r.Nspk == 2)
        assert abs(py_roi_2spk.dist2closest - mat_d2c_2spk) < 0.01, (
            f"dist2closest for 2-spark ROI: Python={py_roi_2spk.dist2closest:.4f} "
            f"vs MATLAB={mat_d2c_2spk:.4f}"
        )
