"""tests/regression/test_regression_export.py — Export parity tests for Phase 10.

Compares Python export outputs against MATLAB reference CSV and MAT files from
tests/data/*/Matlab_Result/.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import scipy.io as sio
import pytest

from sparksimple.features.models import SparkFeature
from sparksimple.filtering.models import FilteredSpark
from sparksimple.roi.cluster import cluster_rois
from sparksimple.export.tables import generate_accepted_csv, generate_rejected_csv
from sparksimple.export.matlab_compat import save_zdata2_mat


REAL_DATA_DIR = Path(__file__).parents[1] / "data" / "Real_Experiment" / "Matlab_Result"
SIM_DATA_DIR = Path(__file__).parents[1] / "data" / "Simulated_Experiment" / "Matlab_Result"


def _load_sparks(zdata2_path: Path) -> list[FilteredSpark]:
    """Load FilteredSpark list from MATLAB zData2.mat."""
    mat2 = sio.loadmat(str(zdata2_path), squeeze_me=True)
    spkF_mat = mat2["spkF"]
    sparks = []
    for i, s in enumerate(spkF_mat):
        good = bool(int(s["good"]))
        fail = str(s["fail"])
        feat = SparkFeature(
            px=int(s["px"]),
            py=int(s["py"]),
            pt=int(s["pt"]),
            x_0b=int(s["px"]) - 1,
            y_0b=int(s["py"]) - 1,
            t_0b=int(s["pt"]) - 1,
            dist2memb=float(s["dist2memb"]),
            fwhm=float(s["FWHM"]),
            timesignal=np.asarray(s["timesignal"]).ravel(),
            timeinterval=np.asarray(s["timeinterval"]).ravel(),
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
    return sparks


@pytest.mark.skipif(
    not (SIM_DATA_DIR / "zData2.mat").exists(),
    reason="Simulated experiment reference not found",
)
def test_regression_accepted_csv_row_count_simulated():
    """Accepted CSV must contain the same number of data rows as MATLAB reference."""
    zdata2 = SIM_DATA_DIR / "zData2.mat"
    zmeta = SIM_DATA_DIR / "zMetaData.mat"
    ref_csv = SIM_DATA_DIR / "yData_NOISE0.csv"

    mat_meta = sio.loadmat(str(zmeta), squeeze_me=True)
    DX = float(mat_meta["DX"])
    DT = float(mat_meta["DT"])
    roiR = float(mat_meta["roiR"])
    mask = mat_meta["mask"]
    temps = int(mat_meta["temps"])
    fo = mat_meta["fo"]

    sparks = _load_sparks(zdata2)
    fov_y, fov_x = mask.shape
    volume = np.zeros((fov_y, fov_x, temps), dtype=np.float64)

    clust = cluster_rois(sparks, volume, mask, dx_um=DX, dt_ms=DT, roi_r_um=roiR)

    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        out_path = f.name

    generate_accepted_csv(sparks, clust.rois, fo, mask, DX, DT, temps, out_path)

    # Count accepted data rows in Python output (section-aware: between spark header and separator)
    with open(out_path, encoding="utf-8") as f:
        py_lines = f.readlines()

    # Count accepted data rows in reference CSV
    with open(str(ref_csv), encoding="latin-1", errors="replace") as f:
        ref_lines = f.readlines()

    def count_accepted_data_rows(lines):
        """Count spark data rows in the accepted spark section (ID,...,ROI header to Roi Data label)."""
        in_section = False
        count = 0
        for ln in lines:
            stripped = ln.strip()
            # Enter section when we find the accepted spark header (ends with ROI, not #sparks)
            if stripped.startswith("ID,") and stripped.rstrip(",").endswith("ROI"):
                in_section = True
                continue
            if in_section:
                # Exit when reaching next named section
                if stripped in ("Roi Data", "ROI time signal", "mean values & stds"):
                    break
                # Count digit-starting rows (skip blank/separator lines)
                if stripped and (stripped[0].isdigit() or stripped[0] == "-"):
                    count += 1
        return count

    n_accepted = sum(1 for s in sparks if s.good)
    py_data_rows = count_accepted_data_rows(py_lines)
    ref_data_rows = count_accepted_data_rows(ref_lines)

    assert py_data_rows == n_accepted, (
        f"Python accepted CSV has {py_data_rows} data rows, expected {n_accepted}"
    )
    assert py_data_rows == ref_data_rows, (
        f"Python accepted CSV rows ({py_data_rows}) != MATLAB reference ({ref_data_rows})"
    )


@pytest.mark.skipif(
    not (SIM_DATA_DIR / "zData2.mat").exists(),
    reason="Simulated experiment reference not found",
)
def test_regression_rejected_csv_row_count_simulated():
    """Rejected CSV must contain the same number of data rows as MATLAB reference."""
    zdata2 = SIM_DATA_DIR / "zData2.mat"
    zmeta = SIM_DATA_DIR / "zMetaData.mat"
    ref_csv = SIM_DATA_DIR / "zDataRejected_NOISE0.csv"

    mat_meta = sio.loadmat(str(zmeta), squeeze_me=True)
    DX = float(mat_meta["DX"])
    DT = float(mat_meta["DT"])
    mask = mat_meta["mask"]
    temps = int(mat_meta["temps"])
    fo = mat_meta["fo"]

    sparks = _load_sparks(zdata2)

    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        out_path = f.name

    generate_rejected_csv(sparks, fo, mask, DX, DT, temps, out_path)

    with open(out_path, encoding="utf-8") as f:
        py_lines = f.readlines()

    with open(str(ref_csv), encoding="latin-1", errors="replace") as f:
        ref_lines = f.readlines()

    def count_rejected_data_rows(lines):
        """Count rejected spark rows after the ID,...,tau header (to end of file)."""
        in_section = False
        count = 0
        for ln in lines:
            stripped = ln.strip()
            if stripped.startswith("ID,") and stripped.rstrip(",").endswith("tau"):
                in_section = True
                continue
            if in_section and stripped and (stripped[0].isdigit() or stripped[0] == "-"):
                count += 1
        return count

    n_rejected = sum(1 for s in sparks if not s.good)
    py_data_rows = count_rejected_data_rows(py_lines)
    ref_data_rows = count_rejected_data_rows(ref_lines)

    assert py_data_rows == n_rejected, (
        f"Python rejected CSV has {py_data_rows} data rows, expected {n_rejected}"
    )
    assert py_data_rows == ref_data_rows, (
        f"Python rejected CSV rows ({py_data_rows}) != MATLAB reference ({ref_data_rows})"
    )


@pytest.mark.skipif(
    not (SIM_DATA_DIR / "zData2.mat").exists(),
    reason="Simulated experiment reference not found",
)
def test_regression_zdata2_mat_field_count():
    """Python-written zData2.mat must have same number of spkF records as MATLAB reference."""
    zdata2 = SIM_DATA_DIR / "zData2.mat"
    zmeta = SIM_DATA_DIR / "zMetaData.mat"

    mat_meta = sio.loadmat(str(zmeta), squeeze_me=True)
    DX = float(mat_meta["DX"])
    DT = float(mat_meta["DT"])
    roiR = float(mat_meta["roiR"])
    mask = mat_meta["mask"]
    temps = int(mat_meta["temps"])

    sparks = _load_sparks(zdata2)
    fov_y, fov_x = mask.shape
    volume = np.zeros((fov_y, fov_x, temps), dtype=np.float64)

    clust = cluster_rois(sparks, volume, mask, dx_um=DX, dt_ms=DT, roi_r_um=roiR)

    with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as f:
        out_path = f.name

    save_zdata2_mat(out_path, sparks, clust.rois)

    # Reload and verify
    loaded = sio.loadmat(out_path, squeeze_me=True)
    assert "spkF" in loaded
    assert "ROIs" in loaded

    n_sparks_py = len(loaded["spkF"])
    n_sparks_matlab = len(sio.loadmat(str(zdata2), squeeze_me=True)["spkF"])
    assert n_sparks_py == n_sparks_matlab, (
        f"spkF count: Python {n_sparks_py} vs MATLAB {n_sparks_matlab}"
    )

    mat_roi_count = len(sio.loadmat(str(zdata2), squeeze_me=True)["ROIs"])
    py_roi_count = len(clust.rois)
    n_roi_mat_file = len(loaded["ROIs"]) if loaded["ROIs"].size > 0 else 0
    assert n_roi_mat_file == py_roi_count, (
        f"ROIs written: Python {n_roi_mat_file} vs expected {py_roi_count}"
    )
    assert py_roi_count == mat_roi_count, (
        f"ROI count: Python {py_roi_count} vs MATLAB {mat_roi_count}"
    )


@pytest.mark.skipif(
    not (SIM_DATA_DIR / "zData2.mat").exists(),
    reason="Simulated experiment reference not found",
)
def test_regression_accepted_csv_column_order_simulated():
    """Accepted CSV data rows must follow MATLAB column order: ID,X,Y,T,T,d2m,d2s,AMP,amp,..."""
    zdata2 = SIM_DATA_DIR / "zData2.mat"
    zmeta = SIM_DATA_DIR / "zMetaData.mat"

    mat_meta = sio.loadmat(str(zmeta), squeeze_me=True)
    DX = float(mat_meta["DX"])
    DT = float(mat_meta["DT"])
    roiR = float(mat_meta["roiR"])
    mask = mat_meta["mask"]
    temps = int(mat_meta["temps"])
    fo = mat_meta["fo"]

    sparks = _load_sparks(zdata2)
    fov_y, fov_x = mask.shape
    volume = np.zeros((fov_y, fov_x, temps), dtype=np.float64)

    clust = cluster_rois(sparks, volume, mask, dx_um=DX, dt_ms=DT, roi_r_um=roiR)

    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        out_path = f.name

    generate_accepted_csv(sparks, clust.rois, fo, mask, DX, DT, temps, out_path)

    with open(out_path, encoding="utf-8") as f:
        lines = f.readlines()

    # Find the header line
    header_line = None
    for ln in lines:
        if ln.startswith("ID,"):
            header_line = ln.strip().rstrip(",")
            break

    assert header_line is not None, "Header line starting with 'ID,' not found"
    cols = header_line.split(",")

    # Expected MATLAB column order for accepted sparks
    expected_cols = [
        "ID", "Xpix", "Ypix", "Tframe", "Tframe",
        "dist2memb", "dist2closest", "AMP", "amp", "BL",
        "RoR", "mass", "t2p", "FDHM", "FWHM", "tau",
        "fldr", "subfldr", "ROI",
    ]
    assert cols == expected_cols, f"Column order mismatch:\n  Got:      {cols}\n  Expected: {expected_cols}"


@pytest.mark.skipif(
    not (SIM_DATA_DIR / "zData2.mat").exists(),
    reason="Simulated experiment reference not found",
)
def test_regression_accepted_csv_spark_ids_match_reference():
    """Accepted spark IDs in Python CSV must match those in MATLAB CSV (same set)."""
    zdata2 = SIM_DATA_DIR / "zData2.mat"
    zmeta = SIM_DATA_DIR / "zMetaData.mat"
    ref_csv = SIM_DATA_DIR / "yData_NOISE0.csv"

    mat_meta = sio.loadmat(str(zmeta), squeeze_me=True)
    DX = float(mat_meta["DX"])
    DT = float(mat_meta["DT"])
    roiR = float(mat_meta["roiR"])
    mask = mat_meta["mask"]
    temps = int(mat_meta["temps"])
    fo = mat_meta["fo"]

    sparks = _load_sparks(zdata2)
    fov_y, fov_x = mask.shape
    volume = np.zeros((fov_y, fov_x, temps), dtype=np.float64)

    clust = cluster_rois(sparks, volume, mask, dx_um=DX, dt_ms=DT, roi_r_um=roiR)

    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        out_path = f.name

    generate_accepted_csv(sparks, clust.rois, fo, mask, DX, DT, temps, out_path)

    def extract_spark_ids(csv_path, encoding="utf-8"):
        ids = []
        with open(csv_path, encoding=encoding, errors="replace") as f:
            in_data_section = False
            for ln in f:
                stripped = ln.strip()
                if stripped.startswith("ID,"):
                    in_data_section = True
                    continue
                if not in_data_section:
                    continue
                if not stripped or stripped == ",":
                    break
                parts = stripped.split(",")
                if parts[0].isdigit():
                    ids.append(int(parts[0]))
        return sorted(ids)

    py_ids = extract_spark_ids(out_path)
    ref_ids = extract_spark_ids(str(ref_csv), encoding="latin-1")

    assert py_ids == ref_ids, (
        f"Accepted spark ID sets differ:\n  Python: {py_ids}\n  MATLAB: {ref_ids}"
    )
