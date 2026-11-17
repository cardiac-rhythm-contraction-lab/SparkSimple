"""tests/integration/test_end_to_end.py — End-to-end integration test for SparkSimple2.

Phase-by-phase approach: loads MATLAB intermediate data (zData2.mat + zMetaData.mat + zData0.mat),
runs Python filtering and ROI clustering on MATLAB's own candidates, then verifies counts and AMP.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
import scipy.io

from sparksimple.config import FilteringConfig
from sparksimple.features.models import SparkFeature
from sparksimple.filtering.apply import filter_sparks
from sparksimple.export.pipeline import run_phase10


def _load_spkF_from_mat(zdata2_path: Path) -> list[SparkFeature]:
    """Load MATLAB spkF struct array from zData2.mat and convert to SparkFeature list."""
    mat = scipy.io.loadmat(str(zdata2_path), squeeze_me=False)
    spkF = mat["spkF"]  # shape (1, N) structured array

    def _scalar(val) -> float:
        arr = np.asarray(val)
        while arr.ndim > 0:
            arr = arr.flat[0]
        try:
            return float(arr)
        except (TypeError, ValueError):
            return float("nan")

    def _int_scalar(val) -> int:
        v = _scalar(val)
        return int(v) if not math.isnan(v) else 0

    candidates: list[SparkFeature] = []
    n = spkF.shape[1]
    for i in range(n):
        s = spkF[0, i]

        px = _int_scalar(s["px"])
        py = _int_scalar(s["py"])
        pt = _int_scalar(s["pt"])

        fwhm_raw = s["FWHM"]
        fwhm = _scalar(fwhm_raw)

        ts_raw = np.asarray(s["timesignal"]).ravel()
        ti_raw = np.asarray(s["timeinterval"]).ravel().astype(int)

        r2 = _scalar(s["r2"])
        amp = _scalar(s["amp"])
        tau = _scalar(s["tau"])
        ror = _scalar(s["ror"])
        t2p = _scalar(s["t2p"])

        fdhm_raw = s["FDHM"]
        fdhm = _scalar(fdhm_raw)

        AMP = _scalar(s["AMP"])
        BL = _scalar(s["BL"])
        mass = _scalar(s["mass"])
        dist2memb = _scalar(s["dist2memb"])

        candidates.append(SparkFeature(
            px=px,
            py=py,
            pt=pt,
            x_0b=px - 1,
            y_0b=py - 1,
            t_0b=pt - 1,
            dist2memb=dist2memb,
            fwhm=fwhm,
            timesignal=ts_raw,
            timeinterval=ti_raw,
            r2=r2,
            amp=amp,
            tau=tau,
            ror=ror,
            t2p=t2p,
            fdhm=fdhm,
            AMP=AMP,
            BL=BL,
            mass=mass,
        ))
    return candidates


def test_full_pipeline_end_to_end_simulated(
    sim_exp_matlab_result: Path,
):
    """Filter MATLAB's 30 candidates and cluster ROIs; compare AMP against MATLAB reference CSV."""
    zdata2_path = sim_exp_matlab_result / "zData2.mat"
    zdata0_path = sim_exp_matlab_result / "zData0.mat"
    zmeta_path = sim_exp_matlab_result / "zMetaData.mat"
    ref_csv_path = sim_exp_matlab_result / "yData_NOISE0.csv"

    for p in (zdata2_path, zdata0_path, zmeta_path, ref_csv_path):
        if not p.exists():
            pytest.skip(f"MATLAB reference file not found: {p}")

    # Load volume (Y, X, T) from MATLAB zData0.mat
    mat0 = scipy.io.loadmat(str(zdata0_path), squeeze_me=False)
    volume = np.asarray(mat0["volum"], dtype=np.float64)  # (Y, X, T)

    # Load metadata from zMetaData.mat
    meta = scipy.io.loadmat(str(zmeta_path), squeeze_me=False)

    def _scalar(val) -> float:
        arr = np.asarray(val)
        while arr.ndim > 0:
            arr = arr.flat[0]
        return float(arr)

    dx_um = _scalar(meta["DX"])
    dt_ms = _scalar(meta["DT"])
    roi_r_um = _scalar(meta["roiR"])
    fo_raw = np.asarray(meta["fo"], dtype=np.float64)
    mask = np.asarray(meta["mask"], dtype=np.uint8)

    # fo may be stored as (1,1) cell → squeeze to 2D
    if fo_raw.ndim == 2 and fo_raw.shape[0] == 1 and fo_raw.dtype == object:
        fo_map = np.asarray(fo_raw[0, 0], dtype=np.float64)
    else:
        fo_map = fo_raw

    # Load MATLAB's 30 spkF candidates
    candidates = _load_spkF_from_mat(zdata2_path)
    assert len(candidates) == 30, f"Expected 30 MATLAB candidates, got {len(candidates)}"

    # Run Python filtering using MATLAB defaults
    cfg = FilteringConfig.from_matlab_defaults()
    filt_res = filter_sparks(
        candidates=candidates,
        config=cfg,
        shape=mask.shape,
    )

    accepted_sparks = filt_res.accepted
    assert len(accepted_sparks) == 20, (
        f"Expected 20 accepted sparks, got {len(accepted_sparks)}. "
        f"Rejected: {len(filt_res.rejected)}"
    )
    assert len(filt_res.records) == 30

    # Run Phase 10 (ROI clustering + CSV export)
    with tempfile.TemporaryDirectory() as tmp_dir:
        p10 = run_phase10(
            sparks=filt_res.records,
            volume=volume,
            mask=mask,
            fo_map=fo_map,
            dx_um=dx_um,
            dt_ms=dt_ms,
            roi_r_um=roi_r_um,
            output_dir=tmp_dir,
            title="E2E_NOISE0",
        )

        assert len(p10.clustering.rois) == 19, (
            f"Expected 19 ROIs, got {len(p10.clustering.rois)}"
        )

        paths = p10.paths
        assert Path(paths.accepted_csv).exists()

        # Parse exported CSV and reference CSV
        def extract_spark_data_rows(lines: list[str]) -> list[list[str]]:
            rows = []
            in_data = False
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("ID,Xpix,Ypix"):
                    in_data = True
                    continue
                if in_data:
                    if stripped.startswith("Roi Data"):
                        break
                    parts = [p.strip() for p in stripped.split(",")]
                    if len(parts) >= 8 and parts[0].isdigit():
                        rows.append(parts)
            return rows

        with open(paths.accepted_csv, encoding="utf-8") as f:
            py_rows = extract_spark_data_rows(f.readlines())

        with open(ref_csv_path, encoding="latin-1", errors="replace") as f:
            ref_rows = extract_spark_data_rows(f.readlines())

        assert len(py_rows) == 20, f"Expected 20 rows in exported CSV, got {len(py_rows)}"
        assert len(ref_rows) == 20, f"Expected 20 rows in MATLAB reference CSV, got {len(ref_rows)}"

        # Sort by Tframe (col 3) then ID (col 0) for stable comparison
        py_sorted = sorted(py_rows, key=lambda r: (int(r[3]), int(r[0])))
        ref_sorted = sorted(ref_rows, key=lambda r: (int(r[3]), int(r[0])))

        # CSV columns: ID(0) Xpix(1) Ypix(2) Tframe(3) ... AMP(7)
        for py_r, ref_r in zip(py_sorted, ref_sorted):
            py_amp = float(py_r[7])
            ref_amp = float(ref_r[7])
            assert py_amp == pytest.approx(ref_amp, rel=0.05, abs=0.02), (
                f"AMP mismatch for spark ID={ref_r[0]}: py={py_amp:.4f} vs ref={ref_amp:.4f}"
            )
