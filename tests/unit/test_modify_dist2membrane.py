"""Tests for the manual dist2memb correction tool."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import scipy.io

from sparksimple.export.matlab_compat import save_zdata2_mat
from sparksimple.features.models import SparkFeature
from sparksimple.filtering.models import FilteredSpark
from sparksimple.tools.modify_dist2membrane import (
    ModifyDist2MembraneState,
    _required_positive_dx,
    compute_dist2memb,
)


def test_compute_dist2memb_from_click_coordinates() -> None:
    dist = compute_dist2memb(xs=13.0, ys=14.0, px=10.0, py=10.0, dx_um=0.25)
    assert dist == math.hypot(13.0 - 10.0, 14.0 - 10.0) * 0.25


# ── _required_positive_dx ────────────────────────────────────────────────────

def test_required_positive_dx_returns_value():
    assert _required_positive_dx(np.array([[0.25]])) == pytest.approx(0.25)


def test_required_positive_dx_rejects_empty_array():
    with pytest.raises(ValueError, match="empty"):
        _required_positive_dx(np.array([]))


@pytest.mark.parametrize("bad_value", [0.0, -0.25, float("nan"), float("inf")])
def test_required_positive_dx_rejects_non_positive(bad_value):
    with pytest.raises(ValueError, match="finite and positive"):
        _required_positive_dx(np.array([[bad_value]]))


def test_required_positive_dx_rejects_non_numeric():
    with pytest.raises(ValueError, match="not numeric"):
        _required_positive_dx(np.array(["oops"], dtype=object))


# ── ModifyDist2MembraneState integration ─────────────────────────────────────

def _spark(spark_id: int) -> FilteredSpark:
    feat = SparkFeature(
        px=5, py=5, pt=1, x_0b=4, y_0b=4, t_0b=0,
        dist2memb=1.0, fwhm=1.5,
        timesignal=np.ones(3), timeinterval=np.arange(1, 4),
        r2=0.95, amp=1.2, tau=20.0, ror=0.1, t2p=3.0, fdhm=4.0,
        AMP=2.2, BL=1.0, mass=10.0,
    )
    return FilteredSpark(id=spark_id, good=True, fail="no", feature=feat)


def _write_result_dir(tmp_path, dx_value) -> Path:
    result_dir = tmp_path / "RF"
    result_dir.mkdir()
    save_zdata2_mat(str(result_dir / "zData2.mat"), [_spark(1)], [])
    meta = {"mask": np.ones((10, 10)), "DX": dx_value}
    scipy.io.savemat(str(result_dir / "zMetaData.mat"), meta)
    scipy.io.savemat(str(result_dir / "zData0.mat"), {"volum": np.ones((10, 10, 3))})
    return result_dir


def test_state_rejects_empty_dx(tmp_path):
    result_dir = _write_result_dir(tmp_path, np.array([]))
    with pytest.raises(ValueError, match="empty"):
        ModifyDist2MembraneState(result_dir)


@pytest.mark.parametrize("bad_dx", [0.0, -0.2, float("nan"), float("inf")])
def test_state_rejects_invalid_dx(tmp_path, bad_dx):
    result_dir = _write_result_dir(tmp_path, bad_dx)
    with pytest.raises(ValueError, match="finite and positive"):
        ModifyDist2MembraneState(result_dir)


def test_state_accepts_valid_dx(tmp_path):
    result_dir = _write_result_dir(tmp_path, 0.2)
    state = ModifyDist2MembraneState(result_dir)
    assert state.dx_um == pytest.approx(0.2)
