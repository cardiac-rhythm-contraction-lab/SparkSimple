"""Unit tests for the browser validation processing backend."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.io

from sparksimple.export.matlab_compat import save_zdata2_mat
from sparksimple.features.models import SparkFeature
from sparksimple.filtering.models import FilteredSpark
from sparksimple.tools.validate_not_merge import (
    _required_positive_metadata,
    apply_manual_validation,
)


def _spark(spark_id: int, good: bool = True) -> FilteredSpark:
    feat = SparkFeature(
        px=5 + spark_id,
        py=5,
        pt=spark_id,
        x_0b=4 + spark_id,
        y_0b=4,
        t_0b=spark_id - 1,
        dist2memb=1.0,
        fwhm=1.5,
        timesignal=np.ones(5),
        timeinterval=np.arange(1, 6),
        r2=0.95,
        amp=1.2,
        tau=20.0,
        ror=0.1,
        t2p=3.0,
        fdhm=4.0,
        AMP=2.2,
        BL=1.0,
        mass=10.0,
    )
    return FilteredSpark(
        id=spark_id,
        good=good,
        fail="no" if good else "manual rejection",
        feature=feat,
    )


def test_apply_manual_validation_overwrites_zdata2_good_values(tmp_path):
    result_dir = tmp_path / "RF"
    result_dir.mkdir()
    scipy.io.savemat(
        result_dir / "zMetaData.mat",
        {
            "mask": np.ones((12, 12)),
            "fo": np.ones((12, 12)),
            "temps": 5,
            "DX": 0.2,
            "DT": 10.0,
            "roiR": 2.0,
        },
    )
    save_zdata2_mat(str(result_dir / "zData2.mat"), [_spark(1), _spark(2), _spark(3)], [])

    apply_manual_validation(
        result_dir,
        {1: True, 2: False, 3: True},
        title="UnitExp",
        report=False,
    )

    loaded = scipy.io.loadmat(str(result_dir / "zData2.mat"), squeeze_me=True)
    spkf = np.asarray(loaded["spkF"]).reshape(-1)
    good_values = [int(np.asarray(rec["good"]).reshape(-1)[0]) for rec in spkf]
    assert good_values == [1, 0, 1]
    assert (result_dir / "yData_UnitExp.csv").exists()
    assert (result_dir / "zValidation.mat").exists()


# ── _required_positive_metadata ─────────────────────────────────────────────

def test_required_positive_metadata_returns_value():
    assert _required_positive_metadata({"DX": 0.2}, "DX", "dx_um") == 0.2


def test_required_positive_metadata_missing_raises_key_error():
    with pytest.raises(KeyError, match="Missing mandatory acquisition metadata"):
        _required_positive_metadata({}, "DX", "dx_um")


@pytest.mark.parametrize("bad_value", [0.0, -1.0, float("nan"), float("inf")])
def test_required_positive_metadata_rejects_non_positive(bad_value):
    with pytest.raises(ValueError, match="finite and positive"):
        _required_positive_metadata({"DX": bad_value}, "DX")


def test_required_positive_metadata_rejects_non_numeric():
    with pytest.raises(ValueError, match="not numeric"):
        _required_positive_metadata({"DX": "not-a-number"}, "DX")


def test_required_positive_metadata_rejects_empty_string():
    with pytest.raises(ValueError, match="not numeric"):
        _required_positive_metadata({"DX": ""}, "DX")


# ── apply_manual_validation must never invent DX/DT ─────────────────────────

def _write_zdata2(result_dir):
    save_zdata2_mat(str(result_dir / "zData2.mat"), [_spark(1), _spark(2)], [])


def test_apply_manual_validation_fails_without_dx(tmp_path):
    result_dir = tmp_path / "RF"
    result_dir.mkdir()
    scipy.io.savemat(
        result_dir / "zMetaData.mat",
        {"mask": np.ones((12, 12)), "fo": np.ones((12, 12)), "temps": 5, "DT": 10.0},
    )
    _write_zdata2(result_dir)

    with pytest.raises(KeyError, match="DX"):
        apply_manual_validation(result_dir, {1: True, 2: False}, title="NoDx", report=False)

    assert not (result_dir / "yData_NoDx.csv").exists()
    assert not (result_dir / "zValidation.mat").exists()


def test_apply_manual_validation_fails_without_dt(tmp_path):
    result_dir = tmp_path / "RF"
    result_dir.mkdir()
    scipy.io.savemat(
        result_dir / "zMetaData.mat",
        {"mask": np.ones((12, 12)), "fo": np.ones((12, 12)), "temps": 5, "DX": 0.2},
    )
    _write_zdata2(result_dir)

    with pytest.raises(KeyError, match="DT"):
        apply_manual_validation(result_dir, {1: True, 2: False}, title="NoDt", report=False)

    assert not (result_dir / "yData_NoDt.csv").exists()
    assert not (result_dir / "zValidation.mat").exists()


@pytest.mark.parametrize("bad_dx", [0.0, -0.2, float("nan"), float("inf")])
def test_apply_manual_validation_fails_on_invalid_dx(tmp_path, bad_dx):
    result_dir = tmp_path / "RF"
    result_dir.mkdir()
    scipy.io.savemat(
        result_dir / "zMetaData.mat",
        {
            "mask": np.ones((12, 12)), "fo": np.ones((12, 12)), "temps": 5,
            "DX": bad_dx, "DT": 10.0,
        },
    )
    _write_zdata2(result_dir)

    with pytest.raises(ValueError, match="finite and positive"):
        apply_manual_validation(result_dir, {1: True, 2: False}, title="BadDx", report=False)

    assert not (result_dir / "yData_BadDx.csv").exists()
    assert not (result_dir / "zValidation.mat").exists()
