"""Unit tests for sparksimple.tools.merge_only."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock
import numpy as np
import pytest

import scipy.io

from sparksimple.export.matlab_compat import save_zdata2_mat
from sparksimple.features.models import SparkFeature
from sparksimple.filtering.models import FilteredSpark
from sparksimple.tools.merge_only import (
    MergeCalibrationError,
    _check_calibration_compatible,
    _find_result_dirs,
    _load_sparks_from_mat,
    _load_metadata_from_mat,
    merge_only,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_spark_dict(
    spark_id: int = 1,
    px: int = 10,
    py: int = 10,
    good: bool = True,
) -> dict:
    return {
        "id": spark_id,
        "px": px,
        "py": py,
        "pt": 5,
        "FWHM": 1.2,
        "fwhm": 1.2,
        "timesignal": np.ones(20),
        "timeinterval": np.arange(1, 21),
        "r2": 0.9,
        "amp": 1.5,
        "tau": 30.0,
        "ror": 0.05,
        "t2p": 15.0,
        "FDHM": 50.0,
        "fdhm": 50.0,
        "AMP": 2.0,
        "BL": 0.5,
        "bl": 0.5,
        "mass": 8.0,
        "dist2memb": 2.0,
        "good": good,
        "fail": "no" if good else "out of field of view (X)",
        "dist2spk": 0.0,
        "roi": 0,
    }


def _make_meta() -> dict:
    return {
        "DX": 0.2,
        "DT": 10.0,
        "temps": 50,
        "mask": np.ones((20, 20), dtype=np.float64),
        "fo": 100.0,
    }


# ── _find_result_dirs ────────────────────────────────────────────────────────

def test_find_result_dirs_finds_matching_subfolders(tmp_path: Path) -> None:
    rf = "iT_03_r2_03"

    result_a = tmp_path / "exp_A" / f"{rf}_1"
    result_b = tmp_path / "exp_B" / f"{rf}_2"
    other = tmp_path / "exp_C" / "SomeOtherFolder"
    for d in (result_a, result_b, other):
        d.mkdir(parents=True)

    (result_a / "zData2.mat").write_bytes(b"dummy")
    (result_b / "zData2.mat").write_bytes(b"dummy")
    # 'other' has no zData2.mat → must be excluded

    found = _find_result_dirs(tmp_path, rf)
    assert set(found) == {result_a, result_b}


def test_find_result_dirs_empty_when_no_match(tmp_path: Path) -> None:
    (tmp_path / "Analysis").mkdir()
    found = _find_result_dirs(tmp_path, "iT_03_r2_03")
    assert found == []


# ── merge_only aggregation ───────────────────────────────────────────────────

def test_merge_only_aggregates_sparks_from_two_folders(tmp_path: Path) -> None:
    """Core: sparks from 2 result folders are pooled and ROI-clustered."""
    rf = "iT_03_r2_03"
    meta = _make_meta()

    sparks_a = [_make_spark_dict(1, px=5, py=5), _make_spark_dict(2, px=6, py=5)]
    sparks_b = [_make_spark_dict(3, px=15, py=15), _make_spark_dict(4, px=15, py=16)]

    root_dir = tmp_path / "root"
    result_a = root_dir / f"{rf}_a"
    result_b = root_dir / f"{rf}_b"
    result_a.mkdir(parents=True)
    result_b.mkdir(parents=True)
    # Create dummy zMetaData.mat so zmeta.exists() is True
    (result_a / "zMetaData.mat").write_bytes(b"dummy")

    with (
        patch(
            "sparksimple.tools.merge_only._find_result_dirs",
            side_effect=lambda root, tag: [result_a, result_b],
        ),
        patch(
            "sparksimple.tools.merge_only._load_sparks_from_mat",
            side_effect=[sparks_a, sparks_b],
        ),
        patch(
            "sparksimple.tools.merge_only._load_metadata_from_mat",
            return_value=meta,
        ),
        patch("sparksimple.tools.merge_only.save_zdata2_mat"),
        patch("sparksimple.tools.merge_only.save_zmetadata_mat"),
        patch("sparksimple.tools.merge_only.generate_accepted_csv"),
        patch("sparksimple.tools.merge_only._generate_report"),
    ):
        out = merge_only(
            experiment_dirs=[root_dir],
            rf_tag=rf,
            roi_r_um_val=2.0,
            output_folder="MERGED",
        )

    assert out == root_dir / "MERGED"


def test_merge_only_reindexes_spark_ids(tmp_path: Path) -> None:
    """IDs must be globally unique (1-based) across merged folders."""
    rf = "iT_03_r2_03"
    meta = _make_meta()

    # Both folders have sparks with id=1 — after merge they must differ
    sparks_a = [_make_spark_dict(1, px=5, py=5)]
    sparks_b = [_make_spark_dict(1, px=6, py=6)]

    root_dir = tmp_path / "root"
    result_dir = root_dir / f"{rf}_x"
    result_dir.mkdir(parents=True)
    (result_dir / "zMetaData.mat").write_bytes(b"dummy")

    captured_sparks: list[list] = []

    def _fake_cluster(sparks, volume, mask, dx_um, dt_ms, roi_r_um):
        captured_sparks.append(list(sparks))
        from sparksimple.roi.models import ClusteringResult
        return ClusteringResult(rois=[], sparks=sparks)

    with (
        patch(
            "sparksimple.tools.merge_only._find_result_dirs",
            side_effect=lambda root, tag: [result_dir, result_dir],
        ),
        patch(
            "sparksimple.tools.merge_only._load_sparks_from_mat",
            side_effect=[sparks_a, sparks_b],
        ),
        patch(
            "sparksimple.tools.merge_only._load_metadata_from_mat",
            return_value=meta,
        ),
        patch("sparksimple.tools.merge_only.cluster_rois", side_effect=_fake_cluster),
        patch("sparksimple.tools.merge_only.save_zdata2_mat"),
        patch("sparksimple.tools.merge_only.save_zmetadata_mat"),
        patch("sparksimple.tools.merge_only.generate_accepted_csv"),
        patch("sparksimple.tools.merge_only._generate_report"),
    ):
        merge_only(
            experiment_dirs=[root_dir],
            rf_tag=rf,
            roi_r_um_val=2.0,
            output_folder="MERGED",
        )

    assert len(captured_sparks) == 1
    pooled = captured_sparks[0]
    assert len(pooled) == 2
    ids = [s["id"] for s in pooled]
    assert ids == [1, 2], f"Expected [1, 2] but got {ids}"


def test_merge_only_raises_when_no_result_dirs(tmp_path: Path) -> None:
    """RuntimeError must be raised when no matching result folders are found."""
    root_dir = tmp_path / "empty"
    root_dir.mkdir()

    with pytest.raises(RuntimeError, match="No result folders"):
        merge_only(
            experiment_dirs=[root_dir],
            rf_tag="iT_03_r2_03",
            roi_r_um_val=2.0,
            output_folder="MERGED",
        )


# ── _load_sparks_from_mat (structural smoke-test with mock) ──────────────────

def test_load_sparks_from_mat_returns_dicts(tmp_path: Path) -> None:
    """Verify that _load_sparks_from_mat converts struct records to dicts."""
    # Use SimpleNamespace instead of MagicMock — MagicMock is iterable,
    # which causes numpy to create an unexpected nested array shape.
    spk_struct = SimpleNamespace(
        px=np.array([[5]]),
        py=np.array([[10]]),
        pt=np.array([[3]]),
        FWHM=np.array([[1.1]]),
        timesignal=np.ones((20, 1)),
        timeinterval=np.arange(1, 21).reshape(1, 20),
        r2=np.array([[0.95]]),
        amp=np.array([[1.5]]),
        tau=np.array([[28.0]]),
        ror=np.array([[0.04]]),
        t2p=np.array([[14.0]]),
        FDHM=np.array([[45.0]]),
        AMP=np.array([[2.1]]),
        BL=np.array([[0.6]]),
        mass=np.array([[7.5]]),
        dist2memb=np.array([[3.0]]),
        id=np.array([[1]]),
        good=np.array([[1.0]]),
        fail=np.array(["no"]),
        dist2spk=np.array([[0.0]]),
        roi=np.array([[0]]),
    )

    # spkF shape (1, 1): matches what scipy.io.loadmat returns for a 1-element struct array
    fake_mat = {"spkF": np.empty((1, 1), dtype=object)}
    fake_mat["spkF"][0, 0] = spk_struct

    with patch("scipy.io.loadmat", return_value=fake_mat):
        result = _load_sparks_from_mat(tmp_path / "zData2.mat")

    assert len(result) == 1
    s = result[0]
    assert s["px"] == 5
    assert s["py"] == 10
    assert s["good"] is True
    assert s["fail"] == "no"
    assert isinstance(s["timesignal"], np.ndarray)


# ── _load_metadata_from_mat strictness ──────────────────────────────────────

def _spark_feature(spark_id: int) -> FilteredSpark:
    feat = SparkFeature(
        px=5, py=5, pt=1, x_0b=4, y_0b=4, t_0b=0,
        dist2memb=1.0, fwhm=1.5,
        timesignal=np.ones(3), timeinterval=np.arange(1, 4),
        r2=0.95, amp=1.2, tau=20.0, ror=0.1, t2p=3.0, fdhm=4.0,
        AMP=2.2, BL=1.0, mass=10.0,
    )
    return FilteredSpark(id=spark_id, good=True, fail="no", feature=feat)


def _write_result_folder(
    path: Path, *, dx=0.2, dt=10.0, temps=50, mask_shape=(20, 20), fo=100.0,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    save_zdata2_mat(str(path / "zData2.mat"), [_spark_feature(1)], [])
    meta: dict = {"DX": dx, "DT": dt, "temps": temps, "fo": fo}
    if mask_shape is not None:
        meta["mask"] = np.ones(mask_shape, dtype=np.float64)
    scipy.io.savemat(str(path / "zMetaData.mat"), meta)


def test_load_metadata_from_mat_returns_validated_fields(tmp_path: Path) -> None:
    _write_result_folder(tmp_path, dx=0.25, dt=12.0, temps=40)
    meta = _load_metadata_from_mat(tmp_path / "zMetaData.mat")
    assert meta["DX"] == pytest.approx(0.25)
    assert meta["DT"] == pytest.approx(12.0)
    assert meta["temps"] == 40
    assert meta["mask"].shape == (20, 20)


def test_load_metadata_from_mat_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(MergeCalibrationError, match="missing"):
        _load_metadata_from_mat(tmp_path / "zMetaData.mat")


@pytest.mark.parametrize("bad_dx", [0.0, -0.2, float("nan"), float("inf")])
def test_load_metadata_from_mat_rejects_invalid_dx(tmp_path: Path, bad_dx) -> None:
    _write_result_folder(tmp_path, dx=bad_dx)
    with pytest.raises(MergeCalibrationError, match="DX"):
        _load_metadata_from_mat(tmp_path / "zMetaData.mat")


@pytest.mark.parametrize("bad_dt", [0.0, -1.0, float("nan"), float("inf")])
def test_load_metadata_from_mat_rejects_invalid_dt(tmp_path: Path, bad_dt) -> None:
    _write_result_folder(tmp_path, dt=bad_dt)
    with pytest.raises(MergeCalibrationError, match="DT"):
        _load_metadata_from_mat(tmp_path / "zMetaData.mat")


def test_load_metadata_from_mat_rejects_missing_mask(tmp_path: Path) -> None:
    _write_result_folder(tmp_path, mask_shape=None)
    with pytest.raises(MergeCalibrationError, match="mask"):
        _load_metadata_from_mat(tmp_path / "zMetaData.mat")


def test_load_metadata_from_mat_rejects_non_positive_temps(tmp_path: Path) -> None:
    _write_result_folder(tmp_path, temps=0)
    with pytest.raises(MergeCalibrationError, match="temps"):
        _load_metadata_from_mat(tmp_path / "zMetaData.mat")


# ── _check_calibration_compatible ───────────────────────────────────────────

def test_check_calibration_compatible_allows_matching_folders() -> None:
    metas = [
        {"DX": 0.2, "DT": 10.0, "source": Path("a")},
        {"DX": 0.201, "DT": 10.05, "source": Path("b")},
    ]
    _check_calibration_compatible(metas)  # must not raise


def test_check_calibration_compatible_rejects_conflicting_dx() -> None:
    metas = [
        {"DX": 0.2, "DT": 10.0, "source": Path("folder_a")},
        {"DX": 0.4, "DT": 10.0, "source": Path("folder_b")},
    ]
    with pytest.raises(MergeCalibrationError, match="folder_a") as exc:
        _check_calibration_compatible(metas)
    assert "folder_b" in str(exc.value)


# ── merge_only end-to-end: validation and duration coherence ───────────────

def test_merge_only_end_to_end_with_valid_metadata(tmp_path: Path) -> None:
    rf = "iT_03_r2_03"
    root = tmp_path / "root"
    result_a = root / f"{rf}_a"
    result_b = root / f"{rf}_b"
    _write_result_folder(result_a, dx=0.2, dt=10.0, temps=50)
    _write_result_folder(result_b, dx=0.2, dt=10.0, temps=30)

    with patch("sparksimple.tools.merge_only._generate_report"):
        out_dir = merge_only(
            experiment_dirs=[root], rf_tag=rf, roi_r_um_val=2.0, output_folder="MERGED",
        )

    assert out_dir == root / "MERGED"
    assert (out_dir / "zDataMERGE.mat").exists()
    assert (out_dir / "zMetaData.mat").exists()
    assert (out_dir / "yData_MERGED.csv").exists()

    merged_meta = scipy.io.loadmat(str(out_dir / "zMetaData.mat"), squeeze_me=True)
    # Total duration must be the SUM of each folder's own duration
    # ((50-1) + (30-1)) frame intervals, never a single folder's temps reused.
    assert int(merged_meta["temps"]) == (50 - 1) + (30 - 1) + 1


def test_merge_only_aborts_on_incompatible_dx_without_writing_output(tmp_path: Path) -> None:
    rf = "iT_03_r2_03"
    root = tmp_path / "root"
    result_a = root / f"{rf}_a"
    result_b = root / f"{rf}_b"
    _write_result_folder(result_a, dx=0.2, dt=10.0, temps=50)
    _write_result_folder(result_b, dx=0.4, dt=10.0, temps=30)

    with pytest.raises(MergeCalibrationError, match=f"{rf}_a"):
        merge_only(experiment_dirs=[root], rf_tag=rf, roi_r_um_val=2.0, output_folder="MERGED")

    assert not (root / "MERGED").exists()


def test_merge_only_aborts_on_incompatible_dt_without_writing_output(tmp_path: Path) -> None:
    rf = "iT_03_r2_03"
    root = tmp_path / "root"
    result_a = root / f"{rf}_a"
    result_b = root / f"{rf}_b"
    _write_result_folder(result_a, dx=0.2, dt=10.0, temps=50)
    _write_result_folder(result_b, dx=0.2, dt=20.0, temps=30)

    with pytest.raises(MergeCalibrationError):
        merge_only(experiment_dirs=[root], rf_tag=rf, roi_r_um_val=2.0, output_folder="MERGED")

    assert not (root / "MERGED").exists()


def test_merge_only_aborts_when_a_folder_has_no_metadata(tmp_path: Path) -> None:
    rf = "iT_03_r2_03"
    root = tmp_path / "root"
    result_a = root / f"{rf}_a"
    result_b = root / f"{rf}_b"
    _write_result_folder(result_a, dx=0.2, dt=10.0, temps=50)
    result_b.mkdir(parents=True)
    save_zdata2_mat(str(result_b / "zData2.mat"), [_spark_feature(1)], [])
    # result_b has zData2.mat but no zMetaData.mat -- must abort, not invent DX=0.

    with pytest.raises(MergeCalibrationError):
        merge_only(experiment_dirs=[root], rf_tag=rf, roi_r_um_val=2.0, output_folder="MERGED")

    assert not (root / "MERGED").exists()
