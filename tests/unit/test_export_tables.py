"""Unit tests for sparksimple.export.tables (CSV export helpers)."""

from __future__ import annotations

import tempfile
import os
import numpy as np
import pytest

from sparksimple.export.tables import (
    _format_val,
    _format_csv_row,
    generate_accepted_csv,
    generate_rejected_csv,
)
from sparksimple.export.matlab_compat import save_zdata2_mat, save_zmetadata_mat
from sparksimple.features.models import SparkFeature
from sparksimple.filtering.models import FilteredSpark
from sparksimple.roi.models import ROIRecord


def _make_spark(spark_id: int, good: bool, px: int = 10, py: int = 10, pt: int = 5) -> FilteredSpark:
    feat = SparkFeature(
        px=px, py=py, pt=pt, x_0b=px - 1, y_0b=py - 1, t_0b=pt - 1,
        dist2memb=2.5, fwhm=1.2, timesignal=np.ones(10), timeinterval=np.arange(1, 11),
        r2=0.95, amp=1.5, tau=25.0, ror=0.05, t2p=15.0, fdhm=40.0,
        AMP=2.5, BL=1.0, mass=10.0,
    )
    setattr(feat, "dist2spk", 0.5 if good else 0.0)
    setattr(feat, "roi", 1 if good else 0)
    return FilteredSpark(
        id=spark_id, good=good,
        fail="no" if good else "previous baseline too large",
        feature=feat,
    )


def _make_roi(roi_id: int, px: int = 10, py: int = 10) -> ROIRecord:
    return ROIRecord(
        id=roi_id, px=px, py=py, Nsparks=1, FWHM=1.2, r2=0.95,
        amp=1.5, tau=25.0, ror=0.05, t2p=15.0, FDHM=40.0,
        AMP=2.5, BL=1.0, dist2memb=2.5, mass=10.0, Nspk=1,
        dist2closest=3.0, signal=np.zeros(10),
    )


class TestFormatVal:
    def test_integer(self):
        assert _format_val(5) == "5"

    def test_float_compact(self):
        result = _format_val(3.14159)
        assert "3.1416" in result or "3.14159" in result

    def test_nan(self):
        assert _format_val(float("nan")) == "NaN"

    def test_inf(self):
        assert _format_val(float("inf")) == "Inf"

    def test_string(self):
        assert _format_val("hello") == "hello"

    def test_none(self):
        assert _format_val(None) == ""

    def test_zero(self):
        assert _format_val(0) == "0"


class TestFormatCsvRow:
    def test_basic_row(self):
        result = _format_csv_row([1, 2.5, "hello"])
        assert result.endswith(",\n")
        parts = result.rstrip("\n").rstrip(",").split(",")
        assert parts[0] == "1"
        assert parts[2] == "hello"

    def test_trailing_comma(self):
        result = _format_csv_row([1, 2, 3])
        assert result[-2:] == "\n" or result.endswith(",\n")


class TestGenerateAcceptedCsv:
    def test_creates_file(self):
        sparks = [_make_spark(1, good=True), _make_spark(2, good=False)]
        rois = [_make_roi(1)]
        mask = np.ones((20, 20))
        fo = np.ones((20, 20)) * 5.0

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            generate_accepted_csv(sparks, rois, fo, mask, 0.1, 10.0, 50, path)
            assert os.path.exists(path)
            with open(path) as f:
                content = f.read()
            assert "ID,Xpix,Ypix" in content
            assert "Roi Data" in content
            assert "ROI time signal" in content
        finally:
            os.unlink(path)

    def test_header_has_no_trailing_comma(self):
        sparks = [_make_spark(1, good=True)]
        rois = [_make_roi(1)]
        mask = np.ones((20, 20))
        fo = 5.0

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            path = f.name
        try:
            generate_accepted_csv(sparks, rois, fo, mask, 0.1, 10.0, 50, path)
            with open(path) as f:
                lines = f.readlines()
            header_line = next(ln for ln in lines if ln.startswith("ID,Xpix"))
            assert not header_line.rstrip("\n").endswith(","), (
                f"Accepted spark header should not end with comma: {header_line!r}"
            )
        finally:
            os.unlink(path)

    def test_roi_data_header_no_trailing_comma(self):
        sparks = [_make_spark(1, good=True)]
        rois = [_make_roi(1)]
        mask = np.ones((20, 20))
        fo = 5.0

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            path = f.name
        try:
            generate_accepted_csv(sparks, rois, fo, mask, 0.1, 10.0, 50, path)
            with open(path) as f:
                lines = f.readlines()
            roi_header = next((ln for ln in lines if ln.startswith("ID,Xpix") and "#sparks" in ln), None)
            assert roi_header is not None
            assert not roi_header.rstrip("\n").endswith(","), (
                f"ROI header should not end with comma: {roi_header!r}"
            )
        finally:
            os.unlink(path)

    def test_roi_data_label_no_trailing_comma(self):
        sparks = [_make_spark(1, good=True)]
        rois = [_make_roi(1)]
        mask = np.ones((20, 20))
        fo = 5.0

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            path = f.name
        try:
            generate_accepted_csv(sparks, rois, fo, mask, 0.1, 10.0, 50, path)
            with open(path) as f:
                lines = f.readlines()
            roi_label = next((ln for ln in lines if ln.strip() == "Roi Data"), None)
            assert roi_label is not None, "Expected 'Roi Data' label line"
        finally:
            os.unlink(path)

    def test_roi_time_signal_label_no_comma(self):
        sparks = [_make_spark(1, good=True)]
        rois = [_make_roi(1)]
        mask = np.ones((20, 20))
        fo = 5.0

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            path = f.name
        try:
            generate_accepted_csv(sparks, rois, fo, mask, 0.1, 10.0, 50, path)
            with open(path) as f:
                lines = f.readlines()
            ts_label = next((ln for ln in lines if ln.strip() == "ROI time signal"), None)
            assert ts_label is not None, "Expected 'ROI time signal' label line (no trailing comma)"
        finally:
            os.unlink(path)

    def test_accepted_data_row_has_trailing_comma(self):
        sparks = [_make_spark(1, good=True)]
        rois = [_make_roi(1)]
        mask = np.ones((20, 20))
        fo = 5.0

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            path = f.name
        try:
            generate_accepted_csv(sparks, rois, fo, mask, 0.1, 10.0, 50, path)
            with open(path) as f:
                lines = f.readlines()
            # Find first data row (after ID header, starts with digit)
            in_data = False
            data_row = None
            for ln in lines:
                if ln.startswith("ID,Xpix"):
                    in_data = True
                    continue
                if in_data and ln.strip() and ln.strip()[0].isdigit():
                    data_row = ln
                    break
            assert data_row is not None, "No accepted data row found"
            assert data_row.rstrip("\n").endswith(","), (
                f"Data row should end with comma: {data_row!r}"
            )
        finally:
            os.unlink(path)

    def test_spark_sorted_by_pt(self):
        s1 = _make_spark(1, good=True, pt=30)
        s2 = _make_spark(2, good=True, pt=10)
        rois = [_make_roi(1)]
        mask = np.ones((20, 20))
        fo = 5.0

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            path = f.name
        try:
            generate_accepted_csv([s1, s2], rois, fo, mask, 0.1, 10.0, 50, path)
            with open(path) as f:
                lines = f.readlines()
            in_data = False
            data_rows = []
            for ln in lines:
                stripped = ln.strip()
                if ln.startswith("ID,Xpix") and "ROI" in stripped and "#sparks" not in stripped:
                    in_data = True
                    continue
                if in_data:
                    if stripped in ("Roi Data", "ROI time signal"):
                        break
                    if stripped and stripped[0].isdigit():
                        data_rows.append(stripped.split(","))
            # s2 (pt=10) should appear before s1 (pt=30)
            assert len(data_rows) == 2
            assert data_rows[0][0] == "2"  # spark id=2 first (pt=10)
            assert data_rows[1][0] == "1"  # spark id=1 second (pt=30)
        finally:
            os.unlink(path)


class TestGenerateRejectedCsv:
    def test_creates_file(self):
        sparks = [_make_spark(1, good=False)]
        mask = np.ones((20, 20))
        fo = 5.0

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            generate_rejected_csv(sparks, fo, mask, 0.1, 10.0, 50, path)
            assert os.path.exists(path)
            with open(path) as f:
                content = f.read()
            assert "Rejected spark" in content
        finally:
            os.unlink(path)

    def test_header_no_trailing_comma(self):
        sparks = [_make_spark(1, good=False)]
        mask = np.ones((20, 20))
        fo = 5.0

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            path = f.name
        try:
            generate_rejected_csv(sparks, fo, mask, 0.1, 10.0, 50, path)
            with open(path) as f:
                lines = f.readlines()
            header = next((ln for ln in lines if ln.startswith("ID,Xpix") and "tau" in ln), None)
            assert header is not None
            assert not header.rstrip("\n").endswith(","), (
                f"Rejected header should not end with comma: {header!r}"
            )
        finally:
            os.unlink(path)


class TestSaveZdata2Mat:
    def test_spkF_and_ROIs_written(self):
        import scipy.io as sio_check
        sparks = [_make_spark(1, good=True), _make_spark(2, good=False)]
        rois = [_make_roi(1), _make_roi(2, px=20)]

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as f:
            path = f.name
        try:
            save_zdata2_mat(path, sparks, rois)
            loaded = sio_check.loadmat(path, squeeze_me=True)
            assert "spkF" in loaded
            assert "ROIs" in loaded
            assert len(loaded["spkF"]) == 2
            assert len(loaded["ROIs"]) == 2
        finally:
            os.unlink(path)

    def test_good_flag_preserved(self):
        import scipy.io as sio_check
        sparks = [_make_spark(1, good=True), _make_spark(2, good=False)]

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as f:
            path = f.name
        try:
            save_zdata2_mat(path, sparks, [])
            loaded = sio_check.loadmat(path, squeeze_me=True)
            spkF = loaded["spkF"]
            assert int(spkF[0]["good"]) == 1
            assert int(spkF[1]["good"]) == 0
        finally:
            os.unlink(path)

    def test_feature_confidence_fields_written(self):
        import scipy.io as sio_check
        spark = _make_spark(1, good=True)
        spark.feature.feature_confidence_flags = "multi_peak_or_overlapping_event;tau_low_confidence"
        spark.feature.tau_confidence = "low"
        spark.feature.ror_confidence = "low"
        spark.feature.fdhm_confidence = "caution"
        spark.feature.amp_confidence = "high"
        spark.feature.peak_confidence = "low"
        spark.feature.peak_timing_confidence = "low"
        spark.feature.multi_peak_count = 2
        spark.feature.shoulder_peak_count = 3
        spark.feature.broad_overlap_event = True

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as f:
            path = f.name
        try:
            save_zdata2_mat(path, [spark], [])
            loaded = sio_check.loadmat(path, squeeze_me=True)
            spk = loaded["spkF"]
            assert "feature_confidence_flags" in spk.dtype.names
            assert "tau_confidence" in spk.dtype.names
            assert "fdhm_confidence" in spk.dtype.names
            assert "shoulder_peak_count" in spk.dtype.names
            assert "broad_overlap_event" in spk.dtype.names
            assert str(spk["feature_confidence_flags"]) == "multi_peak_or_overlapping_event;tau_low_confidence"
            assert str(spk["tau_confidence"]) == "low"
            assert str(spk["fdhm_confidence"]) == "caution"
            assert int(spk["multi_peak_count"]) == 2
            assert int(spk["shoulder_peak_count"]) == 3
            assert int(spk["broad_overlap_event"]) == 1
        finally:
            os.unlink(path)


class TestSaveZmetadataMat:
    def test_fields_written(self):
        import scipy.io as sio_check
        mask = np.ones((10, 10))
        fo = np.ones((10, 10)) * 5.0

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as f:
            path = f.name
        try:
            save_zmetadata_mat(path, fo, mask, 100, 0.2, 10.0, 1.5)
            loaded = sio_check.loadmat(path, squeeze_me=True)
            assert "fo" in loaded
            assert "mask" in loaded
            assert "temps" in loaded
            assert "DX" in loaded
            assert "DT" in loaded
            assert "roiR" in loaded
            assert float(loaded["DX"]) == pytest.approx(0.2)
            assert float(loaded["DT"]) == pytest.approx(10.0)
            assert float(loaded["roiR"]) == pytest.approx(1.5)
        finally:
            os.unlink(path)
