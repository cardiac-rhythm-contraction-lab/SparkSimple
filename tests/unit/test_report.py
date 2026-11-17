"""Unit tests for sparksimple.report (figures + HTML generation)."""

from __future__ import annotations

import re
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

from sparksimple.features.models import SparkFeature
from sparksimple.filtering.models import FilteredSpark
from sparksimple.roi.models import ROIRecord, ClusteringResult
from sparksimple.export.models import ExportSummary, ExportPaths
from sparksimple.export.pipeline import Phase10Result
from sparksimple.report.figures import (
    generate_blank,
    generate_fo_map,
    generate_location_sparks,
    generate_roi_signal,
    generate_mask_image,
    generate_spark_feature_plot,
    generate_spark_overview,
    generate_spark_xt_overview,
    generate_location_rois,
    generate_detection_film,
)
from sparksimple.report.html import generate_report


# ── Helpers ─────────────────────────────────────────────────────────────────

def _make_spark(sid: int, px: int = 10, py: int = 10, good: bool = True) -> FilteredSpark:
    feat = SparkFeature(
        px=px, py=py, pt=5,
        x_0b=px - 1, y_0b=py - 1, t_0b=4,
        dist2memb=1.5,
        fwhm=1.0,
        timesignal=np.linspace(1.0, 2.5, 30),
        timeinterval=np.arange(1, 31),
        r2=0.92,
        amp=1.2,
        tau=20.0,
        ror=0.04,
        t2p=12.0,
        fdhm=35.0,
        AMP=2.2,
        BL=1.0,
        mass=8.0,
    )
    return FilteredSpark(
        id=sid,
        good=good,
        fail="no" if good else "relative amp too small",
        feature=feat,
    )


def _make_roi(rid: int) -> ROIRecord:
    return ROIRecord(
        id=rid, px=12, py=12, Nsparks=2,
        FWHM=1.1, r2=0.91, amp=1.3, tau=22.0,
        ror=0.05, t2p=11.0, FDHM=36.0, AMP=2.3,
        BL=1.0, dist2memb=1.2, mass=9.0, Nspk=2,
        dist2closest=5.0,
        signal=np.sin(np.linspace(0, 2 * np.pi, 50)) * 0.1 + 1.0,
    )


def _make_result(n_accepted: int = 2, n_rejected: int = 1) -> Phase10Result:
    sparks = [_make_spark(i + 1, px=5 + i * 3, py=5 + i * 3) for i in range(n_accepted)]
    sparks += [_make_spark(n_accepted + j + 1, good=False) for j in range(n_rejected)]
    # Simulate dist2spk / roi attributes set by cluster_rois
    for s in sparks:
        s.dist2spk = 2.0
        s.roi = 1
    rois = [_make_roi(1)]
    clustering = ClusteringResult(rois=rois, sparks=sparks)
    summary = ExportSummary(
        n_candidates=len(sparks),
        n_accepted=n_accepted,
        n_rejected=n_rejected,
        spark_freq=0.5,
        spark_density=10.0,
        cell_area_um2=1000.0,
        duration_s=4.0,
        mean_fo=1200.0,
        std_fo=50.0,
    )
    paths = ExportPaths(
        accepted_csv="dummy.csv",
        rejected_csv="dummy_r.csv",
        zdata2_mat="dummy.mat",
        zmetadata_mat="dummy_m.mat",
    )
    return Phase10Result(clustering=clustering, summary=summary, paths=paths)


def _is_valid_png(path: Path) -> bool:
    """Return True if the file starts with the PNG magic bytes."""
    if not path.exists():
        return False
    with path.open("rb") as f:
        sig = f.read(8)
    return sig == b"\x89PNG\r\n\x1a\n"


# ── Tests: figure generation ─────────────────────────────────────────────────

def test_generate_blank(tmp_path: Path) -> None:
    generate_blank(tmp_path)
    out = tmp_path / "extraFigs" / "blank.png"
    assert out.exists(), "blank.png was not created"
    assert _is_valid_png(out), "blank.png is not a valid PNG"


def test_generate_location_sparks(tmp_path: Path) -> None:
    sparks = [_make_spark(1), _make_spark(2, px=20, py=20)]
    mask = np.ones((50, 50), dtype=np.uint8)
    generate_location_sparks(sparks, mask, dx_um=0.2, output_dir=tmp_path)
    out = tmp_path / "extraFigs" / "LocationSparks.png"
    assert out.exists(), "LocationSparks.png was not created"


def test_generate_location_sparks_no_mask(tmp_path: Path) -> None:
    sparks = [_make_spark(1)]
    generate_location_sparks(sparks, mask=None, dx_um=0.2, output_dir=tmp_path)
    assert (tmp_path / "extraFigs" / "LocationSparks.png").exists()


def test_generate_location_rois(tmp_path: Path) -> None:
    rois = [_make_roi(1), _make_roi(2)]
    rois[1].id = 2
    mask = np.ones((50, 50), dtype=np.uint8)
    generate_location_rois(rois, mask, dx_um=0.2, output_dir=tmp_path)
    assert (tmp_path / "extraFigs" / "LocationRois.png").exists()


def test_generate_fo_map(tmp_path: Path) -> None:
    fo_map = np.linspace(100.0, 200.0, 400).reshape(20, 20)
    generate_fo_map(fo_map, tmp_path)
    out = tmp_path / "extraFigs" / "Fo.png"
    assert out.exists(), "Fo.png was not created"
    assert _is_valid_png(out), "Fo.png is not a valid PNG"


def test_generate_mask_image(tmp_path: Path) -> None:
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 1
    generate_mask_image(mask, tmp_path)
    out = tmp_path / "extraFigs" / "mask.png"
    assert out.exists(), "mask.png was not created"
    assert _is_valid_png(out), "mask.png is not a valid PNG"


def test_generate_roi_signal(tmp_path: Path) -> None:
    roi = _make_roi(3)
    sparks = [_make_spark(1), _make_spark(2, good=False)]
    generate_roi_signal(
        roi,
        sparks,
        dx_um=0.2,
        dt_ms=10.0,
        roi_r_um=1.5,
        output_dir=tmp_path,
    )
    out = tmp_path / "roiFeat" / "ROI3.jpg"
    assert out.exists(), "ROI3.jpg was not created"


def test_generate_roi_signal_empty(tmp_path: Path) -> None:
    roi = _make_roi(99)
    roi.signal = np.array([], dtype=np.float64)
    generate_roi_signal(roi, [], dx_um=0.2, dt_ms=10.0, roi_r_um=1.5, output_dir=tmp_path)
    assert (tmp_path / "roiFeat" / "ROI99.jpg").exists()


def test_generate_spark_feature_plot(tmp_path: Path) -> None:
    spark = _make_spark(7)
    generate_spark_feature_plot(spark, volume=None, dx_um=0.2, dt_ms=10.0, output_dir=tmp_path)
    assert (tmp_path / "spkFeat" / "Spk7.png").exists()


def test_generate_spark_overview_no_volume(tmp_path: Path) -> None:
    spark = _make_spark(4)
    generate_spark_overview(spark, volume=None, dx_um=0.2, dt_ms=10.0, output_dir=tmp_path)
    assert (tmp_path / "extraFigs" / "Spk4.png").exists()
    assert (tmp_path / "extraFigs" / "spk4.png").exists()


def test_generate_spark_overview_with_volume(tmp_path: Path) -> None:
    spark = _make_spark(5, px=8, py=8)
    vol = np.random.rand(20, 20, 40).astype(np.float64)
    generate_spark_overview(spark, volume=vol, dx_um=0.2, dt_ms=10.0, output_dir=tmp_path)
    assert (tmp_path / "extraFigs" / "Spk5.png").exists()
    assert (tmp_path / "extraFigs" / "spk5.png").exists()


def test_generate_spark_xt_overview_lowercase_only(tmp_path: Path) -> None:
    spark = _make_spark(6, px=8, py=8)
    vol = np.random.rand(20, 20, 40).astype(np.float64)
    generate_spark_xt_overview(spark, volume=vol, dx_um=0.2, dt_ms=10.0, output_dir=tmp_path)
    assert (tmp_path / "extraFigs" / "spk6.png").exists()


# ── Tests: HTML report ───────────────────────────────────────────────────────

def test_html_contains_title(tmp_path: Path) -> None:
    result = _make_result()
    html_path = generate_report(result, tmp_path, title="TestExp", dx_um=0.2, dt_ms=10.0)
    assert html_path.exists()
    content = html_path.read_text(encoding="utf-8")
    assert "TestExp" in content, "HTML does not contain the experiment title"


def test_html_contains_all_spark_ids(tmp_path: Path) -> None:
    n_accepted = 3
    result = _make_result(n_accepted=n_accepted, n_rejected=1)
    html_path = generate_report(result, tmp_path, title="SpkTest", dx_um=0.2, dt_ms=10.0)
    content = html_path.read_text(encoding="utf-8")
    for i in range(1, n_accepted + 1):
        assert f"selectSpark({i})" in content, f"Missing spark button for ID {i}"


def test_html_contains_roi_buttons(tmp_path: Path) -> None:
    result = _make_result()
    html_path = generate_report(result, tmp_path, title="RoiTest", dx_um=0.2, dt_ms=10.0)
    content = html_path.read_text(encoding="utf-8")
    assert "selectROI(1)" in content, "Missing ROI button for ROI 1"


def test_html_spark_tables_include_decay_r2(tmp_path: Path) -> None:
    result = _make_result(n_accepted=1, n_rejected=1)
    html_path = generate_report(
        result, tmp_path, title="R2Test", dx_um=0.2, dt_ms=10.0
    )
    content = html_path.read_text(encoding="utf-8")

    for table_id in ("tbl-accepted", "tbl-rejected"):
        match = re.search(
            rf'<table id="{table_id}">.*?</table>', content, flags=re.DOTALL
        )
        assert match is not None, f"Missing HTML table: {table_id}"
        table = match.group(0)
        assert ">R²</th>" in table
        assert 'data-val="0.92">0.92</td>' in table


def test_html_structure(tmp_path: Path) -> None:
    result = _make_result()
    html_path = generate_report(result, tmp_path, title="StructTest", dx_um=0.2, dt_ms=10.0)
    content = html_path.read_text(encoding="utf-8")
    for marker in ["roi-strip", "col-left", "col-centre", "col-right", "film-ctrl"]:
        assert marker in content, f"HTML is missing layout element: {marker}"


def test_html_with_no_sparks(tmp_path: Path) -> None:
    result = _make_result(n_accepted=0, n_rejected=0)
    result.clustering.sparks.clear()
    result.clustering.rois.clear()
    html_path = generate_report(result, tmp_path, title="EmptyTest", dx_um=0.2, dt_ms=10.0)
    content = html_path.read_text(encoding="utf-8")
    assert "EmptyTest" in content


def test_generate_detection_film_creates_files(tmp_path: Path) -> None:
    vol = np.random.rand(10, 10, 5).astype(np.float64)
    spark = _make_spark(1, px=5, py=5)
    spark.feature.pt = 3
    generate_detection_film(vol, [spark], [], tmp_path)
    det_dir = tmp_path / "detFilm"
    for i in range(1, 6):
        p = det_dir / f"F1{i:05d}.png"
        assert p.exists(), f"{p.name} was not created"
        with p.open("rb") as f:
            assert f.read(4) == b"\x89PNG", f"{p.name} is not a valid PNG"


def test_generate_detection_film_max_frames(tmp_path: Path) -> None:
    vol = np.random.rand(10, 10, 5).astype(np.float64)
    generate_detection_film(vol, [], [], tmp_path, max_frames=2)
    det_dir = tmp_path / "detFilm"
    files = list(det_dir.iterdir())
    assert len(files) == 2, f"Expected 2 files, got {len(files)}"
    assert (det_dir / "F100001.png").exists()
    assert (det_dir / "F100002.png").exists()
    assert not (det_dir / "F100003.png").exists()


def test_html_with_volume_and_mask(tmp_path: Path) -> None:
    result = _make_result(n_accepted=2)
    vol = np.random.rand(30, 30, 50).astype(np.float64)
    msk = np.ones((30, 30), dtype=np.uint8)
    fo_map = np.linspace(100.0, 200.0, 900).reshape(30, 30)
    html_path = generate_report(
        result, tmp_path, title="VolTest", dx_um=0.2, dt_ms=10.0,
        volume=vol, mask=msk, fo_map=fo_map,
    )
    assert html_path.exists()
    assert (tmp_path / "spkFeat" / "Spk1.png").exists()
    assert (tmp_path / "spkFeat" / "Spk3.png").exists()
    # The XT overview plot is no longer generated in the HTML report.
    assert not (tmp_path / "extraFigs" / "Spk1.png").exists()
    assert not (tmp_path / "extraFigs" / "spk1.png").exists()
    assert not (tmp_path / "extraFigs" / "spk3.png").exists()
    assert (tmp_path / "extraFigs" / "Fo.png").exists()
    assert (tmp_path / "extraFigs" / "mask.png").exists()
    assert (tmp_path / "extraFigs" / "LocationSparks.png").exists()
