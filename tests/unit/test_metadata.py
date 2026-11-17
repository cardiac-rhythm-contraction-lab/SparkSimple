"""tests/unit/test_metadata.py — Unit tests for sparksimple.io.metadata.

Tests cover:
- Parsing Format A XML (Simulated_Experiment/metadata.xml)
- Parsing Format B XML (Real_Experiment/*.xml)
- FileNotFoundError when no XML exists
- Derived properties on ExperimentMetadata
- The _to_um unit conversion helper

Values are compared against what the XML files explicitly contain,
not against arbitrary hardcoded expectations.
"""

import numpy as np
import pytest
from pathlib import Path
import textwrap

from sparksimple.io.metadata import (
    AcquisitionMetadataError,
    _to_um,
    load_metadata,
    validate_stack_against_metadata,
)
from sparksimple.models import ExperimentMetadata


# ── Inline XML fixtures ───────────────────────────────────────────────────────

FORMAT_A_XML = textwrap.dedent("""\
    <?xml version="1.0" encoding="utf-8"?>
    <Dimensions>
      <DimensionDescription DimID="X" NumberOfElements="900" Length="0.000090" Unit="m"/>
      <DimensionDescription DimID="Y" NumberOfElements="220" Length="0.000022" Unit="m"/>
      <DimensionDescription DimID="T" NumberOfElements="360" Length="3.590" Unit="s"/>
    </Dimensions>
""")

FORMAT_B_XML = textwrap.dedent("""\
    <Data><Image><ImageDescription>
      <Channels NumberOfChannels="3"/>
      <Dimensions>X NumberOfElements="512" Length="144.7245" Unit="µm"</Dimensions>
      <Dimensions>Y NumberOfElements="120" Length="33.703" Unit="µm"</Dimensions>
      <Dimensions>T NumberOfElements="1000" Length="9.990" Unit="s"</Dimensions>
    </ImageDescription></Image></Data>
""")

NUMERIC_LAS_XML = textwrap.dedent("""\
    <?xml version="1.0"?>
    <Data><Image><ImageDescription>
      <Channels>
        <ChannelDescription/><ChannelDescription/><ChannelDescription/>
      </Channels>
      <Dimensions>
        <DimensionDescription DimID="1" NumberOfElements="512" Length="3.081415e-05" Unit="m"/>
        <DimensionDescription DimID="2" NumberOfElements="64" Length="3.851769e-06" Unit="m"/>
        <DimensionDescription DimID="4" NumberOfElements="1667" Length="9.695999145508" Unit="s" Voxel="0.00581992745829" TimeIncrement="0.00581992745829"/>
      </Dimensions>
    </ImageDescription></Image></Data>
""")


@pytest.fixture()
def fmt_a_dir(tmp_path):
    (tmp_path / "metadata.xml").write_text(FORMAT_A_XML, encoding="utf-8")
    return tmp_path


@pytest.fixture()
def fmt_b_dir(tmp_path):
    (tmp_path / "MVM_Serie1.xml").write_text(FORMAT_B_XML, encoding="utf-8")
    return tmp_path


@pytest.fixture()
def numeric_las_dir(tmp_path):
    (tmp_path / "Series004.xml").write_text(NUMERIC_LAS_XML, encoding="utf-8")
    return tmp_path


# ── Format A (simulated experiment) ──────────────────────────────────────────

class TestFormatA:
    """Parser tests for Format A (DimensionDescription XML)."""

    def test_loads_without_error(self, fmt_a_dir):
        meta = load_metadata(fmt_a_dir)
        assert isinstance(meta, ExperimentMetadata)

    def test_source_xml_recorded(self, fmt_a_dir):
        meta = load_metadata(fmt_a_dir)
        assert meta.source_xml.endswith(".xml")

    def test_pixel_dimensions(self, fmt_a_dir):
        # X: 900 px, Length=0.000090 m → dx = 90µm/900 = 0.1 µm/pixel
        # Y: 220 px, Length=0.000022 m → dy = 22µm/220 = 0.1 µm/pixel
        meta = load_metadata(fmt_a_dir)
        assert meta.n_pixels_x == 900
        assert meta.n_pixels_y == 220
        assert meta.dx_um == pytest.approx(0.1, rel=1e-4)
        assert meta.dy_um == pytest.approx(0.1, rel=1e-4)

    def test_temporal_resolution(self, fmt_a_dir):
        # T: 360 frames span 359 intervals; 3.590 s / 359 = 10 ms/frame.
        meta = load_metadata(fmt_a_dir)
        assert meta.n_frames == 360
        assert meta.dt_ms == pytest.approx(10.0, rel=1e-4)

    def test_single_channel(self, fmt_a_dir):
        meta = load_metadata(fmt_a_dir)
        assert meta.n_channels == 1

    def test_derived_total_time(self, fmt_a_dir):
        meta = load_metadata(fmt_a_dir)
        assert meta.total_time_s == pytest.approx(3.59, rel=1e-4)

    def test_derived_fov(self, fmt_a_dir):
        meta = load_metadata(fmt_a_dir)
        assert meta.fov_x_um == pytest.approx(90.0, rel=1e-3)


# ── Format B (real experiment) ────────────────────────────────────────────────

class TestFormatB:
    """Parser tests for Format B (LAS-AF quasi-XML)."""

    def test_loads_without_error(self, fmt_b_dir):
        meta = load_metadata(fmt_b_dir)
        assert isinstance(meta, ExperimentMetadata)

    def test_source_xml_recorded(self, fmt_b_dir):
        meta = load_metadata(fmt_b_dir)
        assert meta.source_xml.endswith(".xml")

    def test_pixel_counts(self, fmt_b_dir):
        meta = load_metadata(fmt_b_dir)
        assert meta.n_pixels_x == 512
        assert meta.n_pixels_y == 120
        assert meta.n_frames == 1000

    def test_pixel_size_x(self, fmt_b_dir):
        # X Length=144.7245 µm over 512 pixels → dx ≈ 0.2826 µm/pixel
        meta = load_metadata(fmt_b_dir)
        assert meta.dx_um == pytest.approx(144.7245 / 512, rel=1e-3)

    def test_frame_interval(self, fmt_b_dir):
        # T Length=9.990 s spans 999 frame intervals → dt = 10 ms/frame.
        meta = load_metadata(fmt_b_dir)
        assert meta.dt_ms == pytest.approx(10.0, rel=1e-6)

    def test_multi_channel(self, fmt_b_dir):
        meta = load_metadata(fmt_b_dir)
        assert meta.n_channels == 3


class TestNumericLasFormat:
    def test_numeric_dimensions_use_exact_time_increment(self, numeric_las_dir):
        meta = load_metadata(numeric_las_dir)
        assert meta.source_xml == "Series004.xml"
        assert meta.dx_um == pytest.approx(0.0601839, rel=1e-4)
        assert meta.dt_ms == pytest.approx(5.81992745829, rel=1e-9)
        assert meta.n_channels == 3


# ── Error cases ───────────────────────────────────────────────────────────────

class TestErrorCases:

    def test_missing_xml_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No XML metadata file"):
            load_metadata(tmp_path)

    def test_unrecognised_xml_raises_value_error(self, tmp_path):
        bad_xml = tmp_path / "bad.xml"
        bad_xml.write_text("<root><nothing/></root>", encoding="utf-8")
        with pytest.raises(AcquisitionMetadataError, match="complete X, Y, and T"):
            load_metadata(tmp_path)

    def test_corrupt_leica_temporal_length_is_rejected(self, tmp_path):
        corrupt = NUMERIC_LAS_XML.replace(
            'Length="9.695999145508"', 'Length="1.616323e+04"'
        )
        (tmp_path / "Series004.xml").write_text(corrupt, encoding="utf-8")
        with pytest.raises(AcquisitionMetadataError, match="inconsistent T"):
            load_metadata(tmp_path)

    def test_conflicting_dimension_xmls_are_rejected(self, tmp_path):
        (tmp_path / "a.xml").write_text(FORMAT_A_XML, encoding="utf-8")
        conflict = FORMAT_A_XML.replace('Length="3.590"', 'Length="7.180"')
        (tmp_path / "b.xml").write_text(conflict, encoding="utf-8")
        with pytest.raises(AcquisitionMetadataError, match="conflicting XML calibration"):
            load_metadata(tmp_path)

    def test_stack_shape_must_match_xml(self, fmt_a_dir):
        meta = load_metadata(fmt_a_dir)
        wrong = np.zeros((220, 900, 359), dtype=np.uint16)
        with pytest.raises(AcquisitionMetadataError, match="disagrees with XML"):
            validate_stack_against_metadata(wrong, meta)


# ── ExperimentMetadata calibration invariants ─────────────────────────────────

class TestExperimentMetadataValidation:
    """SparkSimple must never invent DX/DY/DT: __post_init__ is the last line
    of defence even when a caller builds ExperimentMetadata directly."""

    def _kwargs(self, **overrides):
        base = dict(
            dx_um=0.1, dy_um=0.1, dt_ms=10.0,
            n_pixels_x=10, n_pixels_y=10, n_frames=10,
        )
        base.update(overrides)
        return base

    def test_anisotropic_pixels_are_rejected(self):
        with pytest.raises(ValueError, match="scalar spatial calibration"):
            ExperimentMetadata(**self._kwargs(dy_um=0.2))

    def test_isotropy_within_one_percent_tolerance_is_allowed(self):
        meta = ExperimentMetadata(**self._kwargs(dy_um=0.1005))
        assert meta.dy_um == pytest.approx(0.1005)

    @pytest.mark.parametrize("bad_value", [0.0, -1.0, float("nan"), float("inf")])
    def test_dx_um_must_be_finite_and_positive(self, bad_value):
        with pytest.raises(ValueError):
            ExperimentMetadata(**self._kwargs(dx_um=bad_value, dy_um=bad_value))

    @pytest.mark.parametrize("bad_value", [0.0, -1.0, float("nan"), float("inf")])
    def test_dt_ms_must_be_finite_and_positive(self, bad_value):
        with pytest.raises(ValueError):
            ExperimentMetadata(**self._kwargs(dt_ms=bad_value))


# ── Unit helper: _to_um ───────────────────────────────────────────────────────

class TestToUm:

    def test_metres(self):
        assert _to_um(1e-6, "m") == pytest.approx(1.0)

    def test_millimetres(self):
        assert _to_um(1.0, "mm") == pytest.approx(1000.0)

    def test_micrometres(self):
        assert _to_um(5.0, "um") == pytest.approx(5.0)

    def test_micrometres_unicode(self):
        assert _to_um(5.0, "µm") == pytest.approx(5.0)

    def test_nanometres(self):
        assert _to_um(1000.0, "nm") == pytest.approx(1.0)

    def test_unknown_unit_raises(self):
        with pytest.raises(AcquisitionMetadataError, match="unknown spatial unit"):
            _to_um(3.0, "furlongs")
