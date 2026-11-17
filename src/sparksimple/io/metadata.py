"""Strict Leica acquisition-metadata parsing for SparkSimple.

Spatial and temporal calibration are mandatory scientific inputs. This module
never substitutes defaults, guesses units from plausible values, or silently
selects an arbitrary XML file when several calibrated XMLs disagree.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from sparksimple.models import ExperimentMetadata


class AcquisitionMetadataError(ValueError):
    """Acquisition metadata are missing, invalid, ambiguous, or inconsistent."""


_DIMENSION_IDS = {"1": "X", "2": "Y", "3": "Z", "4": "T"}
_REL_TOL = 1e-2
_MAX_SPARK_FRAME_INTERVAL_MS = 1000.0


def load_metadata(folder: str | Path) -> ExperimentMetadata:
    """Parse and cross-check all calibrated XML files in *folder*.

    XML files without acquisition dimensions are ignored. Any XML that
    contains dimension metadata but is malformed or incomplete is an error.
    If multiple XMLs describe the acquisition, their X/Y/T calibration must
    agree.
    """
    folder = Path(folder)
    xml_files = sorted(folder.glob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No XML metadata file found in {folder}")

    parsed: list[ExperimentMetadata] = []
    errors: list[str] = []
    for xml_path in xml_files:
        text = xml_path.read_text(encoding="utf-8", errors="replace")
        try:
            metadata = _try_parse_format_a(text, xml_path.name)
            if metadata is None:
                metadata = _try_parse_format_b(text, xml_path.name)
        except AcquisitionMetadataError as exc:
            errors.append(f"{xml_path.name}: {exc}")
            continue

        if metadata is not None:
            parsed.append(metadata)
        elif _contains_dimension_metadata(text):
            errors.append(f"{xml_path.name}: unrecognised or incomplete dimensions")

    if errors:
        raise AcquisitionMetadataError(
            f"Invalid acquisition metadata in {folder}: " + "; ".join(errors)
        )
    if not parsed:
        raise AcquisitionMetadataError(
            f"No XML in {folder} contains complete X, Y, and T dimensions"
        )

    reference = parsed[0]
    for candidate in parsed[1:]:
        _require_consistent(reference, candidate)
    return reference


def validate_stack_against_metadata(
    stack: np.ndarray, metadata: ExperimentMetadata
) -> None:
    """Fail when TIFF geometry or frame count disagree with the XML."""
    expected = (
        metadata.n_pixels_y,
        metadata.n_pixels_x,
        metadata.n_frames,
    )
    if tuple(stack.shape) != expected:
        raise AcquisitionMetadataError(
            f"TIFF stack shape {tuple(stack.shape)} disagrees with XML "
            f"(Y, X, T)={expected} from {metadata.source_xml}"
        )


def _contains_dimension_metadata(text: str) -> bool:
    return "DimensionDescription" in text or bool(
        re.search(r"<Dimensions>\s*[XYT]", text, re.IGNORECASE)
    )


def _try_parse_format_a(
    text: str, source: str
) -> ExperimentMetadata | None:
    """Parse well-formed Leica ``DimensionDescription`` XML."""
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        if "DimensionDescription" in text:
            raise AcquisitionMetadataError(f"malformed XML: {exc}") from exc
        return None

    elements = list(root.iter("DimensionDescription"))
    if not elements:
        return None

    dims: dict[str, dict[str, str]] = {}
    for elem in elements:
        raw_id = (elem.get("DimID") or "").strip()
        dim_id = _DIMENSION_IDS.get(raw_id, raw_id.upper())
        if dim_id in {"X", "Y", "T"}:
            dims[dim_id] = {
                "n": elem.get("NumberOfElements", ""),
                "length": elem.get("Length", ""),
                "unit": elem.get("Unit", ""),
                "voxel": elem.get("Voxel", ""),
                "time_increment": elem.get("TimeIncrement", ""),
                "time_increment_unit": elem.get("TimeIncrementUnit", ""),
            }

    missing = {"X", "Y", "T"} - set(dims)
    if missing:
        raise AcquisitionMetadataError(
            "missing mandatory dimensions: " + ", ".join(sorted(missing))
        )

    n_x = _positive_int(dims["X"]["n"], "X.NumberOfElements", source)
    n_y = _positive_int(dims["Y"]["n"], "Y.NumberOfElements", source)
    n_t = _positive_int(dims["T"]["n"], "T.NumberOfElements", source)
    dx_um = _spatial_resolution_um(dims["X"], n_x, "X", source)
    dy_um = _spatial_resolution_um(dims["Y"], n_y, "Y", source)
    dt_ms = _temporal_resolution_ms(dims["T"], n_t, source)
    _validate_resolution(dx_um, dy_um, dt_ms, source)

    channel_elements = list(root.iter("ChannelDescription"))
    n_channels = max(1, len(channel_elements))
    return ExperimentMetadata(
        dx_um=dx_um,
        dy_um=dy_um,
        dt_ms=dt_ms,
        n_pixels_x=n_x,
        n_pixels_y=n_y,
        n_frames=n_t,
        n_channels=n_channels,
        source_xml=source,
    )


def _try_parse_format_b(
    text: str, source: str
) -> ExperimentMetadata | None:
    """Parse legacy Leica quasi-XML ``<Dimensions>X ...`` blocks."""
    dim_pattern = re.compile(
        r"<Dimensions>\s*([XYT])\s+"
        r"NumberOfElements=[\"'](\d+)[\"'].*?"
        r"Length=[\"']([0-9.eE+-]+)[\"'].*?"
        r"Unit=[\"']([^\"']+)[\"']",
        re.DOTALL | re.IGNORECASE,
    )
    dims: dict[str, dict[str, str]] = {}
    for match in dim_pattern.finditer(text):
        dims[match.group(1).upper()] = {
            "n": match.group(2),
            "length": match.group(3),
            "unit": match.group(4).strip(),
            "voxel": "",
            "time_increment": "",
            "time_increment_unit": "",
        }
    if not dims:
        return None

    missing = {"X", "Y", "T"} - set(dims)
    if missing:
        raise AcquisitionMetadataError(
            "missing mandatory dimensions: " + ", ".join(sorted(missing))
        )

    n_x = _positive_int(dims["X"]["n"], "X.NumberOfElements", source)
    n_y = _positive_int(dims["Y"]["n"], "Y.NumberOfElements", source)
    n_t = _positive_int(dims["T"]["n"], "T.NumberOfElements", source)
    dx_um = _spatial_resolution_um(dims["X"], n_x, "X", source)
    dy_um = _spatial_resolution_um(dims["Y"], n_y, "Y", source)
    dt_ms = _temporal_resolution_ms(dims["T"], n_t, source)
    _validate_resolution(dx_um, dy_um, dt_ms, source)

    n_ch_match = re.search(
        r"NumberOfChannels\s*=\s*[\"']?(\d+)[\"']?",
        text,
        re.IGNORECASE,
    )
    n_channels = int(n_ch_match.group(1)) if n_ch_match else 1
    return ExperimentMetadata(
        dx_um=dx_um,
        dy_um=dy_um,
        dt_ms=dt_ms,
        n_pixels_x=n_x,
        n_pixels_y=n_y,
        n_frames=n_t,
        n_channels=n_channels,
        source_xml=source,
    )


def _spatial_resolution_um(
    dim: dict[str, str], count: int, axis: str, source: str
) -> float:
    unit = _required_unit(dim.get("unit", ""), f"{axis}.Unit", source)
    length = dim.get("length", "").strip()
    voxel = dim.get("voxel", "").strip()
    derived = None
    if length:
        derived = _to_um(
            _positive_float(length, f"{axis}.Length", source) / count,
            unit,
        )
    if voxel:
        explicit = _to_um(
            _positive_float(voxel, f"{axis}.Voxel", source), unit
        )
        if derived is not None and not math.isclose(
            explicit, derived, rel_tol=_REL_TOL, abs_tol=1e-9
        ):
            raise AcquisitionMetadataError(
                f"inconsistent {axis}: Voxel={explicit:.9g} µm but "
                f"Length/N={derived:.9g} µm"
            )
        return explicit
    if derived is None:
        raise AcquisitionMetadataError(
            f"{source}: mandatory {axis}.Length/Voxel is missing"
        )
    return derived


def _temporal_resolution_ms(
    dim: dict[str, str], count: int, source: str
) -> float:
    unit = _required_unit(dim.get("unit", ""), "T.Unit", source)
    if count < 2:
        raise AcquisitionMetadataError(
            f"{source}: at least two frames are required to derive DT"
        )

    length = dim.get("length", "").strip()
    derived = None
    if length:
        derived = _to_ms(
            _positive_float(length, "T.Length", source), unit
        ) / (count - 1)

    increment = dim.get("time_increment", "").strip()
    voxel = dim.get("voxel", "").strip()
    explicit = None
    if increment:
        increment_unit = dim.get("time_increment_unit", "").strip() or unit
        explicit = _to_ms(
            _positive_float(increment, "T.TimeIncrement", source),
            increment_unit,
        )
    elif voxel:
        explicit = _to_ms(_positive_float(voxel, "T.Voxel", source), unit)

    if explicit is not None:
        if derived is not None and not math.isclose(
            explicit, derived, rel_tol=_REL_TOL, abs_tol=1e-9
        ):
            raise AcquisitionMetadataError(
                f"inconsistent T: explicit DT={explicit:.9g} ms but "
                f"Length/(N-1)={derived:.9g} ms"
            )
        return explicit
    if derived is None:
        raise AcquisitionMetadataError(
            f"{source}: mandatory T.Length/Voxel/TimeIncrement is missing"
        )
    return derived


def _validate_resolution(
    dx_um: float, dy_um: float, dt_ms: float, source: str
) -> None:
    if not math.isclose(dx_um, dy_um, rel_tol=_REL_TOL, abs_tol=1e-9):
        raise AcquisitionMetadataError(
            f"{source}: anisotropic pixels DX={dx_um:.9g} µm and "
            f"DY={dy_um:.9g} µm are unsupported by the scalar spatial model"
        )
    if dt_ms >= _MAX_SPARK_FRAME_INTERVAL_MS:
        raise AcquisitionMetadataError(
            f"{source}: DT={dt_ms:.9g} ms/frame is incompatible with spark "
            "imaging and indicates corrupt temporal metadata"
        )


def _require_consistent(
    reference: ExperimentMetadata, candidate: ExperimentMetadata
) -> None:
    scalar_fields = ("dx_um", "dy_um", "dt_ms")
    integer_fields = ("n_pixels_x", "n_pixels_y", "n_frames")
    for field in scalar_fields:
        left = float(getattr(reference, field))
        right = float(getattr(candidate, field))
        if not math.isclose(left, right, rel_tol=_REL_TOL, abs_tol=1e-9):
            raise AcquisitionMetadataError(
                f"conflicting XML calibration: {reference.source_xml} has "
                f"{field}={left:.9g}, while {candidate.source_xml} has "
                f"{field}={right:.9g}"
            )
    for field in integer_fields:
        left = int(getattr(reference, field))
        right = int(getattr(candidate, field))
        if left != right:
            raise AcquisitionMetadataError(
                f"conflicting XML dimensions: {reference.source_xml} has "
                f"{field}={left}, while {candidate.source_xml} has "
                f"{field}={right}"
            )


def _positive_int(raw: str, field: str, source: str) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise AcquisitionMetadataError(
            f"{source}: invalid {field}={raw!r}"
        ) from exc
    if value <= 0:
        raise AcquisitionMetadataError(
            f"{source}: {field} must be positive, got {value}"
        )
    return value


def _positive_float(raw: str, field: str, source: str) -> float:
    try:
        value = float(str(raw).replace(",", "."))
    except (TypeError, ValueError) as exc:
        raise AcquisitionMetadataError(
            f"{source}: invalid {field}={raw!r}"
        ) from exc
    if not math.isfinite(value) or value <= 0:
        raise AcquisitionMetadataError(
            f"{source}: {field} must be positive and finite, got {value!r}"
        )
    return value


def _required_unit(raw: str, field: str, source: str) -> str:
    unit = str(raw).strip()
    if not unit:
        raise AcquisitionMetadataError(f"{source}: mandatory {field} is missing")
    return unit


def _to_um(value: float, unit: str) -> float:
    """Convert a spatial value to micrometres; unknown units are errors."""
    normalized = unit.strip().lower().replace("μ", "u").replace("µ", "u")
    factors = {"m": 1e6, "mm": 1e3, "um": 1.0, "nm": 1e-3}
    if normalized not in factors:
        raise AcquisitionMetadataError(f"unknown spatial unit {unit!r}")
    return float(value) * factors[normalized]


def _to_ms(value: float, unit: str) -> float:
    """Convert a temporal value to milliseconds; unknown units are errors."""
    normalized = unit.strip().lower().replace("µ", "u").replace("μ", "u")
    factors = {
        "s": 1000.0,
        "ms": 1.0,
        "us": 1e-3,
        "ns": 1e-6,
    }
    if normalized not in factors:
        raise AcquisitionMetadataError(f"unknown temporal unit {unit!r}")
    return float(value) * factors[normalized]
