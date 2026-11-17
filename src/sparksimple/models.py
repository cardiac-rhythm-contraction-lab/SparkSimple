"""sparksimple.models — Shared data structures for the SparkSimple2 pipeline.

All result types and metadata containers live here to avoid circular imports
and to keep I/O, config, and algorithm modules free of cross-dependencies.

MATLAB correspondence
---------------------
ExperimentMetadata  ←  fields saved in zMetaData.mat (DX, DT, roiR, mask, temps)
                        and read by physical_parameters.m / physical_parametersC.m
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path


# ── Experiment geometry & acquisition parameters ──────────────────────────────

@dataclass
class ExperimentMetadata:
    """Physical parameters of one acquisition.

    These values are read from the XML metadata file accompanying each dataset.
    They map directly to the variables DX, DY, DT, NXPIX, NYPIX used throughout
    the MATLAB pipeline (sparkDetect.m, sparkFiltSimple.m, sparkParametersFast2.m).

    Attributes
    ----------
    dx_um : float
        Pixel size along X (spatial), in micrometres.
        MATLAB: DX
    dy_um : float
        Pixel size along Y (spatial), in micrometres.
        MATLAB: DY  (often equal to dx_um for square pixels)
    dt_ms : float
        Frame interval (temporal pixel size), in milliseconds.
        MATLAB: DT
    n_pixels_x : int
        Number of pixels along X.
        MATLAB: NXPIX
    n_pixels_y : int
        Number of pixels along Y.
        MATLAB: NYPIX
    n_frames : int
        Number of frames (time points).
        MATLAB: NumberOfElements for DimID="T"
    n_channels : int
        Number of fluorescence channels.  1 for single-channel acquisitions.
        MATLAB: NumberOfChannels (read from XML header).
    source_xml : str
        Filename (not path) of the XML that was parsed, for traceability.
    """

    dx_um: float
    dy_um: float
    dt_ms: float
    n_pixels_x: int
    n_pixels_y: int
    n_frames: int
    n_channels: int = 1
    source_xml: str = ""

    def __post_init__(self) -> None:
        for name in ("dx_um", "dy_um", "dt_ms"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("n_pixels_x", "n_pixels_y", "n_frames", "n_channels"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if not math.isclose(float(self.dx_um), float(self.dy_um), rel_tol=1e-2, abs_tol=1e-9):
            raise ValueError(
                "SparkSimple uses one scalar spatial calibration, so dx_um "
                f"({self.dx_um!r}) and dy_um ({self.dy_um!r}) must agree within 1%"
            )

    # ── Derived convenience properties ────────────────────────────────────────

    @property
    def total_time_s(self) -> float:
        """Elapsed time from the first frame to the last frame."""
        return max(0, self.n_frames - 1) * self.dt_ms / 1000.0

    @property
    def fov_x_um(self) -> float:
        """Field of view width in micrometres."""
        return self.n_pixels_x * self.dx_um

    @property
    def fov_y_um(self) -> float:
        """Field of view height in micrometres."""
        return self.n_pixels_y * self.dy_um

    @property
    def is_simulated(self) -> bool:
        """True if source_xml is Format A (metadata.xml) or simulated data."""
        return self.source_xml.lower() == "metadata.xml"

    def __repr__(self) -> str:
        return (
            f"ExperimentMetadata("
            f"dx={self.dx_um:.4f}µm, dy={self.dy_um:.4f}µm, dt={self.dt_ms:.4f}ms, "
            f"size=({self.n_pixels_y}×{self.n_pixels_x}), "
            f"frames={self.n_frames}, channels={self.n_channels})"
        )
