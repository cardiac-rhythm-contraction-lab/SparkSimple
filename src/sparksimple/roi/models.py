"""Data models for ROI clustering and per-ROI summary metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class ROIRecord:
    """Represents a spatial Region of Interest (ROI) grouping multiple accepted sparks."""

    id: int  # 1-based ROI identifier
    px: int  # 1-based X centroid pixel coordinate
    py: int  # 1-based Y centroid pixel coordinate
    Nsparks: int  # Number of sparks forming this ROI
    FWHM: float  # Mean FWHM of forming sparks (um)
    r2: float  # Mean decay fit R^2 of forming sparks
    amp: float  # Mean relative amplitude (dF/Fo) of forming sparks
    tau: float  # Mean decay time constant (ms) of forming sparks
    ror: float  # Mean rate of rise (dF/Fo/ms) of forming sparks
    t2p: float  # Mean time-to-peak (ms) of forming sparks
    FDHM: float  # Mean FDHM (ms) of forming sparks
    AMP: float  # Mean peak intensity (dF/Fo) of forming sparks
    BL: float  # Mean pre-spark baseline (dF/Fo) of forming sparks
    dist2memb: float  # Mean distance to membrane (um) of forming sparks
    mass: float  # Mean spark mass (ms * dF/Fo) of forming sparks
    Nspk: int  # Duplicate field matching MATLAB struct contract (Nspk == Nsparks)
    dist2closest: float  # Distance in um to closest neighboring ROI center
    max_diameter_um: float = 0.0  # Largest pairwise member-spark distance
    signal: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))  # 1D mean time trace (length T)


@dataclass
class ClusteringResult:
    """Container for the output of ROI clustering."""

    rois: list[ROIRecord]
    sparks: list[Any]  # List of spark feature objects with updated dist2spk and roi fields
