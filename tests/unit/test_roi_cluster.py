"""Unit tests for sparksimple.roi.cluster (ROI clustering)."""

import numpy as np
import pytest
from sparksimple.filtering.models import FilteredSpark
from sparksimple.features.models import SparkFeature
from sparksimple.roi.cluster import cluster_rois


def _make_dummy_spark(spark_id: int, px: int, py: int, good: bool = True) -> FilteredSpark:
    """Helper to construct a dummy FilteredSpark record."""
    feat = SparkFeature(
        px=px,
        py=py,
        pt=10,
        x_0b=px - 1,
        y_0b=py - 1,
        t_0b=9,
        dist2memb=2.5,
        fwhm=1.2,
        timesignal=np.ones(50),
        timeinterval=np.arange(1, 51),
        r2=0.95,
        amp=1.5,
        tau=25.0,
        ror=0.05,
        t2p=15.0,
        fdhm=40.0,
        AMP=2.5,
        BL=1.0,
        mass=10.0,
    )
    return FilteredSpark(id=spark_id, good=good, fail="no" if good else "out of field of view (X)", feature=feat)


def test_cluster_rois_empty():
    volume = np.ones((50, 50, 20))
    mask = np.ones((50, 50))
    res = cluster_rois([], volume, mask, dx_um=0.1, dt_ms=10.0, roi_r_um=1.5)
    assert len(res.rois) == 0
    assert len(res.sparks) == 0


def test_cluster_rois_single_spark():
    """MATLAB assigns roi=1 to a singleton and leaves ROIs empty."""
    sparks = [_make_dummy_spark(1, px=20, py=20, good=True)]
    volume = np.ones((50, 50, 20))
    mask = np.ones((50, 50))
    res = cluster_rois(sparks, volume, mask, dx_um=0.1, dt_ms=10.0, roi_r_um=1.5)
    assert len(res.rois) == 0
    assert res.sparks[0].feature.dist2spk == 0.0
    assert res.sparks[0].feature.roi == 1


def test_cluster_rois_two_clusters():
    # Group A: 3 sparks near (10, 10) px -> (1.0, 1.0) um
    # Group B: 1 spark near (40, 40) px -> (4.0, 4.0) um
    # Distance between A and B is ~ 4.2 um > 1.5 um (roi_r_um)
    sparks = [
        _make_dummy_spark(1, px=10, py=10, good=True),
        _make_dummy_spark(2, px=11, py=10, good=True),
        _make_dummy_spark(3, px=10, py=11, good=True),
        _make_dummy_spark(4, px=40, py=40, good=True),
        _make_dummy_spark(5, px=20, py=20, good=False),  # Rejected, should be ignored for ROIs
    ]
    volume = np.random.randn(50, 50, 20)
    mask = np.ones((50, 50))

    res = cluster_rois(sparks, volume, mask, dx_um=0.1, dt_ms=10.0, roi_r_um=1.5)

    assert len(res.rois) == 2
    # ROI 1 must be Group A because it has 3 sparks (largest cluster)
    roi1 = res.rois[0]
    roi2 = res.rois[1]

    assert roi1.id == 1
    assert roi1.Nsparks == 3
    assert roi2.id == 2
    assert roi2.Nsparks == 1

    # Check spark ROI assignments
    assert sparks[0].feature.roi == 1
    assert sparks[1].feature.roi == 1
    assert sparks[2].feature.roi == 1
    assert sparks[3].feature.roi == 2
    assert sparks[4].feature.roi == 0  # Rejected spark has roi 0

    # Check dist2closest
    assert roi1.dist2closest > 0.0
    assert roi2.dist2closest > 0.0
