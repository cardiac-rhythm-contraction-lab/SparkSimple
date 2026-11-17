"""Regression tests for the --roi-linkage contract.

These tests pin the wiring of the spatial ROI linkage method (this is ROI
clustering of recurrent sites, not temporal deduplication of sparks):

* extract_features() must never be called with unknown keyword arguments
  (in particular never with linkage_method).
* cluster_rois() must receive linkage_method=cfg.roi.linkage.
* --roi-linkage {average,complete,single} must reach ROIConfig.linkage and
  from there cluster_rois(), on full runs and on checkpoint resumes.
* The flag must be live: each method can yield a different ROI partition.
* The effective value must appear in the run provenance.
* An invalid value must fail loudly instead of silently defaulting to average.
"""

from __future__ import annotations

import inspect
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sparksimple.cli import main
from sparksimple.config import PipelineConfig, ROIConfig
from sparksimple.features.extract import extract_features
from sparksimple.features.models import SparkFeature
from sparksimple.filtering.models import FilteredSpark
from sparksimple.pipeline import run_pipeline
from sparksimple.roi.cluster import cluster_rois

from tests.unit.test_pipeline_resume import (
    _checkpoint,
    _clustering_result,
    _filtering_result,
    _write_checkpoint,
)

LINKAGE_METHODS = ("average", "complete", "single")


# -- 1. Feature extraction must not receive unknown kwargs -------------------

def test_extract_features_signature_has_no_linkage_method():
    """linkage_method belongs to clustering, never to feature extraction."""
    params = inspect.signature(extract_features).parameters
    assert "linkage_method" not in params
    assert not any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    ), "extract_features must not silently swallow unknown kwargs via **kwargs"


def test_extract_features_rejects_linkage_method_keyword():
    with pytest.raises(TypeError):
        extract_features(
            volume=np.ones((3, 3, 3)),
            ix_matrix=np.zeros((3, 3, 3)),
            mask=np.ones((3, 3), dtype=bool),
            dx_um=0.1,
            dt_ms=10.0,
            roi_r_um=1.5,
            linkage_method="average",
        )


@pytest.mark.parametrize("linkage", LINKAGE_METHODS)
def test_pipeline_calls_extract_features_without_unknown_kwargs(tmp_path, linkage):
    """The Phase 8 call site must stay compatible with the real signature."""
    output_dir = tmp_path / "Analysis"
    _write_checkpoint(output_dir, _checkpoint(completed_stage=2))
    allowed = set(inspect.signature(extract_features).parameters)

    with patch("sparksimple.pipeline.extract_features", autospec=True) as extract, \
        patch("sparksimple.pipeline.filter_sparks", return_value=_filtering_result()), \
        patch("sparksimple.pipeline.cluster_rois", return_value=_clustering_result()), \
        patch("sparksimple.pipeline.run_phase10", return_value=MagicMock()), \
        patch("sparksimple.pipeline._save_checkpoint"):
        run_pipeline(
            input_dir=tmp_path,
            output_dir=output_dir,
            cfg=PipelineConfig(roi=ROIConfig(linkage=linkage)),
            title="Features",
            start_stage=2,
        )

    extract.assert_called_once()
    assert extract.call_args.args == ()
    assert set(extract.call_args.kwargs) <= allowed
    assert "linkage_method" not in extract.call_args.kwargs


# -- 2/4. Every value reaches cluster_rois, full run and resume --------------

@pytest.mark.parametrize("linkage", LINKAGE_METHODS)
@pytest.mark.parametrize("start_stage", (2, 3, 4))
def test_configured_linkage_reaches_cluster_rois(tmp_path, linkage, start_stage):
    """Full-run and resume paths both forward cfg.roi.linkage to clustering."""
    output_dir = tmp_path / "Analysis"
    _write_checkpoint(output_dir, _checkpoint(completed_stage=start_stage))

    with patch("sparksimple.pipeline.extract_features", autospec=True), \
        patch("sparksimple.pipeline.filter_sparks", return_value=_filtering_result()), \
        patch(
            "sparksimple.pipeline.cluster_rois", return_value=_clustering_result()
        ) as cluster, \
        patch("sparksimple.pipeline.run_phase10", return_value=MagicMock()), \
        patch("sparksimple.pipeline._save_checkpoint"):
        run_pipeline(
            input_dir=tmp_path,
            output_dir=output_dir,
            cfg=PipelineConfig(roi=ROIConfig(linkage=linkage)),
            title="Clustering",
            start_stage=start_stage,
        )

    cluster.assert_called_once()
    assert cluster.call_args.kwargs["linkage_method"] == linkage


@pytest.mark.parametrize("linkage", LINKAGE_METHODS)
def test_cli_roi_linkage_flag_reaches_pipeline_config(tmp_path, linkage):
    """--roi-linkage must survive CLI parsing into ROIConfig.linkage."""
    fake_result = MagicMock()
    fake_result.phase10 = None

    argv = ["sparksimple", "run", str(tmp_path), "--roi-linkage", linkage]
    with patch("sparksimple.cli.load_metadata", return_value=MagicMock()), \
        patch("sparksimple.cli.load_image_stack", return_value=(MagicMock(), 0)), \
        patch("sparksimple.cli.run_pipeline", return_value=fake_result) as pipeline, \
        patch.object(sys, "argv", argv):
        main()

    cfg = pipeline.call_args.kwargs["cfg"]
    assert cfg.roi.linkage == linkage


# -- 3. The flag is not dead: synthetic geometry separates the methods -------

def _spark(spark_id: int, px: int, py: int) -> FilteredSpark:
    feat = SparkFeature(
        px=px,
        py=py,
        pt=3,
        x_0b=px - 1,
        y_0b=py - 1,
        t_0b=2,
        dist2memb=2.5,
        fwhm=1.2,
        timesignal=np.ones(6),
        timeinterval=np.arange(1, 7),
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
    return FilteredSpark(id=spark_id, good=True, fail="no", feature=feat)


def _roi_count(linkage_method: str, roi_r_um: float) -> int:
    """Cluster a 1 um-spaced collinear chain of three sparks (dx = 1 um/px).

    Chain distances are 1, 1 and 2 um, so the inter-cluster height of the
    final merge is 1 um (single), 1.5 um (average) and 2 um (complete).
    """
    sparks = [_spark(1, px=5, py=5), _spark(2, px=6, py=5), _spark(3, px=7, py=5)]
    res = cluster_rois(
        sparks=sparks,
        volume=np.ones((11, 13, 6)),
        mask=np.ones((11, 13), dtype=bool),
        dx_um=1.0,
        dt_ms=10.0,
        roi_r_um=roi_r_um,
        linkage_method=linkage_method,
    )
    return len(res.rois)


def test_linkage_flag_changes_roi_partition_single_vs_rest():
    """Cutoff 1.4 um: only single-linkage bridges the chain into one ROI."""
    assert _roi_count("single", 1.4) == 1
    assert _roi_count("average", 1.4) == 2
    assert _roi_count("complete", 1.4) == 2


def test_linkage_flag_changes_roi_partition_complete_vs_rest():
    """Cutoff 1.6 um: complete-linkage still refuses to merge the chain."""
    assert _roi_count("single", 1.6) == 1
    assert _roi_count("average", 1.6) == 1
    assert _roi_count("complete", 1.6) == 2


def test_all_three_linkage_methods_are_distinguishable():
    partitions = {m: (_roi_count(m, 1.4), _roi_count(m, 1.6)) for m in LINKAGE_METHODS}
    assert len(set(partitions.values())) == len(LINKAGE_METHODS)


# -- 5. Default stays 'single' -------------------------------------------------
# MATLAB's clusterROIs.m calls linkage(pd) without a method argument, and
# MATLAB's linkage() defaults to 'single' (nearest-neighbour), not 'average'.
# Confirmed empirically against real validation data (ARI=1.0 for 'single').

def test_default_linkage_is_single():
    assert ROIConfig().linkage == "single"
    assert PipelineConfig().roi.linkage == "single"
    assert (
        inspect.signature(cluster_rois).parameters["linkage_method"].default
        == "single"
    )


def test_cli_default_linkage_is_single(tmp_path):
    fake_result = MagicMock()
    fake_result.phase10 = None

    with patch("sparksimple.cli.load_metadata", return_value=MagicMock()), \
        patch("sparksimple.cli.load_image_stack", return_value=(MagicMock(), 0)), \
        patch("sparksimple.cli.run_pipeline", return_value=fake_result) as pipeline, \
        patch.object(sys, "argv", ["sparksimple", "run", str(tmp_path)]):
        main()

    assert pipeline.call_args.kwargs["cfg"].roi.linkage == "single"


# -- Provenance records the effective value -----------------------------------

@pytest.mark.parametrize("linkage", LINKAGE_METHODS)
def test_provenance_records_effective_linkage(tmp_path, linkage):
    output_dir = tmp_path / "Analysis"
    _write_checkpoint(output_dir, _checkpoint(completed_stage=5))

    with patch("sparksimple.pipeline.run_phase10", return_value=MagicMock()), \
        patch("sparksimple.pipeline._save_checkpoint"):
        result = run_pipeline(
            input_dir=tmp_path,
            output_dir=output_dir,
            cfg=PipelineConfig(roi=ROIConfig(linkage=linkage)),
            title="Provenance",
            start_stage=5,
        )

    assert result.provenance["configuration"]["roi"]["linkage"] == linkage


# -- 6. Invalid values fail loudly --------------------------------------------

def test_invalid_linkage_rejected_by_config():
    with pytest.raises(ValueError, match="average, complete, single"):
        ROIConfig(linkage="ward")


def test_invalid_linkage_rejected_by_cluster_rois():
    sparks = [_spark(1, px=5, py=5), _spark(2, px=6, py=5)]
    with pytest.raises(ValueError, match="average, complete, single"):
        cluster_rois(
            sparks=sparks,
            volume=np.ones((11, 13, 6)),
            mask=np.ones((11, 13), dtype=bool),
            dx_um=1.0,
            dt_ms=10.0,
            roi_r_um=1.5,
            linkage_method="ward",
        )


def test_invalid_linkage_rejected_even_without_accepted_sparks():
    """No silent fallback to 'average' on the empty/no-good-spark shortcut."""
    with pytest.raises(ValueError, match="average, complete, single"):
        cluster_rois(
            sparks=[],
            volume=np.ones((11, 13, 6)),
            mask=np.ones((11, 13), dtype=bool),
            dx_um=1.0,
            dt_ms=10.0,
            roi_r_um=1.5,
            linkage_method="ward",
        )


def test_cli_rejects_invalid_linkage(tmp_path):
    argv = ["sparksimple", "run", str(tmp_path), "--roi-linkage", "ward"]
    with patch.object(sys, "argv", argv):
        with pytest.raises(SystemExit) as exc:
            main()
    assert exc.value.code != 0
