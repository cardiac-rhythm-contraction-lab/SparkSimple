"""Unit tests for MATLAB-style SparkSimple2 `sp` resume points."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from sparksimple.detection.fo_reestimate import FoReestimateResult
from sparksimple.detection.local_max import Candidate, CandidateResult
from sparksimple.detection.merge import MergeResult
from sparksimple.detection.wavelet import WaveletResult
from sparksimple.features.models import FeatureExtractionResult, SparkFeature
from sparksimple.filtering.models import FilteredSpark, FilteringResult
import pytest

from sparksimple.models import ExperimentMetadata
from sparksimple.pipeline import PipelineCheckpoint, _save_checkpoint, run_pipeline
from sparksimple.preprocessing.pipeline import PreprocessingResult
from sparksimple.roi.models import ClusteringResult, ROIRecord


def _feature() -> SparkFeature:
    return SparkFeature(
        px=2,
        py=2,
        pt=2,
        x_0b=1,
        y_0b=1,
        t_0b=1,
        dist2memb=1.0,
        fwhm=1.2,
        timesignal=np.ones(3),
        timeinterval=np.arange(1, 4),
        r2=0.95,
        amp=0.5,
        tau=25.0,
        ror=0.05,
        t2p=10.0,
        fdhm=30.0,
        AMP=1.5,
        BL=0.0,
        mass=2.0,
    )


def _filtering_result() -> FilteringResult:
    return FilteringResult(
        records=[
            FilteredSpark(
                id=1,
                good=True,
                fail="no",
                feature=_feature(),
            )
        ]
    )


def _clustering_result() -> ClusteringResult:
    roi = ROIRecord(
        id=1,
        px=2,
        py=2,
        Nsparks=1,
        FWHM=1.2,
        r2=0.95,
        amp=0.5,
        tau=25.0,
        ror=0.05,
        t2p=10.0,
        FDHM=30.0,
        AMP=1.5,
        BL=0.0,
        dist2memb=1.0,
        mass=2.0,
        Nspk=1,
        dist2closest=0.0,
        signal=np.ones(3),
    )
    return ClusteringResult(rois=[roi], sparks=_filtering_result().records)


def _checkpoint(completed_stage: int) -> PipelineCheckpoint:
    volume = np.ones((3, 3, 3), dtype=np.float64)
    mask = np.ones((3, 3), dtype=bool)
    feature_result = FeatureExtractionResult(
        features=[_feature()],
        count=1,
        distance_map=np.ones((3, 3), dtype=np.float64),
    )
    filtering = _filtering_result() if completed_stage >= 4 else None
    clustering = _clustering_result() if completed_stage >= 5 else None

    return PipelineCheckpoint(
        completed_stage=completed_stage,
        metadata=ExperimentMetadata(
            dx_um=0.1,
            dy_um=0.1,
            dt_ms=10.0,
            n_pixels_x=3,
            n_pixels_y=3,
            n_frames=3,
            n_channels=1,
            source_xml="experiment.xml",
        ),
        preprocessing=PreprocessingResult(
            volume_raw=volume,
            volume_norm=volume,
            fo=1.0,
            mask=mask,
            baseline_mode="auto",
            normalisation_mode="df_f0",
            selected_channel=0,
        ),
        wavelet=WaveletResult(
            v2=volume,
            scales=np.array([5, 6], dtype=int),
            filter_window_frames=1,
            is_simulated=False,
        ),
        merge=MergeResult(
            v3_conv=volume,
            umbral=0.1,
            v3_thresholded=volume,
            kernel=np.ones((1, 1, 1), dtype=np.float64),
            centroid_pixel=(1, 1),
        ),
        candidates=CandidateResult(
            ix_matrix=volume,
            candidates=[
                Candidate(
                    y_0b=1,
                    x_0b=1,
                    t_0b=1,
                    y_1b=2,
                    x_1b=2,
                    t_1b=2,
                    step_level=50.0,
                    intensity=1.0,
                )
            ],
            count=1,
            max_candidates=100,
        ),
        fo_reestimate=FoReestimateResult(
            fo_map=np.ones((3, 3), dtype=np.float64),
            volume_renormalised=volume,
            vaux_smoothed=volume,
            candidate_exclusion_mask=np.zeros((3, 3, 3), dtype=bool),
            border_adjustment_mask=np.zeros((3, 3), dtype=bool),
            kernel_3d=np.ones((1, 1, 1), dtype=np.float64),
        ),
        features=feature_result if completed_stage >= 3 else None,
        filtering=filtering,
        clustering=clustering,
        selected_channel_index=1,
        channel_labels=["ch00", "ch01", "ch02"],
    )


def _write_checkpoint(output_dir, checkpoint: PipelineCheckpoint) -> None:
    _save_checkpoint(output_dir / "spark_pipeline_checkpoint.pkl", checkpoint)


def test_start_stage_3_resumes_from_features_and_runs_filter_rois_export(tmp_path):
    output_dir = tmp_path / "Analysis"
    _write_checkpoint(output_dir, _checkpoint(completed_stage=3))
    filtering = _filtering_result()
    clustering = _clustering_result()
    phase10 = MagicMock()

    with patch("sparksimple.pipeline.load_metadata") as load_metadata, \
        patch("sparksimple.pipeline.load_image_stack") as load_image_stack, \
        patch("sparksimple.pipeline.preprocess") as preprocess, \
        patch("sparksimple.pipeline.compute_v2") as compute_v2, \
        patch("sparksimple.pipeline.apply_3d_merge_and_threshold") as merge, \
        patch("sparksimple.pipeline.detect_candidates") as detect_candidates, \
        patch("sparksimple.pipeline.reestimate_fo") as reestimate_fo, \
        patch("sparksimple.pipeline.extract_features") as extract_features, \
        patch("sparksimple.pipeline.filter_sparks", return_value=filtering) as filter_sparks, \
        patch("sparksimple.pipeline.cluster_rois", return_value=clustering) as cluster_rois, \
        patch("sparksimple.pipeline.run_phase10", return_value=phase10) as run_phase10, \
        patch("sparksimple.pipeline._save_checkpoint"):
        result = run_pipeline(
            input_dir=tmp_path,
            output_dir=output_dir,
            title="Resume",
            start_stage=3,
        )

    load_metadata.assert_not_called()
    load_image_stack.assert_not_called()
    preprocess.assert_not_called()
    compute_v2.assert_not_called()
    merge.assert_not_called()
    detect_candidates.assert_not_called()
    reestimate_fo.assert_not_called()
    extract_features.assert_not_called()
    filter_sparks.assert_called_once()
    cluster_rois.assert_called_once()
    assert cluster_rois.call_args.kwargs["linkage_method"] == "single"
    run_phase10.assert_called_once()
    assert result.filtering is filtering
    assert result.clustering is clustering
    assert result.phase10 is phase10


def test_start_stage_5_reuses_checkpoint_rois_and_exports_only(tmp_path):
    output_dir = tmp_path / "Analysis"
    checkpoint = _checkpoint(completed_stage=5)
    _write_checkpoint(output_dir, checkpoint)
    phase10 = MagicMock()

    with patch("sparksimple.pipeline.filter_sparks") as filter_sparks, \
        patch("sparksimple.pipeline.cluster_rois") as cluster_rois, \
        patch("sparksimple.pipeline.run_phase10", return_value=phase10) as run_phase10, \
        patch("sparksimple.pipeline._save_checkpoint"):
        result = run_pipeline(
            input_dir=tmp_path,
            output_dir=output_dir,
            title="ExportOnly",
            start_stage=5,
        )

    filter_sparks.assert_not_called()
    cluster_rois.assert_not_called()
    run_phase10.assert_called_once()
    _, kwargs = run_phase10.call_args
    assert kwargs["clustering"] is result.clustering
    assert kwargs["execution_metadata"]["selected_channel_index"] == 1
    assert kwargs["execution_metadata"]["selected_channel_label"] == "ch01"
    assert result.provenance == kwargs["execution_metadata"]
    assert len(result.clustering.rois) == len(checkpoint.clustering.rois)
    assert result.clustering.rois[0].id == checkpoint.clustering.rois[0].id
    assert result.phase10 is phase10


def _stage1_metadata() -> ExperimentMetadata:
    return ExperimentMetadata(
        dx_um=0.1,
        dy_um=0.1,
        dt_ms=10.0,
        n_pixels_x=3,
        n_pixels_y=3,
        n_frames=3,
        n_channels=1,
        source_xml="experiment.xml",
    )


def _stage1_mocks():
    """Context-manager stack mocking every stage-1..4 computation so run_pipeline
    only exercises metadata loading/override handling end to end."""
    checkpoint = _checkpoint(completed_stage=5)
    return (
        patch("sparksimple.pipeline.load_metadata", return_value=_stage1_metadata()),
        patch(
            "sparksimple.pipeline.load_image_stack",
            return_value=(np.ones((3, 3, 3), dtype=np.float64), 0),
        ),
        patch("sparksimple.pipeline.validate_stack_against_metadata"),
        patch("sparksimple.pipeline.discover_channel_labels", return_value=["ch00"]),
        patch("sparksimple.pipeline.preprocess", return_value=checkpoint.preprocessing),
        patch("sparksimple.pipeline.compute_v2", return_value=checkpoint.wavelet),
        patch("sparksimple.pipeline.apply_3d_merge_and_threshold", return_value=checkpoint.merge),
        patch("sparksimple.pipeline.detect_candidates", return_value=checkpoint.candidates),
        patch("sparksimple.pipeline.reestimate_fo", return_value=checkpoint.fo_reestimate),
        patch("sparksimple.pipeline.extract_features", return_value=checkpoint.features),
        patch("sparksimple.pipeline.filter_sparks", return_value=_filtering_result()),
        patch("sparksimple.pipeline.cluster_rois", return_value=_clustering_result()),
    )


def test_dx_um_override_updates_both_spatial_axes(tmp_path):
    """--dx-um must update dy_um too: SparkSimple has a scalar spatial model."""
    mocks = _stage1_mocks()
    with mocks[0] as load_metadata, mocks[1], mocks[2], mocks[3], mocks[4], \
        mocks[5], mocks[6], mocks[7], mocks[8], mocks[9], mocks[10], mocks[11]:
        result = run_pipeline(
            input_dir=tmp_path,
            output_dir=None,
            title="Override",
            dx_um=0.25,
        )

    assert load_metadata.return_value.dx_um == 0.1  # original metadata untouched
    assert result.metadata.dx_um == 0.25
    assert result.metadata.dy_um == 0.25


def test_dx_um_override_rejects_non_positive_value(tmp_path):
    mocks = _stage1_mocks()
    with mocks[0], mocks[1], mocks[2], mocks[3], mocks[4], mocks[5], mocks[6], \
        mocks[7], mocks[8], mocks[9], mocks[10], mocks[11]:
        with pytest.raises(ValueError, match="--dx-um override"):
            run_pipeline(
                input_dir=tmp_path,
                output_dir=None,
                title="Override",
                dx_um=0.0,
            )


def test_dt_ms_override_rejects_nan(tmp_path):
    mocks = _stage1_mocks()
    with mocks[0], mocks[1], mocks[2], mocks[3], mocks[4], mocks[5], mocks[6], \
        mocks[7], mocks[8], mocks[9], mocks[10], mocks[11]:
        with pytest.raises(ValueError, match="--dt-ms override"):
            run_pipeline(
                input_dir=tmp_path,
                output_dir=None,
                title="Override",
                dt_ms=float("nan"),
            )
