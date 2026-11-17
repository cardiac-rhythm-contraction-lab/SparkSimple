"""sparksimple.pipeline — Top-level pipeline orchestrator composing Phases 1 through 10."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import pickle
import re
import subprocess
from typing import Any

from sparksimple.config import PipelineConfig
from sparksimple.models import ExperimentMetadata
from sparksimple.io.metadata import load_metadata, validate_stack_against_metadata
from sparksimple.io.tiff_loader import (
    discover_channel_labels,
    discover_matched_tiff_files,
    load_image_stack,
)
from sparksimple.preprocessing.pipeline import preprocess, PreprocessingResult
from sparksimple.detection.wavelet import compute_v2, WaveletResult
from sparksimple.detection.merge import apply_3d_merge_and_threshold, MergeResult
from sparksimple.detection.local_max import detect_candidates, CandidateResult
from sparksimple.detection.fo_reestimate import reestimate_fo, FoReestimateResult
from sparksimple.features.extract import extract_features, FeatureExtractionResult
from sparksimple.filtering.apply import filter_sparks, FilteringResult
from sparksimple.roi.cluster import cluster_rois, ClusteringResult
from sparksimple.export.pipeline import run_phase10, Phase10Result


@dataclass
class PipelineRunResult:
    """Complete execution result of the SparkSimple2 pipeline."""

    metadata: ExperimentMetadata
    preprocessing: PreprocessingResult
    wavelet: WaveletResult
    merge: MergeResult
    candidates: CandidateResult
    fo_reestimate: FoReestimateResult
    features: FeatureExtractionResult
    filtering: FilteringResult
    clustering: ClusteringResult
    phase10: Phase10Result | None = None
    provenance: dict[str, Any] | None = None


@dataclass
class PipelineCheckpoint:
    """Restart data used by MATLAB-style `sp` resume points.

    This is intentionally a Python-native checkpoint rather than a MATLAB .mat
    compatibility file: it preserves the rich dataclass objects and arrays
    needed to resume the Python pipeline without recomputing earlier phases.
    """

    completed_stage: int = 0
    metadata: ExperimentMetadata | None = None
    preprocessing: PreprocessingResult | None = None
    wavelet: WaveletResult | None = None
    merge: MergeResult | None = None
    candidates: CandidateResult | None = None
    fo_reestimate: FoReestimateResult | None = None
    features: FeatureExtractionResult | None = None
    filtering: FilteringResult | None = None
    clustering: ClusteringResult | None = None
    phase10: Phase10Result | None = None
    selected_channel_index: int | None = None
    channel_labels: list[str] | None = None
    provenance: dict[str, Any] | None = None


CHECKPOINT_FILENAME = "spark_pipeline_checkpoint.pkl"


def _checkpoint_path(output_dir: str | Path | None) -> Path | None:
    if output_dir is None:
        return None
    return Path(output_dir) / CHECKPOINT_FILENAME


def _save_checkpoint(path: Path | None, checkpoint: PipelineCheckpoint) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(checkpoint, fh, protocol=pickle.HIGHEST_PROTOCOL)


def _load_checkpoint(path: Path | None, start_stage: int) -> PipelineCheckpoint:
    if path is None:
        raise ValueError(
            "output_dir is required when start_stage/sp is greater than 1, "
            "because resume data is stored in the results folder."
        )
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot resume from sp={start_stage}: checkpoint not found: {path}"
        )
    with path.open("rb") as fh:
        checkpoint = pickle.load(fh)
    if not isinstance(checkpoint, PipelineCheckpoint):
        raise TypeError(f"Invalid pipeline checkpoint file: {path}")
    if checkpoint.completed_stage < start_stage:
        raise ValueError(
            f"Cannot resume from sp={start_stage}: checkpoint only contains "
            f"data through stage {checkpoint.completed_stage}."
        )
    return checkpoint


def _require(value: Any, name: str, start_stage: int) -> Any:
    if value is None:
        raise ValueError(
            f"Cannot resume from sp={start_stage}: checkpoint is missing {name}."
        )
    return value


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _channel_index_from_request(
    channel: int | str | None, labels: list[str]
) -> int:
    """Recover the selected channel for legacy checkpoints lacking provenance."""
    if isinstance(channel, int):
        return channel
    if isinstance(channel, str):
        match = re.search(r"(\d+)", channel)
        if match:
            return int(match.group(1))
        raise ValueError(
            f"Cannot derive a numeric channel index from channel tag {channel!r}"
        )
    if len(labels) > 1:
        raise ValueError(
            "Legacy checkpoint has no selected-channel provenance and the input is "
            "multi-channel; resume with an explicit channel."
        )
    if len(labels) == 1:
        match = re.search(r"(\d+)", labels[0])
        if match:
            return int(match.group(1))
    return 0


def _execution_metadata(
    input_path: Path,
    cfg: PipelineConfig,
    meta: ExperimentMetadata,
    selected_channel: int,
    labels: list[str],
    candidates: CandidateResult,
    filtering: FilteringResult,
    clustering: ClusteringResult,
    dx_um_override: float | None,
    dt_ms_override: float | None,
) -> dict[str, Any]:
    selected_label = next(
        (
            label
            for label in labels
            if label.lower() == f"ch{selected_channel:02d}"
        ),
        f"ch{selected_channel:02d}",
    )
    return {
        "input_path": str(input_path.resolve()), "requested_channel": cfg.channel,
        "selected_channel_index": selected_channel, "selected_channel_label": selected_label,
        "available_channels": labels, "dx_um": meta.dx_um, "dy_um": meta.dy_um, "dt_ms": meta.dt_ms,
        "dx_um_cli_override": dx_um_override, "dt_ms_cli_override": dt_ms_override,
        "configuration": asdict(cfg), "NC": candidates.max_candidates,
        "candidate_count": candidates.count, "candidates_examined": filtering.total_count,
        "accepted_sparks": filtering.accepted_count, "rejected_candidates": filtering.rejected_count,
        "roi_count": len(clustering.rois), "candidate_limit_reached": candidates.candidate_limit_reached,
        "sparksimple_version": __import__("sparksimple").__version__, "git_commit": _git_commit(),
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def run_pipeline(
    input_dir: str | Path,
    output_dir: str | Path | None = None,
    cfg: PipelineConfig | None = None,
    title: str = "SparkSimple Run",
    is_simulated: bool | None = None,
    dx_um: float | None = None,
    dt_ms: float | None = None,
    progress_fn: Callable[[str], None] | None = None,
    start_stage: int = 1,
    save_checkpoint: bool = False,
) -> PipelineRunResult:
    """Execute the full SparkSimple2 detection and analysis pipeline (Phases 1-10).

    Composes:
    1-2. Metadata loading & TIFF stack I/O
    3-4. Preprocessing & cell mask segmentation
    5. CWT wavelet transform (V2 volume)
    6A. 3D merge kernel convolution & noise thresholding
    6B. 3D local maxima candidate detection (RegMax3D)
    7. Post-detection Fo baseline re-estimation
    8. Spark feature extraction
    9. Physiological decision-tree filtering
    10. Spatial ROI clustering & CSV/MAT export

    Parameters
    ----------
    input_dir : str | Path
        Directory containing input TIF stack files and metadata XML.
    output_dir : str | Path | None, default None
        Directory to export output CSV files and .mat files. If None, export is skipped.
    cfg : PipelineConfig | None, default None
        Pipeline configuration options. Uses standard defaults if None.
    title : str, default "SparkSimple Run"
        Title recorded in summary exports.
    is_simulated : bool | None, default None
        Override the simulated-experiment flag from metadata. When None, the flag
        is derived from the metadata source file name.
    dx_um : float | None, default None
        Override the spatial pixel size in µm from metadata.
    dt_ms : float | None, default None
        Override the frame interval in ms from metadata.
    progress_fn : Callable[[str], None] | None, default None
        Optional callback invoked with a progress message at the start of each phase.
    start_stage : int, default 1
        MATLAB-style resume point (`sp` in SparkSimple2.m):

        - 1: run everything from detection/preprocessing onward.
        - 2: resume after detection and run features, filtering, ROIs and export.
        - 3: resume after feature extraction and run filtering, ROIs and export.
        - 4: resume after filtering and run ROIs and export.
        - 5: resume after ROI clustering and redo export only.

        Resume points use a Python-native checkpoint stored as
        ``<output_dir>/spark_pipeline_checkpoint.pkl``.  The checkpoint is
        created automatically during normal `start_stage=1` runs.
    save_checkpoint : bool, default False
        When False (default), no checkpoint .pkl file is written to disk.
        Set to True only when using the resume functionality (start_stage > 1),
        which requires the checkpoint to exist from a prior run.

    Returns
    -------
    PipelineRunResult
        Container with results from all pipeline phases.
    """
    input_path = Path(input_dir)
    if cfg is None:
        cfg = PipelineConfig()
    if start_stage not in (1, 2, 3, 4, 5):
        raise ValueError(
            f"start_stage/sp must be one of 1, 2, 3, 4, 5; got {start_stage}"
        )

    def _progress(msg: str) -> None:
        if progress_fn is not None:
            progress_fn(msg)

    def _maybe_save(ckpt: PipelineCheckpoint) -> None:
        if save_checkpoint:
            _save_checkpoint(ckpt_path, ckpt)

    ckpt_path = _checkpoint_path(output_dir)
    if start_stage == 1:
        checkpoint = PipelineCheckpoint()
    else:
        checkpoint = _load_checkpoint(ckpt_path, start_stage)

    if start_stage <= 1:
        # 1-2. Load metadata & stack
        _progress("Phase 1: loading metadata...")
        meta = load_metadata(input_path)

        # Apply CLI overrides
        if dx_um is not None:
            dx_override = float(dx_um)
            if not math.isfinite(dx_override) or dx_override <= 0:
                raise ValueError(
                    f"--dx-um override must be finite and positive; got {dx_um!r}"
                )
            # SparkSimple uses one scalar spatial calibration: an explicit
            # override applies to both in-plane axes, never to X alone,
            # so the isotropy invariant enforced by ExperimentMetadata holds.
            meta = replace(meta, dx_um=dx_override, dy_um=dx_override)
        if dt_ms is not None:
            dt_override = float(dt_ms)
            if not math.isfinite(dt_override) or dt_override <= 0:
                raise ValueError(
                    f"--dt-ms override must be finite and positive; got {dt_ms!r}"
                )
            meta = replace(meta, dt_ms=dt_override)

        _progress(f"    Metadata source: {meta.source_xml}")
        _progress(f"    Frame size (Y x X): {meta.n_pixels_y} x {meta.n_pixels_x} px")
        _progress(f"    Frames (T): {meta.n_frames}")
        _progress(f"    Channels declared in XML: {meta.n_channels}")
        _progress(
            f"    Pixel size: dx={meta.dx_um:.4f} um, dy={meta.dy_um:.4f} um "
            f"| Frame interval: dt={meta.dt_ms:.4f} ms"
        )

        _progress("Phase 2: loading image stack...")
        channel_labels = discover_channel_labels(input_path)
        matched_files = discover_matched_tiff_files(input_path, cfg.channel)
        _progress(f"    Channel selector requested: {cfg.channel!r}")
        _progress(
            f"    Channel tags found in folder: "
            f"{', '.join(channel_labels) if channel_labels else '(none — single-channel filenames)'}"
        )
        _progress(f"    TIFF files matched by selector: {len(matched_files)}")
        if matched_files:
            _progress(f"        first: {matched_files[0].name}")
            if len(matched_files) > 1:
                _progress(f"        last:  {matched_files[-1].name}")

        stack, ch_idx = load_image_stack(
            input_path,
            channel=cfg.channel,
        )
        validate_stack_against_metadata(stack, meta)
        _progress(f"    Loaded stack shape (Y, X, T): {stack.shape}")
        _progress(f"    Selected channel index: {ch_idx}")

        # 3-4. Preprocessing & cell mask
        _progress("Phase 3-4: preprocessing and cell mask segmentation...")
        prep = preprocess(
            stack,
            cfg=cfg,
            selected_channel=ch_idx,
            dx_um=meta.dx_um,
        )

        # 5. CWT Wavelet (V2)
        _progress("Phase 5: wavelet transform (CWT)...")
        v2_res = compute_v2(
            volume_norm=prep.volume_norm,
            dt_ms=meta.dt_ms,
            is_simulated=meta.is_simulated if is_simulated is None else is_simulated,
            input_format="tiff",
            cfg=cfg,
        )

        # 6A. 3D Merge kernel convolution & noise thresholding
        _progress("Phase 6A: 3D merge and noise thresholding...")
        merge_res = apply_3d_merge_and_threshold(
            v2=v2_res.v2,
            mask=prep.mask,
            dx_um=meta.dx_um,
            dt_ms=meta.dt_ms,
            cfg=cfg.detection,
        )

        # 6B. RegMax3D candidate detection
        _progress("Phase 6B: local maxima detection (RegMax3D)...")
        cand_res = detect_candidates(
            v3_thresholded=merge_res.v3_thresholded,
            max_candidates=cfg.detection.noise_connectivity,
            temporal_split_frac=cfg.detection.temporal_split_frac,
            temporal_split_min_frames=cfg.detection.temporal_split_min_frames,
        )

        # 7. Post-detection Fo baseline re-estimation
        _progress("Phase 7: Fo baseline re-estimation...")
        fo_res = reestimate_fo(
            volume_raw=prep.volume_raw,
            v3_thresholded=merge_res.v3_thresholded,
            mask=prep.mask,
            estimated_fo=prep.fo,
            dx_um=meta.dx_um,
            dt_ms=meta.dt_ms,
            roi_r_um=cfg.roi.roi_r_um,
            baseline_mode=cfg.baseline_mode,
            forced_baseline_value=cfg.forced_baseline if cfg.baseline_mode == "forced" else None,
            normalisation_mode=cfg.detection.normalisation_mode,
        )

        checkpoint = PipelineCheckpoint(
            completed_stage=2,
            metadata=meta,
            preprocessing=prep,
            wavelet=v2_res,
            merge=merge_res,
            candidates=cand_res,
            fo_reestimate=fo_res,
            selected_channel_index=ch_idx,
            channel_labels=channel_labels,
        )
        _maybe_save(checkpoint)
    else:
        meta = _require(checkpoint.metadata, "metadata", start_stage)
        prep = _require(checkpoint.preprocessing, "preprocessing", start_stage)
        v2_res = _require(checkpoint.wavelet, "wavelet", start_stage)
        merge_res = _require(checkpoint.merge, "merge", start_stage)
        cand_res = _require(checkpoint.candidates, "candidates", start_stage)
        fo_res = _require(checkpoint.fo_reestimate, "fo_reestimate", start_stage)
        channel_labels = getattr(checkpoint, "channel_labels", None)
        if channel_labels is None:
            channel_labels = discover_channel_labels(input_path)
        ch_idx = getattr(checkpoint, "selected_channel_index", None)
        if ch_idx is None:
            ch_idx = _channel_index_from_request(cfg.channel, channel_labels)

    if start_stage <= 2:
        # 8. Feature extraction
        _progress("Phase 8: spark feature extraction...")
        feat_res = extract_features(
            volume=fo_res.volume_renormalised,
            ix_matrix=cand_res.ix_matrix,
            mask=prep.mask,
            dx_um=meta.dx_um,
            dt_ms=meta.dt_ms,
            roi_r_um=cfg.roi.roi_r_um,
        )
        checkpoint.features = feat_res
        checkpoint.completed_stage = max(checkpoint.completed_stage, 3)
        _maybe_save(checkpoint)
    else:
        feat_res = _require(checkpoint.features, "features", start_stage)

    if start_stage <= 3:
        # 9. Physiological filtering
        _progress("Phase 9: physiological filtering...")
        filt_res = filter_sparks(
            candidates=feat_res.features,
            config=cfg.filtering,
            shape=prep.mask.shape,
        )
        checkpoint.filtering = filt_res
        checkpoint.completed_stage = max(checkpoint.completed_stage, 4)
        _maybe_save(checkpoint)
    else:
        filt_res = _require(checkpoint.filtering, "filtering", start_stage)

    if start_stage <= 4:
        # 10. Spatial ROI clustering
        _progress("Phase 10: ROI clustering...")
        clust_res = cluster_rois(
            sparks=filt_res.records,
            volume=fo_res.volume_renormalised,
            mask=prep.mask,
            dx_um=meta.dx_um,
            dt_ms=meta.dt_ms,
            roi_r_um=cfg.roi.roi_r_um,
            linkage_method=cfg.roi.linkage,
        )
        checkpoint.clustering = clust_res
        checkpoint.completed_stage = max(checkpoint.completed_stage, 5)
        _maybe_save(checkpoint)
    else:
        clust_res = _require(checkpoint.clustering, "clustering", start_stage)

    # Optional export
    p10_res = None
    provenance = None
    if output_dir is not None:
        _progress("Phase 10: export...")
        provenance = _execution_metadata(
            input_path,
            cfg,
            meta,
            ch_idx,
            channel_labels,
            cand_res,
            filt_res,
            clust_res,
            dx_um,
            dt_ms,
        )
        p10_res = run_phase10(
            sparks=filt_res.records,
            volume=fo_res.volume_renormalised,
            mask=prep.mask,
            fo_map=fo_res.fo_map,
            dx_um=meta.dx_um,
            dt_ms=meta.dt_ms,
            roi_r_um=cfg.roi.roi_r_um,
            output_dir=str(output_dir),
            title=title,
            clustering=clust_res,
            execution_metadata=provenance,
        )
        checkpoint.phase10 = p10_res
        provenance_path = Path(output_dir) / "run_metadata.json"
        provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        p10_res.paths.execution_metadata = str(provenance_path)
        checkpoint.provenance = provenance
        _maybe_save(checkpoint)

    return PipelineRunResult(
        metadata=meta,
        preprocessing=prep,
        wavelet=v2_res,
        merge=merge_res,
        candidates=cand_res,
        fo_reestimate=fo_res,
        features=feat_res,
        filtering=filt_res,
        clustering=clust_res,
        phase10=p10_res,
        provenance=provenance,
    )
