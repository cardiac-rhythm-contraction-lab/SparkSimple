"""sparksimple.filtering.apply — Orchestrator for physiological spark filtering.

Directly ports sparkFiltSimple.m loop (lines 60–125).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sparksimple.filtering.models import FilteredSpark, FilteringResult
from sparksimple.filtering.rules import evaluate_candidate

if TYPE_CHECKING:
    from sparksimple.config import FilteringConfig
    from sparksimple.features.models import SparkFeature
    from sparksimple.models import ExperimentMetadata


def filter_sparks(
    candidates: list[SparkFeature],
    config: FilteringConfig,
    metadata: ExperimentMetadata | None = None,
    shape: tuple[int, ...] | None = None,
    margeouts: int = 0,
) -> FilteringResult:
    """Filter spark candidates against physiological acceptance criteria.

    Parameters
    ----------
    candidates : list[SparkFeature]
        List of spark feature records extracted in Phase 8.
    config : FilteringConfig
        Physiological thresholds configuration.
    metadata : ExperimentMetadata | None
        Acquisition metadata providing n_pixels_x (b0) and n_pixels_y (a0).
    shape : tuple[int, ...] | None
        Alternative to metadata: shape of image/volume (..., Y, X) or (Y, X).
    margeouts : int
        Margin pixels to exclude from FOV edges (MATLAB: line 56, default 0).

    Returns
    -------
    FilteringResult
        Structured filtering output containing annotated records, accepted/rejected
        lists, and rejection reason breakdown.
    """
    if metadata is not None:
        fov_x = metadata.n_pixels_x
        fov_y = metadata.n_pixels_y
    elif shape is not None:
        fov_x = shape[-1]
        fov_y = shape[-2] if len(shape) >= 2 else shape[0]
    else:
        # Fallback to max coordinates present in candidates if shape is unspecified
        fov_x = max((c.px for c in candidates), default=1)
        fov_y = max((c.py for c in candidates), default=1)

    records: list[FilteredSpark] = []
    for ii, cand in enumerate(candidates, start=1):
        good, fail = evaluate_candidate(
            candidate=cand,
            config=config,
            fov_x=fov_x,
            fov_y=fov_y,
            margeouts=margeouts,
        )
        record = FilteredSpark(
            id=ii,
            good=good,
            fail=fail,
            feature=cand,
        )
        records.append(record)

    return FilteringResult(records=records)
