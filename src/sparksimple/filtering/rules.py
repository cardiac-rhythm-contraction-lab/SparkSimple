"""sparksimple.filtering.rules — Pure evaluation rules for physiological spark filtering.

Directly ports the 13-stage decision tree in sparkFiltSimple.m (lines 66–121).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from sparksimple.filtering.models import RejectionReason

if TYPE_CHECKING:
    from sparksimple.config import FilteringConfig
    from sparksimple.features.models import SparkFeature


def evaluate_candidate(
    candidate: SparkFeature,
    config: FilteringConfig,
    fov_x: int,
    fov_y: int,
    margeouts: int = 0,
) -> tuple[bool, str]:
    """Evaluate a single spark candidate against the 13-stage MATLAB decision tree.

    First-fail-wins policy. Preserves exact MATLAB evaluation order and error text.

    Parameters
    ----------
    candidate : SparkFeature
        Candidate feature record from Phase 8.
    config : FilteringConfig
        Physiological thresholds configuration.
    fov_x : int
        Field of view width in pixels (MATLAB: b0).
    fov_y : int
        Field of view height in pixels (MATLAB: a0).
    margeouts : int
        Margin pixels to exclude from FOV edges. Defaults to 0 (sparkFiltSimple.m line 56).

    Returns
    -------
    good : bool
        True if accepted, False if rejected.
    fail : str
        Rejection reason string ("no" if accepted).
    """
    # 1. FOV X boundary check (MATLAB line 66)
    # spkF(ii).px < b0 - margeouts + 1 && spkF(ii).px > margeouts
    if not (margeouts < candidate.px < fov_x - margeouts + 1):
        return False, RejectionReason.OUT_OF_FOV_X

    # 2. FOV Y boundary check (MATLAB line 67)
    # spkF(ii).py < a0 - margeouts + 1 && spkF(ii).py > margeouts
    if not (margeouts < candidate.py < fov_y - margeouts + 1):
        return False, RejectionReason.OUT_OF_FOV_Y

    # 3. NaN parameter check (MATLAB line 69)
    # Reject candidates with NaN in any core feature parameter.
    for _attr in ("amp", "tau", "ror", "t2p", "fdhm", "AMP", "BL"):
        if math.isnan(getattr(candidate, _attr, 0.0)):
            return False, RejectionReason.FAILED_PARAMETER_NAN

    # 4. FWHM measurable check (MATLAB line 70)
    # spkF(ii).FWHM > 0  — NaN fwhm is also treated as unmeasurable.
    if math.isnan(candidate.fwhm) or candidate.fwhm <= 0:
        return False, RejectionReason.FWHM_UNMEASURABLE

    # 5. Pre-spark baseline check (MATLAB line 71)
    # spkF(ii).BL <= Bl
    if config.max_baseline is not None and candidate.BL > config.max_baseline:
        return False, RejectionReason.PREVIOUS_BASELINE_TOO_LARGE

    # 6. Relative amplitude check (MATLAB line 72)
    # spkF(ii).amp >= iT
    if config.min_amplitude is not None and candidate.amp < config.min_amplitude:
        return False, RejectionReason.RELATIVE_AMP_TOO_SMALL

    # 7. Decay R² check (MATLAB line 73)
    # spkF(ii).r2 >= r2
    if config.min_decay_r2 is not None and (
        math.isnan(candidate.r2) or candidate.r2 < config.min_decay_r2
    ):
        return False, RejectionReason.DECAY_R2_TOO_SMALL

    # 8. Time-to-peak lower bound (MATLAB line 74)
    # spkF(ii).t2p >= t2(1)
    if config.t2p_range_ms is not None and candidate.t2p < config.t2p_range_ms[0]:
        return False, RejectionReason.T2P_TOO_SMALL

    # 9. Time-to-peak upper bound (MATLAB line 75)
    # spkF(ii).t2p <= t2(2)
    if config.t2p_range_ms is not None and candidate.t2p > config.t2p_range_ms[1]:
        return False, RejectionReason.T2P_TOO_LARGE

    # 10. Tau decay lower bound (MATLAB line 76)
    # spkF(ii).tau >= ta(1)
    if config.tau_range_ms is not None and candidate.tau < config.tau_range_ms[0]:
        return False, RejectionReason.TAU_DECAY_TOO_SMALL

    # 11. Tau decay upper bound (MATLAB line 77)
    # spkF(ii).tau <= ta(2)
    if config.tau_range_ms is not None and candidate.tau > config.tau_range_ms[1]:
        return False, RejectionReason.TAU_DECAY_TOO_LARGE

    # 12. FDHM lower bound (MATLAB line 78)
    # spkF(ii).FDHM >= Fd(1)
    if config.fdhm_range_ms is not None and candidate.fdhm < config.fdhm_range_ms[0]:
        return False, RejectionReason.FDHM_TOO_SMALL

    # 13. FDHM upper bound (MATLAB line 79)
    # spkF(ii).FDHM <= Fd(2)
    if config.fdhm_range_ms is not None and candidate.fdhm > config.fdhm_range_ms[1]:
        return False, RejectionReason.FDHM_TOO_LARGE

    # Passed all checks!
    return True, RejectionReason.ACCEPTED
