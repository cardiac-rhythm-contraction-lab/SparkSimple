"""sparksimple.filtering — Physiological filtering of spark candidates.

MATLAB reference: sparkFiltSimple.m.
Rejection reasons: amplitude, baseline, tau, t2p, FDHM, FWHM, R², border.
"""

from sparksimple.filtering.models import (
    FilteredSpark,
    FilteringResult,
    RejectionReason,
)
from sparksimple.filtering.rules import evaluate_candidate
from sparksimple.filtering.apply import filter_sparks

__all__ = [
    "FilteredSpark",
    "FilteringResult",
    "RejectionReason",
    "evaluate_candidate",
    "filter_sparks",
]
