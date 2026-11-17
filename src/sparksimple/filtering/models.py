"""sparksimple.filtering.models — Dataclasses and reason constants for physiological filtering.

Ports output structures spkF(ii).good and spkF(ii).fail from sparkFiltSimple.m.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sparksimple.features.models import SparkFeature


class RejectionReason:
    """String constants for candidate rejection reasons, matching MATLAB sparkFiltSimple.m verbatim."""

    OUT_OF_FOV_X = "out of field of view (X)"
    OUT_OF_FOV_Y = "out of field of view (Y)"
    FAILED_PARAMETER_NAN = "failed parameter (NaN)"
    FWHM_UNMEASURABLE = "FWHM unmeasurable"
    PREVIOUS_BASELINE_TOO_LARGE = "previous baseline too large"
    RELATIVE_AMP_TOO_SMALL = "relative amp too small"
    DECAY_R2_TOO_SMALL = "decay r2 too small"
    T2P_TOO_SMALL = "t2p too small"
    T2P_TOO_LARGE = "t2p too large"
    TAU_DECAY_TOO_SMALL = "tau decay too small"
    TAU_DECAY_TOO_LARGE = "tau decay too large"
    FDHM_TOO_SMALL = "FDHM too small"
    FDHM_TOO_LARGE = "FDHM too large"
    ACCEPTED = "no"


@dataclass
class FilteredSpark:
    """A detected spark candidate annotated with physiological acceptance status.

    Attributes
    ----------
    id : int
        1-based candidate index (1..N), matching MATLAB ``spkF(ii).id = ii``.
    good : bool
        True if the candidate satisfies all physiological criteria (MATLAB: ``good = 1``),
        False if rejected (MATLAB: ``good = 0``).
    fail : str
        Rejection reason string matching MATLAB ``spkF(ii).fail``. Equal to ``"no"``
        when ``good`` is True.
    feature : SparkFeature
        The underlying candidate feature record extracted in Phase 8.
    """

    id: int
    good: bool
    fail: str
    feature: SparkFeature

    @property
    def is_accepted(self) -> bool:
        """Alias for self.good."""
        return self.good

    @property
    def is_rejected(self) -> bool:
        """Inverse of self.good."""
        return not self.good


@dataclass
class FilteringResult:
    """Output container for the Phase 9 physiological filtering pipeline.

    Attributes
    ----------
    records : list[FilteredSpark]
        Annotated records for all spark candidates in original candidate order.
    accepted : list[FilteredSpark]
        Sublist of accepted spark candidates (where ``good == True``).
    rejected : list[FilteredSpark]
        Sublist of rejected spark candidates (where ``good == False``).
    rejection_counts : dict[str, int]
        Mapping from fail reason string to candidate count.
    """

    records: list[FilteredSpark]
    accepted: list[FilteredSpark] = field(init=False)
    rejected: list[FilteredSpark] = field(init=False)
    rejection_counts: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        self.accepted = [r for r in self.records if r.good]
        self.rejected = [r for r in self.records if not r.good]
        counts = Counter(r.fail for r in self.records)
        self.rejection_counts = dict(counts)

    @property
    def total_count(self) -> int:
        """Total number of candidates evaluated."""
        return len(self.records)

    @property
    def accepted_count(self) -> int:
        """Number of accepted sparks."""
        return len(self.accepted)

    @property
    def rejected_count(self) -> int:
        """Number of rejected candidates."""
        return len(self.rejected)

    def summary(self) -> str:
        """Format a human-readable summary of filtering results."""
        lines = [
            f"FilteringResult: {self.accepted_count}/{self.total_count} accepted "
            f"({self.rejected_count} rejected)",
            "Rejection breakdown:",
        ]
        for reason, count in sorted(self.rejection_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  - {reason!r}: {count}")
        return "\n".join(lines)
