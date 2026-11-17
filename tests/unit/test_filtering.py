"""tests/unit/test_filtering.py — Unit tests for sparksimple.filtering.

Tests cover:
- Reason string constants vs MATLAB sparkFiltSimple.m
- 13-stage decision tree rule checks
- First-fail-wins evaluation priority
- NaN parameter handling
- Border & FOV coordinate checks
- FilteredSpark & FilteringResult dataclass behavior
- filter_sparks pipeline orchestrator
"""

import math
import numpy as np
import pytest

from sparksimple.config import FilteringConfig
from sparksimple.features.models import SparkFeature
from sparksimple.filtering import (
    FilteredSpark,
    FilteringResult,
    RejectionReason,
    evaluate_candidate,
    filter_sparks,
)


def make_dummy_feature(
    px: int = 10,
    py: int = 10,
    pt: int = 20,
    amp: float = 0.5,
    tau: float = 30.0,
    ror: float = 0.05,
    t2p: float = 15.0,
    fdhm: float = 25.0,
    AMP: float = 1.5,
    BL: float = 0.8,
    fwhm: float = 2.0,
    r2: float = 0.85,
) -> SparkFeature:
    """Helper factory for a valid candidate feature record."""
    return SparkFeature(
        px=px,
        py=py,
        pt=pt,
        x_0b=px - 1,
        y_0b=py - 1,
        t_0b=pt - 1,
        dist2memb=5.0,
        fwhm=fwhm,
        timesignal=np.zeros(50),
        timeinterval=np.arange(10, 20),
        r2=r2,
        amp=amp,
        tau=tau,
        ror=ror,
        t2p=t2p,
        fdhm=fdhm,
        AMP=AMP,
        BL=BL,
        mass=100.0,
    )


@pytest.fixture
def matlab_config() -> FilteringConfig:
    return FilteringConfig.from_matlab_defaults()


# ── Reason String Constants ───────────────────────────────────────────────────

class TestRejectionReason:

    def test_rejection_reasons_match_matlab_strings(self):
        assert RejectionReason.OUT_OF_FOV_X == "out of field of view (X)"
        assert RejectionReason.OUT_OF_FOV_Y == "out of field of view (Y)"
        assert RejectionReason.FAILED_PARAMETER_NAN == "failed parameter (NaN)"
        assert RejectionReason.FWHM_UNMEASURABLE == "FWHM unmeasurable"
        assert RejectionReason.PREVIOUS_BASELINE_TOO_LARGE == "previous baseline too large"
        assert RejectionReason.RELATIVE_AMP_TOO_SMALL == "relative amp too small"
        assert RejectionReason.DECAY_R2_TOO_SMALL == "decay r2 too small"
        assert RejectionReason.T2P_TOO_SMALL == "t2p too small"
        assert RejectionReason.T2P_TOO_LARGE == "t2p too large"
        assert RejectionReason.TAU_DECAY_TOO_SMALL == "tau decay too small"
        assert RejectionReason.TAU_DECAY_TOO_LARGE == "tau decay too large"
        assert RejectionReason.FDHM_TOO_SMALL == "FDHM too small"
        assert RejectionReason.FDHM_TOO_LARGE == "FDHM too large"
        assert RejectionReason.ACCEPTED == "no"


# ── Decision Tree Rules ───────────────────────────────────────────────────────

class TestEvaluateCandidate:

    def test_valid_candidate_accepted(self, matlab_config):
        cand = make_dummy_feature()
        good, fail = evaluate_candidate(cand, matlab_config, fov_x=100, fov_y=100)
        assert good is True
        assert fail == RejectionReason.ACCEPTED

    def test_fov_x_rejection(self, matlab_config):
        cand_low = make_dummy_feature(px=0)  # out of FOV X (< 1)
        good_low, fail_low = evaluate_candidate(cand_low, matlab_config, fov_x=100, fov_y=100)
        assert good_low is False
        assert fail_low == RejectionReason.OUT_OF_FOV_X

        cand_high = make_dummy_feature(px=101)  # out of FOV X (> 100)
        good_high, fail_high = evaluate_candidate(cand_high, matlab_config, fov_x=100, fov_y=100)
        assert good_high is False
        assert fail_high == RejectionReason.OUT_OF_FOV_X

    def test_fov_y_rejection(self, matlab_config):
        cand_low = make_dummy_feature(py=0)  # out of FOV Y (< 1)
        good_low, fail_low = evaluate_candidate(cand_low, matlab_config, fov_x=100, fov_y=100)
        assert good_low is False
        assert fail_low == RejectionReason.OUT_OF_FOV_Y

        cand_high = make_dummy_feature(py=101)  # out of FOV Y (> 100)
        good_high, fail_high = evaluate_candidate(cand_high, matlab_config, fov_x=100, fov_y=100)
        assert good_high is False
        assert fail_high == RejectionReason.OUT_OF_FOV_Y


    def test_nan_parameter_rejection(self, matlab_config):
        for field in ["amp", "tau", "ror", "t2p", "fdhm", "AMP", "BL"]:
            kw = {field: float("nan")}
            cand = make_dummy_feature(**kw)
            good, fail = evaluate_candidate(cand, matlab_config, fov_x=100, fov_y=100)
            assert good is False
            assert fail == RejectionReason.FAILED_PARAMETER_NAN, f"Failed for NaN {field}"

    def test_fwhm_unmeasurable_rejection(self, matlab_config):
        cand_zero = make_dummy_feature(fwhm=0.0)
        good_z, fail_z = evaluate_candidate(cand_zero, matlab_config, fov_x=100, fov_y=100)
        assert good_z is False
        assert fail_z == RejectionReason.FWHM_UNMEASURABLE

        cand_nan = make_dummy_feature(fwhm=float("nan"))
        good_n, fail_n = evaluate_candidate(cand_nan, matlab_config, fov_x=100, fov_y=100)
        assert good_n is False
        assert fail_n == RejectionReason.FWHM_UNMEASURABLE

    def test_previous_baseline_too_large(self, matlab_config):
        cand = make_dummy_feature(BL=1.5)  # > Bl=1.0
        good, fail = evaluate_candidate(cand, matlab_config, fov_x=100, fov_y=100)
        assert good is False
        assert fail == RejectionReason.PREVIOUS_BASELINE_TOO_LARGE

    def test_relative_amp_too_small(self, matlab_config):
        cand = make_dummy_feature(amp=0.1)  # < iT=0.3
        good, fail = evaluate_candidate(cand, matlab_config, fov_x=100, fov_y=100)
        assert good is False
        assert fail == RejectionReason.RELATIVE_AMP_TOO_SMALL

    def test_decay_r2_too_small(self, matlab_config):
        cand = make_dummy_feature(r2=0.1)  # < r2=0.3
        good, fail = evaluate_candidate(cand, matlab_config, fov_x=100, fov_y=100)
        assert good is False
        assert fail == RejectionReason.DECAY_R2_TOO_SMALL

        cand_nan = make_dummy_feature(r2=float("nan"))
        good_n, fail_n = evaluate_candidate(cand_nan, matlab_config, fov_x=100, fov_y=100)
        assert good_n is False
        assert fail_n == RejectionReason.DECAY_R2_TOO_SMALL

    def test_t2p_range_rejection(self, matlab_config):
        # t2 range is [0, 200]
        cand_low = make_dummy_feature(t2p=-1.0)
        good_l, fail_l = evaluate_candidate(cand_low, matlab_config, fov_x=100, fov_y=100)
        assert good_l is False
        assert fail_l == RejectionReason.T2P_TOO_SMALL

        cand_high = make_dummy_feature(t2p=250.0)
        good_h, fail_h = evaluate_candidate(cand_high, matlab_config, fov_x=100, fov_y=100)
        assert good_h is False
        assert fail_h == RejectionReason.T2P_TOO_LARGE

    def test_tau_range_rejection(self, matlab_config):
        # ta range is [5, 500]
        cand_low = make_dummy_feature(tau=2.0)
        good_l, fail_l = evaluate_candidate(cand_low, matlab_config, fov_x=100, fov_y=100)
        assert good_l is False
        assert fail_l == RejectionReason.TAU_DECAY_TOO_SMALL

        cand_high = make_dummy_feature(tau=600.0)
        good_h, fail_h = evaluate_candidate(cand_high, matlab_config, fov_x=100, fov_y=100)
        assert good_h is False
        assert fail_h == RejectionReason.TAU_DECAY_TOO_LARGE

    def test_fdhm_range_rejection(self, matlab_config):
        # Fd range is [10, 200]
        cand_low = make_dummy_feature(fdhm=5.0)
        good_l, fail_l = evaluate_candidate(cand_low, matlab_config, fov_x=100, fov_y=100)
        assert good_l is False
        assert fail_l == RejectionReason.FDHM_TOO_SMALL

        cand_high = make_dummy_feature(fdhm=250.0)
        good_h, fail_h = evaluate_candidate(cand_high, matlab_config, fov_x=100, fov_y=100)
        assert good_h is False
        assert fail_h == RejectionReason.FDHM_TOO_LARGE

    def test_first_fail_wins_order(self, matlab_config):
        """If multiple rules fail, earlier rule in decision tree wins."""
        # Fails both NaN check (rule 3) and baseline check (rule 5)
        cand = make_dummy_feature(amp=float("nan"), BL=2.0)
        good, fail = evaluate_candidate(cand, matlab_config, fov_x=100, fov_y=100)
        assert fail == RejectionReason.FAILED_PARAMETER_NAN

        # Fails both baseline (rule 5) and amp (rule 6)
        cand2 = make_dummy_feature(BL=2.0, amp=0.01)
        good2, fail2 = evaluate_candidate(cand2, matlab_config, fov_x=100, fov_y=100)
        assert fail2 == RejectionReason.PREVIOUS_BASELINE_TOO_LARGE


# ── Pipeline Orchestrator ──────────────────────────────────────────────────────

class TestFilterSparksOrchestrator:

    def test_filter_sparks_container_types(self, matlab_config):
        c1 = make_dummy_feature(px=10, py=10)  # valid
        c2 = make_dummy_feature(px=10, py=10, BL=2.0)  # invalid baseline
        candidates = [c1, c2]

        result = filter_sparks(candidates, matlab_config, shape=(100, 100))

        assert isinstance(result, FilteringResult)
        assert result.total_count == 2
        assert result.accepted_count == 1
        assert result.rejected_count == 1

        assert len(result.accepted) == 1
        assert len(result.rejected) == 1

        assert result.accepted[0].id == 1
        assert result.accepted[0].good is True
        assert result.accepted[0].fail == "no"

        assert result.rejected[0].id == 2
        assert result.rejected[0].good is False
        assert result.rejected[0].fail == RejectionReason.PREVIOUS_BASELINE_TOO_LARGE

        assert result.rejection_counts == {
            RejectionReason.ACCEPTED: 1,
            RejectionReason.PREVIOUS_BASELINE_TOO_LARGE: 1,
        }

    def test_empty_candidates_list(self, matlab_config):
        result = filter_sparks([], matlab_config, shape=(100, 100))
        assert result.total_count == 0
        assert result.accepted_count == 0
        assert result.rejected_count == 0
        assert result.rejection_counts == {}
