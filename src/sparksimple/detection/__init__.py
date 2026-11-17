"""sparksimple.detection — Detection package (wavelet detection, thresholding, candidate merging)."""

from sparksimple.detection.convolution import conv3_matlab
from sparksimple.detection.kernel import build_gaussian_2d, build_merge_kernel_3d
from sparksimple.detection.merge import MergeResult, apply_3d_merge_and_threshold
from sparksimple.detection.smooth import smooth_moving_average_1d, smooth_volume_time
from sparksimple.detection.threshold import (
    apply_threshold,
    estimate_noise_threshold,
    get_mask_center_pixel_coords,
)
from sparksimple.detection.wavelet import (
    WaveletResult,
    compute_v2,
    cwt_gaus2_1d,
    cwt_gaus2_volume,
    get_wavelet_scales,
)

from sparksimple.detection.local_max import (
    Candidate,
    CandidateResult,
    detect_candidates,
    extract_candidates,
    regmax_3d,
)
from sparksimple.detection.fo_reestimate import (
    FoReestimateResult,
    apply_border_fo_adjustment,
    build_fo_smoothing_kernel_3d,
    reestimate_fo,
    should_reestimate_fo,
)

__all__ = [
    "smooth_moving_average_1d",
    "smooth_volume_time",
    "get_wavelet_scales",
    "cwt_gaus2_1d",
    "cwt_gaus2_volume",
    "compute_v2",
    "WaveletResult",
    "build_gaussian_2d",
    "build_merge_kernel_3d",
    "conv3_matlab",
    "get_mask_center_pixel_coords",
    "estimate_noise_threshold",
    "apply_threshold",
    "MergeResult",
    "apply_3d_merge_and_threshold",
    "regmax_3d",
    "extract_candidates",
    "detect_candidates",
    "Candidate",
    "CandidateResult",
    "FoReestimateResult",
    "apply_border_fo_adjustment",
    "build_fo_smoothing_kernel_3d",
    "reestimate_fo",
    "should_reestimate_fo",
]
