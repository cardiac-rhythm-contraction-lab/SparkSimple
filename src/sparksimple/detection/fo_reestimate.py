"""sparksimple.detection.fo_reestimate — Post-detection Fo re-estimation module.

Ports lines 252–314 of ``mfiles/sparkDetect.m``:

1. ``should_reestimate_fo`` — checks MATLAB ``if (sum(size(forcedbaseline)) == 2)`` condition.
2. ``build_fo_smoothing_kernel_3d`` — builds 3D Gaussian-expanded moving average kernel B.
3. ``apply_border_fo_adjustment`` — performs border pixel Fo max propagation over local ra x ra neighborhood.
4. ``reestimate_fo`` — orchestrates 3D FFT smoothing, candidate voxel exclusion, pixelwise trace Fo estimation, border max propagation, and volume re-normalisation.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.ndimage import binary_erosion

from sparksimple.detection.convolution import conv3_matlab
from sparksimple.detection.kernel import build_gaussian_2d


@dataclass(frozen=True)
class FoReestimateResult:
    """Dataclass holding post-detection Fo re-estimation outputs.

    Attributes
    ----------
    fo_map : np.ndarray
        2D float64 array of shape ``(Y, X)`` representing the final spatial Fo baseline map.
    volume_renormalised : np.ndarray
        3D float64 array of shape ``(Y, X, T)`` representing the volume re-normalised
        with ``fo_map``.
    vaux_smoothed : np.ndarray
        3D float64 array of shape ``(Y, X, T)`` representing the raw volume smoothed with 3D kernel B.
    candidate_exclusion_mask : np.ndarray
        3D boolean array of shape ``(Y, X, T)`` marking voxels excluded from baseline calculation (V3 > 0).
    border_adjustment_mask : np.ndarray
        2D boolean array of shape ``(Y, X)`` marking cell mask boundary pixels where Fo was adjusted.
    kernel_3d : np.ndarray
        3D float64 smoothing kernel B of shape ``(n_spatial, n_spatial, suptemp)``.
    """

    fo_map: np.ndarray
    volume_renormalised: np.ndarray
    vaux_smoothed: np.ndarray
    candidate_exclusion_mask: np.ndarray
    border_adjustment_mask: np.ndarray
    kernel_3d: np.ndarray


def should_reestimate_fo(forced_baseline: float | np.ndarray | None) -> bool:
    """Determine whether post-detection Fo re-estimation should be performed.

    Mirrors MATLAB ``if (sum(size(forcedbaseline)) == 2)`` (sparkDetect.m line 254).

    In MATLAB:
    - Any scalar (0, 0.0, or positive scalar like 100.0) has ``size(scalar) == [1, 1]`` (sum = 2).
      Therefore scalar forced baselines and auto mode (0) execute re-estimation.
    - Only a 2D matrix of shape ``(Y, X)`` where ``Y + X != 2`` (or empty array) evaluates to False,
      skipping re-estimation.

    Parameters
    ----------
    forced_baseline : float | np.ndarray | None
        Forced baseline value or matrix.

    Returns
    -------
    bool
        True if re-estimation should be performed; False otherwise.
    """
    if forced_baseline is None:
        return True
    if isinstance(forced_baseline, (int, float, np.number)):
        return True
    if isinstance(forced_baseline, np.ndarray):
        if forced_baseline.ndim == 0 or forced_baseline.size == 1:
            return True
        if forced_baseline.ndim == 2 and (forced_baseline.shape[0] + forced_baseline.shape[1] == 2):
            return True
        return False
    return True


def build_fo_smoothing_kernel_3d(dx_um: float, dt_ms: float) -> np.ndarray:
    """Build the 3D spatiotemporal Gaussian moving average kernel B.

    MATLAB reference (sparkDetect.m lines 264–266)::

        suptemp = ceil(68/DT);
        B = gaussiana2d(ceil(2/DX));
        for ii = 2:suptemp, B(:,:,ii) = B(:,:,1); end
        B = B / sum(sum(sum(B)));

    Parameters
    ----------
    dx_um : float
        Spatial resolution in micrometers.
    dt_ms : float
        Temporal resolution in milliseconds.

    Returns
    -------
    np.ndarray
        3D float64 kernel of shape ``(n_spatial, n_spatial, suptemp)`` summing to 1.0.
    """
    n_spatial = int(np.ceil(2.0 / dx_um))
    suptemp = int(np.ceil(68.0 / dt_ms))

    b2d = build_gaussian_2d(n_spatial)
    b3d = np.zeros((n_spatial, n_spatial, suptemp), dtype=np.float64)
    for t in range(suptemp):
        b3d[:, :, t] = b2d

    b3d = b3d / np.sum(b3d)
    return b3d


def apply_border_fo_adjustment(
    fo_map: np.ndarray,
    mask: np.ndarray,
    dx_um: float,
    roi_r_um: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate local maximum Fo values along the cell mask border boundary.

    MATLAB reference (sparkDetect.m lines 287–302)::

        ra = round(roiR/DX); ra = ra - mod(ra, 2) + 1; ker = ones(ra, ra);
        aux = mk - imerode(mk, ker);
        gg = find(aux > 0); mkz = zeros(size(mk));
        Foo = Fo; tots = length(gg);
        for ii = 1:tots
            thismk = mkz;
            thismk(gg(ii)) = 1; thismk = conv2(thismk, ker, 'same');
            Foo(gg(ii)) = max(max(thismk .* Fo));
        end

    Parameters
    ----------
    fo_map : np.ndarray
        2D float64 spatial Fo map of shape ``(Y, X)``.
    mask : np.ndarray
        2D cell mask of shape ``(Y, X)`` (0/1 or bool).
    dx_um : float
        Spatial resolution in micrometers.
    roi_r_um : float
        ROI radius in micrometers (default 1.5).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        - ``foo``: 2D float64 array of shape ``(Y, X)`` after border max propagation.
        - ``border_mask``: 2D boolean array marking border boundary pixels.
    """
    ra = int(np.round(roi_r_um / dx_um))
    ra = ra - (ra % 2) + 1  # ensure odd integer
    ker = np.ones((ra, ra), dtype=bool)

    mask_bool = mask.astype(bool)
    eroded_mask = binary_erosion(mask_bool, structure=ker)
    border_mask = mask_bool & ~eroded_mask

    foo = fo_map.copy()
    rows, cols = fo_map.shape
    half_ra = ra // 2

    border_y, border_x = np.where(border_mask)
    for y, x in zip(border_y, border_x):
        r_min = max(0, y - half_ra)
        r_max = min(rows, y + half_ra + 1)
        c_min = max(0, x - half_ra)
        c_max = min(cols, x + half_ra + 1)

        foo[y, x] = float(np.max(fo_map[r_min:r_max, c_min:c_max]))

    return foo, border_mask


def reestimate_fo(
    volume_raw: np.ndarray,
    v3_thresholded: np.ndarray,
    mask: np.ndarray,
    estimated_fo: float,
    dx_um: float,
    dt_ms: float,
    roi_r_um: float = 1.5,
    baseline_mode: str = "auto",
    forced_baseline_value: float | np.ndarray | None = None,
    normalisation_mode: str = "df_f0",
) -> FoReestimateResult:
    """Re-estimate 2D spatial Fo baseline map and re-normalise raw volume.

    MATLAB reference (sparkDetect.m lines 252–314).

    Parameters
    ----------
    volume_raw : np.ndarray
        3D float raw volume of shape ``(Y, X, T)``. Must be supplied directly
        from ``PreprocessingResult.volume_raw``.
    v3_thresholded : np.ndarray
        3D float thresholded volume of shape ``(Y, X, T)`` from Phase 6A.
        Candidate points have values > 0.
    mask : np.ndarray
        2D binary cell mask of shape ``(Y, X)``.
    estimated_fo : float
        Global initial Fo baseline scalar from Phase 3.
    dx_um : float
        Spatial resolution in micrometers.
    dt_ms : float
        Temporal resolution in milliseconds.
    roi_r_um : float
        ROI radius in micrometers (default 1.5).
    baseline_mode : str
        Baseline estimation mode ("auto" or "forced").
    forced_baseline_value : float | np.ndarray | None
        Forced baseline value or 2D matrix if baseline_mode is "forced".
    normalisation_mode : str
        Normalisation mode ("df_f0" or "f_f0", default "df_f0").

    Returns
    -------
    FoReestimateResult
        Dataclass containing ``fo_map``, ``volume_renormalised``, ``vaux_smoothed``,
        ``candidate_exclusion_mask``, ``border_adjustment_mask``, and ``kernel_3d``.
    """
    # Check MATLAB sum(size(forcedbaseline)) == 2 branching semantics
    fb = forced_baseline_value if baseline_mode == "forced" else None
    if not should_reestimate_fo(fb):
        # 2D matrix forced baseline provided: skip re-estimation
        if isinstance(forced_baseline_value, np.ndarray) and forced_baseline_value.ndim == 2:
            fo_map = forced_baseline_value.astype(np.float64)
        else:
            fo_map = np.full(mask.shape, float(estimated_fo), dtype=np.float64)

        if normalisation_mode == "df_f0":
            vol_renorm = (volume_raw.astype(np.float64) - fo_map[:, :, np.newaxis]) / fo_map[:, :, np.newaxis]
        else:
            vol_renorm = volume_raw.astype(np.float64) / fo_map[:, :, np.newaxis]

        kernel_3d = build_fo_smoothing_kernel_3d(dx_um, dt_ms)
        vaux_smoothed = conv3_matlab(volume_raw, kernel_3d)

        return FoReestimateResult(
            fo_map=fo_map,
            volume_renormalised=vol_renorm,
            vaux_smoothed=vaux_smoothed,
            candidate_exclusion_mask=(v3_thresholded > 0.0),
            border_adjustment_mask=np.zeros(mask.shape, dtype=bool),
            kernel_3d=kernel_3d,
        )

    kernel_3d = build_fo_smoothing_kernel_3d(dx_um, dt_ms)
    vaux_smoothed = conv3_matlab(volume_raw, kernel_3d)

    candidate_exclusion_mask = (v3_thresholded > 0.0)
    vaux = vaux_smoothed.copy()
    vaux[candidate_exclusion_mask] = -1.0

    rows, cols, frames = volume_raw.shape
    fo_pixelwise = np.zeros((rows, cols), dtype=np.float64)

    for y in range(rows):
        for x in range(cols):
            trace = vaux[y, x, :]
            bg_trace = trace[trace != -1.0]
            if bg_trace.size == 0:
                fo_pixelwise[y, x] = float(estimated_fo)
            elif bg_trace.size == 1:
                fo_pixelwise[y, x] = float(bg_trace[0])
            else:
                # MATLAB: min(vec) + std(vec) * 0.5 (ddof=1)
                fo_pixelwise[y, x] = float(np.min(bg_trace) + np.std(bg_trace, ddof=1) * 0.5)

    fo_map, border_adjustment_mask = apply_border_fo_adjustment(
        fo_pixelwise, mask, dx_um, roi_r_um=roi_r_um
    )

    if normalisation_mode == "df_f0":
        vol_renorm = (volume_raw.astype(np.float64) - fo_map[:, :, np.newaxis]) / fo_map[:, :, np.newaxis]
    else:
        vol_renorm = volume_raw.astype(np.float64) / fo_map[:, :, np.newaxis]

    return FoReestimateResult(
        fo_map=fo_map,
        volume_renormalised=vol_renorm,
        vaux_smoothed=vaux_smoothed,
        candidate_exclusion_mask=candidate_exclusion_mask,
        border_adjustment_mask=border_adjustment_mask,
        kernel_3d=kernel_3d,
    )
