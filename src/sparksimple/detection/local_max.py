"""sparksimple.detection.local_max — 3D local maximum detection (RegMax3D port).

Ports MATLAB RegMax3D.m:
- Iterates across 51 threshold levels between min and max of input volume V3.
- Extracts connected components at each threshold using full 3D 26-connectivity.
- Identifies new local maximum objects (components not containing existing candidate marks).
- Computes centroid of new objects and records step level (rank) in candidate index matrix IX.
- Checks the candidate stopping target after each complete level (tope / nc).

SGT-R3c optional extension (disabled by default):
- A temporal valley suppression pre-pass is applied before the RegMax3D loop.
- When two sparks are temporally adjacent at the same (y,x), their V3 blobs can form a
  single 3D connected component, causing RegMax3D to register a single centroid between
  them instead of one per spark. The pre-pass zeroes shallow temporal valleys so that
  the two blobs remain separate components through the full threshold descent.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings
import numpy as np
import scipy.ndimage as ndi


def _matlab_round_pos(val: float) -> int:
    """MATLAB-faithful rounding for positive scalar numbers (floor(val + 0.5))."""
    return int(np.floor(val + 0.5))


def _find_temporal_maxima(profile: np.ndarray) -> np.ndarray:
    """Return 0-based indices of strict interior local maxima in a 1D array.

    A local maximum at index t satisfies profile[t] > profile[t-1] and
    profile[t] > profile[t+1]. Boundary frames (t=0 and t=nt-1) are excluded.

    Parameters
    ----------
    profile : np.ndarray
        1D array of length nT (temporal profile at one spatial position).

    Returns
    -------
    np.ndarray of int
        Indices of local maxima in ascending order.
    """
    if len(profile) < 3:
        return np.array([], dtype=np.intp)
    is_peak = (profile[1:-1] > profile[:-2]) & (profile[1:-1] > profile[2:])
    return np.where(is_peak)[0] + 1  # offset by 1: interior slice starts at t=1


def _temporal_valley_split(
    v3: np.ndarray,
    valley_depth_frac: float = 0.3,
    min_separation_frames: int = 6,
) -> np.ndarray:
    """Break temporal connectivity between adjacent spark blobs at the same spatial position.

    For each (y,x) spatial position with two or more local temporal maxima in the
    V3 profile, checks whether consecutive maxima are separated by a sufficiently
    deep valley. If the valley depth falls below valley_depth_frac * lesser_peak_value,
    the frames forming the valley bridge are zeroed so that the two blobs cannot
    become a single 3D connected component during RegMax3D thresholding.

    Only maxima that are at least min_separation_frames apart are considered.  Pairs
    closer than min_separation_frames are typically within the same merge-kernel temporal
    window (w3 = ceil(2*merge_time_ms / dt_ms)); their V3 profiles merge into a single
    monotone ramp with no valley to zero, so the guard prevents spurious near-neighbour
    splits driven by noise oscillations.

    Parameters
    ----------
    v3 : np.ndarray
        3D volume of shape (nY, nX, nT), non-negative float64.
    valley_depth_frac : float
        Minimum fractional depth required to keep two maxima as separate blobs.
        Depth is measured relative to the lesser of the two peak values.
        Default 0.3: the valley must drop by at least 30% of the smaller peak.
    min_separation_frames : int
        Minimum frame distance between two local maxima before the pair is
        considered for splitting. Pairs closer than this are ignored.
        Default 6: matches the default merge-kernel temporal width
        (ceil(2 * 30 ms / 10 ms/frame) = 6 frames).

    Returns
    -------
    np.ndarray
        Copy of v3 with shallow temporal valley frames zeroed.
    """
    ny, nx, nt = v3.shape
    v3_out = v3.copy()

    if nt < 3:
        return v3_out

    # Only iterate over (y,x) positions that carry any signal (V3 is sparse post-threshold)
    has_signal = np.any(v3 > 0, axis=2)
    nz_y, nz_x = np.nonzero(has_signal)

    for y, x in zip(nz_y, nz_x):
        profile = v3[y, x, :]  # read from original v3, not v3_out, to keep maxima stable
        maxima = _find_temporal_maxima(profile)

        if len(maxima) < 2:
            continue

        for i in range(len(maxima) - 1):
            t1 = int(maxima[i])
            t2 = int(maxima[i + 1])

            # Skip pairs that are too close in time — they lie within the merge-kernel
            # temporal window and their V3 profiles do not show a genuine valley.
            if t2 - t1 < min_separation_frames:
                continue

            lesser_peak = min(float(profile[t1]), float(profile[t2]))
            valley_segment = profile[t1 + 1:t2]
            valley_min = float(np.min(valley_segment))
            valley_depth = lesser_peak - valley_min

            if valley_depth < valley_depth_frac * lesser_peak:
                # Valley is too shallow: zero the bridge frames so the two blobs
                # remain separate connected components at all threshold levels.
                v3_out[y, x, t1 + 1:t2] = 0.0

    return v3_out


@dataclass(frozen=True)
class Candidate:
    """Represents a single 3D local maximum candidate spark location.

    Attributes
    ----------
    y_0b : int
        0-based row (Y) coordinate.
    x_0b : int
        0-based column (X) coordinate.
    t_0b : int
        0-based frame (T) coordinate.
    y_1b : int
        1-based Y coordinate for MATLAB compatibility.
    x_1b : int
        1-based X coordinate for MATLAB compatibility.
    t_1b : int
        1-based T coordinate for MATLAB compatibility.
    step_level : float
        Threshold step level rank assigned by RegMax3D (steps - ii + 1, e.g. 50..1).
    intensity : float | None
        Intensity of V3 at candidate location, if V3 was supplied.
    """

    y_0b: int
    x_0b: int
    t_0b: int
    y_1b: int
    x_1b: int
    t_1b: int
    step_level: float
    intensity: float | None = None


@dataclass
class CandidateResult:
    """Output container for Phase 6B candidate detection.

    Attributes
    ----------
    ix_matrix : np.ndarray
        3D float64 matrix (Y, X, T) matching MATLAB ``IX`` output from RegMax3D.
        Contains step_level values at candidate locations and 0.0 elsewhere.
    candidates : list[Candidate]
        List of candidate objects sorted by descending step_level (highest intensity rank first).
    count : int
        Total candidates, possibly exceeding the end-of-level stopping target.
    max_candidates : int
        Maximum candidate limit applied during detection (tope / nc).
    """

    ix_matrix: np.ndarray
    candidates: list[Candidate]
    count: int
    max_candidates: int
    candidate_limit_reached: bool = False


def regmax_3d(
    v3: np.ndarray,
    max_candidates: int = 50,
    steps: int = 51,
    temporal_split_frac: float = 0.0,
    temporal_split_min_frames: int = 6,
) -> np.ndarray:
    """3D local maximum detection matching MATLAB ``RegMax3D.m``.

    Extends the MATLAB port with an optional temporal valley suppression pre-pass
    (SGT-R3c fix): when two sparks at the same (y,x) are temporally adjacent, their
    V3 blobs would otherwise fuse into one 3D connected component, producing a single
    misplaced centroid. The pre-pass zeroes shallow valleys so each spark is detected
    independently. Set temporal_split_frac=0.0 to disable and restore strict MATLAB
    parity.

    Parameters
    ----------
    v3 : np.ndarray
        3D float64 volume of shape (nY, nX, nT). Typically post-threshold V3 volume.
    max_candidates : int
        Maximum number of candidates to detect (MATLAB ``tope`` / ``nc``). Default: 50.
    steps : int
        Number of threshold levels between min(v3) and max(v3). Default: 51.
    temporal_split_frac : float
        Fractional valley depth threshold for the temporal pre-pass. The valley between
        two temporal local maxima must drop by at least this fraction of the lesser peak
        to be considered naturally separate. Frames forming shallower valleys are zeroed.
        Default 0.0 (disabled). Set to 0.0 to disable the pre-pass entirely.
    temporal_split_min_frames : int
        Minimum frame separation between two temporal local maxima for the pair to be
        considered a split candidate. Pairs closer than this are skipped to avoid noise-
        induced spurious splits within a single spark's merge-kernel footprint.
        Default 6 (matches the default merge-kernel temporal width).

    Returns
    -------
    ix_matrix : np.ndarray
        3D float64 array of shape (nY, nX, nT). Contains step level ranks (steps - ii + 1)
        at detected local maxima locations and 0.0 elsewhere.
    """
    if v3.ndim != 3:
        raise ValueError(f"v3 volume must be 3-dimensional, got shape {v3.shape}")

    ny, nx, nt = v3.shape
    v3_double = v3.astype(np.float64)

    # SGT-R3c: temporal valley suppression pre-pass
    if temporal_split_frac > 0.0:
        v3_double = _temporal_valley_split(
            v3_double,
            valley_depth_frac=temporal_split_frac,
            min_separation_frames=temporal_split_min_frames,
        )

    im_min = float(np.min(v3_double))
    im_max = float(np.max(v3_double))

    rng = np.linspace(im_min, im_max, steps, dtype=np.float64)

    ip = np.zeros((ny, nx, nt), dtype=np.float64)
    contador = 0

    # 3D 26-connectivity structure (3x3x3 all ones)
    struct_26 = np.ones((3, 3, 3), dtype=bool)

    for ii in range(1, steps + 1):
        thresh = rng[steps - ii]  # 0-indexed: steps - ii maps to 51-1=50 down to 51-51=0

        bw = v3_double > thresh
        if not np.any(bw):
            continue

        labeled_array, numcells = ndi.label(bw, structure=struct_26)

        # MATLAB labels in column-major scan order (Y fastest).
        # Boxes conserve live IP reads: a centroid can lie outside its object.
        boxes = ndi.find_objects(labeled_array)
        # Native C-order labels are faster on the pipeline's (Y,X,T) arrays.
        # Recover MATLAB component order by each object's first Fortran pixel.
        ordered = []
        for jj, box in enumerate(boxes, 1):
            plane = labeled_array[box[0], box[1], box[2].start] == jj
            first_x = int(np.flatnonzero(np.any(plane, axis=0))[0])
            first_y = int(np.flatnonzero(plane[:, first_x])[0])
            first = (box[2].start, box[1].start+first_x, box[0].start+first_y)
            ordered.append((first, jj, box))
        for _, jj, box in sorted(ordered):
            component_mask = labeled_array[box] == jj
            ip_vals_in_comp = ip[box][component_mask]
            if len(np.unique(ip_vals_in_comp)) == 1:
                coords = np.nonzero(component_mask)
                aux = tuple(_matlab_round_pos(float(np.mean(c)) + sl.start)
                            for c, sl in zip(coords, box))
                contador += 1
                ip[aux] = float(steps - ii + 1)

        if contador >= max_candidates:
            break

    return ip


def extract_candidates(
    ix_matrix: np.ndarray,
    v3: np.ndarray | None = None,
) -> list[Candidate]:
    """Extract structured Candidate list from candidate index matrix IX.

    Parameters
    ----------
    ix_matrix : np.ndarray
        3D candidate matrix output from ``regmax_3d``.
    v3 : np.ndarray | None
        Optional reference volume V3 to record pixel intensity values at candidate sites.

    Returns
    -------
    candidates : list[Candidate]
        List of candidate objects sorted by descending step_level.
    """
    if ix_matrix.ndim != 3:
        raise ValueError(f"ix_matrix must be 3D, got shape {ix_matrix.shape}")

    y_indices, x_indices, t_indices = np.nonzero(ix_matrix > 0)
    if len(y_indices) == 0:
        return []

    candidates: list[Candidate] = []
    for y, x, t in zip(y_indices, x_indices, t_indices):
        step_lvl = float(ix_matrix[y, x, t])
        intensity_val = float(v3[y, x, t]) if v3 is not None else None

        cand = Candidate(
            y_0b=int(y),
            x_0b=int(x),
            t_0b=int(t),
            y_1b=int(y) + 1,
            x_1b=int(x) + 1,
            t_1b=int(t) + 1,
            step_level=step_lvl,
            intensity=intensity_val,
        )
        candidates.append(cand)

    # Descending step, then MATLAB linear index (Y fastest).
    candidates.sort(key=lambda c: (-c.step_level, c.t_0b, c.x_0b, c.y_0b))
    return candidates


def detect_candidates(
    v3_thresholded: np.ndarray,
    max_candidates: int = 50,
    steps: int = 51,
    temporal_split_frac: float = 0.0,
    temporal_split_min_frames: int = 6,
) -> CandidateResult:
    """Orchestrate Phase 6B candidate detection pipeline.

    Parameters
    ----------
    v3_thresholded : np.ndarray
        Post-threshold 3D volume V3 from Phase 6A.
    max_candidates : int
        Maximum number of candidates allowed (default: 50).
    steps : int
        Number of quantization steps for local maxima thresholding (default: 51).
    temporal_split_frac : float
        Valley depth fraction for the temporal pre-pass (SGT-R3c fix). Default: 0.3.
        Set to 0.0 to disable and reproduce strict MATLAB RegMax3D behaviour.
    temporal_split_min_frames : int
        Minimum frame separation for the temporal split pre-pass. Default: 6.

    Returns
    -------
    CandidateResult
        Dataclass containing ix_matrix, candidates list, count, and max_candidates.
    """
    ix_matrix = regmax_3d(
        v3_thresholded,
        max_candidates=max_candidates,
        steps=steps,
        temporal_split_frac=temporal_split_frac,
        temporal_split_min_frames=temporal_split_min_frames,
    )
    candidates = extract_candidates(ix_matrix, v3=v3_thresholded)
    candidate_limit_reached = len(candidates) >= max_candidates
    if candidate_limit_reached:
        warnings.warn(
            f"RegMax3D candidate limit reached: {len(candidates)} candidates >= NC={max_candidates}. "
            "Repeat with a larger NC before interpreting candidate or spark counts.",
            RuntimeWarning,
            stacklevel=2,
        )
    return CandidateResult(
        ix_matrix=ix_matrix,
        candidates=candidates,
        count=len(candidates),
        max_candidates=max_candidates,
        candidate_limit_reached=candidate_limit_reached,
    )
