"""sparksimple.roi.cluster — Spatial ROI clustering using hierarchical linkage clustering.

Direct port of mfiles/clusterROIs.m.
MATLAB's ``linkage(pd)``, called without a method argument, defaults to
'single' (nearest-neighbour) linkage, not 'average'. Confirmed empirically:
'single' at cutoff=roi_r_um reproduces MATLAB's ROI partition exactly
(Adjusted Rand Index = 1.0) on real validation data; 'average' does not.
"""

from __future__ import annotations

from typing import Any
import warnings
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform

from sparksimple.roi.models import ROIRecord, ClusteringResult


def cluster_rois(
    sparks: list[Any],
    volume: np.ndarray,
    mask: np.ndarray,
    dx_um: float,
    dt_ms: float,
    roi_r_um: float,
    linkage_method: str = "single",
) -> ClusteringResult:
    """Group accepted sparks into spatial ROIs using hierarchical linkage clustering.

    Mirrors MATLAB mfiles/clusterROIs.m lines 54-351.

    Args:
        sparks: List of FilteredSpark, SparkFeature, or dict records containing spark features and good/accepted status.
        volume: 3D float64 volume of shape (Y, X, T).
        mask: 2D cell mask of shape (Y, X).
        dx_um: Spatial pixel resolution in um.
        dt_ms: Frame rate in ms.
        roi_r_um: ROI radius in um.
        linkage_method: Hierarchical linkage method used for the spatial ROI
            clustering: 'single' (MATLAB-compatible default), 'average' or
            'complete'. Any other value raises ValueError.

    Returns:
        ClusteringResult containing rois list and updated sparks list.
    """
    if linkage_method not in ("average", "complete", "single"):
        raise ValueError("linkage_method must be one of: average, complete, single")

    # 1. Identify indices of good (accepted) sparks
    good_indices = []
    for idx, s in enumerate(sparks):
        good = getattr(s, "good", None)
        if good is None:
            good = getattr(s, "is_accepted", True)
        if good:
            good_indices.append(idx)

    # Helper function to get spark attribute
    def get_attr(s: Any, attr: str) -> Any:
        if hasattr(s, attr):
            return getattr(s, attr)
        if hasattr(s, "feature") and hasattr(getattr(s, "feature"), attr):
            return getattr(getattr(s, "feature"), attr)
        if isinstance(s, dict):
            return s.get(attr)
        raise AttributeError(f"Attribute {attr} not found on spark record.")

    # Helper function to set spark attribute on both container and feature if present
    def set_attr(s: Any, attr: str, val: Any) -> None:
        if isinstance(s, dict):
            s[attr] = val
        else:
            setattr(s, attr, val)
            if hasattr(s, "feature") and getattr(s, "feature") is not None:
                setattr(getattr(s, "feature"), attr, val)

    n_good = len(good_indices)

    # Initialize default dist2spk and roi on all sparks
    for s in sparks:
        set_attr(s, "dist2spk", 0.0)
        set_attr(s, "roi", 0)

    if n_good == 0:
        return ClusteringResult(rois=[], sparks=sparks)
    if n_good == 1:
        # clusterROIs.m assigns roi=1 but exports an empty ROIs array.
        set_attr(sparks[good_indices[0]], "roi", 1)
        return ClusteringResult(rois=[], sparks=sparks)

    # Extract coords for good sparks: y, x (1-based)
    coords_px = np.zeros((n_good, 2), dtype=np.float64)
    for k, idx in enumerate(good_indices):
        s = sparks[idx]
        py = float(get_attr(s, "py"))
        px = float(get_attr(s, "px"))
        coords_px[k, 0] = py
        coords_px[k, 1] = px

    # Distance matrix in microns
    coords_um = coords_px * dx_um
    pd = pdist(coords_um)
    sq = squareform(pd) if n_good > 1 else np.zeros((1, 1), dtype=float)

    # Compute dist2spk for each good spark (nearest neighboring good spark)
    for k, idx in enumerate(good_indices):
        dists = sq[k, :]
        dists_nonzero = dists[dists > 0]
        if len(dists_nonzero) > 0:
            set_attr(sparks[idx], "dist2spk", float(np.min(dists_nonzero)))
        else:
            set_attr(sparks[idx], "dist2spk", 0.0)

    # Hierarchical clustering with Cutoff = roi_r_um.
    # MATLAB: h = linkage(pd) uses 'single' method by default (unspecified method arg).
    if n_good == 1:
        idx_clusters = np.array([1], dtype=int)
    else:
        h = linkage(pd, method=linkage_method)
        idx_clusters = fcluster(h, t=roi_r_um, criterion="distance")

    # Count sparks per cluster
    unique_clusters = np.unique(idx_clusters)
    counts = [int(np.sum(idx_clusters == c)) for c in unique_clusters]

    # Sort cluster IDs by spark count descending (stable sort preserving order)
    sorted_order = np.argsort(-np.array(counts), kind="stable")
    ordered_clusters = unique_clusters[sorted_order]

    rois: list[ROIRecord] = []
    coroi = []  # ROI centers [y, x] in 1-based px
    dst_px = int(np.floor(roi_r_um / dx_um + 0.5))
    a_height, b_width, t_temps = volume.shape

    for roi_id_1b, c_idx in enumerate(ordered_clusters, start=1):
        # find good sparks in this cluster
        spk_in_cluster_mask = idx_clusters == c_idx
        cluster_spark_indices = [good_indices[k] for k in np.where(spk_in_cluster_mask)[0]]

        # Assign roi ID to good sparks
        for idx in cluster_spark_indices:
            set_attr(sparks[idx], "roi", roi_id_1b)

        # ROI spatial centroid
        cluster_coords = np.array(
            [[get_attr(sparks[idx], "py"), get_attr(sparks[idx], "px")] for idx in cluster_spark_indices],
            dtype=np.float64,
        )
        mean_coords = np.floor(np.mean(cluster_coords, axis=0) + 0.5)
        y_center = int(mean_coords[0])
        x_center = int(mean_coords[1])
        coroi.append([y_center, x_center])

        # Extract ROI 1D signal
        oy = max(1, y_center - dst_px) - 1  # 0-based start
        fy = min(y_center + dst_px, a_height)  # 0-based end (exclusive)
        ox = max(1, x_center - dst_px) - 1
        fx = min(x_center + dst_px, b_width)

        vol_snippet = volume[oy:fy, ox:fx, :]  # shape (subY, subX, T)
        mask_snippet = mask[oy:fy, ox:fx]  # shape (subY, subX)

        # reshape snippet to (pixels, T) and exclude outside mask
        vol_flat = vol_snippet.reshape(-1, t_temps, order="F")
        mask_flat = mask_snippet.reshape(-1, order="F")
        vol_in_mask = vol_flat[mask_flat != 0, :]

        if vol_in_mask.shape[0] > 0:
            roi_signal = np.mean(vol_in_mask, axis=0)
        else:
            roi_signal = np.full(t_temps, np.nan)

        # Helper to safely retrieve spark metric value
        def get_metric(idx: int, *names: str) -> float:
            s = sparks[idx]
            for name in names:
                try:
                    val = get_attr(s, name)
                    if val is not None:
                        return float(val)
                except AttributeError:
                    continue
            return 0.0

        fwhm_vals = [get_metric(idx, "fwhm", "FWHM") for idx in cluster_spark_indices]
        r2_vals = [get_metric(idx, "r2") for idx in cluster_spark_indices]
        amp_vals = [get_metric(idx, "amp") for idx in cluster_spark_indices]
        AMP_vals = [get_metric(idx, "AMP") for idx in cluster_spark_indices]
        tau_vals = [get_metric(idx, "tau") for idx in cluster_spark_indices]
        ror_vals = [get_metric(idx, "ror") for idx in cluster_spark_indices]
        t2p_vals = [get_metric(idx, "t2p") for idx in cluster_spark_indices]
        fdhm_vals = [get_metric(idx, "fdhm", "FDHM") for idx in cluster_spark_indices]
        bl_vals = [get_metric(idx, "BL", "bl") for idx in cluster_spark_indices]
        d2m_vals = [get_metric(idx, "dist2memb") for idx in cluster_spark_indices]
        mass_vals = [get_metric(idx, "mass") for idx in cluster_spark_indices]

        n_spk = len(cluster_spark_indices)
        cluster_diameter_um = float(np.max(pdist(cluster_coords * dx_um))) if n_spk > 1 else 0.0
        if cluster_diameter_um > roi_r_um:
            warnings.warn(
                f"ROI {roi_id_1b} has maximum member diameter {cluster_diameter_um:.3g} µm "
                f"> clustering cutoff {roi_r_um:.3g} µm (linkage={linkage_method}).",
                RuntimeWarning,
                stacklevel=2,
            )
        roi_record = ROIRecord(
            id=roi_id_1b,
            px=x_center,
            py=y_center,
            Nsparks=n_spk,
            FWHM=float(np.mean(fwhm_vals)),
            r2=float(np.mean(r2_vals)),
            amp=float(np.mean(amp_vals)),
            tau=float(np.mean(tau_vals)),
            ror=float(np.mean(ror_vals)),
            t2p=float(np.mean(t2p_vals)),
            FDHM=float(np.mean(fdhm_vals)),
            AMP=float(np.mean(AMP_vals)),
            BL=float(np.mean(bl_vals)),
            dist2memb=float(np.mean(d2m_vals)),
            mass=float(np.mean(mass_vals)),
            Nspk=n_spk,
            dist2closest=0.0,  # updated below if N_rois > 1
            max_diameter_um=cluster_diameter_um,
            signal=roi_signal,
        )
        rois.append(roi_record)

    # Compute dist2closest for each ROI
    n_rois = len(rois)
    if n_rois > 1:
        coroi_arr = np.array(coroi, dtype=np.float64)  # (n_rois, 2) [y, x]
        pd_rois = squareform(pdist(coroi_arr))
        for i, r in enumerate(rois):
            dists = pd_rois[i, :]
            dists_nonzero = dists[dists > 0]
            if len(dists_nonzero) > 0:
                r.dist2closest = float(np.min(dists_nonzero) * dx_um)
            else:
                r.dist2closest = 0.0
    else:
        for r in rois:
            r.dist2closest = 0.0

    return ClusteringResult(rois=rois, sparks=sparks)
