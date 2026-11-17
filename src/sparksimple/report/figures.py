"""sparksimple.report.figures — Matplotlib figure generation for the HTML report.

All figures use the Agg backend (no display required).
All figures use a white background with a light-theme colour palette.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import numpy as np
from pathlib import Path
from typing import Any
from PIL import Image, ImageDraw


_MISSING = object()

# Light-theme palette used across all figures
_CLR_TRACE   = "#334155"   # main signal line
_CLR_FAINT   = "#94a3b8"   # glow / secondary trace
_CLR_FIT     = "#dc2626"   # decay fit + RoR line
_CLR_SPAN    = "#0369a1"   # FDHM / t2p spans
_CLR_AMP     = "#16a34a"   # amplitude bracket
_CLR_MASS    = "#374151"   # mass annotation
_CLR_TEXT    = "#1e293b"   # axis labels / titles
_CLR_MUTED   = "#64748b"   # tick labels
_CLR_BORDER  = "#cbd5e1"   # spine / colorbar border


def _get(obj: Any, attr: str, default: Any = None) -> Any:
    """Get attribute from a spark or ROI record, falling back to .feature sub-object."""
    if isinstance(obj, dict):
        if attr in obj:
            return obj[attr]
        feature_obj = obj.get("feature")
        if isinstance(feature_obj, dict) and attr in feature_obj:
            return feature_obj[attr]
        if feature_obj is not None:
            v3 = getattr(feature_obj, attr, _MISSING)
            if v3 is not _MISSING:
                return v3
        return default

    v = getattr(obj, attr, _MISSING)
    if v is not _MISSING:
        return v
    feature = getattr(obj, "feature", None)
    if feature is not None:
        v2 = getattr(feature, attr, _MISSING)
        if v2 is not _MISSING:
            return v2
    return default


def _apply_white_style(fig: Any, ax: Any) -> None:
    """Apply white-background light theme to a figure/axes pair."""
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    ax.tick_params(colors=_CLR_MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(_CLR_BORDER)


def generate_blank(output_dir: str | Path) -> None:
    """Save a 1×1 white PNG as extraFigs/blank.png."""
    out = Path(output_dir) / "extraFigs"
    out.mkdir(parents=True, exist_ok=True)
    white = np.ones((1, 1, 3), dtype=np.uint8) * 255
    plt.imsave(str(out / "blank.png"), white)


def generate_location_sparks(
    accepted_sparks: list[Any],
    mask: np.ndarray | None,
    dx_um: float,
    output_dir: str | Path,
    volume: np.ndarray | None = None,
) -> None:
    """Cell temporal mean in the red channel with a white pixel at each spark position.

    Matches MATLAB: R = temporal-mean (normalised to [0,1]), G = B = 0 everywhere
    except at spark pixel positions where R = G = B = 255 (white dot).
    Output is exactly H×W pixels (same dimensions as the cell frame).
    """
    out = Path(output_dir) / "extraFigs"
    out.mkdir(parents=True, exist_ok=True)

    if mask is not None:
        H, W = mask.shape[:2]
    elif volume is not None:
        H, W = volume.shape[:2]
    else:
        H, W = 220, 512

    # Temporal mean of the volume, normalised to [0, 1]
    if volume is not None:
        ima = np.sum(volume, axis=2).astype(np.float64)
    elif mask is not None:
        ima = np.asarray(mask, dtype=np.float64) * 0.5
    else:
        ima = np.zeros((H, W), dtype=np.float64)

    ima_min, ima_max = float(ima.min()), float(ima.max())
    if ima_max > ima_min:
        ima = (ima - ima_min) / (ima_max - ima_min)
    else:
        ima = np.zeros_like(ima)

    # Black outside the cell mask
    if mask is not None:
        ima = ima * (np.asarray(mask) > 0)

    # RGB: temporal mean in red channel only
    r_ch = np.clip(ima * 255, 0, 255).astype(np.uint8)
    rgb = np.stack([r_ch, np.zeros_like(r_ch), np.zeros_like(r_ch)], axis=-1)

    # White single-pixel dot at each accepted spark position
    for s in accepted_sparks:
        x = int(_get(s, "px", 1)) - 1  # 0-based column
        y = int(_get(s, "py", 1)) - 1  # 0-based row
        if 0 <= y < H and 0 <= x < W:
            rgb[y, x] = [255, 255, 255]

    Image.fromarray(rgb, "RGB").save(str(out / "LocationSparks.png"))


def generate_location_sparks2(
    accepted_sparks: list[Any],
    mask: np.ndarray | None,
    output_dir: str | Path,
    volume: np.ndarray | None = None,
) -> None:
    """Green temporal mean with magenta spark-ID labels placed outside the cell.

    Matches MATLAB LocationSparks2:
      R = B = im2  (connector lines + spark ID text, 0 or 255)
      G = ima0     (temporal mean of volume, normalised to [0,255])
    For each accepted spark a ray is cast outward until it exits the cell mask;
    the label is placed just outside the boundary with a connector line.
    """
    from PIL import ImageFont
    out = Path(output_dir) / "extraFigs"
    out.mkdir(parents=True, exist_ok=True)

    if mask is not None:
        H, W = mask.shape[:2]
    elif volume is not None:
        H, W = volume.shape[:2]
    else:
        H, W = 220, 512

    # Temporal mean → g_ch (green background)
    if volume is not None:
        ima = np.sum(volume, axis=2).astype(np.float64)
    elif mask is not None:
        ima = np.asarray(mask, dtype=np.float64) * 0.5
    else:
        ima = np.zeros((H, W), dtype=np.float64)
    ima_min, ima_max = float(ima.min()), float(ima.max())
    if ima_max > ima_min:
        ima = (ima - ima_min) / (ima_max - ima_min)
    else:
        ima = np.zeros_like(ima)
    if mask is not None:
        ima = ima * (np.asarray(mask) > 0)
    g_ch = np.clip(ima * 255, 0, 255).astype(np.uint8)

    m_bool = np.asarray(mask, dtype=bool) if mask is not None else np.ones((H, W), dtype=bool)

    im2_img = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(im2_img)
    try:
        font = ImageFont.load_default(size=14)
    except Exception:
        font = None

    # 16 candidate ray directions
    N_ANGLES = 16
    MARGIN = 4   # pixels outside boundary where label is placed
    angles = np.linspace(0, 2 * np.pi, N_ANGLES + 1)[:-1]
    max_ray = int(np.hypot(H, W)) + 1

    placed: list[tuple[int, int, int, int]] = []  # (cx, cy, tw, th)

    for s in accepted_sparks:
        sid = int(_get(s, "id", 0) or 0)
        sx = int(_get(s, "px", 1) or 1) - 1  # 0-based col
        sy = int(_get(s, "py", 1) or 1) - 1  # 0-based row

        label = str(sid)
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad = max(tw, th) // 2 + 2

        # For each angle find where the ray exits the cell mask
        candidates: list[tuple[int, int, int, int, float]] = []  # (lx, ly, exit_step, ang_idx, score)
        for ai, ang in enumerate(angles):
            dx, dy = float(np.cos(ang)), float(np.sin(ang))
            exit_step = None
            for step in range(1, max_ray):
                rx = int(round(sx + dx * step))
                ry = int(round(sy + dy * step))
                if rx < 0 or rx >= W or ry < 0 or ry >= H:
                    exit_step = step
                    break
                if not m_bool[ry, rx]:
                    exit_step = step
                    break
            if exit_step is None:
                continue
            # Push far enough that the whole text box clears the boundary:
            # centre offset = exit_step + MARGIN + half-text-extent along ray
            text_half = abs(dx) * tw / 2 + abs(dy) * th / 2
            total = exit_step + MARGIN + text_half
            lx = int(round(sx + dx * total))
            ly = int(round(sy + dy * total))
            lx = max(pad, min(W - 1 - pad, lx))
            ly = max(pad, min(H - 1 - pad, ly))
            # Score: short exit (label close to cell) + no overlap
            overlap = any(
                abs(lx - px) < (tw + ptw) // 2 + 3
                and abs(ly - py) < (th + pth) // 2 + 3
                for px, py, ptw, pth in placed
            )
            score = -exit_step + (0 if not overlap else -1000)
            candidates.append((lx, ly, exit_step, ai, score))

        if candidates:
            candidates.sort(key=lambda c: c[4], reverse=True)
            lx, ly = candidates[0][0], candidates[0][1]
        else:
            # Fallback: place at fixed offset (spark may already be at border)
            lx = max(pad, min(W - 1 - pad, sx + MARGIN * 2))
            ly = max(pad, min(H - 1 - pad, sy))

        placed.append((lx, ly, tw, th))
        draw.line([(sx, sy), (lx, ly)], fill=255, width=1)
        draw.text((lx - tw // 2, ly - th // 2), label, fill=255, font=font)

    # Compose: R=im2, G=g_ch, B=im2
    im2 = np.array(im2_img)
    rgb = np.stack([im2, g_ch, im2], axis=-1)
    Image.fromarray(rgb, "RGB").save(str(out / "LocationSparks2.png"))


def generate_location_rois(
    rois: list[Any],
    mask: np.ndarray | None,
    dx_um: float,
    output_dir: str | Path,
    fo_map: np.ndarray | None = None,
    volume: np.ndarray | None = None,
    roi_r_um: float = 1.5,
) -> None:
    """Cell temporal mean as yellow background (R=G=ima, B=0) with white ROI squares.

    Matches MATLAB: temporal mean in R and G channels, B = 0.  A white square
    outline of side 2×(roiR/DX) pixels is drawn around each ROI centroid.
    A white scale bar is added in the corner.
    Output is exactly H×W pixels.
    """
    out = Path(output_dir) / "extraFigs"
    out.mkdir(parents=True, exist_ok=True)

    if mask is not None:
        H, W = mask.shape[:2]
    elif volume is not None:
        H, W = volume.shape[:2]
    elif fo_map is not None:
        H, W = fo_map.shape[:2]
    else:
        H, W = 220, 512

    # Temporal mean → ima, normalised [0,1]
    if volume is not None:
        ima = np.sum(volume, axis=2).astype(np.float64)
    elif fo_map is not None:
        ima = np.asarray(fo_map, dtype=np.float64)
    elif mask is not None:
        ima = np.asarray(mask, dtype=np.float64) * 0.5
    else:
        ima = np.zeros((H, W), dtype=np.float64)

    ima_min, ima_max = float(ima.min()), float(ima.max())
    if ima_max > ima_min:
        ima = (ima - ima_min) / (ima_max - ima_min)
    else:
        ima = np.zeros_like(ima)

    if mask is not None:
        ima = ima * (np.asarray(mask) > 0)

    # Yellow background: R=G=ima, B=0
    rg = np.clip(ima * 255, 0, 255).astype(np.uint8)
    b_ch = np.zeros_like(rg)
    rgb = np.stack([rg, rg, b_ch], axis=-1)

    # White square outlines at ROI positions (half-side = roiR/DX pixels)
    half = max(2, int(round(roi_r_um / max(dx_um, 1e-9))))
    for r in rois:
        x = int(r.px) - 1  # 0-based column
        y = int(r.py) - 1  # 0-based row
        x0, x1 = max(0, x - half), min(W - 1, x + half)
        y0, y1 = max(0, y - half), min(H - 1, y + half)
        # top and bottom edges
        if y0 < H:
            rgb[y0, x0:x1 + 1] = [255, 255, 255]
        if y1 < H:
            rgb[y1, x0:x1 + 1] = [255, 255, 255]
        # left and right edges
        if x0 < W:
            rgb[y0:y1 + 1, x0] = [255, 255, 255]
        if x1 < W:
            rgb[y0:y1 + 1, x1] = [255, 255, 255]

    # ROI ID text centred inside each square
    img = Image.fromarray(rgb, "RGB")
    draw = ImageDraw.Draw(img)
    try:
        from PIL import ImageFont as _IF
        font = _IF.load_default(size=11)
    except Exception:
        font = None
    for r in rois:
        x = int(r.px) - 1
        y = int(r.py) - 1
        x0, x1 = max(0, x - half), min(W - 1, x + half)
        y0, y1 = max(0, y - half), min(H - 1, y + half)
        label = str(int(r.id))
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx = (x0 + x1) // 2 - tw // 2
        ty = (y0 + y1) // 2 - th // 2
        draw.text((tx, ty), label, fill=(255, 255, 255), font=font)

    # Scale bar: white bar of ~5 µm in bottom-right corner
    if dx_um > 0:
        bar_um = 5.0
        bar_px = max(3, int(round(bar_um / dx_um)))
        bar_thick = max(1, H // 80)
        bar_margin = max(3, W // 40)
        bx0 = W - bar_margin - bar_px
        bx1 = W - bar_margin
        by0 = H - bar_margin - bar_thick
        by1 = H - bar_margin
        draw.rectangle([max(0, bx0), max(0, by0), min(W, bx1), min(H, by1)],
                       fill=(255, 255, 255))

    img.save(str(out / "LocationRois.png"))


_SUPERJET_STOPS = [
    (0.00, (0,   0, 143)),
    (0.15, (0,   0, 255)),
    (0.35, (0, 255, 255)),
    (0.55, (255, 255,   0)),
    (0.75, (255, 128,   0)),
    (1.00, (255,   0,   0)),
]


def _superjet_rgb(t: float) -> tuple[float, float, float]:
    """Return (R, G, B) in [0, 1] for t in [0, 1] via superjet colormap."""
    t = max(0.0, min(1.0, t))
    for i in range(len(_SUPERJET_STOPS) - 1):
        t0, c0 = _SUPERJET_STOPS[i]
        t1, c1 = _SUPERJET_STOPS[i + 1]
        if t <= t1:
            f = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            return tuple((c0[j] + f * (c1[j] - c0[j])) / 255.0 for j in range(3))  # type: ignore[return-value]
    return (1.0, 0.0, 0.0)


def generate_roi_heatmap(
    rois: list,
    fo_shape: "tuple[int, int]",
    roi_r_um: float,
    dx_um: float,
    mask: "np.ndarray | None",
    output_dir: "str | Path",
) -> None:
    """Save extraFigs/LocationRoisHeat.png matching MATLAB's output.

    Algorithm (mirrors clusterROIs.m lines 292-295):
      1. Sparse seed image: ROI centroid pixel = Nspk value.
      2. Convert to RGB via superjet (0 → black, v → color at v/max_nspk).
      3. Gaussian blur with σ = roi_r_um / dx_um pixels.
      4. Rescale: blurred × (pre-blur max / post-blur max).
      5. Apply cell mask (background → 0).
    """
    from scipy.ndimage import gaussian_filter
    from PIL import Image as _Image

    H, W = fo_shape
    nspk_vals = [int(getattr(r, "Nspk", 0) or 0) for r in rois]
    max_nspk = max(nspk_vals) if nspk_vals else 1

    # Step 1 — sparse seed
    seed = np.zeros((H, W), dtype=np.float64)
    for r in rois:
        py = int(getattr(r, "py", 0))
        px = int(getattr(r, "px", 0))
        nspk = int(getattr(r, "Nspk", 0) or 0)
        if 0 <= py < H and 0 <= px < W:
            seed[py, px] = max(seed[py, px], nspk)

    # Step 2 — indexed RGB via superjet (vectorised)
    rgb = np.zeros((H, W, 3), dtype=np.float64)
    nonzero = seed > 0
    if nonzero.any():
        t_map = seed / max_nspk  # [0, 1]
        for i in range(len(_SUPERJET_STOPS) - 1):
            t0, c0 = _SUPERJET_STOPS[i]
            t1, c1 = _SUPERJET_STOPS[i + 1]
            band = nonzero & (t_map >= t0) & (t_map <= t1)
            if not band.any():
                continue
            f = np.where(t1 > t0, (t_map - t0) / (t1 - t0), 0.0)
            for ch in range(3):
                rgb[:, :, ch] = np.where(
                    band,
                    (c0[ch] + f * (c1[ch] - c0[ch])) / 255.0,
                    rgb[:, :, ch],
                )

    # Step 3 & 4 — blur + rescale (MATLAB: im2 = blurred * gg / max(blurred))
    gg = float(rgb.max())
    sigma = roi_r_um / dx_um if dx_um > 0 else 5.0
    rgb = gaussian_filter(rgb, sigma=(sigma, sigma, 0))
    post_max = float(rgb.max())
    if post_max > 0 and gg > 0:
        rgb = rgb * (gg / post_max)

    # Step 5 — mask
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        rgb[~m] = 0.0

    out = Path(output_dir) / "extraFigs"
    out.mkdir(parents=True, exist_ok=True)
    img = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
    _Image.fromarray(img).save(str(out / "LocationRoisHeat.png"))


def generate_fo_map_raw(fo_map: np.ndarray, output_dir: str | Path) -> None:
    """Save a pixel-exact F0 map as extraFigs/Fo_raw.png (no axes or margins).

    The output is exactly H×W pixels in viridis colormap, suitable as a
    background for SVG overlays whose viewBox matches the array dimensions.
    """
    out = Path(output_dir) / "extraFigs"
    out.mkdir(parents=True, exist_ok=True)
    fo_arr = np.asarray(fo_map, dtype=np.float64)
    vmin, vmax = float(fo_arr.min()), float(fo_arr.max())
    norm = (fo_arr - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(fo_arr)
    plt.imsave(str(out / "Fo_raw.png"), norm, cmap="viridis")


def generate_fo_map(fo_map: np.ndarray, output_dir: str | Path) -> None:
    """Save the per-pixel F0 baseline map as extraFigs/Fo.png."""
    out = Path(output_dir) / "extraFigs"
    out.mkdir(parents=True, exist_ok=True)

    fo_arr = np.asarray(fo_map, dtype=np.float64)
    if fo_arr.ndim != 2:
        raise ValueError(f"fo_map must be 2D, got shape {fo_arr.shape}")

    fig, ax = plt.subplots(figsize=(6, 5))
    _apply_white_style(fig, ax)

    image = ax.imshow(fo_arr, cmap="viridis", origin="upper", aspect="equal")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=_CLR_MUTED, labelsize=8)
    cbar.outline.set_edgecolor(_CLR_BORDER)

    ax.set_title("F₀ map", fontsize=10, color=_CLR_TEXT, pad=4)
    ax.set_xlabel("X (pixels)", fontsize=9, color=_CLR_TEXT)
    ax.set_ylabel("Y (pixels)", fontsize=9, color=_CLR_TEXT)

    fig.tight_layout(pad=0.5)
    fig.savefig(str(out / "Fo.png"), dpi=100, facecolor="#ffffff")
    plt.close(fig)


def generate_mask_image(mask: np.ndarray, output_dir: str | Path) -> None:
    """Save the binary cell mask as a compact, pixel-exact PNG."""
    out = Path(output_dir) / "extraFigs"
    out.mkdir(parents=True, exist_ok=True)

    mask_arr = np.asarray(mask)
    if mask_arr.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {mask_arr.shape}")

    # The mask is already an image.  Rendering it through a 6×5 inch
    # Matplotlib figure adds large top/bottom margins for this very wide
    # (512×64 in the real data) aspect ratio.
    mask_img = (mask_arr.astype(bool).astype(np.uint8) * 255)
    Image.fromarray(mask_img, mode="L").save(str(out / "mask.png"))


def generate_spark_feature_plot(
    spark: Any,
    volume: np.ndarray | None,
    dx_um: float,
    dt_ms: float,
    output_dir: str | Path,
    max_val: float | None = None,
) -> None:
    """Plot the cropped MATLAB-style spark feature trace for one candidate."""
    out = Path(output_dir) / "spkFeat"
    out.mkdir(parents=True, exist_ok=True)

    spark_id = int(_get(spark, "id", 0))
    timesignal = _get(spark, "timesignal")
    timeinterval = _get(spark, "timeinterval")
    tau = float(_get(spark, "tau", np.nan) or np.nan)
    r2 = float(_get(spark, "r2", np.nan) or np.nan)
    AMP = float(_get(spark, "AMP", np.nan) or np.nan)
    BL = float(_get(spark, "BL", 0.0) or 0.0)
    amp = float(_get(spark, "amp", np.nan) or np.nan)
    ror = float(_get(spark, "ror", np.nan) or np.nan)
    t2p = float(_get(spark, "t2p", np.nan) or np.nan)
    fdhm = float(_get(spark, "fdhm", _get(spark, "FDHM", np.nan)) or np.nan)
    mass = float(_get(spark, "mass", np.nan) or np.nan)

    fig, ax = plt.subplots(figsize=(6, 9))
    _apply_white_style(fig, ax)

    if timesignal is not None and len(timesignal) > 0:
        signal = np.asarray(timesignal, dtype=float)
        if timeinterval is not None and len(timeinterval) > 0:
            interval = np.asarray(timeinterval, dtype=float)
            ot_0b = max(0, int(interval[0]) - 1)
            ft_0b = min(len(signal), int(interval[-1]))
            cut = signal[ot_0b:ft_0b]
            t_ms = interval[: len(cut)] * dt_ms
        else:
            cut = signal
            t_ms = (np.arange(len(cut), dtype=float) + 1.0) * dt_ms

        if len(cut) == 0:
            cut = signal
            t_ms = (np.arange(len(cut), dtype=float) + 1.0) * dt_ms

        # Use pt_refined (metrics-refined peak) when available; fall back to pt.
        pt_refined_val = int(_get(spark, "pt_refined", 0) or 0)
        pt_val = int(_get(spark, "pt", 1) or 1)
        use_pt = pt_refined_val if pt_refined_val > 0 else pt_val
        if timeinterval is not None and len(timeinterval) > 0:
            peak_matches = np.flatnonzero(
                np.asarray(timeinterval[: len(cut)], dtype=int) == use_pt
            )
            if len(peak_matches) > 0:
                i_peak = int(peak_matches[0])
            elif np.isfinite(cut).any():
                i_peak = int(np.nanargmax(cut))
            else:
                i_peak = 0
        else:
            i_peak = max(0, min(use_pt - 1, len(cut) - 1))
        i_peak = max(0, min(i_peak, len(cut) - 1))
        t_peak = float(t_ms[i_peak])

        ax.plot(t_ms, cut, color=_CLR_FAINT, linewidth=2.5, alpha=0.35)
        ax.plot(t_ms, cut, color=_CLR_TRACE, linewidth=1.0)

        decay_a = float(_get(spark, "decay_a", np.nan) or np.nan)
        decay_e = float(_get(spark, "decay_e", np.nan) or np.nan)
        decay_f = float(_get(spark, "decay_f", np.nan) or np.nan)
        if len(t_ms) > 1 and np.isfinite(tau) and tau > 0 and np.isfinite(amp):
            t_fit = t_ms[i_peak:]
            if len(t_fit) > 1:
                if np.isfinite(decay_a) and np.isfinite(decay_e) and np.isfinite(decay_f):
                    y_fit = decay_a * np.exp(-(t_fit - decay_e) / tau) + decay_f
                else:
                    y_fit = amp * np.exp(-(t_fit - t_peak) / tau) + BL
                ax.plot(t_fit, y_fit, color=_CLR_FIT, linewidth=1.4)

        xb = t_peak - t2p if np.isfinite(t2p) else np.nan
        if np.isfinite(ror) and ror != 0 and np.isfinite(xb) and np.isfinite(amp):
            y85 = BL + 0.85 * amp
            t85 = xb + (y85 - BL) / ror
            t_start = max(float(t_ms[0]), min(xb, t_peak))
            t_end = min(float(t_ms[-1]), max(t_start, min(t85, t_peak)))
            if t_end > t_start:
                t_uprise = np.linspace(t_start, t_end, 50)
                y_uprise = ror * (t_uprise - xb) + BL
                ax.plot(t_uprise, y_uprise, color="#ea580c", linewidth=1.4)

        x_min = float(t_ms[0])
        x_max = float(t_ms[-1])
        x_rng = max(x_max - x_min, dt_ms, 1.0)
        finite_cut = cut[np.isfinite(cut)]
        cut_max = float(np.max(finite_cut)) if len(finite_cut) > 0 else np.nan
        y_top_candidates = [
            cut_max,
            AMP if np.isfinite(AMP) else np.nan,
            BL + amp if np.isfinite(amp) else np.nan,
        ]
        finite_y_top = [v for v in y_top_candidates if np.isfinite(v)]
        y_top = max(finite_y_top) if finite_y_top else 1.0
        if max_val is not None and np.isfinite(max_val) and max_val > 0:
            y_top = float(max_val)
        y_top = max(y_top, 1.0)

        if np.isfinite(amp):
            x_bracket = x_min + 0.90 * x_rng
            tick = 0.018 * x_rng
            y0 = BL
            y1 = BL + amp
            ax.vlines(x_bracket, y0, y1, colors=_CLR_AMP, linewidth=1.4)
            ax.hlines(y0, x_bracket - tick, x_bracket + tick,
                      colors=_CLR_AMP, linewidth=1.4)
            ax.hlines(y1, x_bracket - tick, x_bracket + tick,
                      colors="#0d9488", linewidth=1.4)

        if np.isfinite(amp):
            half_level = BL + amp / 2.0
            left = None
            for idx in range(i_peak, -1, -1):
                if cut[idx] <= half_level:
                    left = float(t_ms[idx])
                    break
            right = None
            for idx in range(i_peak, len(cut)):
                if cut[idx] <= half_level:
                    right = float(t_ms[idx])
                    break
            if left is None and np.isfinite(fdhm):
                left = t_peak - fdhm / 2.0
            if right is None and np.isfinite(fdhm):
                right = (left + fdhm) if left is not None else t_peak + fdhm / 2.0
            if left is not None and right is not None and right > left:
                tick_y = 0.01 * y_top
                ax.hlines(half_level, left, right, colors=_CLR_SPAN, linewidth=1.4)
                ax.vlines([left, right], half_level - tick_y, half_level + tick_y,
                          colors=_CLR_SPAN, linewidth=1.1)

        if np.isfinite(xb):
            left = max(x_min, xb)
            right = min(x_max, t_peak)
            if right > left:
                tick_y = 0.01 * y_top
                ax.hlines(BL, left, right, colors="#7c3aed", linewidth=1.4)
                ax.vlines([left, right], BL - tick_y, BL + tick_y,
                          colors="#7c3aed", linewidth=1.1)

        annotations = [
            (f"tau = {tau:.2g} ms (R²= {r2:.3g})", _CLR_FIT),
            (f"ror = {ror:.3g} kHzF",              "#ea580c"),
            (f"BL = {BL:.3g}",                     _CLR_AMP),
            (f"amp = {amp:.3g}",                   "#0d9488"),
            (f"FDHM = {fdhm:.3g} ms",              _CLR_SPAN),
            (f"t2p = {t2p:.3g} ms",                "#7c3aed"),
            (f"mass = {mass:.3g} ΔF/F₀·frame",             "#92400e"),
        ]
        for idx, (text, color) in enumerate(annotations):
            ax.text(
                0.985,
                0.96 - idx * 0.085,
                text,
                ha="right",
                va="top",
                transform=ax.transAxes,
                color=color,
                fontsize=7,
            )

        ax.set_xlabel("Time (ms)", fontsize=9, color=_CLR_TEXT)
        ax.set_ylabel("ΔF/F₀", fontsize=9, color=_CLR_TEXT)
        ax.set_title(f"Spark {spark_id}", fontsize=10, color=_CLR_TEXT, pad=4)
        x_pad = 0.06 * x_rng
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        y_pad = 0.08 * y_top
        ax.set_ylim(-y_pad, y_top + y_pad)
    else:
        ax.text(0.5, 0.5, f"Spark {spark_id}\n(no signal data)",
                ha="center", va="center", transform=ax.transAxes, color=_CLR_MUTED)
        ax.set_title(f"Spark {spark_id}", fontsize=10, color=_CLR_TEXT, pad=4)

    fig.tight_layout(pad=0.5)
    fig.savefig(str(out / f"Spk{spark_id}.png"), dpi=100, facecolor="#ffffff")
    plt.close(fig)


def generate_spark_xt_overview(
    spark: Any,
    volume: np.ndarray | None,
    dx_um: float,
    dt_ms: float,
    output_dir: str | Path,
    save_uppercase: bool = False,
) -> None:
    """Spatial XT image centred on the spark with crosshair marker."""
    out = Path(output_dir) / "extraFigs"
    out.mkdir(parents=True, exist_ok=True)

    spark_id = int(_get(spark, "id", 0))
    fig, ax = plt.subplots(figsize=(5, 3))
    _apply_white_style(fig, ax)

    if volume is not None:
        py = int(_get(spark, "py", 1))
        px = int(_get(spark, "px", 1))
        pt = int(_get(spark, "pt", 1))
        H, W, T = volume.shape
        y_0b = max(0, min(py - 1, H - 1))
        xt_slice = volume[y_0b, :, :]  # (W, T)

        # ΔF/F₀ is centred around zero.  A sequential ``hot`` map maps the
        # negative baseline-dominated values to black, making real positive
        # events almost invisible.  Display positive excursions with a
        # robust upper limit while retaining the underlying data unchanged.
        xt_display = np.clip(np.asarray(xt_slice, dtype=float), 0.0, None)
        finite_positive = xt_display[np.isfinite(xt_display) & (xt_display > 0)]
        vmax = float(np.percentile(finite_positive, 98.5)) if finite_positive.size else 1.0
        vmax = max(vmax, 1e-6)

        ax.imshow(
            xt_display,
            aspect="auto",
            cmap="hot",
            origin="upper",
            norm=PowerNorm(gamma=0.7, vmin=0.0, vmax=vmax),
            extent=[0, T * dt_ms, W * dx_um, 0],
        )
        ax.axvline(x=(pt - 1) * dt_ms, color="#0369a1", linewidth=0.9, alpha=0.85)
        ax.axhline(y=(px - 1) * dx_um, color="#0369a1", linewidth=0.9, alpha=0.85)
        ax.set_xlabel("Time (ms)", fontsize=9, color=_CLR_TEXT)
        ax.set_ylabel("X (µm)", fontsize=9, color=_CLR_TEXT)
    else:
        ax.text(0.5, 0.5, f"Spark {spark_id}\n(volume not available)",
                ha="center", va="center", transform=ax.transAxes, color=_CLR_MUTED)

    ax.set_title(f"Spark {spark_id} — XT", fontsize=10, color=_CLR_TEXT, pad=4)

    fig.tight_layout(pad=0.5)
    fig.savefig(str(out / f"spk{spark_id}.png"), dpi=100, facecolor="#ffffff")
    if save_uppercase:
        fig.savefig(str(out / f"Spk{spark_id}.png"), dpi=100, facecolor="#ffffff")
    plt.close(fig)


def generate_spark_overview(
    spark: Any,
    volume: np.ndarray | None,
    dx_um: float,
    dt_ms: float,
    output_dir: str | Path,
) -> None:
    """Backward-compatible alias for accepted-spark XT overview figures."""
    generate_spark_xt_overview(
        spark=spark,
        volume=volume,
        dx_um=dx_um,
        dt_ms=dt_ms,
        output_dir=output_dir,
        save_uppercase=True,
    )


def generate_detection_film(
    volume: np.ndarray,
    accepted_sparks: list,
    rejected_sparks: list,
    output_dir: "str | Path",
    max_frames: "int | None" = None,
    brightness: float = 1.0,
    mask: "np.ndarray | None" = None,
    fo_map: "np.ndarray | None" = None,
) -> None:
    """Generate detFilm/F1{frame:05d}.png for each timeframe in volume.

    Matches MATLAB:
      - Cell background shown in the red channel (background masked to 0).
        MATLAB normalises the volume to [0,1] using the ~99th-pct as the
        clipping ceiling, then scales by the brightness factor and clips to
        1 before writing 0-255 (see clusterROIs.m); we replicate that by
        mapping the 99th-pct of cell pixels to R=255*brightness.
      - Each spark appears in frames pt-1, pt, pt+1 with a ±2-pixel crosshair:
          accepted → white (R=G=B=255)
          rejected → red   (R=255, G=B=0)
      - Spark ID text is written next to each marker in the same colour.
      - Frame number is printed in the top-left corner.
    Output: exact H×W RGB PNGs in output_dir/detFilm/.
    """
    from PIL import ImageFont
    out = Path(output_dir) / "detFilm"
    out.mkdir(parents=True, exist_ok=True)

    H, W, T = volume.shape
    n_frames = T if max_frames is None else min(max_frames, T)

    # Reconstruct raw-fluorescence-like volume (F0 × F/F0) so the film background
    # shows the real spatial structure of the cell, matching MATLAB's appearance.
    if fo_map is not None:
        fo2d = np.asarray(fo_map, dtype=np.float64)
        if fo2d.shape == (H, W):
            film_vol = fo2d[:, :, np.newaxis] * np.asarray(volume, dtype=np.float64)
        else:
            film_vol = np.asarray(volume, dtype=np.float64)
    else:
        film_vol = np.asarray(volume, dtype=np.float64)

    # Build mask boolean for background zeroing and cell-only normalisation
    if mask is not None:
        m_bool = np.asarray(mask, dtype=bool)
        cell_flat = film_vol[m_bool, :].ravel()
    else:
        m_bool = None
        cell_flat = film_vol.ravel()

    # Normalise: 99th pct of cell pixels → R=255*brightness (matches MATLAB's
    # volum=(volum-min)/(pct99-min); F=F*bf; F(F>1)=1; imwrite(uint8(F*255))).
    vmin = float(cell_flat.min()) if len(cell_flat) > 0 else float(film_vol.min())
    pct99_idx = max(0, int(np.ceil(len(cell_flat) * 0.99)) - 1)
    vmax = float(np.partition(cell_flat, pct99_idx)[pct99_idx])
    if vmax <= vmin:
        vmax = vmin + 1.0
    norm_scale = 255.0 * brightness / (vmax - vmin)

    def _pt(spark: Any) -> int:
        for attr in ("pt", "t", "tframe", "Tframe"):
            v = _get(spark, attr)
            if v is not None:
                return int(v)
        return 0

    # Build coords: (row, col, frame, spark_id, is_accepted) for frames pt-1..pt+1
    coords: list[tuple[int, int, int, int, bool]] = []
    for s in accepted_sparks:
        pt = _pt(s)
        sid = int(_get(s, "id", 0) or 0)
        row = int(_get(s, "py", 1) or 1) - 1
        col = int(_get(s, "px", 1) or 1) - 1
        for delta in (-1, 0, 1):
            t_mark = pt + delta
            if 1 <= t_mark <= n_frames:
                coords.append((row, col, t_mark, sid, True))
    for s in rejected_sparks:
        pt = _pt(s)
        sid = int(_get(s, "id", 0) or 0)
        row = int(_get(s, "py", 1) or 1) - 1
        col = int(_get(s, "px", 1) or 1) - 1
        for delta in (-1, 0, 1):
            t_mark = pt + delta
            if 1 <= t_mark <= n_frames:
                coords.append((row, col, t_mark, sid, False))

    # Group by frame
    by_frame: dict[int, list[tuple[int, int, int, bool]]] = {}
    for row, col, tf, sid, acc in coords:
        by_frame.setdefault(tf, []).append((row, col, sid, acc))

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for t in range(1, n_frames + 1):
        frame = film_vol[:, :, t - 1]
        if m_bool is not None:
            frame = np.where(m_bool, frame, 0.0)
        r_ch = np.clip((frame - vmin) * norm_scale, 0, 255).astype(np.uint8)
        rgb = np.stack([r_ch, np.zeros_like(r_ch), np.zeros_like(r_ch)], axis=-1)
        img = Image.fromarray(rgb, "RGB")
        draw = ImageDraw.Draw(img)

        for row, col, sid, acc in by_frame.get(t, []):
            color = (255, 255, 255) if acc else (255, 0, 0)
            # Hollow circle outline (radius 4 px) so the underlying spark is visible
            r = 4
            draw.ellipse([col - r, row - r, col + r, row + r],
                         outline=color, fill=None, width=1)
            # Spark ID text just below the circle
            draw.text((col - r, row + r + 1), str(sid), fill=color, font=font)

        # Frame number in top-left
        draw.text((1, 1), str(t), fill=(255, 255, 255), font=font)

        img.save(str(out / f"F1{t:05d}.png"))


def generate_roi_signal(
    roi: Any,
    all_sparks: list[Any],
    dx_um: float,
    dt_ms: float,
    roi_r_um: float,
    output_dir: str | Path,
    y_limits: tuple[float, float] | None = None,
) -> None:
    """Time trace of roi.signal with associated spark traces and peak markers."""
    out = Path(output_dir) / "roiFeat"
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(25, 2.5))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    signal = getattr(roi, "signal", None)
    if signal is not None and len(signal) > 0:
        signal_arr = np.asarray(signal, dtype=float)
        t_ms = np.arange(len(signal), dtype=float) * dt_ms
        ax.plot(t_ms, signal_arr, color=_CLR_TRACE, linewidth=1.0, zorder=3)

        associated_sparks: list[Any] = []
        radius_px = float(roi_r_um) / float(dx_um) if dx_um > 0 else 0.0
        roi_px = float(getattr(roi, "px", 0) or 0)
        roi_py = float(getattr(roi, "py", 0) or 0)
        roi_id = int(getattr(roi, "id", 0) or 0)

        for spark in all_sparks:
            spx = _get(spark, "px")
            spy = _get(spark, "py")
            spark_roi = _get(spark, "roi")

            distance_matches = False
            if spx is not None and spy is not None:
                distance = np.hypot(float(spx) - roi_px, float(spy) - roi_py)
                distance_matches = distance < radius_px

            roi_matches = False
            if spark_roi is not None:
                try:
                    roi_matches = int(spark_roi) == roi_id
                except (TypeError, ValueError):
                    roi_matches = False
            if distance_matches or roi_matches:
                associated_sparks.append(spark)

        plotted_arrays = [signal_arr]
        marker_rows: dict[int, int] = {}

        for spark in associated_sparks:
            spark_signal = _get(spark, "timesignal")
            if spark_signal is None or len(spark_signal) == 0:
                continue

            spark_arr = np.asarray(spark_signal, dtype=float)
            plotted_arrays.append(spark_arr)
            spark_t_ms = np.arange(len(spark_arr), dtype=float) * dt_ms

            pt_val = int(_get(spark, "pt", 1) or 1)
            peak_idx = max(0, min(pt_val - 1, len(spark_arr) - 1))
            peak_x = peak_idx * dt_ms
            if peak_x > len(signal_arr) * dt_ms:
                continue

            is_accepted = bool(_get(spark, "is_accepted", _get(spark, "good", True)))
            marker_color = "#16a34a" if is_accepted else "#dc2626"
            peak_y = float(spark_arr[peak_idx])
            if not np.isfinite(peak_y):
                continue
            row_key = int(round(peak_x / max(dt_ms, 1e-9)))
            row = marker_rows.get(row_key, 0)
            marker_rows[row_key] = row + 1

            ax.axvline(peak_x, color=marker_color, linewidth=0.75, alpha=0.95, zorder=4)
            ax.annotate(
                f"spk{int(_get(spark, 'id', 0) or 0)}",
                xy=(peak_x, peak_y),
                xytext=(0, 4 + row * 8),
                textcoords="offset points",
                ha="center",
                va="bottom",
                color=marker_color,
                fontsize=6,
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": marker_color,
                    "linewidth": 0.75,
                    "shrinkA": 0,
                    "shrinkB": 0,
                },
                zorder=5,
            )

        if y_limits is not None:
            y_min, y_max = y_limits
            if y_max <= y_min:
                y_max = y_min + 1.0
            y_pad = 0.08 * (y_max - y_min)
            ax.set_ylim(y_min - y_pad, y_max + y_pad)
        else:
            finite_arrays = [arr[np.isfinite(arr)] for arr in plotted_arrays]
            finite_arrays = [arr for arr in finite_arrays if len(arr) > 0]
            if finite_arrays:
                all_values = np.concatenate(finite_arrays)
                y_min = min(float(np.min(all_values)), 0.0)
                y_max = float(np.max(all_values))
                if y_max <= y_min:
                    y_max = y_min + 1.0
                y_pad = 0.08 * (y_max - y_min)
                ax.set_ylim(y_min - y_pad, y_max + y_pad)
        x_max_ms = len(signal_arr) * dt_ms
        x_pad = 0.03 * x_max_ms
        ax.set_xlim(-x_pad, x_max_ms + x_pad)
        ax.set_xticks(np.arange(0, x_max_ms + 1, 250))
        ax.grid(True, which="major", axis="both", color=_CLR_BORDER,
                linewidth=0.5, linestyle="-", alpha=0.7)
        ax.set_xlabel("Time (ms)", fontsize=8, color=_CLR_TEXT)
        ax.set_ylabel("ΔF/F₀", fontsize=8, color=_CLR_TEXT)
    else:
        ax.text(0.5, 0.5, f"ROI {roi.id} — no signal",
                ha="center", va="center", transform=ax.transAxes, color=_CLR_MUTED)

    ax.set_title(f"ROI {roi.id}", fontsize=9, color=_CLR_TEXT, pad=3)
    ax.tick_params(colors=_CLR_MUTED, labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor(_CLR_BORDER)

    fig.tight_layout(pad=0.3)
    fig.savefig(str(out / f"ROI{roi.id}.jpg"), dpi=100, facecolor="#ffffff")
    plt.close(fig)
