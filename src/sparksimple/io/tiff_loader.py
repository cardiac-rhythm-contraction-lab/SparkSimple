"""sparksimple.io.tiff_loader — Image stack loader for SparkSimple2 datasets.

Reads a time-series of TIFF files from a folder and returns a 3-D NumPy array
with axes (rows, cols, frames) = (Y, X, T), matching the MATLAB convention:

    volum(a, b, ii)   →   stack[ii-1, :, :]   ← MATLAB 1-indexed
    MATLAB size(volum) = [a, b, temps]
    Python  stack.shape = (n_rows, n_cols, n_frames)   i.e. (a, b, temps)

Two naming conventions are supported, discovered from the test datasets:

    Real experiment     MVM_..._z100001_ch00.tif   (z-indexed, multi-channel)
    Simulated           Cabcnew0001_ch00.tif        (sequential, single-channel)

Channel selection:
    - If TIFs have only one channel, use it.
    - A ``channel`` argument always takes precedence.
    - ``channel`` accepts either a 0-indexed int (position among the files
      sharing a frame) or a str tag (e.g. ``"ch01"``).  A str tag mirrors
      MATLAB's ``tagSpk`` in SparkSimple2.m: it pre-filters the file listing
      to only names containing that substring *before* anything else runs,
      so it is immune to the index/ordering assumptions the int form makes.
    - Multi-channel experiments require an explicit channel. SparkSimple
      never infers the calcium channel from brightness.

MATLAB correspondence
---------------------
    sparkDetect.m lines 59–81  (volume building loop)
    sparkDetect.m lines 51–57  (channel selection)
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np


# ── Public API ────────────────────────────────────────────────────────────────

def load_image_stack(
    folder: str | Path,
    channel: int | str | None = None,
) -> tuple[np.ndarray, int]:
    """Load all TIFF frames from *folder* into a (rows, cols, frames) float32 array.

    Parameters
    ----------
    folder : str | Path
        Directory containing the TIFF time-series.
    channel : int | str | None
        0-indexed channel to extract, or a MATLAB-style substring tag (e.g.
        ``"ch01"``) that pre-filters filenames the way SparkSimple2.m's
        ``tagSpk`` does — only files whose name contains the tag are
        considered at all, before any grouping/indexing happens. ``None``
        (default) uses the single channel for single-channel datasets.  For
        multi-channel datasets require an explicit channel. In MATLAB's
        workflow the user pre-filters TIFFs to one channel.

    Returns
    -------
    stack : np.ndarray, shape (n_rows, n_cols, n_frames), dtype float32
        Volume array in (Y, X, T) order matching MATLAB ``volum(a, b, ii)``.
    selected_channel : int
        The 0-indexed channel that was loaded (useful when ``channel=None``).

    Raises
    ------
    FileNotFoundError
        If no TIFF files are found in *folder*.
    ValueError
        If frames have inconsistent dimensions, or requested channel is out of range.
    """
    try:
        import tifffile
    except ImportError as exc:
        raise ImportError(
            "tifffile is required for image loading. "
            "Install it with: pip install tifffile"
        ) from exc

    folder = Path(folder)
    tag = channel if isinstance(channel, str) else None
    tiff_paths = _discover_tiff_series(folder, tag=tag)

    if not tiff_paths:
        if tag is not None:
            raise FileNotFoundError(
                f"No TIFF files matching tag '{tag}' found in {folder}"
            )
        raise FileNotFoundError(f"No TIFF files found in {folder}")

    # ── Read the first frame to determine image geometry ──────────────────────
    first_frame = _read_tiff(tiff_paths[0], tifffile)

    # Single multi-page TIFF (e.g. calcium_stack.tif stacked with tifffile.imwrite):
    # tifffile.imread returns the full (n_pages, Y, X) volume, not one frame.
    # Detect this before the ndim dispatch below misinterprets n_pages as n_rows.
    if len(tiff_paths) == 1 and first_frame.ndim == 3:
        with tifffile.TiffFile(str(tiff_paths[0])) as _tif:
            _n_pages = len(_tif.pages)
        if _n_pages > 1:
            # Shape is (T, Y, X) — reorder to (Y, X, T) to match pipeline convention.
            stack = np.transpose(first_frame.astype(np.float32), (1, 2, 0))
            return stack, 0

    if first_frame.ndim == 2:
        n_rows, n_cols = first_frame.shape
        n_components_in_file = 1
    elif first_frame.ndim == 3:
        n_rows, n_cols, n_components_in_file = first_frame.shape
    else:
        raise ValueError(
            f"Unexpected TIF dimensions {first_frame.shape} in {tiff_paths[0]}"
        )

    # ── Identify unique frame indices and channel structure ───────────────────
    frame_groups = _group_by_frame(tiff_paths)
    n_frames = len(frame_groups)

    # When files are split by channel (e.g., _ch00, _ch01, _ch02),
    # each frame group contains one file per channel.
    n_channels_in_series = max(len(files) for files in frame_groups.values())

    if tag is not None:
        # MATLAB tagSpk-style pre-filter already narrowed tiff_paths to a
        # single logical channel above, so there is nothing left to select
        # among.  A frame group with >1 remaining file means the tag matched
        # more than one channel's files (e.g. tag="ch0" also matches "ch01") —
        # that is ambiguous, so fail loudly instead of silently guessing.
        if n_channels_in_series > 1:
            raise ValueError(
                f"Tag '{tag}' matched {n_channels_in_series} files for at "
                f"least one frame — the tag is ambiguous. Use a more "
                f"specific tag (e.g. 'ch00' instead of 'ch0')."
            )
        # Any remaining ndim==3 shape is an RGB/LUT colour wrapper, not
        # additional acquisition channels — handled per-frame in
        # _load_single_frame via non-zero-plane selection.
        n_ch_file = 1
        n_channels_total = 1
    elif n_channels_in_series > 1:
        # Files split by ``_chNN`` are logical channels even when each TIFF
        # carries an RGB wrapper.  The RGB planes are display payload, not
        # additional acquisition channels.
        n_ch_file = 1
        n_channels_total = n_channels_in_series
    elif n_components_in_file > 1:
        # Channels are packed inside each TIF (less common)
        n_ch_file = n_components_in_file
        n_channels_total = n_ch_file
    else:
        # Channels are in separate files (common in MVM datasets)
        n_ch_file = 1
        n_channels_total = n_channels_in_series

    # ── Channel selection — mirror MATLAB lines 51-57 of sparkDetect.m ────────
    # With a str tag, n_channels_total is already 1, so this always resolves
    # to index 0 — the tag itself was the selection, done via _discover_tiff_series.
    selected_channel = _select_channel(
        n_channels_total, None if tag is not None else channel
    )

    # ── Build volume ──────────────────────────────────────────────────────────
    stack = np.zeros((n_rows, n_cols, n_frames), dtype=np.float32)

    sorted_frames = sorted(frame_groups.keys())
    for t_idx, frame_key in enumerate(sorted_frames):
        files_in_frame = frame_groups[frame_key]
        img = _load_single_frame(files_in_frame, selected_channel, n_ch_file, tifffile)
        if img.shape != (n_rows, n_cols):
            raise ValueError(
                f"Frame {frame_key}: shape {img.shape} != expected ({n_rows}, {n_cols})"
            )
        stack[:, :, t_idx] = img.astype(np.float32)

    if tag is not None:
        # Report the numeric suffix of the matched tag (e.g. "ch01" -> 1) for
        # downstream logging; the internal list index is always 0 once the
        # file set has already been narrowed to a single logical channel.
        m = re.search(r"(\d+)", tag)
        return stack, (int(m.group(1)) if m else 0)

    return stack, selected_channel


# ── Internal helpers ──────────────────────────────────────────────────────────

def _discover_tiff_series(folder: Path, tag: str | None = None) -> list[Path]:
    """Return all TIFF paths in *folder*, sorted by name.

    When *tag* is given, mirrors MATLAB's ``dir([folder '*' tagSpk '*.tif'])``
    channel pre-filter from SparkSimple2.m: only files whose name contains
    *tag* (case-insensitive substring match) are returned — other channels'
    files are never even considered, same as MATLAB's ``tagSpk``.
    """
    patterns = ("*.tif", "*.tiff", "*.TIF", "*.TIFF")
    paths: list[Path] = []
    for pat in patterns:
        paths.extend(folder.glob(pat))
    # Leica LAS-AF folders also contain small ``*LUT.tif`` colour lookup
    # tables.  They are not temporal frames and must not enter the series.
    # Kept even when *tag* is set: unlike MATLAB, this is an explicit check
    # rather than relying on LUT filenames happening not to match the tag.
    paths = [path for path in paths if "lut" not in path.stem.lower()]
    if tag is not None:
        tag_lower = tag.lower()
        paths = [path for path in paths if tag_lower in path.stem.lower()]
    return sorted(set(paths))


def discover_matched_tiff_files(
    folder: str | Path, channel: int | str | None = None
) -> list[Path]:
    """Return the TIFF files ``load_image_stack`` would use for *channel*.

    Exposed for provenance/debug logging so callers can confirm a tag filter
    (e.g. ``"ch01"``) matched the intended files before the volume is built.
    """
    folder = Path(folder)
    tag = channel if isinstance(channel, str) else None
    return _discover_tiff_series(folder, tag=tag)


def discover_channel_labels(folder: str | Path) -> list[str]:
    """Return split-channel labels found in TIFF filenames for provenance."""
    labels: dict[int, str] = {}
    for path in _discover_tiff_series(Path(folder)):
        match = re.search(r"_ch(\d+)$", path.stem, re.IGNORECASE)
        if match:
            labels.setdefault(int(match.group(1)), f"ch{match.group(1)}")
    return [labels[key] for key in sorted(labels)]


def _group_by_frame(tiff_paths: list[Path]) -> dict[str, list[Path]]:
    """Group TIFF files by their frame index.

    Recognises two naming conventions from the reference datasets:

    - ``MVM_..._z100001_ch00.tif``  → frame key = "z100001"
    - ``Cabcnew0001_ch00.tif``       → frame key = "Cabcnew0001"
    - ``file_without_channel.tif``   → frame key = stem

    Files without a ``_ch##`` suffix are treated as a single-channel series
    where each file is one frame.
    """
    # Pattern: anything followed by _chNN at the end
    ch_pattern = re.compile(r"^(.+?)(?:_ch(\d+))?$", re.IGNORECASE)

    groups: dict[str, list[Path]] = {}
    for path in tiff_paths:
        stem = path.stem
        m = ch_pattern.match(stem)
        frame_key = m.group(1) if m else stem
        groups.setdefault(frame_key, []).append(path)

    # Sort files within each group by channel number
    for key in groups:
        groups[key] = sorted(groups[key])

    return groups


def _select_channel(
    n_channels_total: int,
    channel: int | None,
) -> int:
    """Return the 0-indexed channel to use.

    If *channel* is specified, validate and return it. A single-channel input
    resolves to channel zero. Multi-channel inputs without an explicit
    selection are rejected because brightness does not identify calcium.
    """
    if channel is not None:
        if channel >= n_channels_total:
            raise ValueError(
                f"Requested channel {channel} but dataset has "
                f"{n_channels_total} channel(s)."
            )
        return channel

    if n_channels_total == 1:
        return 0

    raise ValueError(
        f"Multi-channel dataset has {n_channels_total} channels; select an explicit channel "
        "(for example channel='ch01'). SparkSimple does not infer the calcium channel "
        "from brightness."
    )


def _load_single_frame(
    files_in_frame: list[Path],
    channel: int,
    n_ch_file: int,
    tifffile,
) -> np.ndarray:
    """Load one frame for the chosen *channel* and return a 2-D array."""
    if n_ch_file > 1:
        # All channels in one file
        img = _read_tiff(files_in_frame[0], tifffile)
        return img[:, :, channel]
    else:
        # One file per channel — files_in_frame[channel] is the right file
        if channel < len(files_in_frame):
            img = _read_tiff(files_in_frame[channel], tifffile)
        else:
            # Fallback: single file, single channel
            img = _read_tiff(files_in_frame[0], tifffile)
        if img.ndim == 3:
            # Leica's split-channel TIFFs may contain an RGB wrapper.  Select
            # the non-empty acquisition plane from the chosen logical file.
            plane = int(np.argmax(np.sum(img, axis=(0, 1))))
            return img[:, :, plane]
        return img


def _read_tiff(path: Path, tifffile) -> np.ndarray:
    """Read a TIFF, falling back to Pillow for LZW files without imagecodecs."""
    try:
        return tifffile.imread(str(path))
    except ValueError as exc:
        if "imagecodecs" not in str(exc).lower() and "lzw" not in str(exc).lower():
            raise
        try:
            from PIL import Image
        except ImportError:
            raise exc
        with Image.open(path) as image:
            return np.asarray(image)
