#!/usr/bin/env python3
"""SparkSimple2.py — Python master script equivalent to MATLAB SparkSimple2.m.

Run from this folder with:

    python SparkSimple2.py

or, on a machine without a graphical file dialog:

    python SparkSimple2.py /path/to/experiment
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path


# Make the local port importable without requiring `pip install -e .`.
sys.path.insert(0, str(Path(__file__).parent / "src"))

from sparksimple.config import (  # noqa: E402
    DetectionConfig,
    FilteringConfig,
    PipelineConfig,
    ROIConfig,
)
from sparksimple.pipeline import run_pipeline  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════════
# ══ SparkSimple2 configuration
# ═══════════════════════════════════════════════════════════════════════════════

# Number of experiments to process.
# N = 0 means: select one root folder and process all its direct subfolders.
N = 1

# Ca²⁺ channel. Either a 0-indexed int (tag_channel=0 == MATLAB tagSpk='ch00'),
# or a MATLAB-style substring tag string (tag_channel="ch01") that pre-filters
# filenames exactly like tagSpk does in SparkSimple2.m — use this form if your
# channel filenames aren't consistently zero-padded (e.g. "ch1" vs "ch01").
tag_channel = "ch01"

# True for simulated data. This forces simulated timing behaviour in the pipeline.
simulated = False

# Name of the results subfolder created inside each experiment folder.
results_folder = "SparkSimple2_results"

# Skip an experiment if <experiment>/<results_folder>/zData2.mat already exists.
skip_processed = False

# ROI radius around each spark candidate centre, in micrometres.
roi_radius_um = 0.5

# Temporal filtering window, in frames.
filter_window_frames = 1

# Spatial merge distance for nearby spark candidates, in micrometres.
merge_distance_um = 1.5

# Temporal merge half-window for nearby spark candidates, in milliseconds.
merge_time_ms = 30.0

# Maximum number of spark candidates to keep searching for.
n_candidates = 100

# iT: minimum relative spark amplitude.
min_amplitude = 0.3

# Bl: maximum allowed pre-spark baseline.
max_baseline = 1.0

# Fd: allowed full-duration-at-half-maximum interval, in milliseconds.
fdhm_range_ms = (10, 200)

# ta: allowed decay tau interval, in milliseconds.
tau_range_ms = (5, 500)

# t2: allowed time-to-peak interval, in milliseconds.
t2p_range_ms = (0, 200)

# r2: minimum R² for the decay fit.
min_decay_r2 = 0.3

# Generate the HTML report (with spark/ROI figures) alongside the CSV/MAT exports.
generate_html_report = True

# Also generate a detection film (one PNG per frame). Implies generate_html_report.
generate_film = True

# Starting point:
#   1 = detection
#   2 = features
#   3 = filter
#   4 = rois
#   5 = export
#
# Resume points use a Python checkpoint stored in the results folder.
# Run once with sp=1 before resuming with sp=2..5.
sp = 1

# ═══════════════════════════════════════════════════════════════════════════════
# ══ End of configuration
# ═══════════════════════════════════════════════════════════════════════════════


LAST_SELECTION_FILE = Path(__file__).with_name(".sparksimple2_last_folder.json")


def _print(msg: str = "") -> None:
    """Print progress immediately, matching MATLAB's chatty style."""
    print(msg, flush=True)


def _load_last_folder() -> Path | None:
    """Load the last selected folder, replacing MATLAB's ultimpaquet.mat."""
    try:
        data = json.loads(LAST_SELECTION_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception:
        return None

    value = data.get("last_folder")
    if not value:
        return None

    path = Path(value)
    return path if path.exists() else None


def _save_last_folder(folder: Path) -> None:
    """Remember the last selected folder for the next interactive run."""
    payload = {"last_folder": str(folder)}
    LAST_SELECTION_FILE.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _select_folder(initial_dir: Path | None = None) -> Path | None:
    """Open a folder-selection dialog and return the chosen folder."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        _print("Could not load tkinter for the folder-selection dialog.")
        _print(f"Detail: {exc}")
        _print()
        _print("Use a path argument instead, for example:")
        _print("    python SparkSimple2.py /path/to/experiment")
        return None

    try:
        root = tk.Tk()
        root.withdraw()
        root.update()

        folder = filedialog.askdirectory(
            initialdir=str(initial_dir) if initial_dir else None,
            title="Select experiment folder",
            mustexist=True,
        )
    except Exception as exc:
        _print("Could not open the folder-selection dialog.")
        _print(f"Detail: {exc}")
        _print()
        _print("If this is a server without a display, use:")
        _print("    python SparkSimple2.py /path/to/experiment")
        return None
    finally:
        try:
            root.destroy()
        except Exception:
            pass

    if not folder:
        return None

    return Path(folder)


def _experiment_folders_from_dialog() -> list[Path]:
    """Collect experiment folders using MATLAB-like interactive selection."""
    routes: list[Path] = []
    last_folder = _load_last_folder()

    if N > 0:
        current_initial = last_folder

        for idx in range(N):
            selected = _select_folder(current_initial)
            if selected is None:
                break

            routes.append(selected)

            # MATLAB uses the parent of the selected folder as the next initial folder.
            current_initial = selected.parent
            _save_last_folder(selected)

            if N > 1:
                if idx == 0:
                    _print("Adding to processing pipeline:")
                _print(f"        {idx + 1}-. {selected}")

    else:
        root_folder = _select_folder(last_folder)
        if root_folder is None:
            return []

        _save_last_folder(root_folder)
        routes = sorted(
            child
            for child in root_folder.iterdir()
            if child.is_dir()
        )

    return routes


def _experiment_folders_from_args(args: list[str]) -> list[Path]:
    """Use command-line folders directly; useful on servers without a display."""
    routes = [Path(arg).expanduser() for arg in args]
    for route in routes:
        if not route.exists():
            raise FileNotFoundError(f"Experiment folder not found: {route}")
        if not route.is_dir():
            raise NotADirectoryError(f"Experiment path is not a folder: {route}")
    return routes


def _parse_tag_channel(value) -> int | str:
    """Accept tag_channel as either a 0-indexed int or a MATLAB tagSpk-style
    substring tag (e.g. "ch01"), matching load_image_stack()'s channel param."""
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return value
    return int(value)


def _build_config() -> PipelineConfig:
    """Build PipelineConfig from the MATLAB-style block above."""
    filtering = FilteringConfig.from_matlab_defaults()
    filtering.min_amplitude = float(min_amplitude)
    filtering.max_baseline = float(max_baseline)
    filtering.fdhm_range_ms = tuple(float(x) for x in fdhm_range_ms)
    filtering.tau_range_ms = tuple(float(x) for x in tau_range_ms)
    filtering.t2p_range_ms = tuple(float(x) for x in t2p_range_ms)
    filtering.min_decay_r2 = float(min_decay_r2)

    detection = DetectionConfig(
        filter_window_frames=int(filter_window_frames),
        merge_distance_um=float(merge_distance_um),
        merge_time_ms=float(merge_time_ms),
        noise_connectivity=int(n_candidates),
    )

    roi = ROIConfig(roi_radius_um=float(roi_radius_um))

    return PipelineConfig(
        detection=detection,
        filtering=filtering,
        roi=roi,
        channel=_parse_tag_channel(tag_channel),
    )


def _process_one_folder(folder: Path) -> bool:
    """Run one experiment folder. Return True when processing completed."""
    folder = folder.resolve()
    output_dir = folder / results_folder
    zdata2 = output_dir / "zData2.mat"

    _print(f"Starting with folder {folder}")

    if skip_processed and zdata2.exists():
        _print("         Folder already processed. Skipping reprocessing...")
        _print("Folder finished (skipped reprocessing).")
        _print()
        return True

    cfg = _build_config()

    result = run_pipeline(
        input_dir=folder,
        output_dir=output_dir,
        cfg=cfg,
        title=folder.name,
        is_simulated=simulated,
        progress_fn=_print,
        start_stage=int(sp),
    )

    do_report = generate_html_report or generate_film
    if do_report and result.phase10 is not None:
        from sparksimple.report.html import generate_report

        _print("Generating HTML report...")
        generate_report(
            result=result.phase10,
            output_dir=output_dir,
            title=folder.name,
            dx_um=result.metadata.dx_um,
            dt_ms=result.metadata.dt_ms,
            roi_r_um=cfg.roi.roi_r_um,
            volume=result.fo_reestimate.volume_renormalised,
            mask=result.preprocessing.mask,
            fo_map=result.fo_reestimate.fo_map,
            generate_film=bool(generate_film),
        )

    _print("Folder finished.")
    _print()
    return True


def main(argv: list[str] | None = None) -> int:
    """MATLAB-style SparkSimple2 master entry point."""
    if argv is None:
        argv = sys.argv[1:]

    if argv:
        try:
            routes = _experiment_folders_from_args(argv)
        except Exception as exc:
            _print(f"Error preparing command-line folders: {exc}")
            return 2
    else:
        routes = _experiment_folders_from_dialog()

    if not routes:
        _print("No experiment folders selected. Nothing to process.")
        return 0

    _print("SparkSimple2 Python")
    _print()

    completed = 0
    failed = 0

    for folder in routes:
        try:
            if _process_one_folder(folder):
                completed += 1
        except Exception as exc:
            failed += 1
            tb = traceback.extract_tb(exc.__traceback__)
            location = ""
            if tb:
                last = tb[-1]
                location = f" In {Path(last.filename).name} (line {last.lineno})"
            _print(f"Skipping due to error: {exc}{location}")
            _print()
            continue

    _print(f"Finished. Completed: {completed}. Failed: {failed}.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
