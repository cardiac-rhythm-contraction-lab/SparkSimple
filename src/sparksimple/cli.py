"""sparksimple.cli — Command-line entry point for the SparkSimple2 analysis pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sparksimple import __version__
from sparksimple.io import load_image_stack, load_metadata
from sparksimple.pipeline import run_pipeline


def _channel_arg(value: str) -> int | str:
    """Parse --channel as an int index, or fall back to a MATLAB tagSpk-style tag."""
    try:
        return int(value)
    except ValueError:
        return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sparksimple",
        description="SparkSimple2: Ca2+ spark detection and analysis pipeline.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"sparksimple {__version__}",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Re-raise exceptions with full traceback instead of a terse error message.",
    )

    subparsers = parser.add_subparsers(dest="command")

    run_p = subparsers.add_parser(
        "run",
        help="Run the full SparkSimple2 analysis pipeline on an experiment folder.",
    )
    run_p.add_argument(
        "experiment_dir",
        help="Path to the experiment folder containing TIF files and metadata.",
    )
    run_p.add_argument(
        "--output-dir",
        metavar="PATH",
        default=None,
        help="Where to write results. Default: <experiment_dir>/Python_Result/",
    )
    run_p.add_argument(
        "--title",
        metavar="TEXT",
        default=None,
        help="Experiment title used for output filenames. Default: name of experiment_dir.",
    )
    run_p.add_argument(
        "--channel",
        type=_channel_arg,
        default=None,
        metavar="INT|TAG",
        help=(
            "Channel to load from multi-channel TIFs. Either a 0-indexed int "
            "(position among same-frame files) or a MATLAB-style substring "
            "tag such as 'ch01' that pre-filters filenames the way "
            "SparkSimple2.m's tagSpk does. Required for multi-channel experiments; "
            "use --channel ch01 for calcium when applicable."
        ),
    )
    run_p.add_argument(
        "--simulated",
        action="store_true",
        default=False,
        help="Mark simulated provenance (TIFF scales remain 5..10 frames).",
    )
    run_p.add_argument(
        "--dx-um",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Override spatial pixel size in µm (reads from metadata if omitted).",
    )
    run_p.add_argument(
        "--dt-ms",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Override frame rate in ms (reads from metadata if omitted).",
    )
    run_p.add_argument(
        "--report",
        action="store_true",
        default=False,
        help="Generate the HTML report alongside CSV/MAT outputs.",
    )
    run_p.add_argument(
        "--film",
        action="store_true",
        default=False,
        help="Generate detection film PNGs (one per frame). Implies --report.",
    )
    run_p.add_argument(
        "--roi-radius",
        type=float,
        default=1.5,
        metavar="FLOAT",
        help="ROI clustering radius in µm. Default: 1.5.",
    )
    run_p.add_argument(
        "--merge-distance-um",
        type=float,
        default=None,
        metavar="FLOAT",
        help=(
            "Phase 6A spatial merge distance in um (DetectionConfig.merge_distance_um). "
            "Default: 3.0 (matches DetectionConfig). The 3D merge kernel spans "
            "2x this distance in pixels; for narrow images (e.g. short line-scan Y "
            "extent) a value that produces a kernel larger than the image will raise "
            "an error -- lower this to fit, e.g. 1.0-1.5 for a 64-row image at "
            "dx=0.06um/px."
        ),
    )
    run_p.add_argument(
        "--spatial-smooth-um",
        type=float,
        default=None,
        metavar="FLOAT",
        help=(
            "Gaussian spatial smoothing kernel width in µm applied before the CWT "
            "(MATLAB fs parameter). Default: off. Requires --dx-um or metadata dx. "
            "Warning: non-zero values reduce amplitude measurements by ~2–8%% and "
            "inflate FWHM. Use the same value in all runs you intend to compare."
        ),
    )
    run_p.add_argument(
        "--filter-window-frames",
        type=int,
        default=1,
        metavar="INT",
        help=(
            "Temporal moving-average window in frames applied per-pixel before the CWT "
            "(MATLAB ft parameter). Default: 1 (off). Values >3 may reduce recall at "
            "high noise levels."
        ),
    )
    run_p.add_argument(
        "--nc", type=int, default=100, metavar="INT",
        help="RegMax3D candidate ceiling (NC). Default: 100; increase and rerun if the limit is reached.",
    )
    run_p.add_argument(
        "--roi-linkage", choices=("average", "complete", "single"), default="single",
        help="Spatial ROI linkage. Default: single (MATLAB-compatible; MATLAB's "
        "linkage() defaults to 'single' when called without a method argument).",
    )

    merge_p = subparsers.add_parser(
        "merge-only",
        help="Merge zData2.mat results from multiple experiments and re-cluster ROIs.",
    )
    merge_p.add_argument(
        "experiment_dirs",
        nargs="*",
        metavar="DIR",
        help="Experiment root folders. If omitted, a folder picker opens.",
    )

    validate_p = subparsers.add_parser(
        "validate",
        help="Open the browser-based manual spark validation tool.",
    )
    validate_p.add_argument(
        "--rf",
        default=None,
        metavar="TAG",
        help="Result subfolder tag. Default: validate_not_merge.RF.",
    )
    validate_p.add_argument(
        "--port",
        type=int,
        default=None,
        metavar="INT",
        help="Local Flask port. Default: 5555.",
    )
    validate_p.add_argument(
        "experiment_dir",
        nargs="?",
        default=None,
        help="Experiment folder. If omitted, a folder picker opens.",
    )

    modify_p = subparsers.add_parser(
        "modify-d2m",
        help="Open the browser UI to manually correct accepted spark dist2memb values.",
    )
    modify_p.add_argument(
        "result_dir",
        nargs="?",
        default=None,
        help="Results folder. If omitted, a folder picker opens.",
    )

    return parser


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _cmd_run(args: argparse.Namespace) -> None:
    from sparksimple.config import PipelineConfig, ROIConfig

    experiment_dir = Path(args.experiment_dir)
    if not experiment_dir.exists():
        raise FileNotFoundError(
            f"Experiment directory not found: {experiment_dir}"
        )

    output_dir = (
        Path(args.output_dir) if args.output_dir
        else experiment_dir / "Python_Result"
    )
    title = args.title or experiment_dir.name
    do_report = args.report or args.film
    generate_film = do_report

    # Phase 1 — validate and load metadata early for fast feedback
    _stderr("Phase 1: loading metadata...")
    load_metadata(experiment_dir)

    # Phase 2 — validate stack
    _stderr("Phase 2: loading image stack...")
    load_image_stack(experiment_dir, channel=args.channel)

    from sparksimple.config import DetectionConfig
    detection_kwargs = dict(
        spatial_smooth_um=args.spatial_smooth_um,
        filter_window_frames=args.filter_window_frames,
        noise_connectivity=args.nc,
    )
    if args.merge_distance_um is not None:
        detection_kwargs["merge_distance_um"] = args.merge_distance_um
    cfg = PipelineConfig(
        detection=DetectionConfig(**detection_kwargs),
        roi=ROIConfig(roi_radius_um=args.roi_radius, linkage=args.roi_linkage),
        channel=args.channel,
    )

    # Phases 3-10 — run the full pipeline (re-loads internally with overrides applied)
    result = run_pipeline(
        input_dir=experiment_dir,
        output_dir=output_dir,
        cfg=cfg,
        title=title,
        is_simulated=args.simulated,
        dx_um=args.dx_um,
        dt_ms=args.dt_ms,
        progress_fn=_stderr,
    )

    if do_report and result.phase10 is not None:
        from sparksimple.report.html import generate_report
        _stderr("Generating HTML report...")
        generate_report(
            result=result.phase10,
            output_dir=output_dir,
            title=title,
            dx_um=result.metadata.dx_um,
            dt_ms=result.metadata.dt_ms,
            roi_r_um=cfg.roi.roi_r_um,
            volume=result.fo_reestimate.volume_renormalised,
            mask=result.preprocessing.mask,
            fo_map=result.fo_reestimate.fo_map,
            generate_film=generate_film,
        )

    n_accepted = len(result.filtering.accepted)
    n_rejected = len(result.filtering.rejected)
    n_rois = len(result.clustering.rois)

    print(f"Accepted sparks : {n_accepted}")
    print(f"Rejected sparks : {n_rejected}")
    print(f"ROIs            : {n_rois}")
    print(f"Output          : {output_dir}")


def main() -> None:
    """Entry point registered in pyproject.toml."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    try:
        if args.command == "run":
            _cmd_run(args)
        elif args.command == "merge-only":
            from pathlib import Path as _Path
            from sparksimple.tools.merge_only import merge_only, RF, roi_r_um, output_folder_name

            dirs = [_Path(d) for d in args.experiment_dirs] if args.experiment_dirs else []
            if not dirs:
                from sparksimple.tools.merge_only import _select_folders, _load_last_folder, _save_last_folder
                dirs = _select_folders(initial_dir=_load_last_folder())
                if not dirs:
                    _stderr("No folders selected.")
                    sys.exit(0)
            _save_last_folder(dirs[0])
            merge_only(experiment_dirs=dirs, rf_tag=RF, roi_r_um_val=roi_r_um, output_folder=output_folder_name)
        elif args.command == "validate":
            from sparksimple.tools.validate_not_merge import PORT, RF, run_server

            run_server(
                Path(args.experiment_dir) if args.experiment_dir else None,
                rf_tag=args.rf or RF,
                port=args.port or PORT,
            )
        elif args.command == "modify-d2m":
            from sparksimple.tools.modify_dist2membrane import (
                run_modify_dist2membrane,
            )

            saved = run_modify_dist2membrane(args.result_dir)
            print(f"Saved updated distances to {saved}")
    except SystemExit:
        raise
    except Exception as exc:
        if getattr(args, "debug", False):
            raise
        _stderr(f"Error: {exc}")
        sys.exit(1)
