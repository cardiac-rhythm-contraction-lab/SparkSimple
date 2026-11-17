#!/usr/bin/env python3
"""SparkSimpleTools.py — Companion-tool launcher for SparkSimple2.

Python equivalent of the MATLAB companion tools:
  MergeOnly.m · ValidateNotMerge.m · ModifyDist2MembraneManual.m

Run from this folder with:

    python SparkSimpleTools.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


# ═══════════════════════════════════════════════════════════════════════════════
# ══ Tool selection
# ═══════════════════════════════════════════════════════════════════════════════
#
# Change `tool` to choose which companion tool to run:
#
#   "merge"     →  MergeOnly       (merge results from several experiments)
#   "validate"  →  ValidateNotMerge (browser-based manual spark-by-spark validation)
#   "dist2memb" →  ModifyDist2MembraneManual (manual distance-to-membrane correction)
#
tool = "merge"

# ═══════════════════════════════════════════════════════════════════════════════
# ══ MergeOnly — configuration
# ═══════════════════════════════════════════════════════════════════════════════

# Prefix (tag) of the results sub-folder inside each experiment folder.
RF = "iT_03_r2_03"

# ROI clustering radius in micrometres.
roi_r_um = 2.0

# Name of the output folder created inside the first selected experiment folder.
output_folder_name = "MERGED"

# ═══════════════════════════════════════════════════════════════════════════════
# ══ ValidateNotMerge — configuration
# ═══════════════════════════════════════════════════════════════════════════════

# Prefix of the results sub-folder to validate.
rf_validate = "iT_03_r2_03"

# Local Flask server port for the validation UI.
port_validate = 5555

# ═══════════════════════════════════════════════════════════════════════════════
# ══ ModifyDist2MembraneManual — configuration
# ═══════════════════════════════════════════════════════════════════════════════

# Local Flask server port for the distance-correction UI.
port_d2m = 5556

# ═══════════════════════════════════════════════════════════════════════════════
# ══ End of configuration
# ═══════════════════════════════════════════════════════════════════════════════


def _print(msg: str = "") -> None:
    print(msg, flush=True)


def _run_merge() -> None:
    from sparksimple.tools.merge_only import (
        _load_last_folder,
        _save_last_folder,
        _select_folders,
        merge_only,
    )

    _print("SparkSimpleTools — MergeOnly")
    _print()
    _print(f"Results tag : {RF}")
    _print(f"ROI radius  : {roi_r_um} um")
    _print(f"Output name : {output_folder_name}")
    _print()

    if len(sys.argv) > 1:
        dirs = [Path(p) for p in sys.argv[1:]]
        missing = [d for d in dirs if not d.exists()]
        if missing:
            for d in missing:
                _print(f"Error: folder not found: {d}")
            sys.exit(1)
    else:
        last = _load_last_folder()
        _print("Select the experiment folders to merge.")
        _print("(Cancel the dialog when all folders have been added)\n")
        dirs = _select_folders(initial_dir=last)
        if not dirs:
            _print("No folder selected. Exiting.")
            return

    _save_last_folder(dirs[0])
    _print(f"\nSelected folders ({len(dirs)}):")
    for d in dirs:
        _print(f"  {d}")
    _print()

    merge_only(
        experiment_dirs=dirs,
        rf_tag=RF,
        roi_r_um_val=roi_r_um,
        output_folder=output_folder_name,
    )


def _run_validate() -> None:
    from sparksimple.tools.validate_not_merge import run_server

    _print("SparkSimpleTools — ValidateNotMerge")
    _print()
    _print(f"Results tag : {rf_validate}")
    _print(f"Port        : {port_validate}")
    _print()

    folder: Path | None = None
    if len(sys.argv) > 1:
        folder = Path(sys.argv[1])
        if not folder.exists():
            _print(f"Error: folder not found: {folder}")
            sys.exit(1)

    run_server(folder, rf_tag=rf_validate, port=port_validate)


def _run_dist2memb() -> None:
    from sparksimple.tools.modify_dist2membrane import run_modify_dist2membrane

    _print("SparkSimpleTools — ModifyDist2MembraneManual")
    _print()
    _print(f"Port : {port_d2m}")
    _print()

    result_dir: str | None = None
    if len(sys.argv) > 1:
        result_dir = sys.argv[1]
        if not Path(result_dir).exists():
            _print(f"Error: folder not found: {result_dir}")
            sys.exit(1)

    import sparksimple.tools.modify_dist2membrane as _m2m
    _m2m.PORT = port_d2m

    saved = run_modify_dist2membrane(result_dir)
    _print(f"\nSaved to: {saved}")


def main() -> None:
    _TOOLS = {
        "merge":     _run_merge,
        "validate":  _run_validate,
        "dist2memb": _run_dist2memb,
    }

    runner = _TOOLS.get(tool)
    if runner is None:
        _print(f'Error: unknown tool: "{tool}"')
        _print(f'Valid options: {", ".join(_TOOLS)}')
        sys.exit(1)

    try:
        runner()
    except KeyboardInterrupt:
        _print("\nInterrupted by user.")
    except Exception as exc:
        import traceback as tb
        _print(f"\nError: {exc}")
        tb.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
