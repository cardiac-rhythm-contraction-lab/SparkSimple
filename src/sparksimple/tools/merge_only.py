#!/usr/bin/env python3
"""sparksimple.tools.merge_only — Companion tool that merges zData2.mat results from
multiple experiment sub-folders and re-runs ROI clustering on the pooled spark set.

Port of mfiles/MergeOnly.m.

Usage
-----
    python -m sparksimple.tools.merge_only          # interactive folder picker
    python -m sparksimple.tools.merge_only /path/to/root1 /path/to/root2 ...
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io

# Allow running as a standalone script from the repo root.
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from sparksimple.roi.cluster import cluster_rois  # noqa: E402
from sparksimple.export.matlab_compat import save_zdata2_mat, save_zmetadata_mat  # noqa: E402
from sparksimple.export.tables import generate_accepted_csv  # noqa: E402

# ═══════════════════════════════════════════════════════════════════════════════
# ══ Configuration — edit these values as needed
# ═══════════════════════════════════════════════════════════════════════════════

# Tag / prefix of the results sub-folder inside each experiment folder.
RF: str = "iT_03_r2_03"

# ROI clustering radius in micrometres.
roi_r_um: float = 2.0

# Name of the merged output folder (created inside the first selected folder).
output_folder_name: str = "MERGED"

# ═══════════════════════════════════════════════════════════════════════════════
# ══ End of configuration
# ═══════════════════════════════════════════════════════════════════════════════

_LAST_FOLDER_FILE = Path(__file__).parent.parent.parent.parent / ".mergeonly_last_folder.json"


# ── Console helpers ─────────────────────────────────────────────────────────

def _print(msg: str = "") -> None:
    print(msg, flush=True)


# ── Persistent last-folder ──────────────────────────────────────────────────

def _load_last_folder() -> Path | None:
    try:
        data = json.loads(_LAST_FOLDER_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None
    value = data.get("last_folder")
    if not value:
        return None
    path = Path(value)
    return path if path.exists() else None


def _save_last_folder(folder: Path) -> None:
    payload = {"last_folder": str(folder)}
    _LAST_FOLDER_FILE.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ── Folder picker ────────────────────────────────────────────────────────────

def _select_folders(initial_dir: Path | None = None) -> list[Path]:
    """Open one or more folder-selection dialogs; return selected paths."""
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox
    except Exception as exc:
        _print(f"Cannot load tkinter: {exc}")
        _print("Pass experiment folders as command-line arguments instead.")
        return []

    selected: list[Path] = []
    try:
        root = tk.Tk()
        root.withdraw()
        root.update()

        while True:
            title_suffix = f" (#{len(selected) + 1})" if selected else ""
            folder = filedialog.askdirectory(
                initialdir=str(initial_dir or (selected[-1].parent if selected else None) or ""),
                title=f"Select experiment folder{title_suffix} — Cancel when done",
                mustexist=True,
            )
            if not folder:
                break
            p = Path(folder)
            if p not in selected:
                selected.append(p)
                initial_dir = p.parent

            more = messagebox.askyesno(
                "Add more?",
                f"Added: {p.name}\n\nAdd another folder?",
            )
            if not more:
                break
    except Exception as exc:
        _print(f"Folder dialog error: {exc}")
    finally:
        try:
            root.destroy()
        except Exception:
            pass

    return selected


# ── .mat loading ─────────────────────────────────────────────────────────────

def _squeeze(arr: Any) -> Any:
    """Recursively squeeze length-1 numpy array wrappers."""
    while isinstance(arr, np.ndarray) and arr.size == 1:
        arr = arr.flat[0]
    return arr


def _scalar(arr: Any, default: float = 0.0) -> float:
    try:
        v = _squeeze(arr)
        return float(v)
    except Exception:
        return default


def _load_sparks_from_mat(zdata2_path: Path) -> list[dict]:
    """Load sparks from a zData2.mat file as a list of plain dicts."""
    mat = scipy.io.loadmat(str(zdata2_path), squeeze_me=False, struct_as_record=False)

    spkF = mat.get("spkF")
    if spkF is None:
        return []

    sparks: list[dict] = []
    spkF = spkF.squeeze()  # (N,) or scalar
    if spkF.ndim == 0:
        spkF = spkF.reshape(1)

    for s in spkF:
        def _f(name: str, default: float = 0.0) -> float:
            try:
                return _scalar(getattr(s, name), default)
            except Exception:
                return default

        def _i(name: str, default: int = 0) -> int:
            return int(_f(name, default))

        def _arr(name: str) -> np.ndarray:
            try:
                v = getattr(s, name)
                if isinstance(v, np.ndarray):
                    return v.squeeze()
                return np.array([], dtype=np.float64)
            except Exception:
                return np.array([], dtype=np.float64)

        good_raw = _f("good", 1.0)
        fail_raw = getattr(s, "fail", "no")
        if isinstance(fail_raw, np.ndarray):
            fail_raw = str(fail_raw.flat[0]) if fail_raw.size > 0 else "no"
        else:
            fail_raw = str(fail_raw)

        sparks.append({
            "px": _i("px"),
            "py": _i("py"),
            "pt": _i("pt"),
            "FWHM": _f("FWHM"),
            "fwhm": _f("FWHM"),
            "timesignal": _arr("timesignal"),
            "timeinterval": _arr("timeinterval"),
            "r2": _f("r2"),
            "amp": _f("amp"),
            "tau": _f("tau"),
            "ror": _f("ror"),
            "t2p": _f("t2p"),
            "FDHM": _f("FDHM"),
            "fdhm": _f("FDHM"),
            "AMP": _f("AMP"),
            "BL": _f("BL"),
            "bl": _f("BL"),
            "mass": _f("mass"),
            "dist2memb": _f("dist2memb"),
            "id": _i("id"),
            "good": bool(good_raw),
            "fail": fail_raw,
            "dist2spk": _f("dist2spk"),
            "roi": _i("roi"),
        })

    return sparks


class MergeCalibrationError(ValueError):
    """Raised when a result folder's acquisition metadata is missing, invalid,
    or incompatible with the other folders being merged.

    merge-only must never invent DX, DT, frame counts, or a cell mask: it
    pools real per-experiment results, so a missing or ambiguous calibration
    aborts the merge before anything is written.
    """


def _required_positive_scalar(mat: dict, key: str, source: Path) -> float:
    if key not in mat:
        raise MergeCalibrationError(f"{source}: missing mandatory metadata {key!r}")
    raw = mat[key]
    try:
        value = float(np.squeeze(raw))
    except (TypeError, ValueError) as exc:
        raise MergeCalibrationError(
            f"{source}: metadata {key!r} is not a numeric scalar ({raw!r})"
        ) from exc
    if not math.isfinite(value) or value <= 0:
        raise MergeCalibrationError(
            f"{source}: metadata {key!r} must be finite and positive; got {value!r}"
        )
    return value


def _load_metadata_from_mat(zmeta_path: Path) -> dict:
    """Load mandatory scalar metadata from a zMetaData.mat file.

    DX, DT, temps and mask are mandatory scientific inputs: none of them is
    ever substituted with a default. A missing, empty, non-numeric, zero,
    negative, NaN, or infinite value aborts with a clear error identifying
    the offending file.
    """
    if not zmeta_path.exists():
        raise MergeCalibrationError(f"{zmeta_path}: zMetaData.mat is missing")
    mat = scipy.io.loadmat(str(zmeta_path), squeeze_me=True)

    dx = _required_positive_scalar(mat, "DX", zmeta_path)
    dt = _required_positive_scalar(mat, "DT", zmeta_path)

    if "temps" not in mat:
        raise MergeCalibrationError(f"{zmeta_path}: missing mandatory metadata 'temps'")
    temps_raw = np.squeeze(mat["temps"])
    try:
        temps_float = float(temps_raw)
    except (TypeError, ValueError) as exc:
        raise MergeCalibrationError(
            f"{zmeta_path}: metadata 'temps' is not numeric ({temps_raw!r})"
        ) from exc
    temps = int(round(temps_float))
    if temps <= 0 or not math.isclose(temps_float, temps, abs_tol=1e-6):
        raise MergeCalibrationError(
            f"{zmeta_path}: 'temps' must be a positive integer; got {temps_raw!r}"
        )

    mask_raw = mat.get("mask")
    if not isinstance(mask_raw, np.ndarray) or mask_raw.size == 0:
        raise MergeCalibrationError(f"{zmeta_path}: missing or empty mandatory 'mask'")
    mask = np.squeeze(mask_raw).astype(np.float64)
    if mask.ndim != 2 or mask.shape[0] == 0 or mask.shape[1] == 0:
        raise MergeCalibrationError(
            f"{zmeta_path}: 'mask' must be a non-empty 2D array; got shape {mask.shape}"
        )

    if "fo" not in mat:
        raise MergeCalibrationError(f"{zmeta_path}: missing mandatory metadata 'fo'")
    fo_raw = mat["fo"]
    if isinstance(fo_raw, np.ndarray) and fo_raw.size > 1:
        fo = fo_raw.astype(np.float64)
    else:
        try:
            fo = float(np.squeeze(fo_raw))
        except (TypeError, ValueError) as exc:
            raise MergeCalibrationError(
                f"{zmeta_path}: metadata 'fo' is not numeric ({fo_raw!r})"
            ) from exc
        if not math.isfinite(fo):
            raise MergeCalibrationError(f"{zmeta_path}: 'fo'={fo!r} must be finite")

    return {
        "DX": dx,
        "DT": dt,
        "temps": temps,
        "mask": mask,
        "fo": fo,
        "source": zmeta_path,
    }


def _check_calibration_compatible(metas: list[dict], rel_tol: float = 1e-2) -> None:
    """Abort with the conflicting folder names if DX or DT disagree.

    Pooling sparks across result folders is only valid when every folder was
    acquired at the same spatial and temporal resolution; silently picking
    one folder's calibration for the merged output would misrepresent every
    other folder's sparks.
    """
    reference = metas[0]
    for other in metas[1:]:
        for field in ("DX", "DT"):
            if not math.isclose(reference[field], other[field], rel_tol=rel_tol, abs_tol=1e-9):
                raise MergeCalibrationError(
                    f"Incompatible {field} across result folders: "
                    f"{reference['source']} has {field}={reference[field]!r}, "
                    f"but {other['source']} has {field}={other[field]!r}"
                )


# ── Result-folder discovery ──────────────────────────────────────────────────

def _find_result_dirs(root: Path, rf_tag: str) -> list[Path]:
    """Return all sub-directories whose name starts with rf_tag, depth-first."""
    found: list[Path] = []
    for candidate in sorted(root.rglob("*")):
        if candidate.is_dir() and candidate.name.startswith(rf_tag):
            if (candidate / "zData2.mat").exists():
                found.append(candidate)
    return found


# ── Dummy merged volume ──────────────────────────────────────────────────────

def _build_dummy_volume(mask: np.ndarray, temps: int) -> np.ndarray:
    """Build a (Y, X, T) volume of ones matching the mask footprint.

    The merged tool does not have access to the original fluorescence volume,
    so we pass a ones-array.  ROI centroid and signal extraction will work
    on this neutral baseline.
    """
    h, w = mask.shape[:2]
    return np.ones((h, w, max(temps, 1)), dtype=np.float64)


# ── Main merge logic ─────────────────────────────────────────────────────────

def merge_only(
    experiment_dirs: list[Path],
    rf_tag: str = RF,
    roi_r_um_val: float = roi_r_um,
    output_folder: str = output_folder_name,
) -> Path:
    """Merge sparks from result sub-folders and re-cluster ROIs.

    Args:
        experiment_dirs: Root experiment folders selected by the user.
        rf_tag: Prefix that identifies result sub-folders.
        roi_r_um_val: ROI clustering radius in µm.
        output_folder: Name of the output folder created under the first experiment.

    Returns:
        Path to the MERGED output directory.
    """
    # ── 1. Collect all result directories ───────────────────────────────────
    all_result_dirs: list[Path] = []
    for root in experiment_dirs:
        found = _find_result_dirs(root, rf_tag)
        _print(f"  {root.name}: {len(found)} result folder(s)")
        all_result_dirs.extend(found)

    if not all_result_dirs:
        raise RuntimeError(
            f"No result folders starting with '{rf_tag}' containing zData2.mat were found."
        )

    _print(f"\nTotal result folders to merge: {len(all_result_dirs)}")

    # ── 2. Load and validate metadata from every folder, and aggregate sparks.
    # All validation happens before anything is written: a bad folder must
    # never leave partial output on disk.
    all_sparks: list[dict] = []
    metas: list[dict] = []

    for rdir in all_result_dirs:
        sparks = _load_sparks_from_mat(rdir / "zData2.mat")
        _print(f"  Loaded {len(sparks):4d} sparks from {rdir}")
        all_sparks.extend(sparks)
        metas.append(_load_metadata_from_mat(rdir / "zMetaData.mat"))

    _check_calibration_compatible(metas)

    # Re-index spark IDs so they are globally unique (1-based)
    for new_id, spk in enumerate(all_sparks, start=1):
        spk["id"] = new_id

    _print(f"\nTotal sparks aggregated: {len(all_sparks)}")

    # ── 3. Reduce per-folder metadata to the pooled calibration ─────────────
    reference = metas[0]
    dx_um = reference["DX"]
    dt_ms = reference["DT"]
    mask = reference["mask"]
    fo = reference["fo"]

    # Total imaging duration is the SUM of each folder's own recording
    # duration (they are independent acquisitions, not one continuous
    # timeline), not the frame count of an arbitrary single folder. DX/DT
    # were just confirmed identical across folders, so the elapsed time of
    # folder i is (temps_i - 1) * dt_ms. `effective_temps` is chosen so that
    # downstream helpers -- which all derive duration as
    # (temps - 1) * dt_ms / 1000 -- reproduce that summed duration exactly.
    total_frame_intervals = sum(max(0, m["temps"] - 1) for m in metas)
    effective_temps = total_frame_intervals + 1
    temps = effective_temps

    volume = _build_dummy_volume(mask, temps)

    # ── 4. Create output directory (only after all validation has passed) ──
    out_dir = experiment_dirs[0] / output_folder
    out_dir.mkdir(parents=True, exist_ok=True)
    _print(f"Output directory: {out_dir}")

    # ── 5. ROI clustering ────────────────────────────────────────────────────

    _print(f"\nClustering ROIs (radius={roi_r_um_val} µm, DX={dx_um} µm/px)...")
    clustering = cluster_rois(
        sparks=all_sparks,
        volume=volume,
        mask=mask,
        dx_um=dx_um,
        dt_ms=dt_ms,
        roi_r_um=roi_r_um_val,
    )
    _print(f"ROIs found: {len(clustering.rois)}")

    # ── 6. Save outputs ──────────────────────────────────────────────────────
    zdata_out = str(out_dir / "zDataMERGE.mat")
    zmeta_out = str(out_dir / "zMetaData.mat")
    csv_out = str(out_dir / "yData_MERGED.csv")

    _print("\nSaving zDataMERGE.mat...")
    save_zdata2_mat(zdata_out, clustering.sparks, clustering.rois)

    _print("Saving zMetaData.mat...")
    save_zmetadata_mat(zmeta_out, fo, mask, temps, dx_um, dt_ms, roi_r_um_val)

    _print("Saving yData_MERGED.csv...")
    generate_accepted_csv(
        sparks=clustering.sparks,
        rois=clustering.rois,
        fo_map=fo,
        mask=mask,
        dx_um=dx_um,
        dt_ms=dt_ms,
        temps=temps,
        output_path=csv_out,
    )

    # ── 7. HTML report ───────────────────────────────────────────────────────
    _print("Generating HTML report...")
    try:
        _generate_report(
            clustering=clustering,
            fo=fo,
            mask=mask,
            dx_um=dx_um,
            dt_ms=dt_ms,
            temps=temps,
            out_dir=out_dir,
            roi_r_um_val=roi_r_um_val,
        )
    except Exception as exc:
        _print(f"Warning: HTML report generation failed: {exc}")

    _print(f"\nDone.  Output: {out_dir}")
    return out_dir


def _generate_report(
    clustering: Any,
    fo: Any,
    mask: np.ndarray,
    dx_um: float,
    dt_ms: float,
    temps: int,
    out_dir: Path,
    roi_r_um_val: float,
) -> None:
    """Build a minimal HTML report reusing the sparksimple report infrastructure."""
    from sparksimple.export.models import ExportSummary, ExportPaths
    from sparksimple.export.pipeline import Phase10Result
    from sparksimple.report.html import generate_report

    sparks = clustering.sparks
    n_accepted = sum(1 for s in sparks if s.get("good", True))
    n_rejected = len(sparks) - n_accepted
    duration_s = max(0, temps - 1) * dt_ms / 1000.0
    in_mask_px = int(np.sum(mask == 1))
    cell_area_um2 = in_mask_px * dx_um * dx_um

    if isinstance(fo, np.ndarray):
        fo_in_mask = fo[mask == 1]
        mean_fo = float(np.mean(fo_in_mask)) if len(fo_in_mask) > 0 else 0.0
        std_fo = float(np.std(fo_in_mask, ddof=1)) if len(fo_in_mask) > 1 else 0.0
    else:
        mean_fo = float(fo)
        std_fo = 0.0

    spk_freq = n_accepted / duration_s if duration_s > 0 else 0.0
    spk_density_per_um2_s = (
        n_accepted / duration_s / cell_area_um2
        if (duration_s > 0 and cell_area_um2 > 0)
        else 0.0
    )
    spk_density_per_mm2_s = spk_density_per_um2_s * 1e6

    summary = ExportSummary(
        n_candidates=len(sparks),
        n_accepted=n_accepted,
        n_rejected=n_rejected,
        spark_freq=spk_freq,
        spark_density=spk_density_per_mm2_s,
        spark_density_per_um2_s=spk_density_per_um2_s,
        spark_density_per_mm2_s=spk_density_per_mm2_s,
        cell_area_um2=cell_area_um2,
        duration_s=duration_s,
        mean_fo=mean_fo,
        std_fo=std_fo,
    )

    paths = ExportPaths(
        accepted_csv=str(out_dir / "yData_MERGED.csv"),
        rejected_csv="",
        zdata2_mat=str(out_dir / "zDataMERGE.mat"),
        zmetadata_mat=str(out_dir / "zMetaData.mat"),
    )

    phase10_result = Phase10Result(
        clustering=clustering,
        summary=summary,
        paths=paths,
    )

    generate_report(
        result=phase10_result,
        output_dir=str(out_dir),
        title="MERGED",
        dx_um=dx_um,
        dt_ms=dt_ms,
        roi_r_um=roi_r_um_val,
        volume=None,
        mask=mask,
        fo_map=fo if isinstance(fo, np.ndarray) else None,
        generate_film=False,
    )


# ── CLI entry point ──────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) > 1:
        dirs = [Path(p) for p in sys.argv[1:]]
        missing = [d for d in dirs if not d.exists()]
        if missing:
            for d in missing:
                _print(f"Error: folder not found: {d}")
            sys.exit(1)
    else:
        last = _load_last_folder()
        _print("MergeOnly — select experiment folders to merge")
        _print("(Cancel the dialog when you have added all folders)\n")
        dirs = _select_folders(initial_dir=last)
        if not dirs:
            _print("No folders selected. Exiting.")
            sys.exit(0)

    _save_last_folder(dirs[0])
    _print(f"\nFolders to process ({len(dirs)}):")
    for d in dirs:
        _print(f"  {d}")
    _print(f"\nResult tag  : {RF}")
    _print(f"ROI radius  : {roi_r_um} µm")
    _print(f"Output name : {output_folder_name}\n")

    try:
        merge_only(
            experiment_dirs=dirs,
            rf_tag=RF,
            roi_r_um_val=roi_r_um,
            output_folder=output_folder_name,
        )
    except Exception as exc:
        _print(f"\nError: {exc}")
        import traceback as tb
        tb.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
