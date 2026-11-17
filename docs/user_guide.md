# SparkSimple2 — User Guide

This guide covers everything needed to run SparkSimple2: installation, data requirements, CLI usage, configuration, output files, the Python API, and the companion post-processing tools.

---

## Table of Contents

1. [What SparkSimple2 does](#what-sparksimple2-does)
2. [Installation](#installation)
3. [Data requirements](#data-requirements)
4. [Running the MATLAB-style master script](#running-the-matlab-style-master-script)
5. [Running the pipeline (CLI)](#running-the-pipeline-cli)
6. [CLI reference](#cli-reference)
7. [Multi-channel experiments](#multi-channel-experiments)
8. [Simulated vs real data](#simulated-vs-real-data)
9. [Output files](#output-files)
10. [Configuration reference](#configuration-reference)
11. [Python API](#python-api)
12. [Companion tools](#companion-tools)
13. [Numerical accuracy](#numerical-accuracy)
14. [Troubleshooting](#troubleshooting)

---

## What SparkSimple2 does

SparkSimple2 detects and characterises Ca²⁺ sparks in confocal frame-scan (x-y-t) image series. It implements a 10-phase pipeline for Ca²⁺ spark detection in cardiac myocytes:

1. **Load** TIFF image stack + XML metadata
2. **Preprocess** — baseline estimation (F₀), normalisation to ΔF/F₀, moving-average smoothing
3. **Wavelet transform** — continuous wavelet transform (CWT) using a Mexican-Hat wavelet across multiple scales → V2 volume
4. **3D merge** — convolve with a Gaussian kernel, threshold at 2.22 × MAD noise level
5. **Candidate detection** — RegMax3D finds local maxima in the thresholded volume
6. **F₀ re-estimation** — local baseline re-estimated per candidate region
7. **Feature extraction** — amplitude, FWHM, FDHM, τ (exponential decay), rise time, mass
8. **Physiological filtering** — discard sparks outside user-defined physiological ranges
9. **ROI clustering** — group spatially close sparks into regions of interest
10. **Export** — CSV files and MATLAB-compatible `.mat` files

---

## Installation

```bash
cd python_port
pip install -e .
```

For development extras (pytest, matplotlib):

```bash
pip install -e ".[dev]"
```

**Dependencies:** Python ≥ 3.10, NumPy ≥ 1.24, SciPy ≥ 1.10, tifffile ≥ 2023.1.

---

## Data requirements

### Folder structure

Each experiment must be a folder containing:

```
MyExperiment/
  metadata.xml          ← required: pixel size, frame rate, dimensions
  frame0001_ch00.tif    ← TIFF time series (any of the naming conventions below)
  frame0002_ch00.tif
  ...
```

The pipeline discovers TIFFs automatically. Two naming conventions are supported:

| Dataset type | File naming example |
|---|---|
| Real (multi-channel, z-indexed) | `MVM_..._z100001_ch00.tif`, `..._ch01.tif`, `..._ch02.tif` |
| Simulated (single-channel) | `Cabcnew0001_ch00.tif`, `Cabcnew0002_ch00.tif` |

### metadata.xml

The metadata file must contain at minimum:

- `dx_um` — pixel size in µm (spatial resolution)
- `dt_ms` — frame interval in ms (temporal resolution)
- `n_pixels_x`, `n_pixels_y` — image dimensions
- `n_frames` — number of frames

This file is produced automatically by the acquisition software (e.g. Leica LAS X).

---

## Running the MATLAB-style master script

The file `SparkSimple2.py` in the root of `python_port` is the Python equivalent
of the original MATLAB `SparkSimple2.m` master script. Use it when you want the
same interactive batch workflow as MATLAB: configure parameters at the top of
the file, select folders with a file picker, and process each experiment into an
`Analysis` subfolder.

```bash
cd python_port
python SparkSimple2.py
```

By default, the script opens one folder-selection dialog (`N = 1`) and processes
the selected experiment. Results are written to:

```
<experiment>/Analysis/
```

### Configuration block

Edit the block at the top of `SparkSimple2.py` before running. It documents every parameter with an inline comment.

| Parameter | Default | Meaning |
|---|---:|---|
| `N` | `1` | Number of experiments to select. `0` means select one root folder and process all direct subfolders. |
| `tag_channel` | `0` | Ca²⁺ channel: 0-indexed int (MATLAB `tagSpk='ch00'` corresponds to `0`), or a substring tag string like `"ch01"` that pre-filters filenames exactly like MATLAB's `tagSpk` — safer than the int form if channel suffixes aren't consistently zero-padded. |
| `simulated` | `False` | Set `True` for simulated data using simulated wavelet timing behaviour. |
| `results_folder` | `"Analysis"` | Results subfolder created inside each experiment. |
| `skip_processed` | `True` | Skip folders where `<experiment>/<results_folder>/zData2.mat` already exists. |
| `roi_radius_um` | `1.5` | ROI radius around each spark candidate centre. |
| `filter_window_frames` | `1` | Temporal filtering window in frames. |
| `merge_distance_um` | `3.0` | Spatial merge distance in µm. |
| `merge_time_ms` | `30.0` | Temporal merge half-window in ms. |
| `n_candidates` | `100` | Maximum number of spark candidates to search for. |
| `min_amplitude` | `0.3` | MATLAB `iT`: minimum relative spark amplitude. |
| `max_baseline` | `1.0` | MATLAB `Bl`: maximum allowed pre-spark baseline. |
| `fdhm_range_ms` | `(10, 200)` | MATLAB `Fd`: valid FDHM interval in ms. |
| `tau_range_ms` | `(5, 500)` | MATLAB `ta`: valid decay tau interval in ms. |
| `t2p_range_ms` | `(0, 200)` | MATLAB `t2`: valid time-to-peak interval in ms. |
| `min_decay_r2` | `0.3` | MATLAB `r2`: minimum R² for the decay fit. |
| `sp` | `1` | MATLAB-style starting point / resume point. |

### Folder selection behaviour

`SparkSimple2.py` uses `tkinter.filedialog.askdirectory()`.

- If `N > 0`, the folder picker opens `N` times.
- If `N == 0`, the folder picker opens once for a root folder, then every direct
  subfolder of that root is processed as one experiment.
- The last selected folder is remembered in `.sparksimple2_last_folder.json`,
  replacing MATLAB's `ultimpaquet.mat` behaviour.

If `tkinter` is unavailable or the machine has no display, pass experiment
folders directly:

```bash
python SparkSimple2.py /path/to/MyExperiment
```

You can pass multiple folders:

```bash
python SparkSimple2.py /path/to/Exp1 /path/to/Exp2 /path/to/Exp3
```

When folders are passed as arguments, no file picker is opened.

### `sp` resume points

`sp` is implemented with a Python-native checkpoint stored in the results folder:

```
<experiment>/<results_folder>/spark_pipeline_checkpoint.pkl
```

Run once with `sp = 1` to create the checkpoint. Later runs can resume from a
later point:

| `sp` | Resume point |
|---:|---|
| `1` | Full run from detection/preprocessing onward. Creates/updates the checkpoint. |
| `2` | Resume after detection; run feature extraction, filtering, ROI clustering and export. |
| `3` | Resume after feature extraction; run filtering, ROI clustering and export. |
| `4` | Resume after filtering; run ROI clustering and export. |
| `5` | Resume after ROI clustering; redo export only. |

The checkpoint stores Python dataclass objects and arrays needed to resume the pipeline without recomputing previous phases. MATLAB-compatible files (`zData2.mat` and `zMetaData.mat`) are still written for downstream analysis.

### Error handling

Like the MATLAB `try/catch` loop, the master script catches errors per folder:
if one experiment fails, the error is printed and the next selected folder is
processed.

---

## Running the pipeline (CLI)

The simplest invocation:

```bash
sparksimple run /path/to/MyExperiment
```

The pipeline prints progress to stderr and a one-line summary to stdout when finished:

```
Phase 1: loading metadata...
Phase 2: loading image stack...
Phase 3-4: preprocessing and cell mask segmentation...
...
Phase 10: export...

Accepted sparks : 42
Rejected sparks : 18
ROIs            : 12
Output          : /path/to/MyExperiment/Python_Result
```

Output is written to `<experiment>/Python_Result/` by default. Use
`--output-dir` to choose a different location.

### Typical invocation for a real multi-channel experiment

Real experiments usually have multiple channels (Ca²⁺ channel + reference). Always specify the channel explicitly:

```bash
sparksimple run /path/to/MVM_220208 --channel 0 --output-dir results/
```

### Typical invocation for simulated data

```bash
sparksimple run /path/to/NOISE0 --simulated --output-dir results/
```

The `--simulated` flag adjusts the CWT scale range for 10 ms frame intervals. Without it the pipeline assumes real experimental data with slower dynamics.

---

## CLI reference

```
sparksimple run <experiment_dir> [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `<experiment_dir>` | *(required)* | Path to the folder containing TIFFs and metadata.xml |
| `--output-dir PATH` | `<experiment>/Python_Result/` | Directory where results are written |
| `--title TEXT` | folder name | Title prefix for output filenames |
| `--channel INT\|TAG` | `0` | 0-indexed channel, or a substring tag like `ch01` (MATLAB `tagSpk` style). **Always set this for multi-channel datasets.** |
| `--simulated` | off | Use CWT scales for simulated data (DT=10 ms). Omit for real experiments. |
| `--dx-um FLOAT` | from metadata | Override pixel size in µm |
| `--dt-ms FLOAT` | from metadata | Override frame interval in ms |
| `--report` | off | Generate an HTML summary report in the output folder |
| `--film` | off | Save per-candidate detection images to `detFilm/` subfolder |
| `--roi-radius FLOAT` | `1.5` | ROI clustering radius in µm |
| `--debug` | off | Print extra diagnostic information during pipeline execution |

---

## Multi-channel experiments

MATLAB's original workflow filters TIFFs to a single channel before running SparkSimple2. In Python all channels are visible, so you **must specify the channel** with `--channel`:

```bash
sparksimple run /path/to/experiment --channel 0
```

If you omit `--channel` on a multi-channel dataset, the pipeline auto-selects the channel with the highest total intensity and emits a warning:

```
UserWarning: Multi-channel dataset: auto-selected channel 2 (max total intensity).
Channel sums: [ch0=51234, ch1=48120, ch2=891456].
Pass channel=<n> to load_image_stack() or --channel <n> on the CLI to override (0-indexed).
```

This auto-selection often picks the wrong channel (e.g. a bright blue reference channel instead of the Ca²⁺ signal). Always override it explicitly.

### Substring tags (MATLAB `tagSpk` equivalent)

`--channel` also accepts a text tag instead of an int, replicating MATLAB's `tagSpk` filter from `SparkSimple2.m`: the filename listing is pre-filtered to only files containing that substring *before* any grouping happens, rather than indexing into an already-grouped list:

```bash
sparksimple run /path/to/experiment --channel ch01
```

Use this form if a dataset's channel suffixes aren't consistently zero-padded (e.g. a mix of `ch1`/`ch01`), since the int form relies on alphabetical sort order matching numeric channel order. A tag that matches more than one channel's files for the same frame (e.g. `ch0` also matching `ch01`) raises a clear "ambiguous tag" error instead of silently picking one.

---

## Simulated vs real data

The `--simulated` flag controls the CWT wavelet scale range:

| Mode | Frame interval | Wavelet scales |
|---|---|---|
| `--simulated` | DT=10 ms | scales 2–5 (shorter sparks) |
| *(default, real)* | DT varies | scales 5–10 (longer sparks) |

Use `--simulated` only for synthetic datasets generated with 10 ms frame intervals. For all real confocal acquisitions use the default (no flag).

---

## Output files

The package CLI writes output to `<experiment>/Python_Result/` by default, or to
the exact folder passed with `--output-dir`. The MATLAB-style master script
writes output to `<experiment>/Analysis/` by default.

| File | Description |
|---|---|
| `yData_<title>.csv` | Accepted sparks with full feature set (see columns below) |
| `zDataRejected_<title>.csv` | Rejected sparks with failure reason |
| `zData2.mat` | MATLAB-compatible data: spark features and ROI traces |
| `zMetaData.mat` | MATLAB-compatible metadata: Fo, mask, frame count, DX, DT, ROI radius |
| `spark_pipeline_checkpoint.pkl` | Python-native checkpoint used by `sp=2..5` resume points |
| `report.html` | HTML summary (only with `--report`) |
| `detFilm/` | Per-candidate detection images (only with `--film`) |

### `yData_<title>.csv` columns

| Column | Unit | Description |
|---|---|---|
| `ID` | — | Spark index |
| `Xpix` | pixels | X position (column) |
| `Ypix` | pixels | Y position (row) |
| `Tframe` | frames | Peak frame |
| `dist2memb` | µm | Distance to cell membrane |
| `dist2closest` | µm | Distance to closest other spark |
| `AMP` | ΔF/F₀ | Peak amplitude (re-estimated F₀) |
| `amp` | ΔF/F₀ | Peak amplitude (global F₀) |
| `BL` | ΔF/F₀ | Local baseline |
| `RoR` | ΔF/F₀/ms | Rate of rise |
| `mass` | ΔF/F₀·frame | `sum(T)` over the snippet, no BL subtraction or DT factor |
| `t2p` | ms | Time to peak |
| `FDHM` | ms | Full duration at half maximum |
| `FWHM` | µm | Full width at half maximum |
| `tau` | ms | Exponential decay time constant |
| `fldr` | — | Folder / experiment index |
| `subfldr` | — | Sub-folder index |
| `ROI` | — | ROI cluster ID |

The file header also includes experiment metadata: pixel size, frame rate, total duration, cell area, mean F₀ ± SD, candidate count, accepted count, spark frequency (spk/s), and spark density (spk/s/mm²).

---

## Configuration reference

When using the Python API directly, pipeline behaviour is controlled via `PipelineConfig` and its sub-configs.

### DetectionConfig

```python
from sparksimple.config import DetectionConfig

DetectionConfig(
    merge_distance_um=3.0,      # Gaussian merge kernel spatial radius (µm)
    merge_time_ms=30.0,         # Gaussian merge kernel temporal radius (ms)
    filter_window_frames=1,     # Temporal smoothing before CWT (1 = no smoothing)
    noise_connectivity=100,     # Max candidate count (RegMax3D ceiling)
)
```

### FilteringConfig

```python
from sparksimple.config import FilteringConfig

# Use MATLAB defaults — recommended
filt = FilteringConfig.from_matlab_defaults()

# Manual configuration
filt = FilteringConfig(
    min_amplitude=0.3,          # Minimum ΔF/F₀ amplitude
    max_baseline=1.0,           # Maximum baseline (rejects drift artefacts)
    min_decay_r2=0.3,           # Minimum R² for exponential decay fit
    t2p_range_ms=(0, 200),      # Time-to-peak range (ms)
    tau_range_ms=(5, 500),      # Decay τ range (ms)
    fdhm_range_ms=(10, 200),    # Duration at half-max range (ms)
)
```

### ROIConfig

```python
from sparksimple.config import ROIConfig

ROIConfig(
    roi_radius_um=1.5,          # Radius for ROI clustering (µm)
)
```

---

## Python API

For scripting or batch processing, the recommended high-level API is
`run_pipeline()`. It is the same orchestrator used by `SparkSimple2.py`.

```python
from sparksimple.config import PipelineConfig, FilteringConfig, ROIConfig
from sparksimple.pipeline import run_pipeline

cfg = PipelineConfig(
    filtering=FilteringConfig.from_matlab_defaults(),
    roi=ROIConfig(roi_radius_um=1.5),
    channel=0,
)

result = run_pipeline(
    input_dir="path/to/experiment",
    output_dir="path/to/experiment/Analysis",
    cfg=cfg,
    title="MyExperiment",
    is_simulated=False,
    progress_fn=print,
    start_stage=1,  # Same meaning as SparkSimple2.py `sp`
)
```

To resume from a checkpoint, pass the same `output_dir` and set
`start_stage=2`, `3`, `4`, or `5`. This uses the same
`spark_pipeline_checkpoint.pkl` file as the master script.

For lower-level development or phase-by-phase validation, the pipeline can also
be driven one phase at a time:

```python
from sparksimple.config import PipelineConfig, FilteringConfig, ROIConfig
from sparksimple.io import load_image_stack, load_metadata
from sparksimple.preprocessing import preprocess
from sparksimple.detection import (
    compute_v2, apply_3d_merge_and_threshold,
    detect_candidates, reestimate_fo,
)
from sparksimple.features import extract_features
from sparksimple.filtering import filter_sparks
from sparksimple.roi import cluster_rois
from sparksimple.export import run_phase10

# Configuration
cfg = PipelineConfig(filtering=FilteringConfig.from_matlab_defaults())

# Load data (always pass channel= for multi-channel experiments)
stack, channel = load_image_stack("path/to/experiment/", channel=0)
meta = load_metadata("path/to/experiment/")
dx_um, dt_ms = meta.dx_um, meta.dt_ms

# Preprocessing (Phases 3–4)
prep = preprocess(stack, cfg)

# CWT (Phase 5)
wav = compute_v2(prep.volume_norm, dt_ms=dt_ms, is_simulated=False, cfg=cfg)

# 3D merge + threshold (Phase 6A)
merged = apply_3d_merge_and_threshold(
    wav.v2, prep.mask, dx_um=dx_um, dt_ms=dt_ms,
    merge_distance_um=cfg.detection.merge_distance_um,
    merge_time_ms=cfg.detection.merge_time_ms,
)

# Candidate detection (Phase 6B)
cands = detect_candidates(
    merged.v3_thresholded,
    max_candidates=cfg.detection.noise_connectivity,
)

# F₀ re-estimation (Phase 7)
reest = reestimate_fo(
    prep.volume_raw, merged.v3_thresholded, prep.mask,
    prep.fo, dx_um=dx_um, dt_ms=dt_ms,
    roi_r_um=cfg.roi.roi_radius_um,
)

# Feature extraction (Phase 8)
feat_res = extract_features(
    reest.volume_renormalised, cands.ix_matrix,
    prep.mask, dx_um=dx_um, dt_ms=dt_ms,
)

# Filtering (Phase 9)
ny, nx, _ = prep.volume_raw.shape
filt = filter_sparks(feat_res.features, cfg.filtering, shape=(ny, nx))

# ROI clustering + export (Phase 10)
result = run_phase10(
    filt.records,
    reest.volume_renormalised,
    prep.mask,
    reest.fo_map,
    dx_um=dx_um,
    dt_ms=dt_ms,
    roi_r_um=cfg.roi.roi_radius_um,
    output_dir="output/",
    title="MyExperiment",
)
print(f"Accepted: {result.summary.n_accepted}, ROIs: {len(result.clustering.rois)}")
```

### Key return types

| Object | Attribute of interest |
|---|---|
| `PreprocessResult` (from `preprocess`) | `.volume_norm`, `.volume_raw`, `.mask`, `.fo` |
| `WaveletResult` (from `compute_v2`) | `.v2` — CWT volume, shape (Y, X, T) |
| `MergeResult` (from `apply_3d_merge_and_threshold`) | `.v3_thresholded`, `.umbral`, `.v3_conv`, `.kernel`, `.centroid_pixel` |
| `CandidateResult` (from `detect_candidates`) | `.ix_matrix` — RegMax3D rank matrix; candidate count = `(ix_matrix > 0).sum()` |
| `FoReestimateResult` (from `reestimate_fo`) | `.fo_map` (2D array), `.volume_renormalised` |
| `FilteredSpark` | `.feature.x_0b`, `.feature.y_0b`, `.feature.t_0b`, `.feature.amp`, `.good`, `.fail` |

---

## Companion tools

SparkSimple2 includes three post-processing tools for interactive review and correction of detection results. Run them via `SparkSimpleTools.py` or the `sparksimple` CLI.

### MergeOnly

Combines `zData2.mat` results from multiple experiment folders, re-indexes spark IDs globally, re-runs ROI clustering, and writes a unified output folder.

**Script (`SparkSimpleTools.py`):** set `tool = "merge"`, adjust `RF`, `roi_r_um`, `output_folder_name`, then:

```bash
cd python_port
python SparkSimpleTools.py
```

**CLI:**

```bash
sparksimple merge-only /path/to/Exp1 /path/to/Exp2 /path/to/Exp3
```

If no folders are given, a multi-folder picker opens. The merged output lands in `<first_experiment>/<output_folder_name>/`.

Output files inside the merged folder:

| File | Description |
|---|---|
| `zDataMERGE.mat` | Combined spark data |
| `zMetaData.mat` | Metadata from the first experiment |
| `yData_MERGED.csv` | Accepted sparks CSV |
| `zDataRejected_MERGED.csv` | Rejected sparks CSV |
| `report.html` | HTML summary report |

| Config variable | Default | Meaning |
|---|---|---|
| `RF` | `"iT_03_r2_03"` | Prefix of the results subfolder inside each experiment folder |
| `roi_r_um` | `2.0` | ROI clustering radius in µm |
| `output_folder_name` | `"MERGED"` | Name of the output folder created inside the first experiment |

### ValidateNotMerge

Opens a browser-based review interface where you can accept (Y) or reject (N) each detected spark individually. On completion it rewrites `zData2.mat`, CSVs, and the HTML report with the updated `good` flags.

**Script (`SparkSimpleTools.py`):** set `tool = "validate"`, adjust `rf_validate` and `port_validate`, then:

```bash
cd python_port
python SparkSimpleTools.py
```

**CLI:**

```bash
sparksimple validate /path/to/Experiment --rf iT_03_r2_03 --port 5555
```

If no folder is given, a folder picker opens. Navigate to `http://localhost:<port>` in a browser.

Keyboard shortcuts:

| Key | Action |
|---|---|
| **Y** | Accept current spark |
| **N** | Reject current spark |
| **←** / **→** | Navigate film frames |
| **Done** button | Save decisions, rewrite outputs, shut down server |

| Config variable | Default | Meaning |
|---|---|---|
| `rf_validate` | `"iT_03_r2_03"` | Results subfolder tag |
| `port_validate` | `5555` | Local Flask server port |

### ModifyDist2MembraneManual

Opens a browser-based overlay where you can click on the fluorescence image to manually set the distance-to-membrane (`dist2memb`) for accepted sparks. Updates `zData2.mat` in place.

**Script (`SparkSimpleTools.py`):** set `tool = "dist2memb"`, adjust `port_d2m`, then:

```bash
cd python_port
python SparkSimpleTools.py
```

**CLI:**

```bash
sparksimple modify-d2m /path/to/Results
```

If no folder is given, a folder picker opens. Navigate to `http://localhost:<port>` in a browser.

The tool detects MERGED results (presence of `zDataMERGE.mat`) automatically and loads the appropriate data file.

| Config variable | Default | Meaning |
|---|---|---|
| `port_d2m` | `5556` | Local Flask server port |

---

## Numerical accuracy

| Stage | Result | Max deviation |
|---|---|---|
| Baseline F₀ | Exact | 0.000 |
| Cell mask | Exact pixel-by-pixel | IoU = 1.000 |
| Temporal smoothing | Machine epsilon | 2.22 × 10⁻¹⁶ |
| CWT / V2 | Float32 stored | < 1 × 10⁻³ |
| 3D convolution | Machine epsilon | 1.55 × 10⁻¹⁵ |
| Noise threshold (MAD) | Exact | 0.000 |
| RegMax3D candidates | Exact | 0.000 |
| F₀ re-estimation | Machine epsilon | 9.95 × 10⁻¹⁴ |
| Feature extraction | Under extended validation | See `docs/feature_extraction_validation_plan.md` |
| Temporal tau fit | Guarded bounded fit | Robust approximation falls back when residual quality is poor |
| Spatial FWHM | Near-exact | < 2.3% |
| Physiological filtering | Exact | 0 mismatches |
| ROI clustering | Exact grouping | ID order may differ for equal-count singletons |
| CSV export | Exact format | Column order, row counts, separators match |

**Notes:**
- Temporal `tau` fitting uses a documented Python-improvement policy: bounded least-squares is the stable baseline, the MATLAB-style robust bisquare approximation may be attempted, and Python falls back to the stable fit when the robust result is poor or misleading. See `docs/feature_extraction_audit/tau_fit_policy.md`.
- Spatial FWHM uses `scipy.optimize.curve_fit` (L2 loss) instead of robust bisquare fitting. The difference is bounded at < 2.3% and does not affect which sparks are accepted.
- ROI label numbering for singletons with equal spark counts may differ due to cluster traversal order; the grouping itself is identical.

---

## Troubleshooting

### No TIFF files found

```
FileNotFoundError: No TIFF files found in /path/to/experiment
```

Check that the folder contains `.tif` or `.tiff` files. The pipeline looks for files directly in the specified folder (not subdirectories).

### Wrong channel loaded

If results show no sparks or unexpectedly high background, the wrong channel may have been loaded. For multi-channel experiments always pass `--channel <n>` explicitly and check the UserWarning that shows channel sums.

### Candidate count is zero

If RegMax3D returns 0 candidates:
- Check that `--simulated` is set correctly (wrong CWT scales can suppress all signal).
- Check `--dx-um` and `--dt-ms` match your acquisition parameters — wrong resolution causes the merge kernel to be the wrong size.
- Try running with `--debug` to inspect intermediate volumes.

### AttributeError on `prep.fo`

`prep.fo` is a scalar float, not an array. Do not call `.min()` or `.max()` on it. Use `float(prep.fo)` if you need a scalar.

### `ix_matrix.sum()` is much larger than expected

`ix_matrix` stores rank values (1–51) at each candidate location, not binary flags. The correct candidate count is `(cands.ix_matrix > 0).sum()`, not `cands.ix_matrix.sum()`.

### `sp=2..5` says the checkpoint is missing

Resume points require a previous run that created:

```
<experiment>/<results_folder>/spark_pipeline_checkpoint.pkl
```

Run `SparkSimple2.py` once with `sp = 1` and the same `results_folder`, then
rerun with the desired `sp` value. If you deleted or moved the `Analysis` folder,
the checkpoint must be recreated.

### Running tests

```bash
cd python_port
python -m pytest tests/ -q
```

Integration tests that require the reference datasets are automatically skipped if the data folders are missing.


## Compatibility changes, 2026-10-06

TIFF input uses frame scales 5–10 even with `--simulated` or
`is_simulated=True`; these flags now affect provenance only and preserve the
actual XML filename. Low-level `get_wavelet_scales`/`compute_v2` accept the new
keyword-only `input_format="ata"` for ceil(12/DT)..ceil(48/DT). This does not
add an ATA loader to the TIFF pipeline. Existing configurable scale fields
retain their names. Comparison/benchmark scripts passing simulated provenance
now also receive TIFF scales. Custom values still override the scale ranges.

The default temporal split is disabled (`temporal_split_frac=0.0`); explicit
nonzero values select a Python extension outside MATLAB-compatible behaviour.
`noise_connectivity`/NC is a stopping target checked after the whole threshold
level, and candidate counts can exceed it. A saturated run is marked by
`candidate_limit_reached` and a warning. Old checkpoints retain their serialized
configuration: restart from stage 1 for a new comparison of the defaults.

Temporal O-3/O-4 guards have been removed. An invalid MATLAB peak window yields
NaN temporal metrics, preserves `mass`, and records the local reason in the
existing `tau_fit_fallback_reason` field. Tau uses bisquare IRLS without the
previous unweighted-R² quality fallback. Export column names and MAT fields
are unchanged. `r2_robust_unweighted` remains diagnostic; `r2` uses robust MSE.

`mass` is not a Ca²⁺ mass or an integral in ms. For a temporal integral of the
same snippet, multiply its value by DT; this is not done in the export.
See [validation and limitations](matlab_parity/report.md).

A single accepted spark receives roi=1 but leaves the ROIs array empty, as in
MATLAB. This removes the previous Python-only singleton ROI record; clients
should read the accepted spark even when no ROI table row is exported. Empty
ROI masks yield NaN signals, without averaging outside the mask.


## Algorithmic compatibility and numerical identity, 2026-10-07

The default follows the principal MATLAB definitions and decision rules; this
is not a claim of identical implementation for every input. RoR uses the same
rising-segment search and linear slope definition, with numerical agreement
in the 18-case comparison. Tau uses the same decay model, starting values,
bounds and bisquare procedure, but its optimized coefficient can differ.

Fixed-state controls verified the bisquare weights and robust R² formulas.
The main differences in the studied cases arise from the numerical initial
fit/Jacobian, amplified by redundant exponential parameters. SciPy remains
in use without changing thresholds to force agreement. A proposed model with
fixed t0 is future scientific work, not an implemented or validated change.

See the [Spanish explanation of compatibility, tau/RoR and remaining FP](matlab_parity/compatibility_scope.md) and the [solver investigation](matlab_parity/solver_investigation_20261007/report.md).
