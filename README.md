# SparkSimple

Automated detection and characterisation of Ca²⁺ sparks in confocal
**frame-scan (x-y-t)** image series of cardiac myocytes.

SparkSimple is the Python implementation of the MATLAB SparkSimple2 pipeline
developed at the Cardiac Rhythm & Contraction Lab. This release (`v2.1.0`)
reproduces the detection logic of SparkSimple2 (MATLAB).

## What it does

Given a folder of Ca²⁺ image frames (TIFF) and an XML metadata file:

1. Estimates the baseline fluorescence (F₀) and computes ΔF/F₀.
2. Segments the cell mask.
3. Detects candidates (normalisation, continuous wavelet transform, 3-D
   spatio-temporal fusion and thresholding, 3-D regional maxima).
4. Measures each candidate: amplitude (ΔF/F₀), FWHM, FDHM, time to peak, rate of
   rise, decay τ, mass and distance to the membrane.
5. Applies physiological filtering and groups accepted sparks into recurrent
   spark sites (ROIs).
6. Exports CSV tables, MATLAB-compatible `.mat` files and an HTML report.

## Installation

```bash
git clone https://github.com/cardiac-rhythm-contraction-lab/SparkSimple.git
cd SparkSimple
pip install -e .
```

Requirements: Python ≥ 3.10, NumPy, SciPy, tifffile, matplotlib, Flask.

## Quick start

```bash
sparksimple run /path/to/experiment/
```

The experiment folder must contain the TIFF frames (one file per frame or a
multi-frame TIFF) and a `metadata.xml` with pixel size and frame interval
(simple `Dimensions` XML or Leica LAS-AF XML). Results are written to
`<experiment>/Python_Result/` (change it with `--output-dir`):

| File | Contents |
|------|----------|
| `yData_<title>.csv` | Accepted sparks and their features |
| `zDataRejected_<title>.csv` | Rejected candidates |
| `zData2.mat`, `zMetaData.mat` | MATLAB-compatible export |
| `run_metadata.json` | Parameters and software version of the run |
| `<title>.html` | Interactive report (only with `--report`) |

`<title>` defaults to the name of the experiment folder (`--title` to change it).

Useful options (`sparksimple run --help` lists all of them):

| Option | Meaning |
|--------|---------|
| `--report` | Also write the HTML report and per-spark figures |
| `--film` | Also write one detection image per frame (implies `--report`) |
| `--channel` | Channel to analyse in multi-channel TIFFs |
| `--dx-um`, `--dt-ms` | Override pixel size (µm) and frame interval (ms) from the metadata |
| `--nc` | Maximum number of candidates per film (default 100). If the run warns that the limit was reached, rerun with a higher value (e.g. `--nc 300`) |

Python API (returns the results in memory; it does not write files):

```python
from sparksimple import run_pipeline

result = run_pipeline("/path/to/experiment/")
print(result.filtering.accepted_count, "sparks accepted")
```

Other commands:

| Command | Purpose |
|---------|---------|
| `sparksimple merge-only` | Merge `zData2.mat` results from several experiments and re-cluster ROIs |
| `sparksimple validate` | Browser-based tool for manual spark validation |
| `sparksimple modify-d2m` | Browser-based tool to correct distance-to-membrane values |

Full parameter and output reference: [docs/user_guide.md](docs/user_guide.md).

## Tests

```bash
pip install -e ".[dev]"
pytest tests/unit -q
```

## Citation

If you use SparkSimple, please cite the software (see
[CITATION.cff](CITATION.cff)). A manuscript describing and validating the
method is in preparation; this section will be updated when it is published.

## License

GNU General Public License v3.0 or later — see [LICENSE](LICENSE).

## Contact

* Principal Investigator: Dr. Leif Hove-Madsen ([@Leif-Hove](https://github.com/Leif-Hove)), Tenured Scientist, IIBB-CSIC — [leif.hove@iibb.csic.es](mailto:leif.hove@iibb.csic.es)
* PhD Student in Biomedical Engineering: Gabriel Manzano Reche ([@GabriManz](https://github.com/GabriManz)) — [gabriel.manzano@iibb.csic.es](mailto:gabriel.manzano@iibb.csic.es)
