"""
Shared pytest fixtures for SparkSimple2 Python port tests.

Provides paths to the two reference datasets (Real and Simulated experiment)
and their corresponding MATLAB result folders, without assuming specific shapes
or hardcoded array sizes.
"""

from __future__ import annotations

import sys
import pytest
from pathlib import Path

# ── Test Data Directory Root ──────────────────────────────────────────────────
_DATA_ROOT = Path(__file__).parent / "data"
_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"

# Make the package importable when running pytest directly from the repo
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))


# ── Path Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def data_root() -> Path:
    """Root path of tests/data/."""
    return _DATA_ROOT


@pytest.fixture(scope="session")
def real_exp_dir() -> Path:
    """Input folder for the real experiment (multichannel TIFs + XML)."""
    return _DATA_ROOT / "Real_Experiment"


@pytest.fixture(scope="session")
def real_exp_matlab_result() -> Path:
    """Folder containing MATLAB outputs for the real experiment.

    Contains: zData0.mat, zData1.mat, zData2.mat, zMetaData.mat,
    yData_*.csv, zDataRejected_*.csv, extraFigs/, spkFeat/, roiFeat/, detFilm/.
    """
    return _DATA_ROOT / "Real_Experiment" / "Matlab_Result"


@pytest.fixture(scope="session")
def sim_exp_dir() -> Path:
    """Input folder for the simulated experiment (single-channel TIFs + metadata.xml)."""
    return _DATA_ROOT / "Simulated_Experiment"


@pytest.fixture(scope="session")
def sim_exp_matlab_result() -> Path:
    """Folder containing MATLAB outputs for the simulated experiment.

    Contains: zData0.mat, zData1.mat, zData2.mat, zMetaData.mat,
    yData_NOISE0.csv, zDataRejected_NOISE0.csv, zValidation.mat,
    extraFigs/, spkFeat/, roiFeat/, detFilm/.
    """
    return _DATA_ROOT / "Simulated_Experiment" / "Matlab_Result"


# ── Data Presence Skip Guards ─────────────────────────────────────────────────

@pytest.fixture(scope="session")
def require_real_data(real_exp_dir: Path):
    """Skip test if the real experiment folder does not exist or is empty."""
    if not real_exp_dir.exists() or not any(real_exp_dir.glob("*.tif")):
        pytest.skip(f"Real experiment data not found in {real_exp_dir}")
    return real_exp_dir


@pytest.fixture(scope="session")
def require_sim_data(sim_exp_dir: Path):
    """Skip test if the simulated experiment folder does not exist or is empty."""
    if not sim_exp_dir.exists() or not any(sim_exp_dir.glob("*.tif")):
        pytest.skip(f"Simulated experiment data not found in {sim_exp_dir}")
    return sim_exp_dir


@pytest.fixture
def matlab_reference_mfiles(tmp_path: Path):
    """Run opt-in MATLAB checks only from independent copied source."""
    import shutil
    source = Path(__file__).resolve().parents[2] / "mfiles"
    if not source.is_dir():
        pytest.skip(f"MATLAB source not found in {source}")
    target = tmp_path / "matlab_reference_mfiles"
    shutil.copytree(source, target)
    return target
