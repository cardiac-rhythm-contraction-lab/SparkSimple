"""tests/regression/test_regression_filtering.py — Parity regression tests for physiological filtering.

Compares Python filter_sparks outputs against MATLAB CLI results saved in
tests/data/Real_Experiment/Matlab_Result/zData2.mat.
"""

from pathlib import Path
import scipy.io as sio
import pytest

from sparksimple.config import FilteringConfig
from sparksimple.features.models import SparkFeature
from sparksimple.filtering import filter_sparks


REAL_DATA_DIR = Path(__file__).parents[1] / "data" / "Real_Experiment" / "Matlab_Result"
ZDATA2_PATH = REAL_DATA_DIR / "zData2.mat"
ZMETADATA_PATH = REAL_DATA_DIR / "zMetaData.mat"


@pytest.mark.skipif(not ZDATA2_PATH.exists(), reason="Real experiment zData2.mat not found")
def test_regression_physiological_filtering_parity():
    """Verify 100% exact parity against MATLAB sparkFiltSimple.m on real experiment dataset."""
    mat_data2 = sio.loadmat(str(ZDATA2_PATH))
    mat_meta = sio.loadmat(str(ZMETADATA_PATH))

    spkF_mat = mat_data2["spkF"].flat
    mask = mat_meta["mask"]
    fov_y, fov_x = mask.shape

    # Extract SparkFeature records from MATLAB spkF struct array
    candidates: list[SparkFeature] = []
    for s in spkF_mat:
        px = int(s["px"][0, 0])
        py = int(s["py"][0, 0])
        pt = int(s["pt"][0, 0])
        fwhm = float(s["FWHM"][0, 0])
        ts = s["timesignal"].ravel()
        ti = s["timeinterval"].ravel()
        r2 = float(s["r2"][0, 0])
        amp = float(s["amp"][0, 0])
        tau = float(s["tau"][0, 0])
        ror = float(s["ror"][0, 0])
        t2p = float(s["t2p"][0, 0])
        fdhm = float(s["FDHM"][0, 0])
        AMP = float(s["AMP"][0, 0])
        BL = float(s["BL"][0, 0])
        mass = float(s["mass"][0, 0])
        dist2memb = float(s["dist2memb"][0, 0])

        feat = SparkFeature(
            px=px,
            py=py,
            pt=pt,
            x_0b=px - 1,
            y_0b=py - 1,
            t_0b=pt - 1,
            dist2memb=dist2memb,
            fwhm=fwhm,
            timesignal=ts,
            timeinterval=ti,
            r2=r2,
            amp=amp,
            tau=tau,
            ror=ror,
            t2p=t2p,
            fdhm=fdhm,
            AMP=AMP,
            BL=BL,
            mass=mass,
        )
        candidates.append(feat)

    # Apply standard MATLAB thresholds from SparkSimple2.m
    config = FilteringConfig.from_matlab_defaults()

    # Execute Python filtering orchestrator
    result = filter_sparks(candidates, config, shape=(fov_y, fov_x))

    assert result.total_count == len(spkF_mat)

    # Compare candidates 1-by-1 against MATLAB output
    mismatches = []
    for ii, (py_rec, s_mat) in enumerate(zip(result.records, spkF_mat), start=1):
        mat_good = bool(s_mat["good"][0, 0])
        mat_fail = str(s_mat["fail"][0])

        if py_rec.good != mat_good or py_rec.fail != mat_fail:
            mismatches.append(
                f"Candidate #{ii}: Python (good={py_rec.good}, fail={py_rec.fail!r}) "
                f"vs MATLAB (good={mat_good}, fail={mat_fail!r})"
            )

    assert not mismatches, f"Filtering parity mismatches found:\n" + "\n".join(mismatches)
