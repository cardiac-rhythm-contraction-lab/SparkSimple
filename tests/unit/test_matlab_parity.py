"""Discrete contracts and same-input MATLAB numeric oracle controls."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import scipy.ndimage as ndi
from sparksimple.config import DetectionConfig
from sparksimple.detection.local_max import regmax_3d, extract_candidates
from sparksimple.detection.wavelet import get_wavelet_scales
from sparksimple.features.metrics import compute_trace_parameters
from sparksimple.features.spatial import interp10, blobfit, talla_diametre
from sparksimple.roi.cluster import cluster_rois

ORACLE = Path(__file__).parents[1]/"fixtures"/"matlab_parity_controls.npz"


def test_matlab_empty_diameter_image_oracle():
    from sparksimple.features.spatial import measure_blob_fwhm
    with np.load(ORACLE) as oracle:
        actual = measure_blob_fwhm(oracle['empty_diameter_image'], .1)
        assert actual == pytest.approx(float(oracle['empty_diameter_fwhm']), abs=1e-12)


def test_empty_spatial_diameter_is_omitted_from_mean(monkeypatch):
    from sparksimple.features import spatial
    # MATLAB Cell2/N10 (808,69,299): the third diameter is [].
    profiles = iter([30.4, .1, None, .1])
    monkeypatch.setattr(spatial, 'blobfit', lambda profile: profile)
    def diameter(profile, height):
        value = next(profiles)
        if value is None:
            raise spatial.EmptyDiameter('no_strict_halfheight_crossing')
        return value
    monkeypatch.setattr(spatial, 'talla_diametre', diameter)
    assert spatial.measure_blob_fwhm(np.ones((47,47)), .1) == pytest.approx(1.0213807118745768)


def test_all_empty_spatial_diameters_give_nan(monkeypatch):
    from sparksimple.features import spatial
    monkeypatch.setattr(spatial, 'blobfit', lambda profile: profile)
    def diameter(profile, height):
        raise spatial.EmptyDiameter('no_strict_halfheight_crossing')
    monkeypatch.setattr(spatial, 'talla_diametre', diameter)
    assert np.isnan(spatial.measure_blob_fwhm(np.ones((47,47)), .1))


def test_simulated_tiff_uses_frame_scales():
    np.testing.assert_array_equal(get_wavelet_scales(10,is_simulated=True),np.arange(5,11))
    np.testing.assert_array_equal(get_wavelet_scales(10,input_format="ata"),np.arange(2,6))
    assert DetectionConfig().temporal_split_frac == 0


def slow_matlab_regmax(v,cap):
    # Independent literal reference: Fortran linear scan, full-component reads.
    ip=np.zeros_like(v); count=0
    for level,threshold in zip(range(51,0,-1),np.linspace(v.min(),v.max(),51)[::-1]):
        labels,n=ndi.label(v.transpose(2,1,0)>threshold,np.ones((3,3,3)))
        labels=labels.transpose(2,1,0)
        for label in range(1,n+1):
            pixels=labels==label
            if np.unique(ip[pixels]).size==1:
                coords=np.column_stack(np.where(pixels))+1
                center=np.floor(coords.mean(axis=0)+.5).astype(int)-1
                ip[tuple(center)]=level;count+=1
        if count>=cap:break
    return ip


@pytest.mark.parametrize("seed",[0,1,2,3])
def test_regmax_full_ix_matches_literal_reference(seed):
    rng=np.random.default_rng(seed)
    v=np.round(rng.uniform(0,1,(8,9,10)),1);v[v<.6]=0
    np.testing.assert_array_equal(regmax_3d(v,max_candidates=3),slow_matlab_regmax(v,3))


def test_regmax_cap_is_checked_after_all_components_in_level():
    v=np.zeros((9,9,9));v[1,7,1]=v[7,1,1]=v[4,4,7]=10
    ix=regmax_3d(v,max_candidates=1)
    assert np.count_nonzero(ix)==3
    c=extract_candidates(ix)
    assert [(s.x_0b,s.y_0b,s.t_0b) for s in c]==[(1,7,1),(7,1,1),(4,4,7)]


@pytest.mark.parametrize("index",[0,1,2,3,4,5,6,7,8,9])
def test_temporal_algebra_on_audited_matlab_trace(index):
    with np.load(ORACLE) as o:
        p=compute_trace_parameters(o[f'trace_{index}'],o[f'axis_{index}'],int(o['centers'][index]))
        ref=o['values'][index]
        # The solver's tau/r2 are evaluated separately; algebra must be exact.
        actual=[p['amp'],p['ror'],p['t2p'],p['fdhm'],p['AMP'],p['mass']]
        np.testing.assert_allclose(actual,ref[[0,2,3,4,5,6]],atol=1e-10,rtol=0,equal_nan=True)
        if index in (1,3):
            assert p['r2']==0
            assert p['tau_fit_fallback_reason'].startswith('matlab_invalid_peak_window:')
        if index==0:
            assert p['AMP']<.3 # MATLAB re-centres on the early fluctuation (O-4 removed).


@pytest.mark.parametrize("index",[0,1,2])
def test_interp10_matches_matlab_filter_and_grid(index):
    with np.load(ORACLE) as o:
        np.testing.assert_allclose(interp10(o[f'interp_in_{index}']),o[f'interp_out_{index}'],atol=2e-11,rtol=0)


@pytest.mark.parametrize("index",[0,1,2,3,4])
def test_gaussian_bisquare_on_same_matlab_profile(index):
    with np.load(ORACLE) as o:
        np.testing.assert_allclose(blobfit(o[f'profile_{index}']),o[f'fit_{index}'],atol=1.2e-7,rtol=0)
        if index==4:
            from sparksimple.features.robust import LocalFitFailure
            with pytest.raises(LocalFitFailure,match='crossing'): talla_diametre(blobfit(o[f'profile_{index}']))
            with pytest.raises(LocalFitFailure,match='crossing'): talla_diametre(o[f'fit_{index}'])
        else:
            assert talla_diametre(blobfit(o[f'profile_{index}']))==talla_diametre(o[f'fit_{index}'])


def test_roi_half_pixel_rounding_changes_center_and_signal():
    sparks=[SimpleNamespace(px=5,py=5,good=True),SimpleNamespace(px=6,py=6,good=True)]
    y,x,t=np.indices((12,12,3));volume=10*y+x+t
    res=cluster_rois(sparks,volume,np.ones((12,12)),1,10,2)
    assert (res.rois[0].px,res.rois[0].py)==(6,6)
    np.testing.assert_array_equal(res.rois[0].signal,np.array([55.,56.,57.]))


def test_unexpected_fit_failure_propagates(monkeypatch):
    import sparksimple.features.robust as robust
    def broken(*args,**kwargs):
        raise RuntimeError("unexpected solver bug")
    monkeypatch.setattr(robust,"bisquare_fit",broken)
    with np.load(ORACLE) as o:
        with pytest.raises(RuntimeError,match="unexpected"):
            compute_trace_parameters(o['trace_2'],o['axis_2'],int(o['centers'][2]))


def test_fwhm_default_roi_is_47_pixels(monkeypatch):
    import sparksimple.features.extract as extraction
    shapes=[]
    def capture(im,dx_um):
        shapes.append(im.shape);return 1.0
    monkeypatch.setattr(extraction,"measure_blob_fwhm",capture)
    volume=np.ones((70,70,60));volume[30:40,30:40,20:30]=2.
    ix=np.zeros_like(volume);ix[35,35,25]=50
    extraction.extract_features(volume,ix,np.ones((70,70)),.1,10)
    assert shapes==[(47,47)] # MATLAB round(1.5*1.5/.1)=23.
