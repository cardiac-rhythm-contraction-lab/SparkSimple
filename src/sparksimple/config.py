"""sparksimple.config — Pipeline configuration dataclasses.

All parameters are documented with their MATLAB counterpart and the script
where they appear.  Default values are taken directly from MATLAB source;
parameters that vary per-experiment or that MATLAB reads interactively are
left as None and must be set explicitly.

MATLAB reference files
----------------------
- sparkDetect.m          (detection + baseline)
- sparkFiltSimple.m      (physiological filtering thresholds)
- SparkSimple2.m         (entry point — top-level parameter passing)

Usage
-----
    cfg = PipelineConfig()                         # all MATLAB defaults
    cfg = PipelineConfig(
        detection=DetectionConfig(signal_sign=-1), # inverted signal
        filtering=FilteringConfig(min_amplitude=0.15),
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ── Detection ─────────────────────────────────────────────────────────────────

@dataclass
class DetectionConfig:
    """Parameters for volume building, baseline and wavelet detection.

    Attributes
    ----------
    spatial_smooth_um : float | None
        Gaussian smoothing width applied to each frame before the volume is
        built, in micrometres. Applied before the cell mask, matching MATLAB.
        MATLAB: ``fs`` in sparkDetect.m (lines 63–65 convert µm to an odd pixel
        kernel via DX; line 76 applies ``conv2(Im, gaussiana2d(fs), 'same')``).
        ``SparkSimple2.m`` line 15 ships ``fs = 0``.
        Note that MATLAB's ``if(fs>0)`` guard is always true, because
        ``fs=fs+mod(fs,2)+1`` forces ``fs >= 1``. What makes ``fs = 0`` a no-op
        is the resulting 1×1 unit kernel, not the guard.
        Requires ``dx_um`` to be passed to ``preprocess()`` when set.
        Default: None — no smoothing, equivalent to the MATLAB default.

    filter_window_frames : int
        Temporal smoothing window applied to each pixel trace before the CWT,
        in frames.
        MATLAB: ``ft = 1`` passed from SparkSimple2.m line 14 as ``filtime`` argument;
        sparkDetect.m uses ``w = filtime; if(w==0), w=1; end``.
        The old commented-out default of filtime=40 was in **milliseconds** and is no
        longer used.  Current SparkSimple2.m default is 1 frame (no smoothing).
        Default: 1

    filter_window_ms : float (Deprecated)
        Deprecated alias for `filter_window_frames`.
    wavelet_scales_real : tuple[int, int]
        Min and max CWT scales (inclusive) for TIFF data, regardless of simulated origin.
        MATLAB: ``scala = [5:10]`` (sparkDetect.m line 142).
        Default: (5, 10)

    wavelet_scales_sim_dt_range_ms : tuple[float, float]
        Time range (ms) used to derive CWT scales for explicit .ata / point-scanned
        data as ``scala = [ceil(lo/DT) : ceil(hi/DT)]``.
        MATLAB: ``scala = [ceil(12/DT) : ceil(48/DT)]`` (sparkDetect.m line 140).
        Default: (12.0, 48.0)

    merge_distance_um : float | None
        Spatial radius (µm) for 3-D convolution kernel used to merge nearby
        candidates after thresholding.
        MATLAB: ``dm`` passed as argument to sparkDetect.m (line 174).
        SparkSimple2.m default: 3 µm.
        Default: 3.0

    merge_time_ms : float | None
        Temporal half-window (ms) for the 3-D merge kernel.
        MATLAB: ``tm`` passed as argument (sparkDetect.m line 175).
        SparkSimple2.m default: 30 ms.
        Default: 30.0

    noise_connectivity : int
        End-of-level candidate stopping target for RegMax3D (``tope`` / ``nc``
        in MATLAB RegMax3D.m).  Despite the field name, this is NOT the 3-D
        voxel connectivity (which is always 26-connected in both MATLAB ``bwlabeln``
        and Python ``scipy.ndimage.label``); it controls the stopping target checked after a whole level.
        Counts may exceed it. The default temporal split is disabled.
        MATLAB: ``nc`` passed as second argument to RegMax3D (sparkDetect.m line 199).
        SparkSimple2.m default: 100.
        Default: 100

    signal_sign : int
        Multiply the raw volume by this value before processing.
        Use +1 for normal Ca²⁺ signals, -1 for inverted (ratiometric) signals.
        MATLAB: ``ss`` passed as argument (sparkDetect.m line 94).
        Default: 1

    baseline_quantile : float
        Quantile of the in-mask pixel intensities used as the global baseline
        estimate Fo when no forced baseline is provided.
        MATLAB: ``quantile(vk, .05)`` (sparkDetect.m line 114).
        Default: 0.05

    normalisation_mode : str
        ``"df_f0"``  → (F - Fo) / Fo  (optB=1, MATLAB default)
        ``"f_f0"``   → F / Fo          (optB=0)
        MATLAB: ``optB = 1`` (sparkDetect.m line 3).
        Default: "df_f0"
    """

    spatial_smooth_um: float | None = None
    filter_window_frames: int = 1
    wavelet_scales_real: tuple[int, int] = (5, 10)
    wavelet_scales_sim_dt_range_ms: tuple[float, float] = (12.0, 48.0)
    merge_distance_um: float = 3.0
    merge_time_ms: float = 30.0
    noise_connectivity: int = 100
    temporal_split_frac: float = 0.0
    temporal_split_min_frames: int = 6
    signal_sign: int = 1
    baseline_quantile: float = 0.05
    normalisation_mode: str = "df_f0"

    def __init__(
        self,
        spatial_smooth_um: float | None = None,
        filter_window_frames: int | None = None,
        wavelet_scales_real: tuple[int, int] = (5, 10),
        wavelet_scales_sim_dt_range_ms: tuple[float, float] = (12.0, 48.0),
        merge_distance_um: float = 3.0,
        merge_time_ms: float = 30.0,
        noise_connectivity: int = 100,
        temporal_split_frac: float = 0.0,
        temporal_split_min_frames: int = 6,
        signal_sign: int = 1,
        baseline_quantile: float = 0.05,
        normalisation_mode: str = "df_f0",
        filter_window_ms: float | int | None = None,
    ) -> None:
        if filter_window_ms is not None:
            import warnings
            warnings.warn(
                "'filter_window_ms' is deprecated and will be removed in a future release. "
                "Use 'filter_window_frames' (temporal window in frames; effective default: 1 frame) instead. "
                "Milliseconds require conversion using the acquisition DT.",
                DeprecationWarning,
                stacklevel=2,
            )
            if filter_window_frames is None:
                filter_window_frames = int(round(filter_window_ms))

        if filter_window_frames is None:
            filter_window_frames = 1

        self.spatial_smooth_um = spatial_smooth_um
        self.filter_window_frames = int(filter_window_frames)
        self.wavelet_scales_real = wavelet_scales_real
        self.wavelet_scales_sim_dt_range_ms = wavelet_scales_sim_dt_range_ms
        self.merge_distance_um = float(merge_distance_um)
        self.merge_time_ms = float(merge_time_ms)
        self.noise_connectivity = noise_connectivity
        self.temporal_split_frac = float(temporal_split_frac)
        self.temporal_split_min_frames = int(temporal_split_min_frames)
        self.signal_sign = signal_sign
        self.baseline_quantile = baseline_quantile
        self.normalisation_mode = normalisation_mode

        self.__post_init__()

    @property
    def filter_window_ms(self) -> float:
        import warnings
        warnings.warn(
            "'filter_window_ms' property is deprecated. Use 'filter_window_frames' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return float(self.filter_window_frames)

    @filter_window_ms.setter
    def filter_window_ms(self, value: float | int) -> None:
        import warnings
        warnings.warn(
            "'filter_window_ms' property setter is deprecated. Use 'filter_window_frames' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.filter_window_frames = int(round(value))

    def __post_init__(self) -> None:
        if self.normalisation_mode not in ("df_f0", "f_f0"):
            raise ValueError(
                f"normalisation_mode must be 'df_f0' or 'f_f0', "
                f"got {self.normalisation_mode!r}"
            )
        if self.signal_sign not in (1, -1):
            raise ValueError(
                f"signal_sign must be 1 or -1, got {self.signal_sign}"
            )
        if not (0.0 < self.baseline_quantile < 1.0):
            raise ValueError(
                f"baseline_quantile must be in (0, 1), got {self.baseline_quantile}"
            )
        if self.noise_connectivity < 1:
            raise ValueError("noise_connectivity (NC) must be at least 1")
        if not (0.0 <= self.temporal_split_frac <= 1.0):
            raise ValueError("temporal_split_frac must be in [0, 1]")
        if self.temporal_split_min_frames < 1:
            raise ValueError("temporal_split_min_frames must be at least 1")


# ── Filtering ─────────────────────────────────────────────────────────────────

@dataclass
class FilteringConfig:
    """Physiological acceptance thresholds applied after feature extraction.

    Mirrors the nested ``if`` chain in sparkFiltSimple.m (lines 70–121).
    Each threshold has a corresponding ``fail`` reason string shown in the
    MATLAB output and stored in ``spkF(ii).fail``.

    Attributes
    ----------
    min_amplitude : float | None
        Minimum ΔF/F₀ amplitude.
        MATLAB: ``iT``, checked as ``spkF(ii).amp >= iT`` (line 72).
        Default: None — value varies per experiment; must be set explicitly.

    max_baseline : float | None
        Maximum allowed pre-spark baseline (ΔF/F₀).
        MATLAB: ``Bl``, checked as ``spkF(ii).BL <= Bl`` (line 71).
        Default: None — idem.

    min_decay_r2 : float | None
        Minimum R² of the exponential decay fit.
        MATLAB: ``r2``, checked as ``spkF(ii).r2 >= r2`` (line 73).
        Default: None — idem.

    t2p_range_ms : tuple[float, float] | None
        Allowed range for time-to-peak (ms).
        MATLAB: ``t2``, checked as ``t2(1) <= t2p <= t2(2)`` (lines 74–75).
        Default: None — idem.

    tau_range_ms : tuple[float, float] | None
        Allowed range for decay time constant τ (ms).
        MATLAB: ``ta``, checked as ``ta(1) <= tau <= ta(2)`` (lines 76–77).
        Default: None — idem.

    fdhm_range_ms : tuple[float, float] | None
        Allowed range for FDHM — Full Duration at Half Maximum (ms).
        MATLAB: ``Fd``, checked as ``Fd(1) <= FDHM <= Fd(2)`` (lines 78–79).
        Default: None — idem.
    """

    min_amplitude: float | None = None
    max_baseline: float | None = None
    min_decay_r2: float | None = None
    t2p_range_ms: tuple[float, float] | None = None
    tau_range_ms: tuple[float, float] | None = None
    fdhm_range_ms: tuple[float, float] | None = None

    @classmethod
    def from_matlab_defaults(cls) -> FilteringConfig:
        """Return a FilteringConfig populated with standard SparkSimple2.m defaults.

        Threshold values (SparkSimple2.m):
        - min_amplitude (iT): 0.3
        - max_baseline (Bl): 1.0
        - min_decay_r2 (r2): 0.3
        - t2p_range_ms (t2): (0.0, 200.0)
        - tau_range_ms (ta): (5.0, 500.0)
        - fdhm_range_ms (Fd): (10.0, 200.0)
        """
        return cls(
            min_amplitude=0.3,
            max_baseline=1.0,
            min_decay_r2=0.3,
            t2p_range_ms=(0.0, 200.0),
            tau_range_ms=(5.0, 500.0),
            fdhm_range_ms=(10.0, 200.0),
        )



# ── ROI ───────────────────────────────────────────────────────────────────────

@dataclass
class ROIConfig:
    """Parameters for ROI clustering and per-ROI signal extraction.

    Attributes
    ----------
    roi_radius_um : float
        Radius of the region of interest around each detected spark centroid,
        in micrometres.
        MATLAB: ``roiR = 1.5`` passed to sparkDetect.m (line 19) and saved in zMetaData.mat.
        Default: 1.5 µm.
    """

    roi_radius_um: float = 1.5
    # MATLAB: clusterROIs.m calls `linkage(pd)` without a method argument, and
    # MATLAB's linkage() defaults to 'single' (nearest-neighbour), not 'average'.
    # Confirmed empirically: 'single' at this cutoff reproduces MATLAB's ROI
    # partition exactly (ARI=1.0) on real validation data; 'average' does not.
    linkage: str = "single"

    def __post_init__(self) -> None:
        if self.linkage not in ("average", "complete", "single"):
            raise ValueError("ROI linkage must be one of: average, complete, single")

    @property
    def roi_r_um(self) -> float:
        return self.roi_radius_um


# ── Top-level pipeline config ─────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """Top-level configuration object for the full SparkSimple2 pipeline.

    Groups all sub-configs and adds pipeline-wide options.

    Attributes
    ----------
    detection : DetectionConfig
    filtering : FilteringConfig
    roi : ROIConfig
    baseline_mode : str
        ``"auto"``    — estimate Fo from data (MATLAB default; ``forcedbaseline`` is scalar 0).
        ``"forced"``  — use a pre-computed Fo map.
        MATLAB: ``forcedbaseline`` argument to sparkDetect.m; scalar 0 triggers auto.
    forced_baseline : float | None
        Value used when ``baseline_mode == "forced"``.
        Must not be None if ``baseline_mode == "forced"``.
    channel : int | str | None
        0-indexed channel to use, or a MATLAB-style substring tag (e.g.
        ``"ch01"``) that pre-filters filenames like SparkSimple2.m's
        ``tagSpk``. ``None`` is valid only for single-channel input;
        multi-channel experiments require an explicit channel.

    Example
    -------
    >>> cfg = PipelineConfig()
    >>> cfg.detection.filter_window_frames
    1
    >>> cfg.filtering.min_amplitude is None  # must be set per-experiment
    True
    """

    detection: DetectionConfig = field(default_factory=DetectionConfig)
    filtering: FilteringConfig = field(default_factory=FilteringConfig.from_matlab_defaults)
    roi: ROIConfig = field(default_factory=ROIConfig)
    baseline_mode: str = "auto"
    forced_baseline: float | None = None
    channel: int | str | None = None

    def __post_init__(self) -> None:
        if self.baseline_mode not in ("auto", "forced"):
            raise ValueError(
                f"baseline_mode must be 'auto' or 'forced', "
                f"got {self.baseline_mode!r}"
            )
        if self.baseline_mode == "forced" and self.forced_baseline is None:
            raise ValueError(
                "forced_baseline must be set when baseline_mode == 'forced'"
            )
        if isinstance(self.channel, int) and self.channel < 0:
            raise ValueError(
                f"channel must be a non-negative integer or a str tag, got {self.channel}"
            )
