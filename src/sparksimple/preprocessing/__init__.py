"""sparksimple.preprocessing — Preprocessing pipeline (Cell Mask, Baseline & Normalisation).

Phases 3 and 4 of the SparkSimple2 Python port.
Ports sparkDetect.m lines 83–131 (first preprocessing block & cellMask4).

Public API
----------
    from sparksimple.preprocessing import (
        preprocess,
        PreprocessingResult,
        cell_mask4,
        apply_signal_sign,
        estimate_baseline_global,
        estimate_baseline_forced,
        normalise_df_f0,
        normalise_f_f0,
        normalise_volume,
        denormalise_df_f0,
    )
"""

from sparksimple.preprocessing.cell_mask import cell_mask4
from sparksimple.preprocessing.pipeline import preprocess, PreprocessingResult
from sparksimple.preprocessing.baseline import (
    apply_signal_sign,
    estimate_baseline_global,
    estimate_baseline_forced,
)
from sparksimple.preprocessing.normalise import (
    normalise_df_f0,
    normalise_f_f0,
    normalise_volume,
    denormalise_df_f0,
)

__all__ = [
    # Pipeline & Cell Mask
    "preprocess",
    "PreprocessingResult",
    "cell_mask4",
    # Baseline
    "apply_signal_sign",
    "estimate_baseline_global",
    "estimate_baseline_forced",
    # Normalisation
    "normalise_df_f0",
    "normalise_f_f0",
    "normalise_volume",
    "denormalise_df_f0",
]
