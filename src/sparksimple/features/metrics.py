"""sparksimple.features.metrics — Spark parameter metrics calculation.

Ports MATLAB sparkParametersFast2.m:
- Peak intensity (AMP) & baseline (BL) calculation
- Spark amplitude (amp = AMP - BL)
- Rate of Rise (ror) from uprise linear fit
- Time-to-peak (t2p) from uprise linear baseline intercept
- Full Duration at Half Maximum (FDHM)
- Decay time constant (tau) & R² from bisquare exponential fit (solver-dependent)
- Spark mass (sum(T), ΔF/F₀·frame; no BL subtraction or DT factor)
"""

from __future__ import annotations

import numpy as np


def _exp_decay_model(x_val, a, b, e, f):
    return a * np.exp(-b * (x_val - e)) + f


def _tau_from_params(params: np.ndarray) -> float:
    b_fit = float(params[1])
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.divide(1.0, b_fit))


def _select_decay_fit(x, y, p0, lower, upper):
    from sparksimple.features.robust import bisquare_fit, LocalFitFailure
    try:
        fit = bisquare_fit(_exp_decay_model, x, y, p0, lower, upper)
    except LocalFitFailure as exc:
        return {"tau": np.nan, "r2": 0.0, "tau_fit_method": "invalid",
                "tau_fit_fallback_reason": "local_decay_fit_failure:" + str(exc)}
    params = fit.params
    return {
        "tau": _tau_from_params(params), "r2": fit.r2,
        "tau_fit_method": "robust_bisquare", "tau_fit_fallback_reason": "none" if fit.converged else "solver_budget_exhausted",
        "tau_lsq": _tau_from_params(fit.lsq_params), "r2_lsq": fit.lsq_r2,
        "tau_robust": _tau_from_params(params), "r2_robust_unweighted": fit.unweighted_r2,
        "decay_points": float(len(y)), "decay_start_ms": float(x[0]), "decay_end_ms": float(x[-1]),
        "decay_a": float(params[0]), "decay_e": float(params[2]), "decay_f": float(params[3]),
    }


def compute_trace_parameters(
    cut: np.ndarray,
    t_axis_ms: np.ndarray,
    peak_idx: int,
) -> dict[str, float]:
    """Compute spark parameters matching MATLAB ``sparkParametersFast2.m``.

    Parameters
    ----------
    cut : np.ndarray
        1D float64 snippet trace.
    t_axis_ms : np.ndarray
        1D float64 time axis in milliseconds corresponding to ``cut``.
    peak_idx : int
        0-based index of candidate peak frame in ``cut``.

    Returns
    -------
    params : dict[str, float]
        Dictionary containing:
        - 'amp': float (amplitude above baseline, AMP - BL)
        - 'tau': float (decay time constant in ms, or NaN)
        - 'ror': float (rate of rise slope in ΔF/F₀ per ms)
        - 't2p': float (time-to-peak in ms)
        - 'fdhm': float (FDHM duration in ms, or NaN)
        - 'AMP': float (peak intensity value)
        - 'BL': float (baseline level before spark)
        - 'mass': float (sum of cut signal)
        - 'r2': float (R² of decay exponential fit, or NaN)
    """
    cut_arr = np.asarray(cut, dtype=np.float64)
    t_arr = np.asarray(t_axis_ms, dtype=np.float64)
    n_cut = len(cut_arr)

    res: dict[str, float] = {
        "amp": np.nan,
        "tau": np.nan,
        "ror": np.nan,
        "t2p": np.nan,
        "fdhm": np.nan,
        "AMP": np.nan,
        "BL": np.nan,
        "mass": float(np.sum(cut_arr)),
        "r2": np.nan,
        "tau_fit_method": "not_attempted",
        "tau_fit_fallback_reason": "insufficient_decay_input",
        "tau_lsq": np.nan,
        "r2_lsq": np.nan,
        "tau_robust": np.nan,
        "r2_robust_unweighted": np.nan,
        "decay_points": 0.0,
        "decay_start_ms": np.nan,
        "decay_end_ms": np.nan,
        "peak_idx_cut": 0,
        "decay_a": np.nan,
        "decay_e": np.nan,
        "decay_f": np.nan,
    }

    if n_cut < 3:
        return res

    # MATLAB receives c1-ot as a 1-based centre: exclude the detection frame.
    if peak_idx <= 0:
        res.update(r2=np.nan, mass=np.nan, tau_fit_method="invalid",
                   tau_fit_fallback_reason="matlab_empty_prepeak_window")
        return res
    centre_idx = int(np.argmax(cut_arr[:peak_idx]))

    # 2. Uprise boundary search aux1 (MATLAB lines 23–25)
    aux1 = centre_idx
    if aux1 >= 9:
        peak_val = cut_arr[centre_idx]
        above_85 = np.where(cut_arr[: centre_idx + 1] > peak_val * 0.85)[0]
        if len(above_85) > 0:
            aux1 = int(above_85[-1])

    uprise = cut_arr[: aux1 + 1]
    ori = 0
    if len(uprise) > 0 and uprise[0] == np.min(uprise):
        ori = 1 if len(uprise) > 1 else 0

    # 3. Fit scanning for uprise transition point aux (MATLAB lines 27–44)
    best_aux = 0
    if aux1 > ori:
        best_dist = 9999.0
        for ii in range(ori, aux1):
            t1 = uprise[: ii + 1]
            x_fit = np.arange(ii, centre_idx + 1, dtype=np.float64)
            y_fit = cut_arr[ii : centre_idx + 1]

            if len(x_fit) >= 2:
                poly = np.polyfit(x_fit, y_fit, 1)
                t3_0 = poly[0] * float(ii) + poly[1]
                dist = abs(float(np.mean(t1)) - t3_0)
            else:
                dist = 1e9

            if dist < best_dist:
                best_dist = dist
                best_aux = ii

    aux = best_aux

    # 4. Linear uprise fit using real time axis x (MATLAB lines 46–56)
    x1 = t_arr[aux : aux1 + 1]
    T1 = cut_arr[aux : aux1 + 1]

    if len(x1) >= 2:
        poly_uprise = np.polyfit(x1, T1, 1)
        ror = float(poly_uprise[0])
        intercept = float(poly_uprise[1])
    else:
        ror = 0.0
        intercept = float(cut_arr[aux])

    # Baseline & Amplitude (MATLAB lines 61–65)
    search_start = centre_idx - 3
    search_end = centre_idx + 9
    if search_start < 0 or search_end > n_cut:
        # MATLAB's local try exits here, preserving mass and r2=0.
        res.update(r2=0.0, tau_fit_method="invalid",
                   tau_fit_fallback_reason=f"matlab_invalid_peak_window:{search_start+1}:{search_end}")
        return res
    rel_max_idx = int(np.argmax(cut_arr[search_start:search_end]))
    peak_idx = search_start + rel_max_idx
    res["peak_idx_cut"] = peak_idx

    bline = float(np.mean(cut_arr[: aux + 1]))
    AMP = float(cut_arr[peak_idx])
    amp = AMP - bline

    res["AMP"] = AMP
    res["BL"] = bline
    res["amp"] = amp
    res["ror"] = ror

    # Time to Peak (MATLAB lines 69–73)
    with np.errstate(divide="ignore", invalid="ignore"):
        xb = np.divide(bline - intercept, ror)
    res["t2p"] = float(t_arr[peak_idx] - xb)

    # 5. Full Duration at Half Maximum (FDHM) (MATLAB lines 80–84)
    half_level = bline + amp / 2.0
    after_peak = np.where(cut_arr[peak_idx:] < half_level)[0]
    before_peak = np.where(cut_arr[: peak_idx + 1] < half_level)[0]

    if len(after_peak) > 0 and len(before_peak) > 0:
        idx_after = peak_idx + int(after_peak[0])
        idx_before = int(before_peak[-1])
        res["fdhm"] = float(t_arr[idx_after] - t_arr[idx_before])
    else:
        res["fdhm"] = np.nan

    # 6. Exponential Decay Fit (Tau & R²) (MATLAB lines 86–110)
    caiguda = cut_arr[peak_idx:]
    xx = t_arr[peak_idx:]

    if len(caiguda) >= 3:
        start_a = max(abs(caiguda))
        start_b = 0.0
        start_e = float(np.min(xx))
        start_f = float(np.min(caiguda))

        p0 = [start_a, start_b, start_e, start_f]
        lower = [0.0, 0.0, 0.99 * float(np.min(xx)), float(np.min(caiguda))]
        upper = [10.0 * max(abs(caiguda)), 1.0, float(np.max(xx)), float(np.max(caiguda))]

        fit = _select_decay_fit(xx, caiguda, p0=p0, lower=lower, upper=upper)
        res.update({key: value for key, value in fit.items() if key != "fitted"})
    else:
        res["tau"] = np.nan
        res["r2"] = np.nan

    return res
