"""Bisquare IRLS controls used by Curve Fitting Toolbox.

Implements fixed initial leverage, trimmed absolute-residual scale and robust
MSE (DuMouchel/O'Brien). SciPy TRF is the bounded inner solver; its iterates
are not claimed identical to MATLAB's trust-region implementation.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import qr
from scipy.optimize import least_squares


class LocalFitFailure(RuntimeError):
    """Expected numerical failure of a candidate's bounded fit."""


@dataclass
class RobustFit:
    params: np.ndarray
    weights: np.ndarray
    leverage: np.ndarray
    r2: float
    unweighted_r2: float
    lsq_params: np.ndarray
    lsq_r2: float
    evaluations: int
    converged: bool


def bisquare_weights(u):
    return np.where(np.abs(u) < 1, (1 - u*u)**2, 0.0)


def robust_r2(y, residuals, weights, leverage, scale, initial_sse, p):
    """MATLAB robust MSE shrinkage, divided by ordinary response SST."""
    n = len(y)
    if n <= p:
        return np.nan
    if np.all((weights < 1e-6) | (weights > 1-1e-6)):
        included = weights > 1-1e-6
        denom = np.count_nonzero(included)-p
        mse = np.sum(residuals[included]**2)/denom if denom > 0 else np.nan
    else:
        u = residuals / np.sqrt(1-leverage) / (scale*4.685)
        phi = u * bisquare_weights(u)
        delta = 1e-4
        dphi = ((u+delta)*bisquare_weights(u+delta)
                - (u-delta)*bisquare_weights(u-delta))/(2*delta)
        m1 = np.mean(dphi)
        m2 = np.sum((1-leverage)*phi**2)/(n-p)
        k = 1+(p/n)*(1-m1)/m1
        mse = (k*np.sqrt(m2)*scale*4.685/m1)**2
    mse = max(mse, (n*mse+p*p*initial_sse/(n-p))/(n+p*p))
    sst = np.sum((y-np.mean(y))**2)
    return float(1-mse*(n-p)/sst) if sst else np.nan


def bisquare_fit(model, x, y, start, lower, upper):
    x, y = np.asarray(x, float), np.asarray(y, float)
    start, lower, upper = map(lambda a: np.asarray(a, float), (start, lower, upper))
    p, n = len(start), len(y)
    if n <= p or not np.all(np.isfinite(y)) or np.any(lower >= upper):
        raise LocalFitFailure("invalid_fit_input_or_bounds")
    def residual(params):
        return model(x, *params)-y
    def solve(params, weights, budget):
        try:
            fit = least_squares(lambda a: np.sqrt(weights)*residual(a), params,
                                bounds=(lower, upper), method="trf",
                                ftol=1e-6, xtol=1e-6, gtol=1e-6,
                                max_nfev=max(1,budget))
        except (ValueError, FloatingPointError) as exc:
            raise LocalFitFailure(str(exc)) from exc
        if not np.all(np.isfinite(fit.x)):
            raise LocalFitFailure("nonfinite_fit_parameters")
        return fit
    # MATLAB defaults: MaxIter=400, MaxFunEvals=600, TolFun=TolX=1e-6.
    # SciPy nfev excludes finite difference probes; report this limitation.
    fit = solve(start, np.ones(n), 600)
    initial = fit.x.copy()
    initial_sse = float(np.sum(residual(initial)**2))
    q, _, _ = qr(fit.jac, mode="economic", pivoting=True)
    h = np.minimum(0.9999, np.sum(q*q, axis=1))
    tiny = 1e-6*np.std(y, ddof=1)/n or 1.0
    evaluations = fit.nfev
    converged = False
    for _ in range(400):
        adjusted = residual(fit.x)/np.sqrt(1-h)
        scale = max(tiny, float(np.median(np.sort(np.abs(adjusted))[p-1:])/0.6745))
        weights = bisquare_weights(adjusted/(4.685*scale))
        previous = fit.x.copy()
        fit = solve(previous, weights, 600-evaluations)
        evaluations += fit.nfev
        converged = bool(np.all(np.abs(fit.x-previous) <= 1e-6*np.maximum(np.abs(fit.x),np.abs(previous))))
        if converged or evaluations >= 600 or fit.status == 0:
            break
    res = residual(fit.x)
    sst = np.sum((y-np.mean(y))**2)
    return RobustFit(fit.x, weights, h,
                     robust_r2(y,res,weights,h,scale,initial_sse,p),
                     float(1-np.sum(res**2)/sst) if sst else np.nan,
                     initial, float(1-initial_sse/sst) if sst else np.nan,
                     evaluations, converged)
