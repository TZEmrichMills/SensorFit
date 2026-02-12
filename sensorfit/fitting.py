"""Fitting functions for sensor data models."""

import math
import numpy as np
from scipy.optimize import curve_fit

from .models import model_IB, model_Exponential, model_GFI, MODEL_FUNCS
from .utils import r2_score, aic, bic, initial_rate_numeric


def _basic_seeds(tt, yy):
    """Generate basic seed values from data."""
    C0 = float(np.median(yy[-max(3, len(yy) // 10) :]))
    amp = float(max(yy[0] - C0, 1e-6))
    dur = float(max(tt) - min(tt) + 1e-9)
    return C0, amp, dur


def fit_IB(tt, yy, maxfev=8000):
    """Fit Inactivation model to data with improved robustness."""
    if len(tt) < 5:
        raise ValueError("Need at least 5 data points for Inactivation fit")
    if len(tt) != len(yy):
        raise ValueError("Time and signal arrays must have same length")
    
    C0, amp, dur = _basic_seeds(tt, yy)
    
    # Try multiple initial parameter guesses
    initial_guesses = [
        [C0, amp, 0.5, 1.0 / max(dur * 0.5, 1e-3), max((yy[0] - yy[-1]) / max(dur, 1e-6) * 0.02, 0.0)],
        [C0, amp * 0.5, 0.3, 1.0 / max(dur * 0.3, 1e-3), max((yy[0] - yy[-1]) / max(dur, 1e-6) * 0.01, 0.0)],
        [C0, amp * 1.5, 0.7, 1.0 / max(dur * 0.7, 1e-3), max((yy[0] - yy[-1]) / max(dur, 1e-6) * 0.03, 0.0)],
    ]
    
    lb = [C0 - 10 * amp - 100, 0, 0, 0, 0]
    ub = [C0 + 10 * amp + 100, 10 * max(yy), 10, 10.0 / max(dur * 0.5, 1e-3), (yy[0] - min(yy)) / max(dur, 1e-6)]
    
    best_result = None
    best_rss = float('inf')
    last_error = None
    
    for p0 in initial_guesses:
        try:
            popt, pcov = curve_fit(model_IB, tt, yy, p0=p0, bounds=(lb, ub), method="trf", maxfev=maxfev)
            
            # Convert to numpy array and check validity
            popt = np.asarray(popt, dtype=float)
            if len(popt) == 0 or np.any(~np.isfinite(popt)):
                continue
            
            yhat = model_IB(tt, *popt)
            yhat = np.asarray(yhat, dtype=float)
            
            # Check for invalid values in fitted data
            if len(yhat) == 0 or np.any(~np.isfinite(yhat)):
                continue
            
            # Ensure arrays are same length
            if len(yhat) != len(yy):
                continue
            
            rss = float(np.sum((yy - yhat) ** 2))
            if not np.isfinite(rss):
                continue
                
            if rss < best_rss:
                best_rss = rss
                best_result = (popt.copy(), yhat.copy())
        except (RuntimeError, ValueError, TypeError, AttributeError, MemoryError) as e:
            last_error = e
            continue
    
    if best_result is None:
        raise RuntimeError(f"Inactivation fit failed with all initial guesses. Last error: {last_error}")
    
    popt, yhat = best_result
    
    # Ensure arrays are properly formatted and valid
    popt = np.asarray(popt, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    
    # Final validation
    if len(popt) == 0 or len(yhat) == 0 or len(yhat) != len(tt):
        raise RuntimeError("Inactivation fit produced invalid array dimensions")
    if np.any(~np.isfinite(popt)) or np.any(~np.isfinite(yhat)):
        raise RuntimeError("Inactivation fit produced invalid values (NaN/Inf)")
    
    rss = float(np.sum((yy - yhat) ** 2))
    n = len(tt)
    k = len(popt)
    
    return {
        "model": "IB",
        "params": popt,  # Keep as numpy array
        "names": MODEL_FUNCS["IB"][1],
        "yhat": yhat,  # Keep as numpy array
        "rss": rss,
        "r2": r2_score(yy, yhat),
        "aic": aic(n, rss, k),
        "bic": bic(n, rss, k),
        "H0_fit": float(model_IB([tt[0]], *popt)[0]),
        "init_rate": initial_rate_numeric(model_IB, popt, t_start=float(tt[0])),
    }


def fit_Exponential(tt, yy, maxfev=12000):
    """Fit exponential model: y(t) = a*t + b + c*exp(-k*(t-t0)) with improved robustness."""
    if len(tt) < 5:
        raise ValueError("Need at least 5 data points for Exponential fit")
    if len(tt) != len(yy):
        raise ValueError("Time and signal arrays must have same length")
    
    C0, amp, dur = _basic_seeds(tt, yy)
    
    # Estimate initial parameters
    slope_est = (yy[-1] - yy[0]) / max(dur, 1e-6)
    
    # Try multiple initial parameter guesses
    initial_guesses = [
        [slope_est * 0.5, C0, amp * 0.5, 1.0 / max(dur * 0.3, 1e-3), float(np.median(tt))],
        [slope_est * 0.3, C0, amp * 0.3, 1.0 / max(dur * 0.2, 1e-3), float(np.percentile(tt, 25))],
        [slope_est * 0.7, C0, amp * 0.7, 1.0 / max(dur * 0.4, 1e-3), float(np.percentile(tt, 75))],
    ]
    
    lb = [
        -abs(slope_est) * 10,
        C0 - 10 * amp - 100,
        -10 * max(abs(yy)),
        0,
        min(tt) - dur,
    ]
    ub = [
        abs(slope_est) * 10,
        C0 + 10 * amp + 100,
        10 * max(abs(yy)),
        10.0 / max(dur * 0.02, 1e-3),
        max(tt) + dur,
    ]
    
    best_result = None
    best_rss = float('inf')
    last_error = None
    
    for p0 in initial_guesses:
        try:
            popt, pcov = curve_fit(model_Exponential, tt, yy, p0=p0, bounds=(lb, ub), method="trf", maxfev=maxfev)
            
            # Convert to numpy array and check validity
            popt = np.asarray(popt, dtype=float)
            if len(popt) == 0 or np.any(~np.isfinite(popt)):
                continue
            
            yhat = model_Exponential(tt, *popt)
            yhat = np.asarray(yhat, dtype=float)
            
            # Check for invalid values in fitted data
            if len(yhat) == 0 or np.any(~np.isfinite(yhat)):
                continue
            
            # Ensure arrays are same length
            if len(yhat) != len(yy):
                continue
            
            rss = float(np.sum((yy - yhat) ** 2))
            if not np.isfinite(rss):
                continue
                
            if rss < best_rss:
                best_rss = rss
                best_result = (popt.copy(), yhat.copy())
        except (RuntimeError, ValueError, TypeError, AttributeError, MemoryError) as e:
            last_error = e
            continue
    
    if best_result is None:
        raise RuntimeError(f"Exponential fit failed with all initial guesses. Last error: {last_error}")
    
    popt, yhat = best_result
    
    # Ensure arrays are properly formatted and valid
    popt = np.asarray(popt, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    
    # Final validation
    if len(popt) == 0 or len(yhat) == 0 or len(yhat) != len(tt):
        raise RuntimeError("Exponential fit produced invalid array dimensions")
    if np.any(~np.isfinite(popt)) or np.any(~np.isfinite(yhat)):
        raise RuntimeError("Exponential fit produced invalid values (NaN/Inf)")
    
    rss = float(np.sum((yy - yhat) ** 2))
    n = len(tt)
    k = len(popt)
    
    a, b, c, k_decay, t0 = popt
    k_decay = float(k_decay)
    
    t_half = math.log(2) / k_decay if k_decay > 0 else np.nan
    t_5pct = math.log(20) / k_decay if k_decay > 0 else np.nan
    
    # Calculate initial rate at the start of the interval (tt[0]), not at t=0
    # Derivative: dy/dt = a - c*k*exp(-k*(t-t0))
    t_start = float(tt[0])
    init_rate = float(a - c * k_decay * np.exp(-k_decay * (t_start - t0)))
    
    return {
        "model": "Exponential",
        "params": popt,
        "names": MODEL_FUNCS["Exponential"][1],
        "yhat": yhat,
        "rss": rss,
        "r2": r2_score(yy, yhat),
        "aic": aic(n, rss, k),
        "bic": bic(n, rss, k),
        "H0_fit": float(model_Exponential([t_start], *popt)[0]),
        "init_rate": init_rate,
        "exp_amp": float(c),
        "exp_k": k_decay,
        "exp_t_half": t_half,
        "exp_t_5pct": t_5pct,
        "linear_slope": float(a),
        "intercept": float(b),
        "t0": float(t0),
    }


def fit_GFI(tt, yy, maxfev=20000):
    """Fit Gompertz-like model to data with improved robustness."""
    if len(tt) < 8:
        raise ValueError("Need at least 8 data points for Gompertz-like fit")
    if len(tt) != len(yy):
        raise ValueError("Time and signal arrays must have same length")
    
    C0, amp, dur = _basic_seeds(tt, yy)
    
    # Try multiple initial parameter guesses
    tail = float(np.median(yy[-max(3, len(yy) // 10) :]))
    thr = tail + 0.05 * max(yy[0] - tail, 1e-9)
    cross = np.where(yy <= thr)[0]
    t0_guess_base = float(tt[cross[0]]) if len(cross) else float(np.median(tt))
    
    initial_guesses = [
        [C0, 0.7 * amp, 0.3 * amp, 0.5, 1.0 / dur, 1.5 / dur, 1.0 / max(dur * 0.2, 1e-3), t0_guess_base],
        [C0, 0.5 * amp, 0.5 * amp, 0.3, 1.0 / (dur * 1.2), 1.0 / dur, 1.0 / max(dur * 0.15, 1e-3), float(np.percentile(tt, 25))],
        [C0, 0.9 * amp, 0.1 * amp, 0.7, 1.0 / (dur * 0.8), 2.0 / dur, 1.0 / max(dur * 0.25, 1e-3), float(np.percentile(tt, 75))],
    ]

    lb = [C0 - 10 * amp - 100, 0, 0, 0, 0, 0, 0, min(tt) - dur]
    ub = [C0 + 10 * amp + 100, 10 * max(yy), 10 * max(yy), 10, 10.0 / dur, 10.0 / dur, 10.0 / dur, max(tt) + dur]

    best_result = None
    best_rss = float('inf')
    last_error = None
    
    for p0 in initial_guesses:
        try:
            popt, pcov = curve_fit(model_GFI, tt, yy, p0=p0, bounds=(lb, ub), method="trf", maxfev=maxfev)
            
            # Convert to numpy array and check validity
            popt = np.asarray(popt, dtype=float)
            if len(popt) == 0 or np.any(~np.isfinite(popt)):
                continue
            
            yhat = model_GFI(tt, *popt)
            yhat = np.asarray(yhat, dtype=float)
            
            # Check for invalid values in fitted data
            if len(yhat) == 0 or np.any(~np.isfinite(yhat)):
                continue
            
            # Ensure arrays are same length
            if len(yhat) != len(yy):
                continue
            
            rss = float(np.sum((yy - yhat) ** 2))
            if not np.isfinite(rss):
                continue
                
            if rss < best_rss:
                best_rss = rss
                best_result = (popt.copy(), yhat.copy())
        except (RuntimeError, ValueError, TypeError, AttributeError, MemoryError) as e:
            last_error = e
            continue
    
    if best_result is None:
        raise RuntimeError(f"Gompertz-like fit failed with all initial guesses. Last error: {last_error}")
    
    popt, yhat = best_result
    
    # Ensure arrays are properly formatted and valid
    popt = np.asarray(popt, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    
    # Final validation
    if len(popt) == 0 or len(yhat) == 0 or len(yhat) != len(tt):
        raise RuntimeError("Gompertz-like fit produced invalid array dimensions")
    if np.any(~np.isfinite(popt)) or np.any(~np.isfinite(yhat)):
        raise RuntimeError("Gompertz-like fit produced invalid values (NaN/Inf)")
    
    rss = float(np.sum((yy - yhat) ** 2))
    n = len(tt)
    k = len(popt)
    
    B = float(popt[2])
    kfast = float(popt[5])
    t_start = float(tt[0])
    init_total = initial_rate_numeric(model_GFI, popt, t_start=t_start)
    init_fast = B * kfast
    init_ib = init_total - init_fast
    
    return {
        "model": "GFI",
        "params": popt,
        "names": MODEL_FUNCS["GFI"][1],
        "yhat": yhat,
        "rss": rss,
        "r2": r2_score(yy, yhat),
        "aic": aic(n, rss, k),
        "bic": bic(n, rss, k),
        "H0_fit": float(model_GFI([t_start], *popt)[0]),
        "init_rate": init_total,
        "fast_B": B,
        "fast_k": kfast,
        "init_rate_fast": init_fast,
        "init_rate_IB": init_ib,
    }

