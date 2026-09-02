"""Fitting functions for sensor data models."""

import math
import numpy as np
from scipy.optimize import curve_fit

from .models import model_Exponential, model_BiExponential, MODEL_FUNCS
from .utils import r2_score, aic, bic


def _basic_seeds(tt, yy):
    """Generate basic seed values from data."""
    C0 = float(np.median(yy[-max(3, len(yy) // 10) :]))
    amp = float(max(yy[0] - C0, 1e-6))
    dur = float(max(tt) - min(tt) + 1e-9)
    return C0, amp, dur


def fit_Exponential(tt, yy, maxfev=12000):
    """Fit exponential model: y(t) = a*t + b + c*exp(-k*(t-t0))."""
    if len(tt) < 5:
        raise ValueError("Need at least 5 data points for Exponential fit")
    if len(tt) != len(yy):
        raise ValueError("Time and signal arrays must have same length")

    C0, amp, dur = _basic_seeds(tt, yy)
    slope_est = (yy[-1] - yy[0]) / max(dur, 1e-6)

    initial_guesses = [
        [slope_est * 0.5, C0, amp * 0.5, 1.0 / max(dur * 0.3, 1e-3), float(np.median(tt))],
        [slope_est * 0.3, C0, amp * 0.3, 1.0 / max(dur * 0.2, 1e-3), float(np.percentile(tt, 25))],
        [slope_est * 0.7, C0, amp * 0.7, 1.0 / max(dur * 0.4, 1e-3), float(np.percentile(tt, 75))],
    ]

    lb = [-abs(slope_est) * 10, C0 - 10 * amp - 100, -10 * max(abs(yy)), 0, min(tt) - dur]
    ub = [abs(slope_est) * 10, C0 + 10 * amp + 100, 10 * max(abs(yy)),
          10.0 / max(dur * 0.02, 1e-3), max(tt) + dur]

    best_result = None
    best_rss = float("inf")
    last_error = None

    for p0 in initial_guesses:
        try:
            popt, _ = curve_fit(model_Exponential, tt, yy, p0=p0, bounds=(lb, ub),
                                method="trf", maxfev=maxfev)
            popt = np.asarray(popt, dtype=float)
            if len(popt) == 0 or np.any(~np.isfinite(popt)):
                continue
            yhat = np.asarray(model_Exponential(tt, *popt), dtype=float)
            if len(yhat) != len(yy) or np.any(~np.isfinite(yhat)):
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
    rss = float(np.sum((yy - yhat) ** 2))
    n, k = len(tt), len(popt)
    a, b, c, k_decay, t0 = popt
    k_decay = float(k_decay)
    t_half = math.log(2) / k_decay if k_decay > 0 else float("nan")
    t_5pct = math.log(20) / k_decay if k_decay > 0 else float("nan")
    t_start = float(tt[0])
    # dy/dt at t_start: a - c*k*exp(-k*(t_start - t0))
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


def fit_BiExponential(tt, yy, maxfev=20000):
    """Fit a two-exponential decay sharing a common offset:

        y(t) = c + A1 * exp(-k1 * (t - t0)) + A2 * exp(-k2 * (t - t0))

    The two rate constants are separated by an order of magnitude in the
    initial guesses so the optimiser doesn't collapse them into one
    exponential; if it does anyway the fit is still valid (equivalent to
    a single exponential + offset), just with A2 ≈ 0 or k1 ≈ k2.
    """
    tt = np.asarray(tt, dtype=float)
    yy = np.asarray(yy, dtype=float)
    if len(tt) < 6:
        raise ValueError("Need at least 6 data points for BiExponential fit")
    if len(tt) != len(yy):
        raise ValueError("Time and signal arrays must have same length")

    C0, amp, dur = _basic_seeds(tt, yy)
    t_start = float(tt[0])

    # Rate seeds cover fast (tau ~ 5% of duration) and slow (tau ~ 60%).
    k_fast = 1.0 / max(dur * 0.05, 1e-3)
    k_slow = 1.0 / max(dur * 0.60, 1e-3)

    initial_guesses = [
        [C0, amp * 0.6, k_fast, amp * 0.4, k_slow, t_start],
        [C0, amp * 0.8, k_fast * 2.0, amp * 0.2, k_slow * 0.5, t_start],
        [C0, amp * 0.3, k_fast * 0.5, amp * 0.7, k_slow * 2.0, t_start],
        # A case where the "fast" component actually rises briefly (overshoot).
        [C0, -amp * 0.2, k_fast, amp * 1.2, k_slow, t_start],
    ]

    amp_bound = 20 * max(abs(amp), abs(C0), 1.0) + 100
    lb = [C0 - 10 * amp - 100, -amp_bound, 0, -amp_bound, 0, t_start - dur]
    ub = [C0 + 10 * amp + 100,  amp_bound, 10.0 / max(dur * 0.01, 1e-3),
                              amp_bound, 10.0 / max(dur * 0.01, 1e-3), t_start + dur]

    best_result = None
    best_rss = float("inf")
    last_error = None

    for p0 in initial_guesses:
        try:
            popt, _ = curve_fit(model_BiExponential, tt, yy, p0=p0, bounds=(lb, ub),
                                method="trf", maxfev=maxfev)
            popt = np.asarray(popt, dtype=float)
            if len(popt) == 0 or np.any(~np.isfinite(popt)):
                continue
            yhat = np.asarray(model_BiExponential(tt, *popt), dtype=float)
            if len(yhat) != len(yy) or np.any(~np.isfinite(yhat)):
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
        raise RuntimeError(
            f"BiExponential fit failed with all initial guesses. Last error: {last_error}"
        )

    popt, yhat = best_result

    # Canonicalise: order the two components so the FASTER decay is first.
    c, A1, k1, A2, k2, t0 = popt
    if k2 > k1:
        A1, k1, A2, k2 = A2, k2, A1, k1
    popt = np.array([c, A1, k1, A2, k2, t0], dtype=float)

    rss = float(np.sum((yy - yhat) ** 2))
    n, k = len(tt), len(popt)

    t_half_1 = math.log(2) / k1 if k1 > 0 else float("nan")
    t_half_2 = math.log(2) / k2 if k2 > 0 else float("nan")

    # dy/dt at t_start: -A1*k1*exp(-k1*(t_start-t0)) - A2*k2*exp(-k2*(t_start-t0))
    dt_s = t_start - t0
    init_rate = float(
        -A1 * k1 * np.exp(-k1 * dt_s) - A2 * k2 * np.exp(-k2 * dt_s)
    )

    return {
        "model": "BiExponential",
        "params": popt,
        "names": MODEL_FUNCS["BiExponential"][1],
        "yhat": yhat,
        "rss": rss,
        "r2": r2_score(yy, yhat),
        "aic": aic(n, rss, k),
        "bic": bic(n, rss, k),
        "H0_fit": float(model_BiExponential([t_start], *popt)[0]),
        "init_rate": init_rate,
        "biexp_offset": float(c),
        "biexp_A_fast": float(A1),
        "biexp_k_fast": float(k1),
        "biexp_t_half_fast": t_half_1,
        "biexp_A_slow": float(A2),
        "biexp_k_slow": float(k2),
        "biexp_t_half_slow": t_half_2,
        "t0": float(t0),
    }
