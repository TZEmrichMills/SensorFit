"""Utility functions for SensorFit."""

import math
import numpy as np


def fmt3(x):
    """Format number to 3 decimal places."""
    return f"{x:.3f}" if np.isfinite(x) else "nan"


def parse_models(s: str):
    """Parse a comma-separated model string into canonical internal tags.

    Accepts case-insensitive aliases (``exp``, ``biexp``, ``linear``) and
    the canonical tags (``Exponential``, ``BiExponential``,
    ``ManualLinear``).  Unknown tokens are dropped; an empty string
    defaults to ``["Exponential"]``.
    """
    aliases = {
        "EXP": "Exponential",
        "EXPONENTIAL": "Exponential",
        "SINGLE_EXP": "Exponential",
        "BIEXP": "BiExponential",
        "BIEXPONENTIAL": "BiExponential",
        "DOUBLE_EXP": "BiExponential",
        "LINEAR": "ManualLinear",
        "MANUALLINEAR": "ManualLinear",
        "MANUAL_LINEAR": "ManualLinear",
    }
    if not s:
        return ["Exponential"]
    parts: list[str] = []
    for token in s.replace(";", ",").split(","):
        tok = aliases.get(token.strip().upper().replace(" ", "_"))
        if tok and tok not in parts:
            parts.append(tok)
    return parts if parts else ["Exponential"]


def r2_score(y, yhat):
    """Calculate R² score."""
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def aic(n, rss, k):
    """Calculate Akaike Information Criterion."""
    rss = max(rss, 1e-15)
    return n * math.log(rss / n) + 2 * k


def bic(n, rss, k):
    """Calculate Bayesian Information Criterion."""
    rss = max(rss, 1e-15)
    return n * math.log(rss / n) + k * math.log(n)


def initial_rate_numeric(f, params, t_start=0.0):
    """
    Calculate initial rate numerically at the start of the interval.
    
    Parameters:
    -----------
    f : callable
        Model function
    params : array-like
        Model parameters
    t_start : float
        Start time of the interval (default 0.0)
    
    Returns:
    --------
    float
        Initial rate (µM/s) at t=t_start
    """
    eps = 1e-6
    H0 = float(f([t_start], *params)[0])
    H_eps = float(f([t_start + eps], *params)[0])
    return -(H_eps - H0) / eps


def downsample(t, y, max_pts=1000):
    """Downsample data to maximum number of points."""
    if len(t) <= max_pts:
        return t, y
    idx = np.linspace(0, len(t) - 1, max_pts).astype(int)
    return t[idx], y[idx]


def tail_turnover(tt, yhat, H0_val, frac=0.25):
    """Calculate tail turnover using linear fit to tail region."""
    n = len(tt)
    if n < 3:
        return np.nan
    w = max(3, int(n * float(frac)))
    A = np.vstack([tt[-w:], np.ones(w)]).T
    m, b = np.linalg.lstsq(A, yhat[-w:], rcond=None)[0]
    # Calculate the y-value of the tail fit line at the start time of the interval
    y_at_start = m * tt[0] + b
    return float(max(0.0, H0_val - y_at_start))

