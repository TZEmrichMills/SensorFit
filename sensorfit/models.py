"""Mathematical models for sensor data fitting."""

import numpy as np


def model_IB(t, C, H0, alpha, kinact, kslow):
    """
    IB model: Inactivation with linear background.
    
    H(t) = C + H0 * exp(-alpha*(1 - exp(-kinact*t))) - kslow*t
    """
    tt = np.asarray(t, dtype=float)
    alpha = np.clip(alpha, 0, np.inf)
    kinact = np.clip(kinact, 0, np.inf)
    kslow = np.clip(kslow, 0, np.inf)
    return C + H0 * np.exp(-alpha * (1.0 - np.exp(-kinact * tt))) - kslow * tt


def model_Exponential(t, a, b, c, k, t0):
    """
    Robust single exponential fit with linear background.
    
    y(t) = a * t + b + c * exp(-k * (t - t0))
    
    Parameters:
    -----------
    a : float
        Linear slope
    b : float
        Intercept
    c : float
        Exponential amplitude
    k : float
        Decay rate constant
    t0 : float
        Time offset
    """
    tt = np.asarray(t, dtype=float)
    a = np.clip(a, -np.inf, np.inf)  # Allow negative slope
    k = np.clip(k, 0, np.inf)  # Decay rate must be non-negative
    c = np.clip(c, -np.inf, np.inf)  # Allow positive or negative amplitude
    return a * tt + b + c * np.exp(-k * (tt - t0))


def model_GFI(t, C, A, B, alpha, kinact, kfast, k, t0):
    """
    GFI model: Gompertz-gated inactivation.
    
    H(t) = C + A * exp(-alpha*(1 - exp(-kinact*t))) * exp(-exp(k*(t - t0))) + B * exp(-kfast*t)
    """
    tt = np.asarray(t, dtype=float)
    inact = np.exp(-np.clip(alpha, 0, np.inf) * (1.0 - np.exp(-np.clip(kinact, 0, np.inf) * tt)))
    gate = np.exp(-np.exp(np.clip(k, 0, np.inf) * (tt - t0)))
    fast = np.clip(B, 0, np.inf) * np.exp(-np.clip(kfast, 0, np.inf) * tt)
    return C + np.clip(A, 0, np.inf) * inact * gate + fast


MODEL_FUNCS = {
    "IB": (model_IB, ["C", "H0", "alpha", "kinact", "kslow"]),
    "Exponential": (model_Exponential, ["a", "b", "c", "k", "t0"]),
    "GFI": (model_GFI, ["C", "A", "B", "alpha", "kinact", "kfast", "k", "t0"]),
}

