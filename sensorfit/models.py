"""Mathematical models for sensor data fitting."""

import numpy as np


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
    k = np.clip(k, 0, np.inf)  # Decay rate must be non-negative
    return a * tt + b + c * np.exp(-k * (tt - t0))


def model_BiExponential(t, c, A1, k1, A2, k2, t0):
    """
    Sum of two exponentials sharing a common offset.

    y(t) = c + A1 * exp(-k1 * (t - t0)) + A2 * exp(-k2 * (t - t0))

    Both decay rates are clipped to non-negative to keep the fit stable;
    the two amplitudes can have either sign so the model can also
    describe an overshoot-then-relax response.  Two exponentials with
    different rate constants often outperform a single one on the
    fast-then-slow kinetics we see after H2O2 injection.
    """
    tt = np.asarray(t, dtype=float)
    k1 = np.clip(k1, 0, np.inf)
    k2 = np.clip(k2, 0, np.inf)
    return c + A1 * np.exp(-k1 * (tt - t0)) + A2 * np.exp(-k2 * (tt - t0))


MODEL_FUNCS = {
    "Exponential": (model_Exponential, ["a", "b", "c", "k", "t0"]),
    "BiExponential": (model_BiExponential, ["c", "A1", "k1", "A2", "k2", "t0"]),
}
