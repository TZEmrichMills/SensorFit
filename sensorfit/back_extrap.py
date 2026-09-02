"""H2O2-injection-start back-extrapolation math.

For reactions that begin with an H2O2 addition and use a two-point
calibration (pre-addition = 0 µM; just-after-addition = nominal µM), the
calibrated values under-estimate the true initial concentration because the
enzyme has already consumed some H2O2 during the instrument deadtime
(~1–2 s).

This module exposes only the pure-math helper used by:
- The 4-point back-extrap calibration mode in ``calibration.select_points``.
- The per-fit back-extrapolation prompt in ``interval_processor`` (added in
  a later commit).

The previous standalone "after-fitting" UI was removed when SensorFit moved
to a per-interval flow: back-extrapolation is now decided inline at the
points where it applies, not as a separate file-level phase.
"""

from __future__ import annotations

import numpy as np

from .models import model_Exponential


def compute_back_extrap(
    fit_results_for_interval: dict[str, dict],
    interval_t_start_abs: float,
    interval_observed_value: float,
    deadtime_s: float,
    nominal_uM: float,
) -> dict:
    """Return back-extrapolation results for one interval.

    Parameters
    ----------
    fit_results_for_interval : dict
        ``{model_name: {init_rate, params, ...}}`` for this interval.
    interval_t_start_abs : float
        Absolute time at the start of the interval (full-trace time
        coordinate the fits were computed in).
    interval_observed_value : float
        Observed [H₂O₂] at the interval start (linear-extrapolation anchor
        when no Exponential fit is available).
    deadtime_s : float
        Seconds between the H₂O₂ addition and the first reliable reading.
    nominal_uM : float
        The nominal H₂O₂ concentration the user added.

    Returns
    -------
    dict with keys:
        ``method`` ("exponential" or "linear")
        ``deadtime_s``, ``nominal_uM``
        ``back_extrap_uM`` — fitted [H₂O₂] at t_start − deadtime
        ``stretch_factor`` — back_extrap_uM / nominal_uM
        ``rate_at_back_uM_per_s`` — derivative of the Exponential at the
            back-extrapolated time (== initial_rate for the linear case)
        ``stretch_initial_rate`` — rate_at_back × stretch_factor (µM/s)
    """
    t_back = float(interval_t_start_abs) - float(deadtime_s)

    if "Exponential" in fit_results_for_interval:
        exp = fit_results_for_interval["Exponential"]
        params = np.asarray(exp.get("params", []), dtype=float)
        if params.size != 5:
            raise ValueError(
                "Exponential fit has unexpected parameter shape; cannot back-extrapolate."
            )
        a, b, c, k_decay, t0 = params
        back_extrap_uM = float(model_Exponential(np.array([t_back]), *params)[0])
        rate_at_back = float(a - c * k_decay * np.exp(-k_decay * (t_back - t0)))
        method = "exponential"
    else:
        # Linear fallback: use whatever init_rate we can find
        rate = None
        for model in ("LinearInitialRate", "ManualLinear", "BiExponential"):
            if model in fit_results_for_interval:
                rate = float(
                    fit_results_for_interval[model].get("init_rate", float("nan"))
                )
                if np.isfinite(rate):
                    break
        if rate is None or not np.isfinite(rate):
            raise ValueError(
                "No fit with a usable initial rate is available for "
                "back-extrapolation.  Fit an Exponential or LinearInitialRate first."
            )
        back_extrap_uM = float(interval_observed_value + rate * (t_back - interval_t_start_abs))
        rate_at_back = rate
        method = "linear"

    nominal = float(nominal_uM)
    if abs(nominal) < 1e-12:
        stretch_factor = float("nan")
    else:
        stretch_factor = back_extrap_uM / nominal
    stretch_initial_rate = (
        rate_at_back * stretch_factor if np.isfinite(stretch_factor) else float("nan")
    )

    return {
        "method": method,
        "deadtime_s": float(deadtime_s),
        "nominal_uM": nominal,
        "back_extrap_uM": back_extrap_uM,
        "stretch_factor": stretch_factor,
        "rate_at_back_uM_per_s": rate_at_back,
        "stretch_initial_rate": stretch_initial_rate,
    }
