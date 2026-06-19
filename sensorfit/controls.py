"""Control-template helpers reused by the per-interval subtraction flow.

Historically this module also held an upfront sample-grouping UI
(``--control-mode``), but that mechanism was removed once per-interval
control subtraction became the single, more intuitive path.  What remains
are the small, generic helpers the per-interval flow still uses:

- ``save_control_template`` / ``load_control_template`` — persist/read a
  calibrated control interval as a 2-column CSV (time zero-based) under
  ``Calibrated/_control_templates/``.
- ``interpolate_control_to_grid`` — align a control trace onto a sample's
  time grid (linear interp + bounded linear extrapolation on the tails).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .calibration import CALIBRATED_COLUMN


CONTROL_TEMPLATES_DIRNAME = "_control_templates"
MAX_EXTRAPOLATION_FRAC = 0.20  # 20 % of control duration


# ──────────────────────────────────────────────────────────────────────────
# Template I/O
# ──────────────────────────────────────────────────────────────────────────

def control_templates_dir(calibrated_dir: Path) -> Path:
    return calibrated_dir / CONTROL_TEMPLATES_DIRNAME


def save_control_template(
    calibrated_dir: Path,
    name: str,
    time_values: np.ndarray,
    h2o2_values: np.ndarray,
) -> Path:
    """Persist a control's reference interval as a 2-column CSV, time zero-based."""
    templates = control_templates_dir(calibrated_dir)
    templates.mkdir(parents=True, exist_ok=True)
    safe_name = name.replace("/", "_").replace(" ", "_")
    out_path = templates / f"{safe_name}.csv"
    t = np.asarray(time_values, dtype=float)
    t0 = float(t[0]) if t.size else 0.0
    df = pd.DataFrame({
        "time_s": t - t0,
        CALIBRATED_COLUMN: np.asarray(h2o2_values, dtype=float),
    })
    df.to_csv(out_path, index=False)
    return out_path


def load_control_template(
    calibrated_dir: Path, template_filename: str
) -> tuple[np.ndarray, np.ndarray]:
    in_path = control_templates_dir(calibrated_dir) / template_filename
    df = pd.read_csv(in_path)
    return df["time_s"].to_numpy(dtype=float), df[CALIBRATED_COLUMN].to_numpy(dtype=float)


# ──────────────────────────────────────────────────────────────────────────
# Alignment math
# ──────────────────────────────────────────────────────────────────────────

def interpolate_control_to_grid(
    control_t: np.ndarray,
    control_y: np.ndarray,
    sample_t: np.ndarray,
    max_extrap_frac: float = MAX_EXTRAPOLATION_FRAC,
) -> tuple[np.ndarray, dict]:
    """Resample control onto the sample's time grid.

    Inside the control's range: linear interpolation.
    Outside: linear extrapolation using the first/last 5 % of the control,
    bounded so we never extend past ``max_extrap_frac`` × control duration.

    Returns
    -------
    (interpolated, info)
        ``interpolated`` has the same length as ``sample_t``.  ``info`` is a
        dict with diagnostic counters (n_extrap_left, n_extrap_right,
        warning: str or None).
    """
    control_t = np.asarray(control_t, dtype=float)
    control_y = np.asarray(control_y, dtype=float)
    sample_t = np.asarray(sample_t, dtype=float)

    if control_t.size < 2:
        raise ValueError("Control template must have at least 2 samples.")

    t_min, t_max = control_t[0], control_t[-1]
    duration = t_max - t_min
    max_extend = duration * max_extrap_frac

    # Tail slopes from the first/last 5 % of the control
    n_tail = max(2, int(round(control_t.size * 0.05)))
    left_slope = float(np.polyfit(control_t[:n_tail], control_y[:n_tail], 1)[0])
    right_slope = float(np.polyfit(control_t[-n_tail:], control_y[-n_tail:], 1)[0])

    out = np.interp(sample_t, control_t, control_y)
    # np.interp clamps to endpoints — overwrite the clamped regions with linear
    # extrapolation, but only within max_extend of either tail.
    n_extrap_left = 0
    n_extrap_right = 0
    warning = None

    left_mask = sample_t < t_min
    if left_mask.any():
        dt = sample_t[left_mask] - t_min
        capped_dt = np.maximum(dt, -max_extend)  # most-negative is -max_extend
        out[left_mask] = control_y[0] + left_slope * capped_dt
        n_extrap_left = int(left_mask.sum())
        if (dt < -max_extend).any():
            warning = (
                f"Sample extends {-dt.min():.2f}s before control start; "
                f"capped at {max_extend:.2f}s of extrapolation."
            )

    right_mask = sample_t > t_max
    if right_mask.any():
        dt = sample_t[right_mask] - t_max
        capped_dt = np.minimum(dt, max_extend)
        out[right_mask] = control_y[-1] + right_slope * capped_dt
        n_extrap_right = int(right_mask.sum())
        if (dt > max_extend).any():
            msg = (
                f"Sample extends {dt.max():.2f}s past control end; "
                f"capped at {max_extend:.2f}s of extrapolation."
            )
            warning = warning + " " + msg if warning else msg

    return out, {
        "n_extrap_left": n_extrap_left,
        "n_extrap_right": n_extrap_right,
        "warning": warning,
    }


# ──────────────────────────────────────────────────────────────────────────
# Multi-control averaging
# ──────────────────────────────────────────────────────────────────────────

def average_controls_on_grid(
    members: list[tuple[np.ndarray, np.ndarray]],
    sample_t: np.ndarray,
    anchor_t0: float | None = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Average multiple control traces after aligning them onto *sample_t*.

    Each member is ``(control_t_zero_based, control_y)``.  If *anchor_t0*
    is given every member is shifted so its t=0 maps to that absolute time;
    otherwise each member's t=0 maps to ``sample_t[0]``.

    Returns ``(averaged_y, individual_interps)`` where *individual_interps*
    has the same length as *members* — each entry is the member interpolated
    onto *sample_t* (useful for the preview plot).
    """
    if not members:
        raise ValueError("Need at least one control member to average.")

    if anchor_t0 is None:
        anchor_t0 = float(sample_t[0])

    interps: list[np.ndarray] = []
    for ct_zero, cy in members:
        shifted = np.asarray(ct_zero, dtype=float) + anchor_t0
        interp, _ = interpolate_control_to_grid(shifted, cy, sample_t)
        interps.append(interp)

    stacked = np.vstack(interps)
    averaged = np.nanmean(stacked, axis=0)
    return averaged, interps
