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


# ──────────────────────────────────────────────────────────────────────────
# Injection-time detection and alignment
# ──────────────────────────────────────────────────────────────────────────

def detect_injection_time(
    time_values: np.ndarray,
    h2o2_values: np.ndarray,
) -> float | None:
    """Estimate the time of the H₂O₂ injection step in a calibrated trace.

    Sample and control runs rarely have their injection at the same offset
    within their selected intervals.  Aligning them at their interval
    *starts* therefore lines up a post-injection sample against a
    pre-injection control, and the whole injection step (~100 µM) gets
    subtracted instead of just the electrode drift.

    The estimate is the **half-height crossing** of the injection rise:
    a point that corresponds between two traces even when their rise
    times differ.  Injection transients (single-sample spikes of several
    hundred µM) are removed with a ~1 s median filter first, and the
    plateau / baseline levels are taken as medians either side of the
    steepest rise rather than as global min/max — a lone overshoot spike
    would otherwise put the "plateau" far above the real one.

    Returns ``None`` when no rising step can be identified (e.g. a trace
    that only decays), in which case callers should fall back to
    start-of-interval alignment.
    """
    t = np.asarray(time_values, dtype=float)
    y = np.asarray(h2o2_values, dtype=float)
    if t.size < 10 or y.size != t.size:
        return None

    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        return None

    # ~1 s median filter kills injection spikes without shifting the edge.
    k = max(3, int(round(1.0 / dt)) | 1)
    k = min(k, max(3, (y.size // 2) * 2 - 1))
    try:
        from scipy.ndimage import median_filter

        ys = median_filter(y, size=k, mode="nearest")
    except Exception:
        ys = y

    with np.errstate(invalid="ignore"):
        grad = np.gradient(ys, t)
    if not np.any(np.isfinite(grad)):
        return None
    i = int(np.nanargmax(grad))
    if not np.isfinite(grad[i]) or grad[i] <= 0:
        return None

    # Plateau / baseline as medians either side of the steepest rise.
    w = max(5, int(round(max(5.0, 0.05 * (t[-1] - t[0])) / dt)))
    plateau = float(np.nanmedian(ys[i:min(i + w, ys.size)]))
    base = float(np.nanmedian(ys[max(0, i - w):i + 1]))
    if not np.isfinite(plateau) or not np.isfinite(base) or (plateau - base) <= 0:
        return None

    half = base + 0.5 * (plateau - base)
    ahead = np.nonzero(ys[i:] >= half)[0]
    if ahead.size:
        return float(t[i + int(ahead[0])])
    behind = np.nonzero(ys[:i + 1] >= half)[0]
    if behind.size:
        return float(t[int(behind[0])])
    return float(t[i])


def suggest_anchor_time(
    sample_t: np.ndarray,
    sample_y: np.ndarray,
) -> float:
    """Default deviation-reference time for control subtraction.

    Placed a short settle margin after the sample's injection so the
    control (aligned to the same injection) is read on its settled
    plateau rather than mid-rise or on an overshoot transient.
    """
    t = np.asarray(sample_t, dtype=float)
    if t.size == 0:
        return 0.0
    duration = float(t[-1] - t[0])
    settle = max(3.0, 0.01 * duration)
    inj = detect_injection_time(t, sample_y)
    base = inj if inj is not None else float(t[0])
    # Never push the anchor past the first third of the interval.
    return float(min(base + settle, t[0] + max(settle, duration / 3.0)))


def align_control_offset(
    sample_t: np.ndarray,
    sample_y: np.ndarray,
    control_t_zero: np.ndarray,
    control_y: np.ndarray,
) -> float:
    """Return the shift to add to a control's zero-based time so its
    injection coincides with the sample's.

    Returns ``0.0`` (start-of-interval alignment, the old behaviour) when
    the injection cannot be located in either trace.
    """
    sample_inj = detect_injection_time(sample_t, sample_y)
    ctrl_inj = detect_injection_time(control_t_zero, control_y)
    if sample_inj is None or ctrl_inj is None:
        return 0.0
    st = np.asarray(sample_t, dtype=float)
    ct = np.asarray(control_t_zero, dtype=float)
    if st.size == 0 or ct.size == 0:
        return 0.0
    # sample_inj is absolute; the control's zero-based injection offset is
    # ctrl_inj - ct[0].  Placing the control's t=0 at sample_t[0] + shift
    # makes the two injections coincide.
    return float((sample_inj - st[0]) - (ctrl_inj - ct[0]))


def deviation_from_anchor(
    control_on_grid: np.ndarray,
    sample_t: np.ndarray,
    anchor_t0: float,
) -> np.ndarray:
    """Return the control's *deviation* from its value at the anchor time.

    ``corrected = sample - deviation`` preserves the sample's absolute
    scale while removing only the drift/background that the control
    reveals.  Concretely::

        deviation(t) = control(t) - control(anchor_t0)

    so if a negative control starts at 100 µM and drifts to 95, the
    deviation at that point is −5, and the sample gains +5 (undoing the
    drift) instead of losing 100 (which would zero out the signal).
    """
    anchor_val = float(np.interp(anchor_t0, sample_t, control_on_grid))
    return control_on_grid - anchor_val
