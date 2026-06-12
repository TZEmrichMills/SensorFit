"""Per-interval processing flow.

This module replaces the file-level ``fitting → turnover`` phases of the
older SensorFit flow with a per-interval state machine:

    for each interval the user wants to define:
        1. select interval bounds
        2. optional control subtraction
             - none / existing-interval picker / new-control modal / back
        3. zero or more fits
             - model: manual-linear | single-exp | IB
             - user-picked fit-range (start, end)
             - preview with initial rate at the chosen start
             - optional back-extrapolation prompt (deadtime TextBox)
             - "Another fit?" loop
        4. one optional Δ[H₂O₂]max
             - method: from-fit | linear-fit | single-point
             - user-picked t=0 (the dynamic anchor)
        5. "Another interval?" → top of loop, or stop and return

Every step is skippable, retryable, and has a Back option that returns to
the prior step within the same interval (or, at step 1, to the calibration
phase via the caller).

The module exposes ``run_per_interval_flow`` for the caller to drive, plus
the ``ProcessedInterval`` / ``FitRecord`` / ``DeltaMaxRecord`` dataclasses
holding the accumulated state.

Pure-math helpers (``delta_max_from_fit``, ``delta_max_from_linear``,
``delta_max_from_point``) are exported for unit testing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
import numpy as np
import pandas as pd

from .calibration import (
    CALIBRATED_COLUMN,
    IntervalSubset,
    add_instruction_banner,
    create_small_button,
    truncate_filename,
)
from .controls import interpolate_control_to_grid
from .fitting import fit_IB, fit_Exponential
from .models import model_Exponential, model_IB
from .zoom_hotkey import install_zoom_keys


# ────────────────────────────────────────────────────────────────────────
# Data classes for the accumulated state
# ────────────────────────────────────────────────────────────────────────


@dataclass
class FitRecord:
    """One fit applied to one interval.

    The model name is the SensorFit display name ("ManualLinear",
    "Exponential", "IB").  Parameters / yhat are stored in absolute
    full-trace time so they overlay correctly on the calibrated trace.
    """

    model: str
    fit_start_s: float
    fit_end_s: float
    params: list[float]
    param_names: list[str]
    yhat: np.ndarray              # fitted curve, in the (fit_start, fit_end) range
    init_rate_uM_per_s: float     # rate at the fit's start (user-chosen)
    init_rate_at_t_s: float       # absolute time at which init_rate was evaluated
    r2: float = float("nan")
    rss: float = float("nan")
    # Back-extrapolation (per-fit, optional)
    back_extrap_applied: bool = False
    back_extrap_deadtime_s: float | None = None
    back_extrap_t0_s: float | None = None
    back_extrap_rate_uM_per_s: float | None = None


@dataclass
class DeltaMaxRecord:
    """The user's chosen Δ[H₂O₂]max for an interval.

    method:
      - "from-fit" — y-intercept of the last fit's asymptote at t_zero
      - "linear"   — y-intercept of a user-drawn line at t_zero
      - "point"    — the y-value of a single picked point (Δmax = y at t_zero
                     − y at end of run; simpler "single anchor" mode)
    """

    method: str
    t_zero_s: float
    value_uM: float
    # For "linear" mode: slope, intercept of the line the user drew
    linear_slope: float | None = None
    linear_intercept: float | None = None


@dataclass
class ProcessedInterval:
    """Everything accumulated about one interval after the per-interval flow."""

    index: int                       # 1-based, per file
    start_time: float
    end_time: float
    data: pd.DataFrame               # time_col + CALIBRATED_COLUMN, possibly corrected
    time_col: str
    control_subtracted: bool = False
    control_source: str | None = None  # human-readable description
    fits: list[FitRecord] = field(default_factory=list)
    delta_max: DeltaMaxRecord | None = None


# ────────────────────────────────────────────────────────────────────────
# Pure-math helpers (unit-testable)
# ────────────────────────────────────────────────────────────────────────


def delta_max_from_fit(fit: FitRecord, t_zero: float) -> float:
    """Δmax estimated from a fitted model.

    For Exponential ``y(t) = a*t + b + c*exp(-k*(t-t0))``, the asymptote is
    the linear baseline ``a*t + b``; the "consumption" at ``t_zero`` is
    ``y(t_zero) − asymptote(t_zero) = c * exp(-k*(t_zero - t0))``.

    For IB ``y(t) = C + H0*exp(...) - kslow*t``, the asymptote is
    ``C - kslow*t``; ``Δmax = y(t_zero) − asymptote(t_zero) = H0*exp(...)``.

    For ManualLinear, the asymptote IS the fit; Δmax is undefined and we
    return NaN (caller should fall back to a different method).
    """
    if fit.model == "Exponential" and len(fit.params) == 5:
        a, b, c, k_decay, t0 = fit.params
        # Δ = y_at_t_zero - asymptote_at_t_zero = c * exp(-k*(t_zero - t0))
        return float(c * np.exp(-k_decay * (t_zero - t0)))
    if fit.model == "IB" and len(fit.params) == 5:
        # H(t) = C + H0*exp(-alpha*(1 - exp(-kinact*t))) - kslow*t
        # Δ at t_zero = H0 * exp(-alpha*(1 - exp(-kinact*t_zero)))
        _C, H0, alpha, kinact, _kslow = fit.params
        return float(H0 * np.exp(-alpha * (1.0 - np.exp(-kinact * t_zero))))
    return float("nan")


def delta_max_from_linear(
    t: np.ndarray, y: np.ndarray,
    line_start_idx: int, line_end_idx: int,
    t_zero: float,
) -> tuple[float, float, float]:
    """Δmax from a straight line fitted between two indices.

    Returns ``(delta_max_uM, slope, intercept)`` where intercept is the
    y-value of the line at ``t_zero``.
    """
    if line_end_idx <= line_start_idx + 1:
        raise ValueError("Linear Δmax needs ≥2 points on the fit segment.")
    t_seg = t[line_start_idx : line_end_idx + 1].astype(float)
    y_seg = y[line_start_idx : line_end_idx + 1].astype(float)
    slope, intercept_at_0 = np.polyfit(t_seg, y_seg, 1)
    y_at_tzero = float(slope * t_zero + intercept_at_0)
    # Δmax = y_at_tzero − y at end-of-run (we use the last point of t/y here
    # as a sensible "asymptote")
    y_end = float(y[-1])
    return y_at_tzero - y_end, float(slope), float(intercept_at_0)


def delta_max_from_point(y_at_tzero: float, y_end: float) -> float:
    """Single-point Δmax: y(t_zero) − y(end)."""
    return float(y_at_tzero) - float(y_end)


# ────────────────────────────────────────────────────────────────────────
# UI: one-interval selection
# ────────────────────────────────────────────────────────────────────────


def select_one_interval(
    time_values: np.ndarray,
    h2o2_values: np.ndarray,
    already_defined: list[tuple[float, float]],
    filename: str | None = None,
    interval_number: int = 1,
):
    """Pick a single interval by clicking start and end.

    ``already_defined`` is drawn faintly so the user can see what's already
    been done.

    Returns one of:
      (start, end) tuple — interval accepted.
      "done"   — user clicked "Done with intervals"; nothing returned.
      "back"   — user wants to go back to the previous phase (calibration).
      "skip"   — user wants to skip this interval (no-op; same as "done"
                 from the caller's perspective unless the caller wants to
                 distinguish).
    """
    fig, ax = plt.subplots(figsize=(11, 6.5))
    try:
        fig.canvas.manager.set_window_title(
            "SensorFit — Pick an interval"
            + (f": {truncate_filename(filename)}" if filename else "")
        )
    except Exception:
        pass
    plt.subplots_adjust(left=0.1, bottom=0.18, right=0.98, top=0.78)
    ax.plot(time_values, h2o2_values, color="tab:green", lw=1.2, label="Calibrated trace")
    for (s_existing, e_existing) in already_defined:
        ax.axvspan(s_existing, e_existing, color="grey", alpha=0.18)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("H2O2 (µM)")

    title = f"Pick interval #{interval_number}: click START then END.  Click Done when finished."
    if filename:
        title = f"{truncate_filename(filename)}\n{title}"
    fig.suptitle(title, fontsize=11)

    add_instruction_banner(
        fig,
        "Click two points: the first sets the interval START, the second the END.  "
        "Existing intervals are shown in grey.  Buttons below: Done (finish), "
        "Back (return to calibration), Skip (this interval).",
        y=0.985,
        width=110,
    )

    pending = {"start": None}
    marker = {"artist": None}
    span = {"artist": None}
    state = {"action": None, "interval": None}

    def _toolbar_active() -> bool:
        toolbar = fig.canvas.toolbar
        if toolbar is None:
            return False
        if getattr(toolbar, "mode", "") in ("zoom rect", "pan/zoom", "zoom", "pan"):
            return True
        is_active = getattr(toolbar, "_active", None)
        if is_active and is_active not in ("", None):
            s = str(is_active).upper()
            if "ZOOM" in s or "PAN" in s:
                return True
        return False

    def on_click(event):
        if event.button != 1 or event.inaxes is not ax or event.xdata is None:
            return
        if _toolbar_active():
            return
        if pending["start"] is None:
            pending["start"] = float(event.xdata)
            if marker["artist"] is not None:
                try:
                    marker["artist"].remove()
                except Exception:
                    pass
            marker["artist"], = ax.plot(
                pending["start"],
                h2o2_values[int(np.abs(time_values - pending["start"]).argmin())],
                "ro", ms=10, markeredgecolor="yellow", markeredgewidth=2, zorder=5,
            )
            fig.canvas.draw_idle()
            print(f"Start: t={pending['start']:.3f} s")
        else:
            end_t = float(event.xdata)
            s, e = sorted((pending["start"], end_t))
            if abs(e - s) < 1e-6:
                print("Interval too short; ignoring.")
                return
            if span["artist"] is not None:
                try:
                    span["artist"].remove()
                except Exception:
                    pass
            span["artist"] = ax.axvspan(s, e, color="orange", alpha=0.25, zorder=1)
            state["interval"] = (s, e)
            fig.canvas.draw_idle()
            print(f"End: t={end_t:.3f} s → interval [{s:.3f}, {e:.3f}] s")

    fig.canvas.mpl_connect("button_press_event", on_click)

    def on_accept(_e=None):
        if state["interval"] is None:
            print("Please click two points to define the interval.")
            return
        state["action"] = "accept"
        plt.close(fig)

    def on_done(_e=None):
        state["action"] = "done"
        plt.close(fig)

    def on_back(_e=None):
        state["action"] = "back"
        plt.close(fig)

    def on_skip(_e=None):
        state["action"] = "skip"
        plt.close(fig)

    def on_retry(_e=None):
        pending["start"] = None
        if marker["artist"] is not None:
            try:
                marker["artist"].remove()
            except Exception:
                pass
            marker["artist"] = None
        if span["artist"] is not None:
            try:
                span["artist"].remove()
            except Exception:
                pass
            span["artist"] = None
        state["interval"] = None
        fig.canvas.draw_idle()
        print("Selection cleared.")

    ax_accept = fig.add_axes([0.10, 0.03, 0.14, 0.05])
    ax_retry = fig.add_axes([0.26, 0.03, 0.10, 0.05])
    ax_done = fig.add_axes([0.38, 0.03, 0.18, 0.05])
    ax_skip = fig.add_axes([0.58, 0.03, 0.14, 0.05])
    ax_back = fig.add_axes([0.74, 0.03, 0.14, 0.05])
    btn_accept = create_small_button(ax_accept, "Accept", "#90ee90", "#7cd47c")
    btn_retry = create_small_button(ax_retry, "Retry", "0.9", "0.8")
    btn_done = create_small_button(ax_done, "Done with intervals", "#ddddff", "#bbbbff")
    btn_skip = create_small_button(ax_skip, "Skip this one", "#ffcc99", "#ffaa66")
    btn_back = create_small_button(ax_back, "Back", "#ddddff", "#bbbbff")
    btn_accept.on_clicked(on_accept)
    btn_retry.on_clicked(on_retry)
    btn_done.on_clicked(on_done)
    btn_skip.on_clicked(on_skip)
    btn_back.on_clicked(on_back)

    install_zoom_keys(fig, ax)
    plt.show()
    plt.close(fig)

    if state["action"] == "accept" and state["interval"] is not None:
        return state["interval"]
    return state["action"] or "done"


# ────────────────────────────────────────────────────────────────────────
# UI: subtraction choice dialog (none / existing / new / back)
# ────────────────────────────────────────────────────────────────────────


def prompt_subtraction_choice(interval_summary: str) -> str:
    """Small dialog asking how (if at all) to subtract a control from this
    interval.  Returns one of: ``"none"``, ``"existing"``, ``"new"``,
    ``"back"``.
    """
    fig, ax = plt.subplots(figsize=(8, 3.4))
    try:
        fig.canvas.manager.set_window_title("SensorFit — Subtraction choice")
    except Exception:
        pass
    ax.axis("off")
    ax.text(
        0.5, 0.62,
        f"{interval_summary}\n\n"
        "Apply control subtraction?\n\n"
        "• None — use this interval as-is.\n"
        "• Subtract existing — pick a control interval already in this session\n"
        "  or in Calibrated/.\n"
        "• Subtract new — process a fresh control file (modal mini-flow).",
        ha="center", va="center", fontsize=10,
    )
    fig.suptitle("Per-interval control subtraction", fontsize=11, fontweight="bold")
    choice = {"value": None}

    def _set(v):
        def _f(_e=None):
            choice["value"] = v
            plt.close(fig)
        return _f

    ax_none = fig.add_axes([0.07, 0.10, 0.18, 0.16])
    ax_existing = fig.add_axes([0.27, 0.10, 0.20, 0.16])
    ax_new = fig.add_axes([0.49, 0.10, 0.18, 0.16])
    ax_back = fig.add_axes([0.75, 0.10, 0.18, 0.16])
    btn_none = create_small_button(ax_none, "None", "#90ee90", "#7cd47c")
    btn_existing = create_small_button(ax_existing, "Subtract existing", "#ffe680", "#ffcd55")
    btn_new = create_small_button(ax_new, "Subtract new", "#ffcc99", "#ffaa66")
    btn_back = create_small_button(ax_back, "Back", "#ddddff", "#bbbbff")
    btn_none.on_clicked(_set("none"))
    btn_existing.on_clicked(_set("existing"))
    btn_new.on_clicked(_set("new"))
    btn_back.on_clicked(_set("back"))

    plt.show()
    plt.close(fig)
    return choice["value"] or "none"


# ────────────────────────────────────────────────────────────────────────
# UI: existing-interval picker
# ────────────────────────────────────────────────────────────────────────


def _discover_existing_control_intervals(
    calibrated_dir: Path | None,
    current_session_intervals: list[ProcessedInterval],
    self_file_stem: str | None,
) -> list[dict]:
    """Build a list of every interval the user could plausibly subtract.

    Each entry is ``{"label": str, "source": str, "load": callable() -> (t, y)}``
    so the caller can lazily load the actual data only when picked.

    Sources:
    1. ``current_session_intervals`` — intervals processed earlier in this
       run (same file or earlier files).
    2. Saved ``Calibrated/*_intervals/interval_*.xlsx`` — anything already on
       disk from prior sessions.

    Intervals from the file currently being processed are EXCLUDED to prevent
    the user from subtracting an interval from itself.
    """
    out: list[dict] = []

    # Current session (in-memory)
    for pi in current_session_intervals:
        # Skip current file's own intervals (we can't know the file stem from
        # ProcessedInterval directly; caller must filter via self_file_stem)
        t_col = pi.time_col
        # Capture by default-argument to avoid late binding
        def _loader(pi=pi, t_col=t_col):
            return (
                pi.data[t_col].to_numpy(dtype=float),
                pi.data[CALIBRATED_COLUMN].to_numpy(dtype=float),
            )
        out.append({
            "label": f"(session) interval #{pi.index}  [{pi.start_time:.1f}–{pi.end_time:.1f} s]",
            "source": "session",
            "load": _loader,
        })

    # On-disk
    if calibrated_dir is not None and calibrated_dir.exists():
        for interval_dir in sorted(calibrated_dir.glob("*_intervals")):
            stem = interval_dir.name.replace("_intervals", "")
            if self_file_stem and stem == self_file_stem:
                continue
            for excel in sorted(interval_dir.glob("interval_*.xlsx")):
                if "_corrected" in excel.stem:
                    continue
                def _loader(excel=excel):
                    df = pd.read_excel(excel)
                    # Find the time and H2O2 columns by name
                    time_col = next((c for c in df.columns if c.lower().startswith("time")), df.columns[0])
                    h_col = next((c for c in df.columns if c == CALIBRATED_COLUMN), None)
                    if h_col is None:
                        # Best effort: assume second column
                        h_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
                    return (
                        df[time_col].to_numpy(dtype=float),
                        df[h_col].to_numpy(dtype=float),
                    )
                out.append({
                    "label": f"(disk) {stem} / {excel.stem}",
                    "source": "disk",
                    "load": _loader,
                })

    return out


def pick_existing_interval(
    candidates: list[dict],
    filename: str | None = None,
):
    """Open a small dialog letting the user pick one of the discovered
    intervals.  Returns the chosen dict (with ``load`` callable) or None
    if cancelled / no candidates.
    """
    if not candidates:
        print("No existing intervals available for subtraction.")
        return None

    n = len(candidates)
    fig_h = 0.7 + 0.30 * max(3, n) + 1.0
    fig = plt.figure(figsize=(9, fig_h))
    fig.suptitle(
        f"Pick a control interval to subtract"
        + (f" from {truncate_filename(filename)}" if filename else ""),
        fontsize=11,
    )
    add_instruction_banner(
        fig,
        "Click a button to pick that interval as the control.  Cancel returns "
        "to the subtraction-choice dialog.",
        y=0.985,
        width=110,
    )

    chosen = {"idx": None}
    btns = []

    def make_picker(i):
        def _f(_e=None):
            chosen["idx"] = i
            plt.close(fig)
        return _f

    # Each row is a button labelled with the candidate's text
    btn_h = 0.06
    spacing = 0.015
    available_h = 0.78
    btn_w = 0.85
    if n * (btn_h + spacing) > available_h:
        btn_h = max(0.03, (available_h - n * spacing) / n)

    for i, c in enumerate(candidates):
        y = 0.85 - (i + 1) * (btn_h + spacing)
        ax_btn = fig.add_axes([0.075, y, btn_w, btn_h])
        b = create_small_button(ax_btn, c["label"], "#ffe680", "#ffcd55")
        b.on_clicked(make_picker(i))
        btns.append(b)

    ax_cancel = fig.add_axes([0.35, 0.03, 0.30, 0.06])
    btn_cancel = create_small_button(ax_cancel, "Cancel", "#ddddff", "#bbbbff")
    btn_cancel.on_clicked(lambda _e=None: plt.close(fig))

    plt.show()
    plt.close(fig)
    return candidates[chosen["idx"]] if chosen["idx"] is not None else None


# ────────────────────────────────────────────────────────────────────────
# UI: subtraction preview
# ────────────────────────────────────────────────────────────────────────


def preview_per_interval_subtraction(
    sample_t: np.ndarray,
    sample_y: np.ndarray,
    control_t: np.ndarray,
    control_y: np.ndarray,
    filename: str | None = None,
    label: str = "Control",
):
    """Show before/after preview of the subtraction and let the user
    accept / re-anchor / skip / back.

    Returns one of:
      np.ndarray — the corrected sample_y (sample − control aligned).
      "skip" — user opted to skip subtraction.
      "back" — user wants to revisit the subtraction-choice dialog.
    """
    anchor = {"t0": float(sample_t[0])}

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(11, 7.4), sharex=True)
    try:
        fig.canvas.manager.set_window_title(
            "SensorFit — Subtraction preview"
            + (f": {truncate_filename(filename)}" if filename else "")
        )
    except Exception:
        pass
    plt.subplots_adjust(left=0.1, bottom=0.18, right=0.98, top=0.78, hspace=0.25)

    ax_top.plot(sample_t, sample_y, color="tab:green", lw=1.3, label="Sample interval")
    line_ctrl, = ax_top.plot([], [], color="tab:red", lw=1.2, alpha=0.85, label=label)
    anchor_line_top = ax_top.axvline(anchor["t0"], color="#8B008B", lw=1.8, ls="--", alpha=0.9, label="Anchor (control t=0)")
    anchor_line_bot = ax_bot.axvline(anchor["t0"], color="#8B008B", lw=1.8, ls="--", alpha=0.9)
    ax_top.set_ylabel("H2O2 (µM)")
    ax_top.legend(loc="upper right", fontsize=9)

    line_corr, = ax_bot.plot([], [], color="tab:blue", lw=1.3, label="Sample − Control")
    ax_bot.axhline(0.0, color="grey", lw=0.6, ls=":")
    ax_bot.set_xlabel("Time (s)")
    ax_bot.set_ylabel("H2O2 (µM, corrected)")
    ax_bot.legend(loc="upper right", fontsize=9)

    add_instruction_banner(
        fig,
        f"{truncate_filename(filename) if filename else ''}  "
        f"Top: sample (green) + control (red).  Bottom: sample − control.  "
        "Click on the upper plot to re-anchor the control's t=0.  Accept records "
        "the subtraction, Skip keeps the original, Back revisits the choice.",
        y=0.985,
        width=110,
    )

    def redraw():
        shifted_t = control_t + anchor["t0"]
        interp, _ = interpolate_control_to_grid(shifted_t, control_y, sample_t)
        line_ctrl.set_data(sample_t, interp)
        line_corr.set_data(sample_t, sample_y - interp)
        anchor_line_top.set_xdata([anchor["t0"], anchor["t0"]])
        anchor_line_bot.set_xdata([anchor["t0"], anchor["t0"]])
        for a in (ax_top, ax_bot):
            a.relim()
            a.autoscale_view()
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes is not ax_top or event.button != 1 or event.xdata is None:
            return
        if getattr(fig.canvas.toolbar, "mode", "") in ("zoom rect", "pan/zoom", "zoom", "pan"):
            return
        anchor["t0"] = float(event.xdata)
        redraw()

    fig.canvas.mpl_connect("button_press_event", on_click)

    decision = {"value": None}

    def on_accept(_e=None):
        decision["value"] = "accept"
        plt.close(fig)

    def on_skip(_e=None):
        decision["value"] = "skip"
        plt.close(fig)

    def on_back(_e=None):
        decision["value"] = "back"
        plt.close(fig)

    def on_reset(_e=None):
        anchor["t0"] = float(sample_t[0])
        redraw()

    ax_accept = fig.add_axes([0.10, 0.03, 0.18, 0.06])
    ax_reset = fig.add_axes([0.30, 0.03, 0.14, 0.06])
    ax_skip = fig.add_axes([0.46, 0.03, 0.18, 0.06])
    ax_back = fig.add_axes([0.66, 0.03, 0.14, 0.06])
    _btn_accept_1 = create_small_button(ax_accept, "Accept & subtract", "#90ee90", "#7cd47c")
    _btn_accept_1.on_clicked(on_accept)
    _btn_reset_2 = create_small_button(ax_reset, "Reset anchor", "0.9", "0.8")
    _btn_reset_2.on_clicked(on_reset)
    _btn_skip_3 = create_small_button(ax_skip, "Skip subtraction", "#ffcc99", "#ffaa66")
    _btn_skip_3.on_clicked(on_skip)
    _btn_back_4 = create_small_button(ax_back, "Back", "#ddddff", "#bbbbff")
    _btn_back_4.on_clicked(on_back)

    redraw()
    install_zoom_keys(fig, [ax_top, ax_bot])
    plt.show()
    plt.close(fig)

    if decision["value"] == "accept":
        shifted_t = control_t + anchor["t0"]
        interp, _ = interpolate_control_to_grid(shifted_t, control_y, sample_t)
        return sample_y - interp
    if decision["value"] == "skip":
        return "skip"
    return "back"


# ────────────────────────────────────────────────────────────────────────
# UI: one fit (model picker → fit-range → preview → back-extrap)
# ────────────────────────────────────────────────────────────────────────


def _pick_model() -> str | None:
    """Small dialog to choose one of the three offered models."""
    fig, ax = plt.subplots(figsize=(8, 3.4))
    try:
        fig.canvas.manager.set_window_title("SensorFit — Pick a fit model")
    except Exception:
        pass
    ax.axis("off")
    ax.text(
        0.5, 0.62,
        "Choose a model to fit to this interval:\n\n"
        "• Manual linear — a straight-line fit on a user-picked range.\n"
        "• Single exponential — robust exp + linear background (Exponential).\n"
        "• Inactivation (IB) — full inactivation kinetic model.",
        ha="center", va="center", fontsize=10,
    )
    fig.suptitle("Pick a fit model", fontsize=11, fontweight="bold")
    pick = {"value": None}

    def _set(v):
        def _f(_e=None):
            pick["value"] = v
            plt.close(fig)
        return _f

    ax_lin = fig.add_axes([0.08, 0.10, 0.22, 0.16])
    ax_exp = fig.add_axes([0.32, 0.10, 0.22, 0.16])
    ax_ib = fig.add_axes([0.56, 0.10, 0.22, 0.16])
    ax_back = fig.add_axes([0.80, 0.10, 0.14, 0.16])
    btn_lin = create_small_button(ax_lin, "Manual linear", "#90ee90", "#7cd47c")
    btn_exp = create_small_button(ax_exp, "Single exp", "#ffe680", "#ffcd55")
    btn_ib = create_small_button(ax_ib, "Inactivation", "#ffcc99", "#ffaa66")
    btn_back = create_small_button(ax_back, "Back", "#ddddff", "#bbbbff")
    btn_lin.on_clicked(_set("ManualLinear"))
    btn_exp.on_clicked(_set("Exponential"))
    btn_ib.on_clicked(_set("IB"))
    btn_back.on_clicked(_set("back"))

    plt.show()
    plt.close(fig)
    return pick["value"]


def _run_fit(
    model: str,
    t_full: np.ndarray, y_full: np.ndarray,
    fit_start_idx: int, fit_end_idx: int,
    init_rate_at_t: float,
) -> FitRecord:
    """Run the chosen fit on ``t_full[fit_start_idx : fit_end_idx + 1]`` and
    return a ``FitRecord``.

    For ManualLinear, fits a degree-1 polynomial through the segment.  For
    Exponential / IB, delegates to ``fit_Exponential`` / ``fit_IB`` from
    ``sensorfit.fitting``.

    ``init_rate_at_t`` is the user-chosen absolute time at which the
    "initial rate" is reported; the model's derivative is evaluated there.
    """
    t = t_full[fit_start_idx : fit_end_idx + 1].astype(float)
    y = y_full[fit_start_idx : fit_end_idx + 1].astype(float)

    if model == "ManualLinear":
        if t.size < 2:
            raise ValueError("Manual linear needs ≥2 points.")
        slope, intercept = np.polyfit(t, y, 1)
        yhat = slope * t + intercept
        rss = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - rss / ss_tot if ss_tot > 0 else float("nan")
        rate_at_t = float(slope)
        return FitRecord(
            model="ManualLinear",
            fit_start_s=float(t[0]),
            fit_end_s=float(t[-1]),
            params=[float(slope), float(intercept)],
            param_names=["slope", "intercept"],
            yhat=yhat,
            init_rate_uM_per_s=rate_at_t,
            init_rate_at_t_s=float(init_rate_at_t),
            r2=r2,
            rss=rss,
        )

    if model == "Exponential":
        fr = fit_Exponential(t, y)
        # Derivative of a*t + b + c*exp(-k*(t-t0)) = a - c*k*exp(-k*(t-t0))
        a, _b, c, k_decay, t0 = fr["params"]
        rate_at_t = float(a - c * k_decay * np.exp(-k_decay * (float(init_rate_at_t) - t0)))
        return FitRecord(
            model="Exponential",
            fit_start_s=float(t[0]),
            fit_end_s=float(t[-1]),
            params=[float(x) for x in fr["params"]],
            param_names=list(fr["names"]),
            yhat=fr["yhat"],
            init_rate_uM_per_s=rate_at_t,
            init_rate_at_t_s=float(init_rate_at_t),
            r2=float(fr.get("r2", float("nan"))),
            rss=float(fr.get("rss", float("nan"))),
        )

    if model == "IB":
        fr = fit_IB(t, y)
        # Derivative of IB model at t_start: H(t) = C + H0*exp(-alpha*(1-exp(-kinact*t))) - kslow*t
        # dH/dt = -H0*alpha*kinact*exp(-kinact*t)*exp(-alpha*(1-exp(-kinact*t))) - kslow
        _C, H0, alpha, kinact, kslow = fr["params"]
        ti = float(init_rate_at_t)
        rate_at_t = float(
            -H0 * alpha * kinact * np.exp(-kinact * ti)
            * np.exp(-alpha * (1.0 - np.exp(-kinact * ti)))
            - kslow
        )
        return FitRecord(
            model="IB",
            fit_start_s=float(t[0]),
            fit_end_s=float(t[-1]),
            params=[float(x) for x in fr["params"]],
            param_names=list(fr["names"]),
            yhat=fr["yhat"],
            init_rate_uM_per_s=rate_at_t,
            init_rate_at_t_s=float(init_rate_at_t),
            r2=float(fr.get("r2", float("nan"))),
            rss=float(fr.get("rss", float("nan"))),
        )

    raise ValueError(f"Unknown fit model: {model}")


def prompt_one_fit(
    interval_t: np.ndarray,
    interval_y: np.ndarray,
    filename: str | None = None,
    fit_index: int = 1,
):
    """Drive a single fit on an interval.

    Steps:
      1. Pick model (ManualLinear / Exponential / IB / Back).
      2. Click fit start + end inside the interval.
      3. Show fit + initial rate at start.  Accept / Retry / Back.
      4. Optional back-extrap: deadtime TextBox, click Extrapolate, accept.

    Returns:
      FitRecord — accepted (possibly with back_extrap_applied=True).
      "skip"    — user skipped this fit.
      "back"    — user wants to revisit the subtraction step.
    """
    model = _pick_model()
    if model is None or model == "back":
        return "back"

    # Step 2: pick fit start + end inside the interval
    fig, ax = plt.subplots(figsize=(11, 6.5))
    try:
        fig.canvas.manager.set_window_title(
            "SensorFit — Pick fit range"
            + (f": {truncate_filename(filename)}" if filename else "")
        )
    except Exception:
        pass
    plt.subplots_adjust(left=0.1, bottom=0.20, right=0.98, top=0.78)
    ax.plot(interval_t, interval_y, color="tab:green", lw=1.2, label="Interval")
    title = f"Fit #{fit_index} ({model}): click START then END of the fit range"
    if filename:
        title = f"{truncate_filename(filename)}\n{title}"
    fig.suptitle(title, fontsize=11)

    add_instruction_banner(
        fig,
        f"Click two points inside the interval to bound the {model} fit.  "
        "On Accept the initial rate at your start point is reported and you "
        "may optionally back-extrapolate to find an earlier t=0.",
        y=0.985,
        width=110,
    )

    pending = {"start": None, "end": None}
    markers: list = []
    state = {"action": None, "fit": None}

    def on_click(event):
        if event.button != 1 or event.inaxes is not ax or event.xdata is None:
            return
        if getattr(fig.canvas.toolbar, "mode", "") in ("zoom rect", "pan/zoom", "zoom", "pan"):
            return
        t_click = float(event.xdata)
        if pending["start"] is None:
            pending["start"] = t_click
            m, = ax.plot(t_click, interval_y[int(np.abs(interval_t - t_click).argmin())], "go", ms=10, zorder=5)
            markers.append(m)
            print(f"Fit start: t={t_click:.3f} s")
        elif pending["end"] is None:
            pending["end"] = t_click
            m, = ax.plot(t_click, interval_y[int(np.abs(interval_t - t_click).argmin())], "ro", ms=10, zorder=5)
            markers.append(m)
            print(f"Fit end: t={t_click:.3f} s")
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_click)

    def on_accept(_e=None):
        if pending["start"] is None or pending["end"] is None:
            print("Pick both start and end first.")
            return
        s, e = sorted((pending["start"], pending["end"]))
        start_idx = int(np.abs(interval_t - s).argmin())
        end_idx = int(np.abs(interval_t - e).argmin())
        if end_idx <= start_idx + 1:
            print("Fit range too narrow.")
            return
        try:
            rec = _run_fit(model, interval_t, interval_y, start_idx, end_idx, init_rate_at_t=s)
        except Exception as exc:
            print(f"Fit failed: {exc}")
            return
        state["fit"] = rec
        state["action"] = "accept"
        plt.close(fig)

    def on_retry(_e=None):
        pending["start"] = None
        pending["end"] = None
        while markers:
            try:
                markers.pop().remove()
            except Exception:
                pass
        fig.canvas.draw_idle()
        print("Fit range cleared.")

    def on_skip(_e=None):
        state["action"] = "skip"
        plt.close(fig)

    def on_back(_e=None):
        state["action"] = "back"
        plt.close(fig)

    ax_accept = fig.add_axes([0.10, 0.04, 0.16, 0.05])
    ax_retry = fig.add_axes([0.28, 0.04, 0.12, 0.05])
    ax_skip = fig.add_axes([0.42, 0.04, 0.16, 0.05])
    ax_back = fig.add_axes([0.60, 0.04, 0.12, 0.05])
    _btn_accept_5 = create_small_button(ax_accept, "Accept fit-range", "#90ee90", "#7cd47c")
    _btn_accept_5.on_clicked(on_accept)
    _btn_retry_6 = create_small_button(ax_retry, "Retry", "0.9", "0.8")
    _btn_retry_6.on_clicked(on_retry)
    _btn_skip_7 = create_small_button(ax_skip, "Skip fit", "#ffcc99", "#ffaa66")
    _btn_skip_7.on_clicked(on_skip)
    _btn_back_8 = create_small_button(ax_back, "Back", "#ddddff", "#bbbbff")
    _btn_back_8.on_clicked(on_back)

    install_zoom_keys(fig, ax)
    plt.show()
    plt.close(fig)

    if state["action"] != "accept" or state["fit"] is None:
        return state["action"] or "back"

    fit_rec: FitRecord = state["fit"]

    # Step 3: preview + offer back-extrapolation
    return _preview_fit_with_back_extrap(
        interval_t, interval_y, fit_rec, filename=filename, fit_index=fit_index
    )


def _preview_fit_with_back_extrap(
    interval_t: np.ndarray,
    interval_y: np.ndarray,
    fit_rec: FitRecord,
    filename: str | None = None,
    fit_index: int = 1,
):
    """Show the fit + initial rate, and offer a back-extrap option (TextBox
    defaulting to 1.5 s).  Accept finalises; Skip drops; Back returns.
    """
    fig, ax = plt.subplots(figsize=(11, 6.6))
    try:
        fig.canvas.manager.set_window_title(
            "SensorFit — Fit preview / back-extrap"
            + (f": {truncate_filename(filename)}" if filename else "")
        )
    except Exception:
        pass
    plt.subplots_adjust(left=0.1, bottom=0.22, right=0.98, top=0.78)
    ax.plot(interval_t, interval_y, color="tab:green", lw=1.2, label="Interval")

    fit_t_full = np.linspace(fit_rec.fit_start_s, fit_rec.fit_end_s, fit_rec.yhat.size)
    line_fit, = ax.plot(fit_t_full, fit_rec.yhat, "r-", lw=1.6, label=f"{fit_rec.model} fit")
    extrap_artists = {"line": None, "dot": None}

    title = (
        f"Fit #{fit_index} ({fit_rec.model})  init rate at t={fit_rec.init_rate_at_t_s:.2f}s "
        f"= {fit_rec.init_rate_uM_per_s:.4f} µM/s"
    )
    if filename:
        title = f"{truncate_filename(filename)}\n{title}"
    fig.suptitle(title, fontsize=11)
    ax.legend(loc="best", fontsize=9)

    info_text = fig.text(
        0.5, 0.15,
        "Optional: type a target deadtime (s) and click Extrapolate to preview "
        "a back-extrap to that earlier t=0.",
        ha="center", fontsize=9, color="dimgrey",
    )

    deadtime_state = {"value": 1.5}
    extrap_state = {"new_rate": None, "t_zero": None}

    ax_dead = fig.add_axes([0.18, 0.08, 0.08, 0.05])
    tb_dead = TextBox(ax_dead, "Deadtime (s) ", initial="1.5")

    def on_dead_submit(text: str):
        try:
            v = float(text)
            if v < 0:
                raise ValueError
        except ValueError:
            print(f"Deadtime must be a non-negative number; got '{text}'.")
            tb_dead.set_val(str(deadtime_state["value"]))
            return
        deadtime_state["value"] = v

    tb_dead.on_submit(on_dead_submit)

    def _clear_extrap():
        for k in ("line", "dot"):
            if extrap_artists[k] is not None:
                try:
                    extrap_artists[k].remove()
                except Exception:
                    pass
                extrap_artists[k] = None

    def on_extrapolate(_e=None):
        _clear_extrap()
        dt = float(deadtime_state["value"])
        t_zero = fit_rec.init_rate_at_t_s - dt

        # Evaluate the fitted curve at t_zero (model-dependent)
        if fit_rec.model == "ManualLinear":
            slope, intercept = fit_rec.params
            y_at_tz = slope * t_zero + intercept
            new_rate = slope  # linear → rate is constant
        elif fit_rec.model == "Exponential":
            y_at_tz = float(model_Exponential(np.array([t_zero]), *fit_rec.params)[0])
            a, _b, c, k_decay, t0 = fit_rec.params
            new_rate = float(a - c * k_decay * np.exp(-k_decay * (t_zero - t0)))
        elif fit_rec.model == "IB":
            y_at_tz = float(model_IB(np.array([t_zero]), *fit_rec.params)[0])
            _C, H0, alpha, kinact, kslow = fit_rec.params
            new_rate = float(
                -H0 * alpha * kinact * np.exp(-kinact * t_zero)
                * np.exp(-alpha * (1.0 - np.exp(-kinact * t_zero)))
                - kslow
            )
        else:
            print(f"Don't know how to back-extrap {fit_rec.model}.")
            return

        line, = ax.plot(
            [t_zero, fit_rec.fit_start_s],
            [y_at_tz, fit_rec.yhat[0]],
            "r--", lw=1.6, alpha=0.85, label="Back-extrap",
        )
        dot, = ax.plot([t_zero], [y_at_tz], "o", ms=12, mfc="none", mec="red", mew=2.5)
        extrap_artists["line"] = line
        extrap_artists["dot"] = dot
        extrap_state["new_rate"] = new_rate
        extrap_state["t_zero"] = t_zero
        info_text.set_text(
            f"Back-extrap to t={t_zero:.3f}s (deadtime={dt:.2f}s).  "
            f"New rate at the earlier t=0: {new_rate:.4f} µM/s.  Accept to record."
        )
        info_text.set_color("black")
        ax.legend(loc="best", fontsize=9)
        fig.canvas.draw_idle()

    decision = {"value": None}

    def on_accept(_e=None):
        decision["value"] = "accept"
        plt.close(fig)

    def on_retry(_e=None):
        decision["value"] = "retry"
        plt.close(fig)

    def on_skip(_e=None):
        decision["value"] = "skip"
        plt.close(fig)

    def on_back(_e=None):
        decision["value"] = "back"
        plt.close(fig)

    ax_extrapolate = fig.add_axes([0.30, 0.08, 0.14, 0.05])
    ax_accept = fig.add_axes([0.48, 0.08, 0.14, 0.05])
    ax_retry = fig.add_axes([0.64, 0.08, 0.10, 0.05])
    ax_skip = fig.add_axes([0.76, 0.08, 0.10, 0.05])
    ax_back = fig.add_axes([0.88, 0.08, 0.10, 0.05])
    _btn_extrapolate_9 = create_small_button(ax_extrapolate, "Extrapolate", "#ffe680", "#ffcd55")
    _btn_extrapolate_9.on_clicked(on_extrapolate)
    _btn_accept_10 = create_small_button(ax_accept, "Accept fit", "#90ee90", "#7cd47c")
    _btn_accept_10.on_clicked(on_accept)
    _btn_retry_11 = create_small_button(ax_retry, "Retry", "0.9", "0.8")
    _btn_retry_11.on_clicked(on_retry)
    _btn_skip_12 = create_small_button(ax_skip, "Skip", "#ffcc99", "#ffaa66")
    _btn_skip_12.on_clicked(on_skip)
    _btn_back_13 = create_small_button(ax_back, "Back", "#ddddff", "#bbbbff")
    _btn_back_13.on_clicked(on_back)

    install_zoom_keys(fig, ax)
    plt.show()
    plt.close(fig)

    if decision["value"] == "accept":
        if extrap_state["new_rate"] is not None:
            fit_rec.back_extrap_applied = True
            fit_rec.back_extrap_deadtime_s = float(deadtime_state["value"])
            fit_rec.back_extrap_t0_s = float(extrap_state["t_zero"])
            fit_rec.back_extrap_rate_uM_per_s = float(extrap_state["new_rate"])
        return fit_rec
    return decision["value"] or "back"


# ────────────────────────────────────────────────────────────────────────
# UI: Δmax screen (three modes + selectable t_zero)
# ────────────────────────────────────────────────────────────────────────


def prompt_delta_max(
    interval_t: np.ndarray,
    interval_y: np.ndarray,
    fits: list[FitRecord],
    filename: str | None = None,
):
    """Drive one Δmax estimation for the interval.

    Three modes, chosen at the top of the screen:
      - From fit  — uses the last fit (if any); Δ at t_zero = c*exp(-k*(t_zero-t0))
                    for Exponential; H0*exp(...) for IB.
      - Linear    — user clicks two points to draw a line; line's y at t_zero
                    minus y at the end of the interval = Δmax.
      - Point     — user clicks ONE point; its y minus y at the end = Δmax.

    In all three modes the user FIRST clicks the t_zero anchor (single click).
    """
    if not fits:
        # Can't do "from-fit" without any fit
        default_mode = "linear"
    else:
        default_mode = "from-fit"

    fig, ax = plt.subplots(figsize=(11, 6.8))
    try:
        fig.canvas.manager.set_window_title(
            "SensorFit — Δ[H₂O₂]max"
            + (f": {truncate_filename(filename)}" if filename else "")
        )
    except Exception:
        pass
    plt.subplots_adjust(left=0.1, bottom=0.22, right=0.98, top=0.74)
    ax.plot(interval_t, interval_y, color="tab:green", lw=1.2, label="Interval")
    if fits:
        for i, f in enumerate(fits):
            t_seg = np.linspace(f.fit_start_s, f.fit_end_s, f.yhat.size)
            ax.plot(t_seg, f.yhat, "-", lw=1.0, alpha=0.7, label=f"fit#{i+1}:{f.model}")
    ax.legend(loc="best", fontsize=8)

    title = "Δ[H₂O₂]max selection"
    if filename:
        title = f"{truncate_filename(filename)}\n{title}"
    fig.suptitle(title, fontsize=11)

    add_instruction_banner(
        fig,
        "Top: choose mode (From fit | Linear | Point).  Then click t=0 once.  "
        "For Linear, additionally click 2 points to define the line.  For Point, "
        "the t=0 click also picks the y-value.  Accept records Δmax.",
        y=0.985,
        width=110,
    )

    mode_state = {"mode": default_mode}
    state = {
        "t_zero_idx": None,
        "linear_clicks": [],  # for Linear mode
        "point_idx": None,    # for Point mode
        "computed": None,     # DeltaMaxRecord on success
        "action": None,
        "markers": [],
        "preview": [],
    }

    btn_axes = {
        "from-fit": fig.add_axes([0.10, 0.86, 0.13, 0.05]),
        "linear":   fig.add_axes([0.24, 0.86, 0.13, 0.05]),
        "point":    fig.add_axes([0.38, 0.86, 0.13, 0.05]),
    }
    btns = {
        m: create_small_button(
            btn_axes[m],
            {"from-fit": "From fit", "linear": "Linear", "point": "Point"}[m],
            "#90ee90" if m == default_mode else "0.85",
            "#7cd47c",
        )
        for m in ("from-fit", "linear", "point")
    }

    def _clear():
        for marker in state["markers"]:
            try:
                marker.remove()
            except Exception:
                pass
        for art in state["preview"]:
            try:
                art.remove()
            except Exception:
                pass
        state["markers"].clear()
        state["preview"].clear()
        state["t_zero_idx"] = None
        state["linear_clicks"].clear()
        state["point_idx"] = None
        state["computed"] = None
        fig.canvas.draw_idle()

    def _set_mode(m):
        def _f(_e=None):
            mode_state["mode"] = m
            for k in btns:
                btns[k].color = "#90ee90" if k == m else "0.85"
            _clear()
            print(f"Δmax mode → {m}")
        return _f

    for m in btns:
        btns[m].on_clicked(_set_mode(m))

    def _compute_and_preview():
        # Need a t_zero in all modes
        tz_idx = state["t_zero_idx"]
        if tz_idx is None:
            return
        t_zero = float(interval_t[tz_idx])
        y_end = float(interval_y[-1])

        if mode_state["mode"] == "from-fit":
            if not fits:
                print("No fit available; pick Linear or Point mode.")
                return
            rec_fit = fits[-1]
            delta = delta_max_from_fit(rec_fit, t_zero)
            if not np.isfinite(delta):
                print(f"Cannot compute Δmax from {rec_fit.model} fit.")
                return
            state["computed"] = DeltaMaxRecord(
                method="from-fit", t_zero_s=t_zero, value_uM=float(delta),
            )

        elif mode_state["mode"] == "linear":
            if len(state["linear_clicks"]) != 2:
                return  # waiting for both line clicks
            i0, i1 = sorted(state["linear_clicks"])
            try:
                delta, slope, intercept = delta_max_from_linear(
                    interval_t, interval_y, i0, i1, t_zero
                )
            except ValueError as exc:
                print(f"Linear Δmax: {exc}")
                return
            # Draw the fit line + extension to t_zero
            t_line = np.linspace(min(t_zero, interval_t[i0]), interval_t[i1], 100)
            y_line = slope * t_line + intercept
            ln, = ax.plot(t_line, y_line, "r--", lw=1.4, label="Linear")
            state["preview"].append(ln)
            state["computed"] = DeltaMaxRecord(
                method="linear", t_zero_s=t_zero, value_uM=float(delta),
                linear_slope=slope, linear_intercept=intercept,
            )

        elif mode_state["mode"] == "point":
            if state["point_idx"] is None:
                return
            y_at_tzero = float(interval_y[state["point_idx"]])
            delta = delta_max_from_point(y_at_tzero, y_end)
            state["computed"] = DeltaMaxRecord(
                method="point", t_zero_s=t_zero, value_uM=float(delta),
            )

        else:
            return

        # Preview text on the figure
        for art in state["preview"]:
            pass  # already drawn
        ax.axvline(t_zero, color="#8B008B", lw=1.4, ls="--", alpha=0.85)
        fig.canvas.draw_idle()
        rec = state["computed"]
        print(f"Δmax preview: method={rec.method}, t_zero={t_zero:.3f}s → {rec.value_uM:.4f} µM")

    def on_click(event):
        if event.button != 1 or event.inaxes is not ax or event.xdata is None:
            return
        if getattr(fig.canvas.toolbar, "mode", "") in ("zoom rect", "pan/zoom", "zoom", "pan"):
            return
        idx = int(np.abs(interval_t - float(event.xdata)).argmin())

        if mode_state["mode"] == "from-fit":
            # Only need t_zero click
            state["t_zero_idx"] = idx
            m, = ax.plot(interval_t[idx], interval_y[idx], "P", ms=12, color="#8B008B", zorder=5)
            state["markers"].append(m)
            _compute_and_preview()

        elif mode_state["mode"] == "linear":
            # First click = t_zero; next two = line endpoints
            if state["t_zero_idx"] is None:
                state["t_zero_idx"] = idx
                m, = ax.plot(interval_t[idx], interval_y[idx], "P", ms=12, color="#8B008B", zorder=5)
                state["markers"].append(m)
                print("Linear Δmax: now click two points to define the line.")
            elif len(state["linear_clicks"]) < 2:
                state["linear_clicks"].append(idx)
                m, = ax.plot(interval_t[idx], interval_y[idx], "ro", ms=10, zorder=5)
                state["markers"].append(m)
                if len(state["linear_clicks"]) == 2:
                    _compute_and_preview()

        elif mode_state["mode"] == "point":
            # Click serves as BOTH t_zero AND the y-value
            state["t_zero_idx"] = idx
            state["point_idx"] = idx
            m, = ax.plot(interval_t[idx], interval_y[idx], "P", ms=12, color="#8B008B", zorder=5)
            state["markers"].append(m)
            _compute_and_preview()

    fig.canvas.mpl_connect("button_press_event", on_click)

    def on_accept(_e=None):
        if state["computed"] is None:
            print("No Δmax computed yet.")
            return
        state["action"] = "accept"
        plt.close(fig)

    def on_retry(_e=None):
        _clear()

    def on_skip(_e=None):
        state["action"] = "skip"
        plt.close(fig)

    def on_back(_e=None):
        state["action"] = "back"
        plt.close(fig)

    ax_accept = fig.add_axes([0.12, 0.04, 0.16, 0.05])
    ax_retry = fig.add_axes([0.30, 0.04, 0.12, 0.05])
    ax_skip = fig.add_axes([0.44, 0.04, 0.16, 0.05])
    ax_back = fig.add_axes([0.62, 0.04, 0.12, 0.05])
    _btn_accept_14 = create_small_button(ax_accept, "Accept Δmax", "#90ee90", "#7cd47c")
    _btn_accept_14.on_clicked(on_accept)
    _btn_retry_15 = create_small_button(ax_retry, "Retry", "0.9", "0.8")
    _btn_retry_15.on_clicked(on_retry)
    _btn_skip_16 = create_small_button(ax_skip, "Skip Δmax", "#ffcc99", "#ffaa66")
    _btn_skip_16.on_clicked(on_skip)
    _btn_back_17 = create_small_button(ax_back, "Back", "#ddddff", "#bbbbff")
    _btn_back_17.on_clicked(on_back)

    install_zoom_keys(fig, ax)
    plt.show()
    plt.close(fig)

    if state["action"] == "accept" and state["computed"] is not None:
        return state["computed"]
    return state["action"] or "skip"


# ────────────────────────────────────────────────────────────────────────
# UI: new-control modal (file picker + stripped processing flow)
# ────────────────────────────────────────────────────────────────────────


def _qt_pick_control_file(input_dir: Path | None) -> Path | None:
    """Open a Qt file dialog and let the user pick a control file.  Falls back
    to None if Qt isn't available."""
    from .calibration_editor import _try_import_qt, QT_AVAILABLE, QT_LIB
    if not QT_AVAILABLE:
        ok, _, _ = _try_import_qt()
        if not ok:
            print("Qt not available; cannot launch file-picker.  Skipping.")
            return None

    if QT_LIB == "PyQt5":
        from PyQt5.QtWidgets import QApplication, QFileDialog
    else:
        from PySide6.QtWidgets import QApplication, QFileDialog

    import sys as _sys
    app = QApplication.instance() or QApplication(_sys.argv)
    start_dir = str(input_dir) if input_dir is not None else ""
    selected, _ = QFileDialog.getOpenFileName(
        None,
        "SensorFit — pick a control file to process",
        start_dir,
        "Excel / CSV (*.xlsx *.xls *.xlsm *.xlsb *.csv *.txt);;All files (*)",
    )
    return Path(selected) if selected else None


# ────────────────────────────────────────────────────────────────────────
# Top-level orchestrator
# ────────────────────────────────────────────────────────────────────────


def run_per_interval_flow(
    time_values: np.ndarray,
    h2o2_values: np.ndarray,
    time_col: str,
    full_frame: pd.DataFrame,
    filename: str | None = None,
    calibrated_dir: Path | None = None,
    session_intervals: list[ProcessedInterval] | None = None,
    new_control_callback=None,
) -> list[ProcessedInterval] | str:
    """Drive the per-interval state machine until the user says "Done".

    ``new_control_callback`` is an optional callable
    ``(picked_path: Path) -> (t: np.ndarray, y: np.ndarray) | None`` that
    the caller provides to process a new control file modal-style and return
    its reference interval.  If None, the "Subtract new" option will tell the
    user to use the grouping UI instead.

    Returns the list of accepted ``ProcessedInterval``s, or the sentinel
    string ``"go_back_to_calibration"`` if the user asked to back out before
    any interval was accepted.
    """
    if session_intervals is None:
        session_intervals = []

    accepted: list[ProcessedInterval] = []
    interval_counter = 1
    self_stem = Path(filename).stem if filename else None

    while True:
        # ── 1) Pick the interval ─────────────────────────────────────
        already = [(p.start_time, p.end_time) for p in accepted]
        sel = select_one_interval(
            time_values, h2o2_values, already,
            filename=filename, interval_number=interval_counter,
        )
        if sel == "done" or sel == "skip":
            if not accepted:
                # User clicked Done before adding anything; that's "no intervals"
                return []
            break
        if sel == "back":
            if not accepted:
                return "go_back_to_calibration"
            # If there are accepted intervals, "Back" just re-opens this step
            continue
        if not isinstance(sel, tuple):
            continue

        s_t, e_t = sel
        mask = (time_values >= s_t) & (time_values <= e_t)
        if not np.any(mask):
            print("Empty interval; pick again.")
            continue

        # Build a working DataFrame for the interval
        interval_df = full_frame.loc[mask, [time_col, CALIBRATED_COLUMN]].copy().reset_index(drop=True)
        interval_t = interval_df[time_col].to_numpy(dtype=float)
        interval_y = interval_df[CALIBRATED_COLUMN].to_numpy(dtype=float)

        # ── 2) Subtraction prompt ────────────────────────────────────
        sub_choice = prompt_subtraction_choice(
            f"Interval #{interval_counter}: [{s_t:.1f}–{e_t:.1f} s]"
        )
        control_subtracted = False
        control_source = None

        if sub_choice == "back":
            # Re-open the interval-selection step
            continue

        if sub_choice == "existing":
            candidates = _discover_existing_control_intervals(
                calibrated_dir, session_intervals, self_stem
            )
            picked = pick_existing_interval(candidates, filename=filename)
            if picked is not None:
                ct, cy = picked["load"]()
                # zero-base the control so its first time aligns with sample's first time
                ct_zero = ct - ct[0]
                result = preview_per_interval_subtraction(
                    interval_t, interval_y, ct_zero, cy,
                    filename=filename, label=picked["label"],
                )
                if isinstance(result, np.ndarray):
                    interval_y = result
                    interval_df[CALIBRATED_COLUMN] = result
                    control_subtracted = True
                    control_source = picked["label"]
                elif result == "back":
                    # Re-open subtraction-choice
                    continue

        elif sub_choice == "new":
            if new_control_callback is None:
                print(
                    "No callback for new-control processing was provided.  "
                    "Falling back to Skip — process the control as a normal "
                    "file and then choose 'Subtract existing'."
                )
            else:
                picked_path = _qt_pick_control_file(
                    calibrated_dir.parent if calibrated_dir is not None else None
                )
                if picked_path is not None:
                    template = new_control_callback(picked_path)
                    if template is not None:
                        ct, cy = template
                        ct_zero = ct - ct[0]
                        result = preview_per_interval_subtraction(
                            interval_t, interval_y, ct_zero, cy,
                            filename=filename, label=f"new: {picked_path.name}",
                        )
                        if isinstance(result, np.ndarray):
                            interval_y = result
                            interval_df[CALIBRATED_COLUMN] = result
                            control_subtracted = True
                            control_source = f"new: {picked_path.name}"
                        elif result == "back":
                            continue

        # ── 3) Multi-fit loop ────────────────────────────────────────
        fits: list[FitRecord] = []
        fit_index = 1
        while True:
            print(f"\nFitting interval #{interval_counter} — fit #{fit_index}")
            res = prompt_one_fit(
                interval_t, interval_y,
                filename=filename, fit_index=fit_index,
            )
            if isinstance(res, FitRecord):
                fits.append(res)
                print(f"  ✓ Fit #{fit_index} ({res.model}) accepted.")
                if not _ask_another("Apply another fit to this interval?"):
                    break
                fit_index += 1
                continue
            if res == "skip":
                print("  → fit skipped.")
                break
            if res == "retry":
                continue
            if res == "back":
                # Back from fitting → revisit subtraction (re-open whole prompt)
                fits.clear()
                break

        # ── 4) Δmax (one per interval, optional) ─────────────────────
        delta = None
        dres = prompt_delta_max(interval_t, interval_y, fits, filename=filename)
        if isinstance(dres, DeltaMaxRecord):
            delta = dres
            print(f"  ✓ Δmax recorded ({delta.method}: {delta.value_uM:.4f} µM)")

        accepted.append(ProcessedInterval(
            index=interval_counter,
            start_time=float(s_t),
            end_time=float(e_t),
            data=interval_df,
            time_col=time_col,
            control_subtracted=control_subtracted,
            control_source=control_source,
            fits=fits,
            delta_max=delta,
        ))
        interval_counter += 1

        if not _ask_another("Define another interval?"):
            break

    return accepted


def _ask_another(question: str) -> bool:
    """Small yes/no popup; returns True if user clicks Yes."""
    fig, ax = plt.subplots(figsize=(7, 2.5))
    try:
        fig.canvas.manager.set_window_title("SensorFit — Continue?")
    except Exception:
        pass
    ax.axis("off")
    ax.text(0.5, 0.55, question, ha="center", va="center", fontsize=11)
    state = {"choice": False}

    def _yes(_e=None):
        state["choice"] = True
        plt.close(fig)

    def _no(_e=None):
        state["choice"] = False
        plt.close(fig)

    ax_yes = fig.add_axes([0.22, 0.10, 0.22, 0.18])
    ax_no = fig.add_axes([0.56, 0.10, 0.22, 0.18])
    _btn_yes_18 = create_small_button(ax_yes, "Yes", "#90ee90", "#7cd47c")
    _btn_yes_18.on_clicked(_yes)
    _btn_no_19 = create_small_button(ax_no, "No", "0.9", "0.8")
    _btn_no_19.on_clicked(_no)
    plt.show()
    plt.close(fig)
    return state["choice"]
