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
    apply_robust_ylim,
    average_window,
    create_small_button,
    truncate_filename,
)
from .controls import (
    align_control_offset,
    average_controls_on_grid,
    deviation_from_anchor,
    interpolate_control_to_grid,
    suggest_anchor_time,
)
from .fitting import fit_IB, fit_Exponential
from .models import model_Exponential, model_IB
from .window import get_window, finish_window
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
    control_n_averaged: int = 0        # how many controls were averaged (0 = none)
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
    t1: float, y1: float,
    t2: float, y2: float,
    t_zero: float,
    max_uM: float = 100.0,
) -> tuple[float, float, float]:
    """Δmax from a straight line through TWO user-picked points.

    Each ``(t_i, y_i)`` is typically already averaged over a window of
    samples around the user's click (so noise on individual samples
    doesn't dominate).  The line through both is evaluated at ``t_zero``;
    Δmax = ``max_uM − y_line(t_zero)``.

    Returns ``(delta_max_uM, slope, intercept_at_0)``.
    """
    if abs(t2 - t1) < 1e-12:
        raise ValueError("Linear Δmax needs two points at different t-values.")
    slope = (float(y2) - float(y1)) / (float(t2) - float(t1))
    intercept_at_0 = float(y1) - slope * float(t1)
    y_at_tzero = slope * float(t_zero) + intercept_at_0
    return float(max_uM) - y_at_tzero, float(slope), float(intercept_at_0)


def delta_max_from_point(y_at_tzero: float, max_uM: float = 100.0) -> float:
    """Single-point Δmax: ``max_uM − y(t_zero)``.

    ``y_at_tzero`` is the windowed-average of the user's click (using the
    Δmax window size, defaulting to ±50 samples).  Δmax is the height
    from there up to ``max_uM``.
    """
    return float(max_uM) - float(y_at_tzero)


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
    fig = get_window().reset(figsize=(11, 6.5))
    ax = fig.add_subplot(111)
    try:
        fig.canvas.manager.set_window_title(
            "SensorFit — Pick an interval"
            + (f": {truncate_filename(filename)}" if filename else "")
        )
    except Exception:
        pass
    plt.subplots_adjust(left=0.1, bottom=0.18, right=0.98, top=0.78)
    ax.plot(time_values, h2o2_values, color="tab:green", lw=1.2, label="Calibrated trace")
    # Keep the injection transient from compressing the trace (see
    # apply_robust_ylim); "r" restores this view, not the spike-wide one.
    apply_robust_ylim(ax, h2o2_values)
    for (s_existing, e_existing) in already_defined:
        ax.axvspan(s_existing, e_existing, color="grey", alpha=0.18)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("H2O2 (µM)")

    # Banner is anchored left-of-centre and kept narrow so it stays clear of
    # the top-right "Done with intervals" button (users have hit Done by
    # mistake — spatial separation is the whole point).
    import textwrap as _tw
    banner_text = _tw.fill(
        "An interval is one experimental run — a stretch of trace you want to fit "
        "(e.g. from H₂O₂ addition to end of consumption).  "
        "Click START then END; Accept to keep it, Retry to re-click, "
        "Skip to move on without saving.",
        width=90,
    )
    fig.text(
        0.42, 0.985, banner_text,
        ha="center", va="top", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.9, pad=0.4),
    )

    title = f"Pick interval #{interval_number}"
    if filename:
        title = f"{truncate_filename(filename)} — {title}"
    # Title sits just above the axes, left-aligned so it doesn't collide
    # with the banner or the top-right Done button.
    fig.text(0.10, 0.86, title, fontsize=11, fontweight="bold", ha="left", va="top")

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
        get_window().stop()

    def on_done(_e=None):
        state["action"] = "done"
        get_window().stop()

    def on_back(_e=None):
        state["action"] = "back"
        get_window().stop()

    def on_skip(_e=None):
        state["action"] = "skip"
        get_window().stop()

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

    def on_remove_last(_e=None):
        if not already_defined:
            print("No previously-defined intervals to remove.")
            return
        state["action"] = "remove_last"
        get_window().stop()

    # Two rows of buttons, with "Done with intervals" deliberately placed
    # in the top-right corner, well away from the frequently-clicked
    # Accept / Retry / Skip cluster along the bottom — users kept hitting
    # it by mistake when it sat between Retry and Skip and it would end the
    # per-file interval flow entirely (no undo).
    #
    # Bottom row (per-interval actions):
    #   Accept | Retry | Skip this one | Remove last | Back → calibration
    # Top-right (whole-file action, spatially separated):
    #   Done with intervals
    ax_accept = fig.add_axes([0.08, 0.03, 0.13, 0.05])
    ax_retry = fig.add_axes([0.22, 0.03, 0.10, 0.05])
    ax_skip = fig.add_axes([0.33, 0.03, 0.13, 0.05])
    ax_remove = fig.add_axes([0.47, 0.03, 0.13, 0.05])
    ax_back = fig.add_axes([0.61, 0.03, 0.15, 0.05])
    ax_done = fig.add_axes([0.83, 0.945, 0.15, 0.045])
    btn_accept = create_small_button(ax_accept, "Accept", "#90ee90", "#7cd47c")
    btn_retry = create_small_button(ax_retry, "Retry", "0.9", "0.8")
    btn_skip = create_small_button(ax_skip, "Skip this one", "#ffcc99", "#ffaa66")
    # Remove last is greyed out (slightly) when there are no accepted intervals.
    remove_colour = "#ffaaaa" if already_defined else "0.85"
    remove_hover = "#ff8888" if already_defined else "0.85"
    btn_remove = create_small_button(ax_remove, "Remove last", remove_colour, remove_hover)
    btn_back = create_small_button(ax_back, "Back → calibration", "#ddddff", "#bbbbff")
    # Give "Done" a distinct muted-purple palette so it is visually as well
    # as spatially separated from the green/orange per-interval buttons.
    btn_done = create_small_button(ax_done, "✓ Done with intervals", "#c8b8e0", "#a898c8")
    btn_accept.on_clicked(on_accept)
    btn_retry.on_clicked(on_retry)
    btn_done.on_clicked(on_done)
    btn_skip.on_clicked(on_skip)
    btn_remove.on_clicked(on_remove_last)
    btn_back.on_clicked(on_back)

    install_zoom_keys(fig, ax)
    get_window().run()

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

    "Existing" and "new" both lead to the multi-control averaging hub,
    which lets the user accumulate one or more controls before accepting.
    """
    fig = get_window().reset(figsize=(9.2, 4.8))
    ax = fig.add_subplot(111)
    try:
        fig.canvas.manager.set_window_title("SensorFit — Subtraction choice")
    except Exception:
        pass
    ax.axis("off")
    ax.text(
        0.5, 0.88,
        interval_summary,
        ha="center", va="center", fontsize=10, style="italic", color="dimgrey",
    )
    ax.text(
        0.5, 0.72,
        "Only use control subtraction if you have a matched control run\n"
        "(e.g. no-enzyme, no-substrate) that spans a similar time to this interval.\n"
        "It removes electrode drift/background so what's left is the enzymatic signal.\n"
        "If you don't have a control, choose 'None' — this is the normal case.",
        ha="center", va="center", fontsize=10, color="#333333",
    )
    ax.text(
        0.06, 0.40,
        "• None — use this interval as-is (no control available or not needed).\n"
        "• Subtract existing — pick a control interval already processed in this\n"
        "   session or saved from a previous session.\n"
        "• Subtract new — pick a control file from disk and process it now, then\n"
        "   return here to subtract it.\n"
        "Both 'existing' and 'new' let you add more than one control; they will be\n"
        "averaged before subtraction.",
        ha="left", va="center", fontsize=9.5,
    )
    fig.suptitle(
        "Control subtraction — apply a no-enzyme (or similar) control to this interval?",
        fontsize=11, fontweight="bold",
    )
    choice = {"value": None}

    def _set(v):
        def _f(_e=None):
            choice["value"] = v
            get_window().stop()
        return _f

    ax_none = fig.add_axes([0.07, 0.05, 0.18, 0.13])
    ax_existing = fig.add_axes([0.27, 0.05, 0.20, 0.13])
    ax_new = fig.add_axes([0.49, 0.05, 0.18, 0.13])
    ax_back = fig.add_axes([0.75, 0.05, 0.18, 0.13])
    btn_none = create_small_button(ax_none, "None", "#90ee90", "#7cd47c")
    btn_existing = create_small_button(ax_existing, "Subtract existing", "#ffe680", "#ffcd55")
    btn_new = create_small_button(ax_new, "Subtract new", "#ffcc99", "#ffaa66")
    btn_back = create_small_button(ax_back, "Back", "#ddddff", "#bbbbff")
    btn_none.on_clicked(_set("none"))
    btn_existing.on_clicked(_set("existing"))
    btn_new.on_clicked(_set("new"))
    btn_back.on_clicked(_set("back"))

    get_window().run()
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


# ────────────────────────────────────────────────────────────────────────
# UI: subtraction preview (used by the averaging hub's single-control
# legacy path is gone; this is kept for potential direct use)
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
    # Shift the control so its H₂O₂ injection lines up with the sample's,
    # then anchor a little after that injection.  Aligning at the interval
    # starts instead would read the control on its pre-injection baseline
    # and subtract the whole injection step rather than the drift.
    try:
        _shift = align_control_offset(sample_t, sample_y, control_t, control_y)
    except Exception:
        _shift = 0.0
    control_t = np.asarray(control_t, dtype=float) + _shift
    anchor = {"t0": suggest_anchor_time(sample_t, sample_y)}
    place_t0 = float(sample_t[0])

    fig = get_window().reset(figsize=(11, 7.4))
    ax_top = fig.add_subplot(211)
    ax_bot = fig.add_subplot(212, sharex=ax_top)
    try:
        fig.canvas.manager.set_window_title(
            "SensorFit — Subtraction preview"
            + (f": {truncate_filename(filename)}" if filename else "")
        )
    except Exception:
        pass
    plt.subplots_adjust(left=0.1, bottom=0.18, right=0.98, top=0.78, hspace=0.25)

    ax_top.plot(sample_t, sample_y, color="tab:green", lw=1.3, label="Sample interval")
    apply_robust_ylim(ax_top, sample_y, control_y)
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

    # Autoscale only on the very first draw.  On subsequent re-anchors the
    # user's current view (including any zoom they set to pick t carefully)
    # is preserved — pressing 'r' resets the view if they want it back.
    view = {"autoscaled": False}

    def redraw():
        shifted_t = control_t + place_t0
        interp, _ = interpolate_control_to_grid(shifted_t, control_y, sample_t)
        dev = deviation_from_anchor(interp, sample_t, anchor["t0"])
        line_ctrl.set_data(sample_t, interp)
        line_corr.set_data(sample_t, sample_y - dev)
        anchor_line_top.set_xdata([anchor["t0"], anchor["t0"]])
        anchor_line_bot.set_xdata([anchor["t0"], anchor["t0"]])
        if not view["autoscaled"]:
            for a in (ax_top, ax_bot):
                a.relim()
                a.autoscale_view()
            view["autoscaled"] = True
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
        get_window().stop()

    def on_skip(_e=None):
        decision["value"] = "skip"
        get_window().stop()

    def on_back(_e=None):
        decision["value"] = "back"
        get_window().stop()

    def on_reset(_e=None):
        anchor["t0"] = suggest_anchor_time(sample_t, sample_y)
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
    get_window().run()

    if decision["value"] == "accept":
        shifted_t = control_t + place_t0
        interp, _ = interpolate_control_to_grid(shifted_t, control_y, sample_t)
        dev = deviation_from_anchor(interp, sample_t, anchor["t0"])
        return sample_y - dev
    if decision["value"] == "skip":
        return "skip"
    return "back"


# ────────────────────────────────────────────────────────────────────────
# UI: multi-select existing-interval picker
# ────────────────────────────────────────────────────────────────────────

def pick_existing_intervals_multi(
    candidates: list[dict],
    filename: str | None = None,
) -> list[dict] | None:
    """Let the user pick one or more existing intervals from a scrollable list.

    Uses a Qt ``QListWidget`` dialog (natively scrollable, handles hundreds
    of items).  Returns a list of chosen candidate dicts, or None if
    cancelled / no candidates.
    """
    if not candidates:
        print("No existing intervals available for subtraction.")
        return None

    return _qt_multi_select_dialog(
        candidates,
        title=(
            "Select one or more control intervals"
            + (f" for {truncate_filename(filename)}" if filename else "")
        ),
        instructions=(
            "Click to select (Ctrl/Cmd-click or Shift-click for multiple).  "
            "Accept adds all selected controls to the averaging set."
        ),
        subtracting_from=filename,
    )


def _qt_multi_select_dialog(
    candidates: list[dict],
    title: str = "Select intervals",
    instructions: str = "",
    subtracting_from: str | None = None,
) -> list[dict] | None:
    """Scrollable multi-select dialog backed by Qt.

    ``subtracting_from`` names the run these controls will be subtracted
    from; it is shown as a prominent banner so the choice of control is
    never made against a half-remembered sample.
    """
    import sys as _sys
    try:
        from .calibration_editor import _try_import_qt, QT_AVAILABLE, QT_LIB
    except Exception:
        QT_AVAILABLE = False
        QT_LIB = None

    if not QT_AVAILABLE:
        try:
            ok, _, QT_LIB = _try_import_qt()
        except Exception:
            ok = False
        if not ok:
            print("Qt not available; falling back to console selection.")
            return _console_multi_select(candidates, subtracting_from)

    if QT_LIB == "PyQt5":
        from PyQt5 import QtWidgets, QtCore
    else:
        from PySide6 import QtWidgets, QtCore

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(_sys.argv)

    dlg = QtWidgets.QDialog()
    dlg.setWindowTitle(title)
    dlg.resize(620, 460)
    layout = QtWidgets.QVBoxLayout(dlg)

    if subtracting_from:
        banner = QtWidgets.QLabel(
            f"Choosing a control to subtract FROM:\n{subtracting_from}"
        )
        banner.setWordWrap(True)
        banner.setStyleSheet(
            "QLabel {"
            " background-color: #fff3cd;"
            " border: 2px solid #d39e00;"
            " border-radius: 4px;"
            " padding: 8px;"
            " font-size: 13px;"
            " font-weight: bold;"
            " color: #5c4400;"
            "}"
        )
        layout.addWidget(banner)

    if instructions:
        lbl = QtWidgets.QLabel(instructions)
        lbl.setWordWrap(True)
        layout.addWidget(lbl)

    lw = QtWidgets.QListWidget()
    lw.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
    for c in candidates:
        lw.addItem(c["label"])
    layout.addWidget(lw)

    btn_box = QtWidgets.QHBoxLayout()
    btn_accept = QtWidgets.QPushButton("Accept selection")
    btn_cancel = QtWidgets.QPushButton("Cancel")
    btn_accept.clicked.connect(dlg.accept)
    btn_cancel.clicked.connect(dlg.reject)
    btn_box.addWidget(btn_accept)
    btn_box.addWidget(btn_cancel)
    layout.addLayout(btn_box)

    if dlg.exec_() == QtWidgets.QDialog.Accepted:
        indices = sorted(idx.row() for idx in lw.selectedIndexes())
        if indices:
            return [candidates[i] for i in indices]
    return None


def _console_multi_select(
    candidates: list[dict],
    subtracting_from: str | None = None,
) -> list[dict] | None:
    """Fallback when Qt is unavailable: numbered console list."""
    if subtracting_from:
        print(f"\n>>> Choosing a control to subtract FROM: {subtracting_from}")
    print("\nAvailable intervals:")
    for i, c in enumerate(candidates):
        print(f"  [{i}] {c['label']}")
    raw = input("Enter indices (comma-separated) or 'c' to cancel: ").strip()
    if raw.lower() == "c":
        return None
    try:
        indices = [int(x.strip()) for x in raw.split(",")]
        chosen = [candidates[i] for i in indices if 0 <= i < len(candidates)]
        return chosen if chosen else None
    except (ValueError, IndexError):
        print("Invalid input; cancelling.")
        return None


# ────────────────────────────────────────────────────────────────────────
# UI: multi-control averaging hub (live preview)
# ────────────────────────────────────────────────────────────────────────

ControlMember = tuple  # (label: str, t_zero_based: ndarray, y: ndarray)


def averaging_hub(
    sample_t: np.ndarray,
    sample_y: np.ndarray,
    initial_members: list[ControlMember],
    filename: str | None = None,
    *,
    existing_candidates: list[dict] | None = None,
    new_control_callback=None,
    calibrated_dir: Path | None = None,
) -> tuple[str, list[ControlMember]]:
    """Show a live-preview hub where the user builds an averaging set.

    Top pane: sample (green), each member control faint, averaged control
    bold red, anchor line.  Bottom pane: sample − averaged control (blue).

    Buttons: Add existing / Add new / Remove last / Accept & subtract /
    Skip / Back.

    Each control is time-shifted so its H₂O₂ injection coincides with the
    sample's (see ``align_control_offset``); without that the anchor lands
    in the control's pre-injection baseline and the whole injection step
    is subtracted instead of just the drift.

    Returns ``(decision, members, anchor_t)`` where decision is
    ``"accept"``, ``"skip"``, or ``"back"``.  The returned members carry
    their alignment shift baked into their time arrays, and ``anchor_t``
    is the deviation reference the user settled on — callers must reuse
    both so the applied subtraction matches this preview.
    """

    def _aligned(member: ControlMember) -> ControlMember:
        """Shift a member's time base so its injection matches the sample's."""
        label, ct, cy = member
        try:
            shift = align_control_offset(sample_t, sample_y, ct, cy)
        except Exception:
            shift = 0.0
        return (label, np.asarray(ct, dtype=float) + shift, cy)

    members: list[ControlMember] = [_aligned(m) for m in initial_members]
    anchor = {"t0": suggest_anchor_time(sample_t, sample_y)}

    fig = get_window().reset(figsize=(11, 7.8))
    ax_top = fig.add_subplot(211)
    ax_bot = fig.add_subplot(212, sharex=ax_top)
    try:
        fig.canvas.manager.set_window_title(
            "SensorFit — Multi-control averaging"
            + (f": {truncate_filename(filename)}" if filename else "")
        )
    except Exception:
        pass
    plt.subplots_adjust(left=0.1, bottom=0.18, right=0.98, top=0.78, hspace=0.25)

    ax_top.plot(sample_t, sample_y, color="tab:green", lw=1.3, label="Sample")
    apply_robust_ylim(ax_top, sample_y)
    anchor_line_top = ax_top.axvline(anchor["t0"], color="#8B008B", lw=1.8, ls="--", alpha=0.9, label="Anchor")
    anchor_line_bot = ax_bot.axvline(anchor["t0"], color="#8B008B", lw=1.8, ls="--", alpha=0.9)
    ax_top.set_ylabel("H₂O₂ (µM)")
    ax_bot.axhline(0.0, color="grey", lw=0.6, ls=":")
    ax_bot.set_xlabel("Time (s)")
    ax_bot.set_ylabel("H₂O₂ (µM, corrected)")

    dynamic_lines: list = []
    line_avg = [None]
    line_corr = [None]
    readout = [fig.text(
        0.5, 0.845, "", ha="center", va="top", fontsize=9, color="#333333",
    )]

    member_colours = [
        "#d4a0a0", "#a0a0d4", "#a0d4a0", "#d4d4a0", "#d4a0d4",
        "#a0d4d4", "#c8a080", "#80c8a0",
    ]

    add_instruction_banner(
        fig,
        "Build your control-averaging set.  "
        "Controls are auto-aligned to this run's H₂O₂ injection, and only "
        "their drift away from the anchor is subtracted — so a 100 µM run "
        "minus a 100 µM control stays near 100, not 0.  "
        "Top: sample (green) + controls (faint) + average (red).  "
        "Bottom: corrected sample.  Click upper plot to move the anchor.",
        y=0.985, width=110,
    )

    view = {"autoscaled": False}

    def redraw():
        for ln in dynamic_lines:
            ln.remove()
        dynamic_lines.clear()
        if line_avg[0] is not None:
            line_avg[0].remove()
            line_avg[0] = None
        if line_corr[0] is not None:
            line_corr[0].remove()
            line_corr[0] = None

        if not members:
            ax_top.legend(loc="upper right", fontsize=9)
            ax_bot.legend(loc="upper right", fontsize=9)
            if not view["autoscaled"]:
                for a in (ax_top, ax_bot):
                    a.relim()
                    a.autoscale_view()
                view["autoscaled"] = True
            fig.canvas.draw_idle()
            return

        pairs = [(ct, cy) for (_label, ct, cy) in members]
        # Members are pre-aligned to the sample's injection, so they are
        # always placed at the interval start; the anchor only chooses the
        # zero-deviation reference time.
        averaged, interps = average_controls_on_grid(
            pairs, sample_t, float(sample_t[0])
        )

        for i, (interp_y, (label, _ct, _cy)) in enumerate(zip(interps, members)):
            c = member_colours[i % len(member_colours)]
            ln, = ax_top.plot(sample_t, interp_y, color=c, lw=0.8, alpha=0.55,
                              label=f"Ctrl {i+1}")
            dynamic_lines.append(ln)

        ln_a, = ax_top.plot(sample_t, averaged, color="tab:red", lw=1.6,
                            label=f"Average ({len(members)} ctrl)")
        line_avg[0] = ln_a

        dev = deviation_from_anchor(averaged, sample_t, anchor["t0"])
        ln_c, = ax_bot.plot(sample_t, sample_y - dev, color="tab:blue",
                            lw=1.3, label="Sample − control deviation")
        line_corr[0] = ln_c

        anchor_line_top.set_xdata([anchor["t0"], anchor["t0"]])
        anchor_line_bot.set_xdata([anchor["t0"], anchor["t0"]])

        # Readout so a misplaced anchor is obvious rather than silent.
        ctrl_at = float(np.interp(anchor["t0"], sample_t, averaged))
        samp_at = float(np.interp(anchor["t0"], sample_t, sample_y))
        drift = float(averaged[-1] - ctrl_at)
        msg = (
            f"At anchor t={anchor['t0']:.1f}s:  control={ctrl_at:.1f} µM   "
            f"sample={samp_at:.1f} µM   |   control drift over run: "
            f"{drift:+.1f} µM"
        )
        colour = "#333333"
        span = float(np.nanmax(sample_y) - np.nanmin(sample_y))
        if span > 0 and abs(samp_at - ctrl_at) > 0.25 * span:
            msg += "\n⚠ Control and sample differ a lot here — the anchor may "
            msg += "sit before an injection. Click the upper plot to move it."
            colour = "#b00020"
        readout[0].set_text(msg)
        readout[0].set_color(colour)

        ax_top.legend(loc="upper right", fontsize=9)
        ax_bot.legend(loc="upper right", fontsize=9)

        if not view["autoscaled"]:
            # Robust limits on both panes: the corrected trace inherits the
            # sample's injection transient, which would otherwise squash it.
            apply_robust_ylim(ax_top, sample_y, averaged)
            apply_robust_ylim(ax_bot, sample_y - dev)
            view["autoscaled"] = True
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

    def _close_with(v):
        def _f(_e=None):
            decision["value"] = v
            get_window().stop()
        return _f

    def _add_existing(_e=None):
        if existing_candidates is None or not existing_candidates:
            print("No existing intervals available.")
            return
        picked = pick_existing_intervals_multi(existing_candidates, filename=filename)
        if picked:
            for p in picked:
                ct, cy = p["load"]()
                ct_zero = ct - ct[0]
                members.append(_aligned((p["label"], ct_zero, cy)))
            redraw()

    def _add_new(_e=None):
        if new_control_callback is None:
            print("No callback for new-control processing.")
            return
        picked_path = _qt_pick_control_file(
            calibrated_dir.parent if calibrated_dir is not None else None,
            subtracting_from=filename,
        )
        if picked_path is not None:
            template = new_control_callback(picked_path)
            if template is not None:
                ct, cy = template
                ct_zero = ct - ct[0]
                members.append(
                    _aligned((f"new: {picked_path.name}", ct_zero, cy))
                )
                redraw()

    def _remove_last(_e=None):
        if members:
            members.pop()
            redraw()

    ax_add_ex  = fig.add_axes([0.03, 0.03, 0.16, 0.06])
    ax_add_new = fig.add_axes([0.20, 0.03, 0.14, 0.06])
    ax_rm      = fig.add_axes([0.35, 0.03, 0.14, 0.06])
    ax_accept  = fig.add_axes([0.51, 0.03, 0.18, 0.06])
    ax_skip    = fig.add_axes([0.70, 0.03, 0.12, 0.06])
    ax_back    = fig.add_axes([0.83, 0.03, 0.12, 0.06])

    _btn_add_ex  = create_small_button(ax_add_ex,  "Add existing…", "#ffe680", "#ffcd55")
    _btn_add_new = create_small_button(ax_add_new, "Add new…",      "#ffcc99", "#ffaa66")
    _btn_rm      = create_small_button(ax_rm,      "Remove last",       "0.9", "0.8")
    _btn_accept  = create_small_button(ax_accept,  "Accept & subtract", "#90ee90", "#7cd47c")
    _btn_skip    = create_small_button(ax_skip,    "Skip",              "#ffcc99", "#ffaa66")
    _btn_back    = create_small_button(ax_back,    "Back",              "#ddddff", "#bbbbff")

    _btn_add_ex.on_clicked(_add_existing)
    _btn_add_new.on_clicked(_add_new)
    _btn_rm.on_clicked(_remove_last)
    _btn_accept.on_clicked(_close_with("accept"))
    _btn_skip.on_clicked(_close_with("skip"))
    _btn_back.on_clicked(_close_with("back"))

    redraw()
    install_zoom_keys(fig, [ax_top, ax_bot])
    get_window().run()

    d = decision["value"] or "skip"
    return (d, members if d == "accept" else [], float(anchor["t0"]))


# ────────────────────────────────────────────────────────────────────────
# UI: one fit — single-pane (model + range + fit + back-extrap all together)
# ────────────────────────────────────────────────────────────────────────


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
    existing_fits: "list[FitRecord] | None" = None,
):
    """Drive a single fit on an interval in ONE pane.

    Replaces the previous 3-pane flow (model picker → fit-range picker →
    preview + back-extrap) with a single screen that contains:

    - the data plot (top, large) with the fitted curve overlaid in red
    - a residuals strip below
    - mode buttons (Manual linear | Single exp | Inactivation) at the top
    - a "Fit" button + a stats line showing init_rate and R²
    - actions along the bottom: Extrapolate to… | Accept fit | Retry |
      Skip | Back

    Flow:
      1. Click twice on the data to define the fit range.  The two clicks
         are marked green (start) and red (end).
      2. Click a model button at the top.
      3. Click "Fit" — the curve is drawn over the data, residuals appear
         in the lower strip, and the init-rate + R² appear at the top.
      4. Optionally click "Extrapolate to…" — your next click on the plot
         picks the t-value to extrapolate to (it can be EARLIER than the
         fit start = back-extrap, or LATER than the fit end = forward
         extrap).  A red dotted line connects the fit to the new
         extrapolation target, with an open red circle marking it.
      5. Accept records the fit (with any back-extrap info).

    Returns:
      FitRecord — accepted (possibly with back_extrap_applied=True).
      "skip"    — user skipped this fit.
      "back"    — user wants to revisit the subtraction step.
    """
    # ── Figure layout: 2-row gridspec (data : residuals = 3 : 1) ──────
    fig = get_window().reset(figsize=(11.5, 7.6))
    try:
        fig.canvas.manager.set_window_title(
            f"SensorFit — Fit #{fit_index}"
            + (f": {truncate_filename(filename)}" if filename else "")
        )
    except Exception:
        pass
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.10,
                          left=0.10, right=0.97, top=0.76, bottom=0.155)
    ax_data = fig.add_subplot(gs[0])
    ax_resid = fig.add_subplot(gs[1], sharex=ax_data)
    ax_data.plot(interval_t, interval_y, color="tab:green", lw=1.2, label="Interval")
    apply_robust_ylim(ax_data, interval_y)
    ax_data.set_ylabel("H2O2 (µM)")
    ax_resid.axhline(0.0, color="grey", lw=0.6, ls=":")
    ax_resid.set_ylabel("residual")
    ax_resid.set_xlabel("Time (s)")

    add_instruction_banner(
        fig,
        "Click twice on the plot to set the fit range; pick a model below; click Fit.  "
        "Extrapolate then prompts a click on the plot for the target t.",
        y=0.985, width=130,
    )

    # ── Controls along the top (between banner and plot) ─────────────
    n_existing = len(existing_fits or [])
    fig.text(
        0.10, 0.895,
        f"Fit #{fit_index}" + (f"  (previous fits on this interval: {n_existing})" if n_existing else ""),
        fontsize=10, fontweight="bold",
    )

    # Model picker buttons
    ax_mlin = fig.add_axes([0.10, 0.845, 0.12, 0.04])
    ax_mexp = fig.add_axes([0.23, 0.845, 0.12, 0.04])
    ax_mib = fig.add_axes([0.36, 0.845, 0.12, 0.04])
    btn_mlin = create_small_button(ax_mlin, "Manual linear", "0.85", "0.75")
    btn_mexp = create_small_button(ax_mexp, "Single exp", "0.85", "0.75")
    btn_mib = create_small_button(ax_mib, "Inactivation", "0.85", "0.75")

    # Action buttons in the same row
    ax_fit = fig.add_axes([0.52, 0.845, 0.08, 0.04])
    ax_clear = fig.add_axes([0.61, 0.845, 0.10, 0.04])
    btn_fit = create_small_button(ax_fit, "Fit", "#90ee90", "#7cd47c")
    btn_clear = create_small_button(ax_clear, "Clear range", "0.9", "0.8")

    # Stats line + status hint
    stats_text = fig.text(
        0.10, 0.815,
        "Pick the fit range by clicking on the plot, then a model.",
        fontsize=9, color="dimgrey", style="italic",
    )

    # ── State ─────────────────────────────────────────────────────────
    mode_state = {"model": None}
    range_state = {"clicks": [], "markers": []}
    fit_state = {"record": None, "fit_artists": []}
    extrap_state = {
        "pending": False,
        "target_t": None,
        "new_rate": None,
        "artists": [],
    }
    final = {"action": None}

    # ── Helpers ───────────────────────────────────────────────────────
    def _set_stats(msg, colour="dimgrey"):
        stats_text.set_text(msg)
        stats_text.set_color(colour)
        fig.canvas.draw_idle()

    def _update_model_buttons():
        for m, btn in (("ManualLinear", btn_mlin), ("Exponential", btn_mexp), ("IB", btn_mib)):
            btn.color = "#ffe680" if mode_state["model"] == m else "0.85"
        fig.canvas.draw_idle()

    def _clear_fit_artists():
        for art in fit_state["fit_artists"]:
            try:
                art.remove()
            except Exception:
                pass
        fit_state["fit_artists"].clear()
        fit_state["record"] = None
        ax_resid.clear()
        ax_resid.axhline(0.0, color="grey", lw=0.6, ls=":")
        ax_resid.set_ylabel("residual")
        ax_resid.set_xlabel("Time (s)")

    def _clear_extrap_artists():
        for art in extrap_state["artists"]:
            try:
                art.remove()
            except Exception:
                pass
        extrap_state["artists"].clear()
        extrap_state["target_t"] = None
        extrap_state["new_rate"] = None
        extrap_state["pending"] = False

    def _clear_range():
        for m in range_state["markers"]:
            try:
                m.remove()
            except Exception:
                pass
        range_state["markers"].clear()
        range_state["clicks"].clear()
        _clear_fit_artists()
        _clear_extrap_artists()
        _set_stats("Pick the fit range by clicking on the plot, then a model.")
        fig.canvas.draw_idle()

    def _draw_range_markers():
        for m in range_state["markers"]:
            try:
                m.remove()
            except Exception:
                pass
        range_state["markers"].clear()
        clicks_sorted = sorted(range_state["clicks"])
        if clicks_sorted:
            idx0 = clicks_sorted[0]
            m0, = ax_data.plot(
                interval_t[idx0], interval_y[idx0], "o",
                ms=12, mfc="#76d275", mec="darkgreen", mew=1.5, zorder=6,
            )
            range_state["markers"].append(m0)
        if len(clicks_sorted) >= 2:
            idx1 = clicks_sorted[-1]
            m1, = ax_data.plot(
                interval_t[idx1], interval_y[idx1], "o",
                ms=12, mfc="#ff6e6e", mec="darkred", mew=1.5, zorder=6,
            )
            range_state["markers"].append(m1)
        fig.canvas.draw_idle()

    def _run_and_draw_fit():
        if mode_state["model"] is None:
            _set_stats("Pick a model first (Manual linear / Single exp / Inactivation).", "darkred")
            return
        if len(range_state["clicks"]) != 2:
            _set_stats("Click two points on the plot to define the fit range.", "darkred")
            return
        i0, i1 = sorted(range_state["clicks"])
        if i1 - i0 < 2:
            _set_stats("Fit range too narrow; pick a wider span.", "darkred")
            return
        try:
            rec = _run_fit(
                mode_state["model"], interval_t, interval_y, i0, i1,
                init_rate_at_t=float(interval_t[i0]),
            )
        except Exception as exc:
            _set_stats(f"Fit failed: {exc}", "darkred")
            return

        _clear_fit_artists()
        _clear_extrap_artists()

        t_fit = interval_t[i0 : i1 + 1]
        y_fit = interval_y[i0 : i1 + 1]
        line_fit, = ax_data.plot(
            t_fit, rec.yhat, color="red", lw=1.8, label=f"{rec.model} fit", zorder=7,
        )
        # Tangent at start (the init-rate slope)
        t_start = float(interval_t[i0])
        y_start = float(rec.yhat[0])
        slope_disp = rec.init_rate_uM_per_s
        tangent_len = max(1.0, (interval_t[i1] - t_start) * 0.05)
        tangent_t = np.array([t_start - tangent_len * 0.5, t_start + tangent_len * 1.5])
        tangent_y = y_start + slope_disp * (tangent_t - t_start)
        line_tan, = ax_data.plot(
            tangent_t, tangent_y, color="#005700", lw=2.2, ls="-", alpha=0.9, zorder=6,
        )
        fit_state["fit_artists"].extend([line_fit, line_tan])
        ax_data.legend(loc="best", fontsize=8)

        resid_vals = y_fit - rec.yhat
        ax_resid.plot(t_fit, resid_vals, color="tab:blue", lw=1.0)
        # Force a symmetric y-axis around 0 so positive and negative
        # residuals are both visible.
        r_max = float(np.max(np.abs(resid_vals))) if resid_vals.size else 1.0
        r_max = max(r_max, 1e-9)
        ax_resid.set_ylim(-r_max * 1.15, r_max * 1.15)

        fit_state["record"] = rec
        r2_str = f"{rec.r2:.4f}" if np.isfinite(rec.r2) else "n/a"
        _set_stats(
            f"{rec.model}  init_rate = {rec.init_rate_uM_per_s:.4f} µM/s  R² = {r2_str}.  "
            "Accept to record, or click Extrapolate to predict a different t₀.",
            colour="#005700",
        )
        fig.canvas.draw_idle()

    def _do_extrapolation(target_t: float):
        rec = fit_state["record"]
        if rec is None:
            return
        if rec.model == "ManualLinear":
            slope, intercept = rec.params
            y_at_target = slope * target_t + intercept
            new_rate = slope
        elif rec.model == "Exponential":
            y_at_target = float(model_Exponential(np.array([target_t]), *rec.params)[0])
            a, _b, c, k_decay, t0 = rec.params
            new_rate = float(a - c * k_decay * np.exp(-k_decay * (target_t - t0)))
        elif rec.model == "IB":
            y_at_target = float(model_IB(np.array([target_t]), *rec.params)[0])
            _C, H0, alpha, kinact, kslow = rec.params
            new_rate = float(
                -H0 * alpha * kinact * np.exp(-kinact * target_t)
                * np.exp(-alpha * (1.0 - np.exp(-kinact * target_t)))
                - kslow
            )
        else:
            return

        _clear_extrap_artists()

        # Sample the actual model on a dense grid between the fit edge
        # and the extrap target so the user sees how the curve behaves
        # at the new t, not just a straight line to the ring.
        t_fit_start, t_fit_end = rec.fit_start_s, rec.fit_end_s
        if target_t < t_fit_start:
            t_curve = np.linspace(target_t, t_fit_start, 120)
        else:
            t_curve = np.linspace(t_fit_end, target_t, 120)
        if rec.model == "ManualLinear":
            slope, intercept = rec.params
            y_curve = slope * t_curve + intercept
        elif rec.model == "Exponential":
            y_curve = model_Exponential(t_curve, *rec.params)
        elif rec.model == "IB":
            y_curve = model_IB(t_curve, *rec.params)
        else:
            y_curve = np.full_like(t_curve, y_at_target, dtype=float)

        ln, = ax_data.plot(
            t_curve, y_curve,
            color="red", lw=1.6, ls=":", alpha=0.9, zorder=6,
        )
        circle, = ax_data.plot(
            [target_t], [y_at_target], "o",
            ms=14, mfc="none", mec="red", mew=2.5, zorder=7,
            label="Extrap target",
        )
        extrap_state["artists"].extend([ln, circle])
        extrap_state["target_t"] = float(target_t)
        extrap_state["new_rate"] = float(new_rate)
        ax_data.legend(loc="best", fontsize=8)

        # Auto-expand axes to include the whole extrap curve (the swept
        # curve may rise above or dip below the target value depending on
        # the model and direction).
        xmin, xmax = ax_data.get_xlim()
        ymin, ymax = ax_data.get_ylim()
        x_margin = (xmax - xmin) * 0.05 or 1.0
        y_margin = (ymax - ymin) * 0.08 or 1.0
        curve_ymin = float(np.min(y_curve))
        curve_ymax = float(np.max(y_curve))
        new_xmin = min(xmin, float(np.min(t_curve)) - x_margin)
        new_xmax = max(xmax, float(np.max(t_curve)) + x_margin)
        new_ymin = min(ymin, curve_ymin - y_margin, y_at_target - y_margin)
        new_ymax = max(ymax, curve_ymax + y_margin, y_at_target + y_margin)
        if (new_xmin, new_xmax, new_ymin, new_ymax) != (xmin, xmax, ymin, ymax):
            ax_data.set_xlim(new_xmin, new_xmax)
            ax_data.set_ylim(new_ymin, new_ymax)

        direction = "back" if target_t < t_fit_start else "forward"
        r2_str = f"{rec.r2:.4f}" if np.isfinite(rec.r2) else "n/a"
        _set_stats(
            f"{rec.model}: {direction}-extrap to t={target_t:.3f}s → new rate = {new_rate:.4f} µM/s "
            f"(original rate at t_start = {rec.init_rate_uM_per_s:.4f}, R² = {r2_str}).",
            colour="#005700",
        )
        fig.canvas.draw_idle()

    # ── Click handler ─────────────────────────────────────────────────
    def on_click(event):
        if event.button != 1 or event.inaxes is not ax_data or event.xdata is None:
            return
        if getattr(fig.canvas.toolbar, "mode", "") in ("zoom rect", "pan/zoom", "zoom", "pan"):
            return
        x = float(event.xdata)

        if extrap_state["pending"]:
            _do_extrapolation(x)
            extrap_state["pending"] = False
            return

        idx = int(np.abs(interval_t - x).argmin())
        if len(range_state["clicks"]) < 2:
            range_state["clicks"].append(idx)
            _draw_range_markers()
            if len(range_state["clicks"]) == 1:
                _set_stats("Now click the END of the fit range.", "dimgrey")
            else:
                if mode_state["model"] is None:
                    _set_stats("Range set.  Now pick a model and click Fit.", "dimgrey")
                else:
                    _set_stats("Range set.  Click Fit to run the fit.", "dimgrey")
        else:
            _set_stats("Range already set.  Click Clear range to redo, or Fit.", "darkred")

    fig.canvas.mpl_connect("button_press_event", on_click)

    # ── Button handlers ───────────────────────────────────────────────
    def _set_model(m):
        def _f(_e=None):
            mode_state["model"] = m
            _update_model_buttons()
            if len(range_state["clicks"]) == 2:
                _set_stats(f"Model: {m}.  Click Fit.", "dimgrey")
            else:
                _set_stats(f"Model: {m}.  Click two points to set the fit range.", "dimgrey")
        return _f

    btn_mlin.on_clicked(_set_model("ManualLinear"))
    btn_mexp.on_clicked(_set_model("Exponential"))
    btn_mib.on_clicked(_set_model("IB"))
    btn_fit.on_clicked(lambda _e=None: _run_and_draw_fit())
    btn_clear.on_clicked(lambda _e=None: _clear_range())

    def on_extrapolate(_e=None):
        if fit_state["record"] is None:
            _set_stats("Run a fit first, then Extrapolate.", "darkred")
            return
        extrap_state["pending"] = True
        _set_stats(
            "Click a point on the plot to extrapolate to "
            "(earlier than fit start = back-extrap; later than fit end = forward).",
            "dimgrey",
        )

    def on_accept(_e=None):
        rec = fit_state["record"]
        if rec is None:
            _set_stats("Run a fit first, then Accept.", "darkred")
            return
        if extrap_state["target_t"] is not None:
            rec.back_extrap_applied = True
            rec.back_extrap_deadtime_s = float(rec.fit_start_s - extrap_state["target_t"])
            rec.back_extrap_t0_s = float(extrap_state["target_t"])
            rec.back_extrap_rate_uM_per_s = float(extrap_state["new_rate"])
        final["action"] = "accept"
        get_window().stop()

    def on_retry(_e=None):
        _clear_range()
        mode_state["model"] = None
        _update_model_buttons()

    def on_skip(_e=None):
        final["action"] = "skip"
        get_window().stop()

    def on_back(_e=None):
        final["action"] = "back"
        get_window().stop()

    ax_extrap = fig.add_axes([0.10, 0.04, 0.18, 0.05])
    ax_accept = fig.add_axes([0.30, 0.04, 0.14, 0.05])
    ax_retry = fig.add_axes([0.46, 0.04, 0.10, 0.05])
    ax_skip = fig.add_axes([0.58, 0.04, 0.12, 0.05])
    ax_back = fig.add_axes([0.72, 0.04, 0.20, 0.05])
    btn_extrap = create_small_button(ax_extrap, "Extrapolate to…", "#ffe680", "#ffcd55")
    btn_extrap.on_clicked(on_extrapolate)
    btn_accept = create_small_button(ax_accept, "Accept fit", "#90ee90", "#7cd47c")
    btn_accept.on_clicked(on_accept)
    btn_retry = create_small_button(ax_retry, "Retry", "0.9", "0.8")
    btn_retry.on_clicked(on_retry)
    btn_skip = create_small_button(ax_skip, "Skip fit", "#ffcc99", "#ffaa66")
    btn_skip.on_clicked(on_skip)
    btn_back = create_small_button(ax_back, "Back → subtraction", "#ddddff", "#bbbbff")
    btn_back.on_clicked(on_back)

    install_zoom_keys(fig, [ax_data, ax_resid])
    get_window().run()

    if final["action"] == "accept" and fit_state["record"] is not None:
        return fit_state["record"]
    return final["action"] or "back"


# ────────────────────────────────────────────────────────────────────────
# UI: Δmax screen (three modes + selectable t_zero)
# ────────────────────────────────────────────────────────────────────────


def _model_asymptote(rec: FitRecord, t: np.ndarray) -> np.ndarray:
    """Return the model's asymptote evaluated at ``t``.

    For Exponential ``y = a*t + b + c*exp(-k*(t-t0))`` the asymptote is the
    linear background ``a*t + b``.  For IB ``y = C + H0*exp(...) - kslow*t``
    it's ``C - kslow*t``.  For ManualLinear, the fit itself.
    """
    t = np.asarray(t, dtype=float)
    if rec.model == "Exponential" and len(rec.params) == 5:
        a, b, _c, _k, _t0 = rec.params
        return a * t + b
    if rec.model == "IB" and len(rec.params) == 5:
        C, _H0, _alpha, _kinact, kslow = rec.params
        return C - kslow * t
    if rec.model == "ManualLinear" and len(rec.params) == 2:
        slope, intercept = rec.params
        return slope * t + intercept
    return np.zeros_like(t)


def _initial_hint(mode: str, has_usable_fit: bool, max_uM: float = 100.0) -> str:
    """Return a short status hint shown above the Δmax plot."""
    if mode == "from-fit":
        if not has_usable_fit:
            return "No usable fit; pick Linear or Point."
        return "From fit: click ONE point to set t₀ (Δmax = y_fit − asymptote at t₀)."
    if mode == "linear":
        return f"Linear: click t₀, then TWO baseline points.  Δmax = {max_uM:.0f} − line@t₀."
    if mode == "point":
        return f"Point: click t₀, then ONE baseline point.  Δmax = {max_uM:.0f} − y_point."
    return ""


def prompt_delta_max(
    interval_t: np.ndarray,
    interval_y: np.ndarray,
    fits: list[FitRecord],
    filename: str | None = None,
    cal_max_uM: float = 100.0,
):
    """Drive one Δmax estimation for the interval.

    All three modes interpret Δmax as ``high − low``:

    - **From fit**: click ONE point to set t₀.  Δmax = ``y_fit(t₀) − asymptote(t₀)``.
      The fit is redrawn boldly in red and its asymptote as a dashed line
      so it's obvious where the measurement comes from.
    - **Linear**: click t₀, then TWO points that estimate the baseline.
      Δmax = ``cal_max_uM − y_line(t₀)``.
    - **Point**: click t₀, then ONE point representing the baseline.
      Δmax = ``cal_max_uM − y_at_point``.

    ``cal_max_uM`` can be changed on-the-fly via the TextBox at top right.
    """
    has_usable_fit = any(f.model in ("Exponential", "IB") for f in fits)
    default_mode = "from-fit" if has_usable_fit else "linear"

    fig = get_window().reset(figsize=(11, 7.2))
    ax = fig.add_subplot(111)
    try:
        fig.canvas.manager.set_window_title(
            "SensorFit — Δ[H₂O₂]max"
            + (f": {truncate_filename(filename)}" if filename else "")
        )
    except Exception:
        pass
    plt.subplots_adjust(left=0.10, bottom=0.20, right=0.98, top=0.66)
    ax.plot(interval_t, interval_y, color="tab:green", lw=1.2, label="Interval")
    apply_robust_ylim(ax, interval_y)
    for i, f in enumerate(fits):
        t_seg = np.linspace(f.fit_start_s, f.fit_end_s, f.yhat.size)
        ax.plot(t_seg, f.yhat, "-", lw=1.0, alpha=0.6, label=f"fit#{i+1}:{f.model}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("H2O2 (µM)")
    ax.legend(loc="best", fontsize=8)

    add_instruction_banner(
        fig,
        "Pick mode; click t₀; Linear/Point also click on the baseline.  "
        "Toggle 'Set max' to click-set max [H₂O₂] (horizontal line).  "
        "Δmax = max_uM − baseline_at_t₀.",
        y=0.985, width=130,
    )

    mode_state = {"mode": default_mode}
    max_uM_state = {"value": float(cal_max_uM)}
    click_target = {"value": "t0"}  # "t0" or "max_uM"
    state = {
        "t_zero_idx": None,
        "linear_clicks": [],
        "point_idx": None,
        "computed": None,
        "action": None,
        "tz_artists": [],
        "baseline_artists": [],
        "span_artists": [],
        "max_uM_artists": [],
        "annotation": None,
    }
    window_state = {"value": 50}

    btn_axes = {
        "from-fit": fig.add_axes([0.10, 0.74, 0.13, 0.05]),
        "linear":   fig.add_axes([0.24, 0.74, 0.13, 0.05]),
        "point":    fig.add_axes([0.38, 0.74, 0.13, 0.05]),
    }
    from_fit_active = "#90ee90" if has_usable_fit else "0.92"
    from_fit_idle = "0.85" if has_usable_fit else "0.92"

    btns = {
        "from-fit": create_small_button(
            btn_axes["from-fit"], "From fit",
            from_fit_active if default_mode == "from-fit" else from_fit_idle,
            "#7cd47c",
        ),
        "linear": create_small_button(
            btn_axes["linear"], "Linear",
            "#90ee90" if default_mode == "linear" else "0.85", "#7cd47c",
        ),
        "point": create_small_button(
            btn_axes["point"], "Point",
            "#90ee90" if default_mode == "point" else "0.85", "#7cd47c",
        ),
    }

    # "Set max" toggle button — switches click mode between t₀/baseline
    # picking and max_uM picking (horizontal line from y-click).
    ax_set_max = fig.add_axes([0.53, 0.74, 0.12, 0.05])
    btn_set_max = create_small_button(ax_set_max, "Set max ↕", "0.85", "#ffd480")

    def _toggle_click_target(_e=None):
        if click_target["value"] == "t0":
            click_target["value"] = "max_uM"
            btn_set_max.color = "#ffd480"
            _update_status(
                "Click on the plot to set max [H₂O₂] (horizontal line).  "
                "Click 'Set max' again to return to t₀/baseline mode.",
                "#995500",
            )
        else:
            click_target["value"] = "t0"
            btn_set_max.color = "0.85"
            _update_status(_initial_hint(mode_state["mode"], has_usable_fit, max_uM_state["value"]))
        fig.canvas.draw_idle()

    btn_set_max.on_clicked(_toggle_click_target)

    # max-µM TextBox (also settable by click) and window TextBox.
    fig.text(
        0.72, 0.795, "max [H₂O₂] (µM)",
        ha="center", va="center", fontsize=9, color="dimgrey",
    )
    ax_max = fig.add_axes([0.67, 0.745, 0.10, 0.04])
    tb_max = TextBox(ax_max, "", initial=f"{cal_max_uM:.2f}")

    fig.text(
        0.88, 0.795, "window ±",
        ha="center", va="center", fontsize=9, color="dimgrey",
    )
    ax_window = fig.add_axes([0.85, 0.745, 0.06, 0.04])
    tb_window = TextBox(ax_window, "", initial=str(window_state["value"]))

    if not has_usable_fit:
        fig.text(
            0.165, 0.715, "(no fit available)",
            ha="center", va="top", fontsize=8, color="dimgrey",
        )

    status_text = fig.text(
        0.10, 0.685,
        _initial_hint(default_mode, has_usable_fit, max_uM_state["value"]),
        fontsize=9, color="dimgrey", style="italic",
    )

    def _update_status(msg, colour="dimgrey"):
        status_text.set_text(msg)
        status_text.set_color(colour)
        fig.canvas.draw_idle()

    def _remove_artists(category: str):
        for art in state[category]:
            try:
                art.remove()
            except Exception:
                pass
        state[category].clear()

    def _clear_span():
        _remove_artists("span_artists")
        if state["annotation"] is not None:
            try:
                state["annotation"].remove()
            except Exception:
                pass
            state["annotation"] = None
        state["computed"] = None

    def _clear_t_zero():
        _remove_artists("tz_artists")
        state["t_zero_idx"] = None
        _clear_span()

    def _clear_baseline():
        _remove_artists("baseline_artists")
        state["linear_clicks"].clear()
        state["point_idx"] = None
        _clear_span()

    def _clear_max_uM_line():
        _remove_artists("max_uM_artists")

    def _draw_max_uM_line(y_value):
        """Horizontal orange dashed line showing the current max [H₂O₂]."""
        _clear_max_uM_line()
        # If the chosen value sits outside the current view, widen the y-limits
        # so the line (and its label) stay visible inside the plot rather than
        # being clipped off-screen and colliding with the status text.
        y0, y1 = ax.get_ylim()
        span = (y1 - y0) or 1.0
        if y_value > y1 - 0.02 * span:
            ax.set_ylim(y0, y_value + 0.06 * span)
        elif y_value < y0 + 0.02 * span:
            ax.set_ylim(y_value - 0.06 * span, y1)
        hl = ax.axhline(
            y_value, color="#E07020", lw=1.8, ls="--", alpha=0.85, zorder=4,
        )
        state["max_uM_artists"].append(hl)
        lbl = ax.text(
            0.015, y_value, f" max = {y_value:.2f} µM",
            transform=ax.get_yaxis_transform(),
            fontsize=8, color="#E07020", va="bottom", ha="left", zorder=5,
        )
        state["max_uM_artists"].append(lbl)
        fig.canvas.draw_idle()

    def _clear_preview():
        _clear_t_zero()
        _clear_baseline()
        fig.canvas.draw_idle()

    def _set_mode(m):
        def _f(_e=None):
            if m == "from-fit" and not has_usable_fit:
                _update_status("From-fit needs an Exponential or IB fit; pick Linear or Point.",
                              "darkred")
                return
            if mode_state["mode"] == m:
                return
            mode_state["mode"] = m
            btns["from-fit"].color = from_fit_active if m == "from-fit" else from_fit_idle
            btns["linear"].color = "#90ee90" if m == "linear" else "0.85"
            btns["point"].color = "#90ee90" if m == "point" else "0.85"
            _clear_preview()
            _update_status(_initial_hint(m, has_usable_fit, max_uM_state["value"]))
            print(f"Δmax mode → {m}")
        return _f

    for m in btns:
        btns[m].on_clicked(_set_mode(m))

    def _on_max_submit(text):
        try:
            v = float(text)
            if not np.isfinite(v) or v <= 0:
                raise ValueError
        except ValueError:
            _update_status(f"max_uM must be a positive number; got '{text}'.", "darkred")
            tb_max.set_val(f"{max_uM_state['value']:.2f}")
            return
        max_uM_state["value"] = v
        if state["t_zero_idx"] is not None:
            # Recompute with the new max — easiest: re-run on existing clicks.
            _replay_after_max_change()
        else:
            _update_status(_initial_hint(mode_state["mode"], has_usable_fit, v))

    tb_max.on_submit(_on_max_submit)

    def _on_window_submit(text):
        try:
            v = int(float(text))
            if v < 0:
                raise ValueError
        except ValueError:
            _update_status(f"window must be a non-negative integer; got '{text}'.", "darkred")
            tb_window.set_val(str(window_state["value"]))
            return
        window_state["value"] = v
        # Re-run the computation with the new window so baseline
        # averages update immediately.
        if state["t_zero_idx"] is not None:
            _replay_after_max_change()
        else:
            _update_status(_initial_hint(mode_state["mode"], has_usable_fit, max_uM_state["value"]))

    tb_window.on_submit(_on_window_submit)

    def _avg_y(idx: int) -> float:
        """Windowed average of interval_y around the given index."""
        return float(average_window(interval_y, idx, window_state["value"]))

    def _draw_t_zero_marker(idx):
        # Just a vertical dashed line at t₀.  The y-value of the click
        # doesn't matter for any of the calculations, so no cross is
        # needed (and the cross was visually noisy).
        ln = ax.axvline(
            interval_t[idx], color="#8B008B", lw=1.6, ls="--", alpha=0.9, zorder=4,
        )
        state["tz_artists"].append(ln)
        fig.canvas.draw_idle()

    def _draw_baseline_point(idx):
        # Marker sits at the WINDOWED-AVERAGE y so the user can see what
        # value is actually being used in the calculation.
        y_avg = _avg_y(idx)
        m, = ax.plot(
            interval_t[idx], y_avg, "o",
            ms=11, mfc="#8B008B", mec="white", mew=1.0, zorder=6,
        )
        state["baseline_artists"].append(m)

    def _draw_baseline_horizontal(y_value):
        """Horizontal purple dashed line at the baseline y-value, so the
        user can see what level is being measured against."""
        hl = ax.axhline(
            y_value, color="#8B008B", lw=1.0, ls=":", alpha=0.75, zorder=4,
        )
        state["baseline_artists"].append(hl)

    def _draw_span(t_zero, y_top, y_bot, label_text):
        arrow = ax.annotate(
            "", xy=(t_zero, y_bot), xytext=(t_zero, y_top),
            xycoords="data", textcoords="data",
            arrowprops=dict(
                arrowstyle="<->", color="#8B008B", lw=2.4,
                shrinkA=0, shrinkB=0,
            ),
            annotation_clip=True, zorder=7,
        )
        state["span_artists"].append(arrow)

        if state["annotation"] is not None:
            try:
                state["annotation"].remove()
            except Exception:
                pass
        state["annotation"] = ax.text(
            0.985, 0.965, label_text,
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="#f7e8ff", edgecolor="#8B008B", alpha=0.95),
            zorder=8,
        )

        # Make sure both endpoints are visible so the arrow + box can
        # never end up clipped.
        ymin, ymax = ax.get_ylim()
        y_lo = min(y_top, y_bot)
        y_hi = max(y_top, y_bot)
        margin = (ymax - ymin) * 0.05 or 1.0
        new_ymin = min(ymin, y_lo - margin)
        new_ymax = max(ymax, y_hi + margin)
        if (new_ymin, new_ymax) != (ymin, ymax):
            ax.set_ylim(new_ymin, new_ymax)

    def _highlight_fit(rec: FitRecord):
        t_fit = np.linspace(rec.fit_start_s, rec.fit_end_s, rec.yhat.size)
        ln_fit, = ax.plot(t_fit, rec.yhat, "r-", lw=2.2, zorder=6)
        state["tz_artists"].append(ln_fit)
        xmin, xmax = float(interval_t[0]), float(interval_t[-1])
        asym_t = np.linspace(xmin, xmax, 60)
        asym_y = _model_asymptote(rec, asym_t)
        ln_asym, = ax.plot(asym_t, asym_y, color="red", lw=1.0, ls="--", alpha=0.7, zorder=5)
        state["tz_artists"].append(ln_asym)
        fig.canvas.draw_idle()

    def _compute_and_preview():
        tz_idx = state["t_zero_idx"]
        if tz_idx is None:
            return
        t_zero = float(interval_t[tz_idx])
        max_uM = float(max_uM_state["value"])

        if mode_state["mode"] == "from-fit":
            if not has_usable_fit:
                return
            rec_fit = next(
                (f for f in reversed(fits) if f.model in ("Exponential", "IB")), None,
            )
            if rec_fit is None:
                return
            delta = delta_max_from_fit(rec_fit, t_zero)
            if not np.isfinite(delta):
                _update_status(f"Cannot compute Δmax from {rec_fit.model} at this t₀.", "darkred")
                return
            if rec_fit.model == "Exponential":
                y_fit_at_t = float(model_Exponential(np.array([t_zero]), *rec_fit.params)[0])
            else:
                y_fit_at_t = float(model_IB(np.array([t_zero]), *rec_fit.params)[0])
            asym_at_t = float(_model_asymptote(rec_fit, np.array([t_zero]))[0])
            state["computed"] = DeltaMaxRecord(
                method="from-fit", t_zero_s=t_zero, value_uM=float(delta),
            )
            _draw_span(
                t_zero, y_fit_at_t, asym_at_t,
                f"Δ[H₂O₂]max = {delta:.3f} µM\nmethod: from-fit ({rec_fit.model})    t₀ = {t_zero:.2f} s",
            )
            _update_status(
                f"Δmax (from-fit) = {delta:.3f} µM.  Accept or pick a different t₀.",
                colour="#005700",
            )

        elif mode_state["mode"] == "linear":
            if len(state["linear_clicks"]) != 2:
                return
            i0, i1 = sorted(state["linear_clicks"])
            # Use windowed averages around each click so individual-sample
            # noise doesn't dominate the line direction.
            y0_avg = _avg_y(i0)
            y1_avg = _avg_y(i1)
            try:
                delta, slope, intercept = delta_max_from_linear(
                    float(interval_t[i0]), y0_avg,
                    float(interval_t[i1]), y1_avg,
                    t_zero, max_uM=max_uM,
                )
            except ValueError as exc:
                _update_status(f"Linear Δmax: {exc}", "darkred")
                return
            t_seg = np.linspace(interval_t[i0], interval_t[i1], 80)
            y_seg = slope * t_seg + intercept
            ln, = ax.plot(t_seg, y_seg, "r--", lw=1.6, zorder=5)
            state["baseline_artists"].append(ln)
            t_low = min(t_zero, float(interval_t[i0]))
            t_high = max(t_zero, float(interval_t[i1]))
            t_ext = np.linspace(t_low, t_high, 80)
            y_ext = slope * t_ext + intercept
            ln2, = ax.plot(t_ext, y_ext, "r:", lw=1.4, zorder=5)
            state["baseline_artists"].append(ln2)

            y_line_at_t = float(slope * t_zero + intercept)
            # Horizontal purple dotted line at the baseline level so the
            # user can see exactly where the span starts from.
            _draw_baseline_horizontal(y_line_at_t)
            state["computed"] = DeltaMaxRecord(
                method="linear", t_zero_s=t_zero, value_uM=float(delta),
                linear_slope=slope, linear_intercept=intercept,
            )
            _draw_span(
                t_zero, max_uM, y_line_at_t,
                f"Δ[H₂O₂]max = {delta:.3f} µM\nmethod: linear    t₀ = {t_zero:.2f} s    max = {max_uM:.1f} µM",
            )
            _update_status(
                f"Δmax (linear) = {delta:.3f} µM = max ({max_uM:.1f}) − line@t₀ ({y_line_at_t:.3f}).",
                colour="#005700",
            )

        elif mode_state["mode"] == "point":
            if state["point_idx"] is None:
                return
            # Use the windowed average around the click so single-sample
            # noise doesn't skew Δmax.
            y_at_point = _avg_y(state["point_idx"])
            delta = delta_max_from_point(y_at_point, max_uM=max_uM)
            # Horizontal purple dotted line through the picked point so the
            # user sees the "remaining H₂O₂" level extended across.
            _draw_baseline_horizontal(y_at_point)
            state["computed"] = DeltaMaxRecord(
                method="point", t_zero_s=t_zero, value_uM=float(delta),
            )
            _draw_span(
                t_zero, max_uM, y_at_point,
                f"Δ[H₂O₂]max = {delta:.3f} µM\nmethod: point    t₀ = {t_zero:.2f} s    max = {max_uM:.1f} µM",
            )
            _update_status(
                f"Δmax (point) = {delta:.3f} µM = max ({max_uM:.1f}) − point ({y_at_point:.3f}).",
                colour="#005700",
            )

        fig.canvas.draw_idle()
        rec = state["computed"]
        if rec is not None:
            print(f"Δmax preview: method={rec.method}, t_zero={t_zero:.3f}s → {rec.value_uM:.4f} µM")

    def _replay_after_max_change():
        """When the user changes max_uM, redo the active mode's preview with
        the current set of clicks (keeping the t₀ and any baseline picks)."""
        # Save the indices, clear the visuals, redraw markers, recompute.
        saved_tz = state["t_zero_idx"]
        saved_lin = list(state["linear_clicks"])
        saved_pt = state["point_idx"]
        _clear_preview()
        state["t_zero_idx"] = saved_tz
        state["linear_clicks"] = saved_lin
        state["point_idx"] = saved_pt
        if saved_tz is None:
            return
        if mode_state["mode"] == "from-fit":
            rec_fit = next(
                (f for f in reversed(fits) if f.model in ("Exponential", "IB")), None,
            )
            if rec_fit is not None:
                _highlight_fit(rec_fit)
        _draw_t_zero_marker(saved_tz)
        for idx in saved_lin:
            _draw_baseline_point(idx)
        if mode_state["mode"] == "point" and saved_pt is not None:
            _draw_baseline_point(saved_pt)
        _compute_and_preview()

    def on_click(event):
        if event.button != 1 or event.inaxes is not ax or event.xdata is None:
            return
        if getattr(fig.canvas.toolbar, "mode", "") in ("zoom rect", "pan/zoom", "zoom", "pan"):
            return

        if click_target["value"] == "max_uM" and event.ydata is not None:
            y_val = float(event.ydata)
            max_uM_state["value"] = y_val
            # set_val fires _on_max_submit, which already recomputes the
            # preview when a t₀ is set — so we must not recompute again here.
            tb_max.set_val(f"{y_val:.2f}")
            _draw_max_uM_line(y_val)
            click_target["value"] = "t0"
            btn_set_max.color = "0.85"
            if state["t_zero_idx"] is None:
                _update_status(
                    _initial_hint(mode_state["mode"], has_usable_fit, max_uM_state["value"]),
                )
            fig.canvas.draw_idle()
            return

        idx = int(np.abs(interval_t - float(event.xdata)).argmin())

        if mode_state["mode"] == "from-fit":
            _clear_preview()
            state["t_zero_idx"] = idx
            rec_fit = next(
                (f for f in reversed(fits) if f.model in ("Exponential", "IB")), None,
            )
            if rec_fit is not None:
                _highlight_fit(rec_fit)
            _draw_t_zero_marker(idx)
            _compute_and_preview()

        elif mode_state["mode"] == "linear":
            if state["t_zero_idx"] is None:
                state["t_zero_idx"] = idx
                _draw_t_zero_marker(idx)
                _update_status(
                    "Linear: now click TWO points along the baseline / asymptote.",
                    "dimgrey",
                )
            elif len(state["linear_clicks"]) < 2:
                state["linear_clicks"].append(idx)
                _draw_baseline_point(idx)
                fig.canvas.draw_idle()
                remaining = 2 - len(state["linear_clicks"])
                if remaining > 0:
                    _update_status(f"Linear: click {remaining} more point.", "dimgrey")
                else:
                    _compute_and_preview()

        elif mode_state["mode"] == "point":
            if state["t_zero_idx"] is None:
                state["t_zero_idx"] = idx
                _draw_t_zero_marker(idx)
                _update_status(
                    "Point: now click ONE point on the curve representing the baseline.",
                    "dimgrey",
                )
            elif state["point_idx"] is None:
                state["point_idx"] = idx
                _draw_baseline_point(idx)
                fig.canvas.draw_idle()
                _compute_and_preview()

    fig.canvas.mpl_connect("button_press_event", on_click)

    def on_accept(_e=None):
        if state["computed"] is None:
            _update_status("No Δmax computed yet.", "darkred")
            return
        state["action"] = "accept"
        get_window().stop()

    def on_retry_t0(_e=None):
        """Clear ONLY the t₀ marker + dashed line (and the From-fit
        highlight + asymptote line), preserving any baseline picks.
        The user can then re-click to set a new t₀ while keeping their
        Linear / Point baseline selection."""
        _clear_t_zero()
        if state["linear_clicks"] or state["point_idx"] is not None:
            _update_status(
                "t₀ cleared.  Click on the plot to set a new t₀; "
                "the existing baseline points are preserved.",
                "dimgrey",
            )
        else:
            _update_status(_initial_hint(mode_state["mode"], has_usable_fit, max_uM_state["value"]))
        fig.canvas.draw_idle()
        print("Δmax: t₀ cleared.")

    def on_retry_baseline(_e=None):
        """Clear ONLY the baseline picks (Linear's 2 points or Point's 1
        point + the horizontal line), preserving the t₀ anchor.  Lets
        the user keep t₀ and try different baseline points."""
        _clear_baseline()
        if mode_state["mode"] == "from-fit":
            # From-fit has no "baseline picks" — fall back to a full reset
            # behaviour for that mode.
            on_retry_t0(None)
            return
        if state["t_zero_idx"] is not None:
            if mode_state["mode"] == "linear":
                _update_status(
                    "Baseline cleared.  Click TWO new points to redefine the line "
                    "(t₀ preserved).",
                    "dimgrey",
                )
            else:  # point
                _update_status(
                    "Baseline cleared.  Click ONE new point representing the "
                    "remaining-H₂O₂ level (t₀ preserved).",
                    "dimgrey",
                )
        else:
            _update_status(_initial_hint(mode_state["mode"], has_usable_fit, max_uM_state["value"]))
        fig.canvas.draw_idle()
        print("Δmax: baseline cleared.")

    def on_skip(_e=None):
        state["action"] = "skip"
        get_window().stop()

    def on_back(_e=None):
        state["action"] = "back"
        get_window().stop()

    # Five buttons across the bottom:
    # Accept | Retry t₀ | Retry baseline | Skip | Back
    ax_accept = fig.add_axes([0.06, 0.04, 0.14, 0.05])
    ax_retry_tz = fig.add_axes([0.22, 0.04, 0.14, 0.05])
    ax_retry_base = fig.add_axes([0.38, 0.04, 0.16, 0.05])
    ax_skip = fig.add_axes([0.56, 0.04, 0.14, 0.05])
    ax_back = fig.add_axes([0.72, 0.04, 0.18, 0.05])
    btn_accept = create_small_button(ax_accept, "Accept Δmax", "#90ee90", "#7cd47c")
    btn_accept.on_clicked(on_accept)
    btn_retry_tz = create_small_button(ax_retry_tz, "Retry t₀", "0.9", "0.8")
    btn_retry_tz.on_clicked(on_retry_t0)
    btn_retry_base = create_small_button(ax_retry_base, "Retry baseline", "0.9", "0.8")
    btn_retry_base.on_clicked(on_retry_baseline)
    btn_skip = create_small_button(ax_skip, "Skip Δmax", "#ffcc99", "#ffaa66")
    btn_skip.on_clicked(on_skip)
    btn_back = create_small_button(ax_back, "Back → fits", "#ddddff", "#bbbbff")
    btn_back.on_clicked(on_back)

    install_zoom_keys(fig, ax)
    get_window().run()

    if state["action"] == "accept" and state["computed"] is not None:
        return state["computed"]
    return state["action"] or "skip"


# ────────────────────────────────────────────────────────────────────────
# UI: new-control modal (file picker + stripped processing flow)
# ────────────────────────────────────────────────────────────────────────


def _qt_pick_control_file(
    input_dir: Path | None,
    subtracting_from: str | None = None,
) -> Path | None:
    """Open a Qt file dialog and let the user pick a control file.  Falls back
    to None if Qt isn't available.

    ``subtracting_from`` names the run the control will be subtracted from
    and is shown in the dialog title, so the sample is never out of sight
    while choosing."""
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
    caption = "SensorFit — pick a control file to process"
    if subtracting_from:
        caption += f"  —  to subtract FROM: {subtracting_from}"
    selected, _ = QFileDialog.getOpenFileName(
        None,
        caption,
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
    cal_max_uM: float = 100.0,
) -> list[ProcessedInterval] | str:
    """Drive the per-interval state machine until the user says "Done".

    ``new_control_callback`` is an optional callable
    ``(picked_path: Path) -> (t: np.ndarray, y: np.ndarray) | None`` that
    the caller provides to process a new control file modal-style and return
    its reference interval.  If None, the "Add new" button in the averaging
    hub will be disabled.

    Returns the list of accepted ``ProcessedInterval``s, or the sentinel
    string ``"go_back_to_calibration"`` if the user asked to back out before
    any interval was accepted.
    """
    if session_intervals is None:
        session_intervals = []

    accepted: list[ProcessedInterval] = []
    interval_counter = 1
    self_stem = Path(filename).stem if filename else None

    # ── Outer loop: one iteration per interval the user accepts ────────
    while True:
        # ── 1) Pick the interval ─────────────────────────────────────
        already = [(p.start_time, p.end_time) for p in accepted]
        sel = select_one_interval(
            time_values, h2o2_values, already,
            filename=filename, interval_number=interval_counter,
        )
        if sel == "done" or sel == "skip":
            if not accepted:
                return []
            break
        if sel == "back":
            # The user wants to back out to calibration regardless of whether
            # they have any accepted intervals — the "Remove last" button
            # handles the case where they just want to drop a few.
            if not accepted:
                return "go_back_to_calibration"
            # If there ARE accepted intervals already, treat Back as "stop
            # adding new intervals" — same outcome as Done from the caller's
            # perspective.
            return accepted if accepted else "go_back_to_calibration"
        if sel == "remove_last":
            if accepted:
                dropped = accepted.pop()
                interval_counter = max(1, interval_counter - 1)
                print(f"Removed interval #{dropped.index}.")
            else:
                print("No accepted intervals to remove.")
            continue
        if not isinstance(sel, tuple):
            continue

        s_t, e_t = sel
        mask = (time_values >= s_t) & (time_values <= e_t)
        if not np.any(mask):
            print("Empty interval; pick again.")
            continue

        # Build the per-interval state.  These are mutable across step
        # transitions inside the inner state machine below.
        interval_df = full_frame.loc[mask, [time_col, CALIBRATED_COLUMN]].copy().reset_index(drop=True)
        interval_t = interval_df[time_col].to_numpy(dtype=float)
        original_y = interval_df[CALIBRATED_COLUMN].to_numpy(dtype=float)
        interval_y = original_y.copy()
        control_subtracted = False
        control_source = None
        control_n_averaged = 0
        fits: list[FitRecord] = []
        delta: DeltaMaxRecord | None = None

        # ── Inner state machine: subtract → fit → delta_max → done ──
        step = "subtract"
        abort_to_pick = False

        while step != "done":
            if step == "subtract":
                sub_choice = prompt_subtraction_choice(
                    f"Interval #{interval_counter}: [{s_t:.1f}–{e_t:.1f} s]"
                )
                if sub_choice == "back":
                    abort_to_pick = True
                    break

                # Reset subtraction state on re-entry (e.g. Back from fit).
                control_subtracted = False
                control_source = None
                control_n_averaged = 0
                interval_y = original_y.copy()
                interval_df[CALIBRATED_COLUMN] = interval_y

                if sub_choice in ("existing", "new"):
                    initial_members: list[ControlMember] = []
                    candidates = _discover_existing_control_intervals(
                        calibrated_dir, session_intervals, self_stem
                    )

                    if sub_choice == "existing":
                        picked = pick_existing_intervals_multi(candidates, filename=filename)
                        if picked:
                            for p in picked:
                                ct, cy = p["load"]()
                                initial_members.append((p["label"], ct - ct[0], cy))

                    elif sub_choice == "new":
                        if new_control_callback is None:
                            print(
                                "No callback for new-control processing.  "
                                "Process the control as a normal file first, "
                                "then choose 'Subtract existing'."
                            )
                        else:
                            picked_path = _qt_pick_control_file(
                                calibrated_dir.parent if calibrated_dir is not None else None,
                                subtracting_from=filename,
                            )
                            if picked_path is not None:
                                template = new_control_callback(picked_path)
                                if template is not None:
                                    ct, cy = template
                                    initial_members.append((f"new: {picked_path.name}", ct - ct[0], cy))

                    if initial_members:
                        hub_decision, hub_members, hub_anchor = averaging_hub(
                            interval_t, interval_y, initial_members,
                            filename=filename,
                            existing_candidates=candidates,
                            new_control_callback=new_control_callback,
                            calibrated_dir=calibrated_dir,
                        )
                        if hub_decision == "accept" and hub_members:
                            # Reuse the hub's own alignment and anchor —
                            # recomputing from interval_t[0] here would
                            # silently apply a different subtraction from
                            # the one the user just previewed.
                            pairs = [(ct, cy) for (_lbl, ct, cy) in hub_members]
                            averaged, _ = average_controls_on_grid(
                                pairs, interval_t, float(interval_t[0])
                            )
                            dev = deviation_from_anchor(
                                averaged, interval_t, hub_anchor
                            )
                            interval_y = original_y - dev
                            interval_df[CALIBRATED_COLUMN] = interval_y
                            control_subtracted = True
                            control_n_averaged = len(hub_members)
                            labels = [lbl for (lbl, _, _) in hub_members]
                            control_source = " | ".join(labels)
                        elif hub_decision == "back":
                            continue

                step = "fit"
                continue

            elif step == "fit":
                fit_index = len(fits) + 1
                print(f"\nFitting interval #{interval_counter} — fit #{fit_index}")
                res = prompt_one_fit(
                    interval_t, interval_y,
                    filename=filename, fit_index=fit_index,
                    existing_fits=fits,
                )
                if isinstance(res, FitRecord):
                    fits.append(res)
                    print(f"  ✓ Fit #{fit_index} ({res.model}) accepted.")
                    if not _ask_another("Apply another fit to this interval?"):
                        step = "delta_max"
                    continue
                if res == "skip":
                    print("  → fit skipped.")
                    step = "delta_max"
                    continue
                if res == "retry":
                    continue
                if res == "back":
                    step = "subtract"
                    continue
                step = "delta_max"

            elif step == "delta_max":
                dres = prompt_delta_max(
                    interval_t, interval_y, fits,
                    filename=filename,
                    cal_max_uM=cal_max_uM,
                )
                if isinstance(dres, DeltaMaxRecord):
                    delta = dres
                    print(f"  ✓ Δmax recorded ({delta.method}: {delta.value_uM:.4f} µM)")
                    step = "done"
                    continue
                if dres == "skip":
                    delta = None
                    step = "done"
                    continue
                if dres == "back":
                    step = "fit"
                    continue
                step = "done"

        if abort_to_pick:
            continue

        accepted.append(ProcessedInterval(
            index=interval_counter,
            start_time=float(s_t),
            end_time=float(e_t),
            data=interval_df,
            time_col=time_col,
            control_subtracted=control_subtracted,
            control_source=control_source,
            control_n_averaged=control_n_averaged,
            fits=fits,
            delta_max=delta,
        ))
        interval_counter += 1
        # No "Define another interval?" yes/no popup — the picker has its own
        # "Done with intervals" button, which is more direct.

    return accepted


def _ask_another(question: str) -> bool:
    """Small yes/no popup; returns True if user clicks Yes."""
    fig = get_window().reset(figsize=(7, 2.5))
    ax = fig.add_subplot(111)
    try:
        fig.canvas.manager.set_window_title("SensorFit — Continue?")
    except Exception:
        pass
    ax.axis("off")
    ax.text(0.5, 0.55, question, ha="center", va="center", fontsize=11)
    state = {"choice": False}

    def _yes(_e=None):
        state["choice"] = True
        get_window().stop()

    def _no(_e=None):
        state["choice"] = False
        get_window().stop()

    ax_yes = fig.add_axes([0.22, 0.10, 0.22, 0.18])
    ax_no = fig.add_axes([0.56, 0.10, 0.22, 0.18])
    _btn_yes_18 = create_small_button(ax_yes, "Yes", "#90ee90", "#7cd47c")
    _btn_yes_18.on_clicked(_yes)
    _btn_no_19 = create_small_button(ax_no, "No", "0.9", "0.8")
    _btn_no_19.on_clicked(_no)
    get_window().run()
    return state["choice"]
