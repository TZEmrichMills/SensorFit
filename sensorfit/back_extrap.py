"""H2O2-injection-start back-extrapolation.

For reactions that begin with an H2O2 addition and use a two-point calibration
(pre-addition = 0 µM; just-after-addition = nominal µM), the calibrated values
under-estimate the true initial concentration because the enzyme has already
consumed some H2O2 during the instrument deadtime (~1–2 s).  Correction:

1. Back-extrapolate the fitted Exponential model to ``t = t_start − deadtime``
   to recover the "true" [H2O2] at the moment of addition.
2. Compute a stretch factor ``back_extrap / nominal_uM`` describing how much
   the original two-point calibration under-estimated.
3. Compute the corrected initial rate: ``rate_at_back × stretch_factor``.
   The first factor captures the steeper slope at the earlier time (the
   exponential has not yet decayed); the second rescales for the calibration
   correction.

If no Exponential fit is available we fall back to a linear extrapolation
using whatever initial-rate is on file.  Less accurate, but works.

This module is invoked from ``calibration_cli.process_file`` as a new phase
between ``fitting`` and ``turnover`` — only for intervals where the user
answers Yes to the per-interval prompt.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
import numpy as np

from .calibration import (
    CALIBRATED_COLUMN,
    add_instruction_banner,
    create_small_button,
    truncate_filename,
)
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
        ``{model_name: {init_rate, params, ...}}`` for this interval — the
        same dict that ``append_fit_summary`` consumes.
    interval_t_start_abs : float
        Absolute time at the start of the interval (in the full-trace time
        coordinate the fits were computed in).
    interval_observed_value : float
        Observed [H2O2] at the interval start (used as the linear-extrapolation
        anchor when no Exponential fit is available).
    deadtime_s : float
        Seconds between the H2O2 addition and the first reliable reading.
    nominal_uM : float
        The nominal H2O2 concentration the user added (the value the
        two-point calibration treated as the "after" point).

    Returns
    -------
    dict with keys:
        ``method`` ("exponential" or "linear")
        ``deadtime_s``, ``nominal_uM``
        ``back_extrap_uM`` — fitted [H2O2] at t_start − deadtime
        ``stretch_factor`` — back_extrap_uM / nominal_uM
        ``rate_at_back_uM_per_s`` — derivative of the fitted Exponential at
            the back-extrapolated time (== initial_rate for the linear case)
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
        # Derivative of a*t + b + c*exp(-k*(t-t0)) is a - c*k*exp(-k*(t-t0))
        rate_at_back = float(a - c * k_decay * np.exp(-k_decay * (t_back - t0)))
        method = "exponential"
    else:
        # Linear fallback: use whatever init_rate we can find
        rate = None
        for model in ("LinearInitialRate", "IB", "GFI"):
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
        # Linear back-extrap: y(t_back) = y(t_start) + rate * (t_back - t_start)
        # Since t_back < t_start and rate is negative for consumption,
        # y(t_back) > y(t_start), as expected.
        back_extrap_uM = float(interval_observed_value + rate * (t_back - interval_t_start_abs))
        rate_at_back = rate  # constant rate under linear assumption
        method = "linear"

    nominal = float(nominal_uM)
    if abs(nominal) < 1e-12:
        stretch_factor = float("nan")
    else:
        stretch_factor = back_extrap_uM / nominal
    stretch_initial_rate = rate_at_back * stretch_factor if np.isfinite(stretch_factor) else float("nan")

    return {
        "method": method,
        "deadtime_s": float(deadtime_s),
        "nominal_uM": nominal,
        "back_extrap_uM": back_extrap_uM,
        "stretch_factor": stretch_factor,
        "rate_at_back_uM_per_s": rate_at_back,
        "stretch_initial_rate": stretch_initial_rate,
    }


# ──────────────────────────────────────────────────────────────────────────
# Interactive per-interval prompt
# ──────────────────────────────────────────────────────────────────────────


def prompt_back_extrap_for_interval(
    subset,
    fit_results_for_interval: dict[str, dict],
    time_col: str,
    filename: str | None = None,
    default_deadtime: float = 1.5,
):
    """Show the interval with its fitted Exponential (if any), let the user
    enter the deadtime + nominal-H2O2, preview the back-extrapolation, and
    accept / skip / go back.

    Returns
    -------
    one of:
      dict  — the result from ``compute_back_extrap``.  Caller should record it
              in fit_summary.
      "skip"           — user chose not to back-extrap for this interval.
      "back"           — user wants to revisit the previous phase.
    """
    t = subset.data[time_col].to_numpy(dtype=float)
    y = subset.data[CALIBRATED_COLUMN].to_numpy(dtype=float)
    t_start_abs = float(t[0])
    observed_value = float(y[0])

    state = {
        "deadtime": float(default_deadtime),
        "nominal": float(observed_value if observed_value > 0 else 100.0),
        "decision": None,
        "result": None,
    }

    fig, ax = plt.subplots(figsize=(11, 6.5))
    plt.subplots_adjust(left=0.1, bottom=0.32, right=0.98, top=0.85)

    ax.plot(t, y, color="tab:green", lw=1.3, label="Calibrated trace")

    # Overlay the Exponential fit if present
    extrap_line = None
    if "Exponential" in fit_results_for_interval:
        exp = fit_results_for_interval["Exponential"]
        yhat = np.asarray(exp.get("yhat", []), dtype=float)
        if yhat.size == t.size:
            ax.plot(t, yhat, color="tab:orange", lw=1.0, ls="--", label="Exponential fit")
        extrap_line, = ax.plot([], [], color="tab:red", lw=1.2, ls="--", label="Back-extrapolation")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("H2O2 (µM)")
    ax.legend(loc="best", fontsize=9)

    title_parts = []
    if filename:
        title_parts.append(truncate_filename(filename))
    title_parts.append(f"Interval #{subset.index}  ({subset.start_time:.1f}–{subset.end_time:.1f} s)")
    fig.suptitle("  |  ".join(title_parts), fontsize=10)

    add_instruction_banner(
        fig,
        "Enter the deadtime (s) and the nominal H₂O₂ added (µM), then press "
        "Compute to preview.  Accept records the back-extrapolation in "
        "fit_summary; Skip leaves this interval as-is.",
        y=0.99,
    )

    info_text = fig.text(
        0.5, 0.245,
        "Awaiting input…", ha="center", va="center",
        fontsize=10, family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9, pad=0.4),
    )

    # Input boxes
    ax_dead = fig.add_axes([0.17, 0.16, 0.10, 0.05])
    ax_nom = fig.add_axes([0.42, 0.16, 0.10, 0.05])
    tb_dead = TextBox(ax_dead, "Deadtime (s) ", initial=f"{state['deadtime']:.2f}")
    tb_nom = TextBox(ax_nom, "Nominal µM ", initial=f"{state['nominal']:.2f}")

    def update_preview(_=None) -> None:
        try:
            state["deadtime"] = float(tb_dead.text)
            state["nominal"] = float(tb_nom.text)
        except ValueError:
            info_text.set_text("Invalid number entered.")
            fig.canvas.draw_idle()
            return
        try:
            result = compute_back_extrap(
                fit_results_for_interval,
                interval_t_start_abs=t_start_abs,
                interval_observed_value=observed_value,
                deadtime_s=state["deadtime"],
                nominal_uM=state["nominal"],
            )
        except ValueError as exc:
            info_text.set_text(str(exc))
            fig.canvas.draw_idle()
            return
        state["result"] = result
        msg = (
            f"method={result['method']}    "
            f"back-extrap [H₂O₂]={result['back_extrap_uM']:.3f} µM    "
            f"stretch factor={result['stretch_factor']:.4f}\n"
            f"rate at true t₀ = {result['rate_at_back_uM_per_s']:.4f} µM/s    "
            f"stretch initial rate = {result['stretch_initial_rate']:.4f} µM/s"
        )
        info_text.set_text(msg)
        # Visualise the back-extrap extension on the plot
        if extrap_line is not None and result["method"] == "exponential":
            params = np.asarray(
                fit_results_for_interval["Exponential"].get("params", []), dtype=float
            )
            tt = np.linspace(t_start_abs - state["deadtime"], t_start_abs, 30)
            yy = model_Exponential(tt, *params)
            extrap_line.set_data(tt, yy)
            ax.relim()
            ax.autoscale_view()
        fig.canvas.draw_idle()

    tb_dead.on_submit(update_preview)
    tb_nom.on_submit(update_preview)

    # Buttons
    def on_compute(_e=None) -> None:
        update_preview()

    def on_accept(_e=None) -> None:
        if state["result"] is None:
            update_preview()
        if state["result"] is None:
            return
        state["decision"] = "accept"
        plt.close(fig)

    def on_skip(_e=None) -> None:
        state["decision"] = "skip"
        plt.close(fig)

    def on_back(_e=None) -> None:
        state["decision"] = "back"
        plt.close(fig)

    ax_compute = fig.add_axes([0.58, 0.16, 0.10, 0.05])
    ax_accept = fig.add_axes([0.10, 0.05, 0.18, 0.06])
    ax_skip = fig.add_axes([0.32, 0.05, 0.18, 0.06])
    ax_back = fig.add_axes([0.54, 0.05, 0.18, 0.06])
    btn_compute = create_small_button(ax_compute, "Compute", "0.9", "0.8")
    btn_accept = create_small_button(ax_accept, "Accept & record", "#90ee90", "#7cd47c")
    btn_skip = create_small_button(ax_skip, "Skip this interval", "#ffcc99", "#ffaa66")
    btn_back = create_small_button(ax_back, "Back", "#ddddff", "#bbbbff")
    btn_compute.on_clicked(on_compute)
    btn_accept.on_clicked(on_accept)
    btn_skip.on_clicked(on_skip)
    btn_back.on_clicked(on_back)

    update_preview()  # initial render with defaults
    plt.show()
    plt.close(fig)

    if state["decision"] == "accept":
        return state["result"]
    if state["decision"] == "skip":
        return "skip"
    return "back"


def offer_back_extrap_for_file(
    subsets,
    fit_results: dict[int, dict[str, dict]],
    time_col: str,
    filename: str | None = None,
    default_deadtime: float = 1.5,
) -> dict[int, dict]:
    """Iterate over each interval that has been fit and offer back-extrapolation.

    Returns
    -------
    dict[interval_index, dict]
        Map from interval index → ``compute_back_extrap`` result for every
        interval the user accepted.  Empty if the user skipped them all.
    """
    out: dict[int, dict] = {}
    if not fit_results:
        return out

    # Top-level yes/no first — most files don't need this, so we avoid a
    # per-interval barrage by default.
    if not _prompt_yes_no_back_extrap():
        return out

    for subset in subsets:
        fits = fit_results.get(subset.index)
        if not fits:
            continue
        result = prompt_back_extrap_for_interval(
            subset,
            fits,
            time_col=time_col,
            filename=filename,
            default_deadtime=default_deadtime,
        )
        if isinstance(result, dict):
            out[subset.index] = result
        elif result == "back":
            # User wants to abort the back-extrap phase entirely
            return out
        # "skip" → just don't record anything for this interval and move on
    return out


def _prompt_yes_no_back_extrap() -> bool:
    """Top-level yes/no before iterating through intervals."""
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.axis("off")
    ax.text(
        0.5, 0.62,
        "H₂O₂-injection-start back-extrapolation?\n\n"
        "If any interval in this file began with an H₂O₂ addition (so the\n"
        "instrument deadtime caused the calibrated [H₂O₂] to be\n"
        "under-estimated), click Yes to be prompted per interval.\n"
        "Otherwise click No to continue.",
        ha="center", va="center", fontsize=10,
    )
    fig.suptitle("Back-extrapolation (opt-in)", fontsize=11, fontweight="bold")
    state = {"choice": False}

    def on_yes(_e=None):
        state["choice"] = True
        plt.close(fig)

    def on_no(_e=None):
        state["choice"] = False
        plt.close(fig)

    ax_yes = fig.add_axes([0.18, 0.10, 0.28, 0.16])
    ax_no = fig.add_axes([0.54, 0.10, 0.28, 0.16])
    btn_yes = create_small_button(ax_yes, "Yes — per interval", "#90ee90", "#7cd47c")
    btn_no = create_small_button(ax_no, "No — skip", "0.9", "0.8")
    btn_yes.on_clicked(on_yes)
    btn_no.on_clicked(on_no)
    plt.show()
    plt.close(fig)
    return state["choice"]
