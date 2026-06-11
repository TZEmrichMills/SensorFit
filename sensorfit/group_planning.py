"""Group-level subtraction planning.

After all controls and samples in a ``ControlGroup`` have been processed
(each through baseline / calibration / intervals / fits / turnover), this
module drives the *planning* step where the user decides which sample
intervals get which subtraction chain.  Then it applies subtractions, lets
the user re-fit / re-turnover the corrected intervals, and returns the
updated state ready to be persisted to ``fit_summary.xlsx``.

Subtraction math is the user's chosen design:

    corrected_sample(t) = sample(t)
                        − mean(sub-group 1's templates)(t)
                        − mean(sub-group 2's templates)(t)
                        − …

Sub-group averaging is done on a common time grid; sequential subtraction
is then just successive ``−``s.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, CheckButtons
import numpy as np
import pandas as pd

from .calibration import (
    CALIBRATED_COLUMN,
    IntervalSubset,
    add_instruction_banner,
    create_small_button,
    truncate_filename,
)
from .controls import (
    ControlGroup,
    ControlSpec,
    interpolate_control_to_grid,
    load_control_template,
)


@dataclass
class SampleState:
    """Everything we need to remember about one sample file as we drive a
    group through the planning phase.
    """

    file_name: str
    calibrated_frame: pd.DataFrame
    time_col: str
    subsets: list[IntervalSubset]
    fit_results: dict[int, dict[str, dict]] = field(default_factory=dict)
    turnover_results: dict[int, float | None] = field(default_factory=dict)
    # populated after planning:
    corrected_subsets: dict[int, IntervalSubset] = field(default_factory=dict)
    correction_meta: dict[int, dict] = field(default_factory=dict)
    refit_results: dict[int, dict[str, dict]] = field(default_factory=dict)
    refit_turnover: dict[int, float | None] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────
# Subtraction chain math
# ──────────────────────────────────────────────────────────────────────────

def _average_subgroup_on_grid(
    templates: list[tuple[np.ndarray, np.ndarray]],
    target_t: np.ndarray,
) -> np.ndarray:
    """Interpolate each template onto ``target_t`` and average element-wise.

    Each ``templates`` entry is ``(time_array, h2o2_array)``.  Returns the
    averaged H2O2 vs target_t.  If only one template, equivalent to plain
    interpolation.  Templates may have different lengths and start/end
    times — they are independently aligned to ``target_t``.
    """
    if not templates:
        return np.zeros_like(target_t, dtype=float)
    aligned = []
    for t, y in templates:
        interp, _info = interpolate_control_to_grid(t, y, target_t)
        aligned.append(interp)
    return np.mean(np.stack(aligned, axis=0), axis=0)


def build_subtraction_chain(
    sample_time: np.ndarray,
    subgroup_templates: list[list[tuple[np.ndarray, np.ndarray]]],
    anchor_t0: float,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Compute the combined correction to subtract from a sample interval.

    Parameters
    ----------
    sample_time : np.ndarray
        The sample interval's time grid (absolute time).
    subgroup_templates : list of list of (t, y)
        Templates grouped by sub-group.  Each inner list is averaged
        element-wise on the sample's grid; the outer list is summed
        (sequential subtractions add up).
    anchor_t0 : float
        The absolute time at which each template's t=0 should sit on the
        sample's grid.  Almost always the sample interval's start time.

    Returns
    -------
    combined : np.ndarray
        Sum of (each sub-group's average) — what gets subtracted from the
        sample.
    per_subgroup : list[np.ndarray]
        Each sub-group's averaged contribution, in the same order as the
        input.  Useful for visualisation.
    """
    per_subgroup = []
    for templates in subgroup_templates:
        shifted = [(t + anchor_t0, y) for (t, y) in templates]
        avg = _average_subgroup_on_grid(shifted, sample_time)
        per_subgroup.append(avg)
    if not per_subgroup:
        return np.zeros_like(sample_time, dtype=float), []
    combined = np.sum(np.stack(per_subgroup, axis=0), axis=0)
    return combined, per_subgroup


def _load_group_subgroup_templates(
    group: ControlGroup, calibrated_dir: Path
) -> list[list[tuple[np.ndarray, np.ndarray]]]:
    """Load every saved template for the group, organised by sub-group.

    Sub-groups whose controls all have ``template_filename = None`` are
    skipped (cannot contribute).  Missing template files raise.
    """
    out: list[list[tuple[np.ndarray, np.ndarray]]] = []
    for sg in group.subgroups:
        tpls = []
        for spec in sg.controls:
            if spec.template_filename is None:
                continue
            t, y = load_control_template(calibrated_dir, spec.template_filename)
            tpls.append((t, y))
        if tpls:
            out.append(tpls)
    return out


# ──────────────────────────────────────────────────────────────────────────
# Planning UI: per-sample-interval picker
# ──────────────────────────────────────────────────────────────────────────

CHOICE_NONE = "None"
CHOICE_FULL = "Apply full chain"


def planning_picker(
    group: ControlGroup,
    samples: list[SampleState],
    calibrated_dir: Path,
) -> dict[tuple[str, int], dict] | None:
    """Open a picker letting the user choose, per sample-interval, whether
    to apply the group's subtraction chain.

    Returns
    -------
    dict[(file_name, interval_index)] → {choice, anchor_t0}
        Only populated for intervals where the user chose ≠ "None".
        ``choice`` is either ``CHOICE_FULL`` (use all sub-groups) or a
        list of sub-group indices to apply.
        ``anchor_t0`` is the absolute time used to align the templates
        on the sample interval (default = interval start).
    Returns ``None`` if the user cancels.
    """
    # Flatten sample intervals into rows
    rows: list[tuple[str, int, float, float]] = []
    for s in samples:
        for sub in s.subsets:
            rows.append((s.file_name, sub.index, sub.start_time, sub.end_time))
    if not rows:
        return {}

    # Default everything to "None" per design choice
    state_choice: dict[tuple[str, int], bool] = {(fn, idx): False for fn, idx, _, _ in rows}

    # Title information
    n_subgroups = len(group.subgroups)
    title = (
        f"Group '{group.name}': planning subtraction\n"
        f"{n_subgroups} sub-group(s) in chain"
    )

    fig = plt.figure(figsize=(9.5, 0.6 + 0.32 * max(4, len(rows)) + 2.0))
    fig.suptitle(title, fontsize=11)
    add_instruction_banner(
        fig,
        "Tick the box for each sample interval that should receive the "
        "group's subtraction chain (averaged within each sub-group, then "
        "subtracted sequentially).  Leave un-ticked for intervals you don't "
        "want corrected (e.g. calibration ladders).  Click Apply to compute "
        "the corrections and preview them, or Cancel to abort.",
        y=0.985,
        width=110,
    )

    labels = [
        f"{fn}  —  interval #{idx}  ({s:.1f}–{e:.1f} s)"
        for fn, idx, s, e in rows
    ]
    initial_state = [state_choice[(fn, idx)] for fn, idx, _, _ in rows]
    rax = fig.add_axes([0.06, 0.18, 0.88, 0.65])
    check = CheckButtons(rax, labels, initial_state)

    decision = {"action": None}

    def on_apply(_e=None):
        decision["action"] = "apply"
        plt.close(fig)

    def on_cancel(_e=None):
        decision["action"] = "cancel"
        plt.close(fig)

    def on_select_all(_e=None):
        for i, state in enumerate(check.get_status()):
            if not state:
                check.set_active(i)

    def on_select_none(_e=None):
        for i, state in enumerate(check.get_status()):
            if state:
                check.set_active(i)

    ax_apply = fig.add_axes([0.10, 0.04, 0.18, 0.07])
    ax_all = fig.add_axes([0.30, 0.04, 0.14, 0.07])
    ax_none = fig.add_axes([0.46, 0.04, 0.14, 0.07])
    ax_cancel = fig.add_axes([0.72, 0.04, 0.18, 0.07])
    btn_apply = create_small_button(ax_apply, "Apply", "#90ee90", "#7cd47c")
    btn_all = create_small_button(ax_all, "Select all", "0.9", "0.8")
    btn_none = create_small_button(ax_none, "Select none", "0.9", "0.8")
    btn_cancel = create_small_button(ax_cancel, "Cancel", "#ddddff", "#bbbbff")
    btn_apply.on_clicked(on_apply)
    btn_all.on_clicked(on_select_all)
    btn_none.on_clicked(on_select_none)
    btn_cancel.on_clicked(on_cancel)

    plt.show()
    plt.close(fig)

    if decision["action"] != "apply":
        return None

    out: dict[tuple[str, int], dict] = {}
    for (fn, idx, start, _end), checked in zip(rows, check.get_status()):
        if checked:
            out[(fn, idx)] = {"choice": CHOICE_FULL, "anchor_t0": float(start)}
    return out


# ──────────────────────────────────────────────────────────────────────────
# Subtraction preview (per accepted interval)
# ──────────────────────────────────────────────────────────────────────────

def preview_and_apply_subtraction(
    sample_state: SampleState,
    interval_index: int,
    group: ControlGroup,
    subgroup_templates: list[list[tuple[np.ndarray, np.ndarray]]],
    anchor_t0: float,
):
    """Open a preview screen showing the sample interval + each sub-group's
    averaged contribution + the corrected result.  User can re-anchor or
    accept / skip.

    Returns
    -------
    one of:
      (corrected_subset, meta_dict) — accepted; `corrected_subset` is a new
        IntervalSubset with the H2O2_uM column replaced by the corrected
        values.  `meta_dict` carries provenance for fit_summary.
      "skip" — user chose to keep the original (uncorrected) interval.
      "back" — user wants to revisit planning.
    """
    sub = next(s for s in sample_state.subsets if s.index == interval_index)
    sample_t = sub.data[sample_state.time_col].to_numpy(dtype=float)
    sample_y = sub.data[CALIBRATED_COLUMN].to_numpy(dtype=float)

    anchor = {"t0": float(anchor_t0)}

    fig, ax = plt.subplots(2, 1, figsize=(11, 7.4), sharex=True)
    plt.subplots_adjust(left=0.1, bottom=0.18, right=0.98, top=0.80, hspace=0.25)

    ax[0].plot(sample_t, sample_y, color="tab:green", lw=1.3, label="Sample interval")
    line_combined, = ax[0].plot([], [], color="tab:red", lw=1.4, alpha=0.85, label="Combined correction")
    sg_lines = []
    sg_colors = plt.cm.Oranges(np.linspace(0.4, 0.9, len(subgroup_templates))) if subgroup_templates else []
    for i, col in enumerate(sg_colors):
        ln, = ax[0].plot([], [], color=col, lw=1.0, ls="--", alpha=0.7,
                         label=f"Sub-group {i + 1} contribution")
        sg_lines.append(ln)
    anchor_line_top = ax[0].axvline(anchor["t0"], color="#8B008B", lw=1.8, ls="--", alpha=0.9, label="Anchor (t=0)")
    anchor_line_bot = ax[1].axvline(anchor["t0"], color="#8B008B", lw=1.8, ls="--", alpha=0.9)
    anchor_text = ax[0].annotate(
        "", xy=(anchor["t0"], 0), xycoords=("data", "axes fraction"),
        xytext=(6, 6), textcoords="offset points",
        color="#8B008B", fontsize=9, fontweight="bold",
    )
    ax[0].set_ylabel("H2O2 (µM)")
    ax[0].legend(loc="upper right", fontsize=8)

    line_corrected, = ax[1].plot([], [], color="tab:blue", lw=1.3, label="Corrected (sample − combined)")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("H2O2 (µM, corrected)")
    ax[1].axhline(0.0, color="grey", lw=0.6, ls=":")
    ax[1].legend(loc="upper right", fontsize=9)

    add_instruction_banner(
        fig,
        f"Group '{group.name}', sample {truncate_filename(sample_state.file_name)}, interval #{interval_index}.  "
        "Top: sample (green) + each sub-group's averaged contribution (orange dashed) + their sum (red).  "
        "Bottom: sample − combined.  Click on the upper plot to re-anchor the controls' t=0.  "
        "Accept records the correction; Skip keeps the original interval as-is.",
        y=0.985,
        width=110,
    )

    def redraw():
        combined, per_sg = build_subtraction_chain(
            sample_t, subgroup_templates, anchor["t0"]
        )
        line_combined.set_data(sample_t, combined)
        for ln, contrib in zip(sg_lines, per_sg):
            ln.set_data(sample_t, contrib)
        line_corrected.set_data(sample_t, sample_y - combined)
        anchor_line_top.set_xdata([anchor["t0"], anchor["t0"]])
        anchor_line_bot.set_xdata([anchor["t0"], anchor["t0"]])
        anchor_text.set_text(f"anchor t={anchor['t0']:.2f} s")
        anchor_text.xy = (anchor["t0"], 0)
        for a in ax:
            a.relim()
            a.autoscale_view()
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes is not ax[0] or event.button != 1 or event.xdata is None:
            return
        toolbar = fig.canvas.toolbar
        if toolbar is not None and getattr(toolbar, "mode", "") in ("zoom rect", "pan/zoom", "zoom", "pan"):
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
    btn_accept = create_small_button(ax_accept, "Accept & subtract", "#90ee90", "#7cd47c")
    btn_reset = create_small_button(ax_reset, "Reset anchor", "0.9", "0.8")
    btn_skip = create_small_button(ax_skip, "Skip this interval", "#ffcc99", "#ffaa66")
    btn_back = create_small_button(ax_back, "Back", "#ddddff", "#bbbbff")
    btn_accept.on_clicked(on_accept)
    btn_reset.on_clicked(on_reset)
    btn_skip.on_clicked(on_skip)
    btn_back.on_clicked(on_back)

    redraw()
    from .zoom_hotkey import install_zoom_keys
    install_zoom_keys(fig, list(ax))
    plt.show()
    plt.close(fig)

    if decision["value"] == "accept":
        combined, _ = build_subtraction_chain(sample_t, subgroup_templates, anchor["t0"])
        corrected_y = sample_y - combined
        corrected_data = sub.data.copy()
        corrected_data[CALIBRATED_COLUMN] = corrected_y
        corrected_sub = IntervalSubset(
            index=sub.index,
            start_time=sub.start_time,
            end_time=sub.end_time,
            data=corrected_data,
        )
        meta = {
            "group_name": group.name,
            "n_subgroups": len(subgroup_templates),
            "anchor_t0": float(anchor["t0"]),
            "subgroup_descriptions": [
                f"Sub-group {i + 1}: " + ", ".join(c.file_name for c in sg.controls if c.template_filename)
                for i, sg in enumerate(group.subgroups)
            ],
        }
        return corrected_sub, meta
    if decision["value"] == "skip":
        return "skip"
    return "back"


# ──────────────────────────────────────────────────────────────────────────
# Top-level driver
# ──────────────────────────────────────────────────────────────────────────

def run_group_planning(
    group: ControlGroup,
    samples: list[SampleState],
    calibrated_dir: Path,
) -> dict[str, SampleState]:
    """Drive the planning phase for one group.

    1) Load all sub-group templates.
    2) Open the per-sample-interval picker.
    3) For each picked interval: preview + accept/skip/back.
    4) Return updated SampleState dict (file_name → state) with
       ``corrected_subsets`` and ``correction_meta`` populated.

    Re-fit / re-turnover is handled by the caller via
    ``offer_refit_for_corrected_intervals``.
    """
    if not samples:
        return {}

    try:
        subgroup_templates = _load_group_subgroup_templates(group, calibrated_dir)
    except FileNotFoundError as exc:
        print(f"Warning: could not load all control templates for group '{group.name}': {exc}")
        return {s.file_name: s for s in samples}

    if not subgroup_templates:
        print(f"Group '{group.name}' has no usable control templates — nothing to subtract.")
        return {s.file_name: s for s in samples}

    picks = planning_picker(group, samples, calibrated_dir)
    if picks is None:
        print(f"Subtraction planning cancelled for group '{group.name}'.")
        return {s.file_name: s for s in samples}
    if not picks:
        print(f"No intervals selected for subtraction in group '{group.name}'.")
        return {s.file_name: s for s in samples}

    by_name = {s.file_name: s for s in samples}
    for (fn, idx), info in picks.items():
        state = by_name[fn]
        result = preview_and_apply_subtraction(
            state, idx, group, subgroup_templates, info["anchor_t0"]
        )
        if isinstance(result, str) and result == "back":
            print("Aborted planning; remaining intervals left uncorrected.")
            break
        if isinstance(result, str) and result == "skip":
            continue
        corrected_sub, meta = result
        state.corrected_subsets[idx] = corrected_sub
        state.correction_meta[idx] = meta

    return by_name


# ──────────────────────────────────────────────────────────────────────────
# Optional re-fit for corrected intervals
# ──────────────────────────────────────────────────────────────────────────

def offer_refit_for_corrected_intervals(
    samples_by_name: dict[str, SampleState],
    interactive_interval_fitting,
    calculate_turnover_before_inactivation,
) -> None:
    """For each corrected interval, ask the user whether to re-fit + re-turnover.

    Mutates ``samples_by_name[*].refit_results`` and ``refit_turnover``
    in place.
    """
    for state in samples_by_name.values():
        if not state.corrected_subsets:
            continue
        # Build a one-shot pseudo-state with just the corrected intervals so
        # the existing fitting GUI can run unchanged.
        corrected_list = list(state.corrected_subsets.values())
        if not _prompt_refit(state.file_name, [s.index for s in corrected_list]):
            continue
        fit_results = interactive_interval_fitting(
            corrected_list,
            state.time_col,
            CALIBRATED_COLUMN,
            filename=f"{state.file_name} (control-subtracted)",
        )
        if isinstance(fit_results, str):
            continue  # user backed out
        state.refit_results = fit_results
        # Turnover
        turnover = calculate_turnover_before_inactivation(
            corrected_list,
            fit_results,
            state.time_col,
            CALIBRATED_COLUMN,
            filename=f"{state.file_name} (control-subtracted)",
        )
        if isinstance(turnover, str):
            state.refit_turnover = {}
        else:
            state.refit_turnover = turnover


def _prompt_refit(file_name: str, idxs: list[int]) -> bool:
    fig, ax = plt.subplots(figsize=(7, 2.7))
    ax.axis("off")
    ax.text(
        0.5, 0.6,
        f"Re-fit corrected intervals for {truncate_filename(file_name)}?\n\n"
        f"Intervals corrected: {', '.join(f'#{i}' for i in idxs)}\n\n"
        "Yes → run the fitting + max-H₂O₂ screens on the subtracted data.\n"
        "No  → keep the original pre-subtraction fits.",
        ha="center", va="center", fontsize=10,
    )
    fig.suptitle("Re-fit on corrected intervals?", fontsize=11, fontweight="bold")
    state = {"choice": False}

    def on_yes(_e=None):
        state["choice"] = True
        plt.close(fig)

    def on_no(_e=None):
        state["choice"] = False
        plt.close(fig)

    ax_yes = fig.add_axes([0.20, 0.10, 0.25, 0.16])
    ax_no = fig.add_axes([0.55, 0.10, 0.25, 0.16])
    btn_yes = create_small_button(ax_yes, "Yes — re-fit", "#90ee90", "#7cd47c")
    btn_no = create_small_button(ax_no, "No — keep originals", "0.9", "0.8")
    btn_yes.on_clicked(on_yes)
    btn_no.on_clicked(on_no)
    plt.show()
    plt.close(fig)
    return state["choice"]
