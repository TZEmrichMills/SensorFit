"""Multi-addition residual activity: rate-ratio computation between intervals
that the user tagged as a sequential H2O2-addition series in one run.

Use case: the user runs a single enzyme reaction with multiple H2O2 additions
(addition 1 → consume → addition 2 → consume → …).  Each addition gets its own
interval, and the user tags them as belonging to one series.  After fitting,
the ratio ``rate_n / rate_1`` describes how much initial-rate activity remains
after each successive addition.

This module provides:
- ``compute_residual_activity``: pure-function that takes the per-interval fit
  results and the user-tagged series, and returns ``{interval_index: ratio}``.
- ``tag_residual_activity_series``: a small matplotlib dialog that lets the
  user check off which intervals belong to the series.  Returns the list of
  selected interval indices.
"""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
import numpy as np

from .calibration import create_small_button, truncate_filename


# Models excluded from "best-model" rate selection (matches calibration.py
# behavior so the ratio mirrors fit_summary's best_model_initial_rate_uM_per_s).
_NON_PRIMARY_MODELS = {"LinearInitialRate"}


def _best_rate(fit_results: dict[str, dict]) -> float:
    """Pick the initial rate from the same model that
    ``append_fit_summary`` uses for ``best_model_initial_rate_uM_per_s``:
    the non-LinearInitialRate fit with the lowest AIC.
    """
    primaries = {k: v for k, v in fit_results.items() if k not in _NON_PRIMARY_MODELS}
    if primaries:
        best = min(primaries.keys(), key=lambda k: primaries[k].get("aic", float("inf")))
    elif "LinearInitialRate" in fit_results:
        best = "LinearInitialRate"
    else:
        return float("nan")
    return float(fit_results[best].get("init_rate", float("nan")))


def compute_residual_activity(
    fit_results: dict[int, dict[str, dict]],
    series_indices: Iterable[int],
) -> dict[int, float]:
    """Return ``{interval_index: rate_ratio_vs_first}`` for intervals in the
    user-tagged series.

    The first interval in the series gets 1.0; each later interval gets its
    best-model initial rate divided by the first interval's best-model initial
    rate.  Intervals outside the series are not in the returned dict.

    NaN propagates: if any rate is NaN, the corresponding ratio is NaN.  If
    the first interval's rate is ~0 (within 1e-12), all ratios are NaN to avoid
    inflating noise into a giant ratio.
    """
    series = [i for i in series_indices if i in fit_results]
    if len(series) < 2:
        # Nothing meaningful to report; flag the first as 1.0 if it exists,
        # otherwise return empty.
        if len(series) == 1:
            return {series[0]: 1.0}
        return {}

    series_sorted = sorted(series)
    first = series_sorted[0]
    rate0 = _best_rate(fit_results[first])
    out: dict[int, float] = {first: 1.0}
    if not np.isfinite(rate0) or abs(rate0) < 1e-12:
        for idx in series_sorted[1:]:
            out[idx] = float("nan")
        return out
    for idx in series_sorted[1:]:
        rate = _best_rate(fit_results[idx])
        if not np.isfinite(rate):
            out[idx] = float("nan")
        else:
            out[idx] = float(rate / rate0)
    return out


def tag_residual_activity_series(
    interval_summaries: list[tuple[int, float, float]],
    filename: str | None = None,
    pre_selected: list[int] | None = None,
) -> list[int]:
    """Open a small matplotlib dialog with one checkbox per interval.

    Parameters
    ----------
    interval_summaries : list of (index, start_time_s, end_time_s)
        One tuple per defined interval.
    filename : optional, shown in the dialog title.
    pre_selected : optional, interval indices to start checked.

    Returns
    -------
    list[int] of interval indices the user checked, in numerical order.  An
    empty list means "no series" (the feature stays opt-in).
    """
    if not interval_summaries:
        return []

    fig = plt.figure(figsize=(6.5, 0.7 + 0.32 * max(3, len(interval_summaries))))
    title = "Tag intervals belonging to one residual-activity series"
    if filename:
        title = f"{truncate_filename(filename)}\n{title}"
    fig.suptitle(title, fontsize=10)

    labels = [
        f"Interval #{idx}  ({start:.1f}–{end:.1f} s)"
        for idx, start, end in interval_summaries
    ]
    initial = [
        (pre_selected is not None) and (idx in pre_selected)
        for idx, _, _ in interval_summaries
    ]

    rax = fig.add_axes([0.1, 0.20, 0.8, 0.65])
    check = CheckButtons(rax, labels, initial)

    decision = {"action": None}

    def on_confirm(_event=None) -> None:
        decision["action"] = "confirm"
        plt.close(fig)

    def on_cancel(_event=None) -> None:
        decision["action"] = "cancel"
        plt.close(fig)

    ax_confirm = fig.add_axes([0.20, 0.04, 0.25, 0.08])
    ax_cancel = fig.add_axes([0.55, 0.04, 0.25, 0.08])
    btn_confirm = create_small_button(ax_confirm, "Save selection", "#90ee90", "#7cd47c")
    btn_cancel = create_small_button(ax_cancel, "Cancel (no series)", "0.9", "0.8")
    btn_confirm.on_clicked(on_confirm)
    btn_cancel.on_clicked(on_cancel)

    plt.show()
    plt.close(fig)

    if decision["action"] != "confirm":
        return []

    selected = [
        interval_summaries[i][0]
        for i, state in enumerate(check.get_status())
        if state
    ]
    return sorted(selected)
