"""Calibration module for converting current measurements to H2O2 concentrations."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence
import textwrap

# Set matplotlib backend for cross-platform compatibility
import matplotlib
# Use TkAgg on Windows, default on Mac/Linux
if sys.platform == "win32":
    try:
        matplotlib.use("TkAgg")
    except ImportError:
        # Fallback to default if TkAgg not available
        pass

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons, TextBox
import numpy as np
import pandas as pd


@dataclass
class CalibrationResult:
    """Result of calibration fit."""
    slope: float
    intercept: float
    mean_currents: list[float]
    concentrations: list[float]


@dataclass
class IntervalSubset:
    """Subset of data for a selected interval."""
    index: int
    start_time: float
    end_time: float
    data: pd.DataFrame


SUPPORTED_EXCEL_EXTS = {".xls", ".xlsx", ".xlsm", ".xlsb"}
SUPPORTED_TEXT_EXTS = {".txt", ".csv"}
SUPPORTED_FILE_EXTS = SUPPORTED_EXCEL_EXTS | SUPPORTED_TEXT_EXTS
CALIBRATED_COLUMN = "H2O2_uM"

# Mapping from internal model names to display names for outputs
MODEL_DISPLAY_NAMES = {
    "IB": "Inactivation",
    "Exponential": "Exponential",
    "GFI": "Gompertz-like",
    "LinearInitialRate": "LinearInitialRate",
}

def get_model_display_name(model_name: str) -> str:
    """Get display name for a model, or return the original if not found."""
    return MODEL_DISPLAY_NAMES.get(model_name, model_name)


def create_small_button(ax, label, color, hovercolor):
    """Helper function to create buttons with smaller size and font."""
    btn = Button(ax, label, color=color, hovercolor=hovercolor)
    # Set smaller font size
    for text in ax.texts:
        text.set_fontsize(9)
    return btn


def add_instruction_banner(
    fig,
    text: str,
    *,
    y: float = 0.97,
    fontsize: int = 9,
    width: int = 95,
):
    """Add a reusable instruction banner along the top of the figure."""
    wrapped = textwrap.fill(text, width=width)
    fig.text(
        0.5,
        y,
        wrapped,
        ha="center",
        va="top",
        fontsize=fontsize,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.9, pad=0.4),
    )


def load_trace(path: Path, time_col_idx: int = 0, current_col_idx: int = 1) -> pd.DataFrame:
    """
    Load trace from Excel or text file.
    
    Supports:
    - Excel files (.xls, .xlsx, .xlsm, .xlsb)
    - Text files (.txt, .csv) with semicolon or comma delimiters
    - European format numbers (comma as decimal separator)
    
    Parameters:
    -----------
    path : Path
        Path to Excel or text file
    time_col_idx : int
        Column index for time (0-indexed, default 0 = 1st column)
    current_col_idx : int
        Column index for current (0-indexed, default 1 = 2nd column)
    
    Returns:
    --------
    DataFrame with time and current columns
    """
    if path.suffix.lower() not in SUPPORTED_FILE_EXTS:
        raise ValueError(f"Unsupported file extension for {path}. Supported: {SUPPORTED_FILE_EXTS}")
    
    # Load based on file type
    if path.suffix.lower() in SUPPORTED_EXCEL_EXTS:
        frame = pd.read_excel(path)
    else:
        # Text file - try to detect delimiter and handle European number format
        # First, read a sample to detect delimiter
        with open(path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            # Count delimiters to determine which is the column separator
            # Semicolon is common in European CSV files
            semicolon_count = first_line.count(';')
            comma_count = first_line.count(',')
            tab_count = first_line.count('\t')
            
            # Choose delimiter based on count (prefer semicolon for European files)
            if semicolon_count > 0 and semicolon_count >= comma_count:
                delimiter = ';'
                # If semicolon is delimiter, comma is likely decimal separator
                decimal_sep = ','
            elif tab_count > 0:
                delimiter = '\t'
                decimal_sep = '.'  # Tab-delimited usually uses dot
            elif comma_count > 0:
                delimiter = ','
                decimal_sep = '.'  # Comma-delimited usually uses dot
            else:
                delimiter = '\t'  # Default to tab
                decimal_sep = '.'
        
        # Read the file with detected settings
        try:
            frame = pd.read_csv(
                path,
                delimiter=delimiter,
                encoding='utf-8',
                decimal=decimal_sep,
                na_values=['', 'NA', 'N/A', 'nan', 'NaN'],
            )
        except Exception:
            # If that failed, try with alternative decimal separator
            alt_decimal = '.' if decimal_sep == ',' else ','
            frame = pd.read_csv(
                path,
                delimiter=delimiter,
                encoding='utf-8',
                decimal=alt_decimal,
                na_values=['', 'NA', 'N/A', 'nan', 'NaN'],
            )
    
    # Use column indices if provided, otherwise try to find by name
    if time_col_idx is not None and time_col_idx < len(frame.columns):
        time_col = frame.columns[time_col_idx]
    else:
        raise ValueError(f"Time column index {time_col_idx} out of range (file has {len(frame.columns)} columns)")
    
    if current_col_idx is not None and current_col_idx < len(frame.columns):
        current_col = frame.columns[current_col_idx]
    else:
        raise ValueError(f"Current column index {current_col_idx} out of range (file has {len(frame.columns)} columns)")
    
    # Extract the two columns and convert to numeric, handling European format if needed
    result = frame[[time_col, current_col]].copy()
    
    # Convert to numeric, replacing comma with dot if needed
    for col in [time_col, current_col]:
        if result[col].dtype == 'object':
            # Try to convert, handling both comma and dot decimal separators
            result[col] = result[col].astype(str).str.replace(',', '.', regex=False)
        result[col] = pd.to_numeric(result[col], errors='coerce')
    
    return result.dropna().reset_index(drop=True)


def truncate_filename(filename: str, max_length: int = 15) -> str:
    """
    Truncate filename if too long: first N chars + "..." + last N chars.
    
    Parameters:
    -----------
    filename : str
        Full filename to truncate
    max_length : int
        Maximum length for prefix and suffix (default 15)
    
    Returns:
    --------
    Truncated filename if longer than 2*max_length + 3, otherwise original
    """
    if len(filename) <= 2 * max_length + 3:
        return filename
    return filename[:max_length] + "..." + filename[-max_length:]


def average_window(signal_values: np.ndarray, center_idx: int, window: int) -> float:
    """Average signal values around a center index."""
    if window <= 0:
        raise ValueError("window must be positive")
    half = max(window // 2, 1)
    start = max(center_idx - half, 0)
    end = min(center_idx + half + 1, signal_values.size)
    return float(signal_values[start:end].mean())


def select_baseline(
    time_values: np.ndarray,
    signal_values: np.ndarray,
    window: int = 50,
    filename: str | None = None,
) -> tuple[np.ndarray, tuple[float, float]] | None | str:
    """
    Select baseline by clicking two points, show corrected data, and get confirmation.
    
    Returns:
    --------
    Tuple of (baseline_corrected_signal, (slope, intercept)) if accepted
    None if dataset is discarded
    "redraw" if user wants to retry baseline selection
    """
    # Step 1: Select baseline points
    while True:
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.subplots_adjust(left=0.1, bottom=0.18, right=0.98, top=0.80)
        ax.plot(time_values, signal_values, "b-", lw=1, label="Raw data")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Current (A)")
        title = "Click two points to define baseline (average of 20 points around each click)"
        if filename:
            display_name = truncate_filename(filename)
            title = f"{display_name}\n{title}"
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add instructions text box
        instructions_text = (
            "Use the toolbar zoom/pan tools to explore the data. Deselect those tools, then click "
            "two points to define the baseline. Each click averages 20 surrounding points."
        )
        add_instruction_banner(fig, instructions_text)

        baseline_points: list[tuple[float, float]] = []
        baseline_line = None
        state = {"action": None}

        def update_baseline_line() -> None:
            nonlocal baseline_line
            if len(baseline_points) == 2:
                if baseline_line is not None:
                    baseline_line.remove()
                t1, y1 = baseline_points[0]
                t2, y2 = baseline_points[1]
                baseline_line, = ax.plot([t1, t2], [y1, y2], "r--", lw=2, label="Baseline")
                ax.legend()
                fig.canvas.draw_idle()

        def on_click(event) -> None:
            # Only register clicks when:
            # 1. Left mouse button
            # 2. Click is inside the axes
            # 3. Navigation toolbar is not in zoom/pan mode
            if event.button != 1 or event.inaxes != ax:
                return
            
            # Check if navigation toolbar is active (zoom/pan mode)
            # When zoom/pan tools are active, clicks are consumed by those tools
            toolbar = fig.canvas.toolbar
            if toolbar is not None:
                # Check various ways the toolbar might indicate an active tool
                mode = getattr(toolbar, 'mode', '')
                # Some backends use '_active' attribute
                is_active = getattr(toolbar, '_active', None)
                
                # If toolbar mode indicates zoom/pan is active, ignore click
                if mode in ('zoom rect', 'pan/zoom', 'zoom', 'pan'):
                    return
                
                # If there's an active tool (like zoom), ignore click
                # The toolbar's _active attribute might be a string like 'ZOOM' or 'PAN'
                if is_active and is_active not in ('', None):
                    # Check if it's a navigation tool (not just None/empty)
                    active_str = str(is_active).upper()
                    if 'ZOOM' in active_str or 'PAN' in active_str:
                        return
            
            # Register the click as a baseline point
            idx = int(np.abs(time_values - event.xdata).argmin())
            avg_current = average_window(signal_values, idx, window)
            avg_time = time_values[idx]
            
            baseline_points.append((float(avg_time), float(avg_current)))
            ax.plot(avg_time, avg_current, "ro", ms=8, zorder=5)
            fig.canvas.draw_idle()
            
            if len(baseline_points) == 2:
                update_baseline_line()
                print(f"\nBaseline points selected:")
                print(f"  Point 1: t={baseline_points[0][0]:.3f} s, I={baseline_points[0][1]:.6e} A")
                print(f"  Point 2: t={baseline_points[1][0]:.3f} s, I={baseline_points[1][1]:.6e} A")
                print("Click 'Continue' to see baseline-corrected data, 'Redraw' to select again, or 'Discard' to skip this file.")

        def on_continue(_event) -> None:
            if len(baseline_points) == 2:
                state["action"] = "continue"
                plt.close(fig)
            else:
                print("Please select two points first.")

        def on_redraw(_event) -> None:
            state["action"] = "redraw"
            plt.close(fig)

        def on_discard(_event) -> None:
            state["action"] = "discard"
            plt.close(fig)

        def on_skip_baseline(_event) -> None:
            """
            Skip baseline selection and use the raw signal as-is.
            
            This allows users to bypass baseline correction when the
            baseline already looks acceptable.
            """
            state["action"] = "skip"
            plt.close(fig)

        fig.canvas.mpl_connect("button_press_event", on_click)

        ax_continue = fig.add_axes([0.25, 0.02, 0.12, 0.04])
        ax_redraw = fig.add_axes([0.39, 0.02, 0.12, 0.04])
        ax_skip = fig.add_axes([0.53, 0.02, 0.12, 0.04])
        ax_discard = fig.add_axes([0.67, 0.02, 0.18, 0.04])
        btn_continue = create_small_button(ax_continue, "Continue", "#90ee90", "#7cd47c")
        btn_redraw = create_small_button(ax_redraw, "Redraw", "#ffcc99", "#ffaa66")
        btn_skip = create_small_button(ax_skip, "Skip baseline", "#d0d0ff", "#a8a8ff")
        btn_discard = create_small_button(ax_discard, "Discard this file", "#ff9999", "#ff6666")
        btn_continue.on_clicked(on_continue)
        btn_redraw.on_clicked(on_redraw)
        btn_skip.on_clicked(on_skip_baseline)
        btn_discard.on_clicked(on_discard)

        print(
            "\nBaseline selection:\n"
            "  • Use the toolbar zoom/pan buttons to explore the data.\n"
            "  • When ready to select points, deselect zoom/pan tools (click the tool again to deactivate).\n"
            "  • Then left-click on the plot to select baseline points.\n"
            "  • Each click averages 20 surrounding points.\n"
            "  • A red dashed line will appear connecting the two points.\n"
            "  • Use buttons: Continue (view corrected data), Redraw (select again), "
            "Skip baseline (use raw signal as-is), or Discard (skip this file)."
        )

        plt.show()
        plt.close(fig)

        action = state["action"]
        if action == "discard":
            return None
        if action == "redraw":
            continue
        if action == "skip":
            print(
                "\nBaseline selection skipped by user. "
                "Proceeding with raw signal as baseline-corrected data."
            )
            corrected_signal = signal_values.copy()
            slope = 0.0
            intercept = 0.0
            return corrected_signal, (slope, intercept)
        if action == "continue" and len(baseline_points) == 2:
            t1, y1 = baseline_points[0]
            t2, y2 = baseline_points[1]
            if abs(t2 - t1) < 1e-9:
                slope = 0.0
                intercept = y1
            else:
                slope = (y2 - y1) / (t2 - t1)
                intercept = y1 - slope * t1
            
            baseline_values = slope * time_values + intercept
            corrected_signal = signal_values - baseline_values
            
            print(f"\nBaseline calculated: slope={slope:.6e} A/s, intercept={intercept:.6e} A")
            print("Baseline correction applied (drift removed).")
            break
    
    # Step 2: Show corrected data and get confirmation
    while True:
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.subplots_adjust(left=0.1, bottom=0.18, right=0.98, top=0.80)
        ax.plot(time_values, signal_values, "b-", lw=1, alpha=0.5, label="Raw data")
        ax.plot(time_values, corrected_signal, "g-", lw=1.5, label="Baseline-corrected data")
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5, label="Zero reference")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Current (A)")
        title = "Baseline-corrected data (drift removed)\nReview and choose: Accept, Retry, or Discard"
        if filename:
            display_name = truncate_filename(filename)
            title = f"{display_name}\n{title}"
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add instructions text box
        instructions_text = (
            "Review the baseline-corrected data (green). Click Accept to continue, Retry to "
            "select baseline again, or Discard this file to skip it."
        )
        add_instruction_banner(fig, instructions_text)

        confirm_state = {"action": None}

        def on_accept(_event) -> None:
            confirm_state["action"] = "accept"
            plt.close(fig)

        def on_retry(_event) -> None:
            confirm_state["action"] = "retry"
            plt.close(fig)

        def on_discard_confirm(_event) -> None:
            confirm_state["action"] = "discard"
            plt.close(fig)

        ax_accept = fig.add_axes([0.35, 0.02, 0.12, 0.04])
        ax_retry = fig.add_axes([0.48, 0.02, 0.12, 0.04])
        ax_discard_confirm = fig.add_axes([0.61, 0.02, 0.18, 0.04])
        btn_accept = create_small_button(ax_accept, "Accept", "#90ee90", "#7cd47c")
        btn_retry = create_small_button(ax_retry, "Retry", "#ffcc99", "#ffaa66")
        btn_discard_confirm = create_small_button(ax_discard_confirm, "Discard this file", "#ff9999", "#ff6666")
        btn_accept.on_clicked(on_accept)
        btn_retry.on_clicked(on_retry)
        btn_discard_confirm.on_clicked(on_discard_confirm)

        print(
            "\nBaseline-corrected data review:\n"
            "  • Green line shows the data after baseline correction (drift removed).\n"
            "  • Blue line (faded) shows the original raw data for reference.\n"
            "  • Use buttons: Accept (continue with corrected data), Retry (select baseline again), Discard (skip this file)."
        )

        plt.show()
        plt.close(fig)

        confirm_action = confirm_state["action"]
        if confirm_action == "discard":
            return None
        if confirm_action == "retry":
            return "redraw"
        if confirm_action == "accept":
            print(f"Baseline correction accepted. Proceeding with corrected data.")
            return corrected_signal, (slope, intercept)
        
        return None


def select_points(
    time_values: np.ndarray,
    signal_values: np.ndarray,
    num_points: int,
    current_calibration_values: list[float],
    update_calibration_callback: Callable[[list[float], bool], None] | None = None,
    filename: str | None = None,
) -> tuple[list[int], list[float]] | str:
    """Select calibration points interactively."""
    fig, ax = plt.subplots(figsize=(14, 7))
    plt.subplots_adjust(left=0.1, bottom=0.18, right=0.95, top=0.80)
    line, = ax.plot(time_values, signal_values, lw=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Current (A)")
    
    selected_indices: list[int] = []
    selected_markers: list = []
    
    # Use a mutable container for num_points so it can be updated
    num_points_ref = {"value": num_points}
    
    state = {
        "calibration_values": current_calibration_values.copy(),
        "selection_enabled": True,
        "action": None,
        "go_back_to_baseline": False,
        "discard": False,
    }

    def update_title() -> None:
        current_num_points = num_points_ref["value"]
        remaining = current_num_points - len(selected_indices)
        if remaining > 0:
            title = f"Select {remaining} more calibration point(s). Use zoom/pan tools, then click when ready."
        else:
            title = f"All {current_num_points} points selected. Use buttons below to Continue or Retry."
        if filename:
            display_name = truncate_filename(filename)
            title = f"{display_name}\n{title}"
        ax.set_title(title)
        fig.canvas.draw_idle()
    
    def on_button_press(event) -> None:
        if event.button != 1 or event.inaxes != ax:
            return
        
        current_num_points = num_points_ref["value"]
        if len(selected_indices) >= current_num_points:
            return
        
        toolbar = fig.canvas.toolbar
        if toolbar is not None:
            mode = getattr(toolbar, 'mode', '')
            is_active = getattr(toolbar, '_active', None)
            if mode in ('zoom rect', 'pan/zoom', 'zoom', 'pan'):
                return
            if is_active and is_active not in ('', None):
                active_str = str(is_active).upper()
                if 'ZOOM' in active_str or 'PAN' in active_str:
                    return
        
        idx = int(np.abs(time_values - event.xdata).argmin())
        selected_indices.append(idx)
        
        marker, = ax.plot(
            time_values[idx],
            signal_values[idx],
            "ro",
            ms=10,
            markeredgecolor="yellow",
            markeredgewidth=2,
            zorder=5,
        )
        selected_markers.append(marker)
        fig.canvas.draw_idle()
        
        update_title()
        
        current_num_points = num_points_ref["value"]
        if len(selected_indices) == current_num_points:
            print(f"\nAll {current_num_points} calibration points selected.")
            print("Click 'Continue' to proceed or 'Retry' to select again.")

    def on_continue(_event) -> None:
        current_num_points = num_points_ref["value"]
        if len(selected_indices) == current_num_points:
            # Save final values
            state["action"] = "continue"
            plt.close(fig)
        else:
            print(f"Please select all {current_num_points} points before continuing.")

    def on_retry(_event) -> None:
        selected_indices.clear()
        while selected_markers:
            marker = selected_markers.pop()
            marker.remove()
        state["action"] = None
        fig.canvas.draw_idle()
        update_title()
        print("Selection cleared. Please select points again.")

    def on_go_back_to_baseline(_event) -> None:
        state["go_back_to_baseline"] = True
        state["action"] = "go_back"
        plt.close(fig)

    def on_discard(_event) -> None:
        state["discard"] = True
        state["action"] = "discard"
        plt.close(fig)

    def on_change_calibration(_event) -> None:
        """Open calibration values editor."""
        from .calibration_editor import edit_calibration_values
        
        result = edit_calibration_values(
            current_values=state["calibration_values"],
            num_points_ref=num_points_ref,
            selected_indices=selected_indices,
            selected_markers=selected_markers,
            update_calibration_callback=update_calibration_callback,
            filename=filename,
        )
        
        if result is not None:
            new_values, apply_to_all = result
            state["calibration_values"] = new_values
            # If count changed, update_title will reflect it
            update_title()
            fig.canvas.draw_idle()
    
    fig.canvas.mpl_connect("button_press_event", on_button_press)

    # Add instructions text box
    instructions_text = (
        f"Click on the plot to select {num_points} calibration points. Use zoom/pan tools to "
        "explore data (deselect them before clicking). Selected points appear as red circles. "
        "Use 'Change calibration values' to edit values or change the number of points."
    )
    add_instruction_banner(fig, instructions_text)
    
    # Create buttons - adjust layout to fit "Change calibration values" button
    ax_change_cal = plt.axes([0.05, 0.11, 0.18, 0.04])
    ax_retry = plt.axes([0.25, 0.02, 0.10, 0.04])
    ax_goback = plt.axes([0.36, 0.02, 0.12, 0.04])
    ax_discard = plt.axes([0.49, 0.02, 0.20, 0.04])
    ax_continue = plt.axes([0.70, 0.02, 0.10, 0.04])
    btn_change_cal = create_small_button(ax_change_cal, "Change calibration values", "#4CAF50", "#45a049")
    btn_retry = create_small_button(ax_retry, "Retry", "#ffcc99", "#ffaa66")
    btn_goback = create_small_button(ax_goback, "Go Back", "#ff9999", "#ff6666")
    btn_discard = create_small_button(ax_discard, "Discard this file", "#ff6666", "#ff4444")
    btn_continue = create_small_button(ax_continue, "Continue", "#90ee90", "#7cd47c")
    
    btn_change_cal.on_clicked(on_change_calibration)
    btn_retry.on_clicked(on_retry)
    btn_goback.on_clicked(on_go_back_to_baseline)
    btn_discard.on_clicked(on_discard)
    btn_continue.on_clicked(on_continue)

    update_title()
    
    print(
        "\nCalibration point selection:\n"
        "  • Use the toolbar zoom/pan buttons to explore the data.\n"
        "  • When ready to select points, deselect zoom/pan tools.\n"
        "  • Then left-click on the plot to select calibration points.\n"
        "  • Selected points will be marked with red circles.\n"
        f"  • Select {num_points} points total, then click 'Continue' or 'Retry'.\n"
        "  • Use 'Change calibration values' to edit calibration values or change the number of points.\n"
        "  • Use 'Go Back' to return to baseline selection.\n"
        "  • Use 'Discard' to skip this file entirely."
    )

    plt.show()
    plt.close(fig)
    
    if state["discard"]:
        return "discard"
    
    if state["go_back_to_baseline"]:
        return "go_back_to_baseline"
    
    final_num_points = num_points_ref["value"]
    if state["action"] != "continue" or len(selected_indices) != final_num_points:
        raise RuntimeError(f"Selection incomplete or cancelled: {len(selected_indices)}/{final_num_points} points selected.")
    
    return selected_indices, state["calibration_values"]


def build_calibration(mean_currents: Sequence[float], concentrations: Sequence[float]) -> CalibrationResult:
    """Build calibration from current values and concentrations."""
    if len(mean_currents) != len(concentrations):
        raise ValueError("Calibration inputs must have matching lengths")
    coeffs = np.polyfit(mean_currents, concentrations, 1)
    slope, intercept = coeffs
    return CalibrationResult(
        slope=slope,
        intercept=intercept,
        mean_currents=list(mean_currents),
        concentrations=list(concentrations),
    )


def apply_calibration(signal_values: pd.Series, calibration: CalibrationResult) -> pd.Series:
    """Apply calibration to convert current to H2O2 concentration."""
    return signal_values * calibration.slope + calibration.intercept


def select_intervals(
    time_values: np.ndarray, 
    calibrated_values: np.ndarray,
    filename: str | None = None,
) -> list[tuple[float, float]] | None | str:
    """
    Select intervals from calibrated data using two-click method.
    
    Returns:
    --------
    list of (start, end) tuples if intervals are selected and confirmed
    None if user wants to go back to calibration point selection
    [] if window is closed without confirming
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(left=0.1, bottom=0.18, right=0.98, top=0.80)
    ax.plot(time_values, calibrated_values, color="tab:green", lw=1.25)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("H2O2 (µM)")
    
    intervals: list[tuple[float, float]] = []
    span_patches: list = []
    boundary_lines: list = []
    start_marker = None
    pending_start: float | None = None
    accepted = {"confirmed": False}
    go_back = {"requested": False}

    def update_title() -> None:
        if pending_start is not None:
            title = f"Interval selection: Start point selected at t={pending_start:.3f} s. Click to select end point."
        elif len(intervals) > 0:
            title = f"Interval selection: {len(intervals)} interval(s) selected. Click to start a new interval, or use buttons below."
        else:
            title = "Interval selection: Click to select start point of first interval."
        if filename:
            display_name = truncate_filename(filename)
            title = f"{display_name}\n{title}"
        ax.set_title(title)
        fig.canvas.draw_idle()

    def add_interval(start: float, end: float) -> None:
        start, end = sorted((float(start), float(end)))
        if abs(end - start) < 1e-6:
            print("Interval too short, ignoring.")
            return
        intervals.append((start, end))
        patch = ax.axvspan(start, end, color="orange", alpha=0.25, zorder=1)
        span_patches.append(patch)
        line_start = ax.axvline(start, color="orange", linestyle="--", alpha=0.7, linewidth=1.5, zorder=2)
        line_end = ax.axvline(end, color="orange", linestyle="--", alpha=0.7, linewidth=1.5, zorder=2)
        boundary_lines.append(line_start)
        boundary_lines.append(line_end)
        fig.canvas.draw_idle()
        print(f"Interval #{len(intervals)}: {start:.3f}–{end:.3f} s ({end-start:.3f} s duration)")

    def on_button_press(event) -> None:
        nonlocal pending_start, start_marker
        
        if event.button != 1 or event.inaxes != ax:
            return
        
        toolbar = fig.canvas.toolbar
        if toolbar is not None:
            mode = getattr(toolbar, 'mode', '')
            is_active = getattr(toolbar, '_active', None)
            if mode in ('zoom rect', 'pan/zoom', 'zoom', 'pan'):
                return
            if is_active and is_active not in ('', None):
                active_str = str(is_active).upper()
                if 'ZOOM' in active_str or 'PAN' in active_str:
                    return
        
        click_time = float(event.xdata)
        
        if pending_start is None:
            pending_start = click_time
            if start_marker is not None:
                start_marker.remove()
            start_marker, = ax.plot(
                pending_start,
                calibrated_values[int(np.abs(time_values - pending_start).argmin())],
                "ro",
                ms=10,
                markeredgecolor="yellow",
                markeredgewidth=2,
                zorder=5,
                label="Start point"
            )
            ax.legend(loc="upper right")
            fig.canvas.draw_idle()
            update_title()
            print(f"Start point selected: t={pending_start:.3f} s")
        else:
            end_time = click_time
            add_interval(pending_start, end_time)
            if start_marker is not None:
                start_marker.remove()
                start_marker = None
            pending_start = None
            update_title()

    def clear_intervals(_event=None) -> None:
        nonlocal pending_start, start_marker
        intervals.clear()
        pending_start = None
        while span_patches:
            patch = span_patches.pop()
            patch.remove()
        while boundary_lines:
            line = boundary_lines.pop()
            line.remove()
        if start_marker is not None:
            start_marker.remove()
            start_marker = None
        fig.canvas.draw_idle()
        update_title()
        print("All intervals cleared.")

    def remove_last_interval(_event=None) -> None:
        nonlocal pending_start, start_marker
        if intervals:
            intervals.pop()
            if span_patches:
                patch = span_patches.pop()
                patch.remove()
            if len(boundary_lines) >= 2:
                boundary_lines.pop().remove()
                boundary_lines.pop().remove()
            fig.canvas.draw_idle()
            update_title()
            print(f"Last interval removed. {len(intervals)} interval(s) remaining.")
        else:
            print("No intervals to remove.")

    def confirm_and_close(_event=None) -> None:
        if pending_start is not None:
            print("Warning: Start point selected but no end point. Interval not added.")
        
        # Check if no intervals selected and show warning
        if len(intervals) == 0:
            # Create a warning dialog
            warning_fig, warning_ax = plt.subplots(figsize=(6, 3))
            warning_ax.axis('off')
            warning_text = (
                "WARNING: No intervals selected!\n\n"
                "You are about to continue without selecting any intervals.\n"
                "This means no interval data will be saved for fitting.\n\n"
                "Are you sure you want to continue?"
            )
            warning_ax.text(0.5, 0.5, warning_text, 
                          ha='center', va='center', 
                          fontsize=11, wrap=True,
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            warning_fig.suptitle("Confirm Action", fontsize=12, fontweight='bold')
            
            warning_state = {"confirmed": False, "cancelled": False}
            
            def on_yes(_event) -> None:
                warning_state["confirmed"] = True
                plt.close(warning_fig)
            
            def on_no(_event) -> None:
                warning_state["cancelled"] = True
                plt.close(warning_fig)
            
            ax_yes = warning_fig.add_axes([0.3, 0.1, 0.2, 0.12])
            ax_no = warning_fig.add_axes([0.5, 0.1, 0.2, 0.12])
            btn_yes = create_small_button(ax_yes, "Yes, Continue", "#ff9999", "#ff6666")
            btn_no = create_small_button(ax_no, "No, Cancel", "#90ee90", "#7cd47c")
            btn_yes.on_clicked(on_yes)
            btn_no.on_clicked(on_no)
            
            plt.show()
            plt.close(warning_fig)
            
            if warning_state["cancelled"]:
                print("Action cancelled. Please select intervals or use 'Go Back' to return to calibration.")
                return  # Don't close the main figure
            elif not warning_state["confirmed"]:
                print("Action cancelled.")
                return
        
        accepted["confirmed"] = True
        plt.close(fig)

    def go_back_to_calibration(_event=None) -> None:
        go_back["requested"] = True
        plt.close(fig)

    def on_key(event) -> None:
        if event.key == "enter":
            confirm_and_close()
        elif event.key in {"escape", "r"}:
            clear_intervals()
        elif event.key == "b":
            go_back_to_calibration()
        elif event.key == "u":
            remove_last_interval()

    fig.canvas.mpl_connect("button_press_event", on_button_press)
    fig.canvas.mpl_connect("key_press_event", on_key)

    discard_interval = {"requested": False}
    
    def on_discard_interval(_event=None) -> None:
        discard_interval["requested"] = True
        accepted["confirmed"] = True  # Mark as confirmed so we exit
        plt.close(fig)

    ax_remove_last = fig.add_axes([0.12, 0.02, 0.11, 0.04])
    ax_reselect = fig.add_axes([0.24, 0.02, 0.11, 0.04])
    ax_goback = fig.add_axes([0.36, 0.02, 0.11, 0.04])
    ax_discard = fig.add_axes([0.48, 0.02, 0.20, 0.04])
    ax_continue = fig.add_axes([0.69, 0.02, 0.11, 0.04])
    btn_remove_last = create_small_button(ax_remove_last, "Remove Last", "#ffddaa", "#ffcc88")
    btn_reselect = create_small_button(ax_reselect, "Reselect", "0.9", "0.8")
    btn_goback = create_small_button(ax_goback, "Go Back", "#ffcc99", "#ffaa66")
    btn_discard = create_small_button(ax_discard, "Discard this file", "#ff6666", "#ff4444")
    btn_continue = create_small_button(ax_continue, "Continue", "#90ee90", "#7cd47c")
    btn_remove_last.on_clicked(remove_last_interval)
    btn_reselect.on_clicked(clear_intervals)
    btn_goback.on_clicked(go_back_to_calibration)
    btn_discard.on_clicked(on_discard_interval)
    btn_continue.on_clicked(confirm_and_close)

    # Add instructions text box
    instructions_text = (
        "Click once to set the START of an interval and click again to set the END. Intervals "
        "are shown as orange spans. Use the buttons below to manage intervals."
    )
    add_instruction_banner(fig, instructions_text)

    update_title()

    print(
        "\nInterval selection (two-click method):\n"
        "  • Use the toolbar zoom/pan buttons to explore the trace.\n"
        "  • Click to select the START point of an interval.\n"
        "  • Click to select the END point of the interval.\n"
        "  • The interval will be marked with an orange span.\n"
        "  • Repeat to select multiple intervals.\n"
        "  • Use buttons: 'Remove Last', 'Reselect', 'Go Back', 'Discard', 'Continue'.\n"
        "  • Keyboard shortcuts: Enter = Continue, Escape/R = Reselect, B = Go Back, U = Remove Last."
    )
    plt.show()
    plt.close(fig)
    
    if discard_interval["requested"]:
        return "discard"
    if go_back["requested"]:
        return None
    return intervals if accepted["confirmed"] else []


def build_interval_subsets(
    frame: pd.DataFrame,
    time_col: str,
    calibrated_col: str,
    intervals: Sequence[tuple[float, float]],
) -> list[IntervalSubset]:
    """Build interval subsets from selected intervals."""
    subsets: list[IntervalSubset] = []
    times = frame[time_col].to_numpy(dtype=float)
    for idx, (start, end) in enumerate(intervals, start=1):
        left, right = (start, end) if start <= end else (end, start)
        mask = (times >= left) & (times <= right)
        subset = frame.loc[mask, [time_col, calibrated_col]].copy()
        if subset.empty:
            print(f"Interval {idx} [{left:.3f}, {right:.3f}] captured 0 points; skipped.")
            continue
        subset.reset_index(drop=True, inplace=True)
        subsets.append(
            IntervalSubset(
                index=idx,
                start_time=float(left),
                end_time=float(right),
                data=subset,
            )
        )
    return subsets


def save_interval_with_fits(
    subset: IntervalSubset,
    time_col: str,
    calibrated_col: str,
    fit_results: dict[str, dict] | None,
    destination_dir: Path,
) -> Path:
    """
    Save interval data with fits to Excel file.
    
    Parameters:
    -----------
    subset : IntervalSubset
        Interval data to save
    time_col : str
        Name of time column
    calibrated_col : str
        Name of calibrated H2O2 column
    fit_results : dict[str, dict] | None
        Dictionary mapping model names to fit results (or None if no fits)
    destination_dir : Path
        Directory to save the Excel file
    
    Returns:
    --------
    Path to saved Excel file
    """
    destination_dir.mkdir(parents=True, exist_ok=True)
    
    # Create DataFrame with raw data
    df = subset.data.copy()
    
    # Add fitted data columns for each model
    if fit_results:
        for model_name, fit_result in fit_results.items():
            # Use display name for column names in Excel output
            display_name = get_model_display_name(model_name)
            df[f"H2O2_uM_fit_{display_name}"] = fit_result["yhat"]
            df[f"residual_uM_{display_name}"] = subset.data[calibrated_col].to_numpy(dtype=float) - fit_result["yhat"]
    
    # Save to Excel
    excel_path = destination_dir / f"interval_{subset.index:02d}.xlsx"
    df.to_excel(excel_path, index=False, engine="openpyxl")
    
    return excel_path


def persist_interval_subsets(subsets: Sequence[IntervalSubset], destination_dir: Path) -> Path:
    """Save interval subsets to CSV files (legacy function, kept for compatibility)."""
    destination_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, float | int | str]] = []
    for subset in subsets:
        csv_path = destination_dir / f"interval_{subset.index:02d}.csv"
        subset.data.to_csv(csv_path, index=False)
        manifest.append(
            {
                "index": subset.index,
                "start_time": subset.start_time,
                "end_time": subset.end_time,
                "points": len(subset.data),
                "file": csv_path.name,
            }
        )
    manifest_path = destination_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return destination_dir


def append_fit_summary(
    summary_path: Path,
    source_file: str,
    interval_index: int,
    start_time: float,
    end_time: float,
    fit_results: dict[str, dict] | None,
    turnover_uM: float | None = None,
    control_subtracted: bool | None = None,
    control_group: str | None = None,
    residual_activity_ratio: float | None = None,
    back_extrap: dict | None = None,
) -> None:
    """
    Append fit parameters to summary Excel file (creates if doesn't exist).
    
    Parameters:
    -----------
    summary_path : Path
        Path to summary Excel file
    source_file : str
        Name of source Excel file
    interval_index : int
        Interval index number
    start_time : float
        Interval start time
    end_time : float
        Interval end time
    fit_results : dict[str, dict] | None
        Dictionary mapping model names to fit results, or None if no fits
    """
    import pandas as pd
    import numpy as np
    from datetime import datetime
    
    # Prepare row data
    row_data = {
        "source_file": source_file,
        "interval_index": interval_index,
        "start_time_s": start_time,
        "end_time_s": end_time,
        "duration_s": end_time - start_time,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Control-subtraction provenance
    if control_subtracted is not None:
        row_data["control_subtracted"] = bool(control_subtracted)
    if control_group is not None:
        row_data["control_group"] = str(control_group)

    # Residual-activity series ratio (1.0 for the series's first interval,
    # rate_n / rate_1 for later intervals).  NaN if this interval isn't part
    # of any tagged series.
    if residual_activity_ratio is not None:
        row_data["residual_activity_ratio"] = float(residual_activity_ratio)

    # H2O2-injection-start back-extrapolation columns
    if back_extrap is not None:
        row_data["back_extrap_applied"] = True
        row_data["back_extrap_deadtime_s"] = float(back_extrap.get("deadtime_s", float("nan")))
        row_data["back_extrap_nominal_uM"] = float(back_extrap.get("nominal_uM", float("nan")))
        row_data["back_extrap_H2O2_at_true_t0_uM"] = float(back_extrap.get("back_extrap_uM", float("nan")))
        row_data["back_extrap_stretch_factor"] = float(back_extrap.get("stretch_factor", float("nan")))
        row_data["back_extrap_stretch_initial_rate_uM_per_s"] = float(
            back_extrap.get("stretch_initial_rate", float("nan"))
        )
    
    # Add fit parameters for each model
    if fit_results:
        # Find best model by AIC (skip LinearInitialRate for best model selection)
        non_linear_models = {k: v for k, v in fit_results.items() if k != "LinearInitialRate"}
        if non_linear_models:
            best_model = min(non_linear_models.keys(), key=lambda k: non_linear_models[k].get("aic", float("inf")))
        elif "LinearInitialRate" in fit_results:
            best_model = "LinearInitialRate"
        else:
            best_model = "none"
        # Use display name for best_model in output
        row_data["best_model"] = get_model_display_name(best_model)
        
        # Add best model's initial rate and turnover right after best_model
        if best_model != "none" and best_model in fit_results:
            best_fit_result = fit_results[best_model]
            row_data["best_model_initial_rate_uM_per_s"] = best_fit_result.get("init_rate", float("nan"))
        else:
            row_data["best_model_initial_rate_uM_per_s"] = float("nan")
        
        # Add turnover value (if calculated) - positioned right after best model initial rate
        if turnover_uM is not None:
            row_data["turnover_before_inactivation_uM"] = float(turnover_uM)
        else:
            row_data["turnover_before_inactivation_uM"] = float("nan")
        
        for model_name, fit_result in fit_results.items():
            # Use display name for column prefixes in Excel output
            display_name = get_model_display_name(model_name)
            prefix = f"{display_name}_"
            
            # For LinearInitialRate, only save initial rate and fit timepoints (relative to interval start)
            if model_name == "LinearInitialRate":
                row_data[f"{prefix}initial_rate_uM_per_s"] = fit_result.get("init_rate", float("nan"))
                
                # Calculate relative times (from interval start, not full run)
                fit_start_abs = fit_result.get("fit_start_time", float("nan"))
                fit_end_abs = fit_result.get("fit_end_time", float("nan"))
                fit_start_rel = fit_start_abs - start_time if not np.isnan(fit_start_abs) else float("nan")
                fit_end_rel = fit_end_abs - start_time if not np.isnan(fit_end_abs) else float("nan")
                fit_duration = fit_end_rel - fit_start_rel if not (np.isnan(fit_start_rel) or np.isnan(fit_end_rel)) else float("nan")
                
                row_data[f"{prefix}fit_start_s"] = fit_start_rel
                row_data[f"{prefix}fit_end_s"] = fit_end_rel
                row_data[f"{prefix}fit_duration_s"] = fit_duration
                row_data[f"{prefix}slope"] = fit_result.get("slope", float("nan"))
                row_data[f"{prefix}intercept"] = fit_result.get("intercept", float("nan"))
                # R2, AIC, BIC, RSS, and H0_fit_uM are not included for LinearInitialRate
            else:
                # Full metrics for other models
                row_data[f"{prefix}R2"] = fit_result.get("r2", float("nan"))
                row_data[f"{prefix}AIC"] = fit_result.get("aic", float("nan"))
                row_data[f"{prefix}BIC"] = fit_result.get("bic", float("nan"))
                row_data[f"{prefix}RSS"] = fit_result.get("rss", float("nan"))
                row_data[f"{prefix}initial_rate_uM_per_s"] = fit_result.get("init_rate", float("nan"))
                row_data[f"{prefix}H0_fit_uM"] = fit_result.get("H0_fit", float("nan"))
                
                # Add model-specific parameters
                params = fit_result.get("params", [])
                param_names = fit_result.get("names", [])
                for name, value in zip(param_names, params):
                    row_data[f"{prefix}param_{name}"] = float(value)
                
                # Add Exponential-specific metrics
                if model_name == "Exponential":
                    row_data[f"{prefix}exp_amp"] = fit_result.get("exp_amp", float("nan"))
                    row_data[f"{prefix}exp_k"] = fit_result.get("exp_k", float("nan"))
                    row_data[f"{prefix}exp_t_half"] = fit_result.get("exp_t_half", float("nan"))
                    row_data[f"{prefix}linear_slope"] = fit_result.get("linear_slope", float("nan"))
                    row_data[f"{prefix}intercept"] = fit_result.get("intercept", float("nan"))
                    row_data[f"{prefix}t0"] = fit_result.get("t0", float("nan"))
                
                # Add GFI-specific metrics
                if model_name == "GFI":
                    row_data[f"{prefix}fast_B"] = fit_result.get("fast_B", float("nan"))
                    row_data[f"{prefix}fast_k"] = fit_result.get("fast_k", float("nan"))
                    row_data[f"{prefix}init_rate_fast"] = fit_result.get("init_rate_fast", float("nan"))
                    row_data[f"{prefix}init_rate_IB"] = fit_result.get("init_rate_IB", float("nan"))
    else:
        row_data["best_model"] = "none"
        row_data["best_model_initial_rate_uM_per_s"] = float("nan")
        # Add turnover value even if no fits (if calculated)
        if turnover_uM is not None:
            row_data["turnover_before_inactivation_uM"] = float(turnover_uM)
        else:
            row_data["turnover_before_inactivation_uM"] = float("nan")
    
    # Read existing file or create new DataFrame
    if summary_path.exists():
        try:
            df = pd.read_excel(summary_path, engine="openpyxl")
        except Exception:
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()

    # Idempotent upsert: if a row for the same (source_file, interval_index)
    # already exists, drop it so the new row replaces it cleanly when a user
    # redoes a file.
    if not df.empty and "source_file" in df.columns and "interval_index" in df.columns:
        same_key = (df["source_file"] == source_file) & (
            df["interval_index"] == interval_index
        )
        if same_key.any():
            df = df.loc[~same_key].reset_index(drop=True)

    # Append new row
    new_row = pd.DataFrame([row_data])
    df = pd.concat([df, new_row], ignore_index=True)
    
    # Reorder columns to put important columns first (best_model, initial_rate, turnover)
    # Get the desired order: basic info, then best_model columns, then model-specific columns
    basic_cols = ["source_file", "interval_index", "start_time_s", "end_time_s", "duration_s", "timestamp"]
    important_cols = [
        "best_model",
        "best_model_initial_rate_uM_per_s",
        "turnover_before_inactivation_uM",
        "control_subtracted",
        "control_group",
        "residual_activity_ratio",
        "back_extrap_applied",
        "back_extrap_deadtime_s",
        "back_extrap_nominal_uM",
        "back_extrap_H2O2_at_true_t0_uM",
        "back_extrap_stretch_factor",
        "back_extrap_stretch_initial_rate_uM_per_s",
    ]
    
    # Build ordered column list
    ordered_cols = []
    # Add basic columns (if they exist)
    for col in basic_cols:
        if col in df.columns:
            ordered_cols.append(col)
    
    # Add important columns (if they exist)
    for col in important_cols:
        if col in df.columns:
            ordered_cols.append(col)
    
    # Add remaining columns (model-specific parameters)
    remaining_cols = [col for col in df.columns if col not in ordered_cols]
    ordered_cols.extend(sorted(remaining_cols))  # Sort remaining for consistency
    
    # Reorder DataFrame
    df = df[ordered_cols]
    
    # Write back to file
    df.to_excel(summary_path, index=False, engine="openpyxl")


def turnover_tailfit(tt: np.ndarray, yhat: np.ndarray, H0_val: float, window_frac: float = 0.25) -> float:
    """
    Tail-line method: fit a line to the last window_frac of the fitted curve 
    and subtract its extrapolated value at the start of the interval from H(0).
    
    Parameters:
    -----------
    tt : np.ndarray
        Time array
    yhat : np.ndarray
        Fitted values array
    H0_val : float
        Initial H2O2 value (H(0))
    window_frac : float
        Fraction of data to use for tail fit (default 0.25 = last 25%)
    
    Returns:
    --------
    float
        Turnover value (H0 - tail_value_at_start), or NaN if calculation fails
    """
    n = len(tt)
    if n < 3:
        return float("nan")
    w = max(3, int(n * window_frac))
    t_tail = tt[-w:]
    y_tail = yhat[-w:]
    # simple least-squares line fit
    p = np.polyfit(t_tail, y_tail, 1)  # slope m, intercept b
    m, b = p[0], p[1]
    # Calculate the y-value of the tail fit line at the start time of the interval
    y_at_start = m * tt[0] + b
    return float(max(0.0, H0_val - y_at_start))


def fit_linear_initial_rate(
    time_values: np.ndarray,
    signal_values: np.ndarray,
    start_idx: int,
    end_idx: int,
) -> dict:
    """
    Fit a straight line to a subset of data points.
    
    Parameters:
    -----------
    time_values : np.ndarray
        Full time array
    signal_values : np.ndarray
        Full signal array
    start_idx : int
        Start index for linear fit
    end_idx : int
        End index for linear fit
    
    Returns:
    --------
    dict with fit results including initial_rate
    """
    t_fit = time_values[start_idx:end_idx+1]
    y_fit = signal_values[start_idx:end_idx+1]
    
    # Fit line: y = slope * t + intercept
    coeffs = np.polyfit(t_fit, y_fit, 1)
    slope = float(coeffs[0])
    intercept = float(coeffs[1])
    
    # Generate fitted values for full time range
    yhat = slope * time_values + intercept
    
    # Calculate metrics
    residuals = signal_values - yhat
    rss = float(np.sum(residuals ** 2))
    n = len(time_values)
    
    # R² for the full dataset
    from .utils import r2_score
    r2 = r2_score(signal_values, yhat)
    
    # Initial rate is the slope (negative because H2O2 is consumed)
    initial_rate = -slope  # Negative because consumption means decreasing
    
    return {
        "model": "LinearInitialRate",
        "params": [intercept, slope],
        "names": ["intercept", "slope"],
        "yhat": yhat,
        "rss": rss,
        "r2": r2,
        "aic": float("nan"),  # Not applicable for linear fit
        "bic": float("nan"),  # Not applicable for linear fit
        "init_rate": initial_rate,
        "H0_fit": float(intercept),  # Value at t=0
        "slope": slope,
        "intercept": intercept,
        "fit_start_idx": start_idx,
        "fit_end_idx": end_idx,
        "fit_start_time": float(time_values[start_idx]),
        "fit_end_time": float(time_values[end_idx]),
    }


def interactive_interval_fitting(
    subsets: Sequence[IntervalSubset],
    time_col: str,
    calibrated_col: str,
    filename: str | None = None,
) -> dict[int, dict[str, dict]] | str:
    """
    Interactive fitting interface for intervals.
    
    Allows user to fit one or more models (Inactivation, Exponential, Gompertz-like) to each interval
    and visually compare the fits.
    
    Note: Internal model names (IB, GFI) are mapped to display names (Inactivation, Gompertz-like) in outputs.
    
    Parameters:
    -----------
    subsets : Sequence[IntervalSubset]
        List of interval subsets to fit
    time_col : str
        Name of time column
    calibrated_col : str
        Name of calibrated H2O2 column
    
    Returns:
    --------
    dict mapping interval index to dict of model results, OR
    "go_back_phase" if user wants to return to interval selection, OR
    "discard_file" if user wants to discard the entire file.
    """
    from .fitting import fit_IB, fit_Exponential, fit_GFI
    from .models import MODEL_FUNCS
    
    if not subsets:
        return {}
    
    # Store fit results for all intervals
    all_fit_results: dict[int, dict[str, dict]] = {}
    
    # Process each interval with navigation support
    current_idx = 0
    while current_idx < len(subsets):
        subset = subsets[current_idx]
        time_values = subset.data[time_col].to_numpy(dtype=float)
        signal_values = subset.data[calibrated_col].to_numpy(dtype=float)
        
        # State for this interval — restore previous fits if navigating back
        if subset.index in all_fit_results:
            interval_fits: dict[str, dict] = dict(all_fit_results[subset.index])
        else:
            interval_fits: dict[str, dict] = {}
        selected_models = {"IB": False, "Exponential": False, "GFI": False, "LinearInitialRate": False}
        # Pre-check models that already have fits (so they show on the plot)
        for model_name in interval_fits:
            if model_name in selected_models:
                selected_models[model_name] = True
        manual_linear_active = False
        manual_linear_points: list[int] = []
        manual_linear_markers: list = []  # Store markers for selected points
        navigation_state = {"action": None}  # "continue", "go_back", "discard", "go_back_phase", "discard_file"
        
        fig, (ax_data, ax_resid) = plt.subplots(
            2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )
        plt.subplots_adjust(left=0.1, bottom=0.22, right=0.75, top=0.82)
        
        # Plot data - smaller dark gray dots
        data_line, = ax_data.plot(time_values, signal_values, "o", ms=3, color="#404040", label="Data", alpha=0.8)
        ax_data.set_ylabel("H₂O₂ (µM)", fontsize=11)
        title = f"Interval #{subset.index}: {subset.start_time:.3f}–{subset.end_time:.3f} s"
        if filename:
            display_name = truncate_filename(filename)
            title = f"{display_name} - {title}"
        ax_data.set_title(title, fontsize=12, fontweight="bold")
        ax_data.grid(True, alpha=0.3)
        ax_data.legend(loc="upper right")
        
        # Residuals plot
        ax_resid.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax_resid.set_xlabel("Time (s)", fontsize=11)
        ax_resid.set_ylabel("Residuals (µM)", fontsize=11)
        ax_resid.grid(True, alpha=0.3)
        
        # Store plot lines for fits
        fit_lines: dict[str, tuple] = {}  # model_name -> (data_line, resid_line)
        resid_lines: dict[str, tuple] = {}
        
        # Model colors - bright, vibrant colors
        model_colors = {
            "IB": "#0066FF",      # Bright blue
            "Exponential": "#00CC00",  # Bright green
            "GFI": "#FF3300",     # Bright red
            "LinearInitialRate": "#FF9900",  # Bright orange
        }
        
        def update_plot():
            """Update the plot with current fits."""
            # Check if figure is still valid by trying to access canvas
            try:
                _ = fig.canvas
            except (AttributeError, RuntimeError):
                return  # Figure has been closed or is invalid
            
            # Clear existing fit lines safely
            for model_name in list(fit_lines.keys()):
                try:
                    if model_name in fit_lines and fit_lines[model_name]:
                        fit_line = fit_lines[model_name][0]
                        if fit_line and hasattr(fit_line, 'axes') and fit_line.axes is not None:
                            fit_line.remove()
                except (AttributeError, ValueError, RuntimeError):
                    pass  # Line may have already been removed or figure closed
                
                try:
                    if model_name in resid_lines and resid_lines[model_name]:
                        resid_line = resid_lines[model_name][0]
                        if resid_line and hasattr(resid_line, 'axes') and resid_line.axes is not None:
                            resid_line.remove()
                except (AttributeError, ValueError, RuntimeError):
                    pass  # Line may have already been removed or figure closed
                
                if model_name in fit_lines:
                    del fit_lines[model_name]
                if model_name in resid_lines:
                    del resid_lines[model_name]
            
            # Plot selected fits
            try:
                for model_name, fit_result in interval_fits.items():
                    if model_name not in selected_models or not selected_models[model_name]:
                        continue
                    
                    # Safely extract yhat and validate
                    yhat = fit_result.get("yhat", None)
                    if yhat is None:
                        continue
                    
                    yhat = np.asarray(yhat, dtype=float)
                    if len(yhat) != len(signal_values) or np.any(~np.isfinite(yhat)):
                        continue
                    
                    residuals = signal_values - yhat
                    
                    # Plot fit with appropriate label
                    if model_name == "LinearInitialRate":
                        label = f"Linear Initial Rate (rate = {fit_result['init_rate']:.3f} µM/s)"
                    else:
                        label = f"{model_name} (R²={fit_result['r2']:.4f}, AIC={fit_result['aic']:.2f})"
                    
                    fit_line, = ax_data.plot(
                        time_values, yhat, "-", lw=1.5, color=model_colors[model_name], 
                        label=label
                    )
                    fit_lines[model_name] = (fit_line,)
                    
                    # Plot residuals
                    resid_line, = ax_resid.plot(
                        time_values, residuals, "-", lw=1.2, color=model_colors[model_name], alpha=0.8
                    )
                    resid_lines[model_name] = (resid_line,)
                
                ax_data.legend(loc="upper right", fontsize=9)
                fig.canvas.draw_idle()
            except (RuntimeError, AttributeError, ValueError):
                # Figure may have been closed during update
                pass
        
        def on_model_select(label):
            """Handle model checkbox selection."""
            model_name = label_to_model.get(label, label)
            if model_name not in selected_models:
                return
            selected_models[model_name] = not selected_models[model_name]
            # Don't auto-fit on checkbox click, user should click "Fit Selected"
            update_plot()
        
        def on_fit_all(_event):
            """Fit all selected models."""
            for model_name in ["IB", "Exponential", "GFI"]:
                if selected_models[model_name]:
                    # Always remove existing fit before re-fitting to ensure fresh fit
                    if model_name in interval_fits:
                        del interval_fits[model_name]
                    
                    try:
                        if model_name == "IB":
                            result = fit_IB(time_values, signal_values)
                        elif model_name == "Exponential":
                            result = fit_Exponential(time_values, signal_values)
                        elif model_name == "GFI":
                            result = fit_GFI(time_values, signal_values)
                        else:
                            continue
                        
                        # Validate fit quality - check for reasonable R² and non-NaN values
                        if not np.isfinite(result.get('r2', np.nan)) or result.get('r2', -1) < -10:
                            display_name = get_model_display_name(model_name)
                            print(f"  ✗ {display_name} fit produced invalid R²={result.get('r2', 'nan')}, rejecting fit")
                            selected_models[model_name] = False
                            # Update checkbox state
                            checkbox_label = model_to_label.get(model_name)
                            if checkbox_label in label_index:
                                idx = label_index[checkbox_label]
                                current_status = check_buttons.get_status()
                                if current_status[idx]:
                                    check_buttons.set_active(idx)
                            continue
                        
                        # Check for NaN or Inf in fitted values
                        yhat = result.get('yhat', None)
                        if yhat is not None:
                            yhat_array = np.asarray(yhat, dtype=float)
                            if len(yhat_array) > 0 and np.any(~np.isfinite(yhat_array)):
                                display_name = get_model_display_name(model_name)
                                print(f"  ✗ {display_name} fit produced invalid values (NaN/Inf), rejecting fit")
                                selected_models[model_name] = False
                                # Update checkbox state
                                checkbox_names = ["IB", "Exponential", "GFI"]
                                if model_name in checkbox_names:
                                    idx = checkbox_names.index(model_name)
                                    current_status = check_buttons.get_status()
                                    if current_status[idx]:
                                        check_buttons.set_active(idx)
                                continue
                        
                        # Store result (yhat is on original time scale)
                        interval_fits[model_name] = result
                        display_name = get_model_display_name(model_name)
                        print(f"  ✓ {display_name} fit complete: R²={result['r2']:.4f}, AIC={result['aic']:.2f}")
                    except Exception as e:
                        display_name = get_model_display_name(model_name)
                        print(f"  ✗ {display_name} fit failed: {e}")
                        # Ensure fit is removed if it was partially created
                        if model_name in interval_fits:
                            del interval_fits[model_name]
                        selected_models[model_name] = False
                        # Update checkbox state (set to unchecked if it was checked)
                        checkbox_label = model_to_label.get(model_name)
                        if checkbox_label in label_index:
                            idx = label_index[checkbox_label]
                            current_status = check_buttons.get_status()
                            if current_status[idx]:
                                check_buttons.set_active(idx)
                        checkbox_label = model_to_label.get(model_name)
                        if checkbox_label in label_index:
                            idx = label_index[checkbox_label]
                            current_status = check_buttons.get_status()
                            if current_status[idx]:
                                check_buttons.set_active(idx)
            
            update_plot()
        
        def apply_linear_fit_from_indices(start_idx: int, end_idx: int, mode: str) -> None:
            """Apply linear fit using data between start_idx and end_idx."""
            if end_idx <= start_idx:
                end_idx = min(start_idx + 1, len(time_values) - 1)
                if end_idx == start_idx:
                    print("  Unable to perform linear fit: need at least two distinct points.")
                    return
            result = fit_linear_initial_rate(time_values, signal_values, start_idx, end_idx)
            interval_fits["LinearInitialRate"] = result
            selected_models["LinearInitialRate"] = True
            print(
                f"  ✓ Linear Initial Rate ({mode}) fit complete: "
                f"rate = {result['init_rate']:.3f} µM/s "
                f"({result['fit_start_time']:.3f}-{result['fit_end_time']:.3f} s)"
            )
            update_plot()
        
        def on_linear_auto(_event):
            """Overlay automatic linear fit (first 10% of points or first 3 seconds, whichever is shorter)."""
            # Ensure we have at least 2 points for a linear fit
            if len(time_values) < 2:
                print("  Unable to perform linear fit: need at least 2 data points.")
                return
            
            # Calculate index based on 10% of points (ensure at least 2 points)
            num_points_10pct = max(2, int(len(time_values) * 0.1))
            idx_10pct = num_points_10pct - 1
            idx_10pct = min(idx_10pct, len(time_values) - 1)
            
            # Calculate index based on first 3 seconds
            auto_end_time = time_values[0] + 3.0
            idx_3sec = int(np.searchsorted(time_values, auto_end_time))
            idx_3sec = min(max(idx_3sec, 1), len(time_values) - 1)
            
            # Use whichever is shorter (smaller index), but ensure at least index 1 (2 points total)
            auto_end_idx = min(idx_10pct, idx_3sec)
            auto_end_idx = max(1, min(auto_end_idx, len(time_values) - 1))
            
            # Ensure we have at least 2 points (start_idx=0, end_idx>=1)
            if auto_end_idx < 1:
                auto_end_idx = 1
            
            apply_linear_fit_from_indices(0, auto_end_idx, "auto")
        
        def on_linear_manual(_event):
            """Enable manual selection of two points for linear fit."""
            nonlocal manual_linear_active, manual_linear_points, manual_linear_markers
            # Clear any existing linear fit first
            if "LinearInitialRate" in interval_fits:
                del interval_fits["LinearInitialRate"]
            selected_models["LinearInitialRate"] = False
            manual_linear_active = True
            manual_linear_points = []
            # Clear existing markers
            for marker in manual_linear_markers:
                try:
                    marker.remove()
                except (AttributeError, ValueError):
                    pass
            manual_linear_markers.clear()
            update_plot()  # Update to remove existing linear fit
            print(
                "  Manual linear fit: click two points on the data to define the fit window."
                " Use zoom/pan if needed; clicks while zoom/pan is active are ignored."
            )
        
        def on_clear_fits(_event):
            """Clear all fits and delete fit data."""
            nonlocal manual_linear_active, manual_linear_points, manual_linear_markers
            # Explicitly delete all fit data
            for model_name in list(interval_fits.keys()):
                del interval_fits[model_name]
            interval_fits.clear()  # Double-check: ensure dictionary is completely empty
            
            # Uncheck all model checkboxes
            current_status = check_buttons.get_status()
            for label, model_name in model_label_map:
                selected_models[model_name] = False
                idx = label_index[label]
                if current_status[idx]:
                    check_buttons.set_active(idx)
            
            # Clear linear initial rate fit
            selected_models["LinearInitialRate"] = False
            manual_linear_active = False
            manual_linear_points.clear()
            
            # Clear markers
            for marker in manual_linear_markers:
                try:
                    marker.remove()
                except (AttributeError, ValueError):
                    pass
            manual_linear_markers.clear()
            
            update_plot()
            print("All fits cleared and deleted.")
        
        def on_continue(_event):
            """Continue to next interval or finish."""
            # Check if any fits have been performed
            if not interval_fits:
                # Show warning dialog
                warning_fig = plt.figure(figsize=(6, 3))
                warning_fig.canvas.manager.set_window_title("Warning: No Fits Applied")
                ax_warning = warning_fig.add_axes([0.1, 0.4, 0.8, 0.3])
                ax_warning.axis('off')
                ax_warning.text(
                    0.5, 0.5,
                    "No fits have been applied to this interval.\n"
                    "Do you want to continue without fitting?",
                    ha="center", va="center", fontsize=11, fontweight="bold",
                    transform=ax_warning.transAxes, wrap=True
                )
                
                state = {"choice": None}
                
                def on_go_back(_event):
                    state["choice"] = "back"
                    plt.close(warning_fig)
                
                def on_continue_anyway(_event):
                    state["choice"] = "continue"
                    plt.close(warning_fig)
                
                ax_back = warning_fig.add_axes([0.2, 0.1, 0.25, 0.12])
                ax_continue_anyway = warning_fig.add_axes([0.55, 0.1, 0.25, 0.12])
                
                btn_back = create_small_button(ax_back, "Go Back", "#90ee90", "#7cd47c")
                btn_continue_anyway = create_small_button(ax_continue_anyway, "Continue Anyway", "#ff9999", "#ff6666")
                
                btn_back.on_clicked(on_go_back)
                btn_continue_anyway.on_clicked(on_continue_anyway)
                
                plt.show()
                plt.close(warning_fig)
                
                if state["choice"] == "back":
                    # User wants to go back, don't close the main figure
                    return
                # If "continue anyway" or window closed, proceed to close main figure
            navigation_state["action"] = "continue"
            plt.close(fig)
        
        def on_go_back_interval(_event):
            """Go back to previous interval, or to previous phase if on first interval."""
            if current_idx > 0:
                navigation_state["action"] = "go_back"
                plt.close(fig)
            else:
                navigation_state["action"] = "go_back_phase"
                plt.close(fig)
        
        def on_discard_interval(_event):
            """Discard current interval without fitting."""
            navigation_state["action"] = "discard"
            # Remove any fits for this interval
            if subset.index in all_fit_results:
                del all_fit_results[subset.index]
            plt.close(fig)
        
        def on_discard_file(_event):
            """Discard the entire file."""
            navigation_state["action"] = "discard_file"
            plt.close(fig)
        
        # Add checkboxes for model selection with equations
        # Make checkbox area taller to accommodate equations
        check_ax = fig.add_axes([0.78, 0.58, 0.2, 0.28], facecolor="0.95")
        model_label_map = [
            ("Inactivation", "IB"),
            ("Exponential", "Exponential"),
            ("Gompertz-like", "GFI"),
        ]
        checkbox_labels = [label for label, _ in model_label_map]
        label_to_model = {label: model_name for label, model_name in model_label_map}
        model_to_label = {model_name: label for label, model_name in model_label_map}
        label_index = {label: idx for idx, label in enumerate(checkbox_labels)}
        check_buttons = CheckButtons(
            check_ax,
            checkbox_labels,
            [False] * len(model_label_map),
        )
        check_buttons.on_clicked(on_model_select)
        check_ax.set_title("Select Models", fontsize=10, fontweight="bold")
        
        # Add equation text next to checkboxes using LaTeX math rendering
        # Equations are displayed below each checkbox label in mathematical notation
        model_equations = {
            "Inactivation": r"$H(t) = C + H_0 e^{-\alpha(1-e^{-k_i t})} - k_s t$",
            "Exponential": r"$y(t) = at + b + ce^{-k(t-t_0)}$",
            "Gompertz-like": r"$H(t) = C + A e^{-\alpha(1-e^{-k_i t})} e^{-e^{k(t-t_0)}} + B e^{-k_f t}$",
        }
        
        # Position equations below checkboxes (small font, mathematical notation)
        y_positions = [0.75, 0.50, 0.25]  # Vertical positions for each checkbox
        
        for i, (label, y_pos) in enumerate(zip(checkbox_labels, y_positions)):
            equation = model_equations[label]
            # Add equation text below checkbox using LaTeX math rendering
            check_ax.text(
                0.05,
                y_pos - 0.08,  # Position slightly below checkbox
                equation,
                fontsize=8,
                verticalalignment='top',
                horizontalalignment='left',
                transform=check_ax.transAxes
            )
        
        # Add buttons (adjusted positions to avoid overlap with checkboxes)
        btn_linear_auto_ax = fig.add_axes([0.78, 0.56, 0.2, 0.05])
        btn_linear_manual_ax = fig.add_axes([0.78, 0.50, 0.2, 0.05])
        btn_fit_all_ax = fig.add_axes([0.78, 0.44, 0.2, 0.05])
        btn_clear_ax = fig.add_axes([0.78, 0.38, 0.2, 0.05])
        btn_goback_ax = fig.add_axes([0.78, 0.28, 0.2, 0.05])
        btn_discard_ax = fig.add_axes([0.78, 0.22, 0.2, 0.05])
        btn_discard_file_ax = fig.add_axes([0.78, 0.16, 0.2, 0.05])
        btn_continue_ax = fig.add_axes([0.78, 0.10, 0.2, 0.05])
        
        btn_linear_auto = create_small_button(btn_linear_auto_ax, "Linear Auto", "#ffb347", "#ffa135")
        btn_linear_manual = create_small_button(btn_linear_manual_ax, "Linear Manual", "#ff7f0e", "#ff9500")
        btn_fit_all = create_small_button(btn_fit_all_ax, "Fit Selected", "#90ee90", "#7cd47c")
        btn_clear = create_small_button(btn_clear_ax, "Clear All", "#ffcc99", "#ffaa66")
        goback_label = "Go Back" if current_idx > 0 else "Back to intervals"
        btn_goback = create_small_button(btn_goback_ax, goback_label, "#ffcc99", "#ffaa66")
        btn_discard = create_small_button(btn_discard_ax, "Discard this interval", "#ff6666", "#ff4444")
        btn_discard_file = create_small_button(btn_discard_file_ax, "Discard this file", "#ff3333", "#cc0000")
        btn_continue = create_small_button(btn_continue_ax, "Continue", "#1f77b4", "#1e6ba8")
        
        btn_linear_auto.on_clicked(on_linear_auto)
        btn_linear_manual.on_clicked(on_linear_manual)
        btn_fit_all.on_clicked(on_fit_all)
        btn_clear.on_clicked(on_clear_fits)
        btn_goback.on_clicked(on_go_back_interval)
        btn_discard.on_clicked(on_discard_interval)
        btn_discard_file.on_clicked(on_discard_file)
        btn_continue.on_clicked(on_continue)
        
        # Instructions banner
        instructions_text = (
            "Check one or more models, then click 'Fit Selected'. 'Linear Auto' overlays the "
            "automatic initial-rate line. 'Linear Manual' lets you click two points to define a "
            "fit range. Multiple models can be displayed simultaneously."
        )
        add_instruction_banner(fig, instructions_text)
        
        print(f"\nFitting interface for Interval #{subset.index} ({current_idx + 1}/{len(subsets)})")
        print("  • Check models to fit (Inactivation, Exponential, Gompertz-like), then click 'Fit Selected'")
        print("  • 'Linear Auto' overlays the automatic initial-rate fit (first few seconds)")
        print("  • 'Linear Manual' lets you click two points to define the linear fit window")
        print("  • Fits will be displayed overlaid for comparison")
        print("  • Use 'Go Back' to return to previous interval")
        print("  • Use 'Discard' to skip this interval without fitting")
        print("  • Click 'Continue' when satisfied with fits")
        
        def handle_manual_click(event):
            nonlocal manual_linear_active, manual_linear_points, manual_linear_markers
            if not manual_linear_active or event.inaxes != ax_data or event.button != 1:
                return
            
            toolbar = fig.canvas.toolbar
            if toolbar is not None:
                mode = getattr(toolbar, "mode", "")
                active = getattr(toolbar, "_active", None)
                if mode in ("zoom rect", "pan/zoom", "zoom", "pan"):
                    return
                if active and active not in ("", None):
                    active_str = str(active).upper()
                    if "ZOOM" in active_str or "PAN" in active_str:
                        return
            
            click_time = float(event.xdata)
            nearest_idx = int(np.abs(time_values - click_time).argmin())
            manual_linear_points.append(nearest_idx)
            
            # Add visual marker for selected point
            try:
                marker, = ax_data.plot(
                    time_values[nearest_idx],
                    signal_values[nearest_idx],
                    "o",
                    ms=10,
                    color="#ff7f0e",
                    markeredgecolor="yellow",
                    markeredgewidth=2,
                    zorder=10,
                    label="Selected point" if len(manual_linear_points) == 1 else None
                )
                manual_linear_markers.append(marker)
                fig.canvas.draw_idle()
            except (AttributeError, RuntimeError):
                pass  # Figure may have been closed
            
            print(f"    • Selected point {len(manual_linear_points)}/2 at t={time_values[nearest_idx]:.3f} s")
            
            if len(manual_linear_points) == 2:
                start_idx, end_idx = sorted(manual_linear_points)
                manual_linear_active = False
                # Clear markers (they'll be replaced by the fit line)
                for marker in manual_linear_markers:
                    try:
                        marker.remove()
                    except (AttributeError, ValueError):
                        pass
                manual_linear_markers.clear()
                manual_linear_points.clear()
                apply_linear_fit_from_indices(start_idx, end_idx, "manual")
        
        click_cid = fig.canvas.mpl_connect("button_press_event", handle_manual_click)
        
        plt.show()
        fig.canvas.mpl_disconnect(click_cid)
        plt.close(fig)
        
        # Handle navigation
        action = navigation_state.get("action")
        if action == "discard_file":
            print("  File discarded by user during fitting.")
            return "discard_file"
        elif action == "go_back_phase":
            print("  Returning to interval selection.")
            return "go_back_phase"
        elif action == "discard":
            # Discard this interval - remove any fits and move to next
            if subset.index in all_fit_results:
                del all_fit_results[subset.index]
            print(f"  Interval #{subset.index} discarded.")
            current_idx += 1
            continue
        elif action == "go_back":
            # Go back to previous interval (only reachable when current_idx > 0)
            current_idx -= 1
            print(f"  Going back to interval #{subsets[current_idx].index}.")
            continue
        elif action == "continue":
            # Store results for this interval and move to next
            if interval_fits:
                all_fit_results[subset.index] = interval_fits
            current_idx += 1
        else:
            # Default: continue to next
            if interval_fits:
                all_fit_results[subset.index] = interval_fits
            current_idx += 1
    
    return all_fit_results


def calculate_turnover_before_inactivation(
    subsets: Sequence[IntervalSubset],
    all_fit_results: dict[int, dict[str, dict]],
    time_col: str,
    calibrated_col: str,
    filename: str | None = None,
) -> dict[int, float | None] | str:
    """
    Calculate maximum H2O2 turnover before inactivation for each interval.
    
    This function presents a GUI for each interval allowing the user to:
    - Attempt an automatic tailfit using a fitted model
    - Manually fit a linear line to the tail of the data
    - Skip the calculation for this interval
    
    Parameters:
    -----------
    subsets : Sequence[IntervalSubset]
        List of interval subsets
    all_fit_results : dict[int, dict[str, dict]]
        Dictionary mapping interval index to fit results from interactive_interval_fitting
    time_col : str
        Name of time column
    calibrated_col : str
        Name of calibrated H2O2 column
    filename : str | None
        Optional filename for display
    
    Returns:
    --------
    dict[int, float | None]
        Dictionary mapping interval index to turnover value (µM), or None if skipped.
    OR "go_back_phase" if user wants to return to the fitting phase.
    OR "discard_file" if user wants to discard the entire file.
    """
    if not subsets:
        return {}
    
    turnover_results: dict[int, float | None] = {}
    
    # Process each interval with navigation support
    current_idx = 0
    while current_idx < len(subsets):
        subset = subsets[current_idx]
        time_values = subset.data[time_col].to_numpy(dtype=float)
        signal_values = subset.data[calibrated_col].to_numpy(dtype=float)
        
        # Get fit results for this interval (if any)
        interval_fits = all_fit_results.get(subset.index, {})
        
        # State for this interval
        tail_fit_result: dict | None = None
        manual_tail_active = False
        manual_tail_points: list[int] = []
        manual_tail_markers: list = []
        navigation_state = {"action": None}  # "continue", "go_back", "discard", "skip"
        
        fig, ax = plt.subplots(figsize=(12, 6))
        plt.subplots_adjust(left=0.1, bottom=0.22, right=0.75, top=0.82)
        
        # Plot data
        ax.plot(time_values, signal_values, "o", ms=3, color="#404040", label="Data", alpha=0.8)
        ax.set_xlabel("Time (s)", fontsize=11)
        ax.set_ylabel("H₂O₂ (µM)", fontsize=11)
        title = f"Calculate maximum H₂O₂ turnover before inactivation\nInterval #{subset.index}: {subset.start_time:.3f}–{subset.end_time:.3f} s"
        if filename:
            display_name = truncate_filename(filename)
            title = f"{display_name} - {title}"
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
        
        # Store plot lines for tail fit
        tail_fit_line = None
        turnover_text = None
        
        def update_plot():
            """Update the plot with current tail fit."""
            nonlocal tail_fit_line, turnover_text
            try:
                _ = fig.canvas
            except (AttributeError, RuntimeError):
                return
            
            # Clear existing tail fit line
            if tail_fit_line is not None:
                try:
                    if hasattr(tail_fit_line, 'axes') and tail_fit_line.axes is not None:
                        tail_fit_line.remove()
                except (AttributeError, ValueError, RuntimeError):
                    pass
                tail_fit_line = None
            
            # Clear existing turnover text
            if turnover_text is not None:
                try:
                    if hasattr(turnover_text, 'axes') and turnover_text.axes is not None:
                        turnover_text.remove()
                except (AttributeError, ValueError, RuntimeError):
                    pass
                turnover_text = None
            
            # Plot tail fit if available
            if tail_fit_result is not None:
                try:
                    yhat = tail_fit_result.get("yhat", None)
                    if yhat is not None:
                        yhat = np.asarray(yhat, dtype=float)
                        if len(yhat) == len(time_values) and np.all(np.isfinite(yhat)):
                            tail_fit_line, = ax.plot(
                                time_values, yhat, "-", lw=2, color="#FF3300", 
                                label=f"Tail fit (turnover = {tail_fit_result.get('turnover', 0):.3f} µM)"
                            )
                            
                            # Display turnover value as text
                            turnover_val = tail_fit_result.get('turnover', 0)
                            turnover_text = ax.text(
                                0.02, 0.98,
                                f"Turnover: {turnover_val:.3f} µM",
                                transform=ax.transAxes,
                                fontsize=12,
                                fontweight="bold",
                                verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8)
                            )
                except (AttributeError, ValueError, RuntimeError):
                    pass
            
            ax.legend(loc="upper right", fontsize=9)
            fig.canvas.draw_idle()
        
        def on_tailfit_auto(_event):
            """Attempt automatic tailfit using best fitted model."""
            nonlocal tail_fit_result
            
            # Find best model (prefer Inactivation, Exponential, or Gompertz-like)
            best_model = None
            best_fit = None
            for model_name in ["IB", "Exponential", "GFI"]:
                if model_name in interval_fits:
                    best_model = model_name
                    best_fit = interval_fits[model_name]
                    break
            
            if best_fit is None:
                print("  ✗ No fitted model available for automatic tailfit. Please fit a model first in the fitting screen.")
                return
            
            try:
                yhat = best_fit.get("yhat", None)
                if yhat is None:
                    print("  ✗ Fitted model has no yhat values.")
                    return
                
                yhat = np.asarray(yhat, dtype=float)
                if len(yhat) != len(time_values) or np.any(~np.isfinite(yhat)):
                    print("  ✗ Fitted model has invalid yhat values.")
                    return
                
                # Get H0 value (initial H2O2 concentration)
                H0_val = signal_values[0]  # Use first data point as H0
                
                # Calculate turnover using tailfit
                turnover = turnover_tailfit(time_values, yhat, H0_val, window_frac=0.25)
                
                if np.isnan(turnover):
                    print("  ✗ Automatic tailfit calculation failed.")
                    return
                
                # Get tail fit line parameters for display
                n = len(time_values)
                w = max(3, int(n * 0.25))
                t_tail = time_values[-w:]
                y_tail = yhat[-w:]
                p = np.polyfit(t_tail, y_tail, 1)
                slope, intercept = p[0], p[1]
                
                # Create full tail fit line for display
                tail_yhat = slope * time_values + intercept
                
                tail_fit_result = {
                    "yhat": tail_yhat,
                    "turnover": turnover,
                    "slope": slope,
                    "intercept": intercept,
                    "method": "auto",
                    "model_used": best_model,
                }
                
                print(f"  ✓ Automatic tailfit complete: turnover = {turnover:.3f} µM (using {best_model} model)")
                update_plot()
            except Exception as e:
                print(f"  ✗ Automatic tailfit failed: {e}")
        
        def apply_manual_tail_fit(start_idx: int, end_idx: int) -> None:
            """Apply manual linear fit to tail region."""
            nonlocal tail_fit_result, manual_tail_active, manual_tail_points, manual_tail_markers
            
            if end_idx <= start_idx:
                end_idx = min(start_idx + 1, len(time_values) - 1)
                if end_idx == start_idx:
                    print("  Unable to perform tail fit: need at least two distinct points.")
                    return
            
            try:
                # Fit line to tail region
                t_tail = time_values[start_idx:end_idx+1]
                y_tail = signal_values[start_idx:end_idx+1]
                p = np.polyfit(t_tail, y_tail, 1)
                slope, intercept = p[0], p[1]
                
                # Create full tail fit line for display
                tail_yhat = slope * time_values + intercept
                
                # Calculate turnover: H0 - y_value_at_interval_start
                H0_val = signal_values[0]
                y_at_start = slope * time_values[0] + intercept
                turnover = max(0.0, H0_val - y_at_start)
                
                tail_fit_result = {
                    "yhat": tail_yhat,
                    "turnover": turnover,
                    "slope": slope,
                    "intercept": intercept,
                    "method": "manual",
                    "fit_start_time": float(time_values[start_idx]),
                    "fit_end_time": float(time_values[end_idx]),
                }
                
                print(f"  ✓ Manual tail fit complete: turnover = {turnover:.3f} µM")
                update_plot()
            except Exception as e:
                print(f"  ✗ Manual tail fit failed: {e}")
        
        def on_tailfit_manual(_event):
            """Enable manual selection of two points for tail fit."""
            nonlocal manual_tail_active, manual_tail_points, manual_tail_markers, tail_fit_result
            # Clear any existing tail fit first
            tail_fit_result = None
            manual_tail_active = True
            manual_tail_points = []
            # Clear existing markers
            for marker in manual_tail_markers:
                try:
                    marker.remove()
                except (AttributeError, ValueError):
                    pass
            manual_tail_markers.clear()
            update_plot()  # Update to remove existing tail fit
            print("  Manual tail fit: click two points on the data to define the tail fit window.")
        
        def on_clear_fit(_event):
            """Clear tail fit."""
            nonlocal tail_fit_result, manual_tail_active, manual_tail_points, manual_tail_markers
            tail_fit_result = None
            manual_tail_active = False
            manual_tail_points.clear()
            # Clear markers
            for marker in manual_tail_markers:
                try:
                    marker.remove()
                except (AttributeError, ValueError):
                    pass
            manual_tail_markers.clear()
            update_plot()
            print("Tail fit cleared.")
        
        def on_skip(_event):
            """Skip this interval without calculating turnover."""
            navigation_state["action"] = "skip"
            plt.close(fig)
        
        def on_continue(_event):
            """Continue to next interval."""
            # Check if tail fit has been performed
            if tail_fit_result is None:
                # Show warning dialog
                warning_fig = plt.figure(figsize=(6, 3))
                warning_fig.canvas.manager.set_window_title("Warning: No Tail Fit Applied")
                ax_warning = warning_fig.add_axes([0.1, 0.4, 0.8, 0.3])
                ax_warning.axis('off')
                ax_warning.text(
                    0.5, 0.5,
                    "No tail fit has been applied to this interval.\n"
                    "Do you want to continue without calculating turnover?",
                    ha="center", va="center", fontsize=11, fontweight="bold",
                    transform=ax_warning.transAxes, wrap=True
                )
                
                state = {"choice": None}
                
                def on_go_back(_event):
                    state["choice"] = "back"
                    plt.close(warning_fig)
                
                def on_continue_anyway(_event):
                    state["choice"] = "continue"
                    plt.close(warning_fig)
                
                ax_back = warning_fig.add_axes([0.2, 0.1, 0.25, 0.12])
                ax_continue_anyway = warning_fig.add_axes([0.55, 0.1, 0.25, 0.12])
                
                btn_back = create_small_button(ax_back, "Go Back", "#90ee90", "#7cd47c")
                btn_continue_anyway = create_small_button(ax_continue_anyway, "Continue Anyway", "#ff9999", "#ff6666")
                
                btn_back.on_clicked(on_go_back)
                btn_continue_anyway.on_clicked(on_continue_anyway)
                
                plt.show()
                plt.close(warning_fig)
                
                if state["choice"] == "back":
                    return
            navigation_state["action"] = "continue"
            plt.close(fig)
        
        def on_go_back(_event):
            """Go back to previous interval, or to previous phase if on first interval."""
            if current_idx > 0:
                navigation_state["action"] = "go_back"
                plt.close(fig)
            else:
                navigation_state["action"] = "go_back_phase"
                plt.close(fig)
        
        def on_discard(_event):
            """Discard this interval."""
            navigation_state["action"] = "discard"
            plt.close(fig)
        
        def on_discard_file(_event):
            """Discard the entire file."""
            navigation_state["action"] = "discard_file"
            plt.close(fig)
        
        # Add buttons
        btn_tailfit_auto_ax = fig.add_axes([0.78, 0.70, 0.2, 0.05])
        btn_tailfit_manual_ax = fig.add_axes([0.78, 0.64, 0.2, 0.05])
        btn_clear_ax = fig.add_axes([0.78, 0.58, 0.2, 0.05])
        btn_skip_ax = fig.add_axes([0.78, 0.50, 0.2, 0.05])
        btn_goback_ax = fig.add_axes([0.78, 0.44, 0.2, 0.05])
        btn_discard_ax = fig.add_axes([0.78, 0.38, 0.2, 0.05])
        btn_discard_file_ax = fig.add_axes([0.78, 0.32, 0.2, 0.05])
        btn_continue_ax = fig.add_axes([0.78, 0.26, 0.2, 0.05])
        
        btn_tailfit_auto = create_small_button(btn_tailfit_auto_ax, "Auto Tailfit", "#ffb347", "#ffa135")
        btn_tailfit_manual = create_small_button(btn_tailfit_manual_ax, "Manual Tail Fit", "#ff7f0e", "#ff9500")
        btn_clear = create_small_button(btn_clear_ax, "Clear Fit", "#ffcc99", "#ffaa66")
        btn_skip = create_small_button(btn_skip_ax, "Skip", "#cccccc", "#aaaaaa")
        goback_label = "Go Back" if current_idx > 0 else "Back to fitting"
        btn_goback = create_small_button(btn_goback_ax, goback_label, "#ffcc99", "#ffaa66")
        btn_discard = create_small_button(btn_discard_ax, "Discard this interval", "#ff6666", "#ff4444")
        btn_discard_file = create_small_button(btn_discard_file_ax, "Discard this file", "#ff3333", "#cc0000")
        btn_continue = create_small_button(btn_continue_ax, "Continue", "#1f77b4", "#1e6ba8")
        
        btn_tailfit_auto.on_clicked(on_tailfit_auto)
        btn_tailfit_manual.on_clicked(on_tailfit_manual)
        btn_clear.on_clicked(on_clear_fit)
        btn_skip.on_clicked(on_skip)
        btn_goback.on_clicked(on_go_back)
        btn_discard.on_clicked(on_discard)
        btn_discard_file.on_clicked(on_discard_file)
        btn_continue.on_clicked(on_continue)
        
        # Instructions banner
        instructions_text = (
            "Calculate maximum H₂O₂ turnover before inactivation. 'Auto Tailfit' uses the best fitted "
            "model to automatically fit the tail. 'Manual Tail Fit' lets you click two points to define "
            "the tail region. Turnover = H₀ - tail_intercept."
        )
        add_instruction_banner(fig, instructions_text)
        
        print(f"\nTurnover calculation for Interval #{subset.index} ({current_idx + 1}/{len(subsets)})")
        print("  • 'Auto Tailfit' attempts automatic calculation using fitted model")
        print("  • 'Manual Tail Fit' lets you select the tail region manually")
        print("  • 'Skip' to skip this interval")
        print("  • 'Continue' when satisfied with the calculation")
        
        def handle_manual_click(event):
            nonlocal manual_tail_active, manual_tail_points, manual_tail_markers
            if not manual_tail_active or event.inaxes != ax or event.button != 1:
                return
            
            toolbar = fig.canvas.toolbar
            if toolbar is not None:
                mode = getattr(toolbar, "mode", "")
                active = getattr(toolbar, "_active", None)
                if mode in ("zoom rect", "pan/zoom", "zoom", "pan"):
                    return
                if active is not None and str(active).upper() in ("ZOOM", "PAN"):
                    return
            
            nearest_idx = int(np.abs(time_values - event.xdata).argmin())
            if nearest_idx < 0 or nearest_idx >= len(time_values):
                return
            
            manual_tail_points.append(nearest_idx)
            
            try:
                marker = ax.plot(
                    time_values[nearest_idx],
                    signal_values[nearest_idx],
                    "o",
                    ms=8,
                    color="#FF3300",
                    markeredgecolor="black",
                    markeredgewidth=2,
                    zorder=10,
                    label="Selected point" if len(manual_tail_points) == 1 else None
                )[0]
                manual_tail_markers.append(marker)
                fig.canvas.draw_idle()
            except (AttributeError, RuntimeError):
                pass
            
            print(f"    • Selected point {len(manual_tail_points)}/2 at t={time_values[nearest_idx]:.3f} s")
            
            if len(manual_tail_points) == 2:
                start_idx, end_idx = sorted(manual_tail_points)
                manual_tail_active = False
                # Clear markers (they'll be replaced by the fit line)
                for marker in manual_tail_markers:
                    try:
                        marker.remove()
                    except (AttributeError, ValueError):
                        pass
                manual_tail_markers.clear()
                manual_tail_points.clear()
                apply_manual_tail_fit(start_idx, end_idx)
        
        click_cid = fig.canvas.mpl_connect("button_press_event", handle_manual_click)
        
        plt.show()
        fig.canvas.mpl_disconnect(click_cid)
        plt.close(fig)
        
        # Handle navigation
        action = navigation_state.get("action")
        if action == "discard_file":
            print("  File discarded by user during turnover calculation.")
            return "discard_file"
        elif action == "go_back_phase":
            print("  Returning to fitting phase.")
            return "go_back_phase"
        elif action == "discard":
            # Discard this interval — store None for consistency with skip
            turnover_results[subset.index] = None
            print(f"  Interval #{subset.index} discarded (no turnover calculated).")
            current_idx += 1
            continue
        elif action == "go_back":
            # Go back to previous interval (only reachable when current_idx > 0)
            current_idx -= 1
            print(f"  Going back to interval #{subsets[current_idx].index}.")
            continue
        elif action == "skip":
            # Skip this interval
            turnover_results[subset.index] = None
            print(f"  Interval #{subset.index} skipped (no turnover calculated).")
            current_idx += 1
        elif action == "continue":
            # Store result and move to next
            if tail_fit_result is not None:
                turnover = tail_fit_result.get("turnover", None)
                turnover_results[subset.index] = turnover
                print(f"  Interval #{subset.index}: turnover = {turnover:.3f} µM")
            else:
                turnover_results[subset.index] = None
                print(f"  Interval #{subset.index}: no turnover calculated.")
            current_idx += 1
        else:
            # Default: skip
            turnover_results[subset.index] = None
            current_idx += 1
    
    return turnover_results


def review_results(
    subsets: Sequence[IntervalSubset],
    all_fit_results: dict[int, dict[str, dict]],
    turnover_results: dict[int, float | None],
    filename: str | None = None,
) -> str:
    """
    Display a summary of all fits and turnover results and let the user
    accept, redo any phase, or discard the file.

    Returns:
    --------
    "accept" — save results and move on.
    "baseline" / "calibration" / "intervals" / "fitting" / "turnover"
        — redo that phase.
    "discard" — discard the entire file.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    plt.subplots_adjust(left=0.05, bottom=0.28, right=0.95, top=0.88)
    ax.axis("off")

    title = "Review Results"
    if filename:
        display_name = truncate_filename(filename)
        title = f"{display_name} — {title}"
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Build summary text
    lines: list[str] = []
    for subset in subsets:
        idx = subset.index
        time_range = f"{subset.start_time:.1f}–{subset.end_time:.1f} s"
        fits = all_fit_results.get(idx, {})
        turnover = turnover_results.get(idx, None)

        if fits:
            model_names = ", ".join(get_model_display_name(m) for m in fits)
        else:
            model_names = "none"

        if turnover is not None:
            turnover_str = f"{turnover:.3f} µM"
        else:
            turnover_str = "—"

        lines.append(
            f"Interval #{idx} ({time_range}):  "
            f"fits = {model_names},  turnover = {turnover_str}"
        )

    summary = "\n".join(lines) if lines else "(no intervals)"
    ax.text(
        0.05, 0.95, summary,
        transform=ax.transAxes,
        fontsize=10, fontfamily="monospace",
        verticalalignment="top",
    )

    state = {"action": None}

    def _make_cb(action_name):
        def cb(_event):
            state["action"] = action_name
            plt.close(fig)
        return cb

    # --- buttons (two rows) ---
    btn_w, btn_h, gap = 0.14, 0.05, 0.015
    # Row 1: redo buttons
    row1_y = 0.14
    labels_row1 = [
        ("Redo baseline",     "baseline",     "#d0d0ff", "#a8a8ff"),
        ("Redo calibration",  "calibration",  "#d0d0ff", "#a8a8ff"),
        ("Redo intervals",    "intervals",    "#d0d0ff", "#a8a8ff"),
        ("Redo fitting",      "fitting",      "#d0d0ff", "#a8a8ff"),
        ("Redo turnover",     "turnover",     "#d0d0ff", "#a8a8ff"),
    ]
    x = 0.05
    for label, action, col, hov in labels_row1:
        bax = fig.add_axes([x, row1_y, btn_w, btn_h])
        btn = create_small_button(bax, label, col, hov)
        btn.on_clicked(_make_cb(action))
        x += btn_w + gap

    # Row 2: accept / discard
    row2_y = 0.05
    accept_ax = fig.add_axes([0.25, row2_y, 0.20, 0.06])
    discard_ax = fig.add_axes([0.55, row2_y, 0.20, 0.06])
    btn_accept = create_small_button(accept_ax, "Accept and save", "#90ee90", "#7cd47c")
    btn_discard = create_small_button(discard_ax, "Discard this file", "#ff6666", "#ff4444")
    btn_accept.on_clicked(_make_cb("accept"))
    btn_discard.on_clicked(_make_cb("discard"))

    add_instruction_banner(
        fig,
        "Review the results below. Click 'Accept and save' to finalise, "
        "or use a 'Redo' button to go back to any processing step.",
    )

    print(
        "\nReview results:\n"
        "  • Check the summary above.\n"
        "  • 'Accept and save' finalises this file.\n"
        "  • 'Redo …' buttons let you repeat any step.\n"
        "  • 'Discard this file' skips the file entirely."
    )

    plt.show()
    plt.close(fig)

    return state.get("action") or "accept"

