#!/usr/bin/env python3
"""
collect_intervals.py
====================
Post-processing script for SensorFit.

Scans a directory for calibrated interval files
(Calibrated/*_intervals/interval_*.xlsx) and combines them into a single
Excel workbook with:

  - Column A: Time (s) — zero-based, common grid
  - Columns B …: H₂O₂ (µM) for each interval, named after the source
    file/folder (e.g. "Nc-0-2 #1", "Ncmet-0_5-3 #1").

By default, data are downsampled by a factor of 6.25 (e.g. 0.08 s → 0.5 s).
Use --downsample-factor to change the factor, or --no-downsample to keep the
original resolution.

By default, shorter intervals are extended to the length of the longest one
using a linear extrapolation of the last 10 data points.  Extrapolated cells
are written in red font.  Use --no-extend to disable this and leave shorter
traces as empty cells (NaN) instead.

Usage
-----
    python -m sensorfit.collect_intervals <directory> [-o output.xlsx]
    python -m sensorfit.collect_intervals <directory> --no-downsample
    python -m sensorfit.collect_intervals <directory> --downsample-factor 10
    python -m sensorfit.collect_intervals <directory> --no-extend

If -o is not given, the output is written as "all_intervals.xlsx" (or
"all_intervals_downsampled.xlsx" when downsampled) in the given directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font


def discover_intervals(root: Path) -> list[Path]:
    """Return sorted list of interval Excel files under root/Calibrated/."""
    calibrated = root / "Calibrated"
    if not calibrated.is_dir():
        # Fall back: maybe the user pointed directly at the Calibrated folder
        calibrated = root

    files = sorted(calibrated.glob("*_intervals/interval_*.xlsx"))
    return files


def label_from_path(p: Path) -> str:
    """
    Derive a human-readable column label from an interval path.

    Example:
        .../Calibrated/Nc-0-2_intervals/interval_01.xlsx  →  "Nc-0-2 #1"
        .../Calibrated/Nc-0-2_intervals/interval_02.xlsx  →  "Nc-0-2 #2"
    """
    folder = p.parent.name  # e.g. "Nc-0-2_intervals"
    sample = folder.replace("_intervals", "")

    stem = p.stem  # e.g. "interval_01"
    num = stem.split("_")[-1].lstrip("0") or "0"

    return f"{sample} #{num}"


def collect(
    root: Path,
    downsample_factor: float | None = 6.25,
    extend: bool = True,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Read every interval file, zero-base the time, and merge into one
    DataFrame with a common time column.

    Parameters
    ----------
    root : Path
        Directory containing Calibrated/*_intervals/ folders.
    downsample_factor : float or None
        Factor by which to increase the time step.  For example, 6.25
        turns a 0.08 s step into 0.5 s.  ``None`` keeps the original
        resolution.
    extend : bool
        If True, shorter intervals are extended to the length of the
        longest one by extrapolating a linear fit of the last 10 points.

    Returns
    -------
    df : pd.DataFrame
        Combined DataFrame with Time (s) as the first column.
    extend_start_rows : dict[str, int]
        Mapping of column label → first row index (0-based, in the data
        portion of the DataFrame, i.e. row 0 = first data row) where
        extrapolated (synthetic) values begin.  Columns that were not
        extended are absent from the dict.
    """
    files = discover_intervals(root)
    if not files:
        raise FileNotFoundError(
            f"No interval files found under {root}/Calibrated/*_intervals/"
        )

    # ------------------------------------------------------------------
    # Pass 1 — read files, collect per-file time steps, zero-base time
    # ------------------------------------------------------------------
    raw: list[tuple[str, np.ndarray, np.ndarray, float]] = []  # (label, t, y, dt)

    for f in files:
        df = pd.read_excel(f, engine="openpyxl")
        time_col = df.columns[0]  # usually "Time (s)"
        h2o2_col = df.columns[1]  # usually "H2O2_uM"

        t = df[time_col].to_numpy(dtype=float)
        y = df[h2o2_col].to_numpy(dtype=float)

        # Zero-base the time
        t_zeroed = t - t[0]

        dt = float(np.median(np.diff(t_zeroed))) if len(t) > 1 else 0.0

        label = label_from_path(f)
        raw.append((label, t_zeroed, y, dt))

    # ------------------------------------------------------------------
    # Verify that all intervals share the same time step
    # ------------------------------------------------------------------
    dt_arr = np.array([dt for (_, _, _, dt) in raw if dt > 0])
    dt_native = round(float(np.median(dt_arr)), 6)

    # Allow ±1 % tolerance
    if np.any(np.abs(dt_arr - dt_native) / dt_native > 0.01):
        unique_dts = sorted(set(round(v, 6) for v in dt_arr))
        print(
            f"WARNING: Not all intervals have the same time step. "
            f"Found steps: {unique_dts} s. Using median ({dt_native} s).",
            file=sys.stderr,
        )

    # ------------------------------------------------------------------
    # Determine output time step (with optional downsampling)
    # ------------------------------------------------------------------
    if downsample_factor is not None and downsample_factor > 1.0:
        dt_out = round(dt_native * downsample_factor, 6)
        print(
            f"Downsampling: native Δt = {dt_native} s  × {downsample_factor}  →  "
            f"output Δt = {dt_out} s"
        )
    else:
        dt_out = dt_native

    # ------------------------------------------------------------------
    # Build common time grid
    # ------------------------------------------------------------------
    max_time = max(t[-1] for (_, t, _, _) in raw)
    common_time = np.round(np.arange(0, max_time + dt_out / 2, dt_out), 6)
    n_rows = len(common_time)

    # ------------------------------------------------------------------
    # Align every series to the common grid
    # ------------------------------------------------------------------
    seen_labels: dict[str, int] = {}
    result = pd.DataFrame({"Time (s)": common_time})
    # Track where real data ends per column (0-based row index of first NaN)
    real_data_end: dict[str, int] = {}

    for label, t_zeroed, y, _ in raw:
        # Deduplicate labels
        if label in seen_labels:
            seen_labels[label] += 1
            label = f"{label} ({seen_labels[label]})"
        else:
            seen_labels[label] = 1

        s = pd.Series(y, index=np.round(t_zeroed, 6), name=label)
        tol = dt_out / 2
        aligned = s.reindex(common_time, method="nearest", tolerance=tol)
        vals = aligned.values

        # Find the last non-NaN index — everything after is missing/short
        finite_mask = np.isfinite(vals)
        if finite_mask.any():
            last_real = int(np.where(finite_mask)[0][-1])
        else:
            last_real = -1

        result[label] = vals

        # Record if this column is shorter than the full grid
        if last_real < n_rows - 1:
            real_data_end[label] = last_real

    # ------------------------------------------------------------------
    # Extend shorter intervals (linear extrapolation of last 10 points)
    # ------------------------------------------------------------------
    extend_start_rows: dict[str, int] = {}

    if extend and real_data_end:
        n_tail = 50  # number of tail points for the linear fit
        extended_count = 0
        for col_label, last_real in real_data_end.items():
            if last_real < 1:
                # Not enough data to extrapolate
                continue

            # Indices for the tail window (up to n_tail real points)
            tail_start = max(0, last_real - n_tail + 1)
            tail_idx = np.arange(tail_start, last_real + 1)
            t_tail = common_time[tail_idx]
            y_tail = result[col_label].values[tail_idx].astype(float)

            # Fit a line: y = slope * t + intercept
            if len(t_tail) < 2:
                continue
            coeffs = np.polyfit(t_tail, y_tail, 1)
            slope, intercept = coeffs[0], coeffs[1]

            # Fill the gap
            gap_idx = np.arange(last_real + 1, n_rows)
            if len(gap_idx) == 0:
                continue
            t_gap = common_time[gap_idx]
            y_extrap = slope * t_gap + intercept

            result.loc[gap_idx, col_label] = y_extrap
            # Record the first synthetic row (0-based DataFrame row index)
            extend_start_rows[col_label] = int(last_real + 1)
            extended_count += 1

        if extended_count:
            print(
                f"Extended {extended_count} interval(s) to full length "
                f"({n_rows} points) using linear tail extrapolation."
            )

    return result, extend_start_rows


def write_excel_with_red_extensions(
    df: pd.DataFrame,
    out_path: Path,
    extend_start_rows: dict[str, int],
) -> None:
    """
    Write *df* to an Excel file.  Cells that correspond to extrapolated
    (synthetic) data are formatted with red font.

    Parameters
    ----------
    df : pd.DataFrame
    out_path : Path
    extend_start_rows : dict[str, int]
        Column label → first 0-based data-row index of synthetic values.
    """
    # Write vanilla Excel first
    df.to_excel(out_path, index=False, engine="openpyxl")

    if not extend_start_rows:
        return  # nothing to colour

    wb = load_workbook(out_path)
    ws = wb.active
    red_font = Font(color="FF0000")

    # Build a map of column label → Excel column index (1-based)
    header_map: dict[str, int] = {}
    for col_idx in range(1, ws.max_column + 1):
        header_val = ws.cell(row=1, column=col_idx).value
        if header_val is not None:
            header_map[str(header_val)] = col_idx

    for col_label, first_synth_row in extend_start_rows.items():
        xl_col = header_map.get(col_label)
        if xl_col is None:
            continue
        # Excel rows: row 1 = header, row 2 = data row 0, etc.
        start_xl_row = first_synth_row + 2
        for xl_row in range(start_xl_row, ws.max_row + 1):
            cell = ws.cell(row=xl_row, column=xl_col)
            cell.font = red_font

    wb.save(out_path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Combine SensorFit calibrated intervals into a single Excel file."
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Root directory containing Calibrated/*_intervals/ folders.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output Excel path (default: <directory>/all_intervals.xlsx, "
        "or all_intervals_downsampled.xlsx when downsampled).",
    )

    ds_group = parser.add_mutually_exclusive_group()
    ds_group.add_argument(
        "--downsample-factor",
        type=float,
        default=6.25,
        help="Downsample factor applied to the native time step "
        "(default: 6.25, e.g. 0.08 s → 0.5 s).",
    )
    ds_group.add_argument(
        "--no-downsample",
        action="store_true",
        default=False,
        help="Disable downsampling; keep the original time resolution.",
    )

    parser.add_argument(
        "--no-extend",
        action="store_true",
        default=False,
        help="Do not extend shorter intervals to the longest one. "
        "Shorter traces will have empty cells instead.",
    )

    args = parser.parse_args(argv)

    root = args.directory.resolve()
    if not root.is_dir():
        print(f"Error: {root} is not a directory.", file=sys.stderr)
        sys.exit(1)

    do_downsample = not args.no_downsample and args.downsample_factor > 1.0
    factor = args.downsample_factor if do_downsample else None
    do_extend = not args.no_extend

    # Default output name
    if args.output:
        out_path = args.output.resolve()
    elif do_downsample:
        out_path = root / "all_intervals_downsampled.xlsx"
    else:
        out_path = root / "all_intervals.xlsx"

    print(f"Scanning: {root}")
    df, extend_start_rows = collect(
        root, downsample_factor=factor, extend=do_extend
    )
    n_intervals = len(df.columns) - 1  # minus Time column
    print(f"Found {n_intervals} interval(s), {len(df)} time points.")

    write_excel_with_red_extensions(df, out_path, extend_start_rows)
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
