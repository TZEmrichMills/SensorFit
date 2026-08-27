"""Command-line interface for calibration module."""
from __future__ import annotations

import argparse
import shutil
import sys
import traceback
from pathlib import Path
from typing import Sequence

import numpy as np

from .calibration import (
    apply_calibration,
    append_fit_summary,
    build_calibration,
    CALIBRATED_COLUMN,
    IntervalSubset,
    load_trace,
    pane_role_context,
    persist_interval_subsets,
    review_results,
    save_interval_with_fits,
    select_baseline,
    select_points,
)
from .interval_processor import (
    ProcessedInterval,
    run_per_interval_flow,
    select_one_interval,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate amperometric traces to H2O2 concentrations")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing input files (Excel: .xlsx, .xls, .xlsm, .xlsb or Text: .txt, .csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: same as input-dir)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="Glob pattern for input files (default: searches for *.xlsx, *.txt, *.csv)",
    )
    parser.add_argument(
        "--time-col",
        type=int,
        default=0,
        help="Column index for time (0-indexed, default 0 = 1st column)",
    )
    parser.add_argument(
        "--current-col",
        type=int,
        default=1,
        help="Column index for current (0-indexed, default 1 = 2nd column)",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=6,
        help="Number of calibration points to select",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=50,
        help="Number of samples to average around each selection",
    )
    parser.add_argument(
        "--calibration-values",
        type=str,
        default="0,20,40,60,80,100",
        help="Comma-separated µM H2O2 concentrations for each addition",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output files if they already exist",
    )
    parser.add_argument(
        "--skip-calibration",
        action="store_true",
        help=(
            "Treat input files as ALREADY-CALIBRATED [H₂O₂] vs time.  Skips the "
            "baseline and calibration phases entirely — SensorFit becomes a "
            "fitting-only tool.  --current-col is interpreted as the µM H₂O₂ "
            "column; the values are copied straight into the calibrated trace."
        ),
    )
    return parser.parse_args(argv)


def _pad_fit_yhat(fit_record, t_interval: np.ndarray) -> np.ndarray:
    """Project a FitRecord's yhat (which spans only the user-picked fit
    sub-range) onto the full interval time grid, NaN outside the range.

    The per-interval flow lets the user fit a sub-range, so ``yhat`` is
    shorter than the interval.  We interpolate it onto the actual sample
    times within [fit_start_s, fit_end_s] and leave the rest NaN so the
    column writes cleanly into the interval Excel.
    """
    yhat_padded = np.full(len(t_interval), np.nan, dtype=float)
    in_fit = (t_interval >= fit_record.fit_start_s) & (t_interval <= fit_record.fit_end_s)
    if in_fit.any():
        fit_t_dense = np.linspace(
            fit_record.fit_start_s, fit_record.fit_end_s, len(fit_record.yhat)
        )
        yhat_padded[in_fit] = np.interp(
            t_interval[in_fit], fit_t_dense, np.asarray(fit_record.yhat, dtype=float)
        )
    return yhat_padded


def _modal_process_control(
    path: Path,
    *,
    time_col_idx: int,
    current_col_idx: int,
    num_points: int,
    window: int,
    calibration_values: list[float],
    update_calibration_callback,
    skip_calibration: bool,
    calibrated_dir: Path,
    force: bool,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Process a freshly-picked control file modal-style and return its
    chosen interval as ``(time, [H₂O₂])``.

    This backs the per-interval flow's "Subtract new control alongside"
    option.  It runs a stripped pipeline — baseline (skippable) →
    calibration → single-interval selection — then saves the control's
    calibrated trace and that interval under ``Calibrated/`` exactly as if
    the file had been processed independently (so it's also available later
    via the "Subtract existing" picker).  No fitting / Δmax / fit_summary
    rows are written (a control template needs none).

    Returns ``None`` if the user discards / cancels at any step.
    """
    print(f"\n[modal control] Processing {path.name} for subtraction…")
    try:
        frame = load_trace(path, time_col_idx, current_col_idx)
    except Exception as exc:
        print(f"[modal control] Could not load {path.name}: {exc}")
        return None
    time_col = frame.columns[0]
    current_col = frame.columns[1]
    time_values = frame[time_col].to_numpy(dtype=float)
    raw = frame[current_col].to_numpy(dtype=float)

    with pane_role_context("control"):
        if skip_calibration:
            frame[CALIBRATED_COLUMN] = raw.copy()
        else:
            # Loop so "Go Back" from calibration re-opens baseline.
            while True:
                signal_values = None
                while True:
                    br = select_baseline(time_values, raw, window=20, filename=path.name)
                    if br is None:
                        print("[modal control] Discarded at baseline.")
                        return None
                    if br == "redraw":
                        continue
                    signal_values, _meta = br
                    break
                frame[current_col] = signal_values

                result = select_points(
                    time_values, signal_values, num_points,
                    calibration_values, update_calibration_callback,
                    filename=path.name, window=window,
                )
                if result == "discard":
                    print("[modal control] Discarded at calibration.")
                    return None
                if result == "go_back_to_baseline":
                    continue  # re-open baseline
                indices, _vals, mean_currents = result
                calibration = build_calibration(mean_currents, _vals)
                frame[CALIBRATED_COLUMN] = apply_calibration(frame[current_col], calibration)
                break

        # Pick one interval as the control template.
        sel = select_one_interval(
            time_values, frame[CALIBRATED_COLUMN].to_numpy(dtype=float),
            already_defined=[], filename=f"{path.name} (control)", interval_number=1,
        )
    if not isinstance(sel, tuple):
        print("[modal control] No interval selected; subtraction cancelled.")
        return None
    s_t, e_t = sel
    mask = (time_values >= s_t) & (time_values <= e_t)
    if not np.any(mask):
        print("[modal control] Empty interval; subtraction cancelled.")
        return None

    # Save "as if processed independently": calibrated trace + interval excel.
    calibrated_dir.mkdir(parents=True, exist_ok=True)
    out_path = calibrated_dir / f"{path.stem}_calibrated.xlsx"
    if not (out_path.exists() and not force):
        try:
            frame.to_excel(out_path, index=False)
            print(f"[modal control] Saved calibrated trace → {out_path.name}")
        except Exception as exc:
            print(f"[modal control] Warning: could not save calibrated trace: {exc}")

    sub_df = frame.loc[mask, [time_col, CALIBRATED_COLUMN]].copy().reset_index(drop=True)
    interval_dir = calibrated_dir / f"{path.stem}_intervals"
    interval_dir.mkdir(parents=True, exist_ok=True)
    try:
        sub_df.to_excel(interval_dir / "interval_01.xlsx", index=False)
        print(f"[modal control] Saved control interval → {interval_dir.name}/interval_01.xlsx")
    except Exception as exc:
        print(f"[modal control] Warning: could not save control interval: {exc}")

    return (
        sub_df[time_col].to_numpy(dtype=float),
        sub_df[CALIBRATED_COLUMN].to_numpy(dtype=float),
    )


def parse_calibration_values(arg: str, num_points: int) -> list[float]:
    """Parse calibration values from string."""
    if not arg.strip():
        raise ValueError("Calibration values cannot be empty")
    parts = [item.strip() for item in arg.split(",") if item.strip()]
    values = [float(p) for p in parts]
    if len(values) != num_points:
        raise ValueError(
            f"Number of --calibration-values entries ({len(values)}) must equal --num-points ({num_points})"
        )
    return values


def process_file(
    path: Path,
    output_dir: Path,
    time_col_idx: int,
    current_col_idx: int,
    num_points: int,
    window: int,
    calibration_values: list[float],
    update_calibration_callback,
    force: bool,
    summary_path: Path,
    calibrated_dir: Path | None = None,
    skip_calibration: bool = False,
) -> tuple[Path | None, Path, bool]:
    """Process a single file through a phase-based state machine.

    Phases: baseline → calibration → intervals (per-interval loop) → review
    → save.  Per-interval control subtraction, fitting and Δmax all happen
    inside the per-interval loop (run_per_interval_flow).  File I/O is
    deferred until the review screen is accepted.

    Parameters
    ----------
    calibrated_dir : Path | None
        Where the Calibrated/ folder lives.  Used for discovering existing
        control intervals (for per-interval subtraction) and saving outputs.
    skip_calibration : bool
        If True, treat ``current_col_idx`` as the already-calibrated [H₂O₂]
        column (µM).  Baseline and calibration phases are skipped entirely
        and the values are copied directly into the calibrated trace.  Use
        when SensorFit is run purely as a fitting tool against pre-processed
        data (typically from another pipeline).

    Returns
    -------
    (out_path, original_path, has_fits)
    """
    print(f"\nProcessing {path}")
    frame = load_trace(path, time_col_idx, current_col_idx)
    time_col = frame.columns[0]
    current_col = frame.columns[1]

    time_values = frame[time_col].to_numpy(dtype=float)
    signal_values_raw = frame[current_col].to_numpy(dtype=float)

    # In skip-calibration mode the "current" column already holds [H₂O₂] (µM).
    # Sanity check: pure-current amperometry data is typically in the µA / nA
    # range (|values| < 0.01).  Pre-calibrated H₂O₂ is typically tens to
    # hundreds of µM.  We don't reject either way (some experiments may sit
    # in unusual ranges) but warn loudly so the user notices if they forgot
    # to pass --current-col for a non-default column.
    if skip_calibration:
        if np.all(np.isfinite(signal_values_raw)):
            abs_max = float(np.max(np.abs(signal_values_raw)))
            if abs_max < 0.1:
                print(
                    f"  ⚠  --skip-calibration is on but the selected column "
                    f"(index {current_col_idx}, header '{current_col}') has |max| "
                    f"= {abs_max:.4g} — looks like raw current, not [H₂O₂] in µM.  "
                    "Re-check --current-col / --time-col."
                )
            else:
                print(
                    f"  Skip-calibration: using column '{current_col}' as [H₂O₂] (µM); "
                    f"range = [{np.min(signal_values_raw):.2f}, {np.max(signal_values_raw):.2f}] µM."
                )

    # ── persistent state across phases ──────────────────────────
    signal_values = None
    calibration = None
    current_calibration_values = calibration_values
    subsets = None
    all_fit_results: dict[int, dict[str, dict]] = {}
    turnover_results: dict[int, float | None] = {}
    # The per-interval flow keeps the full ProcessedInterval list so the
    # save phase can emit one fit_summary row per fit and one Excel column
    # per fit.
    processed_intervals: list = []
    excel_fits_by_index: dict = {}

    # In skip-calibration mode the baseline and calibration phases are bypassed
    # entirely.  The CALIBRATED_COLUMN is populated directly from the input
    # current column (which the user has confirmed holds µM H₂O₂).
    if skip_calibration:
        signal_values = signal_values_raw.copy()
        frame[CALIBRATED_COLUMN] = signal_values
        phase = "intervals"
    else:
        phase = "baseline"

    while True:
        # ────────────────────────────────────────────────────────
        # PHASE: baseline
        # ────────────────────────────────────────────────────────
        if phase == "baseline":
            while True:
                baseline_result = select_baseline(
                    time_values, signal_values_raw, window=20, filename=path.name
                )
                if baseline_result is None:
                    print(f"Dataset {path.name} discarded by user.")
                    return None, path, False
                if baseline_result == "redraw":
                    continue
                # baseline_result is (corrected_signal, meta); only the
                # corrected signal is needed downstream.
                signal_values, _baseline_meta = baseline_result
                frame[current_col] = signal_values
                break
            phase = "calibration"
            continue

        # ────────────────────────────────────────────────────────
        # PHASE: calibration
        # ────────────────────────────────────────────────────────
        elif phase == "calibration":
            print(f"Using calibration values: {current_calibration_values}")
            result = select_points(
                time_values,
                signal_values,
                num_points,
                current_calibration_values,
                update_calibration_callback,
                filename=path.name,
                window=window,
            )
            if result == "discard":
                print(f"\nDataset {path.name} discarded during calibration.")
                return None, path, False
            if result == "go_back_to_baseline":
                print("\nReturning to baseline selection...")
                phase = "baseline"
                continue

            # select_points returns a uniform 3-tuple in BOTH modes:
            # (indices, calibration_values, mean_currents), the mean_currents
            # computed with the in-screen window.
            indices, final_calibration_values, mean_currents = result
            mean_currents = list(mean_currents)
            current_calibration_values = final_calibration_values
            calibration = build_calibration(mean_currents, final_calibration_values)
            frame[CALIBRATED_COLUMN] = apply_calibration(frame[current_col], calibration)

            summary_rows = list(zip(indices, mean_currents, final_calibration_values))
            print("Selected points (index, mean current, µM):")
            for idx, current, conc in summary_rows:
                print(f"  idx={idx:5d}, current={current: .4e} A, conc={conc: .3f} µM")
            print(f"Calibration: µM = {calibration.slope:.4e} * current + {calibration.intercept:.4f}")

            phase = "intervals"
            continue

        # ────────────────────────────────────────────────────────
        # PHASE: intervals  (per-interval loop — selection, optional
        # control subtraction, fits, Δmax all happen inside)
        # ────────────────────────────────────────────────────────
        elif phase == "intervals":
            calibrated_dir_for_flow = (
                calibrated_dir if calibrated_dir is not None else output_dir / "Calibrated"
            )
            # Default Δmax max-µM = largest calibration value (or 100 if we
            # skipped calibration or only have one value).
            try:
                cal_max_uM = max(float(v) for v in (current_calibration_values or []))
            except (TypeError, ValueError):
                cal_max_uM = 100.0
            if not (cal_max_uM and cal_max_uM > 0):
                cal_max_uM = 100.0

            # Callback for "Subtract new control alongside": run a modal
            # baseline→calibration→one-interval pipeline on the picked file
            # and hand back its chosen interval.
            def _new_control_cb(picked_path: Path):
                return _modal_process_control(
                    picked_path,
                    time_col_idx=time_col_idx,
                    current_col_idx=current_col_idx,
                    num_points=num_points,
                    window=window,
                    calibration_values=current_calibration_values,
                    update_calibration_callback=update_calibration_callback,
                    skip_calibration=skip_calibration,
                    calibrated_dir=calibrated_dir_for_flow,
                    force=force,
                )

            _pi = run_per_interval_flow(
                time_values=time_values,
                h2o2_values=frame[CALIBRATED_COLUMN].to_numpy(dtype=float),
                time_col=time_col,
                full_frame=frame,
                filename=path.name,
                calibrated_dir=calibrated_dir_for_flow,
                cal_max_uM=cal_max_uM,
                new_control_callback=_new_control_cb,
            )
            if _pi == "go_back_to_calibration":
                if skip_calibration:
                    print("\nNothing to go back to in skip-calibration mode; reopening intervals.")
                    continue
                print("\nReturning to calibration point selection...")
                phase = "calibration"
                continue
            if not _pi:
                print("No intervals saved.")
                calibrated_dir_local = output_dir / "Calibrated"
                calibrated_dir_local.mkdir(parents=True, exist_ok=True)
                out_path = calibrated_dir_local / f"{path.stem}_calibrated.xlsx"
                frame.to_excel(out_path, index=False)
                return out_path, path, False
            processed_intervals = _pi

            # Build the containers the save phase consumes.  fit_summary
            # emits one row per fit (from pi.fits); the Excel carries one
            # column per fit (excel_fits_by_index); all_fit_results keeps
            # just the first fit for the legacy best-model summary columns.
            subsets = []
            all_fit_results = {}
            excel_fits_by_index = {}
            turnover_results = {}
            for pi in processed_intervals:
                subsets.append(IntervalSubset(
                    index=pi.index,
                    start_time=pi.start_time,
                    end_time=pi.end_time,
                    data=pi.data,
                ))
                if pi.fits:
                    t_interval = pi.data[pi.time_col].to_numpy(dtype=float)
                    excel_fits: dict[str, dict] = {}
                    for fnum, frec in enumerate(pi.fits, start=1):
                        label = frec.model if len(pi.fits) == 1 else f"{frec.model}#{fnum}"
                        excel_fits[label] = {
                            "model": frec.model,
                            "params": np.array(frec.params, dtype=float),
                            "names": frec.param_names,
                            "yhat": _pad_fit_yhat(frec, t_interval),
                            "r2": frec.r2,
                            "rss": frec.rss,
                            "init_rate": frec.init_rate_uM_per_s,
                        }
                    excel_fits_by_index[pi.index] = excel_fits
                    f0 = pi.fits[0]
                    all_fit_results[pi.index] = {
                        f0.model: {
                            "model": f0.model,
                            "params": np.array(f0.params, dtype=float),
                            "names": f0.param_names,
                            "yhat": _pad_fit_yhat(f0, t_interval),
                            "r2": f0.r2,
                            "rss": f0.rss,
                            "init_rate": f0.init_rate_uM_per_s,
                        },
                    }
                turnover_results[pi.index] = (
                    pi.delta_max.value_uM if pi.delta_max is not None else None
                )

            phase = "review"
            continue

        # ────────────────────────────────────────────────────────
        # PHASE: review
        # ────────────────────────────────────────────────────────
        elif phase == "review":
            decision = review_results(
                subsets, all_fit_results, turnover_results,
                filename=path.name,
                skip_calibration=skip_calibration,
                per_interval_mode=True,
            )
            if decision == "discard":
                print(f"\nDataset {path.name} discarded at review.")
                return None, path, False
            if decision == "accept":
                phase = "save"
                continue
            if decision in ("baseline", "calibration") and skip_calibration:
                print("  (baseline/calibration not available in skip-calibration mode; staying at review)")
                continue
            # Per-interval mode only exposes Redo-baseline / Redo-calibration /
            # Redo-intervals.  "intervals" re-runs the whole per-interval flow.
            if decision in ("baseline", "calibration", "intervals"):
                print(f"\nRedoing {decision} phase...")
                phase = decision
                continue
            # Fallback: treat as accept
            phase = "save"
            continue

        # ────────────────────────────────────────────────────────
        # PHASE: save  (terminal — writes files and returns)
        # ────────────────────────────────────────────────────────
        elif phase == "save":
            break

    # ── Deferred file I/O ───────────────────────────────────────
    calibrated_dir = output_dir / "Calibrated"
    calibrated_dir.mkdir(parents=True, exist_ok=True)
    out_path = calibrated_dir / f"{path.stem}_calibrated.xlsx"
    if out_path.exists() and not force:
        raise FileExistsError(f"{out_path} already exists (use --force to overwrite)")
    frame.to_excel(out_path, index=False)
    print(f"Saved calibrated trace to {out_path}")

    interval_dir = calibrated_dir / f"{path.stem}_intervals"
    has_fits = len(all_fit_results) > 0

    # Build a quick map: pi_by_index → ProcessedInterval (for per-fit rows)
    pi_by_index = {pi.index: pi for pi in processed_intervals}

    print(f"\nSaving interval data and fits...")
    for subset in subsets:
        interval_fits = all_fit_results.get(subset.index, None)
        turnover_uM = turnover_results.get(subset.index, None)
        # The Excel gets ALL fits (one yhat/residual column each); the
        # legacy summary still uses the first-fit dict.  Fall back to
        # interval_fits for the control/legacy path (no per-fit detail).
        excel_fits = excel_fits_by_index.get(subset.index, interval_fits)
        excel_path = save_interval_with_fits(
            subset, time_col, CALIBRATED_COLUMN, excel_fits, interval_dir
        )
        print(f"  Saved: {excel_path.name}")

        # Per-interval control-subtraction provenance comes from the
        # ProcessedInterval itself (subtraction is per-interval now).
        pi = pi_by_index.get(subset.index)
        was_subtracted = bool(pi.control_subtracted) if pi is not None else False
        ctrl_source = pi.control_source if (pi is not None and was_subtracted) else None
        common_kwargs = dict(
            summary_path=summary_path,
            source_file=path.name,
            interval_index=subset.index,
            start_time=subset.start_time,
            end_time=subset.end_time,
            control_subtracted=was_subtracted if pi is not None else None,
            control_group=ctrl_source,
            control_n_averaged=pi.control_n_averaged if pi is not None else None,
            calibration_skipped=True if skip_calibration else None,
        )

        if pi is not None and pi.fits:
            # New flow: one row per fit (fit_number = 1, 2, …), each
            # carrying the FitRecord columns + the (denormalised) Δmax.
            for fnum, frec in enumerate(pi.fits, start=1):
                append_fit_summary(
                    fit_results=interval_fits if fnum == 1 else None,
                    turnover_uM=turnover_uM,
                    fit_number=fnum,
                    fit_record=frec,
                    delta_max_record=pi.delta_max,
                    **common_kwargs,
                )
        else:
            # No fits applied OR legacy / control path: one row with
            # fit_number = 0 carrying interval-level + Δmax info only.
            append_fit_summary(
                fit_results=interval_fits,
                turnover_uM=turnover_uM,
                fit_number=0,
                fit_record=None,
                delta_max_record=(pi.delta_max if pi is not None else None),
                **common_kwargs,
            )

    persist_interval_subsets(subsets, interval_dir)

    # ── Print summary ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Fitting Summary for This File")
    print(f"{'='*60}")
    print(f"File: {path.name}")
    print(f"Total intervals: {len(subsets)}")
    intervals_with_fits = len(all_fit_results)
    intervals_without_fits = len(subsets) - intervals_with_fits
    if intervals_with_fits > 0:
        print(f"Intervals with fits applied: {intervals_with_fits}")
        print(f"Intervals without fits: {intervals_without_fits}")
        print(f"\nFit details:")
        for interval_idx, models in all_fit_results.items():
            subset = next((s for s in subsets if s.index == interval_idx), None)
            if subset:
                time_range = f"{subset.start_time:.3f}-{subset.end_time:.3f} s"
                print(f"  Interval #{interval_idx} ({time_range}): {', '.join(models.keys())}")
    else:
        print("  (No fits were applied to any intervals)")
    print(f"\nOutput files:")
    print(f"  Calibrated data: {out_path}")
    print(f"  Interval data: {interval_dir}")
    print(f"  Summary Excel: {summary_path}")
    print(f"{'='*60}\n")

    return out_path, path, has_fits


def main(argv: Sequence[str] | None = None) -> int:
    """Main entry point for calibration CLI."""
    args = parse_args(argv)
    
    # Parse calibration values.  In skip-calibration mode they're unused but
    # may still have been supplied — accept either way.
    if args.skip_calibration:
        print(
            "\n=== Skip-calibration mode ===\n"
            f"Treating input files as already-calibrated [H₂O₂] vs time.\n"
            f"  --time-col    = {args.time_col}\n"
            f"  --current-col = {args.current_col}  (interpreted as µM H₂O₂)\n"
            "Baseline and calibration phases will be bypassed.\n"
        )
        # When skipping calibration we still need a values list for the signature
        # but it's never used.  Tolerate an empty / mismatched --calibration-values.
        try:
            calibration_values = parse_calibration_values(args.calibration_values, args.num_points)
        except ValueError:
            calibration_values = [0.0] * args.num_points
    else:
        calibration_values = parse_calibration_values(args.calibration_values, args.num_points)

    # Set output directory
    if args.output_dir is None:
        args.output_dir = args.input_dir

    # Summary Excel lives inside Calibrated/ so it is never re-discovered as an
    # input file and an interrupted-then-restarted session resumes the same
    # file (Calibrated/ is excluded from file discovery below).
    calibrated_out_dir = args.output_dir / "Calibrated"
    calibrated_out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = calibrated_out_dir / "fit_summary.xlsx"

    # Migrate any legacy fit_summary.xlsx sitting at the root of the output
    # directory (previous SensorFit versions put it there).  Move rather than
    # copy to avoid silent duplication.
    legacy_summary = args.output_dir / "fit_summary.xlsx"
    if legacy_summary.exists() and not summary_path.exists():
        try:
            shutil.move(str(legacy_summary), str(summary_path))
            print(f"Migrated legacy fit_summary.xlsx → {summary_path}")
        except OSError as exc:
            print(f"Warning: could not migrate {legacy_summary}: {exc}")

    if summary_path.exists():
        try:
            import pandas as _pd
            _existing = _pd.read_excel(summary_path, engine="openpyxl")
            n_rows = len(_existing)
            n_files = (
                _existing["source_file"].nunique()
                if "source_file" in _existing.columns
                else 0
            )
            print(
                f"Resuming: fit_summary contains {n_rows} row(s) from {n_files} file(s) — new rows will append."
            )
        except Exception as exc:
            print(f"Existing summary at {summary_path} could not be read ({exc}); will be overwritten on next write.")
    else:
        print(f"Summary Excel: {summary_path} (will be created on first save)")

    # Discover files (exclude already calibrated files and files in Calibrated folder)
    calibrated_dir = args.input_dir / "Calibrated"
    
    # If no pattern specified, search for common file types
    if args.pattern is None:
        all_files = []
        for pattern in ["*.xlsx", "*.xls", "*.xlsm", "*.xlsb", "*.txt", "*.csv"]:
            all_files.extend(args.input_dir.glob(pattern))
        all_files = sorted(set(all_files))  # Remove duplicates and sort
    else:
        all_files = sorted(args.input_dir.glob(args.pattern))
    
    # Filter out:
    # 1. Files that already have "_calibrated" in their name
    # 2. Files in the Calibrated folder
    # 3. Files in the Processed folder
    # 4. Any fit_summary* spreadsheet (belt-and-braces; the summary should now
    #    live in Calibrated/ but a legacy file may still be at the root mid-run)
    files = []
    processed_dir_path = args.input_dir / "Processed"
    for f in all_files:
        # Skip if file is in Calibrated or Processed folder
        # Check if any parent directory is Calibrated or Processed
        if any(p.name == "Calibrated" for p in f.parents) or any(p.name == "Processed" for p in f.parents):
            continue
        # Skip temporary Excel lock files (often start with "~$")
        if f.name.startswith("~$"):
            continue
        # Skip hidden / macOS "AppleDouble" companion files (e.g.
        # "._260721_sample.txt").  macOS creates one of these next to every
        # real file on non-HFS volumes (USB sticks, network/SMB shares); they
        # match "*.txt" but are tiny binary resource forks, not data.
        if f.name.startswith("._") or f.name.startswith("."):
            continue
        # Skip if filename already contains "_calibrated"
        if "_calibrated" in f.stem:
            continue
        # Skip the summary file and any analysis copies of it
        if f.stem.startswith("fit_summary"):
            continue
        files.append(f)
    
    if not files:
        print(f"No unprocessed files matching '{args.pattern}' under {args.input_dir}")
        print("Note: Files with '_calibrated' suffix or in 'Calibrated' folder are excluded.")
        return 1

    processed_dir = args.input_dir / "Processed"
    processed_dir.mkdir(exist_ok=True)
    calibrated_dir.mkdir(exist_ok=True)

    # Use mutable containers for calibration values and num_points
    session_calibration = {"values": calibration_values}
    session_num_points = {"value": args.num_points}

    def update_calibration(new_values: list[float], apply_to_all: bool) -> None:
        if apply_to_all:
            session_calibration["values"] = new_values
            # Update num_points to match the number of calibration values
            new_num_points = len(new_values)
            if new_num_points != session_num_points["value"]:
                session_num_points["value"] = new_num_points
                print(f"Number of calibration points updated for all remaining files: {session_num_points['value']}")
            print(f"Calibration values updated for all remaining files: {new_values}")

    # ── Statistics + small helper to process one file ─────────────────────
    total_files = len(files)
    stats = {"discarded": 0, "successful": 0}

    def _move_to_processed(p: Path) -> bool:
        """Move file to Processed/.  Returns False if move was blocked
        (caller should exit)."""
        processed_path = processed_dir / p.name
        if processed_path.exists():
            print(f"Warning: {processed_path} already exists; skipping move.")
            return True
        try:
            shutil.move(str(p), str(processed_path))
            return True
        except OSError as e:
            if sys.platform == "win32" and "being used by another process" in str(e):
                print(f"Warning: File {p.name} is locked. Please close it and try again.")
                return False
            raise

    def run_one(file_path: Path) -> None:
        """Call process_file and handle the post-processing housekeeping."""
        out_path, original_path, has_fits = process_file(
            path=file_path,
            output_dir=args.output_dir,
            time_col_idx=args.time_col,
            current_col_idx=args.current_col,
            num_points=session_num_points["value"],
            window=args.window,
            calibration_values=session_calibration["values"],
            update_calibration_callback=update_calibration,
            force=args.force,
            summary_path=summary_path,
            calibrated_dir=calibrated_out_dir,
            skip_calibration=args.skip_calibration,
        )
        if not _move_to_processed(original_path):
            raise SystemExit(1)
        if out_path is None:
            stats["discarded"] += 1
            print(f"Dataset discarded; moved file to {processed_dir / original_path.name}")
        else:
            if has_fits:
                stats["successful"] += 1
            print(f"Calibration successful; moved original file to {processed_dir / original_path.name}")

    # ── Process each file in turn ──────────────────────────────────────────
    # Control subtraction is handled per-interval inside each file's flow
    # (the user picks/averages control intervals when defining an interval).
    for file_path in files:
        try:
            run_one(file_path)
        except SystemExit as exc:
            return int(exc.code)
        except Exception as exc:
            print(f"Failed to process {file_path}: {exc}")
            traceback.print_exc()
            return 1

    # ── Final summary ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Processing Complete")
    print(f"{'='*60}")
    print(f"Total files processed: {total_files}")
    print(f"Files discarded: {stats['discarded']}")
    print(f"Files successfully fitted: {stats['successful']}")
    print(f"Summary Excel: {summary_path}")
    print(f"{'='*60}\n")

    # Close the shared interactive window (WindowManager) now that the
    # whole run is finished. Kept open across all steps and files to avoid
    # window-recreation churn (DisplayLink driver crashes).
    try:
        from .window import finish_window
        finish_window()
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())

