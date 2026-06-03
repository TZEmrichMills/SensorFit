"""Command-line interface for calibration module."""

import argparse
import shutil
import sys
from pathlib import Path
from typing import Sequence

from .calibration import (
    apply_calibration,
    append_fit_summary,
    average_window,
    build_calibration,
    build_interval_subsets,
    CALIBRATED_COLUMN,
    calculate_turnover_before_inactivation,
    get_model_display_name,
    interactive_interval_fitting,
    load_trace,
    persist_interval_subsets,
    review_results,
    save_interval_with_fits,
    select_baseline,
    select_intervals,
    select_points,
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
    return parser.parse_args(argv)


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
) -> tuple[Path | None, Path, bool]:
    """Process a single file through a phase-based state machine.

    Phases: baseline → calibration → intervals → fitting → turnover → review → save.
    Every phase can navigate backwards to the previous phase, and the review
    screen can jump to any earlier phase.  File I/O is deferred until the
    review screen is accepted.
    """
    print(f"\nProcessing {path}")
    frame = load_trace(path, time_col_idx, current_col_idx)
    time_col = frame.columns[0]
    current_col = frame.columns[1]

    time_values = frame[time_col].to_numpy(dtype=float)
    signal_values_raw = frame[current_col].to_numpy(dtype=float)

    # ── persistent state across phases ──────────────────────────
    signal_values = None
    calibration = None
    current_calibration_values = calibration_values
    intervals = None
    subsets = None
    all_fit_results: dict[int, dict[str, dict]] = {}
    turnover_results: dict[int, float | None] = {}

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
                signal_values, (_slope, _intercept) = baseline_result
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
            )
            if result == "discard":
                print(f"\nDataset {path.name} discarded during calibration.")
                return None, path, False
            if result == "go_back_to_baseline":
                print("\nReturning to baseline selection...")
                phase = "baseline"
                continue

            indices, final_calibration_values = result
            current_calibration_values = final_calibration_values
            mean_currents = [
                average_window(signal_values, idx, window) for idx in indices
            ]
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
        # PHASE: intervals
        # ────────────────────────────────────────────────────────
        elif phase == "intervals":
            intervals = select_intervals(
                time_values,
                frame[CALIBRATED_COLUMN].to_numpy(dtype=float),
                filename=path.name,
            )
            if intervals == "discard":
                print(f"\nDataset {path.name} discarded during interval selection.")
                return None, path, False
            if intervals is None:
                print("\nReturning to calibration point selection...")
                phase = "calibration"
                continue
            if not intervals:
                print("No intervals selected; skipping subset export.")
                # Still need to write calibrated file
                calibrated_dir = output_dir / "Calibrated"
                calibrated_dir.mkdir(parents=True, exist_ok=True)
                out_path = calibrated_dir / f"{path.stem}_calibrated.xlsx"
                frame.to_excel(out_path, index=False)
                return out_path, path, False

            subsets = build_interval_subsets(frame, time_col, CALIBRATED_COLUMN, intervals)
            if not subsets:
                print("Intervals contained no data points; nothing saved.")
                return None, path, False

            phase = "fitting"
            continue

        # ────────────────────────────────────────────────────────
        # PHASE: fitting
        # ────────────────────────────────────────────────────────
        elif phase == "fitting":
            print(f"\n{'='*60}")
            print("Interactive Fitting Interface")
            print(f"{'='*60}")
            fit_result = interactive_interval_fitting(
                subsets, time_col, CALIBRATED_COLUMN, filename=path.name
            )
            if fit_result == "discard_file":
                print(f"\nDataset {path.name} discarded during fitting.")
                return None, path, False
            if fit_result == "go_back_phase":
                phase = "intervals"
                continue
            all_fit_results = fit_result

            if all_fit_results:
                print(f"\nFit results for {len(all_fit_results)} interval(s):")
                for interval_idx, models in all_fit_results.items():
                    display_names = [get_model_display_name(n) for n in models]
                    print(f"  Interval #{interval_idx}: {', '.join(display_names)}")

            phase = "turnover"
            continue

        # ────────────────────────────────────────────────────────
        # PHASE: turnover
        # ────────────────────────────────────────────────────────
        elif phase == "turnover":
            print(f"\n{'='*60}")
            print("Calculate Maximum H₂O₂ Turnover Before Inactivation")
            print(f"{'='*60}")
            turnover_result = calculate_turnover_before_inactivation(
                subsets, all_fit_results, time_col, CALIBRATED_COLUMN, filename=path.name
            )
            if turnover_result == "discard_file":
                print(f"\nDataset {path.name} discarded during turnover calculation.")
                return None, path, False
            if turnover_result == "go_back_phase":
                phase = "fitting"
                continue
            turnover_results = turnover_result

            if turnover_results:
                n_calc = len([v for v in turnover_results.values() if v is not None])
                print(f"\nTurnover results for {n_calc} interval(s):")
                for interval_idx, turnover in turnover_results.items():
                    if turnover is not None:
                        print(f"  Interval #{interval_idx}: {turnover:.3f} µM")
                    else:
                        print(f"  Interval #{interval_idx}: skipped")

            phase = "review"
            continue

        # ────────────────────────────────────────────────────────
        # PHASE: review
        # ────────────────────────────────────────────────────────
        elif phase == "review":
            decision = review_results(
                subsets, all_fit_results, turnover_results, filename=path.name
            )
            if decision == "discard":
                print(f"\nDataset {path.name} discarded at review.")
                return None, path, False
            if decision == "accept":
                phase = "save"
                continue
            # Any redo option maps directly to a phase name
            if decision in ("baseline", "calibration", "intervals", "fitting", "turnover"):
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

    print(f"\nSaving interval data and fits...")
    for subset in subsets:
        interval_fits = all_fit_results.get(subset.index, None)
        turnover_uM = turnover_results.get(subset.index, None)
        excel_path = save_interval_with_fits(
            subset, time_col, CALIBRATED_COLUMN, interval_fits, interval_dir
        )
        print(f"  Saved: {excel_path.name}")
        append_fit_summary(
            summary_path=summary_path,
            source_file=path.name,
            interval_index=subset.index,
            start_time=subset.start_time,
            end_time=subset.end_time,
            fit_results=interval_fits,
            turnover_uM=turnover_uM,
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
    
    # Parse calibration values
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

    # Track statistics
    total_files = len(files)
    discarded_count = 0
    successfully_fit_count = 0

    for file_path in files:
        try:
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
            )
            # Move processed file to Processed folder (whether successful or discarded)
            processed_path = processed_dir / original_path.name
            if processed_path.exists():
                print(f"Warning: {processed_path} already exists; skipping move.")
            else:
                # Use pathlib for cross-platform path handling
                # Convert to string for shutil.move which handles both platforms
                try:
                    shutil.move(str(original_path), str(processed_path))
                except OSError as e:
                    # Handle Windows file locking issues
                    if sys.platform == "win32" and "being used by another process" in str(e):
                        print(f"Warning: File {original_path.name} is locked. Please close it and try again.")
                        return 1
                    raise
                if out_path is None:
                    discarded_count += 1
                    print(f"Dataset discarded; moved file to {processed_path}")
                else:
                    if has_fits:
                        successfully_fit_count += 1
                    print(f"Calibration successful; moved original file to {processed_path}")
        except Exception as exc:
            print(f"Failed to process {file_path}: {exc}")
            import traceback
            traceback.print_exc()
            return 1
    
    # Display final summary
    print(f"\n{'='*60}")
    print("Processing Complete")
    print(f"{'='*60}")
    print(f"Total files processed: {total_files}")
    print(f"Files discarded: {discarded_count}")
    print(f"Files successfully fitted: {successfully_fit_count}")
    print(f"Summary Excel: {summary_path}")
    print(f"{'='*60}\n")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

