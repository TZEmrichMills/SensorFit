"""Command-line interface for calibration module."""

import argparse
import shutil
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

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
from .controls import (
    ControlGroup,
    interactive_subtract,
    load_control_template,
    load_controls_manifest,
    save_control_template,
    save_controls_manifest,
    select_control_reference_interval,
    show_grouping_ui,
)
from .group_planning import (
    SampleState,
    offer_refit_for_corrected_intervals,
    run_group_planning,
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
        "--control-mode",
        action="store_true",
        help=(
            "Open the control-grouping UI at session start so a 'no-enzyme' (or other) "
            "control trace can be subtracted from selected samples.  Auto-enabled if "
            "Calibrated/controls.json already exists from a prior session."
        ),
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


def _legacy_template_filename(group: "ControlGroup | None") -> str | None:
    """Single-subgroup, single-control legacy shim.

    For groups built before the multi-control restructure (or for new groups
    that happen to contain exactly one sub-group with one control) we still
    use the simple inline `control_subtract` phase inside ``process_file``.
    This helper extracts the one template name; returns None if the group
    has zero or multiple controls.
    """
    if group is None or not group.subgroups:
        return None
    if len(group.subgroups) != 1 or len(group.subgroups[0].controls) != 1:
        return None
    return group.subgroups[0].controls[0].template_filename


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
    control_role: str | None = None,
    control_group: "ControlGroup | None" = None,
    calibrated_dir: Path | None = None,
    skip_calibration: bool = False,
) -> tuple[Path | None, Path, bool, "dict | None"]:
    """Process a single file through a phase-based state machine.

    Phases: baseline → calibration → [control_subtract] → intervals → fitting →
    turnover → review → save.  Every phase can navigate backwards to the
    previous phase, and the review screen can jump to any earlier phase.  File
    I/O is deferred until the review screen is accepted.

    Parameters
    ----------
    control_role : "control" | "sample" | None
        If "control", after interval selection the user is asked which interval
        is the subtraction reference, and the reference is saved as a control
        template.  If "sample", after calibration the user is shown the control
        template aligned to the sample and can choose to subtract.  None means
        the file is processed normally.
    control_group : ControlGroup | None
        Required when control_role is set; carries the group name, the control
        file, and (for samples) the path to the already-saved control template.
    calibrated_dir : Path | None
        Where the Calibrated/ folder lives.  Required for control templates.
    skip_calibration : bool
        If True, treat ``current_col_idx`` as the already-calibrated [H₂O₂]
        column (µM).  Baseline and calibration phases are skipped entirely
        and the values are copied directly into the calibrated trace.  Use
        when SensorFit is run purely as a fitting tool against pre-processed
        data (typically from another pipeline).

    Returns
    -------
    (out_path, original_path, has_fits, control_info)
        ``control_info`` is None for ungrouped files; for samples it is
        ``{"subtracted": bool, "group": str}``; for controls it carries the
        chosen reference interval index.
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
    intervals = None
    subsets = None
    all_fit_results: dict[int, dict[str, dict]] = {}
    turnover_results: dict[int, float | None] = {}
    control_subtracted = False
    control_info: dict | None = None

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
                    return None, path, False, control_info, None
                if baseline_result == "redraw":
                    continue
                # baseline_result is (corrected_signal, meta) where meta is
                # either (slope, intercept) for line mode or a dict for curve.
                # We only need the corrected signal here; the metadata isn't
                # persisted anywhere downstream yet.
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
            )
            if result == "discard":
                print(f"\nDataset {path.name} discarded during calibration.")
                return None, path, False, control_info, None
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

            # Branch to control-subtract phase for sample files whose group
            # already has a saved template.  Controls and ungrouped files
            # bypass straight to interval selection.
            # NOTE: the old single-control subtraction phase is retained for
            # legacy single-control single-subgroup groups (so anyone running
            # with a controls.json built before the multi-control restructure
            # still works the same way).  Groups with >=2 controls or >=2
            # sub-groups go through the new GROUP-LEVEL planning phase after
            # all files in the group are processed, not through this
            # inline-per-file path.
            _legacy_single_ctrl_template = _legacy_template_filename(control_group)
            if (
                control_role == "sample"
                and control_group is not None
                and _legacy_single_ctrl_template is not None
                and len(control_group.subgroups) == 1
                and len(control_group.subgroups[0].controls) == 1
            ):
                phase = "control_subtract"
            else:
                phase = "intervals"
            continue

        # ────────────────────────────────────────────────────────
        # PHASE: control_subtract  (samples only, after calibration)
        # ────────────────────────────────────────────────────────
        elif phase == "control_subtract":
            assert control_group is not None and calibrated_dir is not None
            tpl_name = _legacy_template_filename(control_group)
            try:
                ct, cy = load_control_template(calibrated_dir, tpl_name)
            except Exception as exc:
                print(
                    f"Warning: could not load control template "
                    f"{tpl_name}: {exc}.  Skipping subtraction."
                )
                phase = "intervals"
                continue
            result = interactive_subtract(
                sample_time=time_values,
                sample_h2o2=frame[CALIBRATED_COLUMN].to_numpy(dtype=float),
                control_t=ct,
                control_y=cy,
                filename=path.name,
                group_name=control_group.name,
            )
            if isinstance(result, str) and result == "back":
                phase = "calibration"
                continue
            if isinstance(result, str) and result == "skip":
                print("Subtraction skipped by user.")
                control_subtracted = False
                phase = "intervals"
                continue
            # result is np.ndarray
            frame[CALIBRATED_COLUMN] = result
            control_subtracted = True
            print(
                f"Control template '{control_group.name}' subtracted from "
                f"{path.name} ({len(result)} samples)."
            )
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
                return None, path, False, control_info, None
            if intervals is None:
                if skip_calibration:
                    # No calibration phase to return to — re-open intervals.
                    print("\nNothing to go back to in skip-calibration mode; reopening interval selection.")
                    continue
                print("\nReturning to calibration point selection...")
                phase = "calibration"
                continue
            if not intervals:
                print("No intervals selected; skipping subset export.")
                # Still need to write calibrated file
                calibrated_dir_local = output_dir / "Calibrated"
                calibrated_dir_local.mkdir(parents=True, exist_ok=True)
                out_path = calibrated_dir_local / f"{path.stem}_calibrated.xlsx"
                frame.to_excel(out_path, index=False)
                return out_path, path, False, control_info, None

            subsets = build_interval_subsets(frame, time_col, CALIBRATED_COLUMN, intervals)
            if not subsets:
                print("Intervals contained no data points; nothing saved.")
                return None, path, False, control_info, None

            # For control files: pick the reference interval and save the
            # template to disk so subsequent samples can subtract it.  The
            # template is recorded against the specific ControlSpec within
            # the group's sub-group structure.
            if control_role == "control" and control_group is not None:
                lookup = control_group.find_spec(path.name)
                if lookup is None:
                    # Shouldn't happen — main() puts the file in the group
                    print(f"Warning: control {path.name} not found in group '{control_group.name}'; skipping reference selection.")
                else:
                    subgroup_idx, spec = lookup
                    chosen = select_control_reference_interval(
                        subsets, time_col, filename=path.name
                    )
                    if chosen is None:
                        print("No reference interval chosen — control will not be subtracted from samples.")
                        spec.reference_interval_index = None
                        spec.template_filename = None
                    else:
                        ref = next((s for s in subsets if s.index == chosen), subsets[0])
                        if calibrated_dir is None:
                            calibrated_dir_local = output_dir / "Calibrated"
                        else:
                            calibrated_dir_local = calibrated_dir
                        # Disambiguate templates with the same file appearing
                        # in different groups by prefixing with group name.
                        template_key = f"{control_group.name}__sg{subgroup_idx}__{Path(path.stem).name}"
                        tpl_path = save_control_template(
                            calibrated_dir_local,
                            template_key,
                            ref.data[time_col].to_numpy(dtype=float),
                            ref.data[CALIBRATED_COLUMN].to_numpy(dtype=float),
                        )
                        spec.reference_interval_index = int(ref.index)
                        spec.template_filename = tpl_path.name
                        control_info = {
                            "role": "control",
                            "group": control_group.name,
                            "subgroup_index": subgroup_idx,
                            "reference_interval_index": int(ref.index),
                            "template_filename": tpl_path.name,
                        }
                        print(
                            f"Saved control template for {control_group.name}/"
                            f"{control_group.subgroups[subgroup_idx].name}/{path.name} "
                            f"(interval #{ref.index}) → {tpl_path}"
                        )

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
                return None, path, False, control_info, None
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
                return None, path, False, control_info, None
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
                subsets, all_fit_results, turnover_results,
                filename=path.name,
                skip_calibration=skip_calibration,
            )
            if decision == "discard":
                print(f"\nDataset {path.name} discarded at review.")
                return None, path, False, control_info, None
            if decision == "accept":
                phase = "save"
                continue
            # Any redo option maps directly to a phase name.  In skip-
            # calibration mode, baseline / calibration are not available;
            # but defend in depth in case some other entry point passes one.
            if decision in ("baseline", "calibration") and skip_calibration:
                print(f"  (baseline/calibration not available in skip-calibration mode; staying at review)")
                continue
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
            control_subtracted=(
                control_subtracted if control_role == "sample" else None
            ),
            control_group=(
                control_group.name
                if (control_role == "sample" and control_subtracted and control_group is not None)
                else None
            ),
            calibration_skipped=True if skip_calibration else None,
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

    # Build a control_info payload for sample files so main() can log
    if control_role == "sample" and control_group is not None:
        control_info = {
            "role": "sample",
            "group": control_group.name,
            "subtracted": bool(control_subtracted),
        }

    # Build a SampleState payload so the group orchestration can run
    # planning + optional re-fit later.  Only meaningful for grouped sample
    # files; controls and ungrouped files get None here.
    sample_state: SampleState | None = None
    if control_role == "sample" and control_group is not None and subsets:
        sample_state = SampleState(
            file_name=path.name,
            calibrated_frame=frame.copy(),
            time_col=time_col,
            subsets=list(subsets),
            fit_results=dict(all_fit_results),
            turnover_results=dict(turnover_results),
        )

    return out_path, path, has_fits, control_info, sample_state


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

    # ── Control-mode setup ────────────────────────────────────────────
    # Auto-enable control mode if a manifest already exists from a prior run.
    existing_groups = load_controls_manifest(calibrated_out_dir)
    control_mode = args.control_mode or (existing_groups is not None)
    groups: dict[str, ControlGroup] = {}
    role_of: dict[str, str] = {}       # filename → "control" | "sample"
    group_of: dict[str, ControlGroup] = {}  # filename → its ControlGroup

    if control_mode:
        print("\nControl-mode is ON.  Opening grouping UI...")
        ui_result = show_grouping_ui(files, existing_groups=existing_groups)
        if ui_result is None:
            print("Grouping cancelled.  No subtraction will be performed.")
        else:
            groups = ui_result
            # Carry over already-saved templates from existing_groups so a
            # session can resume without re-processing controls.  Match by
            # group name + control file name + sub-group index.
            if existing_groups:
                for name, g in groups.items():
                    if name not in existing_groups:
                        continue
                    old_group = existing_groups[name]
                    for sg_idx, sg in enumerate(g.subgroups):
                        if sg_idx >= len(old_group.subgroups):
                            break
                        old_sg = old_group.subgroups[sg_idx]
                        for spec in sg.controls:
                            old_spec = next(
                                (c for c in old_sg.controls if c.file_name == spec.file_name),
                                None,
                            )
                            if old_spec and old_spec.template_filename:
                                spec.template_filename = old_spec.template_filename
                                spec.reference_interval_index = old_spec.reference_interval_index
            save_controls_manifest(groups, calibrated_out_dir)
            for g in groups.values():
                for ctrl_name in g.all_control_files():
                    role_of[ctrl_name] = "control"
                    group_of[ctrl_name] = g
                for s in g.sample_files:
                    role_of[s] = "sample"
                    group_of[s] = g
            print(f"  {len(groups)} group(s) defined.  Manifest saved to {calibrated_out_dir / 'controls.json'}")

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

    def run_one(file_path: Path, role: str | None, grp: ControlGroup | None) -> SampleState | None:
        """Call process_file and handle the post-processing housekeeping.
        Returns the SampleState if available; else None."""
        out_path, original_path, has_fits, control_info, sample_state = process_file(
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
            control_role=role,
            control_group=grp,
            calibrated_dir=calibrated_out_dir,
            skip_calibration=args.skip_calibration,
        )
        # Persist manifest after each control so an interrupted session
        # can resume with saved templates intact.
        if role == "control" and groups:
            save_controls_manifest(groups, calibrated_out_dir)
        if not _move_to_processed(original_path):
            raise SystemExit(1)
        if out_path is None:
            stats["discarded"] += 1
            print(f"Dataset discarded; moved file to {processed_dir / original_path.name}")
        else:
            if has_fits:
                stats["successful"] += 1
            print(f"Calibration successful; moved original file to {processed_dir / original_path.name}")
        return sample_state

    # ── Group-by-group orchestration ──────────────────────────────────────
    # Process each group as a unit:
    #   1) all controls (templates saved as we go)
    #   2) all samples (state accumulated)
    #   3) per-group subtraction-planning UI
    #   4) optional re-fit + re-turnover on corrected intervals
    #   5) write "corrected" rows to fit_summary + save corrected interval files
    # Then process any ungrouped files normally.
    by_name: dict[str, Path] = {p.name: p for p in files}

    for group in groups.values():
        print(f"\n{'#' * 60}")
        print(f"Group '{group.name}' — {len(group.all_control_files())} control(s) "
              f"in {len(group.subgroups)} sub-group(s); "
              f"{len(group.sample_files)} sample(s)")
        print(f"{'#' * 60}")

        # 1) Process controls
        for ctrl_name in group.all_control_files():
            ctrl_path = by_name.get(ctrl_name)
            if ctrl_path is None:
                print(f"Skipping control '{ctrl_name}' — not found in input directory.")
                continue
            try:
                run_one(ctrl_path, "control", group)
            except SystemExit as exc:
                return int(exc.code)
            except Exception as exc:
                print(f"Failed to process control {ctrl_path}: {exc}")
                import traceback
                traceback.print_exc()
                return 1

        # 2) Process samples and accumulate SampleState
        sample_states: list[SampleState] = []
        for sample_name in group.sample_files:
            sample_path = by_name.get(sample_name)
            if sample_path is None:
                print(f"Skipping sample '{sample_name}' — not found in input directory.")
                continue
            try:
                state = run_one(sample_path, "sample", group)
                if state is not None:
                    sample_states.append(state)
            except SystemExit as exc:
                return int(exc.code)
            except Exception as exc:
                print(f"Failed to process sample {sample_path}: {exc}")
                import traceback
                traceback.print_exc()
                return 1

        # 3) + 4) Planning + optional re-fit (only meaningful for multi-control
        # or multi-subgroup groups; legacy single-control already did inline
        # subtraction inside process_file)
        is_legacy_single = (
            len(group.subgroups) == 1
            and len(group.subgroups[0].controls) == 1
        )
        if sample_states and not is_legacy_single:
            updated = run_group_planning(group, sample_states, calibrated_out_dir)
            offer_refit_for_corrected_intervals(
                updated,
                interactive_interval_fitting,
                calculate_turnover_before_inactivation,
            )
            # 5) Persist corrected outputs
            _persist_group_corrected_outputs(
                group, updated, calibrated_out_dir, summary_path
            )

    # 6) Process any ungrouped files normally
    for file_path in files:
        if file_path.name in role_of:
            continue
        try:
            run_one(file_path, None, None)
        except SystemExit as exc:
            return int(exc.code)
        except Exception as exc:
            print(f"Failed to process {file_path}: {exc}")
            import traceback
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

    return 0


def _persist_group_corrected_outputs(
    group: ControlGroup,
    sample_states: dict[str, "SampleState"],
    calibrated_dir: Path,
    summary_path: Path,
) -> None:
    """Save corrected interval excels and append `variant=corrected` rows to
    fit_summary.  Pre-subtraction `variant=original` rows were already written
    by process_file's save phase.
    """
    for state in sample_states.values():
        if not state.corrected_subsets:
            continue
        interval_dir = calibrated_dir / f"{Path(state.file_name).stem}_intervals"
        interval_dir.mkdir(parents=True, exist_ok=True)
        for idx, corrected in state.corrected_subsets.items():
            # Per-interval corrected Excel
            out_path = interval_dir / f"interval_{idx:02d}_corrected.xlsx"
            corrected.data.to_excel(out_path, index=False)
            print(f"  Saved corrected interval → {out_path.name}")

            # Choose post-subtraction fits/turnover where available
            post_fits = state.refit_results.get(idx, None)
            post_turn = state.refit_turnover.get(idx, None)
            append_fit_summary(
                summary_path=summary_path,
                source_file=state.file_name,
                interval_index=idx,
                start_time=corrected.start_time,
                end_time=corrected.end_time,
                fit_results=post_fits if post_fits else None,
                turnover_uM=post_turn,
                control_subtracted=True,
                control_group=group.name,
                correction_meta=state.correction_meta.get(idx),
                variant="corrected",
            )


if __name__ == "__main__":
    import sys
    sys.exit(main())

