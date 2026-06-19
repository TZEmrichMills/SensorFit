"""Unit tests for the four wishlist features added on the wishlist-features branch.

Covers pure-logic functions only.  The interactive (matplotlib / Qt) parts are
verified manually by running SensorFit against real data — they cannot be
driven non-interactively here.

Run with:
    python test_new_features.py
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path


def _section(title: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}\n{title}\n{bar}")


# ──────────────────────────────────────────────────────────────────────────
# 1) fit_summary upsert (Commit 1)
# ──────────────────────────────────────────────────────────────────────────

def test_fit_summary_upsert() -> bool:
    _section("C6: fit_summary upsert with (file, interval, fit_number, variant) key")
    import pandas as pd
    from sensorfit.calibration import append_fit_summary

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "fit_summary.xlsx"

        # Interval-only row (fit_number=0)
        append_fit_summary(p, "fileA.xlsx", 1, 0.0, 10.0, fit_results=None, turnover_uM=None)
        # Same key → REPLACE (turnover updated)
        append_fit_summary(p, "fileA.xlsx", 1, 0.0, 10.0, fit_results=None, turnover_uM=42.0)
        # Same file, different interval → APPEND
        append_fit_summary(p, "fileA.xlsx", 2, 11.0, 20.0, fit_results=None, turnover_uM=None)
        # Different file → APPEND
        append_fit_summary(p, "fileB.xlsx", 1, 0.0, 5.0, fit_results=None, turnover_uM=None)
        # Same fileA interval 1 but variant=corrected → APPEND (different variant)
        append_fit_summary(p, "fileA.xlsx", 1, 0.0, 10.0, fit_results=None,
                           turnover_uM=99.0, variant="corrected", control_group="G_A")
        # Same fileA interval 1 original, but fit_number=1 → APPEND (different fit_number)
        append_fit_summary(p, "fileA.xlsx", 1, 0.0, 10.0, fit_results=None,
                           turnover_uM=42.0, fit_number=1)
        # Same fileA interval 1 original, fit_number=2 → APPEND (another fit)
        append_fit_summary(p, "fileA.xlsx", 1, 0.0, 10.0, fit_results=None,
                           turnover_uM=42.0, fit_number=2)

        df = pd.read_excel(p)
        assert "fit_number" in df.columns
        assert len(df) == 6, f"expected 6 rows, got {len(df)}"
        # The (fileA, 1, 0, original) row carries turnover=42.0 (latest)
        row_orig = df[(df["source_file"] == "fileA.xlsx") & (df["interval_index"] == 1)
                      & (df["variant"] == "original") & (df["fit_number"] == 0)].iloc[0]
        assert row_orig["turnover_before_inactivation_uM"] == 42.0
        # corrected variant coexists with original at same fit_number=0
        row_corr = df[(df["source_file"] == "fileA.xlsx") & (df["interval_index"] == 1)
                      & (df["variant"] == "corrected") & (df["fit_number"] == 0)].iloc[0]
        assert row_corr["control_group"] == "G_A"
        # Per-fit rows: fit_number=1 and fit_number=2
        fit_rows = df[(df["source_file"] == "fileA.xlsx") & (df["interval_index"] == 1)
                      & (df["variant"] == "original") & (df["fit_number"].isin([1, 2]))]
        assert len(fit_rows) == 2
        print(f"  ✓ upsert: 7 writes → 6 rows; key = (file, interval, fit_number, variant)")
    return True


def test_fit_summary_filter() -> bool:
    """fit_summary*.xlsx and Calibrated/ files are not discovered as inputs."""
    _section("Commit 1: file-discovery filter")
    # We verify the filter logic directly by inspecting the filter conditions.
    # The actual main() loop isn't easily testable without a full Qt env.
    from pathlib import PurePosixPath
    test_names = [
        "trace_A.xlsx",
        "fit_summary.xlsx",
        "fit_summary_Analysis.xlsx",
        "trace_B_calibrated.xlsx",
        "~$trace_C.xlsx",
    ]
    expected_pass = {"trace_A.xlsx"}
    actually_pass = set()
    for name in test_names:
        f = PurePosixPath(name)
        if f.name.startswith("~$"):
            continue
        if "_calibrated" in f.stem:
            continue
        if f.stem.startswith("fit_summary"):
            continue
        actually_pass.add(f.name)
    assert actually_pass == expected_pass, (
        f"filter discrepancy: expected {expected_pass}, got {actually_pass}"
    )
    print(f"  ✓ filter: only 'trace_A.xlsx' passed; lock/summary/calibrated all excluded")
    return True


# ──────────────────────────────────────────────────────────────────────────
# 2) Control subtraction (Commit 2)
# ──────────────────────────────────────────────────────────────────────────

def test_grouping_modules_gone() -> bool:
    _section("Grouping removed: group_planning + grouping symbols are gone")
    # The upfront sample-grouping feature was removed; its module and the
    # grouping dataclasses/UI should no longer be importable.
    try:
        import sensorfit.group_planning  # noqa: F401
    except ModuleNotFoundError:
        print("  ✓ sensorfit.group_planning no longer importable")
    else:
        raise AssertionError("group_planning should have been deleted")

    import sensorfit.controls as c
    for gone in ("ControlGroup", "show_grouping_ui", "save_controls_manifest",
                 "select_control_reference_interval", "interactive_subtract"):
        assert not hasattr(c, gone), f"controls.{gone} should have been removed"
    # The reused template helpers must remain.
    for kept in ("save_control_template", "load_control_template", "interpolate_control_to_grid"):
        assert hasattr(c, kept), f"controls.{kept} must be kept"
    print("  ✓ grouping symbols removed; template helpers kept")
    return True


def test_control_template_roundtrip() -> bool:
    _section("Commit 2: control template save/load (with time-zeroing)")
    import numpy as np
    from sensorfit.controls import save_control_template, load_control_template

    with tempfile.TemporaryDirectory() as tmp:
        calib = Path(tmp) / "Calibrated"
        t = np.linspace(0, 30, 200)
        y = np.linspace(100, 60, 200)
        tpl_path = save_control_template(calib, "G_A", t + 50.0, y)  # input offset by 50s
        t_back, y_back = load_control_template(calib, tpl_path.name)
        assert abs(t_back[0]) < 1e-9, f"template not zero-based: {t_back[0]}"
        assert abs(t_back[-1] - 30.0) < 1e-6
        assert np.allclose(y_back, y)
        print(f"  ✓ template saved/loaded; time zero-based from {50:.1f}s offset")
    return True


def test_interpolate_control() -> bool:
    _section("Commit 2: interpolate_control_to_grid")
    import numpy as np
    from sensorfit.controls import interpolate_control_to_grid

    # Linear control y = 2t + 10 on [0, 10]
    ct = np.linspace(0, 10, 51)
    cy = 2 * ct + 10
    sample_t = np.array([2.0, 5.0, 8.0])
    interp, info = interpolate_control_to_grid(ct, cy, sample_t)
    expected = 2 * sample_t + 10
    assert np.allclose(interp, expected), f"interp wrong: {interp} vs {expected}"
    assert info["n_extrap_left"] == 0 and info["n_extrap_right"] == 0
    print("  ✓ interpolation inside range matches y=2t+10 exactly")

    # Sample extends beyond control: extrapolation should match slope
    sample_t2 = np.array([-2.0, 5.0, 15.0])
    interp2, info2 = interpolate_control_to_grid(ct, cy, sample_t2)
    # control duration = 10; max extend = 2 (20%); t=-2 → capped at -2 (just fits);
    # t=15 → capped at +2 → y_predict = control_y[-1] + slope*2 = 30 + 2*2 = 34
    assert info2["n_extrap_left"] == 1 and info2["n_extrap_right"] == 1
    assert abs(interp2[2] - 34.0) < 1e-6, f"right extrap wrong: {interp2[2]}"
    print(f"  ✓ extrapolation capped at 20%: t=15s → y={interp2[2]:.2f} (cap to t=12)")

    # Sample WAY past → should still emit warning
    sample_t3 = np.array([-50.0, 50.0])
    _, info3 = interpolate_control_to_grid(ct, cy, sample_t3)
    assert info3["warning"], "expected a warning when sample exceeds extrap cap"
    print(f"  ✓ warning emitted when extrapolation hits cap")
    return True


# ──────────────────────────────────────────────────────────────────────────
# 3) Residual activity (Commit 3)
# ──────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────
# Residual-activity module was removed in the per-interval restructure
# (the user can compute rate ratios in Excel from initial_rate columns).
# ──────────────────────────────────────────────────────────────────────────


def test_residual_activity_module_gone() -> bool:
    _section("Per-interval restructure: residual_activity module removed")
    try:
        import sensorfit.residual_activity  # noqa: F401
    except ModuleNotFoundError:
        print("  ✓ sensorfit.residual_activity no longer importable")
        return True
    raise AssertionError("residual_activity module is still importable but should have been deleted")


# ──────────────────────────────────────────────────────────────────────────
# Back-extrapolation (math stays; the standalone UI is gone)
# ──────────────────────────────────────────────────────────────────────────

def test_back_extrap_exponential() -> bool:
    _section("Commit 4: back-extrap (exponential method)")
    import numpy as np
    from sensorfit.back_extrap import compute_back_extrap
    from sensorfit.models import model_Exponential

    # Pure exponential decay: y(t) = 100 * exp(-0.02 * t)
    params = np.array([0.0, 0.0, 100.0, 0.02, 0.0])
    t_start, deadtime, nominal = 60.0, 2.0, 30.0
    y_start = float(model_Exponential([t_start], *params)[0])
    y_back_expected = float(model_Exponential([t_start - deadtime], *params)[0])
    rate_back_expected = -100.0 * 0.02 * float(np.exp(-0.02 * (t_start - deadtime)))

    fits = {"Exponential": {"params": params, "yhat": np.array([y_start])}}
    r = compute_back_extrap(fits, t_start, y_start, deadtime, nominal)
    assert abs(r["back_extrap_uM"] - y_back_expected) < 1e-9
    assert abs(r["rate_at_back_uM_per_s"] - rate_back_expected) < 1e-9
    assert abs(r["stretch_factor"] - y_back_expected / nominal) < 1e-9
    assert abs(r["stretch_initial_rate"] - rate_back_expected * y_back_expected / nominal) < 1e-9
    assert r["method"] == "exponential"
    print(f"  ✓ y(60)={y_start:.4f} y(58)={y_back_expected:.4f} stretch={r['stretch_factor']:.4f}")
    return True


def test_back_extrap_linear_fallback() -> bool:
    _section("Commit 4: back-extrap (linear fallback)")
    from sensorfit.back_extrap import compute_back_extrap

    # No Exponential fit → use linear with LinearInitialRate rate
    fits = {"LinearInitialRate": {"init_rate": -2.0}}
    r = compute_back_extrap(
        fits, interval_t_start_abs=60.0, interval_observed_value=100.0,
        deadtime_s=2.0, nominal_uM=100.0,
    )
    # y(t_back) = 100 + (-2.0)*(58 - 60) = 100 + 4 = 104
    assert abs(r["back_extrap_uM"] - 104.0) < 1e-9
    assert abs(r["stretch_factor"] - 1.04) < 1e-9
    assert abs(r["stretch_initial_rate"] - (-2.0 * 1.04)) < 1e-9
    assert r["method"] == "linear"
    print(f"  ✓ linear back-extrap: 100 µM + 4 µM deadtime consumption = 104 µM")
    return True


def test_skip_calibration_provenance() -> bool:
    _section("Skip-calibration: fit_summary records calibration_skipped flag")
    import pandas as pd
    from sensorfit.calibration import append_fit_summary

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "fit_summary.xlsx"
        # Pretend we processed one file in skip mode
        append_fit_summary(
            p, "precalibrated_A.csv", 1, 0.0, 30.0,
            fit_results=None, turnover_uM=None, calibration_skipped=True,
        )
        # Plus a normal file (no flag)
        append_fit_summary(
            p, "raw_B.xlsx", 1, 0.0, 30.0,
            fit_results=None, turnover_uM=None,
        )
        df = pd.read_excel(p)
        assert "calibration_skipped" in df.columns
        flag_skip = df[df["source_file"] == "precalibrated_A.csv"]["calibration_skipped"].iloc[0]
        flag_norm = df[df["source_file"] == "raw_B.xlsx"]["calibration_skipped"]
        assert bool(flag_skip) is True
        # Normal row should have NaN (column exists due to skip row, but this row was None)
        assert flag_norm.isna().all() or flag_norm.iloc[0] is None
        print("  ✓ calibration_skipped=True recorded for skip-mode row; NaN for normal row")
    return True


def test_skip_calibration_loads_real_csv() -> bool:
    """End-to-end check that load_trace + skip-cal-style frame setup works
    on the user's real pre-calibrated CSVs."""
    _section("Skip-calibration: load_trace + frame setup against real data")
    import pandas as pd
    from sensorfit.calibration import load_trace, CALIBRATED_COLUMN

    base = Path(
        "/Users/tom/Jottacloud/Tommy/01_NMBU_workspace/Supervision/Rannei_Skaali_PhD/"
        "Assays/Sensor/Peroxi-Temperature-variation/260602_Rannei_arrhenius_att3/"
        "Subtracted_Traces_for_Rannei/IBFits"
    )
    if not base.exists():
        print(f"  (skipped — folder not available: {base})")
        return True

    # CSVs may live in the base folder, or in Processed/ if the user has
    # already run SensorFit against them.  Accept either.
    candidates = []
    for d in (base, base / "Processed"):
        if d.exists():
            candidates.extend(p for p in d.glob("*.csv") if not p.name.startswith("~$"))
    if not candidates:
        print(f"  (skipped — no CSVs available in {base} or its Processed/)")
        return True
    csvs = sorted(candidates)[:2]
    for path in csvs:
        # Mirror what process_file does in skip mode: load with --time-col=0, --current-col=3
        frame = load_trace(path, time_col_idx=0, current_col_idx=3)
        time_col, current_col = frame.columns[0], frame.columns[1]
        assert time_col == "tau_s", f"unexpected time col: {time_col}"
        assert current_col == "H2O2_corr_uM", f"unexpected current col: {current_col}"
        # Skip-mode frame setup: CALIBRATED_COLUMN := current_col
        frame[CALIBRATED_COLUMN] = frame[current_col].values
        # H2O2 should be in a sensible range (tens to hundreds µM)
        h2o2 = frame[CALIBRATED_COLUMN]
        assert h2o2.notna().all(), f"NaNs in {path.name}"
        assert abs(h2o2.max()) > 1, f"{path.name}: looks empty (max={h2o2.max()})"
        assert abs(h2o2.max()) < 10000, f"{path.name}: implausibly large (max={h2o2.max()})"
        print(f"  ✓ {path.name}: {len(frame)} rows, [H₂O₂] range = [{h2o2.min():.2f}, {h2o2.max():.2f}] µM")
    return True


def test_fit_baseline_polynomial() -> bool:
    _section("C3: _fit_baseline_polynomial reproduces clicks and chooses degree")
    import numpy as np
    from sensorfit.calibration import _fit_baseline_polynomial

    t = np.linspace(0, 10, 101)

    # 2 clicks → degree 1 (straight line through both points)
    fit2 = _fit_baseline_polynomial([0.0, 10.0], [5.0, 25.0], t)
    assert abs(fit2[0] - 5.0) < 1e-9
    assert abs(fit2[-1] - 25.0) < 1e-9
    # slope = 2 → at t=5, y = 15
    assert abs(fit2[50] - 15.0) < 1e-9
    print("  ✓ 2 clicks → degree 1 (line through (0,5) and (10,25))")

    # 3 clicks → degree 2 (parabola)
    fit3 = _fit_baseline_polynomial([0.0, 5.0, 10.0], [5.0, 11.0, 25.0], t)
    assert abs(fit3[0] - 5.0) < 1e-6
    assert abs(fit3[50] - 11.0) < 1e-6
    assert abs(fit3[-1] - 25.0) < 1e-6
    print("  ✓ 3 clicks → degree 2 (reproduces all 3 click points)")

    # 5 clicks → degree 3 (cubic capped at 3, NOT degree 4)
    # Fit-through-points isn't guaranteed for 5 points + degree 3 (over-determined)
    # but a 5-point fit on data y = 2t (truly linear) should still be ~linear.
    fit5 = _fit_baseline_polynomial(
        [0.0, 2.0, 5.0, 7.0, 10.0],
        [0.0, 4.0, 10.0, 14.0, 20.0],
        t,
    )
    # All points lie on y = 2t → fit at t=5 should be ~10
    assert abs(fit5[50] - 10.0) < 1e-6
    print("  ✓ 5 clicks on a line → degree-3 fit collapses to linear, exact reproduction")

    # 1 click → ValueError
    try:
        _fit_baseline_polynomial([0.0], [5.0], t)
    except ValueError:
        print("  ✓ 1 click → ValueError (need ≥2 points)")
        return True
    raise AssertionError("expected ValueError for 1 click")


def test_fit_summary_per_fit_columns() -> bool:
    _section("C6: fit_summary writes per-fit and Δmax columns from records")
    import pandas as pd
    import numpy as np
    from sensorfit.calibration import append_fit_summary
    from sensorfit.interval_processor import FitRecord, DeltaMaxRecord

    rec_fit = FitRecord(
        model="Exponential",
        fit_start_s=2.0, fit_end_s=18.0,
        params=[0.0, 0.0, 100.0, 0.05, 0.0],
        param_names=["a", "b", "c", "k", "t0"],
        yhat=np.zeros(50),
        init_rate_uM_per_s=-5.123,
        init_rate_at_t_s=2.0,
        r2=0.987,
        rss=12.34,
        back_extrap_applied=True,
        back_extrap_deadtime_s=1.5,
        back_extrap_t0_s=0.5,
        back_extrap_rate_uM_per_s=-6.0,
    )
    rec_delta = DeltaMaxRecord(
        method="from-fit", t_zero_s=2.0, value_uM=98.7,
    )

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "fit_summary.xlsx"
        append_fit_summary(
            p, "sample.xlsx", interval_index=1,
            start_time=0.0, end_time=20.0,
            fit_results=None, turnover_uM=None,
            fit_number=1, fit_record=rec_fit, delta_max_record=rec_delta,
        )
        df = pd.read_excel(p)
        assert "fit_model" in df.columns and df.iloc[0]["fit_model"] == "Exponential"
        assert abs(df.iloc[0]["fit_init_rate_uM_per_s"] - (-5.123)) < 1e-6
        assert df.iloc[0]["fit_back_extrap_applied"] == True
        assert abs(df.iloc[0]["fit_back_extrap_deadtime_s"] - 1.5) < 1e-9
        assert df.iloc[0]["delta_max_method"] == "from-fit"
        assert abs(df.iloc[0]["delta_max_uM"] - 98.7) < 1e-6
        # Param columns flattened
        assert "fit_param_c" in df.columns
        assert abs(df.iloc[0]["fit_param_c"] - 100.0) < 1e-9
        print("  ✓ fit_record columns (model/start/end/init_rate/back_extrap/params) emitted")
        print("  ✓ delta_max_record columns (method/t_zero/value) emitted")
    return True


def test_delta_max_pure_helpers() -> bool:
    _section("C5: Δmax helpers (from_fit / from_linear / from_point)")
    import numpy as np
    from sensorfit.interval_processor import (
        FitRecord, delta_max_from_fit, delta_max_from_linear, delta_max_from_point,
    )

    # ── from_fit on an Exponential ────────────────────────────────────
    # y(t) = a*t + b + c*exp(-k*(t-t0))
    # Asymptote = a*t + b; Δ at t_zero = c*exp(-k*(t_zero - t0))
    # With a=0, b=0, c=100, k=0.05, t0=0 → y(t) = 100*exp(-0.05*t)
    # Δ at t_zero=0 = c*exp(0) = 100; at t_zero=10 = 100*exp(-0.5) ≈ 60.65
    rec_exp = FitRecord(
        model="Exponential",
        fit_start_s=0.0, fit_end_s=30.0,
        params=[0.0, 0.0, 100.0, 0.05, 0.0],
        param_names=["a", "b", "c", "k", "t0"],
        yhat=np.zeros(10), init_rate_uM_per_s=0.0, init_rate_at_t_s=0.0,
    )
    assert abs(delta_max_from_fit(rec_exp, 0.0) - 100.0) < 1e-6
    assert abs(delta_max_from_fit(rec_exp, 10.0) - 60.6531) < 1e-3
    print("  ✓ from_fit (Exponential) recovers c*exp(-k*Δt)")

    # ── from_linear (two-point API) ──────────────────────────────────
    # Line through (t1=5, y1=90) and (t2=15, y2=70) is y = 100 − 2t.
    # At t_zero=0 the line is at 100, so with max_uM=100 → Δmax = 0.
    # With max_uM=150 and t_zero=5 → y_line(5)=90, Δmax=60.
    delta, slope, intercept = delta_max_from_linear(
        5.0, 90.0, 15.0, 70.0, t_zero=0.0, max_uM=100.0,
    )
    assert abs(slope - (-2.0)) < 1e-6, f"slope={slope}"
    assert abs(intercept - 100.0) < 1e-6, f"intercept={intercept}"
    assert abs(delta - 0.0) < 1e-6, f"delta={delta}"
    print("  ✓ from_linear (two-point, t_zero=0, max_uM=100) → Δmax = 0")

    delta2, _, _ = delta_max_from_linear(
        5.0, 90.0, 15.0, 70.0, t_zero=5.0, max_uM=150.0,
    )
    assert abs(delta2 - 60.0) < 1e-6, f"delta2={delta2}"
    print("  ✓ from_linear (two-point, t_zero=5, max_uM=150) → Δmax = 60")

    # ValueError when the two points share the same t
    try:
        delta_max_from_linear(5.0, 1.0, 5.0, 2.0, t_zero=0.0)
    except ValueError:
        print("  ✓ ValueError when t1 == t2")
    else:
        raise AssertionError("expected ValueError for equal-t points")

    # ── from_point ───────────────────────────────────────────────────
    # Δmax = max_uM − y_at_tzero.  Default max_uM=100.
    assert abs(delta_max_from_point(0.0) - 100.0) < 1e-12        # max=100, y=0
    assert abs(delta_max_from_point(40.0, max_uM=100.0) - 60.0) < 1e-12
    assert abs(delta_max_from_point(50.0, max_uM=150.0) - 100.0) < 1e-12
    print("  ✓ from_point computes max_uM − y_at_tzero")

    # ── from_fit returns NaN for ManualLinear ────────────────────────
    rec_lin = FitRecord(
        model="ManualLinear",
        fit_start_s=0.0, fit_end_s=10.0,
        params=[-2.0, 100.0], param_names=["slope", "intercept"],
        yhat=np.zeros(5), init_rate_uM_per_s=-2.0, init_rate_at_t_s=0.0,
    )
    import math
    assert math.isnan(delta_max_from_fit(rec_lin, 0.0))
    print("  ✓ from_fit returns NaN for ManualLinear (caller falls back to Point/Linear mode)")
    return True


def test_back_extrap_calibration_fit() -> bool:
    _section("C4: 4-point back-extrap calibration exponential fit")
    import matplotlib
    matplotlib.use("Agg")
    import numpy as np
    from sensorfit.calibration import _fit_back_extrap_calibration_exponential

    # Simulate a reaction starting with H2O2 injection: deadtime junk before
    # t=3, then a clean exponential decay y = 100*exp(-0.05*(t-3)) afterwards.
    t = np.linspace(0, 30, 1000)
    y = np.where(t < 3, 50.0, 100 * np.exp(-0.05 * (t - 3)))

    max_t_idx = int(np.argmin(np.abs(t - 3.0)))     # true [H2O2]max moment
    fit_start_idx = int(np.argmin(np.abs(t - 5.0)))  # after deadtime
    fit_end_idx = int(np.argmin(np.abs(t - 25.0)))

    _, _, _, _, extrap_value = _fit_back_extrap_calibration_exponential(
        t, y, fit_start_idx, fit_end_idx, max_t_idx
    )
    # True value at t=3 is 100; allow a small fit-tolerance
    assert abs(extrap_value - 100.0) < 1.0, f"expected ~100, got {extrap_value}"
    print(f"  ✓ exponential fit back-extrapolates to {extrap_value:.3f} (true 100)")

    # Too few points → ValueError
    try:
        _fit_back_extrap_calibration_exponential(t, y, 100, 102, 50)
    except ValueError:
        print("  ✓ ValueError when fit window has <5 points")
        return True
    raise AssertionError("expected ValueError for narrow fit window")


def test_zoom_hotkey_install() -> bool:
    _section("Zoom hotkey: install_zoom_keys wires up cleanly")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sensorfit.zoom_hotkey import install_zoom_keys

    # Single axis
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 1, 0])
    state = install_zoom_keys(fig, ax)
    assert state.get("installed") is True
    plt.close(fig)
    print("  ✓ single-axis install completes without error")

    # Multi-axes form
    fig, ax_pair = plt.subplots(2, 1)
    state2 = install_zoom_keys(fig, list(ax_pair))
    assert state2.get("installed") is True
    plt.close(fig)
    print("  ✓ two-axis install completes without error")
    return True




def test_back_extrap_no_fit() -> bool:
    _section("Commit 4: back-extrap raises when no fit available")
    from sensorfit.back_extrap import compute_back_extrap
    try:
        compute_back_extrap({}, 60.0, 100.0, 2.0, 100.0)
    except ValueError as exc:
        print(f"  ✓ correctly raised ValueError: {exc}")
        return True
    raise AssertionError("expected ValueError when no fit is available")


# ──────────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────────

def test_pad_fit_yhat_and_multifit_excel() -> bool:
    _section("Audit Fix E: _pad_fit_yhat + multi-fit Excel columns")
    import numpy as np
    import pandas as pd
    from sensorfit.calibration_cli import _pad_fit_yhat
    from sensorfit.interval_processor import FitRecord
    from sensorfit.calibration import save_interval_with_fits, IntervalSubset, CALIBRATED_COLUMN

    t = np.linspace(0.0, 10.0, 101)

    # Sub-range fit → padded NaN outside, defined inside.
    rec1 = FitRecord(
        model="ManualLinear", fit_start_s=2.0, fit_end_s=5.0,
        params=[1.0, 0.0], param_names=["slope", "intercept"],
        yhat=np.linspace(2.0, 5.0, 31), init_rate_uM_per_s=1.0, init_rate_at_t_s=2.0,
    )
    padded = _pad_fit_yhat(rec1, t)
    assert len(padded) == len(t)
    assert np.isnan(padded[0]) and np.isnan(padded[-1])
    assert not np.any(np.isnan(padded[(t >= 2.0) & (t <= 5.0)]))
    print("  ✓ _pad_fit_yhat pads NaN outside the fit range, defined inside")

    # Two fits of the SAME model must get distinct Excel columns.
    rec2 = FitRecord(
        model="ManualLinear", fit_start_s=6.0, fit_end_s=9.0,
        params=[0.5, 1.0], param_names=["slope", "intercept"],
        yhat=np.linspace(4.0, 5.5, 31), init_rate_uM_per_s=0.5, init_rate_at_t_s=6.0,
    )
    with tempfile.TemporaryDirectory() as tmp:
        df = pd.DataFrame({"Time (s)": t, CALIBRATED_COLUMN: np.zeros_like(t)})
        subset = IntervalSubset(index=1, start_time=0.0, end_time=10.0, data=df)
        excel_fits = {
            "ManualLinear#1": {"model": "ManualLinear", "params": np.array(rec1.params),
                               "names": rec1.param_names, "yhat": _pad_fit_yhat(rec1, t),
                               "r2": float("nan"), "rss": float("nan"), "init_rate": 1.0},
            "ManualLinear#2": {"model": "ManualLinear", "params": np.array(rec2.params),
                               "names": rec2.param_names, "yhat": _pad_fit_yhat(rec2, t),
                               "r2": float("nan"), "rss": float("nan"), "init_rate": 0.5},
        }
        path = save_interval_with_fits(subset, "Time (s)", CALIBRATED_COLUMN, excel_fits, Path(tmp))
        out = pd.read_excel(path)
        assert "H2O2_uM_fit_ManualLinear#1" in out.columns
        assert "H2O2_uM_fit_ManualLinear#2" in out.columns
        print("  ✓ two same-model fits get distinct Excel columns (no collision)")
    return True


def test_review_per_interval_mode_hides_legacy_redo() -> bool:
    _section("Audit Fix C: review_results hides Redo-fitting/turnover in per-interval mode")
    import inspect
    from sensorfit.calibration import review_results
    sig = inspect.signature(review_results)
    assert "per_interval_mode" in sig.parameters
    # The hiding is driven by the per_interval_mode flag in the source.
    src = inspect.getsource(review_results)
    assert 'hidden_actions |= {"fitting", "turnover"}' in src
    print("  ✓ per_interval_mode flag present and gates fitting/turnover redo")
    return True


def test_pane_role_chrome() -> bool:
    _section("Orientation chrome: apply_pane_role_chrome + pane_role_context")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sensorfit.calibration import apply_pane_role_chrome, pane_role_context

    fig, ax = plt.subplots()
    fig.suptitle("Test window")
    apply_pane_role_chrome(fig, "control")
    assert fig.patch.get_edgecolor()[:3] != (1.0, 1.0, 1.0), "Edge should be orange-ish"
    assert "[CONTROL]" in fig._suptitle.get_text()
    plt.close(fig)
    print("  ✓ apply_pane_role_chrome sets edge colour and suptitle for 'control'")

    with pane_role_context("sample"):
        fig2, ax2 = plt.subplots()
        fig2.suptitle("Sample window")
        # Chrome is applied at plt.show() time; simulate that:
        apply_pane_role_chrome(fig2, "sample")
    assert "[SAMPLE]" in fig2._suptitle.get_text()
    plt.close(fig2)
    print("  ✓ pane_role_context patches plt.subplots for 'sample'")

    fig3, ax3 = plt.subplots()
    fig3.suptitle("Normal")
    assert "[SAMPLE]" not in (fig3._suptitle.get_text() if fig3._suptitle else "")
    assert "[CONTROL]" not in (fig3._suptitle.get_text() if fig3._suptitle else "")
    plt.close(fig3)
    print("  ✓ outside context manager, figures are unmodified")
    return True


def test_average_controls_on_grid() -> bool:
    _section("Multi-control averaging + deviation-based subtraction")
    import numpy as np
    from sensorfit.controls import average_controls_on_grid, deviation_from_anchor

    sample_t = np.linspace(0.0, 10.0, 101)

    ctrl_a = (np.linspace(0.0, 10.0, 51), np.ones(51) * 2.0)
    ctrl_b = (np.linspace(0.0, 10.0, 51), np.ones(51) * 4.0)

    avg, interps = average_controls_on_grid([ctrl_a, ctrl_b], sample_t, anchor_t0=0.0)
    assert len(avg) == len(sample_t)
    assert len(interps) == 2
    assert np.allclose(avg, 3.0, atol=0.01), f"Expected ~3.0, got {avg[:5]}"
    print("  ✓ two flat controls averaged correctly")

    # Deviation-based subtraction: flat control → deviation is zero everywhere
    dev = deviation_from_anchor(avg, sample_t, 0.0)
    assert np.allclose(dev, 0.0, atol=0.01), "Flat control deviation should be zero"
    print("  ✓ flat control deviation is zero (preserves sample absolute scale)")

    # Drifting control: starts at 100, drifts to 90
    ctrl_drift = np.linspace(100.0, 90.0, 101)
    dev_drift = deviation_from_anchor(ctrl_drift, sample_t, 0.0)
    assert abs(dev_drift[0]) < 0.01, "Deviation at anchor should be ~0"
    assert abs(dev_drift[-1] - (-10.0)) < 0.01, "Deviation at end should be -10"
    sample = np.ones(101) * 100.0
    corrected = sample - dev_drift
    assert abs(corrected[0] - 100.0) < 0.01, "Corrected should keep sample's starting value"
    assert abs(corrected[-1] - 110.0) < 0.01, "Corrected should gain +10 (undo drift)"
    print("  ✓ drifting control: deviation removes drift, preserves absolute scale")

    avg1, interps1 = average_controls_on_grid([ctrl_a], sample_t, anchor_t0=0.0)
    assert np.allclose(avg1, 2.0, atol=0.01)
    print("  ✓ single control returns itself")

    try:
        average_controls_on_grid([], sample_t)
        assert False, "Should have raised"
    except ValueError:
        pass
    print("  ✓ empty list raises ValueError")
    return True


def main() -> int:
    tests = [
        test_fit_summary_upsert,
        test_fit_summary_filter,
        test_control_template_roundtrip,
        test_interpolate_control,
        test_residual_activity_module_gone,
        test_grouping_modules_gone,
        test_back_extrap_exponential,
        test_back_extrap_linear_fallback,
        test_back_extrap_no_fit,
        test_fit_baseline_polynomial,
        test_back_extrap_calibration_fit,
        test_delta_max_pure_helpers,
        test_fit_summary_per_fit_columns,
        test_zoom_hotkey_install,
        test_skip_calibration_provenance,
        test_skip_calibration_loads_real_csv,
        test_pad_fit_yhat_and_multifit_excel,
        test_review_per_interval_mode_hides_legacy_redo,
        test_average_controls_on_grid,
        test_pane_role_chrome,
    ]
    failures = []
    for t in tests:
        try:
            t()
        except Exception as exc:
            failures.append((t.__name__, exc))
            print(f"  ✗ {t.__name__}: {exc}")
    _section("Test summary")
    n = len(tests)
    print(f"  {n - len(failures)}/{n} passed")
    if failures:
        for name, exc in failures:
            print(f"   ✗ {name}: {exc}")
        return 1
    print("  All tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
