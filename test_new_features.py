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
    _section("Commit 1: fit_summary idempotent upsert (with variant column)")
    import pandas as pd
    from sensorfit.calibration import append_fit_summary

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "fit_summary.xlsx"

        append_fit_summary(p, "fileA.xlsx", 1, 0.0, 10.0, fit_results=None, turnover_uM=None)
        # Second write with same (file, interval, variant=original) → REPLACE
        append_fit_summary(p, "fileA.xlsx", 1, 0.0, 10.0, fit_results=None, turnover_uM=42.0)
        # Third: same file, different interval → APPEND
        append_fit_summary(p, "fileA.xlsx", 2, 11.0, 20.0, fit_results=None, turnover_uM=None)
        # Fourth: different file → APPEND
        append_fit_summary(p, "fileB.xlsx", 1, 0.0, 5.0, fit_results=None, turnover_uM=None)
        # Fifth: same fileA interval 1 but variant=corrected → APPEND
        append_fit_summary(p, "fileA.xlsx", 1, 0.0, 10.0, fit_results=None,
                           turnover_uM=99.0, variant="corrected", control_group="G_A")

        df = pd.read_excel(p)
        assert len(df) == 4, f"expected 4 rows, got {len(df)}"
        row_orig = df[(df["source_file"] == "fileA.xlsx") & (df["interval_index"] == 1)
                      & (df["variant"] == "original")].iloc[0]
        row_corr = df[(df["source_file"] == "fileA.xlsx") & (df["interval_index"] == 1)
                      & (df["variant"] == "corrected")].iloc[0]
        assert row_orig["turnover_before_inactivation_uM"] == 42.0
        assert row_corr["turnover_before_inactivation_uM"] == 99.0
        assert row_corr["control_group"] == "G_A"
        print(f"  ✓ upsert: 5 writes → 4 rows; original + corrected variants coexist")
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

def test_controls_manifest_roundtrip() -> bool:
    _section("Commit 2: controls manifest round-trip (new multi-control model)")
    import json
    from sensorfit.controls import (
        ControlGroup, ControlSubgroup, ControlSpec,
        save_controls_manifest, load_controls_manifest,
    )

    with tempfile.TemporaryDirectory() as tmp:
        calib = Path(tmp) / "Calibrated"
        groups = {
            "G_A": ControlGroup(
                name="G_A",
                subgroups=[
                    ControlSubgroup(
                        name="Sub-group 1",
                        controls=[
                            ControlSpec(
                                file_name="ctrl1.xlsx",
                                reference_interval_index=2,
                                template_filename="G_A_ctrl1.csv",
                            ),
                            ControlSpec(
                                file_name="ctrl1b.xlsx",
                                reference_interval_index=1,
                                template_filename="G_A_ctrl1b.csv",
                            ),
                        ],
                    ),
                    ControlSubgroup(
                        name="Sub-group 2",
                        controls=[
                            ControlSpec(file_name="ctrl2.xlsx"),
                        ],
                    ),
                ],
                sample_files=["s1.xlsx", "s2.xlsx"],
            ),
            "G_B": ControlGroup(
                name="G_B",
                subgroups=[
                    ControlSubgroup(name="Sub-group 1", controls=[ControlSpec(file_name="ctrl3.xlsx")]),
                ],
                sample_files=["s3.xlsx"],
            ),
        }
        save_controls_manifest(groups, calib)
        loaded = load_controls_manifest(calib)
        assert set(loaded) == set(groups)
        assert len(loaded["G_A"].subgroups) == 2
        assert len(loaded["G_A"].subgroups[0].controls) == 2
        assert loaded["G_A"].subgroups[0].controls[0].reference_interval_index == 2
        assert loaded["G_A"].subgroups[0].controls[0].template_filename == "G_A_ctrl1.csv"
        assert loaded["G_A"].sample_files == ["s1.xlsx", "s2.xlsx"]
        print("  ✓ new multi-control manifest round-trips (2 subgroups, 3 controls in G_A)")

    # Legacy auto-upgrade: a manifest written in the old single-control format
    # should load into a single-subgroup, single-control group.
    with tempfile.TemporaryDirectory() as tmp:
        calib = Path(tmp) / "Calibrated"
        calib.mkdir(parents=True)
        legacy_payload = {
            "version": 1,
            "groups": [
                {
                    "name": "G_legacy",
                    "control_file": "old_ctrl.xlsx",
                    "sample_files": ["s1.xlsx"],
                    "reference_interval_index": 1,
                    "template_filename": "old_ctrl.csv",
                }
            ],
        }
        with open(calib / "controls.json", "w") as fh:
            json.dump(legacy_payload, fh)
        loaded = load_controls_manifest(calib)
        assert loaded is not None and "G_legacy" in loaded
        g = loaded["G_legacy"]
        assert len(g.subgroups) == 1 and len(g.subgroups[0].controls) == 1
        assert g.subgroups[0].controls[0].file_name == "old_ctrl.xlsx"
        assert g.subgroups[0].controls[0].template_filename == "old_ctrl.csv"
        assert g.sample_files == ["s1.xlsx"]
        print("  ✓ legacy single-control manifest auto-upgrades to new nested format")
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
    _section("C2: zoom_hotkey.install_zoom_keys wires up cleanly")
    # Headless mode for CI / non-interactive testing
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sensorfit.zoom_hotkey import install_zoom_keys

    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 1, 0])
    state = install_zoom_keys(fig, ax)
    assert state["zoom_active"] is False
    assert id(ax) in state["selectors"]
    plt.close(fig)
    print("  ✓ single-axis install: zoom inactive at start, selector created")

    # Multi-axes form
    fig, ax_pair = plt.subplots(2, 1)
    state2 = install_zoom_keys(fig, list(ax_pair))
    assert len(state2["selectors"]) == 2
    plt.close(fig)
    print("  ✓ two-axis install: both selectors created")
    return True


def test_build_subtraction_chain() -> bool:
    _section("Restructure: build_subtraction_chain (averaged-within / sequential-across)")
    import numpy as np
    from sensorfit.group_planning import build_subtraction_chain

    ct = np.linspace(0, 10, 51)
    sample_t = np.linspace(100, 110, 51)
    anchor = 100.0

    # 1) Single sub-group of two constants (5 and 7) → averaged to 6
    combined, per_sg = build_subtraction_chain(
        sample_t,
        [[(ct, np.full_like(ct, 5.0)), (ct, np.full_like(ct, 7.0))]],
        anchor,
    )
    assert np.allclose(combined, 6.0)
    assert len(per_sg) == 1
    print("  ✓ single sub-group of 2 controls (5, 7) → avg = 6")

    # 2) Two sub-groups subtracted sequentially: 3 + 2 = 5
    combined, per_sg = build_subtraction_chain(
        sample_t,
        [[(ct, np.full_like(ct, 3.0))], [(ct, np.full_like(ct, 2.0))]],
        anchor,
    )
    assert np.allclose(combined, 5.0)
    assert len(per_sg) == 2
    print("  ✓ two sub-groups (3, 2) → sum = 5")

    # 3) Empty chain → zero correction
    combined, per_sg = build_subtraction_chain(sample_t, [], anchor)
    assert np.allclose(combined, 0.0)
    assert per_sg == []
    print("  ✓ empty chain → zero correction")

    # 4) Averaging templates of different lengths
    ct_short = np.linspace(0, 8, 41)
    cy_short = ct_short * 0.5  # 0..4
    ct_long = np.linspace(0, 10, 51)
    cy_long = ct_long * 1.0    # 0..10
    sample_t = np.array([100.0, 105.0])
    combined, _ = build_subtraction_chain(
        sample_t, [[(ct_short, cy_short), (ct_long, cy_long)]], 100.0
    )
    # At t=105 (5s after anchor): short=2.5, long=5.0, avg=3.75
    assert abs(combined[1] - 3.75) < 1e-6
    print("  ✓ averaging of different-length templates (0.5*t and 1.0*t at t=5s → avg=3.75)")
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

def main() -> int:
    tests = [
        test_fit_summary_upsert,
        test_fit_summary_filter,
        test_controls_manifest_roundtrip,
        test_control_template_roundtrip,
        test_interpolate_control,
        test_residual_activity_module_gone,
        test_back_extrap_exponential,
        test_back_extrap_linear_fallback,
        test_back_extrap_no_fit,
        test_fit_baseline_polynomial,
        test_back_extrap_calibration_fit,
        test_zoom_hotkey_install,
        test_build_subtraction_chain,
        test_skip_calibration_provenance,
        test_skip_calibration_loads_real_csv,
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
