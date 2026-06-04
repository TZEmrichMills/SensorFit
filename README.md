# SensorFit

Interactive tool for calibrating amperometric sensor traces to H₂O₂ concentrations and performing curve-fitting analysis.

## Contents

- [Quick Start](#quick-start)
- [What to Expect When You Run SensorFit](#what-to-expect-when-you-run-sensorfit)
- [Command Line Arguments](#command-line-arguments)
- [Advanced features](#advanced-features) — control subtraction, residual activity, back-extrapolation, resilient summary
- [Post-processing: Collecting Intervals](#post-processing-collecting-intervals)
- [Detailed Installation Guide](#detailed-installation-guide)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

**Prerequisites:** Python 3.9+ ([download here](https://www.python.org/downloads/) — Windows users: tick "Add Python to PATH" during install).

> **Windows users:** Do NOT install into OneDrive, Dropbox, or other cloud-synced folders — use a local path like `C:\Projects\SensorFit`.

**Mac / Linux:**

```bash
git clone https://github.com/TZEmrichMills/SensorFit.git
cd SensorFit
python3 -m venv sensorfit_env
source sensorfit_env/bin/activate
pip install --upgrade pip
pip install -e .
```

**Windows (Command Prompt):**

```cmd
git clone https://github.com/TZEmrichMills/SensorFit.git
cd SensorFit
python -m venv sensorfit_env
sensorfit_env\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

**Run the calibration tool** (always include `--force`):

```bash
python -m sensorfit.calibration_cli --input-dir "/path/to/data" --num-points 6 --calibration-values "0,20,40,60,80,100" --force
```

Each time you come back to use SensorFit, activate the environment first:

```bash
# Mac / Linux
cd /path/to/SensorFit && source sensorfit_env/bin/activate

# Windows
cd C:\path\to\SensorFit
sensorfit_env\Scripts\activate
```

---

## What to Expect When You Run SensorFit

SensorFit processes every data file in your input directory one at a time. For each file you will step through the following interactive screens. Every screen has navigation buttons (continue, go back, redraw, discard, skip, etc.), so you can always correct mistakes or skip steps.

### 1. Baseline Selection

A plot of the raw current trace opens. You have three choices:

- **Define a baseline** — click two points on the plot to draw a baseline line, then click *Continue* to preview the corrected trace. You can *Accept*, *Retry*, or *Discard* the file.
- **Skip baseline** — if the trace already looks flat, click *Skip baseline* to proceed with the raw signal as-is.
- **Discard this file** — skip the file entirely and move on to the next.

### 2. Calibration Point Selection

The (baseline-corrected) current trace is shown. Click on the plateau of each calibration step (e.g. 0, 20, 40, 60, 80, 100 µM H₂O₂) to select calibration points. The number and values of calibration points can be edited via a pop-up table editor. You can go back to baseline selection from here if needed.

### 3. Interval Selection

A calibrated H₂O₂ concentration trace is shown. Click pairs of points to define time intervals you want to analyse (e.g. each enzyme injection). Intervals are highlighted on the plot. You can add multiple intervals, undo, or go back to calibration.

### 4. Curve Fitting (per interval)

For each interval, a fitting interface opens. You can select one or more models to fit:

- **Inactivation (IB)** — enzyme inactivation kinetics
- **Exponential** — single exponential decay with linear background
- **Gompertz-like (GFI)** — Gompertz-gated inactivation with a fast phase
- **Linear Initial Rate** — straight-line fit to a user-selected region (auto or manual)

Fitted curves and residuals are overlaid on the data. Model statistics (R², AIC) are reported. You can go back, discard, or continue to the next step.

### 5. Maximum H₂O₂ Turnover Before Inactivation (per interval)

For each interval, you can calculate the maximum amount of H₂O₂ consumed (or produced) before the enzyme became fully inactivated. Two methods are available:

- **Automatic** — fits a straight line to the tail of the best fitted model and extrapolates back to the interval start.
- **Manual** — click two points to define the tail region yourself.

The turnover value is displayed on the plot and recorded in the summary.

### 6. Next File

Results (calibrated data, interval Excel files, fit parameters) are saved under a `Calibrated/` folder. The original file is moved to a `Processed/` folder. A running `fit_summary.xlsx` is updated with all fit results across files **inside `Calibrated/`** (it used to live at the root of the input directory; see the [Advanced features](#advanced-features) section for the resilience fix). SensorFit then opens the next file and repeats from step 1.

If a session is interrupted (Ctrl-C, crash, lost power), just re-run the same command — SensorFit detects the existing `Calibrated/fit_summary.xlsx` and resumes appending to it. Files already in `Processed/` are skipped automatically.

---

## Command Line Arguments

| Argument | Default | Description |
|---|---|---|
| `--input-dir` | *(required)* | Directory containing input files (.xlsx, .txt, .csv) |
| `--output-dir` | same as input | Output directory |
| `--pattern` | auto-detect | Glob pattern for input files |
| `--time-col` | `0` | Column index for time (0-indexed) |
| `--current-col` | `1` | Column index for current (0-indexed) |
| `--num-points` | `6` | Number of calibration points |
| `--window` | `50` | Samples to average around each click |
| `--calibration-values` | `"0,20,40,60,80,100"` | Comma-separated µM H₂O₂ concentrations |
| `--force` | off | Overwrite existing output files |
| `--control-mode` | off | Open a grouping UI at session start so a control trace (no-enzyme, etc.) can be subtracted from selected sample files. See [Control subtraction](#control-subtraction) below. Auto-enabled if `Calibrated/controls.json` already exists. |

---

## Advanced features

These features are all **opt-in** — if you don't use them, SensorFit behaves exactly as before. Each feature adds extra columns to `fit_summary.xlsx` only when used.

### Control subtraction (multi-control groups, group-as-unit flow)

Many experimental designs involve measuring one or more **control runs** (no-enzyme, no-substrate, buffer-only, etc.) and subtracting them from related sample runs. SensorFit groups controls with samples and handles subtraction at the **interval level**, after all files in the group have been calibrated and intervals delineated.

**Run:**

```bash
python -m sensorfit.calibration_cli --input-dir /path/to/folder --num-points 6 --calibration-values "0,20,40,60,80,100" --control-mode --force
```

**Grouping window (session start).**

A Qt window opens listing every unprocessed file. Each *group* is built as a tree:

```
Group_A
├── Sub-group 1 (averaged)
│   ├── noEnz_noSub_1.xlsx
│   └── noEnz_noSub_2.xlsx
├── Sub-group 2
│   └── noEnz_1.xlsx
└── Samples
    ├── treatment_1.xlsx
    └── treatment_2.xlsx
```

- Controls inside the **same sub-group** are **averaged** before subtraction.
- Different sub-groups are **subtracted sequentially** (top → bottom).
- For each sample interval the user later chooses to correct, the effective subtraction is:
  `sample(t) − mean(Sub-group 1)(t) − mean(Sub-group 2)(t) − …`

Buttons: **New group**, **New sub-group in selected group**, **Add selected files → sub-group (controls)**, **Add selected files → samples**, **↑ / ↓ Move sub-group up/down**, **Remove selected**, **Done**. The grouping is saved to `Calibrated/controls.json` so an interrupted session can resume.

Files left ungrouped are processed normally, with no subtraction step.

**Group-as-unit processing.** Each group is fully processed end-to-end before the next group begins:

1. **Controls** in the group are processed first, one by one, through the normal baseline → calibration → intervals → fits → turnover flow. After interval selection on each control, you pick which interval is the **subtraction reference**; that interval is saved as a CSV in `Calibrated/_control_templates/`.
2. **Samples** in the group are processed next, also through the full pipeline. Their state (calibrated frame, intervals, fits, turnover) is kept in memory for the next step.
3. **Subtraction planning UI** opens once all samples are done: a check-list of every sample interval. Tick the ones you want corrected. Defaults to all-unticked so calibration ladders / blank phases are left alone — you opt in per interval.
4. **Preview & subtract.** For each ticked interval, a preview screen shows the sample (green), each sub-group's averaged contribution (orange dashed), their combined sum (red), the anchor as a **purple dashed vertical line**, and the corrected result below in blue. Click on the upper plot to **re-anchor** the controls' t = 0. **Accept & subtract**, **Skip this interval**, or **Back**.
5. **Optional re-fit.** After subtraction, for each sample with corrected intervals you're asked **"Re-fit corrected intervals?"** Yes runs the existing fitting + turnover GUIs on the subtracted data; No keeps the pre-subtraction fits.

**Outputs.** Each corrected interval is saved alongside the original as `interval_NN_corrected.xlsx` in the sample's `_intervals/` folder. `fit_summary.xlsx` carries **two rows per corrected interval** distinguished by a new `variant` column:
- `variant = "original"` — pre-subtraction fit results (always written).
- `variant = "corrected"` — post-subtraction fit results, with `control_subtracted = True`, `control_group = <name>`, `correction_n_subgroups`, `correction_anchor_t0_s`, and `correction_chain` (a human-readable summary of which controls were applied).

You can filter / pivot by `variant` in Excel to compare before vs. after subtraction directly.

**Legacy single-control groups** (one sub-group with exactly one control file — what PR #1 supported) still work via the original inline `control_subtract` phase during sample processing, with the same UX as before.

### Multi-addition residual activity

If a single run contained **multiple successive H₂O₂ additions** (addition 1 → consume → addition 2 → consume → …), each addition typically becomes its own interval. The ratio of the second interval's initial rate to the first interval's initial rate gives the **residual activity** — how much of the enzyme's original activity remains after the first addition's consumption.

**How to use it:**

1. Define ≥2 intervals during the normal interval-selection step.
2. After confirming intervals, a small dialog asks **"Multi-addition residual-activity series?"** Click Yes.
3. A checkbox dialog lists every interval — tick the ones that belong to the series (usually all of them) and click **Save selection**.
4. After fitting, ratios are written to a new `residual_activity_ratio` column in `fit_summary.xlsx`:
   - The first interval in the series gets `1.0`.
   - Each later interval gets `best_initial_rate / first_interval_initial_rate`.
   - Intervals outside the series get NaN.

This replaces the by-hand computation in the `Rate ratio` table of the older analysis Excels.

### H₂O₂-injection-start back-extrapolation

When a reaction begins with an H₂O₂ injection and you used a **two-point calibration** (pre-addition = 0 µM, just-after-addition = e.g. 100 µM), the calibrated concentrations are systematically under-estimated: the enzyme has already consumed some H₂O₂ during the instrument deadtime (~1–2 s before the first reliable reading). The correction is:

1. **Back-extrapolate** the fitted Exponential model to `t_start − deadtime` to recover the true [H₂O₂] at the moment of addition.
2. Compute the **stretch factor** `back_extrap_uM / nominal_uM` — by how much the calibration under-estimated.
3. Compute the **stretched initial rate** = `rate_at_back × stretch_factor`. This combines the steeper slope at the (earlier) true t₀ with the calibration rescale.

**How to use it:**

1. Fit at least an Exponential model to the relevant interval (the back-extrap also works with a `LinearInitialRate` fit but is less accurate).
2. After the fitting phase, a dialog asks **"H₂O₂-injection-start back-extrapolation?"** Click Yes.
3. For each interval, a preview screen lets you enter the **deadtime** (default 1.5 s) and the **nominal H₂O₂ added** (default = the observed initial value). Click **Compute** to preview the back-extrapolated curve (red, dashed) and the computed values. Click **Accept & record** to save, **Skip** to pass on this interval, or **Back** to abort the back-extrap phase.

Six new columns are added to `fit_summary.xlsx` for accepted intervals: `back_extrap_applied`, `back_extrap_deadtime_s`, `back_extrap_nominal_uM`, `back_extrap_H2O2_at_true_t0_uM`, `back_extrap_stretch_factor`, `back_extrap_stretch_initial_rate_uM_per_s`. These mirror the columns in older hand-computed analyses (`Back extrap`, `Stretch factor`, `Stretch initial rate`).

### Resilient `fit_summary.xlsx`

`fit_summary.xlsx` now lives **inside `Calibrated/`** rather than at the root of the input directory. This prevents two issues that bit earlier versions:

- SensorFit no longer tries to "process" its own summary as input data (the summary used to be discovered as a `.xlsx` in the input folder).
- An interrupted session can be restarted with the same command, and SensorFit will resume appending to the existing summary instead of creating a duplicate.

A legacy `fit_summary.xlsx` at the root of an input directory is **automatically migrated** into `Calibrated/` on first run, so existing experiments continue cleanly.

Re-running a file (move it from `Processed/` back to the root and re-launch) now **replaces** the corresponding row(s) in `fit_summary.xlsx` rather than duplicating them.

---

## Post-processing: Collecting Intervals

After running SensorFit on a batch of files, you can combine all calibrated interval traces into a single Excel workbook for easy plotting and comparison.

```bash
python -m sensorfit.collect_intervals /path/to/experiment/directory
```

This scans `Calibrated/*_intervals/interval_*.xlsx`, zero-bases every interval to start at t = 0, downsamples by 6.25x (0.08 s → 0.5 s by default), extends shorter traces to the length of the longest one (using linear extrapolation, shown in red font), and writes a single `all_intervals_downsampled.xlsx`.

**Options:**

| Flag | Description |
|---|---|
| `-o path.xlsx` | Custom output path |
| `--downsample-factor N` | Change downsample factor (default 6.25) |
| `--no-downsample` | Keep original time resolution |
| `--no-extend` | Don't extend shorter intervals (leave as empty cells) |

**Example output:**

| Time (s) | Sample-A #1 | Sample-B #1 | Sample-C #1 | ... |
|----------|-------------|-------------|-------------|-----|
| 0.0      | 98.29       | 98.14       | 98.59       | ... |
| 0.5      | 98.05       | 98.14       | 98.74       | ... |
| ...      | ...         | ...         | ...         | ... |

---

## Detailed Installation Guide

If the [Quick Start](#quick-start) instructions didn't work, follow these expanded steps.

### Check Python

```bash
python3 --version   # Mac/Linux
python --version     # Windows (or try: py --version)
```

You need Python 3.9 or higher. If not installed, download from [python.org](https://www.python.org/downloads/). **Windows users:** make sure to tick "Add Python to PATH" during installation.

### Clone or Download

If you have `git` installed:

```bash
git clone https://github.com/TZEmrichMills/SensorFit.git
```

Otherwise, download the ZIP from GitHub and extract it.

### Create and Activate a Virtual Environment

A virtual environment keeps SensorFit's dependencies separate from other Python projects.

**Mac / Linux:**

```bash
cd /path/to/SensorFit
python3 -m venv sensorfit_env
source sensorfit_env/bin/activate
```

**Windows (Command Prompt — recommended over PowerShell):**

```cmd
cd C:\path\to\SensorFit
python -m venv sensorfit_env
sensorfit_env\Scripts\activate
```

You should see `(sensorfit_env)` at the start of your prompt.

### Install

```bash
pip install --upgrade pip
pip install -e .
```

This installs SensorFit in editable mode, so any updates to the source code take effect immediately. Dependencies (numpy, pandas, matplotlib, scipy, PyQt5, openpyxl) are installed automatically.

If `pip install -e .` gives an "Access is denied" error on Windows, see [Troubleshooting](#access-is-denied-error-windows).

### Verify

```bash
python -c "import sensorfit; print('SensorFit installed successfully!')"
python -m sensorfit.calibration_cli --help
```

### Updating

If SensorFit has been updated on GitHub, pull the latest changes:

```bash
cd /path/to/SensorFit
git pull
```

Because it's installed in editable mode, the new code is available immediately — no reinstall needed.

---

## Troubleshooting

### "python is not recognized" / "command not found"

Python is not installed or not in your PATH. [Install Python](https://www.python.org/downloads/) and make sure "Add Python to PATH" is ticked. Restart your terminal after installing. Windows users can also try `py` instead of `python`.

### ModuleNotFoundError: No module named 'sensorfit'

Make sure your virtual environment is activated (you should see `(sensorfit_env)` in your prompt) and that you ran `pip install -e .` from inside the SensorFit directory.

### Access is denied error (Windows)

This almost always means the project is inside a cloud-synced folder (OneDrive, Dropbox, etc.).

1. **Move the project** to a local folder such as `C:\Projects\SensorFit`
2. Delete the old `sensorfit_env` folder and create a new one
3. Try again

Other things to try, in order:

- Use `python -m pip install -e .` instead of `pip install -e .`
- Run Command Prompt as Administrator
- Install without editable mode: `pip install .` (you will need to reinstall after updates)
- Temporarily disable antivirus software
- Close any programs or file explorers that have the SensorFit folder open

### PowerShell execution policy error

Use **Command Prompt** instead of PowerShell — it does not have this restriction. If you must use PowerShell, run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### FileExistsError: "already exists (use --force to overwrite)"

Add `--force` to your command. This is needed whenever output files from a previous run already exist.

### PyQt5 won't install

Install PySide6 as an alternative:

```bash
pip install PySide6
pip install -e .
```

### Installation hangs or is very slow

Check your internet connection. Try `pip install --upgrade pip` first. If you're behind a firewall, try: `pip install -e . -i https://pypi.org/simple`.
