# SensorFit

Interactive tool for calibrating amperometric sensor traces to H₂O₂ concentrations and performing curve-fitting analysis.

## Contents

- [Quick Start](#quick-start)
- [What to Expect When You Run SensorFit](#what-to-expect-when-you-run-sensorfit)
- [Command Line Arguments](#command-line-arguments)
- [Advanced features](#advanced-features) — control subtraction (per-interval, multi-control averaging), skip calibration, resilient summary
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

> **⚠️ If you use Anaconda / Miniconda, read this first.** The single most common install failure is having conda's `base` environment active at the same time as `sensorfit_env` — the two overlap and `pip install -e .` can land in the wrong place, so SensorFit later can't find `numpy`. If your terminal prompt starts with `(base)`, do this **before** the steps above:
>
> ```bash
> conda deactivate          # repeat until (base) disappears from your prompt
> ```
>
> Then create and activate `sensorfit_env` as shown, and **verify the venv is really the active Python before installing**:
>
> ```bash
> which python              # Mac / Linux  → must contain .../sensorfit_env/bin/python
> where python              # Windows      → must contain ...\sensorfit_env\Scripts\python.exe
> ```
>
> Only run `pip install -e .` once that path points **inside `sensorfit_env`**. To stop conda auto-activating its base env in every new terminal (recommended):
>
> ```bash
> conda config --set auto_activate_base false
> ```
>
> After this, close and reopen the terminal so the prompt no longer shows `(base)`.

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

SensorFit processes every data file in your input directory one at a time. For each file you will step through these screens. Every step is skippable, retryable, and has a "Back" option, so you can always correct mistakes.

### 1. Baseline Selection

Two modes via the toggle at top-left:

- **Line (default, 2 clicks)** — straight-line baseline, same as previous versions.
- **Curve (≥3 clicks)** — polynomial fit through your points (degree auto: 2 → 1, 3 → 2, 4+ → 3). Smooths through click-noise and extrapolates naturally to the start/end of the run — pick this when the drift is curved rather than linear.

A window-size **TextBox at top-centre** lets you change the click averaging window on the fly (type a new value and press Enter; existing points are re-averaged in place).

Buttons: Continue | Redraw | Skip baseline | Discard this file.

### 2. Calibration Point Selection

Two modes via the toggle at top-left:

- **Standard (default)** — pick N points across a [H₂O₂] ladder (e.g. 0, 20, 40, 60, 80, 100 µM). Linear calibration is built from your points.
- **Back-extrap (4-pt)** — for reactions started by H₂O₂ injection where the deadtime keeps you from a clean max-current point. Pick four points in order: ① baseline / [H₂O₂]=0, ② timepoint of true max-[H₂O₂] (y-value doesn't matter), ③ start of an exponential-fit region (after deadtime), ④ end of the fit region. SensorFit fits a single exponential between ③ and ④ and back-extrapolates to the timepoint of ② — that extrapolated current becomes the calibration's max anchor (drawn as a red ring). Accept builds a 2-point linear calibration from (zero_avg, 0 µM) and (extrap_value, max_µM).

The same window-size TextBox is available here.

A pop-up table editor lets you edit the calibration µM values or change the number of points (Standard mode).

### 3. Per-interval flow

This is the heart of SensorFit. Instead of picking all intervals up-front then fitting them all, each interval flows through subtraction → fits → Δmax before you move on:

For each interval (repeat until you click **Done with intervals**):

1. **Pick the interval** — click START then END. Existing intervals are shown faintly in grey.
2. **Optional control subtraction** — small dialog with four choices:
   - **None** — use the interval as-is.
   - **Subtract existing** — pick any control interval already in this session OR saved in `Calibrated/*_intervals/interval_*.xlsx` from a prior session.
   - **Subtract new alongside** — a Qt file dialog opens; pick a control file; a modal mini-flow runs baseline → calibration → interval-selection on that file (no fitting), saves it as if processed independently, then returns you to a preview where you can re-anchor the control's t = 0 by clicking the upper plot.
   - **Back** — return to the interval picker.
3. **Optional fit(s) — multi-fit supported.** For each fit:
   1. **Pick a model**: Manual linear / Single exponential / Inactivation (IB).
   2. **Click fit start + fit end** inside the interval (these can be inside the interval; they don't have to use its whole range).
   3. **Preview** shows the fit + the **initial rate at your chosen start point**.
   4. **Optional back-extrapolation**: type a deadtime (default 1.5 s) in the TextBox, click **Extrapolate**; you see a red dashed line back to a red circle at the new, earlier t = 0, and the new initial rate at that point. Accept records both the fit and the back-extrap onto the same row.
   5. **"Another fit on this interval?"** — Yes loops back to model-pick; No goes to Δmax.
4. **Optional Δ[H₂O₂]max** (one per interval). Three modes via top-of-window buttons:
   - **From fit** — uses the last fit; Δmax at your chosen t=0 is the height of the y-axis intercept of the asymptote (analytically: `c·exp(-k·(t_zero−t0))` for Exponential; `H0·exp(-α(1-exp(-k·t_zero)))` for IB).
   - **Linear** — click t = 0, then two more points to define a straight line; Δmax = line's y at t = 0 minus y at end of run.
   - **Point** — click ONE point; Δmax = its y-value minus y at end of run.

   For all three modes, you click your **t = 0 anchor first**. This dynamic anchor lets you play with the "effective start" of the interval; the recorded value will reflect that choice. Accept records it.
5. **"Another interval?"** — Yes loops back to step 1; No → review.

### 4. Review and Save

A short text summary of every accepted interval and fit. Accept saves outputs; Discard skips the file.

### 5. Next file

Results — the calibrated trace, per-interval Excel files (one for each interval), and a `fit_summary.xlsx` row per fit — are saved under `Calibrated/`. The original file is moved to `Processed/`. SensorFit then opens the next file.

`fit_summary.xlsx` rows are keyed on `(source_file, interval_index, fit_number, variant)`. `fit_number = 0` is the "interval only" row (no fit applied); `fit_number = 1, 2, …` is one row per fit. Each fit row carries its own initial rate, fit-range, back-extrap (if accepted), and the interval's Δmax — so a single row is a complete record of one fit.

If a session is interrupted, re-run the same command — SensorFit resumes from `Calibrated/fit_summary.xlsx` and skips files already in `Processed/`.

### Zoom shortcuts (every interactive screen)

- Press `z` to enter zoom mode (cursor changes to crosshair); next click-drag zooms into that rectangle. Press `z` again to leave.
- Press `r` to reset the view to the full data extent.

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
| `--skip-calibration` | off | Treat input files as already-calibrated [H₂O₂] vs time and skip the baseline / calibration phases entirely. `--current-col` is interpreted as the µM H₂O₂ column. See [Skip calibration](#skip-calibration-fitting-only-mode) below. |

---

## Advanced features

These features are all **opt-in** — if you don't use them, SensorFit behaves exactly as before. Each feature adds extra columns to `fit_summary.xlsx` only when used.

### Control subtraction

Control subtraction now lives **inside the per-interval flow** (see "What to Expect" above) — when you define an interval you immediately get a dialog asking whether to subtract a control. Two routes:

- **Subtract existing** — pick any interval already in this session OR saved on disk in `Calibrated/*_intervals/`. Useful for re-using a previously-processed control without re-running it.
- **Subtract new alongside** — a Qt file dialog opens; pick a control file; SensorFit runs a modal mini-flow (baseline → calibration → interval-selection only) on that file, saves it under `Calibrated/` as if processed independently, and then drops you into a preview where you can re-anchor the control's t = 0 by clicking the upper plot.

The original (un-subtracted) interval is still saved alongside the corrected one in the same `_intervals/` folder, so you have both for comparison.

`fit_summary.xlsx` rows for control-subtracted intervals carry:
- `control_subtracted = True`
- `control_group` — the control interval(s)/file(s) that were subtracted (multiple averaged controls are joined with ` | `)
- `control_n_averaged` — how many control traces were averaged together (1 for a single control)

**Averaging multiple controls.** Both subtraction routes drop you into an averaging hub where you can add several control traces; they are aligned onto the sample's time grid and averaged before subtraction. The subtraction itself is *deviation-based* — only the control's drift away from its value at the anchor time is removed, so a sample that starts at 100 µM is not zeroed out by a control that also starts at 100 µM:

```
corrected(t) = sample(t) − (control(t) − control(t_anchor))
```

You can re-anchor the control's t = 0 by clicking in the preview, and the corrected trace updates live.

### Skip calibration (fitting-only mode)

If your input files are already calibrated `[H₂O₂] (µM) vs time (s)` — for example because they came from another pipeline or were previously control-subtracted outside SensorFit — pass `--skip-calibration` and SensorFit becomes a fitting-only tool. The baseline and calibration phases are bypassed entirely; you start at interval selection.

**Run:**

```bash
python -m sensorfit.calibration_cli \
  --input-dir "/path/to/already-calibrated-folder" \
  --time-col 0 \
  --current-col 3 \
  --skip-calibration \
  --force
```

Use `--time-col` / `--current-col` to point at the right columns. The "current" column is interpreted as **µM H₂O₂** (no scaling applied) and copied straight into the calibrated trace.

**Sanity check:** on each file SensorFit prints the [H₂O₂] range and warns if the column you picked looks like raw current (|max| < 0.1) instead of pre-calibrated H₂O₂.

**What you still get:**

- Interval selection, fitting (Manual linear / Single exponential / Inactivation), control subtraction, Δ[H₂O₂]max, and back-extrapolation all work as normal.
- The Review screen hides the "Redo baseline" / "Redo calibration" buttons since they're not applicable.
- `fit_summary.xlsx` rows carry a new `calibration_skipped = True` column for these files (NaN for normal files), so you can filter them in Excel.

**Combined with control subtraction:** pre-calibrated controls work too — when you reach the per-interval subtraction step, pick a pre-calibrated control file via "Subtract new alongside" and its calibration phase is skipped as well.

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

### ModuleNotFoundError: No module named 'numpy' (or pandas / scipy / matplotlib)

SensorFit's dependencies aren't installed in the environment you're actually running. The fix is to reinstall them into `sensorfit_env`:

```bash
# Mac / Linux
source sensorfit_env/bin/activate
pip install -e .

# Windows
sensorfit_env\Scripts\activate
python -m pip install -e .
```

**Common cause — conda + venv both active.** If your prompt shows **both** `(sensorfit_env)` and `(base)`, Anaconda's `base` environment is active alongside the SensorFit venv, and the install may have landed in the wrong place. Check which Python you're really using:

```bash
which python     # Mac / Linux  → must contain .../sensorfit_env/bin/python
where python     # Windows      → must contain ...\sensorfit_env\Scripts\python.exe
```

If it does **not** point inside `sensorfit_env`, deactivate conda first (`conda deactivate`, possibly twice until `(base)` disappears), re-activate the venv, and run `pip install -e .` again. To stop conda auto-activating its base env in every new terminal: `conda config --set auto_activate_base false`.

Verify the fix with:

```bash
python -c "import numpy, pandas, scipy, matplotlib, openpyxl; print('all dependencies OK')"
```

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
