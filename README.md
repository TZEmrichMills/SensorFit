# SensorFit

Interactive tool for calibrating amperometric sensor traces to H₂O₂ concentrations and performing curve-fitting analysis.

## Contents

- [Quick Start](#quick-start)
- [What to Expect When You Run SensorFit](#what-to-expect-when-you-run-sensorfit)
- [Command Line Arguments](#command-line-arguments)
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

Results (calibrated data, interval Excel files, fit parameters) are saved under a `Calibrated/` folder. The original file is moved to a `Processed/` folder. A running `fit_summary.xlsx` is updated with all fit results across files. SensorFit then opens the next file and repeats from step 1.

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
