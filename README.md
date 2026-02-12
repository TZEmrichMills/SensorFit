# SensorFit - Calibration and Fitting Tool

A Python tool for calibrating amperometric traces to H2O2 concentrations and performing curve fitting analysis.

## Requirements

- **Python 3.8 or higher** (Python 3.9+ recommended)
- **Windows, Mac, or Linux**

## Installation

### Step 1: Check Python Installation

Open **Command Prompt** (Windows) or **Terminal** (Mac/Linux) and check your Python version:

```cmd
python --version
```

You should see something like `Python 3.9.7` or higher. If you see an error or a version below 3.8, you need to [install Python](https://www.python.org/downloads/) first.

**Windows users:** If `python` doesn't work, try `py`:
```cmd
py --version
```

### Step 2: Navigate to SensorFit Directory

Open **Command Prompt** (Windows) or **Terminal** (Mac/Linux) and navigate to where you've placed the SensorFit folder:

**Windows:**
```cmd
cd C:\path\to\SensorFit
```

**Mac/Linux:**
```bash
cd /path/to/SensorFit
```

**⚠️ CRITICAL for Windows users:** 
- **DO NOT use OneDrive, Dropbox, or other cloud-synced folders** - these will cause "Access is denied" errors
- **Use a local folder instead**, such as:
  - `C:\Users\YourName\Projects\SensorFit`
  - `C:\Projects\SensorFit`
  - `D:\SensorFit` (if you have a D: drive)
- If your SensorFit folder is currently in OneDrive/Dropbox, **move it to a local folder before proceeding**

### Step 3: Create a Virtual Environment

A virtual environment keeps SensorFit's dependencies separate from other Python projects.

**Windows (Command Prompt):**
```cmd
python -m venv sensorfit_env
```

**Mac/Linux:**
```bash
python3 -m venv sensorfit_env
```

**Windows users:** If `python` doesn't work, use `py`:
```cmd
py -m venv sensorfit_env
```

This creates a folder called `sensorfit_env` with a fresh Python environment.

### Step 4: Activate the Virtual Environment

**Windows (Command Prompt - Recommended):**
```cmd
sensorfit_env\Scripts\activate
```

**Mac/Linux:**
```bash
source sensorfit_env/bin/activate
```

**Windows (PowerShell):** If you're using PowerShell and get an execution policy error, either:
- Use Command Prompt instead (recommended - no issues), or
- Run this in PowerShell: `Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process`

After activation, you should see `(sensorfit_env)` at the beginning of your command prompt.

### Step 5: Upgrade pip

```cmd
python -m pip install --upgrade pip
```

### Step 6: Install SensorFit

**Before installing:** Make sure you're NOT in a OneDrive, Dropbox, or other cloud-synced folder. If you are, move the SensorFit folder to a local directory like `C:\Users\YourName\Projects\SensorFit` first.

Run this command (don't forget the "." at the end):

```cmd
python -m pip install -e .
```

**If you get "Access is denied" error**, try these solutions in order:

**Quick Fix 1:** Try without editable mode (if `-e` is causing issues):
```cmd
python -m pip install .
```

**Quick Fix 2:** Run Command Prompt as Administrator:
- Close your current Command Prompt
- Right-click Command Prompt → "Run as administrator"
- Navigate back: `cd C:\path\to\SensorFit`
- Activate: `sensorfit_env\Scripts\activate`
- Try: `python -m pip install -e .`

**Quick Fix 3:** If you're in OneDrive/Dropbox, move the project:
- Copy SensorFit folder to `C:\Users\YourName\Projects\SensorFit`
- Open Command Prompt in the new location
- Activate: `sensorfit_env\Scripts\activate`
- Try: `python -m pip install -e .`

**If none of the above work**, see the detailed [Troubleshooting section](#access-is-denied-error-at-step-6-pip-install) below for more solutions.

This will:
- Install the SensorFit package
- Automatically install all required dependencies (numpy, pandas, matplotlib, scipy, PyQt5, openpyxl)

**Note:** The installation may take a few minutes. If you continue to have issues, see the detailed [Troubleshooting](#troubleshooting) section below.

### Step 7: Verify Installation

Test that everything is installed correctly:

```cmd
python -c "import sensorfit; print('SensorFit installed successfully!')"
```

You should see: `SensorFit installed successfully!`

Or test the command-line tool:

```cmd
python -m sensorfit.calibration_cli --help
```

You should see help text for the calibration tool.

---

## Usage

### Running the Calibration Tool

1. **Activate your virtual environment** (if not already activated):
   ```cmd
   sensorfit_env\Scripts\activate    # Windows
   source sensorfit_env/bin/activate # Mac/Linux
   ```

2. **Run the calibration tool:**
   ```cmd
   python -m sensorfit.calibration_cli --input-dir "C:\SensorData" --num-points 6 --calibration-values "0,20,40,60,80,100" --force
   ```

**⚠️ Important:** Always include `--force` to overwrite existing output files. Without it, the program will stop if output files already exist.

**Note:** You can run this command from any directory once SensorFit is installed.

### Command Line Arguments

- `--input-dir`: Directory containing input Excel files (required)
- `--output-dir`: Output directory (default: same as input-dir)
- `--pattern`: Glob pattern for input files (default: searches for *.xlsx, *.txt, *.csv)
- `--time-col`: Column index for time data (default: 0)
- `--current-col`: Column index for current data (default: 1)
- `--num-points`: Number of calibration points (default: 6)
- `--window`: Number of samples to average around each selection (default: 50)
- `--calibration-values`: Comma-separated µM H2O2 concentrations (default: "0,20,40,60,80,100")
- `--force`: **⚠️ IMPORTANT:** Overwrite existing output files. Always include this flag to avoid errors when output files already exist.

### Example

**Note:** Always include `--force` to overwrite existing files.

```cmd
python -m sensorfit.calibration_cli --input-dir "C:\SensorData" --output-dir "C:\Results" --num-points 6 --calibration-values "0,20,40,60,80,100" --force
```

---

## Features

- **Interactive Baseline Selection**: Click two points to define and correct baseline drift, or skip if the baseline already looks good
- **Calibration Point Selection**: Select calibration points with visual feedback
- **Calibration Value Editor**: Edit calibration values and number of points using a user-friendly Qt-based table interface
- **Interval Selection**: Select time intervals for detailed analysis
- **Curve Fitting**: Fit one or more models (Inactivation, Exponential, Gompertz-like, Linear Initial Rate) to selected intervals
- **Excel Export**: Export calibrated data and fit results to Excel files
- **Summary Reports**: Automatic generation of fit summary Excel files
- **Interval Collection**: Combine all calibrated intervals from a batch run into a single Excel file for easy comparison

---

## Post-processing: Collecting Intervals

After running SensorFit on a set of files, you can combine all calibrated interval
traces into a single Excel workbook for easy plotting and comparison.

### What it does

- Scans `Calibrated/*_intervals/interval_*.xlsx` inside a given directory
- Zero-bases every interval so they all start at t = 0
- Merges them side-by-side on a common time grid
- Column headers are derived from the folder/file names (e.g. `Nc-0-2 #1`, `Ncmet-0_5-3 #1`)
- Shorter traces are padded with empty cells

### Usage

Make sure your virtual environment is activated, then run:

```bash
python -m sensorfit.collect_intervals /path/to/experiment/directory
```

This writes `all_intervals.xlsx` in the given directory.

To specify a custom output path:

```bash
python -m sensorfit.collect_intervals /path/to/experiment/directory -o combined.xlsx
```

### Example

```bash
python -m sensorfit.collect_intervals "C:\SensorData\MyExperiment" --force
```

The resulting Excel file has:

| Time (s) | Sample-A #1 | Sample-B #1 | Sample-C #1 | ... |
|----------|-------------|-------------|-------------|-----|
| 0.00     | 98.29       | 98.14       | 98.59       | ... |
| 0.08     | 98.05       | 98.14       | 98.74       | ... |
| ...      | ...         | ...         | ...         | ... |

---

## Troubleshooting

### "python is not recognized" or "python: command not found"

**Problem:** Python is not installed or not in your PATH.

**Solution:**
1. Download and install Python from [python.org](https://www.python.org/downloads/)
2. **Important:** During installation, check the box "Add Python to PATH"
3. Restart Command Prompt/Terminal and try again

**Windows alternative:** Try using `py` instead of `python`:
```cmd
py --version
py -m venv sensorfit_env
```

### ModuleNotFoundError: No module named 'sensorfit'

**Problem:** The package is not installed.

**Solution:**
1. Make sure you're in the SensorFit directory
2. Make sure your virtual environment is activated (you should see `(sensorfit_env)` in your prompt)
3. Run: `pip install -e .`

### "Access is denied" Error at Step 6 (pip install)

**Problem:** Getting "Access is denied" when running `pip install -e .` on Windows.

**Most Common Causes:**
1. Project is in OneDrive/Dropbox/cloud storage folder
2. Insufficient permissions
3. Antivirus blocking pip
4. File locking issues

**Solutions (try in this order):**

**Solution 1: Use `python -m pip` instead of `pip`**
```cmd
python -m pip install -e .
```
This is more reliable on Windows and uses the correct Python installation.

**Solution 2: Check if you're in OneDrive/Dropbox**
- Look at your current directory path - does it contain "OneDrive" or "Dropbox"?
- If yes, **move the project to a local folder:**
  1. Copy the entire SensorFit folder to `C:\Users\YourName\Projects\SensorFit`
  2. Open Command Prompt and navigate to the new location
  3. Activate environment: `sensorfit_env\Scripts\activate`
  4. Try installation again: `python -m pip install -e .`

**Solution 3: Run Command Prompt as Administrator**
1. Close your current Command Prompt
2. Right-click Command Prompt → "Run as administrator"
3. Click "Yes" when prompted
4. Navigate to SensorFit: `cd C:\path\to\SensorFit`
5. Activate environment: `sensorfit_env\Scripts\activate`
6. Try again: `python -m pip install -e .`

**Solution 4: Install without editable mode**
If editable mode (`-e`) is causing issues, try normal installation:
```cmd
python -m pip install .
```
Note: With normal installation, code changes won't be reflected until you reinstall.

**Solution 5: Check antivirus software**
- Temporarily disable antivirus software
- Or add Python and pip to antivirus exceptions
- Try installation again

**Solution 6: Check for file locks**
- Close any programs that might be using files in the SensorFit folder
- Close any file explorers showing the SensorFit folder
- Try installation again

**Solution 7: Verify folder permissions**
1. Right-click the SensorFit folder → Properties → Security tab
2. Make sure your user account has "Full control" or at least "Modify" permissions
3. If not, click "Edit" and grant yourself full control

**Still not working?** Try a completely fresh installation:
1. Move SensorFit to `C:\Users\YourName\Projects\SensorFit` (local folder, not cloud)
2. Delete the `sensorfit_env` folder if it exists
3. Start over from Step 3 (create new virtual environment)
4. Use `python -m pip install -e .` instead of `pip install -e .`

### PowerShell Execution Policy Error

**Problem:** When activating venv in PowerShell, you see:
```
cannot be loaded because running scripts is disabled on this system
```

**Solution (Easiest):** Use **Command Prompt** instead of PowerShell:
```cmd
sensorfit_env\Scripts\activate
```

Command Prompt doesn't have execution policy restrictions.

**Alternative:** If you must use PowerShell, run this once:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### FileExistsError: "already exists (use --force to overwrite)"

**Problem:** You see an error like:
```
FileExistsError: C:\SensorData\Calibrated\file_calibrated.xlsx already exists (use --force to overwrite)
```

**Solution:** Always include the `--force` flag in your command:
```cmd
python -m sensorfit.calibration_cli --input-dir "C:\SensorData" --num-points 6 --calibration-values "0,20,40,60,80,100" --force
```

The `--force` flag tells the program to overwrite existing output files instead of stopping with an error.

### PyQt5 Installation Fails

**Problem:** PyQt5 won't install (common on some systems).

**Solution:** Install PySide6 instead (the program will automatically use it):
```cmd
pip install PySide6
```

Then continue with the SensorFit installation:
```cmd
pip install -e .
```

### Installation Takes Too Long or Hangs

**Problem:** Installation seems stuck.

**Solutions:**

1. **Check your internet connection** - pip needs to download packages
2. **Try upgrading pip first:**
   ```cmd
   python -m pip install --upgrade pip
   ```
3. **Use a different package index (if behind a firewall):**
   ```cmd
   pip install -e . -i https://pypi.org/simple
   ```

### "No module named 'sensorfit'" After Installation

**Problem:** Installation seemed successful but import fails.

**Solutions:**

1. **Check you're in the right environment:**
   ```cmd
   python -c "import sys; print(sys.executable)"
   ```
   This should show a path containing `sensorfit_env`. If not, activate the environment.

2. **Verify installation location:**
   ```cmd
   python -c "import sys; print('\n'.join(sys.path))"
   ```
   Make sure the SensorFit directory or site-packages is listed.

3. **Reinstall:**
   ```cmd
   pip uninstall sensorfit
   pip install -e .
   ```

### Still Having Issues?

1. **Check Python version:**
   ```cmd
   python --version
   ```
   Must be 3.8 or higher (3.9+ recommended).

2. **Verify virtual environment is activated:**
   - You should see `(sensorfit_env)` at the start of your command prompt
   - If not, run: `sensorfit_env\Scripts\activate` (Windows) or `source sensorfit_env/bin/activate` (Mac/Linux)

3. **Check you're in the SensorFit directory:**
   ```cmd
   dir    # Windows (should see setup.py, README.md, sensorfit folder)
   ls     # Mac/Linux
   ```

4. **Try a fresh installation:**
   ```cmd
   # Deactivate environment
   deactivate
   
   # Delete old environment
   rmdir /s sensorfit_env    # Windows
   rm -rf sensorfit_env      # Mac/Linux
   
   # Start over from Step 3
   ```

---

## Quick Reference

### Activating the Environment (Do this each time you use SensorFit)

**Windows (Command Prompt):**
```cmd
cd C:\path\to\SensorFit
sensorfit_env\Scripts\activate
```

**Mac/Linux:**
```bash
cd /path/to/SensorFit
source sensorfit_env/bin/activate
```

### Running the Tool

**Always include `--force` to overwrite existing files:**

```cmd
python -m sensorfit.calibration_cli --input-dir "C:\SensorData" --num-points 6 --calibration-values "0,20,40,60,80,100" --force
```

### Deactivating the Environment

When you're done, you can deactivate:
```cmd
deactivate
```

---

## Installation Checklist

Before running SensorFit, make sure:

- [ ] Python 3.8+ is installed (`python --version`)
- [ ] You're in the SensorFit directory
- [ ] Virtual environment is created (`sensorfit_env` folder exists)
- [ ] Virtual environment is activated (see `(sensorfit_env)` in prompt)
- [ ] SensorFit is installed (`pip install -e .` completed successfully)
- [ ] Installation verified (`python -c "import sensorfit"` works)

---

## Getting Help

If you continue to have issues:

1. Make sure you've followed all steps in order
2. Check the [Troubleshooting](#troubleshooting) section above
3. Verify your Python version is 3.8 or higher
4. Ensure you're using Command Prompt (Windows) or Terminal (Mac/Linux)
5. Make sure the project is not in a cloud-synced folder (OneDrive, Dropbox, etc.)

For more information:
- [Python venv documentation](https://docs.python.org/3/library/venv.html)
- [Python installation guide](https://www.python.org/about/gettingstarted/)
