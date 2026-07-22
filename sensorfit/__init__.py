"""SensorFit - A package for fitting sensor timecourse data."""

__version__ = "0.4.0"


def _check_dependencies() -> None:
    """Fail early with an actionable message if the scientific stack is
    missing, instead of a cryptic ``ModuleNotFoundError`` deep in a traceback.

    The most common cause is running SensorFit in an environment where
    ``pip install -e .`` never completed — often because conda's ``base`` env
    is active alongside ``sensorfit_env`` and the install landed elsewhere.
    """
    import importlib

    required = ["numpy", "pandas", "scipy", "matplotlib", "openpyxl"]
    missing = [m for m in required if importlib.util.find_spec(m) is None]
    if not missing:
        return

    import sys

    exe = sys.executable
    raise ModuleNotFoundError(
        "SensorFit's dependencies are not installed in the active Python "
        "environment.\n"
        f"  Missing: {', '.join(missing)}\n"
        f"  Active interpreter: {exe}\n\n"
        "Fix — activate the SensorFit environment and (re)install it:\n"
        "  macOS / Linux:\n"
        "    source sensorfit_env/bin/activate\n"
        "    pip install -e .\n"
        "  Windows:\n"
        "    sensorfit_env\\Scripts\\activate\n"
        "    python -m pip install -e .\n\n"
        "Tip: if your shell prompt shows BOTH (sensorfit_env) and (base), "
        "conda's base environment is also active. Run `which python` (macOS/"
        "Linux) or `where python` (Windows) and confirm it points INSIDE "
        "sensorfit_env before installing."
    )


_check_dependencies()

from .models import MODEL_FUNCS
from .fitting import fit_IB, fit_Exponential, fit_GFI
from .utils import parse_models, r2_score, aic, bic
from .calibration import (
    load_trace,
    select_baseline,
    select_points,
    build_calibration,
    apply_calibration,
    persist_interval_subsets,
)
from .controls import (
    average_controls_on_grid,
    deviation_from_anchor,
    save_control_template,
    load_control_template,
    interpolate_control_to_grid,
)
from .back_extrap import compute_back_extrap
from .interval_processor import (
    ProcessedInterval,
    FitRecord,
    DeltaMaxRecord,
    run_per_interval_flow,
    delta_max_from_fit,
    delta_max_from_linear,
    delta_max_from_point,
)

__all__ = [
    "MODEL_FUNCS",
    "fit_IB",
    "fit_Exponential",
    "fit_GFI",
    "parse_models",
    "r2_score",
    "aic",
    "bic",
    # Calibration exports
    "load_trace",
    "select_baseline",
    "select_points",
    "build_calibration",
    "apply_calibration",
    "persist_interval_subsets",
    # Control-template helpers (reused by per-interval subtraction)
    "average_controls_on_grid",
    "deviation_from_anchor",
    "save_control_template",
    "load_control_template",
    "interpolate_control_to_grid",
    # Back-extrapolation (math only)
    "compute_back_extrap",
    # Per-interval processor (the single processing flow)
    "ProcessedInterval",
    "FitRecord",
    "DeltaMaxRecord",
    "run_per_interval_flow",
    "delta_max_from_fit",
    "delta_max_from_linear",
    "delta_max_from_point",
]
