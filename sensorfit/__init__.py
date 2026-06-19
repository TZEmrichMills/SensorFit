"""SensorFit - A package for fitting sensor timecourse data."""

__version__ = "0.4.0"

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
