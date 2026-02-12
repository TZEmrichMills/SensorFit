"""SensorFit - A package for fitting sensor timecourse data."""

__version__ = "0.1.0"

from .models import MODEL_FUNCS
from .fitting import fit_IB, fit_Exponential, fit_GFI
from .utils import parse_models, r2_score, aic, bic
from .calibration import (
    load_trace,
    select_baseline,
    select_points,
    build_calibration,
    apply_calibration,
    select_intervals,
    build_interval_subsets,
    persist_interval_subsets,
    interactive_interval_fitting,
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
    "select_intervals",
    "build_interval_subsets",
    "persist_interval_subsets",
    "interactive_interval_fitting",
]

