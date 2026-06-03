"""SensorFit - A package for fitting sensor timecourse data."""

__version__ = "0.2.0"

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
from .controls import (
    ControlGroup,
    show_grouping_ui,
    save_controls_manifest,
    load_controls_manifest,
    save_control_template,
    load_control_template,
    interpolate_control_to_grid,
    interactive_subtract,
    select_control_reference_interval,
)
from .residual_activity import (
    compute_residual_activity,
    tag_residual_activity_series,
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
    # Control-subtraction exports
    "ControlGroup",
    "show_grouping_ui",
    "save_controls_manifest",
    "load_controls_manifest",
    "save_control_template",
    "load_control_template",
    "interpolate_control_to_grid",
    "interactive_subtract",
    "select_control_reference_interval",
    # Residual-activity exports
    "compute_residual_activity",
    "tag_residual_activity_series",
]

