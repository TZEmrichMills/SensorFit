"""SensorFit - A package for fitting sensor timecourse data."""

__version__ = "0.3.0"

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
    ControlSubgroup,
    ControlSpec,
    show_grouping_ui,
    save_controls_manifest,
    load_controls_manifest,
    save_control_template,
    load_control_template,
    interpolate_control_to_grid,
    interactive_subtract,
    select_control_reference_interval,
)
from .back_extrap import compute_back_extrap
from .group_planning import (
    SampleState,
    build_subtraction_chain,
    planning_picker,
    preview_and_apply_subtraction,
    run_group_planning,
    offer_refit_for_corrected_intervals,
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
    "ControlSubgroup",
    "ControlSpec",
    "show_grouping_ui",
    "save_controls_manifest",
    "load_controls_manifest",
    "save_control_template",
    "load_control_template",
    "interpolate_control_to_grid",
    "interactive_subtract",
    "select_control_reference_interval",
    # Back-extrapolation (math only; standalone UI removed)
    "compute_back_extrap",
    # Group-planning exports
    "SampleState",
    "build_subtraction_chain",
    "planning_picker",
    "preview_and_apply_subtraction",
    "run_group_planning",
    "offer_refit_for_corrected_intervals",
]
