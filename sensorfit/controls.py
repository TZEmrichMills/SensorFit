"""Control-subtraction support: group controls with samples, subtract calibrated
control traces from sample traces during interactive processing.

This module provides:
- ``ControlGroup``: dataclass describing one control + its sample files.
- ``show_grouping_ui``: Qt window for assigning controls to samples at session start.
- ``save_controls_manifest`` / ``load_controls_manifest``: persist groupings under
  ``Calibrated/controls.json`` so an interrupted session can resume.
- ``save_control_template`` / ``load_control_template``: persist the calibrated
  reference interval that will be subtracted.
- ``interpolate_control_to_grid``: align a control template onto a sample's time
  grid (linear interp + bounded linear extrapolation on the tails).
- ``interactive_subtract``: matplotlib screen that previews + confirms the
  subtraction.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Reuse the Qt shim from calibration_editor so we depend on the same backend.
from .calibration_editor import _try_import_qt, QT_AVAILABLE, QT_LIB
from .calibration import (
    CALIBRATED_COLUMN,
    add_instruction_banner,
    create_small_button,
    truncate_filename,
)


CONTROL_TEMPLATES_DIRNAME = "_control_templates"
CONTROLS_MANIFEST_FILENAME = "controls.json"
MAX_EXTRAPOLATION_FRAC = 0.20  # 20 % of control duration


@dataclass
class ControlSpec:
    """One control file inside a sub-group.

    After the control has been processed and the user has chosen the reference
    interval, ``reference_interval_index`` and ``template_filename`` are
    populated.  Until then they are None.
    """

    file_name: str
    reference_interval_index: int | None = None
    template_filename: str | None = None

    def to_dict(self) -> dict:
        return {
            "file_name": self.file_name,
            "reference_interval_index": self.reference_interval_index,
            "template_filename": self.template_filename,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ControlSpec":
        return cls(
            file_name=d["file_name"],
            reference_interval_index=d.get("reference_interval_index"),
            template_filename=d.get("template_filename"),
        )


@dataclass
class ControlSubgroup:
    """A bundle of controls whose templates get **averaged** together at
    subtraction time.  At least one control per sub-group.  Sub-groups inside
    a ``ControlGroup`` are subtracted **sequentially**.
    """

    name: str  # display name, e.g. "Sub-group 1"
    controls: list[ControlSpec] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "controls": [c.to_dict() for c in self.controls],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ControlSubgroup":
        return cls(
            name=d.get("name", "Sub-group"),
            controls=[ControlSpec.from_dict(c) for c in d.get("controls", [])],
        )


@dataclass
class ControlGroup:
    """One experimental "group" — a set of controls (one or more sub-groups)
    plus a set of samples those controls apply to.

    Subtraction math for a sample: ``sample − avg(sub-group 0) − avg(sub-group 1) − …``
    Sub-groups are applied in list order.

    Attributes
    ----------
    name : str
        Display / manifest name.
    subgroups : list[ControlSubgroup]
        Ordered list of control sub-groups.  Empty list = no subtraction
        configured (the group is just a logical bundle).
    sample_files : list[str]
        Sample file names (bare names, relative to input dir).
    """

    name: str
    subgroups: list[ControlSubgroup] = field(default_factory=list)
    sample_files: list[str] = field(default_factory=list)

    # Backward-compat shim: some callers want the "primary" control file (for
    # logging or for the deprecated single-control API).  Returns the first
    # control of the first sub-group, or "" if none.
    @property
    def control_file(self) -> str:  # pragma: no cover - shim only
        if self.subgroups and self.subgroups[0].controls:
            return self.subgroups[0].controls[0].file_name
        return ""

    def all_control_files(self) -> list[str]:
        """Every control filename across all sub-groups, in order."""
        out = []
        for sg in self.subgroups:
            for c in sg.controls:
                out.append(c.file_name)
        return out

    def find_spec(self, file_name: str) -> tuple[int, ControlSpec] | None:
        """Return (subgroup_index, spec) for the named control, or None."""
        for i, sg in enumerate(self.subgroups):
            for c in sg.controls:
                if c.file_name == file_name:
                    return i, c
        return None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "subgroups": [sg.to_dict() for sg in self.subgroups],
            "sample_files": list(self.sample_files),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ControlGroup":
        # Auto-upgrade legacy single-control manifests:
        #   {name, control_file, sample_files, reference_interval_index?, template_filename?}
        if "control_file" in d and "subgroups" not in d:
            spec = ControlSpec(
                file_name=d["control_file"],
                reference_interval_index=d.get("reference_interval_index"),
                template_filename=d.get("template_filename"),
            )
            sg = ControlSubgroup(name="Sub-group 1", controls=[spec])
            return cls(
                name=d["name"],
                subgroups=[sg],
                sample_files=list(d.get("sample_files", [])),
            )
        return cls(
            name=d["name"],
            subgroups=[ControlSubgroup.from_dict(sg) for sg in d.get("subgroups", [])],
            sample_files=list(d.get("sample_files", [])),
        )


# ──────────────────────────────────────────────────────────────────────────
# Manifest I/O
# ──────────────────────────────────────────────────────────────────────────

def save_controls_manifest(
    groups: dict[str, ControlGroup], calibrated_dir: Path
) -> Path:
    """Write groups to Calibrated/controls.json."""
    calibrated_dir.mkdir(parents=True, exist_ok=True)
    out_path = calibrated_dir / CONTROLS_MANIFEST_FILENAME
    payload = {
        "version": 1,
        "groups": [g.to_dict() for g in groups.values()],
    }
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    return out_path


def load_controls_manifest(calibrated_dir: Path) -> dict[str, ControlGroup] | None:
    """Return the persisted manifest, or None if it doesn't exist."""
    in_path = calibrated_dir / CONTROLS_MANIFEST_FILENAME
    if not in_path.exists():
        return None
    with open(in_path) as fh:
        payload = json.load(fh)
    groups = {}
    for d in payload.get("groups", []):
        g = ControlGroup.from_dict(d)
        groups[g.name] = g
    return groups


# ──────────────────────────────────────────────────────────────────────────
# Template I/O
# ──────────────────────────────────────────────────────────────────────────

def control_templates_dir(calibrated_dir: Path) -> Path:
    return calibrated_dir / CONTROL_TEMPLATES_DIRNAME


def save_control_template(
    calibrated_dir: Path,
    group_name: str,
    time_values: np.ndarray,
    h2o2_values: np.ndarray,
) -> Path:
    """Persist a control's reference interval as a 2-column CSV, time zero-based."""
    templates = control_templates_dir(calibrated_dir)
    templates.mkdir(parents=True, exist_ok=True)
    safe_name = group_name.replace("/", "_").replace(" ", "_")
    out_path = templates / f"{safe_name}.csv"
    t = np.asarray(time_values, dtype=float)
    t0 = float(t[0]) if t.size else 0.0
    df = pd.DataFrame({
        "time_s": t - t0,
        CALIBRATED_COLUMN: np.asarray(h2o2_values, dtype=float),
    })
    df.to_csv(out_path, index=False)
    return out_path


def load_control_template(
    calibrated_dir: Path, template_filename: str
) -> tuple[np.ndarray, np.ndarray]:
    in_path = control_templates_dir(calibrated_dir) / template_filename
    df = pd.read_csv(in_path)
    return df["time_s"].to_numpy(dtype=float), df[CALIBRATED_COLUMN].to_numpy(dtype=float)


# ──────────────────────────────────────────────────────────────────────────
# Alignment + subtraction math
# ──────────────────────────────────────────────────────────────────────────

def interpolate_control_to_grid(
    control_t: np.ndarray,
    control_y: np.ndarray,
    sample_t: np.ndarray,
    max_extrap_frac: float = MAX_EXTRAPOLATION_FRAC,
) -> tuple[np.ndarray, dict]:
    """Resample control onto the sample's time grid.

    Inside the control's range: linear interpolation.
    Outside: linear extrapolation using the first/last 5 % of the control,
    bounded so we never extend past ``max_extrap_frac`` × control duration.

    Returns
    -------
    (interpolated, info)
        ``interpolated`` has the same length as ``sample_t``.  ``info`` is a
        dict with diagnostic counters (n_extrap_left, n_extrap_right,
        warning: str or None).
    """
    control_t = np.asarray(control_t, dtype=float)
    control_y = np.asarray(control_y, dtype=float)
    sample_t = np.asarray(sample_t, dtype=float)

    if control_t.size < 2:
        raise ValueError("Control template must have at least 2 samples.")

    t_min, t_max = control_t[0], control_t[-1]
    duration = t_max - t_min
    max_extend = duration * max_extrap_frac

    # Tail slopes from the first/last 5 % of the control
    n_tail = max(2, int(round(control_t.size * 0.05)))
    left_slope = float(
        np.polyfit(control_t[:n_tail], control_y[:n_tail], 1)[0]
    )
    right_slope = float(
        np.polyfit(control_t[-n_tail:], control_y[-n_tail:], 1)[0]
    )

    out = np.interp(sample_t, control_t, control_y)
    # np.interp clamps to endpoints — overwrite the clamped regions with linear
    # extrapolation, but only within max_extend of either tail.
    n_extrap_left = 0
    n_extrap_right = 0
    warning = None

    left_mask = sample_t < t_min
    if left_mask.any():
        dt = sample_t[left_mask] - t_min
        capped_dt = np.maximum(dt, -max_extend)  # most-negative is -max_extend
        out[left_mask] = control_y[0] + left_slope * capped_dt
        n_extrap_left = int(left_mask.sum())
        if (dt < -max_extend).any():
            warning = (
                f"Sample extends {-dt.min():.2f}s before control start; "
                f"capped at {max_extend:.2f}s of extrapolation."
            )

    right_mask = sample_t > t_max
    if right_mask.any():
        dt = sample_t[right_mask] - t_max
        capped_dt = np.minimum(dt, max_extend)
        out[right_mask] = control_y[-1] + right_slope * capped_dt
        n_extrap_right = int(right_mask.sum())
        if (dt > max_extend).any():
            msg = (
                f"Sample extends {dt.max():.2f}s past control end; "
                f"capped at {max_extend:.2f}s of extrapolation."
            )
            warning = warning + " " + msg if warning else msg

    return out, {
        "n_extrap_left": n_extrap_left,
        "n_extrap_right": n_extrap_right,
        "warning": warning,
    }


# ──────────────────────────────────────────────────────────────────────────
# Grouping UI (Qt)
# ──────────────────────────────────────────────────────────────────────────

def show_grouping_ui(
    files: list[Path],
    existing_groups: dict[str, ControlGroup] | None = None,
) -> dict[str, ControlGroup] | None:
    """Open a Qt dialog letting the user map controls to samples.

    Returns the dict of groups (name → ControlGroup), or None if the user
    cancelled.  Files that are not mentioned in any group are processed
    normally (no subtraction).
    """
    if not QT_AVAILABLE:
        ok, _, err = _try_import_qt()
        if not ok:
            raise ImportError(
                "Qt (PyQt5 or PySide6) is required for the control-grouping UI.\n"
                f"Install with: {sys.executable} -m pip install PyQt5\n\n"
                f"Import error: {err}"
            )

    if QT_LIB == "PyQt5":
        from PyQt5.QtWidgets import (
            QApplication, QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
            QListWidget, QListWidgetItem, QTreeWidget, QTreeWidgetItem,
            QLabel, QInputDialog, QMessageBox, QAbstractItemView,
        )
        from PyQt5.QtCore import Qt
    else:
        from PySide6.QtWidgets import (
            QApplication, QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
            QListWidget, QListWidgetItem, QTreeWidget, QTreeWidgetItem,
            QLabel, QInputDialog, QMessageBox, QAbstractItemView,
        )
        from PySide6.QtCore import Qt

    app = QApplication.instance() or QApplication(sys.argv)

    dialog = QDialog()
    dialog.setWindowTitle("SensorFit — Control / Sample Grouping")
    dialog.resize(1000, 650)

    layout = QVBoxLayout(dialog)
    layout.addWidget(QLabel(
        "Group controls with their samples.  Files left ungrouped are processed normally.\n"
        "• Each group can have one or more control sub-groups + a samples list.\n"
        "• Controls inside the same sub-group are AVERAGED before subtraction.\n"
        "• Different sub-groups are subtracted SEQUENTIALLY in order (top → bottom).\n"
        "• Click Done to begin processing — controls first, then samples, then per-group subtraction planning."
    ))

    body = QHBoxLayout()
    layout.addLayout(body, 1)

    # Left: unassigned files
    left_box = QVBoxLayout()
    body.addLayout(left_box, 1)
    left_box.addWidget(QLabel("Unassigned files"))
    file_list = QListWidget()
    file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
    left_box.addWidget(file_list, 1)

    # Right: groups tree
    right_box = QVBoxLayout()
    body.addLayout(right_box, 1)
    right_box.addWidget(QLabel("Groups → sub-groups (averaged) → controls;  Samples"))
    group_tree = QTreeWidget()
    group_tree.setHeaderLabels(["Tree", "Type"])
    group_tree.setColumnWidth(0, 400)
    right_box.addWidget(group_tree, 1)

    # We use Qt.UserRole metadata to identify what each tree item represents.
    # Roles: "group", "subgroup", "samples_holder", "control_file", "sample_file"
    TYPE_ROLE = Qt.UserRole
    META_ROLE = Qt.UserRole + 1  # used to store (group_name, subgroup_idx) etc.

    # State
    file_names = [f.name for f in files]
    groups: dict[str, ControlGroup] = {}
    if existing_groups:
        for g in existing_groups.values():
            groups[g.name] = ControlGroup(
                name=g.name,
                subgroups=[
                    ControlSubgroup(
                        name=sg.name,
                        controls=[
                            ControlSpec(
                                file_name=c.file_name,
                                reference_interval_index=c.reference_interval_index,
                                template_filename=c.template_filename,
                            )
                            for c in sg.controls
                        ],
                    )
                    for sg in g.subgroups
                ],
                sample_files=list(g.sample_files),
            )

    def assigned_names() -> set[str]:
        out = set()
        for g in groups.values():
            out.update(g.all_control_files())
            out.update(g.sample_files)
        return out

    def _set_item_type(item, type_str: str, meta=None) -> None:
        item.setData(0, TYPE_ROLE, type_str)
        item.setData(0, META_ROLE, meta)

    def refresh_views() -> None:
        assigned = assigned_names()
        file_list.clear()
        for name in file_names:
            if name in assigned:
                continue
            file_list.addItem(QListWidgetItem(name))

        group_tree.clear()
        for g in groups.values():
            grp_item = QTreeWidgetItem([g.name, "group"])
            grp_item.setForeground(0, Qt.darkBlue)
            _set_item_type(grp_item, "group", g.name)
            grp_item.setExpanded(True)

            for sg_idx, sg in enumerate(g.subgroups):
                tag = " (averaged)" if len(sg.controls) > 1 else ""
                sg_item = QTreeWidgetItem([f"{sg.name}{tag}", "sub-group"])
                sg_item.setForeground(0, Qt.darkGreen)
                _set_item_type(sg_item, "subgroup", (g.name, sg_idx))
                for spec in sg.controls:
                    label = spec.file_name
                    if spec.template_filename:
                        label += "  ✓"
                    c_item = QTreeWidgetItem([label, "control"])
                    _set_item_type(c_item, "control_file", (g.name, sg_idx, spec.file_name))
                    sg_item.addChild(c_item)
                grp_item.addChild(sg_item)
                sg_item.setExpanded(True)

            samples_item = QTreeWidgetItem(["Samples", ""])
            samples_item.setForeground(0, Qt.darkRed)
            _set_item_type(samples_item, "samples_holder", g.name)
            for s in g.sample_files:
                s_item = QTreeWidgetItem([s, "sample"])
                _set_item_type(s_item, "sample_file", (g.name, s))
                samples_item.addChild(s_item)
            grp_item.addChild(samples_item)
            samples_item.setExpanded(True)

            group_tree.addTopLevelItem(grp_item)

    # ── Tree helpers ──────────────────────────────────────────────────
    def _current_kind() -> tuple[str | None, object | None]:
        item = group_tree.currentItem()
        if item is None:
            return None, None
        return item.data(0, TYPE_ROLE), item.data(0, META_ROLE)

    def _group_of_selection() -> ControlGroup | None:
        kind, meta = _current_kind()
        if kind == "group":
            return groups.get(meta)
        if kind == "subgroup":
            return groups.get(meta[0])
        if kind == "samples_holder":
            return groups.get(meta)
        if kind == "control_file":
            return groups.get(meta[0])
        if kind == "sample_file":
            return groups.get(meta[0])
        return None

    # ── Button actions ────────────────────────────────────────────────
    def new_group(_=None) -> None:
        default = f"Group_{len(groups) + 1}"
        name, ok = QInputDialog.getText(dialog, "Group name", "Group name:", text=default)
        if not ok or not name.strip():
            return
        name = name.strip()
        if name in groups:
            QMessageBox.warning(dialog, "Group name", f"A group named '{name}' already exists.")
            return
        # Start with one empty sub-group so the user can immediately add controls.
        groups[name] = ControlGroup(
            name=name,
            subgroups=[ControlSubgroup(name="Sub-group 1", controls=[])],
            sample_files=[],
        )
        refresh_views()

    def new_subgroup(_=None) -> None:
        g = _group_of_selection()
        if g is None:
            QMessageBox.information(dialog, "Sub-group", "Select a group (or item inside one) on the right first.")
            return
        g.subgroups.append(
            ControlSubgroup(name=f"Sub-group {len(g.subgroups) + 1}", controls=[])
        )
        refresh_views()

    def add_files_as_controls(_=None) -> None:
        files_selected = [i.text() for i in file_list.selectedItems()]
        if not files_selected:
            QMessageBox.information(dialog, "Add as controls", "Select files on the left first.")
            return
        kind, meta = _current_kind()
        # Accept either a sub-group selection or a group selection (defaults
        # to the last sub-group of that group)
        if kind == "subgroup":
            g = groups[meta[0]]
            sg = g.subgroups[meta[1]]
        elif kind == "group":
            g = groups[meta]
            if not g.subgroups:
                g.subgroups.append(ControlSubgroup(name="Sub-group 1", controls=[]))
            sg = g.subgroups[-1]
        else:
            QMessageBox.information(
                dialog,
                "Add as controls",
                "Select the target sub-group (or its parent group) on the right first.",
            )
            return
        for fn in files_selected:
            sg.controls.append(ControlSpec(file_name=fn))
        refresh_views()

    def add_files_as_samples(_=None) -> None:
        files_selected = [i.text() for i in file_list.selectedItems()]
        if not files_selected:
            QMessageBox.information(dialog, "Add as samples", "Select files on the left first.")
            return
        g = _group_of_selection()
        if g is None:
            QMessageBox.information(dialog, "Add as samples", "Select a target group on the right first.")
            return
        for fn in files_selected:
            if fn not in g.sample_files:
                g.sample_files.append(fn)
        refresh_views()

    def move_subgroup(direction: int):
        def _act(_=None):
            kind, meta = _current_kind()
            if kind != "subgroup":
                QMessageBox.information(dialog, "Move", "Select a sub-group to reorder.")
                return
            g = groups[meta[0]]
            idx = meta[1]
            new_idx = idx + direction
            if 0 <= new_idx < len(g.subgroups):
                g.subgroups[idx], g.subgroups[new_idx] = g.subgroups[new_idx], g.subgroups[idx]
                # Refresh names so they match new positions
                for i, sg in enumerate(g.subgroups, start=1):
                    sg.name = f"Sub-group {i}"
                refresh_views()
        return _act

    def remove_selected(_=None) -> None:
        kind, meta = _current_kind()
        if kind is None:
            return
        if kind == "group":
            groups.pop(meta, None)
        elif kind == "subgroup":
            g = groups[meta[0]]
            del g.subgroups[meta[1]]
            for i, sg in enumerate(g.subgroups, start=1):
                sg.name = f"Sub-group {i}"
        elif kind == "control_file":
            g = groups[meta[0]]
            sg = g.subgroups[meta[1]]
            sg.controls = [c for c in sg.controls if c.file_name != meta[2]]
        elif kind == "sample_file":
            g = groups[meta[0]]
            g.sample_files.remove(meta[1])
        elif kind == "samples_holder":
            g = groups[meta]
            g.sample_files.clear()
        refresh_views()

    # Buttons
    btn_row1 = QHBoxLayout()
    btn_row2 = QHBoxLayout()
    layout.addLayout(btn_row1)
    layout.addLayout(btn_row2)
    b_new = QPushButton("New group")
    b_subg = QPushButton("New sub-group in selected group")
    b_addc = QPushButton("Add selected files → sub-group (controls)")
    b_adds = QPushButton("Add selected files → samples")
    b_up = QPushButton("↑ Move sub-group up")
    b_down = QPushButton("↓ Move sub-group down")
    b_rm = QPushButton("Remove selected")
    b_done = QPushButton("Done")
    b_cancel = QPushButton("Cancel")
    for b in (b_new, b_subg, b_addc, b_adds):
        btn_row1.addWidget(b)
    for b in (b_up, b_down, b_rm, b_done, b_cancel):
        btn_row2.addWidget(b)

    b_new.clicked.connect(new_group)
    b_subg.clicked.connect(new_subgroup)
    b_addc.clicked.connect(add_files_as_controls)
    b_adds.clicked.connect(add_files_as_samples)
    b_up.clicked.connect(move_subgroup(-1))
    b_down.clicked.connect(move_subgroup(+1))
    b_rm.clicked.connect(remove_selected)

    result = {"action": None}

    def on_done() -> None:
        # Sanity-check: drop sub-groups with zero controls so saved manifest stays clean
        for g in list(groups.values()):
            g.subgroups = [sg for sg in g.subgroups if sg.controls]
            # Optionally drop groups with zero controls AND zero samples
            if not g.subgroups and not g.sample_files:
                del groups[g.name]
        result["action"] = "done"
        dialog.accept()

    def on_cancel() -> None:
        result["action"] = "cancel"
        dialog.reject()

    b_done.clicked.connect(on_done)
    b_cancel.clicked.connect(on_cancel)

    refresh_views()
    dialog.exec_() if hasattr(dialog, "exec_") else dialog.exec()

    if result["action"] != "done":
        return None
    return groups


# ──────────────────────────────────────────────────────────────────────────
# Reference-interval picker for controls (uses matplotlib like the rest of
# the flow so the user feels at home)
# ──────────────────────────────────────────────────────────────────────────

def select_control_reference_interval(
    subsets: list,
    time_col: str,
    filename: str | None = None,
) -> int | None:
    """Ask the user which of a control file's intervals to use as the
    subtraction reference.  Returns the chosen interval's ``.index`` or None
    if the user cancelled.
    """
    if not subsets:
        return None
    if len(subsets) == 1:
        return subsets[0].index

    fig, ax = plt.subplots(figsize=(10, 6.4))
    plt.subplots_adjust(left=0.1, bottom=0.20, right=0.98, top=0.78)

    colours = plt.cm.tab10.colors
    for i, s in enumerate(subsets):
        c = colours[i % len(colours)]
        ax.plot(
            s.data[time_col].to_numpy(),
            s.data[CALIBRATED_COLUMN].to_numpy(),
            color=c,
            lw=1.4,
            label=f"Interval #{s.index} ({s.start_time:.1f}–{s.end_time:.1f} s)",
        )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("H2O2 (µM)")
    ax.legend(loc="best", fontsize=8)
    # Consolidate filename + instructions into a single banner so the title
    # and banner never overlap.
    header = ""
    if filename:
        header = f"{truncate_filename(filename)}\n"
    add_instruction_banner(
        fig,
        header + (
            "Pick the control-reference interval: choose which interval of this "
            "CONTROL file should be subtracted from the sample files in this "
            "group (click the matching button below)."
        ),
        y=0.97,
        width=95,
    )

    choice = {"idx": None}

    def make_picker(idx_value):
        def _on_click(_event):
            choice["idx"] = idx_value
            plt.close(fig)
        return _on_click

    # Button per interval, laid out across the bottom
    n = len(subsets)
    width = min(0.14, 0.84 / n)
    spacing = 0.86 / n
    btns = []
    for i, s in enumerate(subsets):
        ax_btn = fig.add_axes([0.08 + i * spacing, 0.04, width, 0.06])
        b = create_small_button(ax_btn, f"Use #{s.index}", "#90ee90", "#7cd47c")
        b.on_clicked(make_picker(s.index))
        btns.append(b)

    plt.show()
    plt.close(fig)
    return choice["idx"]


# ──────────────────────────────────────────────────────────────────────────
# Interactive subtraction preview
# ──────────────────────────────────────────────────────────────────────────

def interactive_subtract(
    sample_time: np.ndarray,
    sample_h2o2: np.ndarray,
    control_t: np.ndarray,
    control_y: np.ndarray,
    filename: str | None = None,
    group_name: str | None = None,
):
    """Show the sample trace + aligned control trace + subtracted result.

    Returns
    -------
    one of:
      np.ndarray  — the corrected H2O2 trace (sample − control).
      "skip"       — user opted to skip subtraction for this sample.
      "back"       — user wants to return to the previous phase.
    """
    sample_time = np.asarray(sample_time, dtype=float)
    sample_h2o2 = np.asarray(sample_h2o2, dtype=float)
    control_t = np.asarray(control_t, dtype=float)
    control_y = np.asarray(control_y, dtype=float)

    # Anchor: by default zero the control time at the sample's t-min.
    # The user can re-anchor by clicking on the sample where the control
    # should "start".
    anchor = {"t0": float(sample_time[0])}

    fig, ax = plt.subplots(2, 1, figsize=(11, 7.4), sharex=True)
    plt.subplots_adjust(left=0.1, bottom=0.18, right=0.98, top=0.80, hspace=0.25)

    line_sample, = ax[0].plot(sample_time, sample_h2o2, color="tab:green", lw=1.3, label="Sample (calibrated)")
    line_control, = ax[0].plot([], [], color="tab:red", lw=1.2, alpha=0.75, label="Control template (aligned)")
    # Anchor markers: a tall dashed line spanning each subplot's full height
    # plus a triangle on the time axis.  Much more visible than the implicit
    # "where the red line begins".
    anchor_line_top = ax[0].axvline(
        anchor["t0"], color="#8B008B", lw=1.8, ls="--", alpha=0.9, label="Anchor (control t=0)",
    )
    anchor_line_bot = ax[1].axvline(
        anchor["t0"], color="#8B008B", lw=1.8, ls="--", alpha=0.9,
    )
    anchor_text = ax[0].annotate(
        "", xy=(anchor["t0"], 0), xycoords=("data", "axes fraction"),
        xytext=(6, 6), textcoords="offset points",
        color="#8B008B", fontsize=9, fontweight="bold",
    )
    ax[0].set_ylabel("H2O2 (µM)")
    ax[0].legend(loc="upper right", fontsize=9)

    line_diff, = ax[1].plot([], [], color="tab:blue", lw=1.3, label="Sample − Control")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("H2O2 (µM, corrected)")
    ax[1].axhline(0.0, color="grey", lw=0.6, ls=":")
    ax[1].legend(loc="upper right", fontsize=9)

    warning_text = ax[0].text(
        0.02, 0.96, "", transform=ax[0].transAxes,
        fontsize=8, va="top", color="darkred",
    )

    # Combine filename, group, and instructions into a single banner so they
    # never overlap.  No separate suptitle.
    header_parts = []
    if filename:
        header_parts.append(truncate_filename(filename))
    if group_name:
        header_parts.append(f"control group: {group_name}")
    header_str = "  |  ".join(header_parts) if header_parts else ""
    banner_text = (
        (header_str + "\n") if header_str else ""
    ) + (
        "Top: sample (green) and control (red).  Bottom: sample − control.  "
        "Click on the upper plot to re-anchor the control's t=0 (purple dashed line).  "
        "When alignment looks right, click Accept & subtract."
    )
    add_instruction_banner(fig, banner_text, y=0.985, width=110)

    def redraw() -> None:
        # Shift control onto sample's time axis at the current anchor.
        shifted_t = control_t + anchor["t0"]
        interp, info = interpolate_control_to_grid(shifted_t, control_y, sample_time)
        diff = sample_h2o2 - interp
        line_control.set_data(sample_time, interp)
        line_diff.set_data(sample_time, diff)
        # Update anchor markers
        anchor_line_top.set_xdata([anchor["t0"], anchor["t0"]])
        anchor_line_bot.set_xdata([anchor["t0"], anchor["t0"]])
        anchor_text.set_text(f"anchor t={anchor['t0']:.2f} s")
        anchor_text.xy = (anchor["t0"], 0)
        for a in ax:
            a.relim()
            a.autoscale_view()
        warning_text.set_text(info["warning"] or "")
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes is not ax[0] or event.button != 1 or event.xdata is None:
            return
        toolbar = fig.canvas.toolbar
        if toolbar is not None:
            mode = getattr(toolbar, "mode", "")
            if mode in ("zoom rect", "pan/zoom", "zoom", "pan"):
                return
        anchor["t0"] = float(event.xdata)
        redraw()
        print(f"Re-anchored control at t={anchor['t0']:.3f} s")

    fig.canvas.mpl_connect("button_press_event", on_click)

    decision = {"value": None}

    def on_accept(_event=None) -> None:
        decision["value"] = "accept"
        plt.close(fig)

    def on_skip(_event=None) -> None:
        decision["value"] = "skip"
        plt.close(fig)

    def on_back(_event=None) -> None:
        decision["value"] = "back"
        plt.close(fig)

    def on_reset(_event=None) -> None:
        anchor["t0"] = float(sample_time[0])
        redraw()
        print("Anchor reset to sample start.")

    ax_accept = fig.add_axes([0.15, 0.03, 0.18, 0.05])
    ax_reset = fig.add_axes([0.35, 0.03, 0.14, 0.05])
    ax_skip = fig.add_axes([0.51, 0.03, 0.18, 0.05])
    ax_back = fig.add_axes([0.71, 0.03, 0.14, 0.05])
    btn_accept = create_small_button(ax_accept, "Accept & subtract", "#90ee90", "#7cd47c")
    btn_reset = create_small_button(ax_reset, "Reset anchor", "0.9", "0.8")
    btn_skip = create_small_button(ax_skip, "Skip subtraction", "#ffcc99", "#ffaa66")
    btn_back = create_small_button(ax_back, "Back", "#ddddff", "#bbbbff")
    btn_accept.on_clicked(on_accept)
    btn_reset.on_clicked(on_reset)
    btn_skip.on_clicked(on_skip)
    btn_back.on_clicked(on_back)

    redraw()
    plt.show()
    plt.close(fig)

    if decision["value"] == "accept":
        shifted_t = control_t + anchor["t0"]
        interp, _ = interpolate_control_to_grid(shifted_t, control_y, sample_time)
        return sample_h2o2 - interp
    if decision["value"] == "skip":
        return "skip"
    return "back"
