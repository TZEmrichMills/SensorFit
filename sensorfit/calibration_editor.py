"""GUI module for editing calibration values during calibration workflow."""

from __future__ import annotations

import sys
from typing import Callable

# Try to import Qt - prefer PyQt5, fall back to PySide6
# We'll do lazy imports at runtime to avoid issues with different Python environments
QT_AVAILABLE = False
QT_LIB = None
QT_IMPORT_ERROR = None

def _try_import_qt():
    """Try to import Qt libraries. Returns (success, lib_name, error_msg)."""
    global QT_AVAILABLE, QT_LIB, QT_IMPORT_ERROR
    
    # Try PyQt5 first
    try:
        from PyQt5.QtWidgets import (
            QApplication, QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
            QTableWidget, QTableWidgetItem, QLabel, QMessageBox, QStyledItemDelegate, QLineEdit
        )
        from PyQt5.QtCore import Qt
        QT_AVAILABLE = True
        QT_LIB = "PyQt5"
        QT_IMPORT_ERROR = None
        return True, "PyQt5", None
    except ImportError as e:
        pyqt5_error = str(e)
    
    # Try PySide6 as fallback
    try:
        from PySide6.QtWidgets import (
            QApplication, QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
            QTableWidget, QTableWidgetItem, QLabel, QMessageBox, QStyledItemDelegate, QLineEdit
        )
        from PySide6.QtCore import Qt
        QT_AVAILABLE = True
        QT_LIB = "PySide6"
        QT_IMPORT_ERROR = None
        return True, "PySide6", None
    except ImportError as e2:
        pyside6_error = str(e2)
    
    # Both failed
    QT_AVAILABLE = False
    QT_LIB = None
    QT_IMPORT_ERROR = f"PyQt5: {pyqt5_error}, PySide6: {pyside6_error}"
    return False, None, QT_IMPORT_ERROR

# Try importing at module load time
_try_import_qt()


def edit_calibration_values(
    current_values: list[float],
    num_points_ref: dict[str, int],
    selected_indices: list[int],
    selected_markers: list,
    update_calibration_callback: Callable[[list[float], bool], None] | None = None,
    filename: str | None = None,
) -> tuple[list[float], bool] | None:
    """
    Open a Qt GUI window to edit calibration values.
    
    Parameters:
    -----------
    current_values : list[float]
        Current calibration values (µM H2O2 concentrations)
    num_points_ref : dict[str, int]
        Mutable reference to number of points (will be updated if count changes)
    selected_indices : list[int]
        Currently selected point indices (may be cleared if count decreases)
    selected_markers : list
        List of marker objects (may be cleared if count decreases)
    update_calibration_callback : Callable[[list[float], bool], None] | None
        Callback function to update calibration values (values, apply_to_all)
    filename : str | None
        Optional filename for display
    
    Returns:
    --------
    tuple[list[float], bool] | None
        (new_values, apply_to_all) if saved, None if cancelled
    """
    # Import sys at the start (needed for QApplication)
    import sys
    
    # Re-check Qt availability at runtime in case imports failed at module load
    if not QT_AVAILABLE:
        # Try importing again at runtime
        success, lib_name, error_msg = _try_import_qt()
        
        if not success:
            # Provide helpful diagnostic information
            error_msg = (
                "Qt (PyQt5 or PySide6) is required for the calibration editor.\n\n"
                f"Python executable: {sys.executable}\n"
                f"Python version: {sys.version.split()[0]}\n"
                f"Python path (first 3 entries): {sys.path[:3]}\n\n"
                "Please ensure PyQt5 is installed in the same Python environment you're using:\n"
                f"  {sys.executable} -m pip install PyQt5\n\n"
                "Or install PySide6:\n"
                f"  {sys.executable} -m pip install PySide6"
            )
            if QT_IMPORT_ERROR:
                error_msg += f"\n\nImport error details: {QT_IMPORT_ERROR}"
            raise ImportError(error_msg)
    
    # Import Qt classes (they should already be imported, but ensure they're available)
    if QT_LIB == "PyQt5":
        from PyQt5.QtWidgets import (
            QApplication, QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
            QTableWidget, QTableWidgetItem, QLabel, QMessageBox, QStyledItemDelegate, QLineEdit
        )
        from PyQt5.QtCore import Qt
    elif QT_LIB == "PySide6":
        from PySide6.QtWidgets import (
            QApplication, QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
            QTableWidget, QTableWidgetItem, QLabel, QMessageBox, QStyledItemDelegate, QLineEdit
        )
        from PySide6.QtCore import Qt
    
    # Get or create QApplication instance
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    # Create dialog
    dialog = QDialog()
    dialog.setWindowTitle("Edit Calibration Values")
    if filename:
        # Truncate filename if too long
        max_length = 30
        if len(filename) > max_length:
            display_name = filename[:max_length] + "..."
        else:
            display_name = filename
        dialog.setWindowTitle(f"{display_name} - Edit Calibration Values")
    
    dialog.setMinimumSize(500, 400)
    dialog.resize(600, 500)
    
    # Main layout
    main_layout = QVBoxLayout(dialog)
    
    # Title label
    title_text = "Edit Calibration Values"
    if filename:
        max_length = 30
        if len(filename) > max_length:
            display_name = filename[:max_length] + "..."
        else:
            display_name = filename
        title_text = f"{display_name} - {title_text}"
    
    title_label = QLabel(title_text)
    title_label.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
    main_layout.addWidget(title_label)
    
    # Instructions label
    inst_label = QLabel(
        "Instructions: Click on a cell to edit the value (text will be automatically selected). "
        "Use 'Add' to add more calibration points, 'Remove' to remove the last one. "
        "Choose 'Apply to This File' to apply changes only to current file, "
        "or 'Apply to this and all subsequent files' to apply to all remaining files."
    )
    inst_label.setStyleSheet("font-style: italic; padding: 8px; font-size: 10px; background-color: #FFF9C4; border-radius: 5px;")
    inst_label.setWordWrap(True)
    main_layout.addWidget(inst_label)
    
    # Working copy of values
    edit_values = current_values.copy()
    
    # Create table
    table = QTableWidget()
    table.setColumnCount(2)
    table.setHorizontalHeaderLabels(["Plateau #", "µM H₂O₂"])
    table.setRowCount(len(edit_values))
    
    # Populate table
    for i, val in enumerate(edit_values):
        # Plateau number (read-only)
        plateau_item = QTableWidgetItem(str(i + 1))
        plateau_item.setFlags(plateau_item.flags() & ~Qt.ItemIsEditable)
        plateau_item.setTextAlignment(Qt.AlignCenter)
        table.setItem(i, 0, plateau_item)
        
        # Value (editable)
        value_item = QTableWidgetItem(f"{val:.1f}")
        value_item.setTextAlignment(Qt.AlignCenter)
        table.setItem(i, 1, value_item)
    
    # Custom delegate that selects all text when editing starts
    class SelectAllLineEdit(QLineEdit):
        """QLineEdit that selects all text on initial focus."""
        def __init__(self, parent=None):
            super().__init__(parent)
            self._select_all_on_focus = True

        def focusInEvent(self, event):
            super().focusInEvent(event)
            if self._select_all_on_focus:
                self.selectAll()
                self._select_all_on_focus = False

        def focusOutEvent(self, event):
            super().focusOutEvent(event)
            self._select_all_on_focus = True

        def keyPressEvent(self, event):
            self._select_all_on_focus = False
            super().keyPressEvent(event)

        def mousePressEvent(self, event):
            self._select_all_on_focus = False
            super().mousePressEvent(event)

    class SelectAllDelegate(QStyledItemDelegate):
        """Custom delegate that selects all text when editing starts."""
        def createEditor(self, parent, option, index):
            """Create editor."""
            return SelectAllLineEdit(parent)
        
        def setEditorData(self, editor, index):
            """Set editor data and select all text."""
            super().setEditorData(editor, index)
            editor.selectAll()
    
    # Set custom delegate for the value column (column 1)
    delegate = SelectAllDelegate()
    table.setItemDelegateForColumn(1, delegate)
    
    # Style table
    table.setAlternatingRowColors(True)
    table.setSelectionBehavior(QTableWidget.SelectItems)
    table.horizontalHeader().setStretchLastSection(True)
    table.verticalHeader().setVisible(False)
    # Enable editing on single click, double click, or key press
    table.setEditTriggers(QTableWidget.DoubleClicked | QTableWidget.SelectedClicked | QTableWidget.AnyKeyPressed)
    
    # Handle cell clicks to start editing
    def on_cell_clicked(row, col):
        """Start editing when a value cell is clicked."""
        if col == 1:  # Only for value column
            item = table.item(row, col)
            if item:
                table.editItem(item)
    
    table.cellClicked.connect(on_cell_clicked)
    
    main_layout.addWidget(table)
    
    # Button layout
    button_layout = QHBoxLayout()
    
    # Add/Remove buttons
    add_button = QPushButton("➕ Add")
    add_button.setStyleSheet("background-color: #81C784; color: white; font-weight: bold; padding: 4px; font-size: 10px;")
    remove_button = QPushButton("➖ Remove")
    remove_button.setStyleSheet("background-color: #E57373; color: white; font-weight: bold; padding: 4px; font-size: 10px;")
    
    def on_add_row():
        """Add a new calibration value row."""
        if len(edit_values) > 0:
            new_val = edit_values[-1] + 20.0
        else:
            new_val = 0.0
        edit_values.append(new_val)
        
        # Add row to table
        row = table.rowCount()
        table.insertRow(row)
        
        # Plateau number
        plateau_item = QTableWidgetItem(str(row + 1))
        plateau_item.setFlags(plateau_item.flags() & ~Qt.ItemIsEditable)
        plateau_item.setTextAlignment(Qt.AlignCenter)
        table.setItem(row, 0, plateau_item)
        
        # Value
        value_item = QTableWidgetItem(f"{new_val:.1f}")
        value_item.setTextAlignment(Qt.AlignCenter)
        table.setItem(row, 1, value_item)
        
        print(f"✓ Added row. Total: {len(edit_values)}")
    
    def on_remove_row():
        """Remove the last calibration value row."""
        if len(edit_values) <= 1:
            QMessageBox.warning(dialog, "Warning", "Cannot remove - need at least one calibration value.")
            return
        edit_values.pop()
        table.removeRow(table.rowCount() - 1)
        print(f"✓ Removed row. Total: {len(edit_values)}")
    
    add_button.clicked.connect(on_add_row)
    remove_button.clicked.connect(on_remove_row)
    
    button_layout.addWidget(add_button)
    button_layout.addWidget(remove_button)
    button_layout.addStretch()
    
    main_layout.addLayout(button_layout)
    
    # Bottom button layout
    bottom_button_layout = QHBoxLayout()
    
    # Save buttons
    save_current_button = QPushButton("Apply to This File")
    save_current_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 6px; font-size: 10px;")
    
    save_all_button = QPushButton("Apply to this and all subsequent files")
    save_all_button.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 6px; font-size: 10px;")
    
    cancel_button = QPushButton("Cancel")
    cancel_button.setStyleSheet("background-color: #E57373; color: white; font-weight: bold; padding: 6px; font-size: 10px;")
    
    result = {"saved": False, "apply_to_all": False, "values": None}
    
    def on_save_current():
        """Save and apply to current file only."""
        # Read values from table
        validated = []
        for row in range(table.rowCount()):
            item = table.item(row, 1)
            if item is None:
                QMessageBox.warning(dialog, "Error", f"Row {row + 1} has no value. Please fill all rows.")
                return
            try:
                val = float(item.text())
                validated.append(val)
            except ValueError:
                QMessageBox.warning(dialog, "Error", f"Invalid value in row {row + 1}: '{item.text()}'. Please enter a valid number.")
                return
        
        if len(validated) < 1:
            QMessageBox.warning(dialog, "Error", "Need at least one calibration value.")
            return
        
        result["saved"] = True
        result["apply_to_all"] = False
        result["values"] = validated
        dialog.accept()
    
    def on_save_all():
        """Save and apply to all remaining files."""
        # Read values from table
        validated = []
        for row in range(table.rowCount()):
            item = table.item(row, 1)
            if item is None:
                QMessageBox.warning(dialog, "Error", f"Row {row + 1} has no value. Please fill all rows.")
                return
            try:
                val = float(item.text())
                validated.append(val)
            except ValueError:
                QMessageBox.warning(dialog, "Error", f"Invalid value in row {row + 1}: '{item.text()}'. Please enter a valid number.")
                return
        
        if len(validated) < 1:
            QMessageBox.warning(dialog, "Error", "Need at least one calibration value.")
            return
        
        result["saved"] = True
        result["apply_to_all"] = True
        result["values"] = validated
        dialog.accept()
    
    def on_cancel():
        """Cancel editing."""
        result["saved"] = False
        dialog.reject()
    
    save_current_button.clicked.connect(on_save_current)
    save_all_button.clicked.connect(on_save_all)
    cancel_button.clicked.connect(on_cancel)
    
    bottom_button_layout.addWidget(save_current_button)
    bottom_button_layout.addWidget(save_all_button)
    bottom_button_layout.addWidget(cancel_button)
    
    main_layout.addLayout(bottom_button_layout)
    
    print("\nCalibration values editor opened.")
    print("  • Click on a cell to edit the value (text will be automatically selected)")
    print("  • Use 'Add' to add more calibration points")
    print("  • Use 'Remove' to remove the last calibration point")
    print("  • Choose 'Apply to This File' to apply changes only to current file")
    print("  • Choose 'Apply to this and all subsequent files' to apply changes to all remaining files")
    
    # Show dialog modally
    dialog.exec_()
    
    # Process the result
    if result["saved"] and result["values"] is not None:
        new_values = result["values"]
        apply_to_all = result["apply_to_all"]
        
        # Update num_points if count changed
        old_num = num_points_ref["value"]
        new_num = len(new_values)
        
        if new_num != old_num:
            num_points_ref["value"] = new_num
            print(f"  Number of calibration points changed: {old_num} → {new_num}")
            
            # If we reduced the count and have too many selected points, clear excess
            if len(selected_indices) > new_num:
                excess = len(selected_indices) - new_num
                for _ in range(excess):
                    idx = selected_indices.pop()
                    if selected_markers:
                        marker = selected_markers.pop()
                        try:
                            marker.remove()
                        except Exception:
                            pass
                print(f"  Cleared {excess} excess selected point(s)")
        
        # Update via callback if provided
        if update_calibration_callback:
            update_calibration_callback(new_values, apply_to_all)
        
        print(f"✓ Calibration values updated: {new_values}")
        if apply_to_all:
            print("  (Applied to all remaining files)")
        else:
            print("  (Applied to current file only)")
        
        return new_values, apply_to_all
    
    # Cancelled
    print("Calibration values editing cancelled.")
    return None
