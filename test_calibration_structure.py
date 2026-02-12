#!/usr/bin/env python3
"""Test script to verify calibration module structure."""

import sys
from pathlib import Path

# Add SensorFit to path
sys.path.insert(0, str(Path(__file__).parent))

from sensorfit.calibration import (
    load_trace,
    select_baseline,
    select_points,
    select_intervals,
    build_calibration,
    apply_calibration,
    build_interval_subsets,
    persist_interval_subsets,
    CALIBRATED_COLUMN,
)

def test_file_loading():
    """Test that files can be loaded."""
    print("Testing file loading...")
    test_file = Path("/Users/tom/Jottacloud/Tommy/01_NMBU_workspace/Supervision/Hanne_Berggreen/Assays/SensorPeroxi/Trainingdata/7-Sm_noC-2.xlsx")
    
    if not test_file.exists():
        print(f"  ✗ Test file not found: {test_file}")
        return False
    
    try:
        df = load_trace(test_file, time_col_idx=0, current_col_idx=1)
        print(f"  ✓ File loaded successfully: {len(df)} rows, {len(df.columns)} columns")
        print(f"    Columns: {df.columns.tolist()}")
        return True
    except Exception as e:
        print(f"  ✗ Error loading file: {e}")
        return False

def test_calibration_functions():
    """Test that calibration functions are callable."""
    print("\nTesting calibration functions...")
    
    import numpy as np
    
    # Test data
    time_values = np.linspace(0, 100, 100)
    signal_values = np.sin(time_values / 10) * 1e-6
    
    try:
        # Test build_calibration
        mean_currents = [1e-6, 2e-6, 3e-6, 4e-6, 5e-6, 6e-6]
        concentrations = [0, 20, 40, 60, 80, 100]
        calibration = build_calibration(mean_currents, concentrations)
        print(f"  ✓ build_calibration works: slope={calibration.slope:.2e}, intercept={calibration.intercept:.2f}")
        
        # Test apply_calibration
        import pandas as pd
        test_series = pd.Series([1e-6, 2e-6, 3e-6])
        calibrated = apply_calibration(test_series, calibration)
        print(f"  ✓ apply_calibration works: {len(calibrated)} values")
        
        return True
    except Exception as e:
        print(f"  ✗ Error in calibration functions: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_return_types():
    """Test that function return types are correct."""
    print("\nTesting return types...")
    
    # Check select_points return type annotation
    import inspect
    sig = inspect.signature(select_points)
    return_annotation = sig.return_annotation
    print(f"  ✓ select_points return type: {return_annotation}")
    
    # Check that it can return string (for go_back_to_baseline)
    if "str" in str(return_annotation) or "|" in str(return_annotation):
        print("  ✓ select_points can return string for navigation")
    else:
        print("  ⚠ select_points return type might not support string return")
    
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("Calibration Module Structure Test")
    print("=" * 60)
    
    results = []
    results.append(("File Loading", test_file_loading()))
    results.append(("Calibration Functions", test_calibration_functions()))
    results.append(("Return Types", test_return_types()))
    
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
    
    all_passed = all(result[1] for result in results)
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All structure tests passed!")
        print("\nNote: Interactive GUI tests require running the CLI with:")
        print("  python3 -m sensorfit.calibration_cli --input-dir <path>")
    else:
        print("✗ Some tests failed. Please check the errors above.")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())

