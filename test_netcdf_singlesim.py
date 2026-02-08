#!/usr/bin/env python3
"""
Test single simulation with netcdf output type.
"""
import sys
import tests.fixtures.test_case_catalog as cases

# Create test case with netcdf output
print("Creating test case with TRITON_processed_output_type='nc'...")
test_case = cases.Local_TestCases.retrieve_norfolk_single_sim_test_case(
    start_from_scratch=False,
    additional_analysis_configs={"TRITON_processed_output_type": "nc"}
)

analysis = test_case.analysis
print(f"Analysis ID: {analysis.cfg_analysis.analysis_id}")
print(f"Output type: {analysis.cfg_analysis.TRITON_processed_output_type}")

# Try to process timeseries
print("\nProcessing timeseries...")
try:
    analysis.process_all_sim_timeseries_serially()
    print("✓ Timeseries processing completed")
except Exception as e:
    print(f"✗ Timeseries processing failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check if outputs were created
print("\nChecking outputs...")
import tests.utils_for_testing as tst_ut
try:
    tst_ut.assert_timeseries_processed(analysis)
    print("✓ All timeseries processed successfully")
except AssertionError as e:
    print(f"✗ Assertion failed: {e}")
    sys.exit(1)

print("\n✓ All tests passed!")
