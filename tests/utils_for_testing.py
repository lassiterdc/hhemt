import os
import pytest
import socket
import pandas as pd
from pathlib import Path
from TRITON_SWMM_toolkit.analysis import TRITONSWMM_analysis


def uses_slurm() -> bool:
    """Return True when running under SLURM."""
    return "SLURM_JOB_ID" in os.environ


def is_scheduler_context() -> bool:
    """Return True when any known HPC scheduler env var is present."""
    scheduler_vars = (
        "SLURM_JOB_ID",  # SLURM
        "PBS_JOBID",  # PBS
        "LSB_JOBID",  # LSF
        "COBALT_JOBID",  # Cobalt
    )
    return any(v in os.environ for v in scheduler_vars)


def on_frontier() -> bool:
    """Return True when hostname indicates Frontier."""
    return "frontier" in socket.getfqdn()


def on_UVA_HPC() -> bool:
    """Return True when hostname indicates UVA HPC."""
    return "virginia" in socket.getfqdn()


def write_snakefile(analysis: TRITONSWMM_analysis, content: str):
    """Write Snakefile content to the analysis directory and return the path."""
    snakefile_path = analysis.analysis_paths.analysis_dir / "Snakefile"
    snakefile_path.write_text(content)
    return snakefile_path


def assert_snakefile_has_rules(content: str, rules: list[str]):
    """Assert that each rule name appears in a Snakefile string."""
    missing = [rule for rule in rules if f"rule {rule}" not in content]
    if missing:
        pytest.fail(f"Missing rules in Snakefile: {missing}")


def assert_snakefile_has_flags(content: str, flags: list[str]):
    """Assert that each CLI flag appears in a Snakefile string."""
    missing = [flag for flag in flags if flag not in content]
    if missing:
        pytest.fail(f"Missing flags in Snakefile: {missing}")


def assert_system_setup(analysis: TRITONSWMM_analysis):
    """Assert compilation and system-level inputs were created successfully."""
    assert analysis._system.compilation_successful, "TRITON compilation failed"
    dem = analysis._system.processed_dem_rds
    assert dem.shape == (1, 537, 551), "Problems with DEM creation"  # type: ignore
    manning = analysis._system.mannings_rds
    assert manning.shape == (1, 537, 551), "Problems with Mannings creation"  # type: ignore


def assert_scenarios_setup(analysis: TRITONSWMM_analysis):
    """Assert that all scenarios were created."""
    if not analysis.all_scenarios_created:
        print("\n    - ".join(analysis.scenarios_not_created))
        pytest.fail("One or more scenario setups failed")


def assert_scenarios_run(analysis: TRITONSWMM_analysis):
    """Assert that all simulations completed successfully."""
    if not analysis.all_sims_run:
        print("\n    - ".join(analysis.scenarios_not_run))
        pytest.fail("One or more simulations failed to run")


def assert_timeseries_processed(analysis: TRITONSWMM_analysis, which: str = "both"):
    """Assert that requested timeseries outputs were processed."""
    # performance tseries
    if not analysis.all_TRITONSWMM_performance_timeseries_processed:
        print(
            "\n    - ".join(analysis.TRITONSWMM_performance_time_series_not_processed)
        )
        pytest.fail("TRITONSWMM performance timeseries processing failed")
    # TRITON time series
    if which in ("both", "TRITON") and not analysis.all_TRITON_timeseries_processed:
        print("\n    - ".join(analysis.TRITON_time_series_not_processed))
        pytest.fail("TRITON timeseries processing failed")
    # SWMM time series
    if which in ("both", "SWMM") and not analysis.all_SWMM_timeseries_processed:
        print("\n    - ".join(analysis.SWMM_time_series_not_processed))
        pytest.fail("SWMM timeseries processing failed")


def assert_analysis_summaries_created(
    analysis: TRITONSWMM_analysis, which: str = "both"
):
    """Assert that output summaries were created for requested modes."""
    if not analysis.TRITONSWMM_performance_analysis_summary_created:
        pytest.fail("TRITONSWMM analysis performance summary missing")
    if which in ("both", "TRITON") and not analysis.TRITON_analysis_summary_created:
        pytest.fail("TRITON analysis summary missing")
    if which in ("both", "SWMM"):
        if not analysis.SWMM_node_analysis_summary_created:
            pytest.fail("SWMM node analysis summary missing")
        if not analysis.SWMM_link_analysis_summary_created:
            pytest.fail("SWMM link analysis summary missing")


def assert_resource_usage_matches_config(analysis: TRITONSWMM_analysis):
    """
    Assert that actual resource usage matches expected configuration.

    Uses the validate_resource_usage function from consolidate_workflow to check
    that simulations used the expected compute resources (MPI tasks, OMP threads,
    GPUs, backend).

    Parameters
    ----------
    analysis : TRITONSWMM_analysis
        The analysis object containing scenario status

    Raises
    ------
    AssertionError
        If resource usage doesn't match expected configuration
    """
    from TRITON_SWMM_toolkit.consolidate_workflow import validate_resource_usage

    # Use the validation function without a logger (prints to stdout instead)
    validation_passed = validate_resource_usage(analysis, logger=None)

    if not validation_passed:
        pytest.fail(
            "Resource usage validation failed: actual compute resources did not match "
            "expected configuration. See output above for details."
        )


def assert_scenario_status_csv_created(analysis: TRITONSWMM_analysis):
    """
    Assert that scenario_status.csv was created with required resource usage columns.

    Validates:
    - CSV file exists in analysis directory
    - CSV can be read successfully
    - All expected resource usage columns are present
    - Basic data integrity (matching number of scenarios)
    """
    csv_path = analysis.analysis_paths.analysis_dir / "scenario_status.csv"

    # Check file exists
    if not csv_path.exists():
        pytest.fail(f"scenario_status.csv not found at {csv_path}")

    # Read CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        pytest.fail(f"Failed to read scenario_status.csv: {e}")

    # Check for required resource usage columns
    required_columns = [
        "scenarios_setup",
        "scen_runs_completed",
        "scenario_directory",
        "actual_nTasks",
        "actual_omp_threads",
        "actual_gpus",
        "actual_total_gpus",
        "actual_gpu_backend",
        "actual_build_type",
        "actual_wall_time_s",
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        pytest.fail(
            f"scenario_status.csv missing required columns: {missing_columns}\n"
            f"Available columns: {list(df.columns)}"
        )

    # Verify row count matches number of scenarios
    if analysis.cfg_analysis.toggle_sensitivity_analysis:
        # For sensitivity analysis, count scenarios across all sub-analyses
        expected_rows = sum(
            len(sub_analysis.df_sims)
            for sub_analysis in analysis.sensitivity.sub_analyses.values()
        )
    else:
        expected_rows = len(analysis.df_sims)

    if len(df) != expected_rows:
        pytest.fail(
            f"scenario_status.csv has {len(df)} rows, expected {expected_rows} "
            f"(one per scenario)"
        )


def assert_analysis_workflow_completed_successfully(
    analysis: TRITONSWMM_analysis, which: str = "both"
):
    """Assert that an end-to-end workflow completed successfully."""
    assert_system_setup(analysis)
    assert_scenarios_setup(analysis)
    assert_scenarios_run(analysis)
    assert_timeseries_processed(analysis, which=which)
    assert_analysis_summaries_created(analysis, which=which)
    assert_scenario_status_csv_created(analysis)
    assert_resource_usage_matches_config(analysis)
