import os
import pytest
import socket
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


def assert_analysis_workflow_completed_successfully(
    analysis: TRITONSWMM_analysis, which: str = "both"
):
    """Assert that an end-to-end workflow completed successfully."""
    assert_system_setup(analysis)
    assert_scenarios_setup(analysis)
    assert_scenarios_run(analysis)
    assert_timeseries_processed(analysis, which=which)
    assert_analysis_summaries_created(analysis, which=which)
