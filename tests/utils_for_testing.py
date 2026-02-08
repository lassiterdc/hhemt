import os
import socket

import pandas as pd
import pytest

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
    """
    Assert that each rule name appears in a Snakefile string.

    For 'run_simulation' and 'process_outputs', accepts either:
    - The exact rule name (old monolithic pattern)
    - Model-specific variants (e.g., run_triton, run_tritonswmm, run_swmm)

    This allows tests to work with both old and new multi-model workflows.
    """
    missing = []
    for rule in rules:
        if rule in ["run_simulation", "process_outputs"]:
            # For these rules, check if ANY model-specific variant exists
            base = "run" if rule == "run_simulation" else "process"
            has_variant = any(
                f"rule {base}_{model}" in content
                for model in ["triton", "tritonswmm", "swmm"]
            )
            has_exact = f"rule {rule}" in content
            if not (has_variant or has_exact):
                missing.append(rule)
        else:
            # For other rules, require exact match
            if f"rule {rule}" not in content:
                missing.append(rule)

    if missing:
        pytest.fail(f"Missing rules in Snakefile: {missing}")


def assert_snakefile_has_flags(content: str, flags: list[str]):
    """Assert that each CLI flag appears in a Snakefile string."""
    missing = [flag for flag in flags if flag not in content]
    if missing:
        pytest.fail(f"Missing flags in Snakefile: {missing}")


def assert_system_setup(analysis: TRITONSWMM_analysis):
    """Assert compilation and system-level inputs were created successfully."""
    cfg_sys = analysis._system.cfg_system

    # Check compilations based on enabled models
    if cfg_sys.toggle_tritonswmm_model:
        assert analysis._system.compilation_successful, "TRITON-SWMM compilation failed"

    if cfg_sys.toggle_triton_model:
        assert (
            analysis._system.compilation_triton_only_successful
        ), "TRITON-only compilation failed"

    if cfg_sys.toggle_swmm_model:
        assert analysis._system.compilation_swmm_successful, "SWMM compilation failed"

    # Check system inputs (DEM, Mannings) - these are always required
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


def assert_analysis_summaries_created(analysis: TRITONSWMM_analysis):
    """
    Assert that analysis-level consolidated summaries were created for all enabled model types.

    Checks actual file existence on disk, not just log flags, for robust validation.
    """
    enabled = get_enabled_model_types(analysis)
    missing = []

    paths = analysis.analysis_paths

    if "tritonswmm" in enabled:
        for desc, path in [
            ("TRITONSWMM TRITON summary", paths.output_tritonswmm_triton_summary),
            ("TRITONSWMM SWMM node summary", paths.output_tritonswmm_node_summary),
            ("TRITONSWMM SWMM link summary", paths.output_tritonswmm_link_summary),
            (
                "TRITONSWMM performance summary",
                paths.output_tritonswmm_performance_summary,
            ),
        ]:
            if path is None or not path.exists():
                missing.append(desc)

    if "triton" in enabled:
        path = paths.output_triton_only_summary
        if path is None or not path.exists():
            missing.append("TRITON-only summary")

        for desc, path in [
            (
                "TRITON-only performance summary",
                paths.output_triton_only_performance_summary,
            ),
        ]:
            if path is None or not path.exists():
                missing.append(desc)

    if "swmm" in enabled:
        for desc, path in [
            ("SWMM-only node summary", paths.output_swmm_only_node_summary),
            ("SWMM-only link summary", paths.output_swmm_only_link_summary),
        ]:
            if path is None or not path.exists():
                missing.append(desc)

    if missing:
        pytest.fail(
            f"Missing analysis-level consolidated summaries:\n"
            + "\n".join(f"  - {m}" for m in missing)
        )


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
    analysis: TRITONSWMM_analysis,
):
    """Assert that an end-to-end workflow completed successfully."""
    assert_system_setup(analysis)
    assert_scenarios_setup(analysis)
    assert_scenarios_run(analysis)
    assert_timeseries_processed(analysis)
    assert_analysis_summaries_created(analysis)
    assert_scenario_status_csv_created(analysis)
    assert_resource_usage_matches_config(analysis)


# ========== Multi-Model Assertion Helpers ==========


def assert_triton_compiled(analysis: TRITONSWMM_analysis):
    """
    Assert that TRITON-only (without SWMM coupling) was compiled successfully.

    This checks for the TRITON-only build which uses -DTRITON_ENABLE_SWMM=OFF.
    """
    system = analysis._system
    if not system.compilation_triton_only_successful:
        pytest.fail("TRITON-only compilation failed")


def assert_swmm_compiled(analysis: TRITONSWMM_analysis):
    """
    Assert that standalone SWMM (EPA SWMM) was compiled successfully.

    This checks for the standalone SWMM executable build.
    """
    system = analysis._system
    if not system.compilation_swmm_successful:
        pytest.fail("SWMM compilation failed")


def assert_tritonswmm_compiled(analysis: TRITONSWMM_analysis):
    """
    Assert that TRITON-SWMM (coupled model) was compiled successfully.

    This is the existing compilation check for the coupled model.
    """
    if not analysis._system.compilation_successful:
        pytest.fail("TRITON-SWMM compilation failed")


def assert_model_simulation_run(
    analysis: TRITONSWMM_analysis,
    model_type: str,
):
    """
    Assert that simulations completed for a specific model type.

    Parameters
    ----------
    analysis : TRITONSWMM_analysis
        The analysis object
    model_type : str
        One of "triton", "tritonswmm", or "swmm"

    Raises
    ------
    pytest.fail
        If any simulation for the model type did not complete
    """
    valid_types = ("triton", "tritonswmm", "swmm")
    if model_type not in valid_types:
        pytest.fail(f"model_type must be one of {valid_types}, got '{model_type}'")

    df_status = analysis.df_status

    # Check if model_type column exists (for multi-model support)
    if "model_type" in df_status.columns:
        model_rows = df_status[df_status["model_type"] == model_type]
        if model_rows.empty:
            pytest.skip(f"No simulations of type '{model_type}' in this analysis")
        failed = model_rows[~model_rows["scen_runs_completed"]]
    else:
        # Legacy mode: check all simulations (assumes single model type)
        failed = df_status[~df_status["scen_runs_completed"]]

    if not failed.empty:
        failed_dirs = failed["scenario_directory"].tolist()
        pytest.fail(
            f"{len(failed)} {model_type} simulation(s) failed to complete:\n"
            + "\n".join(f"  - {d}" for d in failed_dirs[:5])
            + (
                f"\n  ... and {len(failed_dirs) - 5} more"
                if len(failed_dirs) > 5
                else ""
            )
        )


def assert_model_outputs_processed(
    analysis: TRITONSWMM_analysis,
    model_type: str,
):
    """
    Assert that outputs were processed for a specific model type by checking actual output files.

    Parameters
    ----------
    analysis : TRITONSWMM_analysis
        The analysis object
    model_type : str
        One of "triton", "tritonswmm", or "swmm"

    Raises
    ------
    pytest.fail
        If any output processing for the model type did not complete
    """
    valid_types = ("triton", "tritonswmm", "swmm")
    if model_type not in valid_types:
        pytest.fail(f"model_type must be one of {valid_types}, got '{model_type}'")

    # Check for model-specific output files by inspecting actual file paths
    missing_outputs = []

    for event_iloc in analysis.df_sims.index:
        proc = analysis._retrieve_sim_run_processing_object(event_iloc)
        paths = proc.scen_paths

        if model_type == "triton":
            # Check TRITON-only output files
            required_paths = [
                ("TRITON-only timeseries", paths.output_triton_only_timeseries),
                ("TRITON-only summary", paths.output_triton_only_summary),
                (
                    "TRITON-only performance timeseries",
                    paths.output_triton_only_performance_timeseries,
                ),
                (
                    "TRITON-only performance summary",
                    paths.output_triton_only_performance_summary,
                ),
            ]

        elif model_type == "tritonswmm":
            # Check coupled TRITON-SWMM output files
            required_paths = [
                (
                    "TRITONSWMM TRITON timeseries",
                    paths.output_tritonswmm_triton_timeseries,
                ),
                ("TRITONSWMM TRITON summary", paths.output_tritonswmm_triton_summary),
                (
                    "TRITONSWMM SWMM node timeseries",
                    paths.output_tritonswmm_node_time_series,
                ),
                (
                    "TRITONSWMM SWMM link timeseries",
                    paths.output_tritonswmm_link_time_series,
                ),
                ("TRITONSWMM SWMM node summary", paths.output_tritonswmm_node_summary),
                ("TRITONSWMM SWMM link summary", paths.output_tritonswmm_link_summary),
                (
                    "TRITONSWMM performance timeseries",
                    paths.output_tritonswmm_performance_timeseries,
                ),
                (
                    "TRITONSWMM performance summary",
                    paths.output_tritonswmm_performance_summary,
                ),
            ]

        elif model_type == "swmm":
            # Check standalone SWMM output files
            required_paths = [
                ("SWMM-only node timeseries", paths.output_swmm_only_node_time_series),
                ("SWMM-only link timeseries", paths.output_swmm_only_link_time_series),
                ("SWMM-only node summary", paths.output_swmm_only_node_summary),
                ("SWMM-only link summary", paths.output_swmm_only_link_summary),
            ]

        # Check each required path exists
        for desc, path in required_paths:
            if path is None:
                missing_outputs.append(
                    f"{desc} (path not configured) - scenario {event_iloc}"
                )
            elif not path.exists():
                missing_outputs.append(f"{desc} ({path.name}) - scenario {event_iloc}")

    if missing_outputs:
        pytest.fail(
            f"{model_type} output processing incomplete:\n"
            + "\n".join(f"  - {m}" for m in missing_outputs[:10])
            + (
                f"\n  ... and {len(missing_outputs) - 10} more"
                if len(missing_outputs) > 10
                else ""
            )
        )


def assert_enabled_models_match_config(analysis: TRITONSWMM_analysis):
    """
    Assert that the enabled models in df_status match the system config toggles.

    This verifies that:
    - toggle_triton_model -> model_type="triton" rows exist
    - toggle_tritonswmm_model -> model_type="tritonswmm" rows exist
    - toggle_swmm_model -> model_type="swmm" rows exist
    """
    cfg_sys = analysis._system.cfg_system
    df_status = analysis.df_status

    if "model_type" not in df_status.columns:
        pytest.skip("model_type column not yet implemented in df_status")

    model_types_present = set(df_status["model_type"].unique())

    expected_models = set()
    if cfg_sys.toggle_triton_model:
        expected_models.add("triton")
    if cfg_sys.toggle_tritonswmm_model:
        expected_models.add("tritonswmm")
    if cfg_sys.toggle_swmm_model:
        expected_models.add("swmm")

    if model_types_present != expected_models:
        pytest.fail(
            f"Model types in df_status ({model_types_present}) don't match "
            f"enabled toggles ({expected_models})"
        )


def get_enabled_model_types(analysis: TRITONSWMM_analysis) -> list[str]:
    """
    Get list of enabled model types based on system config toggles.

    Returns
    -------
    list[str]
        List of enabled model types: "triton", "tritonswmm", and/or "swmm"
    """
    cfg_sys = analysis._system.cfg_system
    models = []
    if cfg_sys.toggle_triton_model:
        models.append("triton")
    if cfg_sys.toggle_tritonswmm_model:
        models.append("tritonswmm")
    if cfg_sys.toggle_swmm_model:
        models.append("swmm")
    return models


def normalize_swmm_link_vars(link_vars: set[str]) -> set[str]:
    """Normalize SWMM link variable names for cross-parser comparison.

    The .out binary parser (pyswmm) reports 'capacity' while the .rpt text parser
    reports 'capacity_setting'. Both represent the same physical quantity: the
    fraction of conduit filled (0-1 range). This normalizes to 'capacity_setting'.

    Parameters
    ----------
    link_vars : set[str]
        Set of variable names from SWMM output parsing

    Returns
    -------
    set[str]
        Normalized set with 'capacity' replaced by 'capacity_setting' if present
    """
    if "capacity" in link_vars:
        return (link_vars - {"capacity"}) | {"capacity_setting"}
    return link_vars
