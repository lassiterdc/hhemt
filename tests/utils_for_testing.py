import os
import socket
from pathlib import Path

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


def assert_scenarios_setup(analysis: TRITONSWMM_analysis, verbose: bool = False):
    """Assert that all scenarios were created.

    Parameters
    ----------
    analysis : TRITONSWMM_analysis
        The analysis to check
    verbose : bool, optional
        If True, print list of failed scenarios (default: False, use pytest -v to enable)
    """
    if not analysis.all_scenarios_created:
        if verbose:
            print("\n  Failed scenarios:\n    - " + "\n    - ".join(analysis.scenarios_not_created))
        pytest.fail(
            f"Scenario setup failed for {len(analysis.scenarios_not_created)} "
            f"of {len(analysis.df_sims)} scenarios. Run with pytest -v for details."
        )


def assert_scenarios_run(analysis: TRITONSWMM_analysis, verbose: bool = False):
    """Assert that all simulations completed successfully.

    Parameters
    ----------
    analysis : TRITONSWMM_analysis
        The analysis to check
    verbose : bool, optional
        If True, print list of failed simulations (default: False, use pytest -v to enable)
    """
    if not analysis.all_sims_run:
        if verbose:
            print("\n  Failed simulations:\n    - " + "\n    - ".join(analysis.scenarios_not_run))
        pytest.fail(
            f"Simulation failed for {len(analysis.scenarios_not_run)} "
            f"of {len(analysis.df_sims)} scenarios. Run with pytest -v for details."
        )


def assert_timeseries_processed(
    analysis: TRITONSWMM_analysis, which: str = "both", verbose: bool = False
):
    """Assert that requested timeseries outputs were processed.

    Parameters
    ----------
    analysis : TRITONSWMM_analysis
        The analysis to check
    which : str, optional
        Which timeseries to check: "both", "TRITON", or "SWMM" (default: "both")
    verbose : bool, optional
        If True, print list of failed processing (default: False, use pytest -v to enable)
    """
    # performance tseries
    if not analysis.all_TRITONSWMM_performance_timeseries_processed:
        if verbose:
            print(
                "\n  Failed TRITONSWMM performance:\n    - "
                + "\n    - ".join(analysis.TRITONSWMM_performance_time_series_not_processed)
            )
        pytest.fail(
            f"TRITONSWMM performance timeseries processing failed for "
            f"{len(analysis.TRITONSWMM_performance_time_series_not_processed)} scenarios. "
            "Run with pytest -v for details."
        )
    # TRITON time series
    if which in ("both", "TRITON") and not analysis.all_TRITON_timeseries_processed:
        if verbose:
            print(
                "\n  Failed TRITON timeseries:\n    - "
                + "\n    - ".join(analysis.TRITON_time_series_not_processed)
            )
        pytest.fail(
            f"TRITON timeseries processing failed for "
            f"{len(analysis.TRITON_time_series_not_processed)} scenarios. "
            "Run with pytest -v for details."
        )
    # SWMM time series
    if which in ("both", "SWMM") and not analysis.all_SWMM_timeseries_processed:
        if verbose:
            print(
                "\n  Failed SWMM timeseries:\n    - "
                + "\n    - ".join(analysis.SWMM_time_series_not_processed)
            )
        pytest.fail(
            f"SWMM timeseries processing failed for "
            f"{len(analysis.SWMM_time_series_not_processed)} scenarios. "
            "Run with pytest -v for details."
        )


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

# ========== Phase 6d.2: New Assertion Helpers (2026-02-09) ==========


def assert_model_outputs_exist(
    analysis: TRITONSWMM_analysis,
    model_types: list[str] | None = None,
    check_timeseries: bool = True,
    check_summaries: bool = True,
    verbose: bool = False,
) -> None:
    """Assert expected outputs exist for all enabled model types.

    This helper consolidates the common pattern of checking outputs across
    multiple model types (TRITON, TRITON-SWMM coupled, SWMM standalone).

    Parameters
    ----------
    analysis : TRITONSWMM_analysis
        The analysis to check
    model_types : list[str], optional
        Model types to check. If None, auto-detects from enabled models.
        Valid values: ["triton", "tritonswmm", "swmm"]
    check_timeseries : bool, default True
        Check if timeseries outputs exist
    check_summaries : bool, default True
        Check if analysis-level summary files exist
    verbose : bool, default False
        Print detailed list of missing outputs (use pytest -v to enable)

    Raises
    ------
    AssertionError
        If any expected outputs are missing

    Examples
    --------
    Check all enabled models have outputs:
        >>> assert_model_outputs_exist(analysis)

    Check only timeseries, skip summaries:
        >>> assert_model_outputs_exist(analysis, check_summaries=False)

    Check specific model types:
        >>> assert_model_outputs_exist(analysis, model_types=["tritonswmm"])

    Notes
    -----
    Uses existing helper functions (assert_timeseries_processed,
    assert_analysis_summaries_created) internally for consistent behavior.
    """
    if model_types is None:
        model_types = get_enabled_model_types(analysis)

    missing_outputs = []

    # Check timeseries if requested
    if check_timeseries:
        try:
            assert_timeseries_processed(analysis, which="both", verbose=verbose)
        except AssertionError as e:
            missing_outputs.append(f"Timeseries: {str(e)}")

    # Check summaries if requested
    if check_summaries:
        try:
            assert_analysis_summaries_created(analysis)
        except AssertionError as e:
            missing_outputs.append(f"Summaries: {str(e)}")

    if missing_outputs:
        error_msg = f"Model output validation failed for {len(missing_outputs)} check(s):\n"
        error_msg += "\n".join(f"  - {msg}" for msg in missing_outputs)
        if not verbose:
            error_msg += "\nRun with pytest -v for detailed output lists."
        pytest.fail(error_msg)


def assert_file_exists(path: Path, description: str | None = None) -> None:
    """Assert file exists with clear error message.

    Standardized file existence check that always includes the path
    in the error message for easy debugging.

    Parameters
    ----------
    path : Path
        Path to check
    description : str, optional
        Description of what this file is (e.g., "Snakefile", "SWMM input")
        If provided, included in error message for context

    Raises
    ------
    AssertionError
        If file does not exist, with message showing path and description

    Examples
    --------
    Basic usage:
        >>> assert_file_exists(snakefile_path)

    With description for context:
        >>> assert_file_exists(snakefile_path, "Snakefile")
        >>> assert_file_exists(swmm_inp, "SWMM input file")

    Notes
    -----
    Replaces inconsistent patterns:
        assert path.exists()  # Bad: no context in failure
        assert path.exists(), f"Missing {path}"  # Better but inconsistent

    Standard pattern provides:
        - Always includes path in message
        - Optional description for context
        - Consistent error format across test suite
    """
    if not path.exists():
        desc = f" ({description})" if description else ""
        pytest.fail(f"Expected file{desc} not found: {path}")


def assert_phases_complete(
    analysis: TRITONSWMM_analysis,
    phases: list[str] | None = None,
    verbose: bool = False,
) -> None:
    """Assert specified workflow phases completed for all scenarios.

    Uses WorkflowStatus to check completion state of workflow phases.
    Useful for verifying prerequisites before testing later phases.

    Parameters
    ----------
    analysis : TRITONSWMM_analysis
        The analysis to check
    phases : list[str], optional
        Phases to check. Valid values: ["setup", "preparation", "simulation",
        "processing", "consolidation"]. If None, checks all phases.
    verbose : bool, default False
        Print detailed status for each phase (use pytest -v to enable)

    Raises
    ------
    AssertionError
        If any specified phase is incomplete, with details of which phases failed

    Examples
    --------
    Check setup and preparation before testing execution:
        >>> assert_phases_complete(analysis, phases=["setup", "preparation"])

    Check all phases:
        >>> assert_phases_complete(analysis)

    With verbose output:
        >>> assert_phases_complete(analysis, phases=["simulation"], verbose=True)

    Notes
    -----
    Leverages the WorkflowStatus infrastructure added in Tier 3 Phase 2.
    Phase names match those returned by analysis.get_workflow_status().
    """
    if phases is None:
        phases = ["setup", "preparation", "simulation", "processing", "consolidation"]

    status = analysis.get_workflow_status()

    incomplete = []
    phase_status_map = {
        "setup": status.setup,
        "preparation": status.preparation,
        "simulation": status.simulation,
        "processing": status.processing,
        "consolidation": status.consolidation,
    }

    for phase_name in phases:
        if phase_name not in phase_status_map:
            pytest.fail(
                f"Invalid phase name: '{phase_name}'. "
                f"Valid phases: {list(phase_status_map.keys())}"
            )

        phase_obj = phase_status_map[phase_name]
        if not phase_obj.complete:
            if verbose:
                print(f"\n  {phase_name.capitalize()}: {phase_obj.progress:.0%} complete")
                if phase_obj.failed_items:
                    print(f"    Failed items: {phase_obj.failed_items}")
            incomplete.append(f"{phase_name} ({phase_obj.progress:.0%})")

    if incomplete:
        error_msg = f"Phase completion check failed for {len(incomplete)} phase(s):\n"
        error_msg += "\n".join(f"  - {phase}" for phase in incomplete)
        error_msg += f"\n\nTotal progress: {status.simulations_completed}/{status.total_simulations} simulations"
        if not verbose:
            error_msg += "\nRun with pytest -v for phase details."
        pytest.fail(error_msg)


def assert_model_simulations_complete(
    analysis: TRITONSWMM_analysis,
    model_type: str,
    verbose: bool = False,
) -> None:
    """Assert all simulations completed for specific model type.

    Checks model-specific completion status. Useful when testing individual
    model outputs in multi-model analyses.

    Parameters
    ----------
    analysis : TRITONSWMM_analysis
        The analysis to check
    model_type : {"triton", "tritonswmm", "swmm"}
        Model type to check
    verbose : bool, default False
        Print list of incomplete simulations (use pytest -v to enable)

    Raises
    ------
    AssertionError
        If any simulations incomplete for this model type
    ValueError
        If model_type is invalid or model not enabled

    Examples
    --------
    Check TRITON-SWMM coupled simulations:
        >>> assert_model_simulations_complete(analysis, "tritonswmm")

    Check TRITON-only simulations:
        >>> assert_model_simulations_complete(analysis, "triton")

    With verbose output showing which scenarios failed:
        >>> assert_model_simulations_complete(analysis, "swmm", verbose=True)

    Notes
    -----
    Model type must be enabled in system config or this will raise ValueError.
    Uses scenario-level model_run_completed() checks for each event.
    """
    valid_types = ["triton", "tritonswmm", "swmm"]
    if model_type not in valid_types:
        raise ValueError(
            f"Invalid model_type: '{model_type}'. Valid types: {valid_types}"
        )

    enabled_models = get_enabled_model_types(analysis)
    if model_type not in enabled_models:
        raise ValueError(
            f"Model type '{model_type}' not enabled. "
            f"Enabled models: {enabled_models}"
        )

    incomplete_scenarios = []

    for event_iloc in analysis.df_sims.index:
        run = analysis._retrieve_sim_runs(event_iloc)
        scenario = run._scenario

        if not scenario.model_run_completed(model_type):
            incomplete_scenarios.append(event_iloc)

    if incomplete_scenarios:
        if verbose:
            print(f"\n  Incomplete {model_type} simulations (event_iloc):")
            print("    - " + "\n    - ".join(map(str, incomplete_scenarios)))

        pytest.fail(
            f"{model_type.upper()} simulations incomplete for "
            f"{len(incomplete_scenarios)} of {len(analysis.df_sims)} scenarios. "
            "Run with pytest -v for scenario list."
        )
