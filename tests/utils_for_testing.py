import os
import socket
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from TRITON_SWMM_toolkit.analysis import TRITONSWMM_analysis

if TYPE_CHECKING:
    import xarray as xr


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
            has_variant = any(f"rule {base}_{model}" in content for model in ["triton", "tritonswmm", "swmm"])
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


# ========== Validation assertion helpers (thin wrappers) ==========
#
# Each `assert_*` helper below is a thin pytest wrapper around the
# corresponding `check_*` function in
# `src/TRITON_SWMM_toolkit/analysis_validation.py`. The validator module is
# the single source of truth for the analysis-validation logic; the wrappers
# convert structured `CheckResult` records into `pytest.fail(...)` calls,
# preserving the existing public signature (positional + keyword args, the
# `verbose` flag, etc.) so existing test code does not need to change.
#
# To add or modify a check's logic, edit the `check_*` function in
# `analysis_validation.py` — the wrapper does not need editing unless the
# public test signature changes.
#
# The single-model-type helpers further down this file (assert_triton_compiled,
# assert_swmm_compiled, assert_tritonswmm_compiled, assert_model_simulation_run)
# are NOT wrappers: they are more granular per-model checks that have no
# direct counterpart in `analysis_validation.py` and remain as direct attribute
# checks.


def _verbose_detail_block(result, key: str = "scenario_dir") -> str:
    """Format per-scenario failure details for the `verbose=True` path."""
    if not result.details:
        return ""
    lines = []
    for d in result.details:
        if "scenario_dir" in d:
            lines.append(f"    - {d['scenario_dir']}")
        elif "detail" in d:
            lines.append(f"    - {d['detail']}")
    return "\n  Failed scenarios:\n" + "\n".join(lines) if lines else ""


def assert_system_setup(analysis: TRITONSWMM_analysis):
    """Pytest wrapper around analysis_validation.check_system_setup."""
    from TRITON_SWMM_toolkit.analysis_validation import check_system_setup
    result = check_system_setup(analysis)
    if not result.passed:
        msg = result.summary
        if result.details:
            msg += "\n  " + "\n  ".join(f"- {d.get('detail', '')}" for d in result.details)
        pytest.fail(msg)


def assert_scenarios_setup(analysis: TRITONSWMM_analysis, verbose: bool = False):
    """Pytest wrapper around analysis_validation.check_scenarios_setup.

    Parameters
    ----------
    analysis : TRITONSWMM_analysis
    verbose : bool, optional
        If True, append per-scenario fail list to the failure message.
    """
    from TRITON_SWMM_toolkit.analysis_validation import check_scenarios_setup
    result = check_scenarios_setup(analysis)
    if not result.passed:
        msg = result.summary
        if verbose:
            msg += _verbose_detail_block(result)
        else:
            msg += " Run with pytest -v for details."
        pytest.fail(msg)


def assert_scenarios_run(analysis: TRITONSWMM_analysis, verbose: bool = False):
    """Pytest wrapper around analysis_validation.check_scenarios_run."""
    from TRITON_SWMM_toolkit.analysis_validation import check_scenarios_run
    result = check_scenarios_run(analysis)
    if not result.passed:
        msg = result.summary
        if verbose:
            msg += _verbose_detail_block(result)
        else:
            msg += " Run with pytest -v for details."
        pytest.fail(msg)


def assert_timeseries_processed(analysis: TRITONSWMM_analysis, which: str = "both", verbose: bool = False):
    """Pytest wrapper around analysis_validation.check_timeseries_processed.

    Parameters
    ----------
    which : str, optional
        Which timeseries model-types to check: "both" (default), "TRITON", or "SWMM".
        Forwarded to check_timeseries_processed.
    """
    from TRITON_SWMM_toolkit.analysis_validation import check_timeseries_processed
    result = check_timeseries_processed(analysis, which=which)
    if not result.passed:
        msg = result.summary
        if verbose:
            msg += _verbose_detail_block(result)
        else:
            msg += " Run with pytest -v for details."
        pytest.fail(msg)


def assert_analysis_summaries_created(analysis: TRITONSWMM_analysis):
    """Pytest wrapper around analysis_validation.check_analysis_summaries_created."""
    from TRITON_SWMM_toolkit.analysis_validation import check_analysis_summaries_created
    result = check_analysis_summaries_created(analysis)
    if not result.passed:
        msg = result.summary
        if result.details:
            msg += "\n" + "\n".join(f"  - {d.get('detail', '')}" for d in result.details)
        pytest.fail(msg)


def assert_resource_usage_matches_config(analysis: TRITONSWMM_analysis):
    """Pytest wrapper around analysis_validation.check_resource_usage."""
    from TRITON_SWMM_toolkit.analysis_validation import check_resource_usage
    result = check_resource_usage(analysis)
    if not result.passed:
        pytest.fail(
            f"{result.summary}: actual compute resources did not match expected configuration. "
            "See output above for details."
        )


def assert_scenario_status_csv_created(analysis: TRITONSWMM_analysis):
    """Pytest wrapper around analysis_validation.check_scenario_status_csv."""
    from TRITON_SWMM_toolkit.analysis_validation import check_scenario_status_csv
    result = check_scenario_status_csv(analysis)
    if not result.passed:
        msg = result.summary
        if result.details:
            msg += "\n" + "\n".join(f"  - {d.get('detail', '')}" for d in result.details)
        pytest.fail(msg)


def assert_analysis_workflow_completed_successfully(
    analysis: TRITONSWMM_analysis,
):
    """Pytest wrapper around analysis_validation.validate_analysis aggregator.

    Runs all 7 checks; if any failed, fails with a multi-line summary listing
    each failed check's headline message. Per-scenario detail rows are
    available via the structured CheckResult records but are NOT printed here
    (use individual assert_* helpers with verbose=True for that).
    """
    from TRITON_SWMM_toolkit.analysis_validation import validate_analysis
    report = validate_analysis(analysis)
    if not report.overall_passed:
        failed = [c for c in report.checks if not c.passed]
        lines = [f"  ✗ {c.name}: {c.summary}" for c in failed]
        msg = (
            f"Analysis workflow validation failed ({len(failed)} of {len(report.checks)} checks failed):\n"
            + "\n".join(lines)
        )
        pytest.fail(msg)


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
        failed = model_rows[~model_rows["run_completed"]]
    else:
        # Legacy mode: check all simulations (assumes single model type)
        failed = df_status[~df_status["run_completed"]]

    if not failed.empty:
        failed_dirs = failed["scenario_directory"].tolist()
        pytest.fail(
            f"{len(failed)} {model_type} simulation(s) failed to complete:\n"
            + "\n".join(f"  - {d}" for d in failed_dirs[:5])
            + (f"\n  ... and {len(failed_dirs) - 5} more" if len(failed_dirs) > 5 else "")
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
                missing_outputs.append(f"{desc} (path not configured) - scenario {event_iloc}")
            elif not path.exists():
                missing_outputs.append(f"{desc} ({path.name}) - scenario {event_iloc}")

    if missing_outputs:
        pytest.fail(
            f"{model_type} output processing incomplete:\n"
            + "\n".join(f"  - {m}" for m in missing_outputs[:10])
            + (f"\n  ... and {len(missing_outputs) - 10} more" if len(missing_outputs) > 10 else "")
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
        pytest.fail(f"Model types in df_status ({model_types_present}) don't match enabled toggles ({expected_models})")


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
            pytest.fail(f"Invalid phase name: '{phase_name}'. Valid phases: {list(phase_status_map.keys())}")

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
        raise ValueError(f"Invalid model_type: '{model_type}'. Valid types: {valid_types}")

    enabled_models = get_enabled_model_types(analysis)
    if model_type not in enabled_models:
        raise ValueError(f"Model type '{model_type}' not enabled. Enabled models: {enabled_models}")

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


def assert_datasets_equal(
    ds_reference: "xr.Dataset",
    ds_actual: "xr.Dataset",
    dataset_name: str,
    skip_vars: list[str] | None = None,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> None:
    """
    Assert two datasets have identical data values (order-agnostic).

    Parameters
    ----------
    ds_reference : xr.Dataset
        Reference dataset
    ds_actual : xr.Dataset
        Actual (freshly generated) dataset
    dataset_name : str
        Name for error messages (e.g., "TRITONSWMM_TRITON.zarr")
    skip_vars : list[str], optional
        Variable names to skip (e.g., performance timers)
    rtol : float
        Relative tolerance for np.allclose
    atol : float
        Absolute tolerance for np.allclose

    Raises
    ------
    AssertionError
        If datasets differ in dimensions, variables, or values
    """
    import numpy as np

    skip_vars = skip_vars or []

    # Check dimensions match (order-agnostic)
    ref_dims = set(ds_reference.dims)
    actual_dims = set(ds_actual.dims)
    if ref_dims != actual_dims:
        raise AssertionError(
            f"{dataset_name}: Dimension names differ.\n  Reference: {sorted(ref_dims)}\n  Actual: {sorted(actual_dims)}"
        )

    # Check data variables match (order-agnostic, excluding skipped)
    ref_vars = set(ds_reference.data_vars) - set(skip_vars)
    actual_vars = set(ds_actual.data_vars) - set(skip_vars)
    if ref_vars != actual_vars:
        missing = ref_vars - actual_vars
        extra = actual_vars - ref_vars
        raise AssertionError(
            f"{dataset_name}: Data variables differ.\n"
            f"  Missing in actual: {sorted(missing) if missing else 'none'}\n"
            f"  Extra in actual: {sorted(extra) if extra else 'none'}"
        )

    # Compare each data variable
    mismatched_vars = []
    for var in ref_vars:
        ref_da = ds_reference[var]
        actual_da = ds_actual[var]

        # Get dimension names (order-agnostic comparison)
        ref_dims_list = list(ref_da.dims)
        actual_dims_list = list(actual_da.dims)

        # Check dimension names match (order doesn't matter)
        if set(ref_dims_list) != set(actual_dims_list):
            mismatched_vars.append(f"{var}: dimension names differ (ref={ref_dims_list}, actual={actual_dims_list})")
            continue

        # Transpose actual to match reference dimension order if needed
        if ref_dims_list != actual_dims_list:
            actual_da = actual_da.transpose(*ref_dims_list)

        # Use coordinate-based alignment to handle coordinate reversal (e.g., y-coords)
        # This ensures we're comparing the same geographic locations even if coordinates
        # are in different order (e.g., ascending vs descending)
        try:
            actual_da_aligned = actual_da.reindex_like(ref_da, method=None)
        except Exception as e:
            mismatched_vars.append(f"{var}: coordinate alignment failed ({str(e)})")
            continue

        ref_values = ref_da.values
        actual_values = actual_da_aligned.values

        # Now shapes should match after transpose and reindex
        if ref_values.shape != actual_values.shape:
            mismatched_vars.append(
                f"{var}: shape mismatch even after transpose and reindex (ref={ref_values.shape}, actual={actual_values.shape})"
            )
            continue

        # Compare values (handle string dtypes specially)
        if np.issubdtype(ref_values.dtype, np.str_) or np.issubdtype(ref_values.dtype, np.object_):
            # String comparison
            if not np.array_equal(ref_values, actual_values):
                mismatched_vars.append(f"{var}: string values differ")
        else:
            # Numeric comparison with tolerance
            if not np.allclose(ref_values, actual_values, rtol=rtol, atol=atol, equal_nan=True):
                max_diff = np.nanmax(np.abs(ref_values - actual_values))
                mismatched_vars.append(f"{var}: numeric values differ (max_abs_diff={max_diff:.2e})")

    if mismatched_vars:
        raise AssertionError(
            f"{dataset_name}: {len(mismatched_vars)} variable(s) have mismatched values:\n  "
            + "\n  ".join(mismatched_vars)
        )


def _time_dim(da) -> str | None:
    """Return the first time-like dimension of a DataArray, or None if absent.

    Zarrs emitted by different toolkit code paths use different time-dim
    names: SWMM per-node/link timeseries use ``date_time``; TRITON 2D
    timeseries use ``timestep_min``; some legacy code paths use ``time``.
    Per-link or per-node summary variables with no time dim at all return
    None — callers should skip them rather than reduce.
    """
    for d in da.dims:
        if d in ("date_time", "timestep_min", "time"):
            return d
    return None


def assert_hydraulic_components_exercised(analysis) -> None:
    """Verify every enabled hydraulic component produced non-zero time-variant output.

    Raises AssertionError enumerating any component that failed.
    """
    import xarray as xr

    failures: list[str] = []

    enabled = [
        mt
        for mt in ("triton", "tritonswmm", "swmm")
        if getattr(analysis._system.cfg_system, f"toggle_{mt}_model")
    ]

    for event_iloc in analysis.df_sims.index:
        run = analysis._retrieve_sim_runs(event_iloc)
        scen = run._scenario
        sp = scen.scen_paths

        for model_type in enabled:
            _ = scen.get_log(model_type)

            if model_type == "tritonswmm":
                node_sum = sp.output_tritonswmm_node_summary
                if node_sum is None or not node_sum.exists():
                    failures.append(
                        f"event {event_iloc} tritonswmm: node summary missing at {node_sum}"
                    )
                else:
                    ds = xr.open_zarr(node_sum)
                    if not any(
                        float(ds[v].max()) > 0
                        for v in ds.data_vars
                        if ds[v].dtype.kind in "fi"
                    ):
                        failures.append(
                            f"event {event_iloc} tritonswmm: node summary has no positive totals"
                        )
                link_ts = sp.output_tritonswmm_link_time_series
                if link_ts is None or not link_ts.exists():
                    failures.append(
                        f"event {event_iloc} tritonswmm: link timeseries missing at {link_ts}"
                    )
                else:
                    ds = xr.open_zarr(link_ts)
                    flow_var_max = max(
                        (
                            float(ds[v].std(_time_dim(ds[v])).max())
                            for v in ds.data_vars
                            if "flow" in v.lower() and ds[v].dtype.kind in "fi" and _time_dim(ds[v])
                        ),
                        default=0.0,
                    )
                    if flow_var_max <= 0:
                        failures.append(
                            f"event {event_iloc} tritonswmm: link flow variance is zero"
                        )
                triton_ts = sp.output_tritonswmm_triton_timeseries
                _assert_triton_depth(triton_ts, event_iloc, "tritonswmm", failures)

            elif model_type == "triton":
                triton_ts = sp.output_triton_only_timeseries
                _assert_triton_depth(triton_ts, event_iloc, "triton", failures)

            elif model_type == "swmm":
                node_ts = sp.output_swmm_only_node_time_series
                if node_ts is None or not node_ts.exists():
                    failures.append(
                        f"event {event_iloc} swmm: node timeseries missing at {node_ts}"
                    )
                else:
                    ds = xr.open_zarr(node_ts)
                    inflow_var_max = max(
                        (
                            float(ds[v].std(_time_dim(ds[v])).max())
                            for v in ds.data_vars
                            if ("inflow" in v.lower() or "runoff" in v.lower())
                            and ds[v].dtype.kind in "fi"
                            and _time_dim(ds[v])
                        ),
                        default=0.0,
                    )
                    if inflow_var_max <= 0:
                        failures.append(
                            f"event {event_iloc} swmm: node inflow/runoff variance is zero"
                        )
                link_ts = sp.output_swmm_only_link_time_series
                if link_ts is None or not link_ts.exists():
                    failures.append(
                        f"event {event_iloc} swmm: link timeseries missing at {link_ts}"
                    )
                else:
                    ds = xr.open_zarr(link_ts)
                    flow_var_max = max(
                        (
                            float(ds[v].std(_time_dim(ds[v])).max())
                            for v in ds.data_vars
                            if "flow" in v.lower() and ds[v].dtype.kind in "fi" and _time_dim(ds[v])
                        ),
                        default=0.0,
                    )
                    if flow_var_max <= 0:
                        failures.append(
                            f"event {event_iloc} swmm: link flow variance is zero"
                        )

    weather_ds = xr.open_dataset(analysis.cfg_analysis.weather_timeseries)
    bc_var = analysis.cfg_analysis.weather_time_series_storm_tide_datavar
    if bc_var in weather_ds and float(weather_ds[bc_var].std()) <= 0:
        failures.append("storm-tide boundary variance is zero")

    if failures:
        raise AssertionError(
            "assert_hydraulic_components_exercised failed:\n  - "
            + "\n  - ".join(failures)
        )


def _assert_triton_depth(triton_ts, event_iloc, model_type: str, failures: list[str]) -> None:
    """Verify that TRITON emitted a time-varying 2D water field.

    The toolkit's TRITON-output zarr uses ``wlevel_m`` (water level at cell
    center) with dims ``(timestep_min, y, x)``; ``depth`` is an older naming
    that no longer appears. Accept either as the depth proxy.
    """
    import xarray as xr

    if triton_ts is None or not triton_ts.exists():
        failures.append(
            f"event {event_iloc} {model_type}: triton timeseries missing at {triton_ts}"
        )
        return
    ds = xr.open_zarr(triton_ts)
    depth_var = next(
        (v for v in ("wlevel_m", "depth") if v in ds.data_vars),
        None,
    )
    if depth_var is None:
        failures.append(
            f"event {event_iloc} {model_type}: no water-level var "
            f"(wlevel_m / depth) in {triton_ts}"
        )
        return
    # Reduce over every dim except the time dim so we get one value per
    # timestep, then count timesteps where the field is positive somewhere.
    t_dim = _time_dim(ds[depth_var])
    if t_dim is None:
        failures.append(
            f"event {event_iloc} {model_type}: {depth_var} has no time dim in {triton_ts}"
        )
        return
    non_time_dims = tuple(d for d in ds[depth_var].dims if d != t_dim)
    nonzero_ts = int((ds[depth_var] > 0).any(dim=non_time_dims).sum().values)
    if nonzero_ts < 2:
        failures.append(
            f"event {event_iloc} {model_type}: TRITON {depth_var} non-zero "
            f"at only {nonzero_ts} timesteps"
        )
