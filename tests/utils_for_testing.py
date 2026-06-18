import os
import socket
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from hhemt.analysis import TRITONSWMM_analysis

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
# `src/hhemt/analysis_validation.py`. The validator module is
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
    from hhemt.analysis_validation import check_system_setup
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
    from hhemt.analysis_validation import check_scenarios_setup
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
    from hhemt.analysis_validation import check_scenarios_run
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
    from hhemt.analysis_validation import check_timeseries_processed
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
    from hhemt.analysis_validation import check_analysis_summaries_created
    result = check_analysis_summaries_created(analysis)
    if not result.passed:
        msg = result.summary
        if result.details:
            msg += "\n" + "\n".join(f"  - {d.get('detail', '')}" for d in result.details)
        pytest.fail(msg)


def assert_resource_usage_matches_config(analysis: TRITONSWMM_analysis):
    """Pytest wrapper around analysis_validation.check_resource_usage."""
    from hhemt.analysis_validation import check_resource_usage
    result = check_resource_usage(analysis)
    if not result.passed:
        pytest.fail(
            f"{result.summary}: actual compute resources did not match expected configuration. "
            "See output above for details."
        )


def assert_scenario_status_csv_created(analysis: TRITONSWMM_analysis):
    """Pytest wrapper around analysis_validation.check_scenario_status_csv."""
    from hhemt.analysis_validation import check_scenario_status_csv
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
    from hhemt.analysis_validation import validate_analysis
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
                f"{var}: shape mismatch even after transpose and reindex "
                f"(ref={ref_values.shape}, actual={actual_values.shape})"
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


# ---------------------------------------------------------------------------
# Rerun-trigger empirical test helpers
# ---------------------------------------------------------------------------


def snapshot_scenario_output_mtimes(
    analysis, *, kind: str
) -> dict[tuple[str | None, str], dict[Path, float]]:
    """Return {(sa_id, event_id): {output_file: mtime_float}} for every per-scenario output.

    Scenario id is the stable event-slug computed by scenario.compute_event_id_slug
    (e.g., "year.9_event_type.compound_event_id.1") — NOT the row label of df_sims.
    The slug is exposed as TRITONSWMM_scenario.event_id (scenario.py:60: event_id = sim_id_str).

    Computes the slug + sim_folder directly via compute_event_id_slug — does NOT
    instantiate TRITONSWMM_scenario. Instantiation triggers a side-effect chain
    (TRITONSWMM_scenario → TRITONSWMM_run → TRITONSWMM_sim_post_processing →
    _log_write_status → LogField.set → log.write) that rewrites log_tritonswmm.json
    and would corrupt this snapshot.

    Excludes per-model JSON logs (log_triton.json, log_tritonswmm.json, log_swmm.json)
    from the walked file set: these files are touched on every scenario instantiation
    via the side-effect chain above. The same chain runs during the post-workflow
    `_update_master_analysis_log` step that fires after every submit_workflow call
    (including for untouched scenarios). That bookkeeping write makes the per-model
    JSON log mtime drift between baseline and rerun even when Snakemake correctly
    skips the simulation rule — which would be a false positive for the assertion.
    The simulation/processing output zarrs and raw outputs remain in scope; those
    only change when Snakemake's simulation/process/consolidate rules actually run.

    Keys are typed tuples to avoid substring-collision pitfalls with string-concat
    namespacing (e.g., sa_id="sa_1" vs sa_id="extra_sa_1" both endswith "::sa_1"):

      - kind='multi_sim': key is (None, event_id).
      - kind='sensitivity': key is (str(sa_id), event_id), disambiguating identical
        event_ids across sub-analyses.
    """
    from hhemt.scenario import compute_event_id_slug

    snapshot: dict[tuple[str | None, str], dict[Path, float]] = {}
    excluded_log_files = {"log_triton.json", "log_tritonswmm.json", "log_swmm.json"}

    def _walk(sim_dir: Path) -> dict[Path, float]:
        return {
            p: p.stat().st_mtime
            for p in sim_dir.rglob("*")
            if p.is_file() and p.name not in excluded_log_files
        }

    def _slug_and_sim_folder(ana, iloc):
        indexers = ana._retrieve_weather_indexer_using_integer_index(iloc)
        slug = compute_event_id_slug(indexers)
        sim_folder = ana.analysis_paths.simulation_directory / slug
        return slug, sim_folder

    if kind == "multi_sim":
        for iloc in analysis.df_sims.index:
            slug, sim_folder = _slug_and_sim_folder(analysis, iloc)
            snapshot[(None, slug)] = _walk(sim_folder)
    elif kind == "sensitivity":
        for sa_id, sub in analysis.sensitivity.sub_analyses.items():
            sa_id_str = str(sa_id)
            for iloc in sub.df_sims.index:
                slug, sim_folder = _slug_and_sim_folder(sub, iloc)
                snapshot[(sa_id_str, slug)] = _walk(sim_folder)
    else:
        raise ValueError(f"unknown kind: {kind!r}")

    return snapshot


def mutate_scenario_csv(
    analysis,
    *,
    kind: str,
    donor_key: tuple[str | None, str],
    remove_key: tuple[str | None, str],
) -> tuple[Path, tuple[str | None, str]]:
    """Copy the source scenario CSV to a tmp path, drop the row whose scenario
    identifier == remove_key, append a fresh row cloned from donor_key with a
    synthetic identifier guaranteed not to collide. Return (tmp_csv_path, new_key).
    Keys are typed (sa_id, event_id) tuples matching snapshot_scenario_output_mtimes;
    sa_id=None for multi_sim, sa_id=str for sensitivity.

    multi_sim:
      - Source CSV: analysis.cfg_analysis.weather_events_to_simulate (analysis.py:164-166).
      - Indexer columns are listed in cfg_analysis.weather_event_indices.
      - Scenario id is the stable slug built by scenario.compute_event_id_slug
        from a row's indexer-column values (scenario.py:30-38, 60).
      - There is NO `event_id` column in the source CSV; its presence in the test
        fixture (`test_data/.../weather_indices.csv` with values 0,1,...) is
        incidental and not used by the toolkit.
      - Mutation: drop rows whose slug == remove_id; clone donor row, bump the
        last indexer column to a synthetic integer that produces a fresh slug.

    sensitivity:
      - Source CSV (or XLSX): analysis.cfg_analysis.sensitivity_analysis.
      - The `sa_id` column is required by stipulation
        `library/docs/stipulations/hhemt/sensitivity csvs require sa_id column.md`.
      - Mutation: drop the row whose sa_id == remove_id; clone donor row with a
        new synthetic sa_id matching `^[A-Za-z0-9_.]+$`.
    """
    if kind == "multi_sim":
        from hhemt.scenario import compute_event_id_slug

        donor_event_id = donor_key[1]  # sa_id is None for multi_sim
        remove_event_id = remove_key[1]
        src = Path(analysis.cfg_analysis.weather_events_to_simulate)
        df = pd.read_csv(src)
        indexer_cols = list(analysis.cfg_analysis.weather_event_indices)

        def _row_slug(row) -> str:
            return compute_event_id_slug({c: row[c] for c in indexer_cols})

        slugs = df.apply(_row_slug, axis=1)
        donor_row = df[slugs == donor_event_id].iloc[0].copy()
        # Capture victim's iloc positions BEFORE removal so we can delete stale model logs.
        # The toolkit's completion check uses iloc-indexed log names (model_triton_evt{N}.log),
        # NOT slug-based names. After removing the victim and appending a new scenario, the new
        # scenario inherits the victim's iloc, causing the completion check to read the victim's
        # log and declare the new scenario "already done" before running.
        victim_ilocs = df.index[slugs == remove_event_id].tolist()
        df = df[slugs != remove_event_id].copy()
        bumped_col = indexer_cols[-1]
        donor_orig_bumped_val = donor_row[bumped_col]  # save before overwrite in loop below
        existing_slugs = set(df.apply(_row_slug, axis=1).tolist())
        n = 9000
        while True:
            donor_row[bumped_col] = n
            candidate_slug = compute_event_id_slug({c: donor_row[c] for c in indexer_cols})
            if candidate_slug not in existing_slugs:
                break
            n += 1
        new_key: tuple[str | None, str] = (None, candidate_slug)
        df = pd.concat([df, donor_row.to_frame().T], ignore_index=True)

        # Expand the weather NetCDF so that ds.sel({bumped_col: n}) succeeds for the new
        # synthetic scenario. The bumped indexer column is the weather-dataset dimension, so
        # bumping to a value not in the NetCDF would fail scenario preparation. We clone the
        # donor's slice with the new coordinate value.
        import xarray as xr
        weather_nc_src = Path(analysis.cfg_analysis.weather_timeseries)
        with xr.open_dataset(weather_nc_src) as ds_weather:
            ds_weather = ds_weather.load()  # into memory so we can close file and write new one
            donor_slice = ds_weather.sel({bumped_col: [donor_orig_bumped_val]})
            new_slice = donor_slice.assign_coords({bumped_col: [n]})
            ds_expanded = xr.concat([ds_weather, new_slice], dim=bumped_col)
        new_weather_nc = analysis.analysis_paths.analysis_dir / f"_rerun_test_input_{kind}_weather.nc"
        ds_expanded.to_netcdf(new_weather_nc)

        # Delete stale iloc-indexed model logs for the victim's row positions. Without this,
        # the new scenario at victim_iloc passes the log-based completion check (reading the
        # victim's old log) and the model run is skipped, leaving outputs missing.
        simlog_dir = analysis.analysis_paths.simlog_directory
        for model_type in ("triton", "tritonswmm", "swmm"):
            for vic_iloc in victim_ilocs:
                stale_log = simlog_dir / f"model_{model_type}_evt{vic_iloc}.log"
                stale_log.unlink(missing_ok=True)
    elif kind == "sensitivity":
        donor_sa_id = donor_key[0]
        remove_sa_id = remove_key[0]
        src = Path(analysis.cfg_analysis.sensitivity_analysis)
        if src.suffix.lower() in (".xlsx", ".xls"):
            df = pd.read_excel(src)
        else:
            df = pd.read_csv(src)
        donor_row = df[df["sa_id"].astype(str) == str(donor_sa_id)].iloc[0].copy()
        used_sa_ids = set(df["sa_id"].astype(str).tolist())
        n = 9000
        while f"rerun_test_{n}" in used_sa_ids:
            n += 1
        new_sa_id = f"rerun_test_{n}"
        donor_row["sa_id"] = new_sa_id
        df = df[df["sa_id"].astype(str) != str(remove_sa_id)].copy()
        df = pd.concat([df, donor_row.to_frame().T], ignore_index=True)
        # Cloned-row event_id slug is identical to the donor row's event_id (only sa_id changed).
        new_key = (new_sa_id, donor_key[1])
    else:
        raise ValueError(f"unknown kind: {kind!r}")

    out = analysis.analysis_paths.analysis_dir / f"_rerun_test_input_{kind}.csv"
    df.to_csv(out, index=False)
    # Preserve the original CSV's mtime on the mutated copy so Snakemake's mtime trigger
    # does not fire for untouched scenarios. The trigger should only fire for the new/removed
    # scenario (via the input trigger on the expanded scenario set), not for unchanged ones.
    import os
    src_stat = os.stat(src)
    os.utime(out, (src_stat.st_atime, src_stat.st_mtime))
    return out, new_key


def reinstantiate_analysis_pointing_at_csv(analysis, *, kind: str, mutated_csv_path: Path):
    """Re-instantiate a TRITONSWMM_analysis from the same configs but with the
    weather/sensitivity CSV path swapped to mutated_csv_path.

    Implementation: write a modified copy of the analysis YAML to a tmp path with
    the relevant field updated, then call Toolkit.from_configs (toolkit.py:96-148).
    Returns the public .analysis attribute (toolkit.py:94).
    """
    import yaml

    from hhemt.toolkit import Toolkit

    sys_yaml = analysis._system.system_config_yaml
    ana_yaml = analysis.analysis_config_yaml

    with open(ana_yaml) as fp:
        ana_dict = yaml.safe_load(fp)

    if kind == "multi_sim":
        ana_dict["weather_events_to_simulate"] = str(mutated_csv_path)
        # Point at the expanded weather NetCDF if mutate_scenario_csv created one alongside the CSV
        expanded_weather_nc = mutated_csv_path.parent / f"_rerun_test_input_{kind}_weather.nc"
        if expanded_weather_nc.exists():
            ana_dict["weather_timeseries"] = str(expanded_weather_nc)
    elif kind == "sensitivity":
        ana_dict["sensitivity_analysis"] = str(mutated_csv_path)
    else:
        raise ValueError(f"unknown kind: {kind!r}")

    new_ana_yaml = mutated_csv_path.with_suffix(".analysis.yaml")
    with open(new_ana_yaml, "w") as fp:
        yaml.safe_dump(ana_dict, fp)

    # Preserve the original YAML's mtime on the new YAML so Snakemake's mtime trigger
    # does not fire for untouched scenarios. Without this, the freshly-written YAML is
    # newer than all baseline output files, causing mtime-triggered reruns for every scenario.
    import os
    orig_stat = os.stat(ana_yaml)
    os.utime(new_ana_yaml, (orig_stat.st_atime, orig_stat.st_mtime))

    return Toolkit.from_configs(sys_yaml, new_ana_yaml).analysis


def assert_rerun_trigger_correctness(
    *,
    before: dict[tuple[str | None, str], dict[Path, float]],
    after: dict[tuple[str | None, str], dict[Path, float]],
    added_key: tuple[str | None, str],
    removed_key: tuple[str | None, str],
    baseline_scenario_keys: list[tuple[str | None, str]],
) -> None:
    """Validate the four-part rerun-trigger contract.

    Keys are typed (sa_id, event_id) tuples matching snapshot_scenario_output_mtimes;
    sa_id=None for multi_sim, sa_id=str for sensitivity. Tuple-equality eliminates
    the substring-suffix collision class that string-concat namespacing would carry.
    """
    # (a) added scenario has outputs in 'after' with at least one file
    assert added_key in after, (
        f"added scenario {added_key!r} produced no outputs after rerun; "
        f"after keys={list(after)}"
    )

    # (b) untouched scenarios have IDENTICAL mtimes
    untouched = [k for k in baseline_scenario_keys if k != removed_key and k != added_key]
    for key in untouched:
        if key not in after:
            raise AssertionError(
                f"untouched scenario {key!r} disappeared from rerun analysis state"
            )
        before_files = before[key]
        after_files = after[key]
        for path, before_mtime in before_files.items():
            after_mtime = after_files.get(path)
            if after_mtime is None:
                raise AssertionError(
                    f"untouched scenario {key!r}: file {path} disappeared after rerun"
                )
            if after_mtime != before_mtime:
                raise AssertionError(
                    f"untouched scenario {key!r}: file {path} mtime changed "
                    f"({before_mtime} -> {after_mtime}); rerun-triggers misfired"
                )

    # (c) removed scenario is no longer in 'after' analysis state
    assert removed_key not in after, (
        f"removed scenario {removed_key!r} still present in rerun analysis state"
    )

    # (d) removed scenario's old output mtimes (orphans on disk) are unchanged
    before_removed_files = before[removed_key]
    for path, before_mtime in before_removed_files.items():
        if not path.exists():
            continue  # acceptable: file was deleted
        if path.stat().st_mtime != before_mtime:
            raise AssertionError(
                f"removed scenario {removed_key!r}: orphan file {path} was re-touched "
                f"by rerun (mtime {before_mtime} -> {path.stat().st_mtime})"
            )


def assert_alive_set_reconciled(builder, expected_rule_tokens: list[str]) -> None:
    """Assert ``_reconcile_inflight_submissions`` returns the expected alive set.

    Per sentinel-system-v2 Phase 2 — standardized helper for v2 reconcile tests.
    """
    alive = builder._reconcile_inflight_submissions()
    observed_tokens = sorted(t for t, _ in alive)
    if observed_tokens != sorted(expected_rule_tokens):
        pytest.fail(f"alive set mismatch: expected {sorted(expected_rule_tokens)}, got {observed_tokens}")


def assert_du_sentinel_present(
    scope_dir: Path,
    *,
    expected_min_bytes: int | None = None,
) -> None:
    """Assert that ``{scope_dir}/_status/_du.json`` exists, is parseable, and reports
    at least ``expected_min_bytes`` of disk utilization (when provided).

    Used by Phase 1 V-P1.3 and Phase 2 V-P2.4 per F-I Flag 8 / D6 standardized-assertion
    preference — keeps DU sentinel test assertions in the toolkit's helper layer rather
    than scattered inline ``stat`` / ``json.loads`` calls.
    """
    import json
    sentinel = scope_dir / "_status" / "_du.json"
    if not sentinel.exists():
        raise AssertionError(f"DU sentinel missing at {sentinel}")
    try:
        payload = json.loads(sentinel.read_text())
    except (OSError, ValueError) as e:
        raise AssertionError(f"DU sentinel at {sentinel} is unparseable: {e}") from e
    if "disk_utilization_bytes" not in payload:
        raise AssertionError(
            f"DU sentinel at {sentinel} is missing the disk_utilization_bytes field: {payload!r}"
        )
    if expected_min_bytes is not None:
        actual = int(payload["disk_utilization_bytes"])
        if actual < expected_min_bytes:
            raise AssertionError(
                f"DU sentinel at {sentinel} reports disk_utilization_bytes={actual}, "
                f"below expected_min_bytes={expected_min_bytes}"
            )


def assert_du_mtime_preserved(
    scope_dir: Path,
    *,
    prior_mtime: float,
) -> None:
    """Assert that ``{scope_dir}/_status/_du.json``'s mtime is unchanged from ``prior_mtime``.

    Used by V-P1.3 to verify the compare-and-write contract preserves mtime across
    idempotent workflow re-invocations. Captures the file's current ``st_mtime`` and
    compares for exact equality to the prior value. Any drift fails the assertion —
    the contract is bit-for-bit mtime preservation, not approximate-time preservation.
    """
    sentinel = scope_dir / "_status" / "_du.json"
    if not sentinel.exists():
        raise AssertionError(
            f"DU sentinel missing at {sentinel} during mtime-preservation check"
        )
    current_mtime = sentinel.stat().st_mtime
    if current_mtime != prior_mtime:
        raise AssertionError(
            f"DU sentinel at {sentinel} mtime advanced: prior={prior_mtime} "
            f"current={current_mtime} (compare-and-write contract violated)"
        )
