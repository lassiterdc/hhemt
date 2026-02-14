"""
Test SWMM THREADS parameter dynamic configuration.

Validates Phase 1 and Phase 2 of enable_swmm_threading_control implementation.
"""

import pytest
import tests.fixtures.test_case_catalog as cases
import tests.utils_for_testing as tst_ut

pytestmark = pytest.mark.skipif(
    tst_ut.is_scheduler_context(), reason="Only runs on non-HPC systems."
)


def test_swmm_threads_updated_in_inp_files():
    """
    Test that n_omp_threads configuration dynamically updates THREADS parameter.

    Validates:
    1. THREADS parameter in hydro.inp matches n_omp_threads
    2. THREADS parameter in full.inp matches n_omp_threads
    3. Both files are created during scenario preparation
    4. Configuration value propagates correctly from analysis config
    """
    # Get test case with cached system inputs (faster)
    case = cases.Local_TestCases.retrieve_norfolk_multi_sim_test_case(
        start_from_scratch=False
    )
    analysis = case.analysis

    # Verify configuration
    expected_threads = analysis.cfg_analysis.n_omp_threads
    # Note: actual value depends on test config, just verify it propagates correctly
    assert expected_threads >= 1, f"n_omp_threads must be >= 1, got {expected_threads}"

    # Prepare scenarios directly (without Snakemake workflow)
    # This is faster than full workflow and sufficient to test .inp file modification
    for event_iloc in analysis.df_sims.index:
        proc = analysis._retrieve_sim_run_processing_object(event_iloc)
        proc._scenario.prepare_scenario(
            overwrite_scenario_if_already_set_up=True,
            rerun_swmm_hydro_if_outputs_exist=True,
        )

    # Verify all scenarios have updated THREADS parameter
    for event_iloc in analysis.df_sims.index:
        proc = analysis._retrieve_sim_run_processing_object(event_iloc)
        paths = proc.scen_paths

        # Check hydrology.inp (if hydrology enabled)
        if analysis._system.cfg_system.toggle_use_swmm_for_hydrology:
            tst_ut.assert_file_exists(paths.swmm_hydro_inp, "SWMM hydrology .inp")
            with open(paths.swmm_hydro_inp, "r") as fp:
                content = fp.read()
                expected_line = f"THREADS              {expected_threads}"
                assert expected_line in content, (
                    f"hydro.inp for event {event_iloc} missing '{expected_line}'. "
                    f"Found in [OPTIONS]: {_extract_options_section(content)}"
                )

        # Check full.inp (if full model enabled)
        if analysis._system.cfg_system.toggle_swmm_model:
            tst_ut.assert_file_exists(paths.swmm_full_inp, "SWMM full .inp")
            with open(paths.swmm_full_inp, "r") as fp:
                content = fp.read()
                expected_line = f"THREADS              {expected_threads}"
                assert expected_line in content, (
                    f"full.inp for event {event_iloc} missing '{expected_line}'. "
                    f"Found in [OPTIONS]: {_extract_options_section(content)}"
                )


def test_swmm_threads_different_values():
    """
    Test that different n_omp_threads values are correctly applied.

    Validates:
    1. Configuration override works
    2. Different thread counts are written correctly
    3. Method works for various thread values (1, 4, 8)
    """
    case = cases.Local_TestCases.retrieve_norfolk_single_sim_test_case(
        start_from_scratch=False
    )
    analysis = case.analysis

    # Test with different thread counts
    for n_threads in [1, 4, 8]:
        # Override configuration
        analysis.cfg_analysis.n_omp_threads = n_threads

        # Prepare single scenario directly
        event_iloc = analysis.df_sims.index[0]
        proc = analysis._retrieve_sim_run_processing_object(event_iloc)
        proc._scenario.prepare_scenario(
            rerun_swmm_hydro_if_outputs_exist=True,
        )

        # Verify the updated value
        paths = proc.scen_paths

        if analysis._system.cfg_system.toggle_swmm_model:
            with open(paths.swmm_full_inp, "r") as fp:
                content = fp.read()
                expected_line = f"THREADS              {n_threads}"
                assert expected_line in content, (
                    f"full.inp should have THREADS={n_threads} but got: "
                    f"{_extract_options_section(content)}"
                )


def _extract_options_section(inp_content: str) -> str:
    """Extract [OPTIONS] section from .inp file for debugging."""
    lines = inp_content.split("\n")
    in_options = False
    options_lines = []

    for line in lines:
        if "[OPTIONS]" in line:
            in_options = True
            options_lines.append(line)
        elif in_options and line.startswith("["):
            break
        elif in_options:
            options_lines.append(line)

    return "\n".join(options_lines)
