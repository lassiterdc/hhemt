"""
Test SWMM THREADS parameter dynamic configuration.

Validates Phase 1 and Phase 2 of enable_swmm_threading_control implementation.
"""

import pytest

import tests.fixtures.test_case_catalog as cases


@pytest.mark.usefixtures("tritonswmm_cpu_compiled")
def test_swmm_threads_different_values():
    """
    Test that different n_omp_threads values are correctly applied.

    Validates:
    1. Configuration override works
    2. Different thread counts are written correctly
    3. Method works for various thread values (1, 4, 8)
    """
    case = cases.Local_TestCases.retrieve_norfolk_single_sim_test_case(start_from_scratch=False)
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
            with open(paths.swmm_full_inp) as fp:
                content = fp.read()
                expected_line = f"THREADS              {n_threads}"
                assert expected_line in content, (
                    f"full.inp should have THREADS={n_threads} but got: {_extract_options_section(content)}"
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
