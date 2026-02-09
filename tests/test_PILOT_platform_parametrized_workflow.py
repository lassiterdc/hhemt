"""PILOT: Platform-parametrized workflow generation tests.

This is a Phase 6a.1 pilot to validate the parametrization approach before
applying it broadly. These tests run the same workflow generation logic
across multiple platforms (local, UVA, Frontier) using pytest parametrization.

Status: PILOT - Experimental
Original tests: test_PC_04, test_UVA_02, test_frontier_03
Next steps: If successful, expand to more test types and retire originals
"""

import pytest

import tests.fixtures.test_case_catalog as cases
import tests.utils_for_testing as tst_ut


# Platform configuration for parametrization
PLATFORMS = [
    pytest.param(
        ("local", cases.Local_TestCases, "retrieve_norfolk_multi_sim_test_case"),
        id="local",
    ),
    pytest.param(
        ("uva", cases.UVA_TestCases, "retrieve_norfolk_UVA_multisim_1cpu_case"),
        marks=pytest.mark.skipif(not tst_ut.on_UVA_HPC(), reason="UVA platform only"),
        id="uva",
    ),
    pytest.param(
        ("frontier", cases.Frontier_TestCases, "retrieve_norfolk_frontier_multisim_cpu_serial_case"),
        marks=pytest.mark.skipif(not tst_ut.on_frontier(), reason="Frontier platform only"),
        id="frontier",
    ),
]


@pytest.fixture
def platform_analysis(request):
    """Parametrized fixture providing analysis for different platforms.

    This fixture demonstrates the unified approach: instead of separate
    fixtures per platform, we parametrize a single fixture and use
    pytest.mark.skipif to handle platform availability.
    """
    platform_name, catalog_class, method_name = request.param

    # Get the retrieval method from the catalog class
    retrieve_method = getattr(catalog_class, method_name)

    # Call it with start_from_scratch=True
    case = retrieve_method(start_from_scratch=True)

    return case.analysis


@pytest.mark.parametrize("platform_analysis", PLATFORMS, indirect=True)
def test_workflow_generation_parametrized(platform_analysis):
    """Test Snakemake workflow generation across platforms (PILOT).

    This single test runs on local/UVA/Frontier depending on platform
    availability. It validates that workflow generation logic is
    platform-agnostic.

    Original tests:
    - test_PC_04_multisim_with_snakemake::test_snakemake_local_workflow_generation_and_write
    - test_UVA_02_multisim_with_snakemake::test_snakemake_slurm_workflow_generation_and_write
    - test_frontier_03_snakemake_multisim_CPU::test_snakemake_slurm_workflow_generation_and_write
    """
    analysis = platform_analysis

    # Generate Snakefile content (same logic across all platforms)
    snakefile_content = analysis._workflow_builder.generate_snakefile_content(
        process_system_level_inputs=True,
        compile_TRITON_SWMM=True,
        prepare_scenarios=True,
        process_timeseries=True,
    )

    # Write Snakefile
    snakefile_path = tst_ut.write_snakefile(analysis, snakefile_content)

    # Validate Snakefile was created
    tst_ut.assert_file_exists(snakefile_path, "Snakefile")
    assert len(snakefile_path.read_text()) > 100, "Snakefile is too short"

    # Validate required rules present
    content = snakefile_path.read_text()
    tst_ut.assert_snakefile_has_rules(
        content,
        [
            "all",
            "setup",
            "prepare_scenario",
            "run_simulation",
            "process_outputs",
            "consolidate",
        ],
    )

    # Platform-specific validation could go here if needed
    # For now, workflow generation is fully platform-agnostic


@pytest.mark.parametrize("platform_analysis", PLATFORMS, indirect=True)
def test_workflow_config_flags_parametrized(platform_analysis):
    """Test Snakefile config flags across platforms (PILOT).

    Validates that configuration parameters are correctly embedded
    in generated Snakefile content (compression level, which models, etc.).

    Original tests:
    - test_PC_04_multisim_with_snakemake::test_snakemake_workflow_config_generation
    """
    analysis = platform_analysis

    # Generate Snakefile with specific config options
    snakefile_content = analysis._workflow_builder.generate_snakefile_content(
        process_system_level_inputs=False,
        compile_TRITON_SWMM=True,
        prepare_scenarios=True,
        process_timeseries=True,
        which="both",
        clear_raw_outputs=True,
        compression_level=5,
    )

    # Validate config flags are present in Snakefile
    tst_ut.assert_snakefile_has_flags(
        snakefile_content,
        [
            "--compression-level 5",
            "--which both",
            f"--system-config {analysis._system.system_config_yaml}",
            f"--analysis-config {analysis.analysis_config_yaml}",
        ],
    )


# NOTE: Execution tests (dry_run, end_to_end) are NOT included in this pilot
# because they have platform-specific differences (SLURM vs local).
# Those will be addressed in Phase 6a.2 after validating this pilot approach.
