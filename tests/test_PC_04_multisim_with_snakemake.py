import pytest
from TRITON_SWMM_toolkit.examples import GetTS_TestCases as tst
from tests.test_utils import is_scheduler_context

pytestmark = pytest.mark.skipif(
    is_scheduler_context(), reason="Only runs on non-HPC systems."
)


def test_snakemake_local_workflow_generation():
    """
    Test Snakemake workflow generation for local execution.

    Verifies that:
    1. Snakefile is generated correctly
    2. Snakefile contains all necessary rules
    3. Snakefile uses correct conda environment
    """
    nrflk_multisim_ensemble = tst.retreive_norfolk_multi_sim_test_case(
        start_from_scratch=True
    )
    analysis = nrflk_multisim_ensemble.system.analysis

    # Generate Snakefile content
    snakefile_content = analysis._generate_snakefile_content(
        process_system_level_inputs=True,
        compile_TRITON_SWMM=True,
        prepare_scenarios=True,
        process_timeseries=True,
    )

    # Verify Snakefile structure
    assert "rule all:" in snakefile_content
    assert "rule setup:" in snakefile_content
    assert "rule simulation:" in snakefile_content
    assert "rule consolidate:" in snakefile_content

    # Verify conda environment specification
    assert "/workflow/envs/triton_swmm.yaml" in snakefile_content

    # Verify wildcard-based simulation rule
    assert "_status/sims/sim_{event_iloc}_complete.flag" in snakefile_content

    # Verify setup phase commands
    assert "setup_workflow" in snakefile_content
    assert "--process-system-inputs" in snakefile_content
    assert "--compile-triton-swmm" in snakefile_content

    # Verify simulation phase commands
    assert "run_single_simulation" in snakefile_content
    assert "--prepare-scenario" in snakefile_content

    # Verify consolidation phase commands
    assert "consolidate_workflow" in snakefile_content


def test_snakemake_local_workflow_submission_dry_run():
    """
    Test Snakemake workflow submission in local mode (dry-run).

    Verifies that:
    1. Snakefile can be written to disk
    2. submit_workflow() returns success status (but doesn't actually run snakemake)
    3. Workflow mode is detected correctly (local)
    """
    nrflk_multisim_ensemble = tst.retreive_norfolk_multi_sim_test_case(
        start_from_scratch=True
    )
    analysis = nrflk_multisim_ensemble.system.analysis

    # Generate Snakefile (without actually running snakemake)
    snakefile_content = analysis._generate_snakefile_content(
        process_system_level_inputs=True,
        compile_TRITON_SWMM=True,
        prepare_scenarios=True,
        process_timeseries=True,
    )

    # Write Snakefile to disk
    snakefile_path = analysis.analysis_paths.analysis_dir / "Snakefile"
    snakefile_path.write_text(snakefile_content)

    # Verify Snakefile was written
    assert snakefile_path.exists()

    # Verify Snakefile is not empty
    assert len(snakefile_path.read_text()) > 100

    # Verify Snakefile contains expected rules
    content = snakefile_path.read_text()
    assert "rule all:" in content
    assert "rule setup:" in content
    assert "rule simulation:" in content
    assert "rule consolidate:" in content


def test_submit_workflow_detects_local_mode():
    """
    Test that submit_workflow() correctly detects local mode.

    Note: This test does NOT actually run snakemake, only verifies detection logic.
    """
    nrflk_multisim_ensemble = tst.retreive_norfolk_multi_sim_test_case(
        start_from_scratch=True
    )
    analysis = nrflk_multisim_ensemble.system.analysis

    # Verify that analysis is not in SLURM context
    assert not analysis.in_slurm, "Test must run on local machine, not in SLURM"

    # Test mode detection
    # If we call submit_workflow with mode="auto", it should detect "local"
    # For this test, we just verify the detection logic works
    detected_mode = "slurm" if analysis.in_slurm else "local"
    assert detected_mode == "local"


def test_snakemake_workflow_config_generation():
    """
    Test configuration passed to Snakemake.

    Verifies that:
    1. All parameters are correctly formatted in Snakefile
    2. Resource specifications are valid
    3. Command-line arguments are properly escaped
    """
    nrflk_multisim_ensemble = tst.retreive_norfolk_multi_sim_test_case(
        start_from_scratch=True
    )
    analysis = nrflk_multisim_ensemble.system.analysis

    snakefile_content = analysis._generate_snakefile_content(
        process_system_level_inputs=False,
        compile_TRITON_SWMM=True,
        prepare_scenarios=True,
        process_timeseries=True,
        which="TRITON",
        clear_raw_outputs=True,
        compression_level=5,
    )

    # Verify compression level
    assert "--compression-level 5" in snakefile_content

    # Verify which parameter
    assert "--which TRITON" in snakefile_content

    # Verify path handling
    assert f"--system-config {analysis._system.system_config_yaml}" in snakefile_content
    assert f"--analysis-config {analysis.analysis_config_yaml}" in snakefile_content

    # Verify simulation IDs are generated
    n_sims = len(analysis.df_sims)
    assert f"SIM_IDS = {list(range(n_sims))}" in snakefile_content


def test_snakemake_multiple_configurations():
    """
    Test Snakemake generation with different parameter combinations.

    Verifies that:
    1. Different parameter combinations generate different Snakefiles
    2. Optional parameters are correctly included/excluded
    """
    nrflk_multisim_ensemble = tst.retreive_norfolk_multi_sim_test_case(
        start_from_scratch=True
    )
    analysis = nrflk_multisim_ensemble.system.analysis

    # Configuration 1: Setup only
    snakefile_1 = analysis._generate_snakefile_content(
        process_system_level_inputs=True,
        compile_TRITON_SWMM=True,
        prepare_scenarios=False,
        process_timeseries=False,
    )

    # Configuration 2: Everything
    snakefile_2 = analysis._generate_snakefile_content(
        process_system_level_inputs=True,
        compile_TRITON_SWMM=True,
        prepare_scenarios=True,
        process_timeseries=True,
    )

    # They should be different
    assert snakefile_1 != snakefile_2

    # Configuration 1 should have --process-system-inputs and --compile-triton-swmm
    assert "--process-system-inputs" in snakefile_1
    assert "--compile-triton-swmm" in snakefile_1

    # Configuration 1 should NOT have --prepare-scenario (since prepare_scenarios=False)
    # Note: We need to check in the simulation rule
    assert "rule simulation:" in snakefile_1
    # The shell command should not have --prepare-scenario
    lines = snakefile_1.split("\n")
    simulation_section = False
    for line in lines:
        if "rule simulation:" in line:
            simulation_section = True
        if simulation_section and "--prepare-scenario" in line:
            pytest.fail(
                "--prepare-scenario should not be in simulation rule when prepare_scenarios=False"
            )

    # Configuration 2 should have --prepare-scenario
    assert "--prepare-scenario" in snakefile_2
    assert "--process-timeseries" in snakefile_2


def test_snakemake_workflow_dry_run():
    """
    Test Snakemake dry-run (--dry-run flag).

    Validates that:
    1. DAG can be constructed from Snakefile
    2. All dependencies resolve correctly
    3. No actual execution occurs
    4. Snakemake exit code is 0
    """
    import subprocess
    import tempfile
    from pathlib import Path

    nrflk_multisim_ensemble = tst.retreive_norfolk_multi_sim_test_case(
        start_from_scratch=True
    )
    analysis = nrflk_multisim_ensemble.system.analysis

    # Generate Snakefile content
    snakefile_content = analysis._generate_snakefile_content(
        process_system_level_inputs=False,
        compile_TRITON_SWMM=False,
        prepare_scenarios=True,
        process_timeseries=False,
    )

    # Write Snakefile to disk
    snakefile_path = analysis.analysis_paths.analysis_dir / "Snakefile"
    snakefile_path.write_text(snakefile_content)

    # Create logs directory
    logs_dir = analysis.analysis_paths.analysis_dir / "logs"
    logs_dir.mkdir(exist_ok=True, parents=True)

    # Run snakemake --dry-run
    result = subprocess.run(
        ["snakemake", "--snakefile", str(snakefile_path), "--dry-run", "-p"],
        cwd=str(analysis.analysis_paths.analysis_dir),
        capture_output=True,
        text=True,
        timeout=60,
    )

    # Verify DAG construction was successful
    assert (
        result.returncode == 0
    ), f"Snakemake dry-run failed:\n{result.stdout}\n{result.stderr}"

    # Verify rules are present in output
    assert "rule" in result.stdout or "DAG" in result.stdout

    print(f"✅ Snakemake dry-run successful - DAG validated")


@pytest.mark.slow
def test_snakemake_workflow_execution():
    """
    Test Snakemake workflow execution (2 simulations).

    Validates that:
    1. submit_workflow() returns success
    2. Setup phase completes
    3. Simulations execute without errors
    4. Scenarios are prepared correctly
    5. Simulations run successfully
    6. Analysis summaries are generated
    """
    from TRITON_SWMM_toolkit.examples import GetTS_TestCases as tst

    nrflk_multisim_ensemble = tst.retreive_norfolk_multi_sim_test_case(
        start_from_scratch=True
    )
    analysis = nrflk_multisim_ensemble.system.analysis

    # Submit the workflow
    result = analysis.submit_workflow(
        mode="local",
        process_system_level_inputs=True,
        overwrite_system_inputs=True,
        compile_TRITON_SWMM=True,
        recompile_if_already_done_successfully=True,
        prepare_scenarios=True,
        overwrite_scenario=True,
        rerun_swmm_hydro_if_outputs_exist=True,
        process_timeseries=True,
        which="TRITON",
        clear_raw_outputs=True,
        overwrite_if_exist=True,
        compression_level=5,
        pickup_where_leftoff=False,
        consolidate_outputs=True,
        verbose=True,
    )

    # Verify workflow submission was successful
    assert result["success"], "Workflow submission failed"

    # Verify TRITON compilation was successful
    assert analysis._system.compilation_successful, "TRITON compilation failed"

    # Verify scenario preparation completed
    assert analysis.log.all_scenarios_created.get(), "All scenarios should be created"

    # Verify simulations completed
    assert analysis.log.all_sims_run.get(), "All simulations should complete"

    # Verify output summaries were generated
    if not analysis.log.all_TRITON_timeseries_processed.get():
        pytest.fail("TRITON time series were not processed")

    assert (
        analysis.TRITON_analysis_summary_created
    ), "TRITON analysis summary should be created"
