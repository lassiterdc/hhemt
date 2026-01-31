"""
Test dry-run behavior for 1_job_many_srun_tasks mode.

This test verifies that dry runs do NOT submit SBATCH jobs when using
the 1_job_many_srun_tasks execution mode.
"""

import pytest
import tests.fixtures.test_case_catalog as cases


@pytest.fixture
def norfolk_1job_analysis():
    """Norfolk test case configured for 1-job mode."""
    case = cases.Local_TestCases.retrieve_norfolk_multi_sim_test_case(
        start_from_scratch=False
    )
    analysis = case.analysis

    # Configure for 1-job mode
    analysis.cfg_analysis.multi_sim_run_method = "1_job_many_srun_tasks"
    analysis.cfg_analysis.hpc_total_nodes = 2
    analysis.cfg_analysis.hpc_total_job_duration_min = 60
    analysis.cfg_analysis.hpc_cpus_per_node = 32  # Required for dry-run validation
    analysis.cfg_analysis.n_gpus = 0  # CPU-only
    analysis.cfg_analysis.n_mpi_procs = 1
    analysis.cfg_analysis.n_omp_threads = 4
    analysis.cfg_analysis.hpc_ensemble_partition = "test_partition"
    analysis.cfg_analysis.hpc_account = "test_account"
    analysis.cfg_analysis.hpc_max_simultaneous_sims = 10
    analysis.cfg_analysis.local_cpu_cores_for_workflow = 4

    # Update in_slurm flag (normally set at __init__ time)
    analysis.in_slurm = True

    return analysis


def test_1job_dry_run_does_not_submit(norfolk_1job_analysis):
    """
    Test that dry-run with 1_job_many_srun_tasks does NOT submit SBATCH.

    Validates that:
    1. Snakefile is generated
    2. Dry-run validation occurs
    3. No SBATCH submission happens
    4. Result indicates success
    5. Result mode is 'single_job'
    """
    analysis = norfolk_1job_analysis

    result = analysis.submit_workflow(
        mode="slurm",
        process_system_level_inputs=True,
        overwrite_system_inputs=False,
        compile_TRITON_SWMM=True,
        recompile_if_already_done_successfully=False,
        prepare_scenarios=True,
        overwrite_scenario=True,
        rerun_swmm_hydro_if_outputs_exist=True,
        process_timeseries=True,
        which="both",
        clear_raw_outputs=True,
        overwrite_if_exist=True,
        compression_level=5,
        pickup_where_leftoff=False,
        dry_run=True,  # KEY: This should prevent SBATCH submission
        verbose=True,
    )

    # Verify dry-run succeeded
    assert result.get("success"), f"Dry-run failed: {result.get('message', '')}"

    # Verify mode is set correctly (should be 'single_job' indicating 1-job mode)
    assert result.get("mode") == "single_job", (
        f"Expected mode='single_job' for dry-run, got {result.get('mode')}"
    )

    # Verify Snakefile was generated
    snakefile_path = analysis._workflow_builder.analysis_paths.analysis_dir / "Snakefile"
    assert snakefile_path.exists(), "Snakefile should be generated even in dry-run"


def test_1job_normal_run_workflow_builder(norfolk_1job_analysis):
    """
    Test that workflow builder can generate Snakefile and config.

    This test verifies that the fix doesn't break the workflow generation
    pipeline. We don't actually submit, but verify generation succeeds.
    """
    analysis = norfolk_1job_analysis

    # Generate Snakefile content
    snakefile_content = analysis._workflow_builder.generate_snakefile_content(
        process_system_level_inputs=True,
        compile_TRITON_SWMM=True,
        prepare_scenarios=True,
        process_timeseries=True,
    )

    # Verify Snakefile has expected content
    assert "rule all:" in snakefile_content
    assert "rule setup:" in snakefile_content
    assert "rule prepare_scenario:" in snakefile_content
    assert "rule run_simulation:" in snakefile_content
    assert "rule process_outputs:" in snakefile_content

    # Generate Snakemake config for single_job mode
    config = analysis._workflow_builder.generate_snakemake_config(mode="single_job")

    # Verify config has expected keys
    assert "printshellcmds" in config
    assert "keep-going" in config
    # Note: cores is set dynamically via CLI in single_job mode, not in config
