"""
Test Snakemake profile generation for 1_job_many_srun_tasks mode.

These tests verify that the Snakemake profile is correctly configured
for dynamic concurrency (cores passed via CLI, not hardcoded in profile).
"""

import pytest
import yaml
import tests.fixtures.test_case_catalog as cases


@pytest.fixture
def norfolk_1job_cpu_only():
    """Norfolk test case configured for 1-job mode (CPU-only)."""
    case = cases.Local_TestCases.retrieve_norfolk_multi_sim_test_case(
        start_from_scratch=False
    )
    analysis = case.analysis

    # Configure for 1-job mode with CPU-only
    analysis.cfg_analysis.multi_sim_run_method = "1_job_many_srun_tasks"
    analysis.cfg_analysis.hpc_total_nodes = 2
    analysis.cfg_analysis.hpc_total_job_duration_min = 60
    analysis.cfg_analysis.n_gpus = 0  # CPU-only
    analysis.cfg_analysis.n_mpi_procs = 1
    analysis.cfg_analysis.n_omp_threads = 4

    return analysis


@pytest.fixture
def norfolk_1job_with_gpus():
    """Norfolk test case configured for 1-job mode with GPUs."""
    case = cases.Local_TestCases.retrieve_norfolk_multi_sim_test_case(
        start_from_scratch=False
    )
    analysis = case.analysis

    # Configure for 1-job mode with GPUs
    analysis.cfg_analysis.multi_sim_run_method = "1_job_many_srun_tasks"
    analysis.cfg_analysis.hpc_total_nodes = 2
    analysis.cfg_analysis.hpc_total_job_duration_min = 60
    analysis.cfg_analysis.hpc_gpus_per_node = 8  # Frontier-like
    analysis.cfg_analysis.n_gpus = 1  # Use GPUs
    analysis.cfg_analysis.n_mpi_procs = 1
    analysis.cfg_analysis.n_omp_threads = 4

    return analysis


def test_1job_profile_no_cores_cpu_only(norfolk_1job_cpu_only):
    """Verify 1-job profile doesn't hardcode cores (CPU-only)."""
    from TRITON_SWMM_toolkit.workflow import SnakemakeWorkflowBuilder

    analysis = norfolk_1job_cpu_only
    workflow_builder = SnakemakeWorkflowBuilder(analysis)

    # Generate profile
    config_dict = workflow_builder.generate_snakemake_config(mode="single_job")
    config_dir = workflow_builder.write_snakemake_config(config_dict, mode="single_job")

    # Read the config file
    config_file = config_dir / "config.yaml"
    assert config_file.exists(), "Config file should be created"

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # Cores should NOT be in profile (passed via CLI in SBATCH script)
    assert (
        "cores" not in config
    ), "cores should not be hardcoded in profile for 1-job mode"

    # Should have keep-going and latency-wait
    assert config["keep-going"] is True, "Should continue on failures"
    assert config["latency-wait"] == 30, "Should have latency-wait"

    # No GPU resources for CPU-only
    assert "resources" not in config, "Should not have GPU resources for CPU-only mode"


def test_1job_profile_with_gpu_resources(norfolk_1job_with_gpus):
    """Verify 1-job profile does NOT include GPU resources (passed via CLI instead)."""
    from TRITON_SWMM_toolkit.workflow import SnakemakeWorkflowBuilder

    analysis = norfolk_1job_with_gpus
    workflow_builder = SnakemakeWorkflowBuilder(analysis)

    # Generate profile
    config_dict = workflow_builder.generate_snakemake_config(mode="single_job")
    config_dir = workflow_builder.write_snakemake_config(config_dict, mode="single_job")

    # Read the config file
    config_file = config_dir / "config.yaml"
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # GPU resources should NOT be in profile (passed via CLI in SBATCH script)
    assert (
        "resources" not in config
    ), "GPU resources should not be in profile (passed via CLI)"

    # Should not have cores either
    assert "cores" not in config, "cores should not be hardcoded"


# Note: GPU-specific validation tests removed since GPU resources are now
# calculated in the SBATCH script (not in the profile). See SBATCH generation
# tests for GPU validation.
