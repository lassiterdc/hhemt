"""
Test SBATCH script generation for 1_job_many_srun_tasks mode.

These tests verify that the SBATCH submission script is correctly generated
with dynamic concurrency (no --ntasks, using --exclusive instead).
"""

import pytest
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
    analysis.cfg_analysis.hpc_ensemble_partition = "test_partition"
    analysis.cfg_analysis.hpc_account = "test_account"

    # Update in_slurm flag (normally set at __init__ time)
    analysis.in_slurm = True

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
    analysis.cfg_analysis.hpc_ensemble_partition = "test_partition"
    analysis.cfg_analysis.hpc_account = "test_account"

    # Update in_slurm flag (normally set at __init__ time)
    analysis.in_slurm = True

    return analysis


def test_1job_sbatch_script_cpu_only(norfolk_1job_cpu_only):
    """Verify SBATCH script for 1-job mode (CPU-only)."""
    from TRITON_SWMM_toolkit.workflow import SnakemakeWorkflowBuilder

    analysis = norfolk_1job_cpu_only
    workflow_builder = SnakemakeWorkflowBuilder(analysis)

    # Generate and write the snakemake config
    config = workflow_builder.generate_snakemake_config(mode="single_job")
    config_dir = workflow_builder.write_snakemake_config(config, mode="single_job")

    # Create a dummy snakefile path (doesn't need to exist for this test)
    snakefile_path = analysis.analysis_paths.analysis_dir / "Snakefile"

    # Generate SBATCH script
    script_path = workflow_builder._generate_single_job_submission_script(
        snakefile_path, config_dir
    )

    # Read the generated script
    assert script_path.exists(), "SBATCH script should be created"
    script_content = script_path.read_text()

    # Verify correct SBATCH directives
    assert "--exclusive" in script_content, "Should use --exclusive"
    assert "--nodes=2" in script_content, "Should request 2 nodes"
    assert (
        "--ntasks" not in script_content
    ), "Should NOT have --ntasks (dynamic concurrency)"
    assert (
        "--cpus-per-task" not in script_content
    ), "Should NOT have --cpus-per-task (dynamic concurrency)"
    assert (
        "--mem=0" not in script_content
    ), "Should NOT have --mem=0 (redundant with --exclusive)"

    # Verify dynamic CPU calculation
    assert (
        "TOTAL_CPUS=$((SLURM_CPUS_ON_NODE * SLURM_JOB_NUM_NODES))" in script_content
    ), "Should calculate TOTAL_CPUS dynamically"
    assert (
        "--cores $TOTAL_CPUS" in script_content
    ), "Should pass dynamic cores to Snakemake"

    # Verify GPU directive NOT present for CPU-only
    assert (
        "--gres=gpu:" not in script_content
    ), "Should not have GPU directive for CPU-only"
    assert (
        "TOTAL_GPUS" not in script_content
    ), "Should not calculate TOTAL_GPUS for CPU-only"
    assert (
        "--resources gpu=" not in script_content
    ), "Should not pass GPU resources for CPU-only"

    # Verify error handling for missing SLURM env vars
    assert (
        'if [ -z "$SLURM_CPUS_ON_NODE" ]' in script_content
    ), "Should check for SLURM_CPUS_ON_NODE"
    assert (
        "Cannot determine CPU allocation" in script_content
    ), "Should have error message"
    assert "exit 1" in script_content, "Should exit with error code"

    # Verify other standard directives
    assert "#SBATCH --partition=test_partition" in script_content
    assert "#SBATCH --account=test_account" in script_content
    assert "#SBATCH --time=01:00:00" in script_content  # 60 minutes -> 01:00:00


def test_1job_sbatch_script_with_gpus(norfolk_1job_with_gpus):
    """Verify SBATCH script for 1-job mode with GPUs."""
    from TRITON_SWMM_toolkit.workflow import SnakemakeWorkflowBuilder

    analysis = norfolk_1job_with_gpus
    workflow_builder = SnakemakeWorkflowBuilder(analysis)

    # Generate the snakemake config
    config = workflow_builder.generate_snakemake_config(mode="single_job")
    config_dir = workflow_builder.write_snakemake_config(config, mode="single_job")

    # Create a dummy snakefile path
    snakefile_path = analysis.analysis_paths.analysis_dir / "Snakefile"

    # Generate SBATCH script
    script_path = workflow_builder._generate_single_job_submission_script(
        snakefile_path, config_dir
    )

    # Read the generated script
    script_content = script_path.read_text()

    # Verify GPU directive is present in SBATCH
    assert (
        "--gres=gpu:8" in script_content
    ), "Should request 8 GPUs per node with --gres"

    # Verify GPU calculation in bash script
    assert (
        "TOTAL_GPUS=$((SLURM_JOB_NUM_NODES * 8))" in script_content
    ), "Should calculate TOTAL_GPUS dynamically (2 nodes × 8 GPUs/node = 16 total)"

    # Verify GPU resources passed via CLI
    assert (
        "--resources gpu=$TOTAL_GPUS" in script_content
    ), "Should pass GPU resources via CLI argument"

    # Verify still has other correct directives
    assert "--exclusive" in script_content
    assert "--nodes=2" in script_content
    assert "--ntasks" not in script_content
    assert "--cpus-per-task" not in script_content


def test_1job_sbatch_script_error_if_cpus_not_set(norfolk_1job_cpu_only):
    """Verify SBATCH script includes error handling for missing SLURM_CPUS_ON_NODE."""
    from TRITON_SWMM_toolkit.workflow import SnakemakeWorkflowBuilder

    analysis = norfolk_1job_cpu_only
    workflow_builder = SnakemakeWorkflowBuilder(analysis)

    config = workflow_builder.generate_snakemake_config(mode="single_job")
    config_dir = workflow_builder.write_snakemake_config(config, mode="single_job")
    snakefile_path = analysis.analysis_paths.analysis_dir / "Snakefile"

    script_path = workflow_builder._generate_single_job_submission_script(
        snakefile_path, config_dir
    )

    script_content = script_path.read_text()

    # Should contain error handling for missing SLURM env var
    assert 'if [ -z "$SLURM_CPUS_ON_NODE" ]' in script_content
    assert "ERROR: SLURM_CPUS_ON_NODE not set" in script_content
    assert "Cannot determine CPU allocation" in script_content
    assert "exit 1" in script_content


def test_1job_sbatch_requires_hpc_total_nodes(norfolk_1job_cpu_only):
    """Verify that SBATCH script generation fails without hpc_total_nodes."""
    from TRITON_SWMM_toolkit.workflow import SnakemakeWorkflowBuilder

    analysis = norfolk_1job_cpu_only
    analysis.cfg_analysis.hpc_total_nodes = None  # Remove required field

    workflow_builder = SnakemakeWorkflowBuilder(analysis)
    config = workflow_builder.generate_snakemake_config(mode="single_job")
    config_dir = workflow_builder.write_snakemake_config(config, mode="single_job")
    snakefile_path = analysis.analysis_paths.analysis_dir / "Snakefile"

    # Should raise assertion error
    with pytest.raises(AssertionError, match="hpc_total_nodes required"):
        workflow_builder._generate_single_job_submission_script(
            snakefile_path, config_dir
        )


def test_1job_sbatch_requires_hpc_gpus_per_node_when_using_gpus(norfolk_1job_with_gpus):
    """Verify that GPU mode requires hpc_gpus_per_node."""
    from TRITON_SWMM_toolkit.workflow import SnakemakeWorkflowBuilder

    analysis = norfolk_1job_with_gpus
    analysis.cfg_analysis.hpc_gpus_per_node = None  # Remove required field

    workflow_builder = SnakemakeWorkflowBuilder(analysis)
    config = workflow_builder.generate_snakemake_config(mode="single_job")
    config_dir = workflow_builder.write_snakemake_config(config, mode="single_job")
    snakefile_path = analysis.analysis_paths.analysis_dir / "Snakefile"

    # Should raise assertion error during SBATCH script generation
    with pytest.raises(
        AssertionError, match="hpc_gpus_per_node required when using GPUs"
    ):
        workflow_builder._generate_single_job_submission_script(
            snakefile_path, config_dir
        )


def test_1job_sbatch_conda_initialization_present(norfolk_1job_cpu_only):
    """Verify that SBATCH script includes conda initialization for non-interactive shells."""
    from TRITON_SWMM_toolkit.workflow import SnakemakeWorkflowBuilder

    analysis = norfolk_1job_cpu_only
    workflow_builder = SnakemakeWorkflowBuilder(analysis)

    config = workflow_builder.generate_snakemake_config(mode="single_job")
    config_dir = workflow_builder.write_snakemake_config(config, mode="single_job")
    snakefile_path = analysis.analysis_paths.analysis_dir / "Snakefile"

    script_path = workflow_builder._generate_single_job_submission_script(
        snakefile_path, config_dir
    )

    script_content = script_path.read_text()

    # Verify conda initialization logic is present
    assert (
        "Initialize conda for non-interactive shell" in script_content
    ), "Should include conda initialization comment"
    assert (
        'if [ -f "${CONDA_PREFIX}/../etc/profile.d/conda.sh" ]' in script_content
    ), "Should check for conda.sh using CONDA_PREFIX"
    assert (
        'source "${CONDA_PREFIX}/../etc/profile.d/conda.sh"' in script_content
    ), "Should source conda.sh from CONDA_PREFIX path"
    assert (
        'eval "$(${CONDA_EXE} shell.bash hook)' in script_content
    ), "Should use using CONDA_EXE"
    assert (
        "ERROR: Cannot find conda initialization. CONDA_EXE and CONDA_PREFIX are both unset"
        in script_content
    ), "Should warn if conda.sh not found"

    # Verify conda initialization happens BEFORE conda activate
    init_pos = script_content.find("Initialize conda for non-interactive shell")
    activate_pos = script_content.find("conda activate triton_swmm_toolkit")
    assert init_pos > 0, "Conda initialization should be present"
    assert activate_pos > 0, "Conda activation should be present"
    assert (
        init_pos < activate_pos
    ), "Conda initialization must come before conda activate"
