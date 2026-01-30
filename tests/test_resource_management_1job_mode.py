"""
Test resource management for 1_job_many_srun_tasks mode.

These tests verify that _get_simulation_resource_requirements correctly returns
per-simulation requirements (without totals based on max_concurrent).
"""

import pytest
import tests.fixtures.test_case_catalog as cases


@pytest.fixture
def norfolk_analysis_cpu_only():
    """Norfolk test case configured with CPU-only simulation."""
    case = cases.Local_TestCases.retrieve_norfolk_multi_sim_test_case(
        start_from_scratch=False
    )
    analysis = case.analysis

    # Configure basic CPU-only simulation
    analysis.cfg_analysis.n_gpus = 0
    analysis.cfg_analysis.n_mpi_procs = 2
    analysis.cfg_analysis.n_omp_threads = 4
    analysis.cfg_analysis.n_nodes = 1
    analysis.cfg_analysis.mem_gb_per_cpu = 2

    return analysis


@pytest.fixture
def norfolk_analysis_with_gpus():
    """Norfolk test case configured with GPU simulation."""
    case = cases.Local_TestCases.retrieve_norfolk_multi_sim_test_case(
        start_from_scratch=False
    )
    analysis = case.analysis

    # Configure GPU simulation
    analysis.cfg_analysis.n_gpus = 2
    analysis.cfg_analysis.n_mpi_procs = 2
    analysis.cfg_analysis.n_omp_threads = 4
    analysis.cfg_analysis.n_nodes = 1
    analysis.cfg_analysis.mem_gb_per_cpu = 4

    return analysis


def test_sim_requirements_returns_per_sim_only(norfolk_analysis_cpu_only):
    """Verify method returns only per-simulation requirements (no totals)."""
    analysis = norfolk_analysis_cpu_only

    resource_manager = analysis._resource_manager
    sim_reqs = resource_manager._get_simulation_resource_requirements()

    # Should have per-sim requirements
    assert "n_nodes" in sim_reqs
    assert "n_cpus_per_sim" in sim_reqs
    assert "n_gpus" in sim_reqs
    assert "mem_mb_per_sim" in sim_reqs

    # Should NOT have totals (these are no longer calculated)
    assert "total_nodes" not in sim_reqs
    assert "total_cpus" not in sim_reqs
    assert "total_gpus" not in sim_reqs
    assert "total_mem_mb" not in sim_reqs

    # Verify per-sim calculations
    # CPUs per sim = n_mpi_procs × n_omp_threads = 2 × 4 = 8
    assert sim_reqs["n_cpus_per_sim"] == 8
    assert sim_reqs["n_nodes"] == 1
    assert sim_reqs["n_gpus"] == 0
    # mem_mb_per_sim = mem_gb_per_cpu × cpus_per_sim × 1000 = 2 × 8 × 1000 = 16000
    assert sim_reqs["mem_mb_per_sim"] == 16000


def test_sim_requirements_gpu_calculations(norfolk_analysis_with_gpus):
    """Verify GPU resource calculations."""
    analysis = norfolk_analysis_with_gpus

    resource_manager = analysis._resource_manager
    sim_reqs = resource_manager._get_simulation_resource_requirements()

    # Verify per-sim GPU requirements
    assert sim_reqs["n_gpus"] == 2

    # Verify memory calculation with different mem_gb_per_cpu
    # mem_mb_per_sim = 4 GB/CPU × 8 CPUs × 1000 = 32000 MB
    assert sim_reqs["mem_mb_per_sim"] == 32000

    # Should NOT have total_gpus
    assert "total_gpus" not in sim_reqs


def test_sim_requirements_multi_node_config():
    """Verify per-simulation requirements with multi-node configuration."""
    case = cases.Local_TestCases.retrieve_norfolk_multi_sim_test_case(
        start_from_scratch=True
    )
    analysis = case.analysis

    # Configure multi-node simulation
    analysis.cfg_analysis.n_nodes = 4
    analysis.cfg_analysis.n_mpi_procs = 8
    analysis.cfg_analysis.n_omp_threads = 2
    analysis.cfg_analysis.mem_gb_per_cpu = 1

    resource_manager = analysis._resource_manager
    sim_reqs = resource_manager._get_simulation_resource_requirements()

    # Verify calculations
    assert sim_reqs["n_nodes"] == 4
    assert sim_reqs["n_cpus_per_sim"] == 16  # 8 × 2
    assert sim_reqs["mem_mb_per_sim"] == 16000  # 1 × 16 × 1000


# Note: Sensitivity analysis tests would require setting up sub-analyses,
# which is more complex. The basic mechanism (finding MAX per-sim requirements)
# is tested implicitly through the workflow integration tests.
