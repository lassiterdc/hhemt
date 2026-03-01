"""
Unit tests for srun command construction and SLURM preflight checks.

These tests verify:
- srun command strings contain correct flags (--overlap, --cpu-bind=cores)
- GPU mode uses --gpus-per-task=1 (gpus alloc) or --ntasks-per-gpu=1 (gres alloc)
- CPU preflight raises RuntimeError on under-allocation
- GPU preflight raises RuntimeError on detectable under-allocation
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from TRITON_SWMM_toolkit.run_simulation import TRITONSWMM_run


def _make_run(
    run_mode: str,
    n_mpi_procs: int = 2,
    n_omp_threads: int = 4,
    n_gpus: int = 0,
    in_slurm: bool = False,
    gpu_alloc_mode: str = "gpus",
    multi_sim_run_method: str | None = None,
):
    """Build a minimal TRITONSWMM_run with mocked scenario and analysis.

    Args:
        multi_sim_run_method: Override the run method. If None, derived from
            in_slurm (True → "1_job_many_srun_tasks", False → "local").
            Use "batch_job" to test preflight checks that are skipped for
            1_job_many_srun_tasks mode.
    """
    if multi_sim_run_method is None:
        multi_sim_run_method = "1_job_many_srun_tasks" if in_slurm else "local"
    cfg = MagicMock()
    cfg.run_mode = run_mode
    cfg.n_mpi_procs = n_mpi_procs
    cfg.n_omp_threads = n_omp_threads
    cfg.n_gpus = n_gpus
    cfg.n_nodes = 1
    cfg.multi_sim_run_method = multi_sim_run_method
    cfg.hpc_additional_modules = []
    cfg.additional_modules_needed_to_run_TRITON_SWMM_on_hpc = []

    sys_cfg = MagicMock()
    sys_cfg.preferred_slurm_option_for_allocating_gpus = gpu_alloc_mode
    sys_cfg.additional_modules_needed_to_run_TRITON_SWMM_on_hpc = []

    system = MagicMock()
    system.cfg_system = sys_cfg

    analysis = MagicMock()
    analysis.cfg_analysis = cfg
    analysis.in_slurm = in_slurm

    scenario = MagicMock()
    scenario._analysis = analysis
    scenario._system = system
    scenario.model_run_completed.return_value = False
    scenario.scen_paths.sim_tritonswmm_executable = Path("/fake/TRITONSWMM")
    scenario.scen_paths.triton_swmm_cfg = Path("/fake/TRITONSWMM.cfg")

    run = TRITONSWMM_run.__new__(TRITONSWMM_run)
    run._scenario = scenario
    run._analysis = analysis
    return run


def _get_launch_cmd(run: TRITONSWMM_run) -> str:
    """Extract the launch command string from prepare_simulation_command."""
    # Patch out anything that touches the filesystem or logs
    with patch.object(run, "_analysis_level_model_logfile", return_value=Path("/fake/run.log")):
        with patch.object(
            run, "_retrieve_hotstart_file_for_incomplete_triton_or_tritonswmm_simulation", return_value=None
        ):
            result = run.prepare_simulation_command(pickup_where_leftoff=False, verbose=False)
    assert result is not None
    cmd, env, logfile, _ = result
    # The full command is passed to bash -lc; extract the srun portion
    return cmd[2]  # cmd = ["bash", "-lc", "<full_cmd_string>"]


# ---------------------------------------------------------------------------
# Command string correctness
# ---------------------------------------------------------------------------


def test_cpu_srun_includes_overlap():
    """CPU mode srun command must include --overlap for resource sharing in batch_job mode."""
    run = _make_run("mpi", in_slurm=True)
    full_cmd = _get_launch_cmd(run)
    assert "--overlap" in full_cmd


def test_cpu_srun_cpu_bind_cores():
    """CPU mode srun command must use --cpu-bind=cores."""
    run = _make_run("mpi", in_slurm=True)
    full_cmd = _get_launch_cmd(run)
    assert "--cpu-bind=cores" in full_cmd


def test_gpu_srun_includes_overlap():
    """GPU mode srun command must include --overlap for resource sharing in batch_job mode."""
    run = _make_run("gpu", n_gpus=2, in_slurm=True)
    full_cmd = _get_launch_cmd(run)
    assert "--overlap" in full_cmd


def test_gpu_srun_cpu_bind_cores():
    """GPU mode srun command must use --cpu-bind=cores."""
    run = _make_run("gpu", n_gpus=2, in_slurm=True)
    full_cmd = _get_launch_cmd(run)
    assert "--cpu-bind=cores" in full_cmd


def test_gpu_srun_gpus_mode_uses_gpus_per_task():
    """GPU mode with gpus allocation (Frontier) must include --gpus-per-task=1."""
    run = _make_run("gpu", n_gpus=2, in_slurm=True, gpu_alloc_mode="gpus")
    full_cmd = _get_launch_cmd(run)
    assert "--gpus-per-task=1" in full_cmd
    assert "--ntasks-per-gpu" not in full_cmd


def test_gpu_srun_gres_mode_uses_ntasks_per_gpu():
    """GPU mode with gres allocation (UVA) must include --ntasks-per-gpu=1."""
    run = _make_run("gpu", n_gpus=2, in_slurm=True, gpu_alloc_mode="gres")
    full_cmd = _get_launch_cmd(run)
    assert "--ntasks-per-gpu=1" in full_cmd
    assert "--gpus-per-task" not in full_cmd


def test_cpu_srun_gpu_flags_absent():
    """CPU mode srun command must NOT include any GPU binding flags."""
    run = _make_run("mpi", in_slurm=True)
    full_cmd = _get_launch_cmd(run)
    assert "--ntasks-per-gpu" not in full_cmd
    assert "--gpus-per-task" not in full_cmd


# ---------------------------------------------------------------------------
# CPU preflight
# ---------------------------------------------------------------------------


def test_cpu_preflight_raises_on_under_allocation():
    """RuntimeError when SLURM allocates fewer CPUs than configured."""
    run = _make_run("mpi", n_mpi_procs=4, n_omp_threads=4, in_slurm=True, multi_sim_run_method="batch_job")
    # Allocation provides 4 CPUs total, config requires 4×4=16
    slurm_env = {
        "SLURM_JOB_ID": "12345",
        "SLURM_NTASKS": "2",
        "SLURM_CPUS_PER_TASK": "2",  # 2×2=4, need 16
        "SLURM_CPUS_ON_NODE": "4",
    }
    with patch.dict(os.environ, slurm_env, clear=False):
        with pytest.raises(RuntimeError, match="SLURM"):
            with patch.object(run, "_analysis_level_model_logfile", return_value=Path("/fake/run.log")):
                with patch.object(
                    run, "_retrieve_hotstart_file_for_incomplete_triton_or_tritonswmm_simulation", return_value=None
                ):
                    run.prepare_simulation_command(pickup_where_leftoff=False, verbose=False)


def test_cpu_preflight_passes_on_sufficient_allocation():
    """No error when SLURM allocation exactly matches configuration."""
    run = _make_run("mpi", n_mpi_procs=2, n_omp_threads=4, in_slurm=True)
    # Allocation provides 8 CPUs, config requires 2×4=8
    slurm_env = {
        "SLURM_JOB_ID": "12345",
        "SLURM_NTASKS": "2",
        "SLURM_CPUS_PER_TASK": "4",
        "SLURM_CPUS_ON_NODE": "8",
    }
    with patch.dict(os.environ, slurm_env, clear=False):
        with patch.object(run, "_analysis_level_model_logfile", return_value=Path("/fake/run.log")):
            with patch.object(
                run, "_retrieve_hotstart_file_for_incomplete_triton_or_tritonswmm_simulation", return_value=None
            ):
                result = run.prepare_simulation_command(pickup_where_leftoff=False, verbose=False)
    assert result is not None


# ---------------------------------------------------------------------------
# GPU preflight
# ---------------------------------------------------------------------------


def test_gpu_preflight_raises_on_under_allocation():
    """RuntimeError when SLURM GPU allocation is less than configured n_gpus."""
    run = _make_run("gpu", n_gpus=4, in_slurm=True, multi_sim_run_method="batch_job")
    slurm_env = {
        "SLURM_JOB_ID": "12345",
        "SLURM_GPUS": "2",  # only 2 allocated, need 4
        "SLURM_NTASKS": "4",
        "SLURM_CPUS_PER_TASK": "4",
    }
    with patch.dict(os.environ, slurm_env, clear=False):
        with pytest.raises(RuntimeError, match="GPU"):
            with patch.object(run, "_analysis_level_model_logfile", return_value=Path("/fake/run.log")):
                with patch.object(
                    run, "_retrieve_hotstart_file_for_incomplete_triton_or_tritonswmm_simulation", return_value=None
                ):
                    run.prepare_simulation_command(pickup_where_leftoff=False, verbose=False)


def test_gpu_preflight_passes_on_sufficient_allocation():
    """No error when SLURM GPU allocation matches configuration."""
    run = _make_run("gpu", n_gpus=2, in_slurm=True)
    slurm_env = {
        "SLURM_JOB_ID": "12345",
        "SLURM_GPUS": "2",
        "SLURM_NTASKS": "2",
        "SLURM_CPUS_PER_TASK": "4",
    }
    with patch.dict(os.environ, slurm_env, clear=False):
        with patch.object(run, "_analysis_level_model_logfile", return_value=Path("/fake/run.log")):
            with patch.object(
                run, "_retrieve_hotstart_file_for_incomplete_triton_or_tritonswmm_simulation", return_value=None
            ):
                result = run.prepare_simulation_command(pickup_where_leftoff=False, verbose=False)
    assert result is not None


def test_gpu_preflight_skips_when_no_gpu_vars(capsys):
    """Preflight is skipped (not an error) when no SLURM GPU vars are detectable."""
    run = _make_run("gpu", n_gpus=4, in_slurm=True, multi_sim_run_method="batch_job")
    # No SLURM GPU vars — preflight should skip and print diagnostic
    slurm_env = {
        "SLURM_JOB_ID": "12345",
        "SLURM_NTASKS": "4",
        "SLURM_CPUS_PER_TASK": "4",
    }
    # Remove any GPU vars that might be in the real environment
    clean_env = {k: v for k, v in os.environ.items() if k not in ("SLURM_GPUS", "SLURM_GPUS_ON_NODE", "SLURM_JOB_GPUS")}
    clean_env.update(slurm_env)
    with patch.dict(os.environ, clean_env, clear=True):
        with patch.object(run, "_analysis_level_model_logfile", return_value=Path("/fake/run.log")):
            with patch.object(
                run, "_retrieve_hotstart_file_for_incomplete_triton_or_tritonswmm_simulation", return_value=None
            ):
                result = run.prepare_simulation_command(pickup_where_leftoff=False, verbose=False)
    assert result is not None
    assert "[GPU-PREFLIGHT]" in capsys.readouterr().out
