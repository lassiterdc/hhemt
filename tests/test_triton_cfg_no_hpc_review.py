"""Review tests: a run with no HPC system config behaves as one whose HPC config holds its defaults.

Laptop tier. Two properties the coder's own tests do not hold:

1. The None branch of ``triton_launch_env`` and ``triton_cfg_settings`` agrees with the REAL field defaults of
   ``hpc_system_config`` (the coder's comparisons use a hand-written ``SimpleNamespace`` of the four defaults, so a
   default changed in the model would leave both sides green while the None branch drifted).
2. The call site that reached the AttributeError in production, ``prepare_simulation_command`` with
   ``cfg_hpc_system`` None, completes and exports the same launch environment as a default HPC config.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from hhemt.config.hpc_system import hpc_system_config
from hhemt.run_simulation import triton_launch_env
from hhemt.triton_cfg import triton_cfg_settings
from tests.test_srun_command_construction import _make_run
from tests.test_triton_launch_env_wiring import _FULL_ANALYSIS


def _real_default_hpc():
    """An ``hpc_system_config`` carrying every field default, built without validation of the required fields."""
    return hpc_system_config.model_construct()


@pytest.mark.parametrize("run_mode", ["serial", "openmp", "mpi", "hybrid", "gpu"])
@pytest.mark.parametrize("cray", [False, True])
def test_no_hpc_config_equals_the_real_model_defaults_for_the_launch(run_mode, cray):
    """Input class: ``cfg_hpc_system`` None against the real ``hpc_system_config`` field defaults, every run mode.

    Why: the None branch hard-codes the four defaults; it must equal what the model gives, or a local run launches
    differently from a default HPC config. Kill: changing a None-branch literal (``nic`` to ``"NUMA"``, ``hsa_xnack``
    to ``True``) reds it. Second correct implementation: reading the defaults from ``hpc_system_config.model_fields``.
    Control: a real config with ``gpu_aware_mpi`` True must differ on a Cray build.
    """
    default = triton_launch_env(_FULL_ANALYSIS, _real_default_hpc(), run_mode, cray=cray)
    assert triton_launch_env(_FULL_ANALYSIS, None, run_mode, cray=cray) == default
    aware = hpc_system_config.model_construct(gpu_aware_mpi=True)
    assert triton_launch_env(_FULL_ANALYSIS, aware, run_mode, cray=True) != triton_launch_env(
        _FULL_ANALYSIS, None, run_mode, cray=True
    )


def test_no_hpc_config_equals_the_real_model_defaults_for_the_cfg_settings():
    """Input class: ``cfg_hpc_system`` None against the real field defaults.

    Kill: a None branch that sets GPU_DIRECT 1.
    """
    assert triton_cfg_settings(_FULL_ANALYSIS, None) == triton_cfg_settings(_FULL_ANALYSIS, _real_default_hpc())
    aware = hpc_system_config.model_construct(gpu_aware_mpi=True)
    assert triton_cfg_settings(_FULL_ANALYSIS, aware)["GPU_DIRECT"] == 1
    assert triton_cfg_settings(_FULL_ANALYSIS, None)["GPU_DIRECT"] == 0


@pytest.mark.parametrize("run_mode", ["serial", "mpi"])
def test_prepare_simulation_command_with_no_hpc_config_exports_the_default_launch_environment(run_mode):
    """Input class: the production call site (``prepare_simulation_command``) with ``cfg_hpc_system`` None.

    Pre-fix this raised AttributeError ('NoneType' object has no attribute 'gpu_aware_mpi'). The assertion turns on
    the exported TRITON-owned variables being those of a default config, a property true before and after the change.
    """
    envs = {}
    for label, hpc in (("none", None), ("default", _real_default_hpc())):
        run = _make_run(run_mode, n_mpi_procs=2 if run_mode == "mpi" else 1, n_omp_threads=1, in_slurm=False)
        for key, value in vars(_FULL_ANALYSIS).items():
            setattr(run._analysis.cfg_analysis, key, value)
        run._analysis.cfg_hpc_system = hpc
        with patch.object(run, "_analysis_level_model_logfile", return_value=Path("/fake/run.log")):
            with patch.object(
                run, "_retrieve_hotstart_file_for_incomplete_triton_or_tritonswmm_simulation", return_value=None
            ):
                cmd, env, _logfile, _ = run.prepare_simulation_command(pickup_where_leftoff=False, verbose=False)
        envs[label] = {
            k: env[k] for k in ("MPICH_OFI_NIC_POLICY", "HSA_XNACK", "CUDA_MANAGED_FORCE_DEVICE_ALLOC") if k in env
        }
        envs[label]["cmd_tail"] = cmd[2][-40:]
    assert envs["none"] == envs["default"]
    if run_mode == "mpi":
        assert envs["none"]["MPICH_OFI_NIC_POLICY"] == "BLOCK"
