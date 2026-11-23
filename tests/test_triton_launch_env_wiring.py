"""The TRITON settings that are environment variables reach the launch (``run_simulation``), and a default adds nothing.

Laptop tier: ``prepare_simulation_command`` is driven on a stub run (mocked scenario and analysis, as the existing
launch-form tests do), and no solver is compiled or started. Each field is shown arriving in the env export string
at a non-default value, and absent at its default. The command line stays ``{exe} {cfg}``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hhemt.run_simulation import triton_launch_env
from tests.test_srun_command_construction import _make_run


def _launch(run_mode="serial", analysis=None, hpc=None):
    run = _make_run(run_mode, n_mpi_procs=2 if run_mode == "mpi" else 1, n_omp_threads=1, in_slurm=False)
    for key, value in {
        "TRITON_kokkos_print_configuration": False,
        "TRITON_kokkos_disable_warnings": False,
        "TRITON_kokkos_tune_internals": False,
        "TRITON_kokkos_tools_libs": None,
        "TRITON_kokkos_tools_args": None,
        "TRITON_extra_env": {},
        **(analysis or {}),
    }.items():
        setattr(run._analysis.cfg_analysis, key, value)
    for key, value in {
        "gpu_aware_mpi": False,
        "mpich_ofi_nic_policy": None,
        "hsa_xnack": None,
        "cuda_managed_force_device_alloc": None,
        "pmix_gds": None,
        **(hpc or {}),
    }.items():
        setattr(run._analysis.cfg_hpc_system, key, value)
    with patch.object(run, "_analysis_level_model_logfile", return_value=Path("/fake/run.log")):
        with patch.object(
            run, "_retrieve_hotstart_file_for_incomplete_triton_or_tritonswmm_simulation", return_value=None
        ):
            cmd, env, _logfile, _ = run.prepare_simulation_command(pickup_where_leftoff=False, verbose=False)
    return cmd[2], env


@pytest.mark.parametrize(
    ("analysis", "exported"),
    [
        ({"TRITON_kokkos_print_configuration": True}, ("KOKKOS_PRINT_CONFIGURATION", "1")),
        ({"TRITON_kokkos_disable_warnings": True}, ("KOKKOS_DISABLE_WARNINGS", "1")),
        ({"TRITON_kokkos_tune_internals": True}, ("KOKKOS_TUNE_INTERNALS", "1")),
        ({"TRITON_kokkos_tools_libs": "/t/lib.so"}, ("KOKKOS_TOOLS_LIBS", "/t/lib.so")),
        ({"TRITON_kokkos_tools_args": "--x"}, ("KOKKOS_TOOLS_ARGS", "--x")),
        ({"TRITON_extra_env": {"MY_VAR": "7"}}, ("MY_VAR", "7")),
    ],
)
def test_an_analysis_environment_field_arrives_in_the_launch_environment(analysis, exported):
    _cmd, env = _launch(analysis=analysis)
    name, value = exported
    assert env[name] == value


@pytest.mark.parametrize(
    ("hpc", "exported"),
    [
        ({"hsa_xnack": True}, ("HSA_XNACK", "1")),
        ({"hsa_xnack": False}, ("HSA_XNACK", "0")),
        ({"cuda_managed_force_device_alloc": True}, ("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1")),
        ({"mpich_ofi_nic_policy": "NUMA"}, ("MPICH_OFI_NIC_POLICY", "NUMA")),
        ({"pmix_gds": "hash"}, ("PMIX_MCA_gds", "hash")),
    ],
)
def test_an_hpc_environment_field_arrives_in_the_launch_environment(hpc, exported):
    _cmd, env = _launch("mpi", hpc=hpc)
    name, value = exported
    assert env[name] == value


def test_every_default_adds_no_variable_and_the_nic_policy_keeps_todays_value():
    _cmd, env = _launch("serial")
    owned = {k for k in env if k.startswith(("KOKKOS_", "HSA_", "CUDA_MANAGED", "MPICH_", "PMIX_"))}
    assert owned == set()
    _cmd, env = _launch("mpi")
    assert env["MPICH_OFI_NIC_POLICY"] == "BLOCK"


def test_the_command_line_stays_the_executable_and_the_cfg():
    cmd, _env = _launch("serial", analysis={"TRITON_kokkos_print_configuration": True})
    assert "KOKKOS_PRINT_CONFIGURATION" in cmd
    assert cmd.rstrip().endswith("TRITONSWMM.cfg"), cmd[-120:]


_FULL_ANALYSIS = SimpleNamespace(
    TRITON_courant_number=0.5,
    TRITON_hextra_m=0.001,
    TRITON_fixed_timestep=False,
    TRITON_domain_decomposition="static",
    TRITON_domain_decomposition_check_factor=None,
    TRITON_progress_print_iterations=10000,
    TRITON_kokkos_print_configuration=False,
    TRITON_kokkos_disable_warnings=False,
    TRITON_kokkos_tune_internals=False,
    TRITON_kokkos_tools_libs=None,
    TRITON_kokkos_tools_args=None,
    TRITON_extra_env={},
)


def _full_hpc(**over):
    return SimpleNamespace(
        **{
            "gpu_aware_mpi": False,
            "mpich_ofi_nic_policy": None,
            "hsa_xnack": None,
            "cuda_managed_force_device_alloc": None,
            "pmix_gds": None,
            **over,
        }
    )


def test_a_gpu_aware_mpi_sets_the_gpu_direct_key_and_on_a_cray_platform_the_support_variable():
    hpc = _full_hpc()
    hpc.gpu_aware_mpi = True
    env, settings = triton_launch_env(_FULL_ANALYSIS, hpc, "gpu", cray=True)
    assert env["MPICH_GPU_SUPPORT_ENABLED"] == "1" and settings["GPU_DIRECT"] == 1
    env, _settings = triton_launch_env(_FULL_ANALYSIS, hpc, "gpu", cray=False)
    assert "MPICH_GPU_SUPPORT_ENABLED" not in env
    env, settings = triton_launch_env(_FULL_ANALYSIS, _full_hpc(), "gpu", cray=True)
    assert "MPICH_GPU_SUPPORT_ENABLED" not in env and settings["GPU_DIRECT"] == 0


@pytest.mark.parametrize("run_mode", ["serial", "openmp", "mpi", "hybrid", "gpu"])
@pytest.mark.parametrize("cray", [False, True])
def test_a_run_with_no_hpc_system_config_gets_the_launch_of_a_default_one(run_mode, cray):
    """A local run has no HPC system config; the launch is that of an HPC config whose four fields are defaults.

    Input class: ``cfg_hpc_system`` None, for every run mode and both platform kinds, against a default config. The
    control, a GPU-aware MPI on a Cray build, must differ from the None result, so the equality is not one that
    holds for any value of the argument.
    """
    default = triton_launch_env(_FULL_ANALYSIS, _full_hpc(), run_mode, cray=cray)
    assert triton_launch_env(_FULL_ANALYSIS, None, run_mode, cray=cray) == default
    assert "MPICH_GPU_SUPPORT_ENABLED" not in default[0] and default[1]["GPU_DIRECT"] == 0
    aware = _full_hpc()
    aware.gpu_aware_mpi = True
    assert triton_launch_env(_FULL_ANALYSIS, aware, run_mode, cray=True) != triton_launch_env(
        _FULL_ANALYSIS, None, run_mode, cray=True
    )


def test_a_gpu_aware_mpi_on_a_cpu_run_is_refused_when_the_configs_are_validated():
    from hhemt.validation import validate_analysis_config

    analysis = SimpleNamespace(**{**vars(_FULL_ANALYSIS), "run_mode": "serial"})
    with (
        patch("hhemt.validation._validate_weather_data"),
        patch("hhemt.validation._validate_run_mode_consistency"),
        patch("hhemt.validation._validate_toggle_dependencies_analysis"),
        patch("hhemt.validation._validate_hpc_configuration"),
    ):
        refused = validate_analysis_config(analysis, SimpleNamespace(gpu_aware_mpi=True))
        admitted = validate_analysis_config(analysis, SimpleNamespace(gpu_aware_mpi=False))
        on_gpu = validate_analysis_config(
            SimpleNamespace(**{**vars(analysis), "run_mode": "gpu"}), SimpleNamespace(gpu_aware_mpi=True)
        )
    assert [e.field for e in refused.errors] == ["hpc_system.gpu_aware_mpi"]
    assert "run_mode" in refused.errors[0].message
    assert not admitted.errors and not on_gpu.errors


def test_the_resolved_settings_and_the_extra_environment_are_recorded_in_the_model_log():
    run = _make_run("serial", n_mpi_procs=1, n_omp_threads=1, in_slurm=False)
    analysis = run._analysis.cfg_analysis
    for key, value in {
        "TRITON_courant_number": 0.45,
        "TRITON_hextra_m": 0.002,
        "TRITON_fixed_timestep": False,
        "TRITON_domain_decomposition": "static",
        "TRITON_domain_decomposition_check_factor": None,
        "TRITON_progress_print_iterations": 10000,
        "TRITON_kokkos_print_configuration": True,
        "TRITON_extra_env": {"MY_VAR": "7"},
    }.items():
        setattr(analysis, key, value)
    recorded: list = []
    run._scenario.get_log.return_value.triton_launch_settings.set.side_effect = recorded.append
    with patch.object(run, "_analysis_level_model_logfile", return_value=Path("/fake/run.log")):
        with patch.object(
            run, "_retrieve_hotstart_file_for_incomplete_triton_or_tritonswmm_simulation", return_value=None
        ):
            run.prepare_simulation_command(pickup_where_leftoff=False, verbose=False)
    (entry,) = recorded
    assert entry["cfg"]["COURANT"] == 0.45 and entry["cfg"]["HEXTRA"] == 0.002
    assert entry["env"]["KOKKOS_PRINT_CONFIGURATION"] == "1" and entry["env"]["MY_VAR"] == "7"


def test_pmix_gds_set_exports_the_hash_store_and_unset_exports_nothing():
    """Input class: the field set to its one value, unset, and absent (no HPC config); only the set case exports."""
    env, _settings = triton_launch_env(_FULL_ANALYSIS, _full_hpc(pmix_gds=None), "mpi")
    assert "PMIX_MCA_gds" not in env
    env, _settings = triton_launch_env(_FULL_ANALYSIS, None, "mpi")
    assert "PMIX_MCA_gds" not in env
    env, _settings = triton_launch_env(_FULL_ANALYSIS, _full_hpc(pmix_gds="hash"), "mpi")
    assert env["PMIX_MCA_gds"] == "hash"


def _profile_yaml(tmp_path, **extra):
    import yaml

    body = {"hpc_name": "t", "partitions": {"standard": {"max_runtime": 60}}, **extra}
    path = tmp_path / "hpc_profile_for_message_test.yaml"
    path.write_text(yaml.safe_dump(body))
    return path


def test_the_pmix_gds_hash_value_loads_from_a_profile_file(tmp_path):
    from hhemt.config.loaders import load_hpc_system_config

    assert load_hpc_system_config(_profile_yaml(tmp_path, pmix_gds="hash")).pmix_gds == "hash"
    assert load_hpc_system_config(_profile_yaml(tmp_path)).pmix_gds is None


def test_a_pmix_gds_value_other_than_hash_is_refused_with_a_message_naming_the_file_the_field_and_the_fix(tmp_path):
    """The refusal names the profile file, the field, the one allowed value and the failure it fixes (CS-1 ruling)."""
    from hhemt.config.loaders import load_hpc_system_config
    from hhemt.exceptions import ConfigurationError

    path = _profile_yaml(tmp_path, pmix_gds="shmem2")
    with pytest.raises(ConfigurationError) as raised:
        load_hpc_system_config(path)
    text = str(raised.value)
    for needle in (path.name, "pmix_gds", "hash", "gds_shmem2", "shmem2", "pmix_gds: hash"):
        assert needle in text, (needle, text)


def test_pmix_mca_gds_in_the_extra_environment_is_refused_with_a_message_naming_the_key_and_the_fix(tmp_path):
    """The refusal names the key, the field it sits in and the literal profile edit that replaces it (CS-1 ruling)."""
    from pydantic import ValidationError

    from tests.test_triton_cfg_render import _analysis

    with pytest.raises(ValidationError) as raised:
        _analysis(tmp_path, TRITON_extra_env={"PMIX_MCA_gds": "hash"})
    text = str(raised.value)
    for needle in ("PMIX_MCA_gds", "TRITON_extra_env", "pmix_gds: hash"):
        assert needle in text, (needle, text)
