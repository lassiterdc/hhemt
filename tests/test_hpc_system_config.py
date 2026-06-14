"""Tests for the per-HPC-system configuration model (config/hpc_system.py).

Phase 1 of hpc-system-profile-config. Covers:
  (a) a valid UVA-shaped config loads;
  (b) an unknown top-level key raises (extra="forbid");
  (c) the executor_profile_extras reject-guard raises ConfigurationError on a
      set-resources / default-resources clobber of a toolkit-emitted resource;
  (d) the preflight max_runtime bound helper rejects a requested runtime that
      exceeds the partition cap.
"""

import pytest
from pydantic import ValidationError

from TRITON_SWMM_toolkit.config.hpc_system import (
    PartitionSpec,
    hpc_system_config,
)
from TRITON_SWMM_toolkit.config.loaders import load_hpc_system_config
from TRITON_SWMM_toolkit.exceptions import ConfigurationError


def _uva_shaped_dict() -> dict:
    """A representative UVA-shaped config dict."""
    return {
        "system_name": "uva-rivanna",
        "default_account": "{your-allocation}",
        "login_node": "login1.hpc.virginia.edu",
        "default_execution_mode": "batch_job",
        "gpu_allocation_flavor": "gres",
        "additional_modules": ["apptainer", "gcc/11.4.0"],
        "partitions": {
            "gpu": {
                "max_runtime": 4320,
                "max_mem_mb": 384000,
                "max_cpus_per_task": 64,
                "max_gpu": 4,
                "available_gpu_models": ["a6000", "a100", "v100"],
                "gpus_per_node": 4,
                "cpus_per_node": 64,
                "gpu_hardware": "a6000",
                "gpu_compilation_backend": "CUDA",
            },
            "standard": {
                "max_runtime": 4320,
                "max_cpus_per_task": 40,
                "supports_mpi": True,
            },
        },
    }


# (a) ----------------------------------------------------------------------
def test_valid_uva_shaped_config_loads():
    cfg = hpc_system_config.model_validate(_uva_shaped_dict())
    assert cfg.system_name == "uva-rivanna"
    assert cfg.gpu_allocation_flavor == "gres"
    assert set(cfg.partitions) == {"gpu", "standard"}
    gpu = cfg.partitions["gpu"]
    assert isinstance(gpu, PartitionSpec)
    assert gpu.gpu_hardware == "a6000"
    assert gpu.gpu_compilation_backend == "CUDA"
    assert gpu.max_runtime == 4320
    # executor_profile_extras defaults to an empty dict (per-instance, not shared)
    assert cfg.executor_profile_extras == {}


def test_valid_config_loads_from_yaml(tmp_path):
    import yaml

    p = tmp_path / "hpc_system_config.yaml"
    p.write_text(yaml.safe_dump(_uva_shaped_dict()))
    cfg = load_hpc_system_config(p)
    assert cfg.system_name == "uva-rivanna"
    assert cfg.partitions["gpu"].max_gpu == 4


# (b) ----------------------------------------------------------------------
def test_unknown_top_level_key_raises():
    bad = _uva_shaped_dict()
    bad["bogus_key"] = 123
    with pytest.raises(ValidationError):
        hpc_system_config.model_validate(bad)


def test_unknown_partition_key_raises():
    bad = _uva_shaped_dict()
    bad["partitions"]["gpu"]["bogus_partition_field"] = 1
    with pytest.raises(ValidationError):
        hpc_system_config.model_validate(bad)


# (c) ----------------------------------------------------------------------
def test_reject_guard_set_resources_clobber():
    bad = _uva_shaped_dict()
    bad["executor_profile_extras"] = {"set-resources": {"run_triton": {"tasks": 4}}}
    with pytest.raises(ConfigurationError, match="tasks"):
        hpc_system_config.model_validate(bad)


def test_reject_guard_default_resources_clobber():
    bad = _uva_shaped_dict()
    bad["executor_profile_extras"] = {"default-resources": ["mem_mb=8000"]}
    with pytest.raises(ConfigurationError, match="mem_mb"):
        hpc_system_config.model_validate(bad)


def test_reject_guard_allows_non_toolkit_extras():
    ok = _uva_shaped_dict()
    ok["executor_profile_extras"] = {
        "slurm-init-seconds-before-status-checks": 40,
        "set-resources": {"run_triton": {"slurm_account": "{your-allocation}"}},
    }
    # slurm_account is NOT in the toolkit-emitted-resource guard set, and a
    # plugin-level key is unrelated — both pass the guard.
    cfg = hpc_system_config.model_validate(ok)
    assert cfg.executor_profile_extras["slurm-init-seconds-before-status-checks"] == 40


# (d) ----------------------------------------------------------------------
def test_check_runtime_within_cap_rejects_over_cap():
    cfg = hpc_system_config.model_validate(_uva_shaped_dict())
    with pytest.raises(ConfigurationError, match="exceeds"):
        cfg.check_runtime_within_cap("gpu", 5000)


def test_check_runtime_within_cap_allows_under_cap():
    cfg = hpc_system_config.model_validate(_uva_shaped_dict())
    # at or under the 4320-min cap is fine (no raise)
    cfg.check_runtime_within_cap("gpu", 4320)
    cfg.check_runtime_within_cap("gpu", 100)


def test_check_runtime_within_cap_unknown_partition_raises():
    cfg = hpc_system_config.model_validate(_uva_shaped_dict())
    with pytest.raises(ConfigurationError, match="Unknown partition"):
        cfg.check_runtime_within_cap("nonexistent", 10)


def test_check_runtime_no_cap_partition_imposes_no_limit():
    d = _uva_shaped_dict()
    d["partitions"]["uncapped"] = {"max_cpus_per_task": 8}  # no max_runtime
    cfg = hpc_system_config.model_validate(d)
    # no max_runtime declared -> no cap enforced
    cfg.check_runtime_within_cap("uncapped", 999999)
