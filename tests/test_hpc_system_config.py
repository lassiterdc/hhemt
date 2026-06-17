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


@pytest.mark.parametrize(
    "retired_key, retired_val",
    [
        ("gpu_hardware", "a100"),
        ("gpu_compilation_backend", "CUDA"),
        ("preferred_slurm_option_for_allocating_gpus", "gres"),
        ("additional_modules_needed_to_run_TRITON_SWMM_on_hpc", "cuda/12.4"),
    ],
)
def test_system_config_shim_pops_and_warns_on_retired_hpc_keys(retired_key, retired_val):
    """Phase-4 (4c): the system_config pop-and-warn shim (folded into
    validate_toggle_dependencies) lets an un-migrated YAML carrying any of the four
    retired HPC keys LOAD with a DeprecationWarning instead of being rejected by
    extra="forbid". The retired key is dropped (the field no longer exists)."""
    from pathlib import Path

    import yaml as _yaml

    from TRITON_SWMM_toolkit.config.loaders import load_system_config

    # A known-good system config (the shipped template) + a retired HPC key.
    template = Path("test_data/norfolk_coastal_flooding/template_system_config.yaml")
    base = load_system_config(template)
    d = base.model_dump(mode="json")
    assert _yaml  # imported for symmetry with the YAML-load path
    d[retired_key] = retired_val

    cfg_type = type(base)
    with pytest.warns(DeprecationWarning, match=retired_key):
        cfg = cfg_type.model_validate(d)
    # Popped, not rejected, and the field genuinely no longer exists on the model.
    assert not hasattr(cfg, retired_key)
    assert retired_key not in cfg_type.model_fields


@pytest.mark.parametrize(
    "retired_key, retired_val",
    [
        ("hpc_account", "acct"),
        ("hpc_login_node", "login1.example.edu"),
        ("hpc_gpus_per_node", 8),
        ("hpc_cpus_per_node", 64),
        ("python_path", "/opt/conda/bin/python"),
        ("hpc_max_simultaneous_sims", 32),
    ],
)
def test_analysis_config_shim_pops_and_warns_on_retired_hpc_keys(retired_key, retired_val):
    """Phase-4 (4d): the analysis_config pop-and-warn shim (folded into
    check_consistency) lets an un-migrated YAML carrying any of the six retired HPC
    fields LOAD with a DeprecationWarning instead of being rejected by
    extra="forbid". The retired key is dropped (the field no longer exists); the two
    partition selectors are KEPT."""
    from pathlib import Path

    from TRITON_SWMM_toolkit.config.analysis import analysis_config
    from TRITON_SWMM_toolkit.config.loaders import load_analysis_config

    template = Path("test_data/norfolk_coastal_flooding/template_analysis_config.yaml")
    base = load_analysis_config(template)
    d = base.model_dump(mode="json")
    d[retired_key] = retired_val

    with pytest.warns(DeprecationWarning, match=retired_key):
        cfg = analysis_config.model_validate(d)
    assert not hasattr(cfg, retired_key)
    assert retired_key not in analysis_config.model_fields
    # The two partition SELECTORS are KEPT (D-A).
    assert "hpc_ensemble_partition" in analysis_config.model_fields
    assert "hpc_setup_and_analysis_processing_partition" in analysis_config.model_fields
