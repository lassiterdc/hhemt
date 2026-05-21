import os
from pathlib import Path

from TRITON_SWMM_toolkit.platform_configs import PlatformConfig

# Re-exports from version_migration.constants so the CI Check A
# (Phase 4 scripts/check_layout_version.py) can grep without importing.
from TRITON_SWMM_toolkit.version_migration.constants import (  # noqa: F401
    LAYOUT_VERSION,
    MINIMUM_SUPPORTED_VERSION,
)

APP_NAME = "TRITON_SWMM_toolkit"
NORFOLK_EX = "norfolk_coastal_flooding"
NORFOLK_ANALYSIS_CONFIG = "template_analysis_config.yaml"
NORFOLK_SYSTEM_CONFIG = "template_system_config.yaml"
NORFOLK_CASE_CONFIG = "case.yaml"

# POST PROCESSING

LST_COL_HEADERS_NODE_FLOOD_SUMMARY = [
    "node_id",
    "hours_flooded",
    "max_flow_cms",
    "time_of_max_flood_d_hr_mn",
    "tot_flooded_vol_10e6_ltr",
    "max_ponded_depth_m",
]
LST_COL_HEADERS_NODE_FLOW_SUMMARY = [
    "node_id",
    "type",
    "max_lateral_inflow_cms",
    "max_total_inflow_cms",
    "time_of_max_flow_d_hr_mn",
    "lateral_inflow_vol_10e6_ltr",
    "total_inflow_vol_10e6_ltr",
    "flow_balance_error_percent",
]
LST_COL_HEADERS_LINK_FLOW_SUMMARY = [
    "link_id",
    "type",
    "max_flow_cms",
    "time_of_max_flow_d_hr_mn",
    "max_velocity_mps",
    "max_over_full_flow",
    "max_over_full_depth",
]

TEST_SYSTEM_DIRNAME = "tests"
CASE_SYSTEM_DIRNAME = "cases"
TEST_N_REPORTING_TSTEPS_PER_SIM = 12
TEST_TRITON_REPORTING_TIMESTEP_S = 10

# Platform presets
FRONTIER_DEFAULT_PLATFORM_CONFIG = PlatformConfig(
    name="frontier",
    hpc_ensemble_partition="batch",  # or batch or extended
    hpc_setup_and_analysis_processing_partition="batch",
    hpc_account="***REMOVED***",  # "***REMOVED***",
    multi_sim_run_method="1_job_many_srun_tasks",
    additional_modules="PrgEnv-amd Core/24.07 craype-accel-amd-gfx90a miniforge3/23.11.0-0 libfabric/1.22.0",
    gpu_compilation_backend="HIP",
    hpc_gpus_per_node=8,
    hpc_cpus_per_node=64,
    toggle_triton_model=False,
    toggle_tritonswmm_model=True,
    toggle_swmm_model=False,
    target_processed_output_type="zarr",  # nc
    preferred_slurm_option_for_allocating_gpus="gpus",  # gres
)

UVA_DEFAULT_PLATFORM_CONFIG = PlatformConfig(
    name="uva",
    hpc_ensemble_partition="standard",
    hpc_setup_and_analysis_processing_partition="standard",
    hpc_account="***REMOVED***",
    multi_sim_run_method="batch_job",
    additional_modules="miniforge gcc/12.4.0 openmpi/4.1.4 cuda/12.2.2",
    gpu_compilation_backend="CUDA",
    gpu_hardware="a6000",  # a100
    hpc_gpus_per_node=8,
    example_data_dir=Path("/scratch") / os.getenv("USER", "unknown") / "triton_swmm_toolkit_data",
    toggle_triton_model=False,
    toggle_tritonswmm_model=True,
    toggle_swmm_model=False,
    hpc_max_simultaneous_sims=1000,
    hpc_total_job_duration_min=60 * 8,
    preferred_slurm_option_for_allocating_gpus="gres",  #  gpus
    target_processed_output_type="zarr",  # nc
    hpc_login_node="login1.hpc.virginia.edu",
)

# Globus endpoint identifiers
# UUIDs are stable public identifiers for Globus collections.
# Find them at app.globus.org > Collections > search by name.
# Per-user paths (usernames, experiment dirs) belong in configs/transfers/ YAML,
# not here — see configs/transfers/template_transfer.yaml.
UVA_GLOBUS_COLLECTION_NAME = "UVA Standard Security Storage"
UVA_GLOBUS_COLLECTION_UUID = "af187d15-768f-4449-8670-d00e1eb1ce6a"
UVA_GLOBUS_SCRATCH_BASE = "/scratch/{username}"  # expand with os.getenv("USER")

FRONTIER_GLOBUS_COLLECTION_NAME = "OLCF DTN (Globus 5)"
FRONTIER_GLOBUS_COLLECTION_UUID = "36d521b3-c182-4071-b7d5-91db5d380d42"
FRONTIER_GLOBUS_SCRATCH_BASE = "/lustre/orion/***REMOVED***/scratch/{username}"
FRONTIER_GLOBUS_PROJECT_BASE = "/lustre/orion/***REMOVED***/proj-shared"

DESKTOP_GLOBUS_COLLECTION_NAME = "Desktop"
DESKTOP_GLOBUS_COLLECTION_UUID = "***REMOVED***"

LAPTOP_GLOBUS_COLLECTION_NAME = "personal_laptop"
LAPTOP_GLOBUS_COLLECTION_UUID = "***REMOVED***"

# System-name-to-endpoint mapping for PostRunTransferConfig.
# Keys are system names matching PlatformConfig.name values.
# Values are (source_uuid, scratch_base, needs_data_access, session_domain) tuples.
# needs_data_access: whether the endpoint requires a data_access dependent
#   scope at auth time.  UVA (Globus 5 mapped collection) does; OLCF DTN does not.
# session_domain: identity domain required by the endpoint's access policy.
#   OLCF requires sso.ccs.ornl.gov; UVA has no domain restriction (None).
GLOBUS_SYSTEM_ENDPOINTS: dict[str, tuple[str, str, bool, str | None]] = {
    "uva": (UVA_GLOBUS_COLLECTION_UUID, UVA_GLOBUS_SCRATCH_BASE, True, None),
    "frontier": (
        FRONTIER_GLOBUS_COLLECTION_UUID,
        FRONTIER_GLOBUS_SCRATCH_BASE,
        False,
        "sso.ccs.ornl.gov",
    ),
}


# ============================================================================
# Status-flag builders — single source of truth for `_status/*.flag` paths.
#
# Phase 1 scope (Option C of the sensitivity-master reprocess gap plan,
# 2026-05-21): centralize ONLY the patterns referenced by the new code in
# `workflow.py::generate_reprocess_master_snakefile_content` and the lines
# being rewritten there. The other ~58 hardcoded flag-name occurrences
# elsewhere in the codebase are out of scope for this plan; folding them in
# is tracked as a follow-up. Adopt-the-principle, don't expand-the-patch.
#
# Naming convention: descriptive function names communicate purpose at the
# call site (the on-disk `c_run_*` / `d_process_*` / `e_consolidate_*`
# string outputs are unchanged — those are the persistent contract).
# Per the accepted-decision wildcard-charset stipulation, sa_id and
# event_id must match `^[A-Za-z0-9_.]+$`; callers are responsible for
# validating at CSV/config load time. The `_validate_id_fragment` helper
# below provides a runtime fast-fail at the builder call site for
# path-fragment-unsafe values (forbidden: `/`, `\`, `.flag`, whitespace).
# ============================================================================

STATUS_DIR_NAME: str = "_status"


def _validate_id_fragment(name: str, value: str) -> None:
    """Reject path-fragment-unsafe values in flag-name builder inputs.

    sa_id / event_id end up baked into Snakemake rule names and on-disk
    flag file paths. Forbidden characters can corrupt rule names or paths.
    """
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string; got {value!r}")
    for ch in ("/", "\\", ".flag", " ", "\t", "\n"):
        if ch in value:
            raise ValueError(f"{name}={value!r} contains forbidden fragment {ch!r}")


def sim_run_flag_per_sa(model_type: str, sa_id: str, event_id: str) -> str:
    """Per-sa per-event simulation completion flag (sensitivity workflow).

    sa_id and event_id must match `^[A-Za-z0-9_.]+$` per the
    accepted-decision wildcard-charset stipulation; this builder fast-fails
    on path-fragment-unsafe inputs via `_validate_id_fragment`.
    """
    _validate_id_fragment("sa_id", sa_id)
    _validate_id_fragment("event_id", event_id)
    return f"{STATUS_DIR_NAME}/c_run_{model_type}_sa-{sa_id}_evt-{event_id}_complete.flag"


def process_timeseries_flag_per_sa(model_type: str, sa_id: str, event_id: str) -> str:
    """Per-sa per-event process_timeseries completion flag (sensitivity workflow).

    Same wildcard-charset contract as `sim_run_flag_per_sa`.
    """
    _validate_id_fragment("sa_id", sa_id)
    _validate_id_fragment("event_id", event_id)
    return f"{STATUS_DIR_NAME}/d_process_{model_type}_sa-{sa_id}_evt-{event_id}_complete.flag"


def consolidate_subanalysis_flag(sa_id: str) -> str:
    """Per-sa consolidate completion flag (sensitivity workflow)."""
    _validate_id_fragment("sa_id", sa_id)
    return f"{STATUS_DIR_NAME}/e_consolidate_sa-{sa_id}_complete.flag"


def consolidate_master_flag() -> str:
    """Master consolidate completion flag (sensitivity workflow)."""
    return f"{STATUS_DIR_NAME}/f_consolidate_master_complete.flag"


def sa_inputs_fingerprint_flag(sa_id: str) -> str:
    """Per-sa input fingerprint file used as an mtime-trigger sentinel for per-sa rules."""
    _validate_id_fragment("sa_id", sa_id)
    return f"{STATUS_DIR_NAME}/sa-{sa_id}_inputs.json"
