from enum import Enum
from TRITON_SWMM_toolkit.platform_configs import PlatformConfig
import os
from pathlib import Path

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
    additional_modules="miniforge gcc/11.4.0 openmpi/4.1.4 cuda/12.2.2",
    gpu_compilation_backend="CUDA",
    gpu_hardware="a6000",  # a100
    hpc_gpus_per_node=8,
    example_data_dir=Path("/scratch")
    / os.getenv("USER", "unknown")
    / "triton_swmm_toolkit_data",
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
FRONTIER_GLOBUS_COLLECTION_UUID = None  # TODO: confirm at app.globus.org > Collections > search "OLCF DTN"
FRONTIER_GLOBUS_SCRATCH_BASE = "/lustre/orion/***REMOVED***/scratch/{username}"
FRONTIER_GLOBUS_PROJECT_BASE = "/lustre/orion/***REMOVED***/proj-shared"

DESKTOP_GLOBUS_COLLECTION_NAME = "Desktop"
DESKTOP_GLOBUS_COLLECTION_UUID = "***REMOVED***"
