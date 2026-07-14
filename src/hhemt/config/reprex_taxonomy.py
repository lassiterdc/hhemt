"""ADR-10 config Path-field -> reprex taxonomy classifier (reproducibility-system C12).

`field_bucket(field_name)` returns the ADR-10 USER / HPC / EXPERIMENT bucket for a
config **Path** field, derived from its `PathPolicy` in the bundle
`_PATH_FIELD_POLICY` table. The classifier is *provably total over the config
Path-field domain* via two composed totalities:
  (1) config Path field -> PathPolicy, enforced by
      tests/test_bundle.py::test_all_path_fields_have_policy (bidirectional
      set-equality between _PATH_FIELD_POLICY keys and
      enumerate_path_fields(system_config) | enumerate_path_fields(analysis_config));
  (2) PathPolicy -> bucket, enforced by
      tests/test_reprex_taxonomy.py::test_policy_to_bucket_is_total
      (set(_POLICY_TO_BUCKET) == set(PathPolicy)).

Domain is config Path fields ONLY. A field_name outside _PATH_FIELD_POLICY raises
KeyError -- the full non-path taxonomy (toggles, case_name, gpu_hardware, ...) is
deferred to bundle-reprex-roundtrip C8. The "hpc" return value is part of the
shared ADR-10 vocabulary but is UNREACHABLE over the Path-field domain: no config
Path field is HPC-bucketed, because HPC identity lives entirely in
hpc_system_config (partition / account / gpu_hardware), which carries zero Path
fields.

No I/O, no runtime state, no Pydantic introspection at classify time -- the
classifier reads only the static policy table (live introspection via
enumerate_path_fields is the exhaustiveness test's job, not the classify path's).

Package-scope import invariant: this module does ``from hhemt.bundle._path_policy
import ...``, which executes ``hhemt.bundle.__init__`` (NOT stdlib-only -- it eagerly
imports ``_emit`` et al.). Acyclicity therefore rests on the invariant that NO module
reachable from ``hhemt.bundle.__init__`` imports ``hhemt.config.reprex_taxonomy``.
Adding a top-level ``from hhemt.config.reprex_taxonomy import field_bucket`` to any
``hhemt.bundle`` module (e.g. an emit-time bucketing call in ``_emit.py``) WOULD form
an import cycle -- route any such call through a function-body local import instead.
"""

from __future__ import annotations

from typing import Literal

from hhemt.bundle._path_policy import _PATH_FIELD_POLICY, PathPolicy

Bucket = Literal["user", "hpc", "experiment"]

# PathPolicy -> ADR-10 bucket. Total over all 6 PathPolicy values (enforced by
# tests/test_reprex_taxonomy.py::test_policy_to_bucket_is_total). No policy maps
# to "hpc": HPC identity lives in hpc_system_config (non-path), so "hpc" is
# unreachable over this domain.
_POLICY_TO_BUCKET: dict[PathPolicy, Bucket] = {
    PathPolicy.FORCED_DOT: "experiment",
    PathPolicy.BUNDLE_RELATIVE: "experiment",
    PathPolicy.BUNDLE_RELATIVE_OR_NONE: "experiment",
    PathPolicy.BUNDLE_RELATIVE_LIST: "experiment",
    PathPolicy.HELPER_RESOLVED: "experiment",  # unused today; total-map filler
    PathPolicy.IS_NONE_ACCEPTABLE: "user",  # nulled at emit = machine-local host path
}

# Explicit non-Path config-field -> bucket table (C8 full-taxonomy extension).
# Total, WITH _PATH_FIELD_POLICY, over model_fields(system_config) |
# model_fields(analysis_config) -- enforced by
# tests/test_reprex_taxonomy.py::test_field_bucket_is_total (bidirectional).
# RULE: HPC-execution fields -> "hpc"; every other non-Path field -> "experiment";
# no non-Path field is "user" (the two software-dir Path fields are already "user"
# via IS_NONE_ACCEPTABLE -> "user"). The membership below is the RULE enumerated;
# the totality test is what guarantees completeness as fields are added/removed.
_FIELD_BUCKET: dict[str, Bucket] = {
    # ---- HPC-execution fields (hpc: bundled + flagged + revisable on the target machine) ----
    "run_mode": "hpc",
    "n_mpi_procs": "hpc",
    "n_omp_threads": "hpc",
    "n_gpus": "hpc",
    "n_nodes": "hpc",
    "multi_sim_run_method": "hpc",
    "hpc_total_nodes": "hpc",
    "hpc_total_job_duration_min": "hpc",
    "hpc_mem_allocation_for_sim_output_processing_mb": "hpc",
    "hpc_mem_allocation_for_analysis_output_consolidation_mb": "hpc",
    "hpc_mem_allocation_for_setup_mb": "hpc",
    "hpc_runtime_min_for_setup": "hpc",
    "hpc_max_wait_for_inflight_min": "hpc",
    "local_gpus_for_workflow": "hpc",
    "mem_gb_per_cpu": "hpc",
    "hpc_time_min_per_sim": "hpc",
    "hpc_restart_times_simulate": "hpc",
    "hpc_restart_times_other": "hpc",
    "hpc_ensemble_partition": "hpc",
    "hpc_setup_and_analysis_processing_partition": "hpc",
    "execution_environment": "hpc",  # native<->container is HPC-revisable (UVA->Frontier, FQ3/FQ4)
    "local_cpu_cores_for_workflow": "hpc",  # machine-local exec resource (sibling of local_gpus_for_workflow)
    # ---- every remaining non-Path field of system_config + analysis_config -> "experiment" ----
    "SWMM_git_URL": "experiment",
    "SWMM_tag_key": "experiment",
    "TRITONSWMM_branch_key": "experiment",
    "TRITONSWMM_git_URL": "experiment",
    "TRITON_raw_output_type": "experiment",
    "TRITON_reporting_timestep_s": "experiment",
    "additional_SBATCH_params": "experiment",
    "analysis_description": "experiment",
    "analysis_id": "experiment",
    "clear_raw": "experiment",
    "constant_mannings": "experiment",
    "crs": "experiment",
    "dataset_license": "experiment",
    "dem_building_height": "experiment",
    "dem_outside_watershed_height": "experiment",
    "eda": "experiment",
    "force_rerun": "experiment",
    "hydraulic_timestep_s": "experiment",
    "is_subanalysis": "experiment",
    "landuse_description_colname": "experiment",
    "landuse_lookup_class_id_colname": "experiment",
    "landuse_lookup_mannings_colname": "experiment",
    "landuse_plot_color_colname": "experiment",
    "manhole_diameter": "experiment",
    "manhole_loss_coefficient": "experiment",
    "ncols": "experiment",
    "nrows": "experiment",
    "open_boundaries": "experiment",
    "process_append_batch_memory_budget_mb": "experiment",
    "process_append_batch_timesteps": "experiment",
    "process_output_target_chunksize_mb": "experiment",
    "process_store_float32": "experiment",
    "process_timestep_chunk": "experiment",
    "processed_xllcorner": "experiment",
    "processed_yllcorner": "experiment",
    "rainfall_units": "experiment",
    "report": "experiment",
    "storm_tide_units": "experiment",
    "subcatchment_raingage_mapping_gage_id_colname": "experiment",
    "target_dem_resolution": "experiment",
    "target_processed_output_type": "experiment",
    "toggle_sensitivity_analysis": "experiment",
    "toggle_storm_tide_boundary": "experiment",
    "toggle_swmm_model": "experiment",
    "toggle_triton_model": "experiment",
    "toggle_tritonswmm_model": "experiment",
    "toggle_use_constant_mannings": "experiment",
    "toggle_use_swmm_for_hydrology": "experiment",
    "weather_event_indices": "experiment",
    "weather_time_series_spatial_mean_rainfall_datavar": "experiment",
    "weather_time_series_storm_tide_datavar": "experiment",
    "weather_time_series_timestep_dimension_name": "experiment",
}


def all_field_bucket(field_name: str) -> Bucket:
    """Provably-total user/hpc/experiment bucket over EVERY config field.

    Path fields delegate to their ``_PATH_FIELD_POLICY`` -> bucket mapping;
    non-Path fields read the explicit ``_FIELD_BUCKET`` table. Totality over
    ``model_fields(system_config) | model_fields(analysis_config)`` is enforced
    by ``test_field_bucket_is_total`` (bidirectional set-equality).

    Raises:
        KeyError: ``field_name`` is neither a Path field nor a bucketed non-Path
            field -- a new config field was added without a ``_FIELD_BUCKET``
            entry. The totality test fails loudly on the same condition.
    """
    if field_name in _PATH_FIELD_POLICY:
        return _POLICY_TO_BUCKET[_PATH_FIELD_POLICY[field_name]]
    try:
        return _FIELD_BUCKET[field_name]
    except KeyError:
        raise KeyError(
            f"{field_name!r} has no reprex bucket: add it to _FIELD_BUCKET "
            f"(user/hpc/experiment). All config fields must be bucketed "
            f"(test_field_bucket_is_total)."
        ) from None


def column_bucket(column_name: str) -> Bucket:
    """Bucket a sensitivity CSV/XLSX column by its prefix root (Gotcha 54).

    ``hpc.`` root -> "hpc" (the HPC-revisable axis, e.g. ``hpc.partition``).
    ``system.``/``analysis.`` root -> strip the prefix and delegate to
    ``all_field_bucket`` on the underlying field. A bare (unprefixed) column
    delegates directly.
    """
    if column_name.startswith("hpc."):
        return "hpc"
    for prefix in ("system.", "analysis."):
        if column_name.startswith(prefix):
            return all_field_bucket(column_name[len(prefix) :])
    return all_field_bucket(column_name)


def field_bucket(field_name: str) -> Bucket:
    """Return the ADR-10 USER/HPC/EXPERIMENT bucket for a config Path field.

    Args:
        field_name: A Pydantic Path-field name from system_config or
            analysis_config (a ``_PATH_FIELD_POLICY`` key).

    Returns:
        The reprex bucket: "user" | "hpc" | "experiment".

    Raises:
        KeyError: ``field_name`` is not a config Path field (not in
            ``_PATH_FIELD_POLICY``). Non-path fields (e.g. ``case_name``,
            ``gpu_hardware``) are out of scope for this foundation slice -- see
            bundle-reprex-roundtrip C8 for the full taxonomy.
    """
    try:
        policy = _PATH_FIELD_POLICY[field_name]
    except KeyError:
        raise KeyError(
            f"{field_name!r} is not a config Path field; field_bucket classifies "
            f"only _PATH_FIELD_POLICY keys. Non-path fields are out of scope "
            f"(see bundle-reprex-roundtrip C8)."
        ) from None
    return _POLICY_TO_BUCKET[policy]
