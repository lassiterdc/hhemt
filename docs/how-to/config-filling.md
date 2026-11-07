# Fill in your configuration

## Goal

Produce a working system config and analysis config for your own study area, and confirm that both load before you spend any compute.

## Prerequisites

- A working [installation](installation.md).
- A DEM for your watershed, a SWMM `.inp` model, and a weather NetCDF.

## Key references

- [Configuration schema](../reference/config-schema.md) gives the meaning of each field.
- [HPC-profile setup](hpc-profile-setup.md) describes the HPC system config.
- [Operating on an analysis while jobs are in flight](in-flight-operations.md) covers the commands for an analysis that is running.

## 1. Start from the templates

Start from the [system config template](https://github.com/lassiterdc/hhemt/blob/main/test_data/norfolk_coastal_flooding/template_system_config.yaml) and the [analysis config template](https://github.com/lassiterdc/hhemt/blob/main/test_data/norfolk_coastal_flooding/template_analysis_config.yaml) on the repository. Save each template as a file of your own, such as `my_system.yaml` and `my_analysis.yaml`.

The templates define a serial, local configuration. The steps below replace a placeholder path, set a value only you know for your study area, or opt into a feature that the default leaves off.

## 2. Point the system config at your data

Replace every `/path/to/...` value in `my_system.yaml`. Set `crs.horizontal_epsg` to the projected CRS of your DEM. Authoring by hand without a case study also means replacing every `${DATA_DIR}` and `${PACKAGE_DIR}` placeholder.

Set `target_dem_resolution` to the cell size to simulate at. The setting coarsens the full-resolution DEM and has the largest effect on runtime of any single field.

A DEM with burned-in sentinel elevations for buildings or for the area outside the watershed needs `dem_building_height` and `dem_outside_watershed_height` set to the exact sentinel values. Report figures drop cells equal to the building value from the DEM colorbar range. They paint every cell at or above the lower of the values that are set, less `wall_threshold_buffer_m`, as a wall. A value that differs from your raster reports no error and shows as a colorbar stretched by rooftop cells or as real terrain painted as wall. Leave both fields unset when your raster carries no sentinels.

## 3. Resolve the toggles

Each toggle that you flip makes other fields required. Set the toggles first and then fill the fields that they demand. The [toggle-dependency table](../reference/config-schema.md#toggle-dependent-required-fields) lists each pairing. The common pairings are below.

- A landuse raster for Manning's *n* in place of a constant requires the lookup file, the raster, and the column names that the schema lists.
- SWMM for hydrology requires the hydrology `.inp` and a subcatchment-to-raingage mapping.

## 4. Point the analysis config at your events

Set `weather_timeseries`, `weather_events_to_simulate`, and the `weather_*` schema fields to the dimension and variable names as they appear in your NetCDF. The toolkit uses the names to find your data.

The template carries the names of the Norfolk weather file. Replace the values with yours.

```yaml
weather_timeseries: /path/to/events.nc
weather_events_to_simulate: /path/to/events_to_simulate.csv
weather_event_indices: [year, event_type, event_id]
weather_time_series_storm_tide_datavar: waterlevel_m
weather_time_series_timestep_dimension_name: timestep
weather_time_series_spatial_mean_rainfall_datavar: mm_per_hr
```

Set `hhemt_sha` to the full 40-hex commit of hhemt that you run. `git rev-parse HEAD` in your clone prints the hash. The template carries a placeholder. The field has no default, and every process that constructs the analysis refuses to start when the running toolkit is at a different commit.

## 5. Choose the run mode

The default is `run_mode: serial` with `multi_sim_run_method: local`. A `run_mode` beyond the default needs the fields that the mode requires from `n_mpi_procs`, `n_omp_threads`, `n_gpus`, and `n_nodes`. Config load cross-validates the fields, so a mismatch fails at load and never at dispatch.

The [glossary](../reference/glossary.md#multi_sim_run_method) defines the `multi_sim_run_method` values. A `batch_job` analysis config requires `hpc_total_job_duration_min`, `hpc_time_min_per_sim`, `hpc_ensemble_partition`, and `hpc_setup_and_analysis_processing_partition` to load. A `1_job_many_srun_tasks` config requires `hpc_ensemble_partition` and `hpc_setup_and_analysis_processing_partition` to load, and preflight also requires `hpc_total_nodes` and `hpc_total_job_duration_min`.

--8<-- "hpc-system-config-role.md"

Author the HPC system config as [HPC-profile setup](hpc-profile-setup.md) describes.

## 6. Verify before you commit compute

A dry run checks every step above in seconds and submits no job.

```bash
hhemt run --system-config my_system.yaml --analysis-config my_analysis.yaml --dry-run
```

The dry run exits 0 and prints the workflow that a real run executes. A non-zero exit names the offending field. Preflight validation accumulates every error and reports the errors together, so one round of fixes covers all of them.

For a stronger check, `analysis.test()` runs a minimal subset of your own analysis end to end against a solver you compiled already. The [interactive Python tutorial](../tutorials/interactive-python.ipynb) shows the call. A real analysis also needs a compiled solver, so compile the solver once first, following [Compile the solver](compiling-the-solver.md).

!!! warning "Edit the sensitivity XLSX and never the CSV"
    With `toggle_sensitivity_analysis: true`, edit the sensitivity XLSX and leave the derived `sensitivity_analysis_definition.csv` alone, because the workflow overwrites a direct CSV edit. [Rerun triggers](../reference/rerun-triggers.md) gives the reason.
