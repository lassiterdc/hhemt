# Fill in your configuration

hhemt takes two user configs (system + analysis) plus an optional HPC-system profile. Start from the in-repo templates: `test_data/norfolk_coastal_flooding/template_system_config.yaml` and `template_analysis_config.yaml`. The `${DATA_DIR}` / `${PACKAGE_DIR}` placeholders are filled automatically when you load a case study via `TRITON_SWMM_example`; if you author a config by hand, replace them with real paths.

## Annotated system config

```yaml
# --- CRS ---
crs:
  horizontal_epsg: 32147   # projected CRS of your DEM/GIS inputs (meters or ftUS)
  vertical_epsg: 5703      # vertical datum of DEM elevations

# --- Filepaths (inputs) ---
watershed_gis_polygon: '/path/to/watershed.shp'   # analysis-extent polygon
system_directory: '/path/to/system'               # where DEM/Manning's/compiled binaries land
DEM_fullres: '/path/to/fullres_dem_m.tif'         # full-resolution DEM raster
landuse_lookup_file: '/path/to/landuse_lookup.csv'   # required only if toggle_use_constant_mannings: false
landuse_raster: '/path/to/landuse.tif'               # required only if toggle_use_constant_mannings: false

# --- Software sources (cloned + compiled by the toolkit) ---
TRITONSWMM_software_directory: '/path/to/triton'
SWMM_software_directory: '/path/to/swmm'
TRITONSWMM_git_URL: 'https://code.ornl.gov/hydro/triton.git'
TRITONSWMM_branch_key: '<commit-sha-or-branch>'
SWMM_git_URL: 'https://github.com/USEPA/Stormwater-Management-Model.git'
SWMM_tag_key: 'v5.2.4'

# GPU backend: 'HIP' (AMD/ROCm) or 'CUDA' (NVIDIA); null for CPU-only.
# On HPC, this DERIVES from the resolved partition's PartitionSpec — leave null here.
gpu_compilation_backend: null

triton_swmm_configuration_template: '/path/to/TRITON_SWMM_definition_template.cfg'
SWMM_hydraulics: '/path/to/hydraulics_model_template.inp'   # 2D-coupling hydraulics model
SWMM_hydrology: '/path/to/hydrology_model_template.inp'     # required only if toggle_use_swmm_for_hydrology: true
subcatchment_raingage_mapping: '/path/to/subcatchment_raingage_map.csv'   # with toggle_use_swmm_for_hydrology
subcatchment_raingage_mapping_gage_id_colname: 'raingage_id'

# --- Manning's lookup column names (with toggle_use_constant_mannings: false) ---
landuse_description_colname: 'original_description'
landuse_lookup_class_id_colname: 'CLASS_ID'
landuse_lookup_mannings_colname: 'mannings'
landuse_plot_color_colname: 'plot_color'

# --- Toggles (see the dependency table below) ---
toggle_use_swmm_for_hydrology: true
toggle_use_constant_mannings: false
toggle_triton_model: false
toggle_tritonswmm_model: true
toggle_swmm_model: false

# --- Constants / parameters ---
dem_outside_watershed_height: 9999   # meters; sentinel elevation outside the watershed
dem_building_height: 80              # meters; building-footprint burn height
target_dem_resolution: 3.5           # meters; coarsen the full-res DEM to this cell size
```

## Annotated analysis config

```yaml
analysis_id: 'single_sim'   # analysis folder name; created under system_directory if absent

# --- Weather-input schema (names of dims/vars in your weather NetCDF) ---
weather_event_indices: ['year', 'event_type', 'event_id']
weather_time_series_storm_tide_datavar: 'waterlevel_m'
weather_time_series_timestep_dimension_name: 'timestep'
weather_time_series_spatial_mean_rainfall_datavar: 'mm_per_hr'
weather_timeseries: '/path/to/event_timeseries.nc'
storm_tide_boundary_line_gis: '/path/to/variable_bc.shp'
weather_events_to_simulate: '/path/to/events.csv'   # which events to run
analysis_description:

# --- Execution (add run_mode / multi_sim_run_method / n_* for non-serial) ---
run_mode: "serial"              # serial | openmp | mpi | hybrid | gpu
multi_sim_run_method: "local"   # local | batch_job | 1_job_many_srun_tasks
local_cpu_cores_for_workflow: 4

# --- Units ---
rainfall_units: 'mm/hr'
storm_tide_units: 'm'

# --- Toggles ---
toggle_sensitivity_analysis: false   # true requires the sensitivity_analysis XLSX path
toggle_storm_tide_boundary: true

# --- Parameters ---
target_processed_output_type: 'zarr'   # zarr or nc (per-scenario tier)
TRITON_raw_output_type: 'bin'          # bin or asc
manhole_diameter: 1.2
manhole_loss_coefficient: 0.1
hydraulic_timestep_s: 0.01
TRITON_reporting_timestep_s: 120
open_boundaries: 1
report: {}
clear_raw: none       # none | all | [tritonswmm, triton, swmm]
force_rerun: none      # none | all | [tritonswmm, triton, swmm]
```

The template ships a serial/local config. For a non-serial run, ADD the execution fields the mode requires: `run_mode` plus the matching `n_mpi_procs` / `n_omp_threads` / `n_gpus` / `n_nodes` (cross-validated at config load), and set `multi_sim_run_method` for how simulations are dispatched.

## Toggle-dependent required fields

| If you set… | you must also provide… |
|---|---|
| `toggle_use_constant_mannings: true` | `constant_mannings` |
| `toggle_use_constant_mannings: false` | `landuse_lookup_file`, `landuse_raster`, + 3 colname fields |
| `toggle_use_swmm_for_hydrology: true` | `SWMM_hydrology`, `subcatchment_raingage_mapping`, mapping colname |
| `toggle_sensitivity_analysis: true` | `sensitivity_analysis` (XLSX path) |
| any of `processed_xllcorner`/`yllcorner`/`ncols`/`nrows` | all four (all-or-nothing) |

!!! warning
    Edit the sensitivity XLSX, never the derived `sensitivity_analysis_definition.csv` — the CSV is re-derived on every `analysis.run()` (see the [rerun FAQ](../explanation/rerun-faq.md)).

## See also
- [HPC-profile setup](hpc-profile-setup.md)
- [Operating on an analysis while jobs are in flight](in-flight-operations.md)
