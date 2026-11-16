# Set observation points and initial conditions

## Goal

Record flood depth and unit discharge at chosen points through time, and start TRITON from water that already lies on the surface. Both inputs are optional and apply to the coupled model and to the TRITON-only model.

## Prerequisites

- A working [installation](installation.md).
- A system config whose DEM has been processed. The check of a point or a raster reads the processed DEM.
- For the points, a CSV of locations in the horizontal CRS of the system config.
- For the initial conditions, ESRI ASCII grids on the grid of the processed DEM.

## Key references

- [Output data model](../reference/output-data-model.md) describes the analysis tree that holds the observation groups.
- [Config filling](config-filling.md) orders the remaining fields of both configs.
- [In-flight operations](in-flight-operations.md) covers the resume of an interrupted analysis.

## 1. Write the points file

The points file is a CSV with the header `id,x,y`.

```text
id,x,y
north_gauge,436120.5,3992410.0
river_mouth,436480.0,3991980.5
```

The rules for the file are in the list.

- The header reads `id,x,y`, and no other column is allowed.
- Each `id` is a unique, non-empty string. The `id` names the point in the results.
- Each `x` and `y` is a finite number in the horizontal CRS of the system config, `crs.horizontal_epsg`. The coordinates match the DEM.
- The file lists at least 1 point and is UTF-8 text.
- The row order is the identity of the points. The results keep the order of the file.

Point `observation_points` in the system config at the file.

```yaml
observation_points: /path/to/observation_points.csv
```

Every point must lie inside the extent of the processed DEM and on a cell that TRITON computes on. A cell with the NODATA value is refused. A cell at or above the smaller of `dem_outside_watershed_height` and `dem_building_height` is refused when either one is set, because the system burns the heights into cells that are not terrain.

The preflight check of `analysis.validate()` reads the file and names each offending point by `id`. The check against the DEM runs when a scenario is prepared, after system setup. The check names the point and the cell. The toolkit writes the `x,y` rows of the file into `observation_points.txt` in the folder of each scenario in the order of the CSV. The rendered cfg sets `time_series_flag=1` and `observation_loc_file`.

## 2. Set the interval

Set `TRITON_observation_interval_s` in the analysis config to the seconds of simulated time between writes.

```yaml
TRITON_observation_interval_s: 600
```

The value must be greater than 0. With the field unset, TRITON writes at the reporting step, `TRITON_reporting_timestep_s`. An interval set without `observation_points` is refused at preflight, because TRITON would record no series.

TRITON writes the state at time 0, then at the first step at or after each multiple of the interval, then at the end of the simulation. The toolkit labels each periodic row with the nominal time, `k` times the interval, and not with the time of the step that TRITON recorded. The final row keeps the actual time of the final step. When the simulated duration is a whole number of intervals, the final write can fall on the last period, and the series then holds 1 row for both.

TRITON prints depth and both unit discharges at every point, because the toolkit holds `print_option` at `huv`.

## 3. Read the results

The processing step reads the text files that TRITON wrote under `series/` of the model output folder. The step writes 3 variables into the per-scenario gridded series store of the same model, on the dimensions `point_elapsed_s` and `obs_point`. No store beyond the existing per-scenario stores is created.

| Variable | Meaning | Unit |
|---|---|---|
| `point_flood_depth` | Flood depth at the point | `m` |
| `point_unit_discharge_x` | Unit discharge in x | `m2/s` |
| `point_unit_discharge_y` | Unit discharge in y | `m2/s` |

The consolidation of the analysis appends the variables to 2 groups of `analysis_datatree.zarr` for each model.

| Model | Series group | Summary group |
|---|---|---|
| Coupled | `tritonswmm/triton_observation/timeseries` | `tritonswmm/triton_observation/summary` |
| TRITON-only | `triton_only/triton_observation/timeseries` | `triton_only/triton_observation/summary` |

The series group holds the 3 variables named earlier on the dimensions `event_iloc`, `elapsed_s` and `obs_point`. The summary group holds `max_point_flood_depth`, in `m`, and `time_of_max_point_flood_depth`, in `s`, on the dimensions `event_iloc` and `obs_point`.

Each point carries the integer index `obs_point`, the string `obs_point_id` from the file, the coordinates `obs_x` and `obs_y`, and the DEM cell as `obs_row` and `obs_col`. Every event of an analysis shares 1 point set, because the points belong to the system. A sensitivity sub-analysis that changes the DEM resolution changes `obs_row` and `obs_col` in the group of the member, so read the 2 coordinates within a member and never across members.

The consolidation writes the observation groups whenever `observation_points` is set. The groups stay on when `toggle_consolidate_timeseries` is off. `clear_raw` removes the text files of `series/` once the store holds the rows. The sensitivity master holds the groups of each member under `member_{id}/` and then the group path.

The section on [opening the consolidated output](../reference/output-data-model.md#opening-the-consolidated-output) opens the tree. Select a point by the identifier in the string coordinate.

```python
depth = group["point_flood_depth"].swap_dims(obs_point="obs_point_id").sel(obs_point_id="river_mouth")
```

## 4. Resume an interrupted run

An uninterrupted run keeps the labels `k` times the interval. A resume starts again from a checkpoint and appends to the text files of TRITON, so the rows between the checkpoint and the interruption appear twice. The toolkit keeps the last occurrence of each row.

A resumed run labels the later rows from the resume time `R`. The restart row takes the label `R`, and each later periodic row takes `R + j` times the interval. When `R` equals the time of a row that the restart replaced, the restart row takes the label of that row and the grid continues.

The `elapsed_s` axis of the consolidated group holds the shared grid plus the labels of any resumed scenario. An event that wrote no row at a label holds NaN there. The processing step refuses a series whose row count disagrees with the simulated duration and names the file and both counts.

## 5. Start from initial conditions

The analysis config has 3 fields for the initial rasters. Each is an ESRI ASCII grid.

| Field | Content | Unit |
|---|---|---|
| `TRITON_initial_depth_raster` | Initial water depth, or water-surface elevation | `m`, or the unit you name |
| `TRITON_initial_qx_raster` | Initial unit discharge in x | `m2/s` |
| `TRITON_initial_qy_raster` | Initial unit discharge in y | `m2/s` |

The header of each grid must equal the header of the processed DEM in the keys `ncols`, `nrows`, `xllcorner`, `yllcorner` and `cellsize`. Take the header from the processed DEM. A grid that names cell centres with `xllcenter` or `yllcenter` is refused, because the half-cell shift would be silent. A grid that differs is refused at scenario preparation, and the message names the raster and the differing key.

A field left unset writes no line to the cfg, and the cfg is unchanged on every line that TRITON reads. TRITON then starts dry and with no flow. Each field that is set writes `h_infile`, `qx_infile` or `qy_infile`.

On a resume, the arrays of the checkpoint replace the initial rasters. The checkpoint cfg carries the raster settings without effect.

### The form TRITON reads

TRITON reads each initial-condition file without a header. TRITON takes the number of columns from the single-space-separated words of the first line, and the number of rows from the number of lines. TRITON reads a word with no decimal point as an integer, so `1e-05` becomes a wrong value. A tab, a run of spaces, a leading space, or a CRLF line end gives empty or merged words.

The toolkit therefore never passes your grid to TRITON. The toolkit parses your grid and writes a copy that holds only the values, in fixed-point form with a decimal point, separated by single spaces, north row first. The cfg names the copy. The copy sits in `triton_initial_conditions/` beside the processed DEM, and the file name carries a hash of your grid and of the settings in this section.

A copy that exists is reused. The copy holds each value to 5 decimals, so a value finer than 1e-5 is rounded and a value under 5e-6 becomes 0. A byte-order mark at the start of your grid is accepted. A file that is not UTF-8 text is refused with the field and the file name.

### Cells with no data

The toolkit detects a no-data cell on your values. A cell is no-data when the cell equals the `NODATA_value` of the header or when the cell is not finite.

- A no-data cell of `TRITON_initial_qx_raster` or `TRITON_initial_qy_raster` becomes 0.
- A no-data cell of `TRITON_initial_depth_raster` takes the depth in `TRITON_initial_depth_nodata_fill`, in meters. The field has no default. With the field unset, a depth raster that holds a no-data cell is refused. The message gives the count of the no-data cells, the first cell, and the line to write, `TRITON_initial_depth_nodata_fill: 0`.

The toolkit logs the count of the cells that were filled for each raster.

### Depth or water-surface elevation

A grid of depths and a grid of elevations look alike, so `TRITON_initial_depth_raster_kind` is required whenever the depth raster is set, and the field has no default.

```yaml
TRITON_initial_depth_raster: /path/to/initial_depth.asc
TRITON_initial_depth_raster_kind: depth
```

Under `depth`, the cells are depths in meters. A negative cell is refused, and the message gives the count and the first cell. The toolkit never clips a negative depth.

Under `water_surface_elevation`, the cells are levels. 2 more fields are then required, and neither has a default.

```yaml
TRITON_initial_depth_raster: /path/to/initial_level.asc
TRITON_initial_depth_raster_kind: water_surface_elevation
TRITON_initial_depth_raster_units: ft
TRITON_initial_depth_raster_datum_offset: 0.0
```

`TRITON_initial_depth_raster_units` is one of `m`, `cm`, `mm`, `in`, `ft` or `ftUS`. `ftUS` is the US survey foot. `TRITON_initial_depth_raster_datum_offset` is the number of meters to add to each level to bring the level to the datum of the processed DEM. The config refuses both fields under `depth`.

The toolkit applies 4 steps in this order.

1. Detect the no-data cells on your values.
2. Convert every other cell to meters with the unit, then add the offset.
3. Subtract the processed DEM from the converted cells and set a negative result to 0. A cell below the ground is dry, and the toolkit logs how many cells were set dry. A cell outside the watershed carries the wall height of the processed DEM, so the cell is dry too.
4. Give each no-data cell the fill as a depth. The no-data cells skip the subtraction. With the fill unset, the raster is refused, as the no-data steps in the previous section state.

## 6. Check the config

Run `analysis.validate()` ahead of any job submission. The check rejects a malformed points file and an interval with no points. A non-positive interval is refused when the analysis config loads. A mismatch between a raster and the DEM, a negative depth, a no-data cell with no fill, and a point outside the DEM appear when the first scenario is prepared and before any solver starts. The analysis config refuses a depth raster that has no `TRITON_initial_depth_raster_kind` when the config loads. The config also refuses a negative fill. The config refuses an offset that is set under `depth`, and a unit that is set there.
