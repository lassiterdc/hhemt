# Use a runnable SWMM model

## Goal

Point a system config at one plain SWMM model and let the toolkit derive the coupled, hydrology, and standalone models from the model.

## Prerequisites

- A working [installation](installation.md).
- A SWMM 5.2 `.inp` file that runs in SWMM as written, in `CMS` flow units, with no `${...}` placeholder.
- A rain-gauge mapping CSV that names a gauge for every subcatchment.

## Key references

- [Coupled inflow nodes](../reference/coupled-inflow-nodes.md) describes the selection of nodes that exchange water with the surface.
- [Config filling](config-filling.md) gives the order for filling the rest of the system config.

## 1. Name the model

Set `SWMM_model` in the system config to the path of the `.inp` file.

```yaml
SWMM_model: /path/to/model.inp
```

The toolkit derives the models in the table from this file.

| Model | Derivation |
|---|---|
| Standalone | The file, with `THREADS` set from `swmm_threads`. |
| Hydraulics | The drainage network without the subcatchment side, with `THREADS` set from `swmm_threads`. TRITON reads this model for the coupled run. |
| Hydrology | The subcatchment side without the network, routed with `KINWAVE`. SWMM runs this model first to produce runoff hydrographs. |

For each event the toolkit rewrites only the parts in the list.

- The simulation window in `[OPTIONS]`.
- The `[RAINGAGES]` section.
- The rain series and the series of each outfall that a water-level boundary record names.

A section that a model keeps reaches SWMM as written. The toolkit changes only the listed parts and the settings in the table. The toolkit also converts the outfalls that the config converts. A section that a model drops is absent from the model. A step option such as `ROUTING_STEP` therefore belongs in the file.

No model that the toolkit writes holds a data row in `[TIMESERIES]`, because every series reads a file. The toolkit extracts each inline series of `SWMM_model` once into a directory of the system directory that is named for the digest of the model. Every derived model points at the extracted file. A series of `SWMM_model` that already reads a file keeps the file, and the toolkit writes the absolute path that SWMM resolves for the file. The toolkit never edits `SWMM_model`.

`swmm_threads` sets the thread count of SWMM when SWMM routes the network by dynamic wave. The default is 1. SWMM never uses more threads than the OpenMP thread count of the run mode, and the check refuses a larger value. The hydrology model always runs on 1 thread.

## 2. Map the subcatchment to rain gauges

Every gauge named in `[SUBCATCHMENTS]` must appear in the `subcatchment_raingage_mapping` CSV, in the column named by `subcatchment_raingage_mapping_gage_id_colname`. The check names each gauge that is missing.

## 3. Decide each outfall

Every outfall of the coupled model is a stated decision, and the check names each outfall that has none.

| Outfall type in the model | Decision | Refusal text |
|---|---|---|
| `TIMESERIES` | Name the outfall in a water-level record of the external-boundary layer, and the record converts the outfall. Otherwise keep the outfall. | The check offers both lines. |
| `TIDAL` | Keep the outfall. Matching a `TIDAL` outfall is not supported. | The check offers the keep line. |
| `FREE`, `NORMAL`, or `FIXED` | Convert the outfall or keep the outfall. | The check offers both lines. |

An outfall that a water-level record names is converted by the record and appears in no list. A `TIMESERIES` or `TIDAL` outfall in `swmm_outfalls_to_couple` is refused, and the refusal offers a record or the keep line.

The system config converts a free, normal, or fixed outfall.

```yaml
swmm_outfalls_to_couple: [OUTFALL_A]
```

The analysis config keeps an outfall. A kept outfall drains as the model says. A kept `TIMESERIES` or `TIDAL` outfall reads the series that the model defines, unchanged for every event.

```yaml
swmm_outfalls_to_keep: [OUTFALL_B]
```

Name each outfall in 1 place only. The section on [converting outfalls](../reference/coupled-inflow-nodes.md#converting-outfalls) gives the conversion and the resulting rim.

## 4. Pin a TRITON build that reads the coupling list

The toolkit gives the solver an explicit list of coupled nodes. A TRITON build that ignores the list couples the nodes of the `[INFLOWS]` section instead. The preflight check refuses a run when the nodes of the build differ from the toolkit list. Set `TRITONSWMM_branch_key` to a commit that reads `swmm_coupling_nodes`. A container run needs the images rebuilt at the pinned commit.

## 5. Check the config

Call `analysis.validate()` before you submit any job. The check refuses a `${...}` placeholder in the model, a listed outfall that is no row of `[OUTFALLS]`, an outfall with no decision, and a subcatchment whose gauge is missing from the mapping. The same check covers the external-boundary layer, as the next section describes. Each refusal names the offender.

## External boundaries and outfalls

The external-boundary layer is a GIS layer of lines along the edge of the DEM. Each record sets 1 boundary condition of TRITON along the line of the record. Set `toggle_external_boundaries` to `true` and name the layer in `external_boundary_gis`. The toolkit converts the layer to the horizontal CRS of the system and refuses a layer with no CRS.

### The record types

The column that `external_boundary_type_attribute` names holds the type of each record. The toolkit writes 1 row of the TRITON location file for each record, in record order.

| Type | Row written | Boundary condition |
|---|---|---|
| `water_level` | `1,x1,y1,x2,y2,"tseries_{k}.txt"` | A water level that varies with time. Only this type writes a series file. |
| `free_outflow` | `0,x1,y1,x2,y2,0.0` | The boundary copies the inner cell to the boundary cell. |
| `normal_slope` | `2,x1,y1,x2,y2,{slope}` | Depth and discharge on the boundary cell and on the adjacent inner cell, set from a normal-flow slope. |
| `froude_number` | `3,x1,y1,x2,y2,{froude}` | Depth and discharge on the boundary cell and on the adjacent inner cell, set from a Froude number. |

For the `normal_slope` and `froude_number` types, the orientation factor of the edge directs the discharge, and the factor points outward on the west and east edges. A segment overrides `open_boundaries` on the cells of the segment.

The column that `external_boundary_value_attribute` names holds the slope or the Froude number. The column is required when any record has type `normal_slope` or `froude_number`, and the check refuses the column when no record has either type. The value must be a finite number above 0.

A record snaps to 1 edge of the DEM with 2 distinct cells. No 2 records may share a cell, so a segment must stop 1 cell short of a corner that another segment uses. Preflight prints the transformed endpoints of each record and the cells that the endpoints snap to, and the scenario log records them. With `toggle_external_boundaries` off, the location file has no row and no water-level file is written.

### The water level of each boundary

A `water_level` record names the outfalls that the record replaces in the column that `water_level_boundary_outfall_attribute` names. The column holds the outfall names separated by white space. A name that holds white space is not supported.

`water_level_boundary_series_source` decides where the series of the record comes from. The field is required when any `water_level` record exists.

With `weather_time_series`, `water_level_boundary_outfall_variables` maps the SWMM name of each named outfall to a variable of the event weather dataset. Every named outfall is a key of the map. The outfalls of 1 record map to 1 variable, so outfalls that need different variables go in different records.

```yaml
water_level_boundary_series_source: weather_time_series
water_level_boundary_outfall_variables: {E143250: waterlevel_m, E147007: waterlevel_m}
water_level_boundary_variable_units: {waterlevel_m: m}
water_level_boundary_variable_datum_offsets: {waterlevel_m: 0}
```

`water_level_boundary_variable_units` gives the unit of each variable, and `water_level_boundary_variable_datum_offsets` gives the datum offset in meters. The unit is `m`, `cm`, `mm`, `in`, `ft`, or `ftUS`. The toolkit multiplies the variable by the factor of the unit, adds the offset, and writes the same values in meters to the TRITON file and the SWMM file. Both maps need an entry for every variable that the first map names.

With the model source, the series of a record is the series that the outfalls of the record read in `SWMM_model`.

```yaml
water_level_boundary_series_source: swmm_model
```

The toolkit applies the rule that SWMM applies. A row that begins with a date sets the current date of the series, and the date before the first dated row is 01/01/2004. A time reads as decimal hours or as `hh:mm`. A relative file name resolves against the directory that holds `SWMM_model`.

The model series must cover the simulation window of the event, and the check names both ranges when the series falls short. An analysis of more than 1 event is refused under this source. The model values are meters, so the config sets no unit and no offset.

Under either source, the outfalls of 1 record must read the same series in `SWMM_model`. The check compares the rows of the series element by element. Only the time and the value of each row count. A difference of name makes no difference, and a difference of form makes none either. The same holds for the notation of a time, such as `1.5` against `01:30`. Under the model source, setting `water_level_boundary_unequal_outfall_series` to `average` replaces the refusal. The record then takes the mean of the outfalls at each time, the series must share the times, and the file header names the outfalls and the word `averaged`.

Each record writes `extbc/tseries_{k}.txt` for TRITON and `dats/water_level_{k}.dat` for SWMM, where `k` is the zero-padded record index. The first column of the TRITON file is the hours since the start of the event.

### Distances from outfalls to lines

For every outfall that a `water_level` record names, preflight computes the planar distance in the system CRS from the `[COORDINATES]` point of the outfall to the line of the record, before snapping. Preflight prints 1 line for each record, and the toolkit writes the distances to the header of the series file and to the scenario log.

```text
record 0: E143250 15.3 m, E147007 10.5 m
```

The distances are informational until `water_level_boundary_outfall_max_distance_m` is set. A set value refuses each named outfall that lies farther from the line.

### Node depths from the DEM

The toolkit writes the SWMM depth of each converted outfall from the DEM. The depth is the DEM elevation at the cell that TRITON gives the node, less the invert elevation. SWMM still raises the depth of a node at the end of a conduit to the crown, so the depth is the larger of the ground depth and the crown. Where the ground of a node lies above the crown, the depth rises from the crown to the ground. Otherwise SWMM keeps the crown. An outfall with no conduit end takes the depth from the DEM alone.

The check refuses a node whose cell lies outside the grid. The check also refuses a node whose cell has no data and a node whose cell lies at or below the invert. The refusal names the invert and the cell elevation.

`dem_rim_resolution` chooses the DEM. The field is required when any outfall is converted or `swmm_node_rim_from_dem` is `true`, and the check refuses the field when neither holds.

- `processed` reads the DEM that TRITON loads. With `swmm_node_rim_from_dem` on, TRITON takes the exchange rim of every coupled node from the same grid.
- `full` reads `DEM_fullres`. With `swmm_node_rim_from_dem` on, the toolkit writes the depth of every coupled node from the full-resolution DEM and writes no TRITON key, because TRITON would overwrite the rims with the processed grid. TRITON then takes the rim as the larger of the written depth and the crown of the pipe.

Under `full`, preflight prints a notice. A rim read at a point can differ from the average ground of the coarse cell that TRITON uses for the exchange, by as much as the relief of the cell. At Norfolk the outfall `E147007` has a full-resolution depth of 1.3065 m, a crown at 1.676 m, and a processed depth of 1.076 m. The rim is 1.676 m under `full` and 1.076 m under `processed` with the key on.

### Refusals

The check collects every refusal and raises once, so 1 call lists all the offenders. Each refusal prints the field, the problem, the current value, the fix, and a line to write or a layer column to edit where either applies. The examples show the field, the problem, the fix, and the line to write.

An outfall with a variable water level and no decision:

```text
[ERROR] analysis.swmm_outfalls_to_keep
  Problem: 2 outfalls of SWMM_model have a variable water level and no decision: OF_A (TIMESERIES series_a), OF_B (TIMESERIES series_b).
  Fix: Name each in the water_level_boundary_outfall_attribute column of the external-boundary record that replaces it, so it becomes a coupled node and the water level is applied on the TRITON surface; or keep it as an outfall that reads the model's series.
  Write:
    swmm_outfalls_to_keep: [OF_A, OF_B]
```

A variable-water-level outfall in the conversion list:

```text
[ERROR] system.swmm_outfalls_to_couple
  Problem: outfall(s) OF_A have a variable water level (TIMESERIES) and cannot be listed in swmm_outfalls_to_couple
  Fix: Name each in the outfall column of an external-boundary water_level record, which converts it, or list it in swmm_outfalls_to_keep to keep it an outfall that reads the model's series
  Write:
    swmm_outfalls_to_keep: [OF_A]
```

Outfalls of 1 record that read different series:

```text
[ERROR] analysis.water_level_boundary_outfall_attribute
  Problem: record 0 names outfalls OF_A (series series_a) and OF_B (series series_b) whose series differ from row 1: (datetime.datetime(2020, 1, 1, 1, 0), 0.7) against (datetime.datetime(2020, 1, 1, 1, 0), 0.9)
  Fix: Put each outfall in its own record, or make the model's series equal. The averaging option does not apply here, because under the weather source no run reads the outfalls' own series.
```

Records that share a cell:

```text
[ERROR] analysis.external_boundary_gis
  Problem: record pair(s) [(0, 1)] share a DEM cell after snapping; a segment must stop one cell short of a corner another segment uses
  Fix: Shorten one of the lines so the snapped segments share no cell
```

A node with no usable DEM depth:

```text
[ERROR] analysis.dem_rim_resolution
  Problem: the DEM gives no usable depth for node(s) E143250 (invert -1.079 m, cell elevation -1.500 m): each needs a finite cell elevation, not no-data, above its invert
  Fix: Raise the DEM there, lower the invert, or choose the other dem_rim_resolution
```

## Move from templates

Earlier versions took a SWMM file with placeholders for each role and a TRITON configuration template. A config that sets a field of the removed set is refused at load, and the message names the replacement. Merge the SWMM files into 1 runnable model, fill the placeholders once, and decide each outfall. The toolkit renders the TRITON cfg from the config. The TRITON settings that the template fixed are analysis config fields, for example `TRITON_courant_number` and `TRITON_hextra_m`.

## The Norfolk example

The Norfolk case study ships `hague_model.inp` as the runnable model. The system config template converts no outfall directly. Both map layers are GeoPackage files in EPSG:32147. They are the watershed polygon `norfolk_wshed_epsg32147_state_plane_m.gpkg` and the boundary layer `variable_bc.gpkg`. The boundary layer holds one record of type `water_level`, and the `outfalls` column names `E143250` and `E147007`. The analysis config template maps both outfalls to the weather variable `waterlevel_m`, sets the unit to `m` and the offset to 0, reads the series from the weather file, chooses the `processed` DEM, and keeps no outfall. The template pins `TRITONSWMM_branch_key` to commit `deefa6e6fcb26184a60eb8242dc82dcc3cab9618`, the head of the `develop` branch of the TRITON fork, and sets `TRITONSWMM_git_URL` to `https://github.com/lassiterdc/triton.git`.

On the processed grid, the ground depth of the 2 outfalls is 0.97 m and 1.08 m. The crowns lie at 1.52 m and 1.68 m. SWMM therefore keeps the crown. The converted junctions are rimmed at the crown of the highest connecting conduit. The published coupled Norfolk run used commit `15eb18a5d25afe5da295cb4b559a62669dbe5bc3` and a rim of 0 for the converted nodes. A new run therefore departs from the published coupled results. The size of the difference is unmeasured.
