# Output data model

Every variable below carries CF-1.13 attributes. `src/hhemt/cf_conventions.py` is the single source of truth for `standard_name`, `long_name`, `units`, and `cell_methods`, and the tables on this page derive from that module.

## The output tiers

`system.process.consolidate_system_datatree()` writes the per-system tier, and no workflow calls the method.

| Tier | Artifact | Format |
|---|---|---|
| Per-scenario | `sims/{event_id}/processed/` | Flat Zarr by default or NetCDF, selected by `target_processed_output_type` |
| Per-analysis | `analysis_datatree.zarr` | Hierarchical `xarray.DataTree`, always Zarr |
| Per-experiment | `experiment_datatree.zarr` | Hierarchical `xarray.DataTree`, always Zarr |
| Per-system | `system_datatree.zarr` | Hierarchical `xarray.DataTree`, always Zarr, written only by an explicit call as [the per-system tier](#the-per-system-tier) describes |

NetCDF is available for every model. With `target_processed_output_type: nc`, each per-scenario artifact that the output type governs is one NetCDF file under `sims/{event_id}/processed/` in place of a Zarr store. The enabled models make no difference, whether `toggle_triton_model`, `toggle_tritonswmm_model`, or `toggle_swmm_model`. The NetCDF artifacts are listed below.

- The gridded TRITON timeseries and summaries.
- The performance timeseries and summaries.
- The SWMM node and link timeseries and summaries.

The hydrology capture summaries `hydrology_inflow_summary.zarr` and `hydrology_rpt_summary.zarr` stay Zarr under either output type.

Set the output type in the analysis config.

```yaml
target_processed_output_type: nc
```

A regular analysis writes the per-scenario and per-analysis tiers in the analysis directory. A sensitivity analysis writes a directory per member, and each member carries per-scenario and per-analysis tiers under `members/member_{member_id}/`. The experiment root holds `experiment_datatree.zarr`, with a node per completed member.

`sensitivity_datatree.zarr` is the name that an experiment tree carried before the tree was unified. An explicit command converts a tree that still carries that name, and the next workflow never converts the tree automatically. The command is `python -m hhemt.version_migration migrate {analysis_dir} --apply`. Without `--apply` the command reports the pending conversion and changes no file. The conversion can leave a superseded copy of the tree beside the promoted tree, named `experiment_datatree.zarr.superseded-v0022`.

A multi-member sensitivity experiment can also carry a leftover `analysis_datatree.zarr` beside the experiment tree when the root held both that store and a tree under the earlier name before the conversion. The same conversion moves the leftover store aside to `analysis_datatree.zarr.superseded-v0025`. The conversion reads the analysis config that governs the experiment, and the command reads a config only when `--system-config` and `--analysis-config` are passed together, so pass both. Without both flags the command stops and moves no file. The command also stops when the current store stays unidentified.

hhemt keeps each superseded copy on purpose, and you remove the copy yourself.

**The per-scenario tier is flat and the consolidated tiers are hierarchical.** Open the flat tier as a plain `Dataset` and the consolidated tiers as `DataTree` objects.

The completion flag, and no directory listing, signals that a per-scenario TRITON timeseries store is finished. The flag is a sibling file named after the store, for example `TRITONSWMM_TRITON_tseries.zarr.done`. The flag is written once every chapter of the store has been verified and merged. An existing store with no flag sibling is an interrupted write. A `.chapters` sibling directory exists while the store is being written, and the directory is normally removed as soon as the flag lands. A process killed between the flag write and the removal leaves the directory behind indefinitely. A `.chapters` directory beside a flagged store is a harmless leftover and signals no incomplete result. A forced rerun of the scenario removes the flag and the `.chapters` directory.

### The per-scenario tier

The per-scenario processing step `hhemt.process_timeseries_runner` writes the tier. For each scenario the step calls `write_timeseries_outputs` and then `write_summary_outputs` on the post-processing object of the scenario. Together the calls convert the raw output of the solver into the processed stores. The tier lives in `{analysis_dir}/sims/{event_id}/processed/`, a directory for each scenario.

### The per-analysis tier

`consolidate_to_datatree()` on the `process` object of the analysis writes the tier. The call assembles the per-scenario summaries into `{analysis_dir}/analysis_datatree.zarr`. Only a regular analysis binds that path. A sensitivity analysis keeps the store in the directory of each member, under `members/member_{member_id}/`.

### The per-experiment tier

`consolidate_sensitivity_datatree()` on a sensitivity analysis writes the tier. The call first makes sure that each member has an `analysis_datatree.zarr`. The call then assembles the members into `{analysis_dir}/experiment_datatree.zarr` at the experiment root, with a node for each completed member. Only a sensitivity analysis binds this path.

### The per-system tier

??? note "Write timing and held analyses of the per-system tier"
    `system.process.consolidate_system_datatree()` writes the tier into `system_datatree.zarr` in the system directory. The tier is written only when that method is called explicitly. No workflow, command-line command, or test calls the method. The writes use append mode over the analyses bound to the system, so the store holds the analyses that were bound when some call ran. Only analyses bound at some call have a group in the store.

## The default state

With none of the four keys below set, a run keeps the consolidated time series in the consolidated tree of the analysis (the experiment tree for a sensitivity analysis) and removes every raw and intermediate output.

| Key | Default | Effect |
|---|---|---|
| `toggle_consolidate_timeseries` | `true` | Every model that writes a series also writes a `timeseries` branch in the tree. |
| `toggle_clear_scenario_timeseries_after_consolidation` | unset, which resolves to the consolidation toggle (`true`) | The per-scenario stores are cleared once the tree holds their read-back copy. |
| `clear_raw` | `all` | The raw solver output of every enabled model is deleted after processing. |
| `remove_after_processing` | `all` | Every artifact class is removed once the summaries are verified. |

The `timeseries` class of `remove_after_processing` is never applied by the process runner. With consolidation on and the clearing switch on, the consolidation clears the per-scenario series after the read-back. With the clearing switch off, the series stay in the per-scenario store, and the runner logs that. With consolidation off, the per-scenario series are the processed output, and the class is skipped with a logged reason.

The scenario and member directories are removed only when the clearing switch resolves to `true`, `clear_raw` covers every enabled model and `remove_after_processing` elects every class. Setting any one of the three to its keeping value retains the directories for reprocessing.

## Opening the consolidated output

```python
import xarray as xr

tree = xr.open_datatree(
    "path/to/analysis_datatree.zarr",
    engine="zarr",
    chunks="auto",
    consolidated=False,
)
```

Pass `consolidated=False`, because the stores are written without consolidated metadata.

The root of an experiment tree carries a `parameters` dataset that describes **every defined** member. Only **completed** members appear as `member_*` nodes. A tree with fewer nodes than parameter rows is an expected partial-completion state.

Every group of an analysis tree is written per model and artifact, and each group holds a `summary` branch and a `timeseries` branch. The `summary` branch holds the statistics per event and entity. The `timeseries` branch holds the consolidated series. The branches follow opposite write rules. The consolidation rewrites each `summary` group in place, group by group, whenever the inputs change. A `timeseries` branch is append-only. The consolidation adds a scenario at a time to the branch, and neither a rebuild of the summaries nor a reprocess delete removes or rewrites the branch. A rebuild of the summaries therefore keeps the series, and the series stay when `toggle_clear_scenario_timeseries_after_consolidation` has deleted the per-scenario stores.

With `toggle_clear_scenario_timeseries_after_consolidation` on, the consolidation also deletes the per-scenario summary stores of each model, scenario by scenario, once every slice of the scenario has been read back from the `summary` branches of `analysis_datatree.zarr` and found equal. The model log of the scenario records the deletion as `consolidated_summaries_cleared` before the delete, with the groups each store fed and the variable names each group received, and sets the record to done afterwards. From the first clearing a `summary` group only gains events. A cleared event stays as stored, an event that still has per-scenario stores is read back and rewritten in place when the values differ, and a new event is appended, with any variable the group lacks created first and filled with NaN at the earlier events. Every `summary` group stores one event per chunk, so an append rewrites no chunk of an earlier event. A cleared scenario may not leave the analysis, and only a new simulation replaces the summaries of a cleared scenario. The reprocess delete keeps every `summary` branch of an analysis that records a clearing.

| Group | Holds |
|---|---|
| `tritonswmm/triton` | Coupled model, 2D surface results |
| `tritonswmm/swmm_node` | Coupled model, SWMM node results |
| `tritonswmm/swmm_link` | Coupled model, SWMM link results |
| `tritonswmm/performance` | Coupled model, performance summary |
| `triton_only/triton` | TRITON-only model, 2D surface results |
| `triton_only/performance` | TRITON-only model, performance summary |
| `swmm_only/swmm_node` | SWMM-only model, node results |
| `swmm_only/swmm_link` | SWMM-only model, link results |
| `swmm_only/system` | SWMM-only model, whole-system results |

The `summary` branch of a group sits at `{group}/summary`, so the coupled SWMM node summary is `tritonswmm/swmm_node/summary`. The group holds no data variable. With `toggle_consolidate_timeseries` on, which is the default, every model that writes a series also writes a `timeseries` branch. The coupled model writes `tritonswmm/swmm_node/timeseries`, `tritonswmm/swmm_link/timeseries` and `tritonswmm/triton/timeseries`, the SWMM-only model writes `swmm_only/swmm_node/timeseries`, `swmm_only/swmm_link/timeseries` and `swmm_only/system/timeseries`, and the TRITON-only model writes `triton_only/triton/timeseries`. The `swmm_only/system` group holds 6 series that describe the whole system, `system_flooding_rate`, `system_outfall_flow`, `system_stored_volume`, `system_lateral_inflow`, `system_runoff` and `system_rainfall`, and the `summary` branch of the group holds the maximum and the last value of each. The SWMM-only node group also carries the rows of the Runoff Quantity Continuity table as `runoff_continuity_` variables in `mm`. The coupled model has neither, because the hydraulics model of the coupled build routes no runoff and holds flooded water as ponded volume. A layout-30 standalone SWMM analysis needs the scenarios reprocessed before a series consolidation, because the system series store is new in layout 31. The performance groups have no `timeseries` branch. Each branch sits on an `elapsed_s` axis of seconds since the start of the event, with an `event_start_datetime` coordinate on `event_iloc`. The gridded group carries a separate axis that includes the instant 0. The field defaults to `true`. Once consolidation is on, `toggle_clear_scenario_timeseries_after_consolidation` left unset takes the value of `toggle_consolidate_timeseries`, so the per-scenario time-series copies of every model, the gridded TRITON stores included, are deleted after the consolidated tree holds their verified slice. Setting the field to `false` keeps the per-scenario copies. Setting the field to `true` with consolidation off is refused at preflight. From layout 31 no separate `analysis_timeseries.zarr` store exists. Layout 30 kept the series in that store, and migration 31 copies each group of the store into the tree and removes the store after reading the copy back equal.

An analysis tree carries a consolidation stamp as the root attributes `consolidation_version` and `consolidation_inputs_fingerprint`. A write removes the stamp before writing a group and sets the stamp last, and the reprocess delete removes the stamp before deleting a `summary` branch. A tree without the stamp is read as an interrupted write, and the next consolidation rewrites every `summary` group.

A sensitivity master keeps both branches of each member in the experiment tree, at `member_{id}/{group}/summary` and `member_{id}/{group}/timeseries`. No separate `sensitivity_timeseries.zarr` store exists from layout 31. Migration 31 copies an existing one into the master and removes the store after reading the copy back equal. The master copies one member at a time and reads every copied group back equal. With `toggle_clear_scenario_timeseries_after_consolidation` on, the master then removes the `analysis_datatree.zarr` of the member. The master records the removal in the log of the member as `member_tree_moved_to_master` before the delete, so the master holds the only copy. The reprocess delete of the master removes the `summary` branches and keeps the `timeseries` branches. The system tree copies the `summary` branches only.

With `toggle_clear_scenario_timeseries_after_consolidation` on, the summaries of an analysis live only in the `summary` branches of the tree of that analysis, and the system tree points to the tree and copies no branch. The group of such an analysis in `system_datatree.zarr` is a root dataset that holds no leaf. The group attribute `analysis_tree_form` is `pointer`, and the group attribute `analysis_tree_relpath` names the tree of the analysis relative to the system directory. The path names the experiment tree for a sensitivity master, and a member whose tree moved into the master has no group, because the pointer to the master covers the member. A group of an analysis with the switch off copies the `summary` branches and carries `analysis_tree_form` as `copy`. A later call rewrites a group whose form differs from the form that call writes. `open_system_trees(system)` returns the tree of every analysis in either form and raises `ProcessingError` for a pointer whose tree is absent. A bundle carries the `summary` groups of the `analysis_datatree.zarr` of each `_test` group as the reference outputs, because the clearing deleted the per-scenario stores that held the summaries.

## Variables

A variable without a `standard_name` has no applicable CF standard name. The CF table omits the variable, and hhemt leaves the attribute unset.

**The unit is stored and no variable name spells the unit.** Every stored numeric variable at every tier carries a `units` attribute, and no variable name carries a unit token. The SWMM node depth is `depth` with `units: m`, and the flow of a link is `flow` with `units: m3 s-1`. A string-valued variable has no unit and carries no `units` attribute. The string-valued variables are `type`, `event_id`, and the producing-stamp coordinates. Some variables state no single unit and carry a `comment` that gives the reason. A variable that holds different units along one dimension is one such case. The water-quality variables below and the `max_abs_diff` difference of the raw byte-identity artifact are examples. `flow_balance_error_unsplit` is another case. The variable exists only in a store migrated from layout 27 or below, where a variable held a percentage on some rows and an error in litres on the other rows, and the store keeps no record of the unit per row.

### 2D surface results from TRITON

| Variable | Units | `standard_name` | Meaning |
|---|---|---|---|
| `max_flood_depth` | `m` | `flood_water_thickness` | Maximum flood depth. The solver keeps a running maximum of depth and updates the maximum every solver step, and TRITON reports the maximum as `MH` at the final print. The maximum can sit above the largest depth printed at any output timestep. |
| `flood_depth` | `m` | `flood_water_thickness` | Flood-depth timeseries |
| `flood_depth_last_tstep` | `m` | `flood_water_thickness` | Flood depth at the final timestep |
| `time_of_max_flood_depth` | `min` | n/a | Time of the print with the largest printed depth |
| `max_unit_discharge` | `m2 s-1` | n/a | Maximum unit discharge, the flow rate per meter of flow width |
| `unit_discharge_x` | `m2 s-1` | n/a | Unit discharge x-component |
| `unit_discharge_y` | `m2 s-1` | n/a | Unit discharge y-component |
| `unit_discharge_x_at_time_of_max_unit_discharge` | `m2 s-1` | n/a | Unit discharge x-component at the time of maximum unit discharge |
| `unit_discharge_y_at_time_of_max_unit_discharge` | `m2 s-1` | n/a | Unit discharge y-component at the time of maximum unit discharge |
| `time_of_max_unit_discharge` | `min` | n/a | Time of maximum unit discharge |
| `max_flood_velocity` | `m s-1` | `flood_water_speed` | Maximum over output timesteps of the flood velocity \|q\| / h using q and h of the same timestep. The value is NaN where h is at or below `flood_velocity_min_depth_m`, default 0.001 m. |
| `time_of_max_flood_velocity` | `min` | n/a | Time of the maximum flood velocity |
| `x_flood_velocity_at_time_of_max_flood_velocity` | `m s-1` | n/a | Flood-velocity x-component q_x / h at that time, along the grid x axis |
| `y_flood_velocity_at_time_of_max_flood_velocity` | `m s-1` | n/a | Flood-velocity y-component q_y / h at that time, along the grid y axis |
| `final_surface_flood_volume` | `m3` | n/a | Final surface flood volume |

!!! note "Unit discharge differs from velocity"
    The `QX` and `QY` rasters of the solver are depth-integrated flow per meter of flow width in `m2 s-1`. The rows whose names contain `unit_discharge` are therefore unit discharges, or the time of their maximum. The CF standard-name table has no name for the quantity, so the rows carry no `standard_name`, and the flood-depth rows keep theirs. The TRITON tier writes no variable named `max_velocity`. The name belongs to SWMM link nodes `tritonswmm_swmm_link` and `swmm_only_link`, where the variable is the maximum conduit velocity.

    The `flood_velocity` rows are velocities, meaning unit discharge divided by the depth of the same output timestep. Each is a maximum over output timesteps only, so a peak between consecutive prints escapes the maximum. The value is NaN where the depth is at or below `flood_velocity_min_depth_m` or the cell is never wet. Each row carries a `comment` attribute that states the formula and the threshold.

### Network results from SWMM

| Variable | Units | `standard_name` | Meaning |
|---|---|---|---|
| `total_inflow_vol` | `10^6 L` | n/a | Total inflow volume |
| `max_flow` | `m3 s-1` | n/a | Link tier, maximum flow magnitude from the SWMM Link Flow Summary "Maximum \|Flow\|" |
| `time_of_max_flow` | `min` | n/a | Link tier, time of that maximum |
| `max_velocity` | `m s-1` | n/a | Link tier, maximum conduit velocity |
| `max_flood_rate` | `m3 s-1` | n/a | Node tier, maximum flooding rate from the SWMM Node Flooding Summary "Maximum Rate" |
| `time_of_max_total_inflow` | `min` | n/a | Node tier, time of the maximum total inflow from the SWMM Node Inflow Summary "Time of Max Occurrence" |
| `max_ponded_depth` | `m` | n/a | Node tier, maximum ponded depth. The variable is present only where SWMM headed the last column of the Node Flooding Summary `Depth`, under Dynamic Wave routing. The value is NaN with a warning where the column is `Volume`. |
| `flow_balance_error` | `percent` | n/a | Node tier, flow balance error where SWMM printed a percent. The value is NaN on a row where SWMM printed litres. |
| `flow_balance_error_volume` | `L` | n/a | Node tier, flow balance error where SWMM printed an absolute error followed by `ltr`. The value is NaN on a row where SWMM printed a percent. |
| `max_over_full_flow` | `1` | n/a | Maximum flow as a fraction of full-flow capacity |
| `max_over_full_depth` | `1` | n/a | Maximum depth as a fraction of full depth |
| `depth`, `head` | `m` | n/a | Node tier time series of water depth and hydraulic head at each report step |
| `inflow_flow`, `flooding` | `m3 s-1` | n/a | Node tier time series of total inflow and flooding rate at each report step |
| `flow`, `velocity` | `m3 s-1`, `m s-1` | n/a | Link tier time series of signed flow and velocity at each report step |
| `link_depth` | `m` | n/a | Link tier time series of water depth at each report step |
| `capacity` / `capacity_setting` | `1` | n/a | Link tier time series of the SWMM link capacity in a standalone SWMM workflow, or of capacity-or-setting in a coupled workflow, at each report step |
| `{series}_last`, `{series}_max` | the units of the series | n/a | For each series above, the value at the final report step and the largest value over the report steps. The `flow_max` and `velocity_max` of a link are the largest SIGNED values. The largest magnitudes of the report are `max_flow` and `max_velocity`. |
| `max_lateral_inflow`, `max_total_inflow` | `m3 s-1` | n/a | Node tier. Maximum lateral inflow and maximum total inflow from the SWMM Node Inflow Summary |
| `lateral_inflow_vol` | `10^6 L` | n/a | Node tier, total lateral inflow volume |
| `hours_flooded` | `h` | n/a | Node tier, hours that the node was flooded from the SWMM Node Flooding Summary |
| `time_of_max_flood` | `min` | n/a | Node tier, time of `max_flood_rate` |
| `tot_flooded_vol` | `10^6 L` | n/a | Node tier, total flood volume |
| `flow_balance_error_unsplit` | none | n/a | Node tier, migrated stores only. The flow balance error as layout 27 stored the error, a percent on some rows and litres on other rows. The variable carries a `comment` and no `units`. |

`max_over_full_flow` and `max_over_full_depth` are dimensionless, and CF expresses that as units `1`. A value above 1 means that the conduit exceeded the design capacity. Every `time_of_max_*` variable is in minutes elapsed from the SWMM REPORT start, clamped at 0, at minute resolution. The value equals the time since the simulation start only when the report start of the model equals the start date.

### Hydrology and runoff-model results

The hydrology report summary `processed/hydrology_rpt_summary.zarr` holds the Subcatchment Runoff Summary of the hydrology model, a row per subcatchment, as SWMM printed the summary. The variables are `total_precip`, `total_runon`, `total_evap`, `total_infil`, `imperv_runoff`, `perv_runoff`, and `total_runoff_depth` in `mm`, `total_runoff_volume` in `10^6 L`, `peak_runoff` in `m3 s-1`, and `runoff_coeff` in `1`. The runoff-model store `processed/hydrology_inflow_summary.zarr` holds `inflow` in `m3 s-1` and `total_inflow_volume` in `m3` per node. Each variable carries a `units` attribute.

### Water-quality results from quality-routed SWMM models

The variables are written only when the SWMM report prints `Water Quality .......... YES`. SWMM prints that line when the model declares at least one pollutant and `IGNORE_QUALITY` is not `YES`. For any other model none of the names, coordinates, or dimensions below exist, and the outputs match the outputs of a model with no pollutants. The variables are carried along a `pollutant` dimension. The coordinate of the dimension lists the pollutant identifiers in the `[POLLUTANTS]` order of the model. The coordinate `pollutant_conc_units` gives the declared concentration unit of each pollutant, either `mg/L`, `ug/L`, or `#/L`.

| Variable | Dimensions | Units | Meaning |
|---|---|---|---|
| `pollutant_conc` | node, time, pollutant | per `pollutant_conc_units` | Node concentration at each report step from the time-series file |
| `link_pollutant_conc` | link, time, pollutant | per `pollutant_conc_units` | Link concentration at each report step from the time-series file |
| `pollutant_max_conc` | node, pollutant | per `pollutant_conc_units` | Maximum node concentration over the event |
| `pollutant_time_of_max_conc` | node, pollutant | `min` | Minutes from the `Starting Date` of the report to the first step at the maximum |
| `outfall_total_load` | node, pollutant | `kg`, or a count for `#/L` | Total mass discharged at each outfall, NaN at other nodes |
| `outfall_total_vol` | node | `10^6 L` | Total volume discharged at each outfall |
| `outfall_emc` | node, pollutant | per `pollutant_conc_units` | Event mean concentration, the total load over the total volume, NaN where the volume is zero |
| `link_total_load` | link, pollutant | `kg`, or a count for `#/L` | Total mass carried by each link over the workflow |
| `washoff_total_load` | subcatchment, pollutant | `kg`, or a count for `#/L` | Total washoff per subcatchment, in the hydrology summary and on the node dataset for a standalone model |
| `qual_routing_{term}` | pollutant | `kg`, or a count for `#/L` | A variable per row of the SWMM `Quality Routing Continuity` table. The terms are `dry_weather_inflow`, `wet_weather_inflow`, `groundwater_inflow`, `rdii_inflow`, `external_inflow`, `external_outflow`, `flooding_loss`, `exfiltration_loss`, `mass_reacted`, `initial_stored_mass`, and `final_stored_mass`. |
| `qual_routing_continuity_error` | pollutant | `percent` | Continuity error of that table |
| `runoff_qual_{term}` | pollutant | `kg`, or a count for `#/L` | A variable per row of the `Runoff Quality Continuity` table. The terms are `initial_buildup`, `surface_buildup`, `wet_deposition`, `sweeping_removal`, `infiltration_loss`, `bmp_removal`, `surface_runoff`, and `remaining_buildup`. |
| `runoff_qual_continuity_error` | pollutant | `percent` | Continuity error of that table |
| `pollutant_stats_status` coordinate on coupled summaries | event | n/a | `computed`, `computed: resumed by exact replay`, `computed: resumed from a snapshot that carries SWMM water-quality state`, or the reason that the statistics are withheld |

??? note "SWMM pollutant output and the UG/L declaration of trace pollutants"
    SWMM prints the mass of a count pollutant `#/L` as a base-10 logarithm. The toolkit linearizes every mass value and maps a printed `0.000` to 0. Concentrations in the coupled model come from the SWMM report. The report prints concentrations to 0.001 of the pollutant unit, so declare a trace pollutant in `UG/L`. The continuity percentage errors are read as printed. A variable omits `units` and carries a `comment` when the pollutants share no single unit. The concentration series carry no peak statistic under the generic `_max` and `_last` names, so the columns compared across simulations are the same with and without water quality.

## Global attributes

Tree roots carry `Conventions: "CF-1.13"`, `analysis_id`, and `system_id`.

A consolidated tree also carries provenance. A deterministic RO-Crate core sits in the root attribute `ro_crate_metadata`, with a co-located `ro-crate-metadata.json` sidecar beside the store. The embedded core is deterministic by construction and byte-identical across reruns, because the volatile fields of wall-clock times, host, and job id live only in the sidecar.

## A caveat on performance columns

The `performance.*` columns hold timing records.

- Only `performance.Total`, `performance.Simulation`, and `performance.Init` carry wallclock semantics. The category columns are slowest-rank cost.
- On a **hotstart-resumed** simulation the columns are cumulative across every allocation, so the values exceed the elapsed time that a scheduler reports for the final allocation. The cumulative value is the intended value.
- The `perf_*` columns of `df_status` and `scenario_status.csv` read the same values, so both points apply to the columns.

## See also

- [Configuration schema](config-schema.md) lists the inputs that produce the outputs.
- [The interactive analysis report](example-report.md) shows the rendered view of the same data.
- [FAIR scope table](fair-scope-table.md) gives the archival posture of each artifact class.
