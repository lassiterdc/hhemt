# Output data model

What a completed analysis leaves on disk, and what is inside it. This is the
reference for the toolkit's actual deliverable. If you are writing analysis code
against hhemt's results, this is the page you want.

Every variable below carries CF-1.13 attributes. `src/hhemt/cf_conventions.py` is
the single source of truth for `standard_name`, `long_name`, `units`, and
`cell_methods`; the table on this page is derived from it.

## The output tiers

hhemt writes results at four tiers, and they differ in both shape and format.
The per-system tier is written only by an explicit
`system.process.consolidate_system_datatree()` call, which no run makes.

| Tier | Artifact | Format |
|---|---|---|
| Per-scenario | `sims/{event_id}/processed/` | Flat Zarr (default) or NetCDF, selected by `target_processed_output_type`; see below |
| Per-analysis | `analysis_datatree.zarr` | Hierarchical `xarray.DataTree`, always Zarr |
| Per-experiment | `experiment_datatree.zarr` | Hierarchical `xarray.DataTree`, always Zarr |
| Per-system | `system_datatree.zarr` | Hierarchical `xarray.DataTree`, always Zarr. Written only by an explicit `system.process.consolidate_system_datatree()` call; each call adds the analyses bound to the system at that call |

NetCDF is available for every model. With `target_processed_output_type: nc`,
each per-scenario artifact the output type governs is one NetCDF file under
`sims/{event_id}/processed/` in place of a Zarr store, whichever of
`toggle_triton_model`, `toggle_tritonswmm_model` and `toggle_swmm_model` are
enabled. Those artifacts are the gridded TRITON timeseries and summaries, the performance timeseries
and summaries, and the SWMM node and link timeseries and summaries. The hydrology capture summaries,
`hydrology_inflow_summary.zarr` and `hydrology_rpt_summary.zarr`, stay Zarr under either output type.

A regular analysis writes the per-scenario and per-analysis tiers in its own
directory, and writes no experiment tree. A sensitivity analysis writes one
directory per member, and each member carries its own per-scenario and
per-analysis tiers under `members/member_{member_id}/`. The experiment root is
where `experiment_datatree.zarr` is written, with one node per completed member.

`sensitivity_datatree.zarr` is the retired pre-unification name for the
experiment tree. You convert a tree that still carries it with an explicit
command, never automatically on the next run:
`python -m hhemt.version_migration migrate {analysis_dir} --apply`. Without
`--apply` the command reports what it would do and changes nothing. The
conversion can leave a superseded copy of the tree beside the promoted one, named
`experiment_datatree.zarr.superseded-v0022`.

A sensitivity experiment with two or more members can also carry a leftover
`analysis_datatree.zarr` beside its experiment tree, when its root held both that
store and a tree under the retired name before conversion. The same conversion
moves that leftover store aside to `analysis_datatree.zarr.superseded-v0025`.
That step reads the analysis config the experiment is governed by, and the
command reads a config only when `--system-config` and `--analysis-config` are
passed together, so pass both. Without them, or when the command cannot tell
which store is current, it stops and moves nothing.

Each superseded copy is kept on purpose, and nothing removes it for you.

**The per-scenario tier is flat and the consolidated tiers are hierarchical.**
That distinction matters when you open them: the flat tier is a plain `Dataset`,
the consolidated tiers are `DataTree`s.

Treat the completion flag, not the directory listing, as the signal that a
per-scenario TRITON timeseries store is finished. The flag is a sibling file
named after the store, for example `TRITONSWMM_TRITON_tseries.zarr.done`, and it
is written only once every chapter of the store has been verified and merged. An
existing store with no flag beside it is an interrupted write, not a simulation
that produced nothing. A `.chapters` sibling directory exists while the store is
being written, and it is normally removed as soon as the flag lands. A process
killed between those two steps leaves it behind indefinitely, so a `.chapters`
directory beside a flagged store is harmless leftover rather than an incomplete
result. A forced rerun of the scenario removes both the flag and the `.chapters`
directory.

### The per-scenario tier

The per-scenario processing step, `hhemt.process_timeseries_runner`, writes this
tier. For each scenario it calls `write_timeseries_outputs` and then
`write_summary_outputs` on the scenario's post-processing object, which converts
the solver's raw output into the processed stores. The tier lives in
`{analysis_dir}/sims/{event_id}/processed/`, one directory for each scenario.

### The per-analysis tier

`consolidate_to_datatree()` on the analysis's `process` object writes this tier. It
assembles the per-scenario summaries into
`{analysis_dir}/analysis_datatree.zarr`. Only an analysis that is not a sensitivity
analysis binds that path. A sensitivity analysis has the store in each member's
directory, under `members/member_{member_id}/`.

### The per-experiment tier

`consolidate_sensitivity_datatree()` on a sensitivity analysis writes this tier. It
first makes sure each member has its own `analysis_datatree.zarr`, then assembles
the members into `{analysis_dir}/experiment_datatree.zarr` at the experiment root,
with one node for each completed member. A regular analysis does not bind this
path.

### The per-system tier

`system.process.consolidate_system_datatree()` writes this tier, into
`system_datatree.zarr` in the system directory. It is written only when that
method is called explicitly. No run, command-line command or test calls it.
Its writes use append mode over the analyses bound to the system, so the store
holds the analyses that were bound when some call ran. An analysis that was not
bound at any call has no group in it.

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

`consolidated=False` is not optional: these stores are written without
consolidated metadata.

For an experiment tree, the root carries a `parameters` dataset describing
**every defined** member, while only **completed** members appear as
`member_*` nodes. A tree with fewer nodes than parameter rows is therefore an
expected partial-completion state, not a corrupt store.

## Variables

A variable with no `standard_name` has no applicable CF standard name: the CF
table does not cover it, and inventing one would be worse than leaving it unset.

**The unit is stored, not spelled in the name.** Every stored numeric variable, at every
tier, carries a `units` attribute, and no variable name carries a unit token: the SWMM node depth is `depth` with
`units: m`, and a link's flow is `flow` with `units: m3 s-1`. A string-valued variable (`type`, `event_id`, the producing-stamp
coordinates) has no unit and carries no `units` attribute. Two cases state no single unit
and carry a `comment` saying why instead: a variable that holds different units along one
dimension (the water-quality variables below, and the `max_abs_diff` difference of the raw
byte-identity artifact), and `flow_balance_error_unsplit`, which exists only in a store
migrated from layout 27 or below, where one variable held a percentage on some rows and
an error in litres on the others and the store does not record which.

### 2D surface results (TRITON)

| Variable | Units | `standard_name` | Meaning |
|---|---|---|---|
| `max_flood_depth` | `m` | `flood_water_thickness` | Maximum flood depth: the solver's running maximum of depth, updated every solver step (TRITON `MH` at the final print). In coupled runs at solver commits before `b3820a44` the SWMM exchange is applied after that update, so at a coupled manhole cell it can be below a printed depth |
| `flood_depth` | `m` | `flood_water_thickness` | Flood-depth timeseries |
| `flood_depth_last_tstep` | `m` | `flood_water_thickness` | Flood depth at the final timestep |
| `time_of_max_flood_depth` | `min` | n/a | Time of the print at which the printed depth is largest |
| `max_unit_discharge` | `m2 s-1` | n/a | Maximum unit discharge (flow rate per metre of flow width) |
| `unit_discharge_x` | `m2 s-1` | n/a | Unit discharge x-component |
| `unit_discharge_y` | `m2 s-1` | n/a | Unit discharge y-component |
| `unit_discharge_x_at_time_of_max_unit_discharge` | `m2 s-1` | n/a | Unit discharge x-component at the time of maximum unit discharge |
| `unit_discharge_y_at_time_of_max_unit_discharge` | `m2 s-1` | n/a | Unit discharge y-component at the time of maximum unit discharge |
| `time_of_max_unit_discharge` | `min` | n/a | Time at which maximum unit discharge occurred |
| `max_flood_velocity` | `m s-1` | `flood_water_speed` | Maximum over output timesteps of the flood velocity \|q\| / h, with q and h of the same timestep; NaN where h is at or below `flood_velocity_min_depth_m` (default 0.001 m) |
| `time_of_max_flood_velocity` | `min` | n/a | Time at which the maximum flood velocity occurred |
| `x_flood_velocity_at_time_of_max_flood_velocity` | `m s-1` | n/a | Flood-velocity x-component (q_x / h) at that time, along the grid x axis |
| `y_flood_velocity_at_time_of_max_flood_velocity` | `m s-1` | n/a | Flood-velocity y-component (q_y / h) at that time, along the grid y axis |
| `final_surface_flood_volume` | `m3` | n/a | Final surface flood volume |

!!! note "Unit discharge is not velocity"
    The solver's `QX` and `QY` rasters are depth-integrated flow per metre of flow width
    (`m2 s-1`), so the six rows above whose names contain `unit_discharge` are unit discharges
    (or the time of their maximum). The CF standard-name table has no name for that quantity,
    which is why those rows carry no `standard_name`; the flood-depth rows keep theirs. The TRITON tier
    writes no variable named `max_velocity`: that name appears only on a SWMM link node
    (`tritonswmm_swmm_link`, `swmm_only_link`), where it is the maximum conduit velocity.

    The four `flood_velocity` rows are velocities: unit discharge divided by the depth of the same
    output timestep. They are a maximum over output timesteps only, so a peak between two prints is
    not seen, and they are NaN where the depth is at or below `flood_velocity_min_depth_m` or never
    wet. Each carries a `comment` attribute stating the formula and the threshold used.

### Network results (SWMM)

| Variable | Units | `standard_name` | Meaning |
|---|---|---|---|
| `total_inflow_vol` | `10^6 L` | n/a | Total inflow volume |
| `max_flow` | `m3 s-1` | n/a | Link tier: maximum flow magnitude (SWMM's Link Flow Summary "Maximum \|Flow\|") |
| `time_of_max_flow` | `min` | n/a | Link tier: time of that maximum |
| `max_velocity` | `m s-1` | n/a | Link tier: maximum conduit velocity |
| `max_flood_rate` | `m3 s-1` | n/a | Node tier: maximum flooding rate (SWMM's Node Flooding Summary "Maximum Rate") |
| `time_of_max_total_inflow` | `min` | n/a | Node tier: time of the maximum total inflow (SWMM's Node Inflow Summary "Time of Max Occurrence") |
| `max_ponded_depth` | `m` | n/a | Node tier: maximum ponded depth; present only where SWMM headed the Node Flooding Summary's last column `Depth` (Dynamic Wave routing), NaN with a warning where it is `Volume` |
| `flow_balance_error` | `percent` | n/a | Node tier: flow balance error where SWMM printed a percent; NaN on a row where it printed litres |
| `flow_balance_error_volume` | `L` | n/a | Node tier: flow balance error where SWMM printed an absolute error followed by `ltr`; NaN on a row where it printed a percent |
| `max_over_full_flow` | `1` | n/a | Maximum flow as a fraction of full-flow capacity |
| `max_over_full_depth` | `1` | n/a | Maximum depth as a fraction of full depth |
| `depth`, `head` | `m` | n/a | Node tier time series: water depth and hydraulic head at each report step |
| `inflow_flow`, `flooding` | `m3 s-1` | n/a | Node tier time series: total inflow and flooding rate at each report step |
| `flow`, `velocity` | `m3 s-1`, `m s-1` | n/a | Link tier time series: signed flow and velocity at each report step |
| `link_depth` | `m` | n/a | Link tier time series: water depth at each report step |
| `capacity` / `capacity_setting` | `1` | n/a | Link tier time series: SWMM's link capacity (a standalone SWMM run) or capacity-or-setting (a coupled run) at each report step |
| `{series}_last`, `{series}_max` | the series' | n/a | For each of the series above: the value at the final report step, and the largest value over the report steps. A link's `flow_max` and `velocity_max` are the largest SIGNED values; `max_flow` and `max_velocity` are the report's largest magnitudes |
| `max_lateral_inflow`, `max_total_inflow` | `m3 s-1` | n/a | Node tier: maximum lateral and total inflow (SWMM's Node Inflow Summary) |
| `lateral_inflow_vol` | `10^6 L` | n/a | Node tier: total lateral inflow volume |
| `hours_flooded` | `h` | n/a | Node tier: hours the node was flooded (SWMM's Node Flooding Summary) |
| `time_of_max_flood` | `min` | n/a | Node tier: time of `max_flood_rate` |
| `tot_flooded_vol` | `10^6 L` | n/a | Node tier: total flood volume |
| `flow_balance_error_unsplit` | none | n/a | Node tier, migrated stores only: the flow balance error as layout 27 stored it, a percent on some rows and litres on others; carries a `comment` and no `units` |

The two ratio variables are dimensionless, which CF expresses as units `1`. A
value above 1 means the conduit exceeded its design capacity. Every `time_of_max_*` variable is in
minutes elapsed from the SWMM REPORT start, clamped at 0 and at one-minute resolution;
that equals the time since the simulation start only when the model's report start equals
its start date.

### Hydrology and runoff-model results

The hydrology report summary (`processed/hydrology_rpt_summary.zarr`) holds the Subcatchment
Runoff Summary of the hydrology model, one row per subcatchment, as SWMM printed it:
`total_precip`, `total_runon`, `total_evap`, `total_infil`, `imperv_runoff`, `perv_runoff` and
`total_runoff_depth` in `mm`; `total_runoff_volume` in `10^6 L`; `peak_runoff` in `m3 s-1`; and
`runoff_coeff` in `1`. The runoff-model store (`processed/hydrology_inflow_summary.zarr`)
holds `inflow` (`m3 s-1`) and `total_inflow_volume` (`m3`) per node. Each variable carries its `units`
attribute.

### Water-quality results (SWMM, quality-routed models only)

These variables are written only when the SWMM report prints `Water Quality .......... YES`,
which SWMM does when the model declares at least one pollutant and `IGNORE_QUALITY` is not `YES`.
For any other model none of the names, coordinates or dimensions below exist, and the outputs
are identical to those of a model with no pollutants. They are carried along a `pollutant`
dimension, whose coordinate lists the pollutant identifiers in the model's `[POLLUTANTS]` order;
the coordinate `pollutant_conc_units` gives each pollutant's declared concentration unit
(`mg/L`, `ug/L` or `#/L`).

| Variable | Dimensions | Units | Meaning |
|---|---|---|---|
| `pollutant_conc` | node, time, pollutant | per `pollutant_conc_units` | Node concentration at each report step (time-series file) |
| `link_pollutant_conc` | link, time, pollutant | per `pollutant_conc_units` | Link concentration at each report step (time-series file) |
| `pollutant_max_conc` | node, pollutant | per `pollutant_conc_units` | Maximum node concentration over the event |
| `pollutant_time_of_max_conc` | node, pollutant | `min` | Minutes from the report's `Starting Date` to the first step at the maximum |
| `outfall_total_load` | node, pollutant | `kg` (a count for `#/L`) | Total mass discharged at each outfall; NaN at other nodes |
| `outfall_total_vol` | node | `10^6 L` | Total volume discharged at each outfall |
| `outfall_emc` | node, pollutant | per `pollutant_conc_units` | Event mean concentration: total load over total volume; NaN where the volume is zero |
| `link_total_load` | link, pollutant | `kg` (a count for `#/L`) | Total mass carried by each link over the run |
| `washoff_total_load` | subcatchment, pollutant | `kg` (a count for `#/L`) | Total washoff per subcatchment (in the hydrology summary; on the node dataset for a standalone model) |
| `qual_routing_{term}` | pollutant | `kg` (a count for `#/L`) | One variable per row of SWMM's `Quality Routing Continuity` table: `dry_weather_inflow`, `wet_weather_inflow`, `groundwater_inflow`, `rdii_inflow`, `external_inflow`, `external_outflow`, `flooding_loss`, `exfiltration_loss`, `mass_reacted`, `initial_stored_mass`, `final_stored_mass` |
| `qual_routing_continuity_error` | pollutant | `percent` | That table's continuity error |
| `runoff_qual_{term}` | pollutant | `kg` (a count for `#/L`) | One variable per row of `Runoff Quality Continuity`: `initial_buildup`, `surface_buildup`, `wet_deposition`, `sweeping_removal`, `infiltration_loss`, `bmp_removal`, `surface_runoff`, `remaining_buildup` |
| `runoff_qual_continuity_error` | pollutant | `percent` | That table's continuity error |
| `pollutant_stats_status` (coordinate, coupled summaries) | event | n/a | `computed`, `computed: resumed by exact replay`, `computed: resumed from a snapshot that carries SWMM water-quality state`, or the reason the statistics are withheld |

SWMM prints the mass of a count pollutant (`#/L`) as a base-10 logarithm; the toolkit linearizes
every mass value, mapping a printed `0.000` to 0. Concentrations in the coupled model come from the
SWMM report, which prints them to 0.001 of the pollutant's unit, so a trace pollutant is best
declared in `UG/L`. The two continuity percentage errors are read as printed. A variable's `units`
is omitted, with a `comment`, when the pollutants do not all share one unit. The concentration series
carry no peak statistic under the generic `_max`/`_last` names, so the set of columns compared across
simulations is the same with and without water quality.

## Global attributes

Tree roots carry `Conventions: "CF-1.13"`, `analysis_id`, and `system_id`.

A consolidated tree also carries provenance: a deterministic RO-Crate core in the
root attribute `ro_crate_metadata`, with a co-located `ro-crate-metadata.json`
sidecar beside the store. The embedded core is deterministic by construction
(byte-identical across reruns) because the volatile fields (wall-clock times,
host, job id) live only in the sidecar.

## A caveat on performance columns

The `performance.*` columns are timing records, not physical results:

- Only `performance.Total`, `performance.Simulation` and `performance.Init` carry
  wallclock semantics. The category columns are slowest-rank cost.
- On a **hotstart-resumed** simulation these are cumulative across every
  allocation, so they will exceed what a scheduler reports as elapsed time for
  the final allocation. That is correct, not double-counting.
- The `perf_*` columns of `df_status` and `scenario_status.csv` read these same
  values, so both points hold for them too.

## See also

- [Configuration schema](config-schema.md): the inputs that produce these outputs.
- [The interactive analysis report](example-report.md): the rendered view of the same data.
- [FAIR scope table](fair-scope-table.md): the archival posture of each artifact class.
