# Output data model

What a completed analysis leaves on disk, and what is inside it. This is the
reference for the toolkit's actual deliverable. If you are writing analysis code
against hhemt's results, this is the page you want.

Every variable below carries CF-1.13 attributes. `src/hhemt/cf_conventions.py` is
the single source of truth for `standard_name`, `long_name`, `units`, and
`cell_methods`; the table on this page is derived from it.

## The output tiers

hhemt writes results at four tiers, and they differ in both shape and format.

| Tier | Artifact | Format |
|---|---|---|
| Per-scenario | `sims/{event_id}/processed/` | Flat Zarr (default) or NetCDF, selected by `target_processed_output_type` (NetCDF is not supported when a TRITON model is enabled; see below) |
| Per-analysis | `analysis_datatree.zarr` | Hierarchical `xarray.DataTree`, always Zarr |
| Per-experiment | `experiment_datatree.zarr` | Hierarchical `xarray.DataTree`, always Zarr |
| Per-system | `system_datatree.zarr` | Hierarchical `xarray.DataTree`, always Zarr |

NetCDF is not available for every model. `hhemt run` and `Toolkit.from_configs`
refuse `target_processed_output_type: nc` at preflight whenever
`toggle_triton_model` or `toggle_tritonswmm_model` is enabled, because the
gridded TRITON timeseries does not support NetCDF output. With only
`toggle_swmm_model` enabled, `nc` is accepted and applies to the per-scenario
summaries and the SWMM node and link timeseries.

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

### 2D surface results (TRITON)

| Variable | Units | `standard_name` | Meaning |
|---|---|---|---|
| `max_wlevel_m` | `m` | `sea_surface_height_above_geoid` | Maximum water level over the simulation |
| `wlevel_m` | `m` | `sea_surface_height_above_geoid` | Water-level timeseries |
| `wlevel_m_last_tstep` | `m` | `sea_surface_height_above_geoid` | Water level at the final timestep |
| `max_velocity_mps` | `m s-1` | `sea_water_speed` | Maximum flood velocity |
| `velocity_x_mps` | `m s-1` | `sea_water_x_velocity` | Flood-velocity x-component |
| `velocity_y_mps` | `m s-1` | `sea_water_y_velocity` | Flood-velocity y-component |
| `time_of_max_velocity_min` | `minutes` | n/a | Time at which maximum velocity occurred |
| `final_surface_flood_volume_m3` | `m3` | n/a | Final surface flood volume |

!!! warning "`max_velocity_mps` means two different things"
    On a TRITON surface node it is the maximum 2D flood velocity, as above. On a
    SWMM link node (`tritonswmm_swmm_link`, `swmm_only_link`) the same name and
    the same `standard_name` carry **maximum conduit velocity** instead. The
    `long_name` attribute on the variable is what distinguishes them; check it
    before comparing this variable across nodes.

### Network results (SWMM)

| Variable | Units | `standard_name` | Meaning |
|---|---|---|---|
| `total_inflow_vol_10e6_ltr` | `10^6 L` | n/a | Total inflow volume |
| `max_flow_cms` | `m3 s-1` | n/a | Maximum flow rate |
| `max_over_full_flow` | `1` | n/a | Maximum flow as a fraction of full-flow capacity |
| `max_over_full_depth` | `1` | n/a | Maximum depth as a fraction of full depth |

The two ratio variables are dimensionless, which CF expresses as units `1`. A
value above 1 means the conduit exceeded its design capacity.

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
| `pollutant_time_of_max_conc_min` | node, pollutant | `minutes` | Minutes from the simulation start to the first step at the maximum |
| `outfall_total_load` | node, pollutant | `kg` (a count for `#/L`) | Total mass discharged at each outfall; NaN at other nodes |
| `outfall_total_vol_10e6_ltr` | node | `1e6 L` | Total volume discharged at each outfall |
| `outfall_emc` | node, pollutant | per `pollutant_conc_units` | Event mean concentration: total load over total volume; NaN where the volume is zero |
| `link_total_load` | link, pollutant | `kg` (a count for `#/L`) | Total mass carried by each link over the run |
| `washoff_total_load` | subcatchment, pollutant | `kg` (a count for `#/L`) | Total washoff per subcatchment (in the hydrology summary; on the node dataset for a standalone model) |
| `qual_routing_{term}` | pollutant | `kg` (a count for `#/L`) | One variable per row of SWMM's `Quality Routing Continuity` table: `dry_weather_inflow`, `wet_weather_inflow`, `groundwater_inflow`, `rdii_inflow`, `external_inflow`, `external_outflow`, `flooding_loss`, `exfiltration_loss`, `mass_reacted`, `initial_stored_mass`, `final_stored_mass` |
| `qual_routing_continuity_error_perc` | pollutant | `percent` | That table's continuity error |
| `runoff_qual_{term}` | pollutant | `kg` (a count for `#/L`) | One variable per row of `Runoff Quality Continuity`: `initial_buildup`, `surface_buildup`, `wet_deposition`, `sweeping_removal`, `infiltration_loss`, `bmp_removal`, `surface_runoff`, `remaining_buildup` |
| `runoff_qual_continuity_error_perc` | pollutant | `percent` | That table's continuity error |
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

## See also

- [Configuration schema](config-schema.md): the inputs that produce these outputs.
- [The interactive analysis report](example-report.md): the rendered view of the same data.
- [FAIR scope table](fair-scope-table.md): the archival posture of each artifact class.
