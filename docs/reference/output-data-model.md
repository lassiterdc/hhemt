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
