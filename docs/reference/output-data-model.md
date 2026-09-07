# Output data model

What a completed analysis leaves on disk, and what is inside it. This is the
reference for the toolkit's actual deliverable. If you are writing analysis code
against hhemt's results, this is the page you want.

Every variable below carries CF-1.13 attributes. `src/hhemt/cf_conventions.py` is
the single source of truth for `standard_name`, `long_name`, `units`, and
`cell_methods`; the table on this page is derived from it.

## The three output tiers

hhemt writes results at three levels, and they differ in both shape and format.

| Tier | Artifact | Format |
|---|---|---|
| Per-scenario | `sims/{event_id}/processed/` | Flat Zarr (default) or NetCDF, selected by `target_processed_output_type` |
| Per-analysis | `analysis_datatree.zarr` | Hierarchical `xarray.DataTree`, always Zarr |
| Per-system | `system_datatree.zarr` | Hierarchical `xarray.DataTree`, always Zarr |

A sensitivity analysis additionally writes `experiment_datatree.zarr` at the
master level, with one node per completed member. Trees written before the store
unification carry `sensitivity_datatree.zarr` (or, for a regular analysis,
`analysis_datatree.zarr`) and are migrated in place on the next run.

**The per-scenario tier is flat and the consolidated tiers are hierarchical.**
That distinction matters when you open them: the flat tier is a plain `Dataset`,
the consolidated tiers are `DataTree`s.

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

For a sensitivity master, the root carries a `parameters` dataset describing
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
