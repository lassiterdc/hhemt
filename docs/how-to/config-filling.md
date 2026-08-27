# Fill in your configuration

**Goal:** produce a working system config and analysis config for your own study
area, and confirm they load before you spend any compute on them.

**Prerequisites:** a working install ([Installation](installation.md)), a DEM for
your watershed, a SWMM `.inp` model, and a weather NetCDF.

**Field-by-field meanings are not on this page** — they are in the
[configuration schema reference](../reference/config-schema.md). This page is the
order to do things in.

## 1. Copy the templates

```bash
cp test_data/norfolk_coastal_flooding/template_system_config.yaml   my_system.yaml
cp test_data/norfolk_coastal_flooding/template_analysis_config.yaml my_analysis.yaml
```

The templates ship a **serial, local** configuration. Everything below is either
replacing a placeholder path or opting into something the default does not do.

## 2. Point the system config at your data

Replace every `/path/to/...` value in `my_system.yaml`, and set `crs.horizontal_epsg`
to the projected CRS of your DEM. If you author by hand rather than loading a case
study, also replace any `${DATA_DIR}` / `${PACKAGE_DIR}` placeholders.

Set `target_dem_resolution` to the cell size you actually want to simulate at —
this coarsens the full-resolution DEM, and it is the single field with the largest
effect on runtime.

## 3. Resolve the toggles

Each toggle you flip makes other fields required. Set the toggles first, then fill
what they demand — the [toggle-dependency table](../reference/config-schema.md#toggle-dependent-required-fields)
lists each pairing. The common ones:

- Using a landuse raster for Manning's *n* rather than a constant? You owe the
  lookup file, the raster, and three column names.
- Using SWMM for hydrology? You owe the hydrology `.inp` and a subcatchment-to-raingage
  mapping.

## 4. Point the analysis config at your events

Set `weather_timeseries`, `weather_events_to_simulate`, and the four
`weather_*` schema fields to the dimension and variable names **as they appear in
your NetCDF**. These are not conventions the toolkit imposes; they are how it finds
your data.

## 5. Choose how it runs

The default is `run_mode: serial` / `multi_sim_run_method: local`. To go beyond
that, set `run_mode` and **add** the fields that mode requires —
`n_mpi_procs` / `n_omp_threads` / `n_gpus` / `n_nodes`. They are cross-validated at
config load, so a mismatch fails immediately rather than at dispatch.

Running on a cluster also needs a third config: see
[HPC-profile setup](hpc-profile-setup.md).

## 6. Verify before you commit compute

Everything above is checkable in seconds without submitting a single job.

```bash
hhemt run --system-config my_system.yaml --analysis-config my_analysis.yaml --dry-run
```

**Verifiable end state:** the dry run exits 0 and prints the workflow it would
execute. A non-zero exit names the offending field — preflight validation
accumulates every error and reports them together, so you fix one round of
problems rather than discovering them one at a time.

For a stronger check that actually compiles and runs a minimal subset of your own
analysis, use `analysis.test()` — see the
[Norfolk end-to-end tutorial](../tutorials/norfolk-end-to-end.md).

!!! warning "Sensitivity analyses: edit the XLSX, never the CSV"
    With `toggle_sensitivity_analysis: true`, the derived
    `sensitivity_analysis_definition.csv` is re-generated from your XLSX on
    **every** `analysis.run()`. Direct CSV edits are silently overwritten before
    the workflow plans, which reads as a no-op run. See the
    [rerun FAQ](../explanation/rerun-faq.md).

## See also

- [Configuration schema reference](../reference/config-schema.md) — what each field means.
- [HPC-profile setup](hpc-profile-setup.md)
- [Operating on an analysis while jobs are in flight](in-flight-operations.md)
